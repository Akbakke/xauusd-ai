#!/usr/bin/env python3
"""
SNIPER runtime size overlay – EU timing-based gate (P4.1).

This overlay reduces EU mid/late exposure while keeping EU early.
- EU Early (07:00-09:00 UTC): multiplier = 1.00
- EU Mid (09:00-10:20 UTC): multiplier = 0.0 (block)
- EU Late (10:20-12:00 UTC): multiplier = 0.0 (block)

Config shape (policy YAML):

sniper_q4_eu_timing_overlay:
  enabled: true
  multipliers:
    EU_EARLY: 1.00
    EU_MID: 0.0
    EU_LATE: 0.0
    OVERLAP: 1.00
    US_EARLY: 1.00
    US_MID: 1.00
    US_LATE: 1.00
  default_multiplier: 1.00
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Mapping, Tuple, Union

import logging
import pandas as pd

logger = logging.getLogger(__name__)

OVERLAY_IMPL_ID = "q4_eu_timing_overlay_v1_20251229"


def _get_session_timing(entry_time: Union[str, float, int, pd.Timestamp, datetime], session: str) -> str:
    """
    Determine timing within session (Early/Mid/Late).
    
    Session boundaries (UTC):
    - EU: 07:00-12:00 (5 hours)
      - Early: 07:00-09:00 (first 33%)
      - Mid: 09:00-10:20 (middle 33%)
      - Late: 10:20-12:00 (last 33%)
    - OVERLAP: 12:00-16:00 (4 hours)
      - Early: 12:00-13:20 (first 33%)
      - Mid: 13:20-14:40 (middle 33%)
      - Late: 14:40-16:00 (last 33%)
    - US: 16:00-22:00 (6 hours)
      - Early: 16:00-18:00 (first 33%)
      - Mid: 18:00-20:00 (middle 33%)
      - Late: 20:00-22:00 (last 33%)
    """
    try:
        if isinstance(entry_time, str):
            ts = pd.Timestamp(entry_time)
        elif isinstance(entry_time, (int, float)):
            ts = pd.Timestamp(entry_time, unit='s')
        elif isinstance(entry_time, datetime):
            ts = pd.Timestamp(entry_time)
        else:
            ts = entry_time
        
        # Convert to UTC if timezone-aware
        if ts.tzinfo is not None:
            ts_utc = ts.tz_convert("UTC")
        else:
            ts_utc = ts.tz_localize("UTC")
        
        hour = ts_utc.hour
        minute = ts_utc.minute
        total_minutes = hour * 60 + minute
        
        if session == "EU":
            # EU: 07:00-12:00 UTC
            if 7 <= hour < 9:
                return "EU_EARLY"
            elif 9 <= hour < 10 or (hour == 10 and minute < 20):
                return "EU_MID"
            elif (hour == 10 and minute >= 20) or (hour == 11) or (hour == 12 and minute == 0):
                return "EU_LATE"
        elif session == "OVERLAP":
            # OVERLAP: 12:00-16:00 UTC
            if 12 <= hour < 13 or (hour == 13 and minute < 20):
                return "OVERLAP_EARLY"
            elif (hour == 13 and minute >= 20) or (hour == 14) or (hour == 15 and minute < 40):
                return "OVERLAP_MID"
            elif (hour == 15 and minute >= 40) or (hour == 16):
                return "OVERLAP_LATE"
        elif session == "US":
            # US: 16:00-22:00 UTC
            if 16 <= hour < 18:
                return "US_EARLY"
            elif 18 <= hour < 20:
                return "US_MID"
            elif 20 <= hour < 22:
                return "US_LATE"
        
        # Default: return session name if timing cannot be determined
        return session
    except Exception as e:
        logger.warning(f"[EU_TIMING_OVERLAY] Failed to determine timing: {e}")
        return session


def apply_q4_eu_timing_overlay(
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
    Apply EU timing-based size overlay (P4.1).
    
    Returns:
        units_out (int), overlay_meta (dict)
    """
    cfg = cfg or {}
    
    session_s = session or "UNKNOWN"
    enabled = bool(cfg.get("enabled", False))
    multipliers = cfg.get("multipliers") or {}
    default_mult = float(cfg.get("default_multiplier", 1.0))
    
    # Determine timing within session
    timing_key = _get_session_timing(entry_time, session_s)
    
    # Get multiplier for this timing
    try:
        mult = float(multipliers.get(timing_key, multipliers.get(session_s, default_mult)))
    except Exception:
        mult = default_mult
    
    reason: str = "init"
    overlay_applied: bool = False
    units_out: int = int(base_units)
    
    # Apply overlay
    if not enabled:
        reason = "disabled"
    elif session_s not in ["EU", "OVERLAP", "US"]:
        reason = f"not_eu_overlap_us:{session_s}"
    elif abs(mult - 1.0) < 1e-9:
        reason = "multiplier_1.0"
    else:
        # Preserve sign for short trades and round absolute units
        sign = 1 if base_units >= 0 else -1
        try:
            units_abs = abs(int(base_units))
        except Exception:
            units_abs = abs(int(float(base_units)))
        units_out_abs = int(round(units_abs * mult))
        if units_out_abs == 0 and units_abs > 0:
            logger.info(
                "[SNIPER_EU_TIMING] Size overlay produced 0 units (base=%s, mult=%.3f, timing=%s); "
                "trade will be blocked.",
                base_units,
                mult,
                timing_key,
            )
            units_out_abs = 0  # Allow 0 to block trade
        units_out = sign * units_out_abs
        overlay_applied = True
        reason = f"timing_gate:{timing_key}"
    
    overlay_meta: Dict[str, Any] = {
        "overlay_name": "EU_TIMING_SIZE",
        "overlay_applied": overlay_applied,
        "session": session_s,
        "timing": timing_key,
        "multiplier": mult,
        "size_before_units": base_units,
        "size_after_units": units_out,
        "reason": reason,
        "impl_id": OVERLAY_IMPL_ID,
        "impl_file": __file__,
    }
    
    return units_out, overlay_meta


__all__ = ["apply_q4_eu_timing_overlay"]

