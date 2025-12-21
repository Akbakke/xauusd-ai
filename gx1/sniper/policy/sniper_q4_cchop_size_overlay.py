#!/usr/bin/env python3
"""
SNIPER runtime size overlay – Q4 × C_CHOP session-based gate.

This overlay is a pure policy-layer adjustment:
- No changes to entry/exit logic or models.
- Only scales units for trades in Q4 × C_CHOP, by session.

Config shape (policy YAML):

sniper_q4_cchop_overlay:
  enabled: true
  multipliers:
    EU: 1.00
    OVERLAP: 1.00
    US: 0.50
  default_multiplier: 1.00
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Mapping, Tuple, Union

import logging
import pandas as pd

from gx1.sniper.analysis.regime_classifier import classify_regime
from gx1.sniper.policy.sniper_regime_size_overlay import compute_quarter


logger = logging.getLogger(__name__)

OVERLAY_IMPL_ID = "q4_cchop_overlay_v2_20251218_1930"


def apply_q4_cchop_overlay(
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
    Apply Q4 × C_CHOP session-based size overlay.

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
    multipliers = cfg.get("multipliers") or {}
    default_mult = float(cfg.get("default_multiplier", 1.0))
    try:
        mult = float(multipliers.get(session_s, default_mult))
    except Exception:
        mult = default_mult

    regime_class: Any = None
    regime_reason: str = "missing_fields"
    reason: str = "init"
    overlay_applied: bool = False
    units_out: int = int(base_units)

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
    elif regime_class != "C_CHOP":
        reason = f"not_c_chop:{regime_class}"
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
            logger.warning(
                "[SNIPER_Q4_CCHOP] Size overlay produced 0 units (base=%s, mult=%.3f); "
                "keeping minimum of 1 unit in same direction.",
                base_units,
                mult,
            )
            units_out_abs = 1
        units_out = sign * units_out_abs
        overlay_applied = True
        reason = "Q4_C_CHOP_session_gate"

    overlay_meta: Dict[str, Any] = {
        "overlay_name": "Q4_C_CHOP_SESSION_SIZE",
        "overlay_applied": overlay_applied,
        "quarter": quarter,
        "regime_class": regime_class,
        "regime_reason": regime_reason,
        "session": session_s,
        "multiplier": mult,
        "size_before_units": base_units,
        "size_after_units": units_out,
        "reason": reason,
        "impl_id": OVERLAY_IMPL_ID,
        "impl_file": __file__,
    }

    return units_out, overlay_meta


__all__ = ["apply_q4_cchop_overlay"]


