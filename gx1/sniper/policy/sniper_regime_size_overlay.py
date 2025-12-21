#!/usr/bin/env python3
"""
SNIPER runtime size overlay – Q4 × B_MIXED gate

Motivasjon (fra SNIPER_2025_REGIME_SPLIT_REPORT__20251218_142940.md):
- Q4 baseline, regime=B_MIXED: 1,086 trades, EV ≈ −6.4 bps (EV-lekkasje)
- Q4 baseline, regime=C_CHOP: 3,389 trades, EV ≈ +29.3 bps

Denne overlayen reduserer kun eksponering i den dokumenterte lekkasjen
Q4 × B_MIXED ved å skalere units/size ned (default 0.30), uten å endre
entry/exit-regler, thresholds eller modeller.

Bruk:
- compute_quarter(entry_time) -> "Q1"|"Q2"|"Q3"|"Q4"
- apply_size_overlay(base_units, entry_time, trend_regime, vol_regime,
                     atr_bps, spread_bps, session, cfg)

cfg-format (dict):
- enabled: bool
- q4_b_mixed_multiplier: float (default 0.30)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Mapping, Tuple, Union

import logging
import pandas as pd

from gx1.sniper.analysis.regime_classifier import classify_regime


@dataclass
class SizeOverlayConfig:
    enabled: bool = False
    q4_b_mixed_multiplier: float = 0.30

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | None) -> "SizeOverlayConfig":
        if cfg is None:
            return cls()
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            q4_b_mixed_multiplier=float(
                cfg.get("q4_b_mixed_multiplier", 0.30)
            ),
        )


def compute_quarter(entry_time: Union[str, float, int, pd.Timestamp, datetime]) -> str:
    """
    Compute calendar quarter ("Q1".."Q4") from an entry_time value.

    Accepts:
    - ISO8601 string
    - epoch seconds (int/float)
    - pandas.Timestamp
    - datetime.datetime
    """
    ts: pd.Timestamp
    if isinstance(entry_time, pd.Timestamp):
        ts = entry_time
    elif isinstance(entry_time, datetime):
        ts = pd.Timestamp(entry_time)
    elif isinstance(entry_time, (int, float)):
        # Treat numeric as seconds since epoch
        ts = pd.to_datetime(entry_time, unit="s", utc=True)
    elif isinstance(entry_time, str):
        ts = pd.to_datetime(entry_time, utc=True, errors="coerce")
    else:
        # Fallback: let pandas try
        ts = pd.to_datetime(entry_time, utc=True, errors="coerce")

    if ts is None or pd.isna(ts):
        return "UNKNOWN"

    month = ts.month
    if 1 <= month <= 3:
        return "Q1"
    if 4 <= month <= 6:
        return "Q2"
    if 7 <= month <= 9:
        return "Q3"
    if 10 <= month <= 12:
        return "Q4"
    return "UNKNOWN"


def apply_size_overlay(
    base_units: int,
    entry_time: Union[str, float, int, pd.Timestamp, datetime],
    trend_regime: Any,
    vol_regime: Any,
    atr_bps: Any,
    spread_bps: Any,
    session: Any,
    cfg: Mapping[str, Any] | None,
) -> Tuple[int, Dict[str, Any]]:
    """
    Apply SNIPER size overlay for Q4 × B_MIXED.

    Returns:
        units_out (int), overlay_meta (dict)
    """
    config = SizeOverlayConfig.from_dict(cfg)

    # Base meta (even when overlay is disabled or no-op)
    overlay_meta: Dict[str, Any] = {
        "overlay_applied": False,
        "overlay_name": "Q4_B_MIXED_SIZE",
        "size_before_units": base_units,
        "size_after_units": base_units,
        "multiplier": 1.0,
        "quarter": None,
        "regime_class": None,
        "regime_reason": None,
        "session": session,
        "reason": "disabled" if not config.enabled else "not_triggered",
    }

    # If overlay disabled, just return base units as-is
    if not config.enabled:
        return int(base_units), overlay_meta

    quarter = compute_quarter(entry_time)
    overlay_meta["quarter"] = quarter

    # Build row-like dict for reuse of classify_regime()
    row = {
        "trend_regime": trend_regime,
        "vol_regime": vol_regime,
        "atr_bps": atr_bps,
        "spread_bps": spread_bps,
        "session": session,
    }
    regime_class, regime_reason = classify_regime(row)
    overlay_meta["regime_class"] = regime_class
    overlay_meta["regime_reason"] = regime_reason

    # Fail-safe: if required fields are missing, do not change size
    if quarter == "UNKNOWN" or regime_class is None:
        overlay_meta["reason"] = "missing_fields"
        return int(base_units), overlay_meta

    # Only gate Q4 × B_MIXED
    if quarter == "Q4" and regime_class == "B_MIXED":
        multiplier = float(config.q4_b_mixed_multiplier)

        # Preserve sign for short trades and round absolute units
        sign = 1 if base_units >= 0 else -1
        try:
            units_abs = abs(int(base_units))
        except Exception:
            units_abs = abs(int(float(base_units)))
        units_out_abs = int(round(units_abs * multiplier))
        if units_out_abs == 0 and units_abs > 0:
            logging.getLogger(__name__).warning(
                "[SNIPER_OVERLAY] Size overlay produced 0 units (base=%s, mult=%.3f); "
                "keeping minimum of 1 unit in same direction.",
                base_units,
                multiplier,
            )
            units_out_abs = 1
        units_out = sign * units_out_abs

        overlay_meta["overlay_applied"] = True
        overlay_meta["multiplier"] = multiplier
        overlay_meta["size_after_units"] = units_out
        overlay_meta["reason"] = "Q4_B_MIXED_gate"

        # Additional fail-closed guard: if this ever triggers outside Q4×B_MIXED
        # (should not happen), caller can inspect overlay_meta to detect issues.
        return units_out, overlay_meta

    # Sanity: if overlay_applied ever ends up True outside Q4×B_MIXED,
    # we treat it as a bug and keep base size.
    overlay_meta["reason"] = "no_gate"  # Q1–Q3 or other regimes in Q4
    return int(base_units), overlay_meta


__all__ = ["SizeOverlayConfig", "compute_quarter", "apply_size_overlay"]


