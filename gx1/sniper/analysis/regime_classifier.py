"""
Offline regime classification for SNIPER trades (no replay, no policy changes).

We define three coarse regime classes:
- A_TREND : trending / directional regimes with sufficient volatility and acceptable spreads
- B_MIXED : everything that is neither clearly trending nor clearly choppy
- C_CHOP  : low-vol / choppy / range-bound regimes

Classification is based only on fields that are already present in SNIPER trade journals:
- trend_regime (string label, e.g. 'TREND_UP', 'TREND_DOWN', 'TREND_NEUTRAL', 'MR', 'RANGE')
- vol_regime (string label, e.g. 'LOW', 'MEDIUM', 'HIGH', 'EXTREME')
- atr_bps (float, approximate volatility level)
- spread_bps (float, microstructure cost proxy)
- session (ASIA/EU/OVERLAP/US) – not directly used here but available for future refinements

This module is intentionally read‑only analytics and does not change any runtime logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple


@dataclass(frozen=True)
class RegimeClass:
    """Simple names for SNIPER regime classes."""

    A_TREND: str = "A_TREND"
    B_MIXED: str = "B_MIXED"
    C_CHOP: str = "C_CHOP"


# Canonical label groups (lower‑cased)
TREND_LABELS_TREND = {
    "trend",
    "trending",
    "trend_up",
    "trend_down",
    "bull_trend",
    "bear_trend",
    "uptrend",
    "downtrend",
}

TREND_LABELS_CHOP = {
    "chop",
    "choppy",
    "range",
    "ranging",
    "range_bound",
    "rangebound",
    "noise",
    "mean_revert",
    "mr",
    "flat",
}

VOL_LABELS_ULTRA_LOW = {
    "ultra_low",
    "very_low",
}

VOL_LABELS_LOW = {
    "low",
}

VOL_LABELS_HIGH = {
    "high",
    "very_high",
    "extreme",
}


def _norm_str(value) -> str:
    """Normalize a string field to lowercase for comparison."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.strip().lower()


def _is_trending(trend_label: str) -> bool:
    """Return True if trend_regime suggests a trending environment."""
    t = _norm_str(trend_label)
    if not t:
      return False
    if t in TREND_LABELS_TREND:
        return True
    # Many SNIPER/FARM trend labels include 'trend' explicitly
    if "trend_up" in t or "trend_down" in t:
        return True
    return False


def _is_choppy(trend_label: str) -> bool:
    """Return True if trend_regime suggests chop / range / noisy regime."""
    t = _norm_str(trend_label)
    if not t:
        return False
    if t in TREND_LABELS_CHOP:
        return True
    # Common neutral / range-ish labels
    if "neutral" in t or "sideways" in t:
        return True
    if "range" in t:
        return True
    if "chop" in t:
        return True
    return False


def _is_ultra_low_vol(vol_label: str, atr_bps: float | None) -> bool:
    """
    Heuristic for ultra-low volatility:
    - Explicit vol_regime label in ULTRA/VERY_LOW set, OR
    - ATR in bps below a small threshold (e.g. < 10 bps).
    """
    v = _norm_str(vol_label)
    if v in VOL_LABELS_ULTRA_LOW:
        return True
    try:
        if atr_bps is not None and float(atr_bps) < 10.0:
            return True
    except Exception:
        pass
    return False


def _is_spread_high(spread_bps: float | None) -> bool:
    """
    Heuristic for 'high' spreads.

    For XAUUSD M5, spreads in the ~5–15 bps range are typical;
    treat something like >= 20 bps as 'high' and thus less suitable
    for a clean trending regime classification.
    """
    try:
        if spread_bps is None:
            return False
        return float(spread_bps) >= 20.0
    except Exception:
        return False


def classify_regime(row: Mapping[str, object]) -> Tuple[str, str]:
    """
    Classify a single trade row into (regime_class, reason).

    The input is expected to behave like a dict / pandas.Series with at least:
    - 'trend_regime'
    - 'vol_regime'
    - 'atr_bps'
    - 'spread_bps'

    The rules are:
    1) If clearly trending (trend_regime in TREND group) AND not ultra-low vol
       AND spreads are not high → A_TREND.
    2) Else if clearly choppy/range/flat or ultra-low vol → C_CHOP.
    3) Otherwise → B_MIXED.

    This logic is intentionally simple and transparent, and can be refined
    later without changing core trade entry/exit rules.
    """
    trend_label = row.get("trend_regime")
    vol_label = row.get("vol_regime")
    atr_bps = row.get("atr_bps")
    spread_bps = row.get("spread_bps")

    t_trend = _norm_str(trend_label)
    t_vol = _norm_str(vol_label)

    trending = _is_trending(t_trend)
    choppy = _is_choppy(t_trend)
    ultra_low = _is_ultra_low_vol(t_vol, atr_bps)
    spread_high = _is_spread_high(spread_bps)

    # Case 1: trending, not ultra-low vol, spreads acceptable
    if trending and not ultra_low and not spread_high:
        reason = (
            f"A_TREND: trend_regime={t_trend!r}, vol_regime={t_vol!r}, "
            f"atr_bps={atr_bps}, spread_bps={spread_bps}"
        )
        return RegimeClass.A_TREND, reason

    # Case 2: obviously choppy / range / ultra-low vol
    if choppy or ultra_low:
        reason = (
            f"C_CHOP: trend_regime={t_trend!r}, vol_regime={t_vol!r}, "
            f"atr_bps={atr_bps}, spread_bps={spread_bps}"
        )
        return RegimeClass.C_CHOP, reason

    # Fallback: mixed regime (not clearly trend or chop)
    reason = (
        f"B_MIXED: trend_regime={t_trend!r}, vol_regime={t_vol!r}, "
        f"atr_bps={atr_bps}, spread_bps={spread_bps}"
    )
    return RegimeClass.B_MIXED, reason


__all__ = ["RegimeClass", "classify_regime"]


