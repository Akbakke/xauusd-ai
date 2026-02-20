"""
ONE TRUTH feature contract: v13_refined

Goal:
- Information-dense, more stationary, ATR-normalized where possible
- No chart patterns
- Deterministic + order-sensitive
"""

from __future__ import annotations

import hashlib
from typing import List


FEATURE_CONTRACT_ID = "v13_refined"

# Explicitly chosen refined feature set (order-sensitive).
FEATURES_ORDERED: List[str] = [
    # Costs (must be alive)
    "spread_bps",
    "spread_atr_ratio",
    # Volatility (core)
    "atr",
    "std50",
    "volatility",
    "atr_z_200",
    # Returns / momentum (mostly stationary)
    "ret_1",
    "ret_5",
    "ret_20",
    "roc20",
    "roc100",
    "momentum",
    "ret_1_atr",
    # Volume proxies (dimensionless)
    "rvol_20",
    "rvol_60",
    "rvol_ratio",
    "vol_ratio",
    "spread_vs_rvol60",
    # Trend/level relative (ATR-normalized)
    "price_vs_ema20_atr",
    "price_vs_ema50_atr",
    "ema_spread_atr",
    "sma_spread_atr",
    "ema20_slope_atr",
    "ema50_slope_atr",
    # Bollinger derived (relative)
    "bb_width_atr",
    "bb_pos",
    "bb_z",
    # Oscillators (normalized)
    "rsi",
    "macd_atr",
    # Regime (continuous)
    "trend_regime_tf24h",
    # Candle geometry (ATR-normalized + dimensionless)
    "body_size_atr",
    "range_atr",
    "upper_shadow_atr",
    "lower_shadow_atr",
    "wick_asym",
    "high_low_ratio",
]

FEATURE_LIST_TEXT = "|".join(FEATURES_ORDERED)
FEATURE_LIST_SHA256 = hashlib.sha256(FEATURE_LIST_TEXT.encode("utf-8")).hexdigest()


def _validate_contract() -> None:
    assert len(set(FEATURES_ORDERED)) == len(FEATURES_ORDERED), "Duplicate feature in FEATURES_ORDERED"


_validate_contract()

