"""
ONE TRUTH feature contract: v13_refined2

Derived from v13_refined with:
- OSCILLATOR family removed (rsi, macd)
- TREND family compressed: drop SMA-based spread and BB-derived features
  (keep EMA-based relatives/slopes only).

No chart patterns. Deterministic + order-sensitive.
"""

from __future__ import annotations

import hashlib
from typing import List


FEATURE_CONTRACT_ID = "v13_refined2"

# Explicit list (order-sensitive). Must match schema_manifest.required_all_features exactly.
FEATURES_ORDERED: List[str] = [
    # Costs (must be alive)
    "spread_bps",
    "spread_atr_ratio",
    # Volatility (core)
    "atr",
    "std50",
    "volatility",
    "atr_z_200",
    # Returns / momentum
    "ret_1",
    "ret_5",
    "ret_20",
    "roc20",
    "roc100",
    "momentum",
    "ret_1_atr",
    # Volume proxies
    "rvol_20",
    "rvol_60",
    "rvol_ratio",
    "vol_ratio",
    "spread_vs_rvol60",
    # Trend relatives (EMA-based only)
    "price_vs_ema20_atr",
    "price_vs_ema50_atr",
    "ema_spread_atr",
    "ema20_slope_atr",
    "ema50_slope_atr",
    # Regime
    "trend_regime_tf24h",
    # Candle geometry
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

