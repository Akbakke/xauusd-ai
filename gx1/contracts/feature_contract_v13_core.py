"""
ONE TRUTH feature contract: v13_core

Properties:
- Deterministic (hard-coded list in canonical order)
- Read-only (no runtime file reads)
- Strict (no fallback)
- No chart-pattern features
"""

from __future__ import annotations

import hashlib
from typing import List


FEATURE_CONTRACT_ID = "v13_core"

# Explicitly chosen 30–40-ish feature list (order-sensitive).
# NOTE: Must be verified against the real prebuilt schema/parquet via verify_v13_core_contract.py.
FEATURES_ORDERED: List[str] = [
    # Price / levels
    "mid",
    "ema_20",
    "ema_50",
    "sma_20",
    "sma_50",
    "bb_lower",
    "bb_upper",
    # Costs
    "spread_bps",
    "spread_atr_ratio",
    # Volatility
    "atr",
    "true_range",
    "std50",
    "volatility",
    # Returns / momentum
    "ret_1",
    "ret_5",
    "ret_20",
    "roc20",
    "roc100",
    "momentum",
    # Volume proxies
    "rvol_20",
    "rvol_60",
    "rvol_ratio",
    "vol_ratio",
    # Regime
    "trend_regime_tf24h",
    "atr_regime_id",
    # Oscillators
    "rsi",
    "macd",
    # Candle geometry
    "body_size",
    "body_size_pct",
    "upper_shadow",
    "upper_shadow_pct",
    "lower_shadow",
    "lower_shadow_pct",
    "wick_asym",
    "range",
    "high_low_ratio",
]

FEATURE_LIST_TEXT = "|".join(FEATURES_ORDERED)
FEATURE_LIST_SHA256 = hashlib.sha256(FEATURE_LIST_TEXT.encode("utf-8")).hexdigest()


def _validate_contract() -> None:
    assert len(set(FEATURES_ORDERED)) == len(FEATURES_ORDERED), "Duplicate feature in FEATURES_ORDERED"


_validate_contract()

