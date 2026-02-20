"""
ONE TRUTH feature contract: v13_refined3_prune20

Derived deterministically from v13_refined3 audit outputs:
- FEATURE_XGB_CONTRIBUTION.json (gain_norm_total + shap_mean_abs)
- FEATURE_REDUNDANCY_MATRIX.json (clusters at |rho|>=0.90)

Rules:
- Order-sensitive list
- No duplicates
- SHA256 over "|".join(FEATURES_ORDERED)

NOTE: This is a positive-list contract. Anything not listed is ignored by design.
"""

from __future__ import annotations

import hashlib
from typing import List


FEATURE_CONTRACT_ID = "v13_refined3_prune20"


# Explicit list (order-sensitive). Must match schema_manifest.required_all_features exactly.
FEATURES_ORDERED: List[str] = [
    "spread_bps",
    "spread_atr_ratio",
    "atr",
    "std50",
    "volatility",
    "atr_z_200",
    "ret_1",
    "ret_5",
    "ret_20",
    "roc100",
    "momentum",
    "rvol_20",
    "rvol_60",
    "vol_ratio",
    "spread_vs_rvol60",
    "ema_50",
    "price_vs_ema50_atr",
    "trend_regime_tf24h",
    "wick_asym",
    "high_low_ratio",
]


FEATURE_LIST_TEXT = "|".join(FEATURES_ORDERED)
FEATURE_LIST_SHA256 = hashlib.sha256(FEATURE_LIST_TEXT.encode("utf-8")).hexdigest()


def _validate_contract() -> None:
    assert len(set(FEATURES_ORDERED)) == len(FEATURES_ORDERED), "Duplicate feature in FEATURES_ORDERED"
    assert len(FEATURES_ORDERED) == 20, f"Expected 20 features, got {len(FEATURES_ORDERED)}"
    # Required anchors (hard)
    for x in ["spread_bps", "atr", "ret_1", "ret_5", "ret_20"]:
        assert x in FEATURES_ORDERED, f"Missing required anchor feature: {x}"
    assert any(x in FEATURES_ORDERED for x in ["wick_asym", "range_atr"]), "Must include >=1 candle geometry feature"
    assert any(x in FEATURES_ORDERED for x in ["ema_20", "ema_50", "price_vs_ema50_atr"]), "Must include >=1 trend feature"


_validate_contract()

