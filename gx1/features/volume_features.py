#!/usr/bin/env python3
"""
Volume / order-flow per-M5-bar features — ONE TRUTH (2026-05-26).

These derive purely from the raw `volume` (+ `close` for the signed variant)
columns, both of which are present in the canonical feature frame at TRAINING
time (canonical_v2/v3 parquet) AND at SERVING time (augment_canonical_v3 input).
Computing them in a single shared function — called by the V10 builder and the
live ctx augmenter — guarantees identical train/serve values without a costly
canonical-pipeline regeneration.

XAUUSD OANDA `volume` is tick-volume (count of price updates) — a robust proxy
for participation/activity. All features are self-normalising (z-score / ratio /
percentile) so absolute tick-count scale and broker differences wash out.

Wired into the V10 seq via signal_bridge_v3.PER_BAR_PRICE_STATE_FIELDS_V3.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

# Ordered, frozen — the contract appends these to PER_BAR_PRICE_STATE_FIELDS_V3.
VOLUME_FEATURE_NAMES = [
    "vol_z_20",          # volume z-score over trailing 20 M5 bars (surge detector)
    "vol_ratio_5_20",    # SMA5(vol)/SMA20(vol) - 1 (fast-vs-slow activity)
    "vol_pct_96",        # rolling percentile rank of vol over trailing 96 bars (regime)
    "signed_vol_z_20",   # vol_z_20 * sign(ret_1) (directional participation)
]
VOLUME_FEATURE_COUNT = len(VOLUME_FEATURE_NAMES)

_Z_WIN = 20
_RATIO_FAST = 5
_RATIO_SLOW = 20
_PCT_WIN = 96
_CLIP = 6.0  # clip z-scores to ±6σ so a single bad tick-print can't dominate


def compute_volume_features(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Compute the VOLUME_FEATURE_NAMES from `df['volume']` (+ `df['close']`).

    Returns a dict name -> float32 ndarray (len == len(df)). NaN-safe: warmup
    bars and missing/zero volume resolve to neutral 0.0. Causal (trailing
    windows only) — no future leakage.
    """
    n = len(df)
    if n == 0:
        return {k: np.zeros(0, dtype=np.float32) for k in VOLUME_FEATURE_NAMES}

    if "volume" not in df.columns:
        # Fail-closed-neutral: no volume → all features 0.0 (model sees "no signal").
        return {k: np.zeros(n, dtype=np.float32) for k in VOLUME_FEATURE_NAMES}

    vol = pd.to_numeric(df["volume"], errors="coerce").astype(np.float64)
    vol = vol.fillna(0.0).clip(lower=0.0)

    # z-score over trailing 20 bars
    mean20 = vol.rolling(_Z_WIN, min_periods=max(2, _Z_WIN // 2)).mean()
    std20 = vol.rolling(_Z_WIN, min_periods=max(2, _Z_WIN // 2)).std(ddof=0)
    vol_z = ((vol - mean20) / std20.replace(0.0, np.nan)).clip(-_CLIP, _CLIP)

    # fast/slow ratio - 1 (centered at 0)
    sma_fast = vol.rolling(_RATIO_FAST, min_periods=2).mean()
    sma_slow = vol.rolling(_RATIO_SLOW, min_periods=2).mean()
    vol_ratio = (sma_fast / sma_slow.replace(0.0, np.nan)) - 1.0

    # rolling percentile rank over trailing 96 bars (fraction of window <= current)
    def _pct_rank(x: np.ndarray) -> float:
        last = x[-1]
        return float((x <= last).mean())

    vol_pct = vol.rolling(_PCT_WIN, min_periods=max(8, _PCT_WIN // 8)).apply(
        _pct_rank, raw=True
    )

    # signed by short-term return direction
    if "close" in df.columns:
        close = pd.to_numeric(df["close"], errors="coerce").astype(np.float64)
        ret1 = close.diff()
        sign = np.sign(ret1.fillna(0.0).to_numpy())
    else:
        sign = np.zeros(n, dtype=np.float64)
    signed_vol_z = vol_z.to_numpy() * sign

    out = {
        "vol_z_20": vol_z.to_numpy(),
        "vol_ratio_5_20": vol_ratio.to_numpy(),
        "vol_pct_96": vol_pct.to_numpy(),
        "signed_vol_z_20": signed_vol_z,
    }
    # NaN/inf warmup → neutral 0.0; cast float32.
    return {
        k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        for k, v in out.items()
    }


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """In-place-safe: returns df with VOLUME_FEATURE_NAMES columns added/overwritten."""
    feats = compute_volume_features(df)
    for k, arr in feats.items():
        df[k] = arr
    return df
