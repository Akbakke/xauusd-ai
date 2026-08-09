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

Wired into the Entry base surface via
entry_model_native_signal_v1.MODEL_NATIVE_BASE_FIELDS.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

# Ordered, frozen — the model-native contract appends this exact volume tail.
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

# A selected row is independent of the caller's window start only when the
# shared owner receives this many trailing source rows ending at that row.
# Consumers selecting an N-row model window therefore need
# ``N + VOLUME_FEATURE_PREFIX_ROWS`` source rows before computing and slicing.
VOLUME_FEATURE_REQUIRED_HISTORY_ROWS = max(_Z_WIN, _RATIO_FAST, _RATIO_SLOW, _PCT_WIN)
VOLUME_FEATURE_PREFIX_ROWS = VOLUME_FEATURE_REQUIRED_HISTORY_ROWS - 1


def compute_volume_features(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Compute the VOLUME_FEATURE_NAMES from `df['volume']` (+ `df['close']`).

    Returns a dict name -> float32 ndarray (len == len(df)). Sources are exact,
    numeric and finite. Causal warmup values use expanding trailing windows;
    missing or malformed market data is never converted into neutral evidence.
    vol_pct_96 uses the mid-rank tie convention, so a single-element warmup
    window ranks its own row at 0.5 (the mid-rank of one tied value), not 1.0.
    """
    n = len(df)
    if n == 0:
        raise RuntimeError("VOLUME_FEATURE_SOURCE_EMPTY")
    for name in ("volume", "close"):
        if name not in df.columns:
            raise RuntimeError(f"VOLUME_FEATURE_SOURCE_MISSING: {name}")
        if list(df.columns).count(name) != 1:
            raise RuntimeError(f"VOLUME_FEATURE_SOURCE_DUPLICATE: {name}")

    try:
        vol_values = pd.to_numeric(df["volume"], errors="raise").to_numpy(dtype=np.float64)
        close_values = pd.to_numeric(df["close"], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("VOLUME_FEATURE_SOURCE_NOT_NUMERIC") from exc
    if vol_values.shape != (n,) or close_values.shape != (n,):
        raise RuntimeError(
            "VOLUME_FEATURE_SOURCE_SHAPE_INVALID: "
            f"volume={vol_values.shape} close={close_values.shape} rows={n}"
        )
    if not np.isfinite(vol_values).all() or not np.isfinite(close_values).all():
        raise RuntimeError("VOLUME_FEATURE_SOURCE_NONFINITE")
    if np.any(vol_values < 0.0):
        raise RuntimeError("VOLUME_FEATURE_SOURCE_VOLUME_NEGATIVE")
    if np.any(close_values <= 0.0):
        raise RuntimeError("VOLUME_FEATURE_SOURCE_CLOSE_NOT_POSITIVE")

    vol = pd.Series(vol_values, index=df.index, dtype=np.float64)

    # z-score over trailing 20 bars
    mean20 = vol.rolling(_Z_WIN, min_periods=1).mean().to_numpy(dtype=np.float64)
    std20 = vol.rolling(_Z_WIN, min_periods=1).std(ddof=0).to_numpy(dtype=np.float64)
    vol_z = np.zeros(n, dtype=np.float64)
    np.divide(vol_values - mean20, std20, out=vol_z, where=std20 > 0.0)
    vol_z = np.clip(vol_z, -_CLIP, _CLIP)

    # fast/slow ratio - 1 (centered at 0)
    sma_fast = vol.rolling(_RATIO_FAST, min_periods=1).mean().to_numpy(dtype=np.float64)
    sma_slow = vol.rolling(_RATIO_SLOW, min_periods=1).mean().to_numpy(dtype=np.float64)
    vol_ratio = np.zeros(n, dtype=np.float64)
    np.divide(sma_fast, sma_slow, out=vol_ratio, where=sma_slow > 0.0)
    vol_ratio[sma_slow > 0.0] -= 1.0

    # rolling percentile rank over trailing 96 bars, mid-rank tie convention:
    # (count below + half the ties, including the row itself) / window length.
    # Tie-inclusive "<=" would bias quiet sessions upward and pin row 0 at 1.0;
    # mid-rank yields the neutral 0.5 for a single-element window.
    def _pct_rank(x: np.ndarray) -> float:
        last = x[-1]
        return float(((x < last).sum() + 0.5 * (x == last).sum()) / len(x))

    vol_pct = vol.rolling(_PCT_WIN, min_periods=1).apply(
        _pct_rank, raw=True
    ).to_numpy(dtype=np.float64)

    # signed by short-term return direction
    sign = np.zeros(n, dtype=np.float64)
    if n > 1:
        sign[1:] = np.sign(close_values[1:] - close_values[:-1])
    signed_vol_z = vol_z * sign

    out = {
        "vol_z_20": vol_z,
        "vol_ratio_5_20": vol_ratio,
        "vol_pct_96": vol_pct,
        "signed_vol_z_20": signed_vol_z,
    }
    emitted = {name: np.asarray(values, dtype=np.float32) for name, values in out.items()}
    if tuple(emitted) != tuple(VOLUME_FEATURE_NAMES):
        raise RuntimeError("VOLUME_FEATURE_OUTPUT_ORDER_INVALID")
    for name, values in emitted.items():
        if values.shape != (n,) or not np.isfinite(values).all():
            raise RuntimeError(
                f"VOLUME_FEATURE_OUTPUT_INVALID: {name} shape={values.shape}"
            )
    return emitted


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """In-place-safe: returns df with VOLUME_FEATURE_NAMES columns added/overwritten."""
    feats = compute_volume_features(df)
    for k, arr in feats.items():
        df[k] = arr
    return df
