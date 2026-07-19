"""ONE-TRUTH swing-structure ctx features for the V10 entry contract.

Computes the 5 swing-structure features the V10 ctx_cont contract carries:
  dist_last_swing_high_atr, dist_last_swing_low_atr,
  bars_since_swing_high, bars_since_swing_low, retracement_from_last_impulse.

LOOKAHEAD-SAFE: a swing pivot at bar j (high[j] strictly exceeds its `lookback`
neighbours on BOTH sides) is only REFLECTED into the features from bar j+lookback
— never AT bar j — so the value at bar i uses only pivots confirmed by bar i. The
live decision bar (last row) is therefore causal, and train == serve bit-for-bit.

WHY THIS FILE (rule 7): until 2026-06-24 this math lived in TWO copies — the live
augmenter (v12_ctx_augment_live._add_swing_features) and the V10 training-dataset
builder (build_entry_v10_ctx_training_dataset_v3) — and one of them reflected the
pivot AT bar j (a 2-bar look-ahead). Both now delegate here so the computation can
only ever exist once. smc_v1.py was considered but it owns the smc_* family with a
different lookback (3); these are the entry ctx_cont swing features (lookback 2).
Do NOT re-implement this elsewhere — import it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SWING_FEATURE_NAMES_V1 = (
    "dist_last_swing_high_atr",
    "dist_last_swing_low_atr",
    "bars_since_swing_high",
    "bars_since_swing_low",
    "retracement_from_last_impulse",
)


def compute_swing_structure_features(
    high, low, close, *, lookback: int = 2, atr_period: int = 14, eps: float = 1e-9
) -> dict[str, np.ndarray]:
    """Return {feature_name: float32 ndarray} for the 5 swing-structure features.

    high/low/close: equal-length, chronologically-ordered 1-D array-likes (np or pd).
    `lookback` is BOTH the pivot half-window and the confirmation lag (a pivot at j is
    reflected from bar j+lookback). The first/last `lookback` bars can never be pivots.
    Bit-identical to the pre-2026-06-24 live `_add_swing_features` (causal-fixed variant).
    """
    h = np.asarray(high, dtype=np.float64)
    low_values = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    if h.ndim != 1 or low_values.ndim != 1 or c.ndim != 1:
        raise RuntimeError(
            "SWING_STRUCTURE_SOURCE_NOT_1D: "
            f"high={h.shape} low={low_values.shape} close={c.shape}"
        )
    if not (len(h) == len(low_values) == len(c)) or len(c) == 0:
        raise RuntimeError(
            "SWING_STRUCTURE_SOURCE_LENGTH_INVALID: "
            f"high={len(h)} low={len(low_values)} close={len(c)}"
        )
    if (
        not np.isfinite(h).all()
        or not np.isfinite(low_values).all()
        or not np.isfinite(c).all()
    ):
        raise RuntimeError("SWING_STRUCTURE_SOURCE_NONFINITE")
    if np.any(h <= 0.0) or np.any(low_values <= 0.0) or np.any(c <= 0.0):
        raise RuntimeError("SWING_STRUCTURE_SOURCE_NONPOSITIVE")
    if np.any(h < low_values) or np.any(h < c) or np.any(low_values > c):
        raise RuntimeError("SWING_STRUCTURE_SOURCE_GEOMETRY_INVALID")
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
        raise RuntimeError(f"SWING_STRUCTURE_LOOKBACK_INVALID: {lookback!r}")
    if isinstance(atr_period, bool) or not isinstance(atr_period, int) or atr_period < 1:
        raise RuntimeError(f"SWING_STRUCTURE_ATR_PERIOD_INVALID: {atr_period!r}")
    if not np.isfinite(float(eps)) or float(eps) <= 0.0:
        raise RuntimeError(f"SWING_STRUCTURE_EPS_INVALID: {eps!r}")
    n = len(c)

    # ATR = TR rolling-mean (pandas rolling, matching the live/train convention).
    prev_close = np.empty(n, dtype=np.float64)
    if n:
        prev_close[0] = c[0]
        prev_close[1:] = c[:-1]
    tr = np.maximum(
        np.abs(h - low_values),
        np.maximum(np.abs(h - prev_close), np.abs(low_values - prev_close)),
    )
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=1).mean().to_numpy()
    atr_safe = np.clip(atr, eps, None)

    # Pivot detection: strict, full ±lookback window (so the first/last `lookback` bars
    # are never pivots — same edge convention as the live decision bar).
    pivot_high = np.zeros(n, dtype=bool)
    pivot_low = np.zeros(n, dtype=bool)
    for i in range(lookback, n - lookback):
        if h[i] > h[i - lookback:i].max() and h[i] > h[i + 1:i + lookback + 1].max():
            pivot_high[i] = True
        if (
            low_values[i] < low_values[i - lookback : i].min()
            and low_values[i]
            < low_values[i + 1 : i + lookback + 1].min()
        ):
            pivot_low[i] = True

    # Confirmation-lag forward fill: reflect a pivot at bar j only from bar j+lookback.
    last_high_vals = np.empty(n, dtype=np.float64)
    last_low_vals = np.empty(n, dtype=np.float64)
    last_high_idx = np.empty(n, dtype=np.int64)
    last_low_idx = np.empty(n, dtype=np.int64)
    last_high = float(h[0]) if n else 0.0
    last_low = float(low_values[0]) if n else 0.0
    last_hi_i = 0
    last_lo_i = 0
    for i in range(n):
        j = i - lookback
        if j >= 0 and pivot_high[j]:
            last_high = float(h[j])
            last_hi_i = j
        if j >= 0 and pivot_low[j]:
            last_low = float(low_values[j])
            last_lo_i = j
        last_high_vals[i] = last_high
        last_low_vals[i] = last_low
        last_high_idx[i] = last_hi_i
        last_low_idx[i] = last_lo_i

    idx = np.arange(n, dtype=np.int64)
    denom = np.maximum(last_high_vals - last_low_vals, eps)
    retracement = np.zeros(n, dtype=np.float64)
    up_mask = last_high_idx > last_low_idx
    down_mask = last_low_idx > last_high_idx
    retracement[up_mask] = (last_high_vals[up_mask] - c[up_mask]) / denom[up_mask]
    retracement[down_mask] = (c[down_mask] - last_low_vals[down_mask]) / denom[down_mask]
    retracement = np.clip(retracement, 0.0, 1.0)

    result = {
        "dist_last_swing_high_atr": ((c - last_high_vals) / atr_safe).astype(np.float32),
        "dist_last_swing_low_atr": ((c - last_low_vals) / atr_safe).astype(np.float32),
        "bars_since_swing_high": (idx - last_high_idx).astype(np.float32),
        "bars_since_swing_low": (idx - last_low_idx).astype(np.float32),
        "retracement_from_last_impulse": retracement.astype(np.float32),
    }
    if tuple(result) != SWING_FEATURE_NAMES_V1 or any(
        values.shape != (n,) or not np.isfinite(values).all()
        for values in result.values()
    ):
        raise RuntimeError("SWING_STRUCTURE_OUTPUT_INVALID")
    return result
