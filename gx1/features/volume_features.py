#!/usr/bin/env python3
"""
Volume / order-flow native-bar features — ONE TRUTH (2026-08-14).

These derive purely from the raw `volume` column, which is present in the
canonical feature frame at TRAINING time (canonical_v2/v3 parquet) AND at
SERVING time (augment_canonical_v3 input).
Computing them in a single shared function — called by the V10 builder and the
live ctx augmenter — guarantees identical train/serve values without a costly
canonical-pipeline regeneration.

The admitted XAUUSD OANDA `volume` source is a positive integer tick count
(number of price updates), not traded lots or exchange volume. Horizons are
therefore native closed-bar horizons: 20 rows means 20 M1 bars on an M1 frame
and 20 M5 bars on an M5 frame. This owner never resamples or mixes clocks; its
caller owns closed-bar selection and chronological row order.

Wired into the Entry base surface via
entry_model_native_signal_v1.MODEL_NATIVE_BASE_FIELDS.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

# Ordered, frozen — the model-native contract appends this exact volume tail.
VOLUME_FEATURE_NAMES = [
    "vol_z_20",          # tick-count z-score over trailing 20 native bars
    "vol_ratio_5_20",    # SMA5(vol)/SMA20(vol) - 1 (fast-vs-slow activity)
    "vol_pct_96",        # rolling percentile rank of vol over trailing 96 bars (regime)
]
VOLUME_FEATURE_COUNT = len(VOLUME_FEATURE_NAMES)

_Z_WIN = 20
_RATIO_FAST = 5
_RATIO_SLOW = 20
_PCT_WIN = 96

# Each feature is unavailable until its complete declared trailing window is
# present. The model surface has a longer common upstream warmup, but the owner
# keeps feature-local availability explicit instead of fabricating neutral
# evidence at the start of an arbitrary caller chunk.
VOLUME_FEATURE_WARMUP_ROWS = {
    "vol_z_20": _Z_WIN - 1,
    "vol_ratio_5_20": max(_RATIO_FAST, _RATIO_SLOW) - 1,
    "vol_pct_96": _PCT_WIN - 1,
}

# A selected row is independent of the caller's window start only when the
# shared owner receives this many trailing source rows ending at that row.
# Consumers selecting an N-row model window therefore need
# ``N + VOLUME_FEATURE_PREFIX_ROWS`` source rows before computing and slicing.
VOLUME_FEATURE_REQUIRED_HISTORY_ROWS = 1 + max(VOLUME_FEATURE_WARMUP_ROWS.values())
VOLUME_FEATURE_PREFIX_ROWS = VOLUME_FEATURE_REQUIRED_HISTORY_ROWS - 1


def compute_volume_features(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Compute the VOLUME_FEATURE_NAMES from exact observed `df['volume']`.

    Returns a dict name -> float32 ndarray (len == len(df)). Sources are exact,
    numeric and finite. `volume` must match the admitted positive-integer tick
    count source. Each feature emits NaN until its complete causal trailing
    window exists; missing history is never converted into neutral evidence.
    `vol_pct_96` uses the mid-rank tie convention on its complete window.
    """
    n = len(df)
    if n == 0:
        raise RuntimeError("VOLUME_FEATURE_SOURCE_EMPTY")
    if "volume" not in df.columns:
        raise RuntimeError("VOLUME_FEATURE_SOURCE_MISSING: volume")
    if list(df.columns).count("volume") != 1:
        raise RuntimeError("VOLUME_FEATURE_SOURCE_DUPLICATE: volume")
    if df["volume"].map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise RuntimeError("VOLUME_FEATURE_SOURCE_VOLUME_NOT_POSITIVE_INTEGER")

    try:
        vol_values = pd.to_numeric(df["volume"], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("VOLUME_FEATURE_SOURCE_NOT_NUMERIC") from exc
    if vol_values.shape != (n,):
        raise RuntimeError(
            "VOLUME_FEATURE_SOURCE_SHAPE_INVALID: "
            f"volume={vol_values.shape} rows={n}"
        )
    if not np.isfinite(vol_values).all():
        raise RuntimeError("VOLUME_FEATURE_SOURCE_NONFINITE")
    if np.any(vol_values <= 0.0) or not np.equal(
        vol_values, np.floor(vol_values)
    ).all():
        raise RuntimeError("VOLUME_FEATURE_SOURCE_VOLUME_NOT_POSITIVE_INTEGER")
    vol = pd.Series(vol_values, index=df.index, dtype=np.float64)

    # z-score over trailing 20 bars
    mean20 = vol.rolling(_Z_WIN, min_periods=_Z_WIN).mean().to_numpy(dtype=np.float64)
    std20 = vol.rolling(_Z_WIN, min_periods=_Z_WIN).std(ddof=0).to_numpy(dtype=np.float64)
    vol_z = np.full(n, np.nan, dtype=np.float64)
    complete_z = np.isfinite(mean20) & np.isfinite(std20)
    varying_z = complete_z & (std20 > 0.0)
    np.divide(vol_values - mean20, std20, out=vol_z, where=varying_z)
    # A complete constant window has exactly zero activity surprise.
    vol_z[complete_z & (std20 == 0.0)] = 0.0

    # fast/slow ratio - 1 (centered at 0)
    sma_fast = vol.rolling(_RATIO_FAST, min_periods=_RATIO_FAST).mean().to_numpy(
        dtype=np.float64
    )
    sma_slow = vol.rolling(_RATIO_SLOW, min_periods=_RATIO_SLOW).mean().to_numpy(
        dtype=np.float64
    )
    vol_ratio = np.full(n, np.nan, dtype=np.float64)
    complete_ratio = np.isfinite(sma_fast) & np.isfinite(sma_slow)
    np.divide(sma_fast, sma_slow, out=vol_ratio, where=complete_ratio)
    vol_ratio[complete_ratio] -= 1.0

    # rolling percentile rank over trailing 96 bars, mid-rank tie convention:
    # (count below + half the ties, including the row itself) / window length.
    # Tie-inclusive "<=" would bias a constant full window upward; mid-rank
    # yields 0.5 when all 96 observations are tied.
    def _pct_rank(x: np.ndarray) -> float:
        last = x[-1]
        return float(((x < last).sum() + 0.5 * (x == last).sum()) / len(x))

    vol_pct = vol.rolling(_PCT_WIN, min_periods=_PCT_WIN).apply(
        _pct_rank, raw=True
    ).to_numpy(dtype=np.float64)

    out = {
        "vol_z_20": vol_z,
        "vol_ratio_5_20": vol_ratio,
        "vol_pct_96": vol_pct,
    }
    emitted = {name: np.asarray(values, dtype=np.float32) for name, values in out.items()}
    if tuple(emitted) != tuple(VOLUME_FEATURE_NAMES):
        raise RuntimeError("VOLUME_FEATURE_OUTPUT_ORDER_INVALID")
    for name, values in emitted.items():
        warmup = min(VOLUME_FEATURE_WARMUP_ROWS[name], n)
        warmup_is_nan = np.isnan(values[:warmup]).all()
        available_is_finite = np.isfinite(values[warmup:]).all()
        if values.shape != (n,) or not warmup_is_nan or not available_is_finite:
            raise RuntimeError(
                "VOLUME_FEATURE_OUTPUT_INVALID: "
                f"{name} shape={values.shape} warmup={warmup}"
            )
    return emitted


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """In-place-safe: returns df with VOLUME_FEATURE_NAMES columns added/overwritten."""
    feats = compute_volume_features(df)
    for k, arr in feats.items():
        df[k] = arr
    return df
