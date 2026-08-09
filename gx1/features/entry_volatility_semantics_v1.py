"""One-truth semantics for ATR-ratio compression and Bollinger squeeze.

The canonical H1/M15 fields are ``ATR14 / ATR100``: values below one mean
compression and values above one mean expansion.  The canonical Bollinger
field is ``bandwidth / rolling_mean(bandwidth) - 1``: negative values mean a
squeeze and positive values mean expansion.  Keeping these transforms here
prevents feature families from silently assigning the opposite meaning.
"""
from __future__ import annotations

import numpy as np


def _finite_vector(values: np.ndarray, *, field: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise RuntimeError(f"ENTRY_VOLATILITY_SEMANTICS_INVALID: {field}")
    return arr


def center_atr_ratio(values: np.ndarray) -> np.ndarray:
    """Return the signed centered ATR-ratio ``tanh(log2(ratio))`` in [-1, 1].

    This is the one-truth transform for every ``ATR_a / ATR_b`` ratio field:
    0 at ratio 1, ``-tanh(1)`` at ratio 0.5, ``+tanh(1)`` at ratio 2.  It fails
    closed on non-finite or non-positive input; consumers must call this owner
    instead of re-deriving it with a soft floor that would read corrupt input
    as maximal compression.
    """

    ratio = _finite_vector(values, field="atr14_over_atr100")
    if (ratio <= 0.0).any():
        raise RuntimeError("ENTRY_VOLATILITY_SEMANTICS_ATR_RATIO_NOT_POSITIVE")
    centered = np.tanh(np.log(ratio) / np.log(2.0))
    return np.clip(centered, -1.0, 1.0).astype(np.float32, copy=False)


def atr_ratio_compression_pressure(values: np.ndarray) -> np.ndarray:
    """Return [0,1] pressure where lower ATR14/ATR100 means more compression."""

    return np.maximum(-center_atr_ratio(values), 0.0).astype(
        np.float32, copy=False
    )


def atr_ratio_expansion_pressure(values: np.ndarray) -> np.ndarray:
    """Return [0,1] pressure where higher ATR14/ATR100 means more expansion."""

    return np.maximum(center_atr_ratio(values), 0.0).astype(
        np.float32, copy=False
    )


def bollinger_squeeze_pressure(values: np.ndarray) -> np.ndarray:
    """Return [0,1] squeeze from ``bandwidth/mean_bandwidth - 1``."""

    relative_delta = _finite_vector(values, field="bb_width_over_mean_minus_one")
    return np.clip(-relative_delta, 0.0, 1.0).astype(np.float32, copy=False)


def bollinger_expansion_pressure(values: np.ndarray) -> np.ndarray:
    """Return [0,1] expansion from ``bandwidth/mean_bandwidth - 1``."""

    relative_delta = _finite_vector(values, field="bb_width_over_mean_minus_one")
    return np.clip(relative_delta, 0.0, 1.0).astype(np.float32, copy=False)


__all__ = [
    "atr_ratio_compression_pressure",
    "atr_ratio_expansion_pressure",
    "bollinger_expansion_pressure",
    "bollinger_squeeze_pressure",
    "center_atr_ratio",
]
