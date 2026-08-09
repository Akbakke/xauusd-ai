"""One-truth causal M5 micro-structure features for model-native Entry."""

from __future__ import annotations

import numpy as np
import pandas as pd


MICRO_FEATURE_NAMES_V1 = (
    "micro_momentum_3",
    "micro_momentum_5",
    "micro_acceleration",
    "wick_ratio",
    "distance_ema_fast",
)
MICRO_EMA_SPAN_V1 = 5


def compute_micro_structure_features(
    high,
    low,
    close,
    *,
    ema_span: int = MICRO_EMA_SPAN_V1,
    eps: float = 1e-9,
) -> dict[str, np.ndarray]:
    """Return the exact causal five-field micro surface.

    Momentum values without enough preceding bars have the contract-defined
    boundary value zero. No future row or external feature copy participates.
    """

    h = np.asarray(high, dtype=np.float64)
    low_values = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    if h.ndim != 1 or low_values.ndim != 1 or c.ndim != 1:
        raise RuntimeError(
            "MICRO_STRUCTURE_SOURCE_NOT_1D: "
            f"high={h.shape} low={low_values.shape} close={c.shape}"
        )
    if not (len(h) == len(low_values) == len(c)) or len(c) == 0:
        raise RuntimeError(
            "MICRO_STRUCTURE_SOURCE_LENGTH_INVALID: "
            f"high={len(h)} low={len(low_values)} close={len(c)}"
        )
    if (
        not np.isfinite(h).all()
        or not np.isfinite(low_values).all()
        or not np.isfinite(c).all()
    ):
        raise RuntimeError("MICRO_STRUCTURE_SOURCE_NONFINITE")
    if np.any(h <= 0.0) or np.any(low_values <= 0.0) or np.any(c <= 0.0):
        raise RuntimeError("MICRO_STRUCTURE_SOURCE_NONPOSITIVE")
    if np.any(h < low_values) or np.any(h < c) or np.any(low_values > c):
        raise RuntimeError("MICRO_STRUCTURE_SOURCE_GEOMETRY_INVALID")
    if isinstance(ema_span, bool) or not isinstance(ema_span, int) or ema_span < 1:
        raise RuntimeError(f"MICRO_STRUCTURE_EMA_SPAN_INVALID: {ema_span!r}")
    if not np.isfinite(float(eps)) or float(eps) <= 0.0:
        raise RuntimeError(f"MICRO_STRUCTURE_EPS_INVALID: {eps!r}")

    close_series = pd.Series(c)
    # Momentum/acceleration/EMA distance are emitted in bps of the current
    # close (repo ret_* convention: materialize_build_canonical_features_v1
    # pct_change * 10000). Raw USD diffs are era proxies on gold's multi-year
    # price drift and are forbidden as decision inputs. The fillna(0.0)
    # boundary rows keep the contract-defined warmup value; they are masked by
    # the global warmup downstream.
    momentum_3 = (
        close_series.diff(3).fillna(0.0).to_numpy(dtype=np.float64) / c * 1e4
    )
    momentum_5 = (
        close_series.diff(5).fillna(0.0).to_numpy(dtype=np.float64) / c * 1e4
    )
    acceleration = (
        close_series.diff().diff().fillna(0.0).to_numpy(dtype=np.float64) / c * 1e4
    )
    wick_ratio = (h - c) / (h - low_values + float(eps))
    # A zero-range bar carries no close-location evidence; emit the neutral
    # 0.5 (basic_v1._v1_clv's documented convention for the identical
    # degenerate case) instead of the fabricated 0.0 ("closed at high").
    wick_ratio = np.where(h == low_values, 0.5, wick_ratio)
    ema_fast = close_series.ewm(span=ema_span, adjust=False).mean().to_numpy()

    result = {
        "micro_momentum_3": momentum_3.astype(np.float32),
        "micro_momentum_5": momentum_5.astype(np.float32),
        "micro_acceleration": acceleration.astype(np.float32),
        "wick_ratio": wick_ratio.astype(np.float32),
        "distance_ema_fast": ((c - ema_fast) / c * 1e4).astype(np.float32),
    }
    if tuple(result) != MICRO_FEATURE_NAMES_V1 or any(
        values.shape != c.shape or not np.isfinite(values).all()
        for values in result.values()
    ):
        raise RuntimeError("MICRO_STRUCTURE_OUTPUT_INVALID")
    return result
