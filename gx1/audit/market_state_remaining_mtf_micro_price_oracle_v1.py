"""Independent closed-bar price oracle for five V4 MTF micro fields.

This does not call the production primitive or read a cache. A later single-pass
TRAIN audit compares these float32 outputs with cache columns 218..222.
"""
from __future__ import annotations

import numpy as np

FIELDS = (
    "close_return_3_bps",
    "close_return_5_bps",
    "close_return_acceleration_1_bps",
    "close_distance_below_high_range_fraction",
    "close_range_observed",
)
CACHE_COLUMNS = (218, 219, 220, 221, 222)


def closed_micro_price5_reference(high, low, close) -> dict[str, np.ndarray]:
    """Use only each closed bar and earlier closes; preserve the causal NaN prefix."""
    h, l, c = (np.asarray(value, dtype=np.float64) for value in (high, low, close))
    if (h.ndim != 1 or l.ndim != 1 or c.ndim != 1 or
            len(h) != len(l) or len(l) != len(c) or not len(c)):
        raise ValueError("MICRO_PRICE5_SOURCE_SHAPE")
    if (not np.isfinite(h).all() or not np.isfinite(l).all() or
            not np.isfinite(c).all() or np.any(l <= 0) or
            np.any(h < l) or np.any(c < l) or np.any(c > h)):
        raise ValueError("MICRO_PRICE5_SOURCE_VALUES")
    n = len(c)
    r3 = np.full(n, np.nan, dtype=np.float64)
    r5 = np.full(n, np.nan, dtype=np.float64)
    acceleration = np.full(n, np.nan, dtype=np.float64)
    if n > 3:
        r3[3:] = (c[3:] / c[:-3] - 1.0) * 10000.0
    if n > 5:
        r5[5:] = (c[5:] / c[:-5] - 1.0) * 10000.0
    if n > 2:
        acceleration[2:] = ((c[2:] / c[1:-1] - 1.0) -
                            (c[1:-1] / c[:-2] - 1.0)) * 10000.0
    width = h - l
    observed = width > 0.0
    distance = np.zeros(n, dtype=np.float64)
    distance[observed] = (h[observed] - c[observed]) / width[observed]
    return dict(zip(FIELDS, (r3.astype(np.float32), r5.astype(np.float32),
                             acceleration.astype(np.float32),
                             distance.astype(np.float32),
                             observed.astype(np.float32)), strict=True))
