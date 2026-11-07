"""Independent closed-price oracle for four V4 MTF return/volatility fields.

The later shared TRAIN audit supplies the exact native closed TF prices and
compares these outputs with cache columns 191, 192, 193 and 228.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

FIELDS = ("ret_1", "ret_20", "rvol_20", "chart.local_kama_efficiency_30")
CACHE_COLUMNS = (191, 192, 193, 228)


def closed_local_returns4_reference(close) -> dict[str, np.ndarray]:
    c = np.asarray(close, dtype=np.float64)
    if c.ndim != 1 or not len(c) or not np.isfinite(c).all() or np.any(c <= 0):
        raise ValueError("LOCAL_RETURNS4_CLOSE_INVALID")
    n = len(c)
    ret1 = np.full(n, np.nan, dtype=np.float64)
    ret20 = np.full(n, np.nan, dtype=np.float64)
    rvol20 = np.full(n, np.nan, dtype=np.float64)
    efficiency30 = np.full(n, np.nan, dtype=np.float64)
    if n > 1:
        one_bar_fraction = c[1:] / c[:-1] - 1.0
        ret1[1:] = one_bar_fraction * 10000.0
    if n > 20:
        ret20[20:] = (c[20:] / c[:-20] - 1.0) * 10000.0
        # Twenty complete one-bar returns; sample std matches the declared
        # pandas rolling.std default (ddof=1), with no partial warmup window.
        rvol20[20:] = (sliding_window_view(one_bar_fraction, 20).std(axis=1, ddof=1)
                        * 10000.0 * np.sqrt(20.0))
    if n > 30:
        absolute_steps = np.abs(np.diff(c))
        distance = sliding_window_view(absolute_steps, 30).sum(axis=1)
        positive = distance > 0.0
        values = efficiency30[30:]
        values[positive] = np.abs(c[30:] - c[:-30])[positive] / distance[positive]
    return dict(zip(FIELDS, (ret1.astype(np.float32), ret20.astype(np.float32),
                             rvol20.astype(np.float32), efficiency30.astype(np.float32)),
                    strict=True))
