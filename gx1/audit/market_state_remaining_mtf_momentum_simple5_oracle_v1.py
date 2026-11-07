"""Independent causal oracle for five non-divergence V4 MTF momentum fields.

RSI14 and positive ATR14 are separately qualified upstream inputs. This owner
reconstructs columns 111, 112 and 123..125 without calling production helpers.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

FIELDS = ("rsi_extreme_event_age_bars", "rsi_extreme_last_event_side",
          "macd_line_atr", "macd_hist_atr", "stoch_k_14")
CACHE_COLUMNS = (111, 112, 123, 124, 125)


def _sma_seeded_ema(values: np.ndarray, span: int) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) >= span:
        result[span - 1] = values[:span].mean()
        alpha = 2.0 / (span + 1.0)
        for i in range(span, len(values)):
            result[i] = result[i - 1] + alpha * (values[i] - result[i - 1])
    return result


def _suffix_or_all_missing(values: np.ndarray) -> bool:
    finite = np.isfinite(values)
    return bool(not finite.any() or finite[np.flatnonzero(finite)[0]:].all())


def closed_momentum_simple5_reference(high, low, close, rsi14, atr14_positive) -> dict[str, np.ndarray]:
    if np.asarray(rsi14).dtype != np.float64 or np.asarray(atr14_positive).dtype != np.float64:
        raise ValueError("MOMENTUM_SIMPLE5_RAW_F64_REQUIRED")
    h, l, c, rsi, atr = (np.asarray(x, dtype=np.float64) for x in
                          (high, low, close, rsi14, atr14_positive))
    n = len(c)
    if (not n or any(x.ndim != 1 or len(x) != n for x in (h, l, c, rsi, atr))
            or not np.isfinite(h).all() or not np.isfinite(l).all()
            or not np.isfinite(c).all() or np.any(l <= 0) or np.any(h < l)
            or np.any(c < l) or np.any(c > h)
            or np.isinf(rsi).any() or np.isinf(atr).any()
            or not _suffix_or_all_missing(rsi) or not _suffix_or_all_missing(atr)
            or np.any((rsi[np.isfinite(rsi)] < 0) | (rsi[np.isfinite(rsi)] > 100))
            or np.any(atr[np.isfinite(atr)] <= 0)):
        raise ValueError("MOMENTUM_SIMPLE5_SOURCE_INVALID")
    age = np.full(n, np.nan, dtype=np.float64)
    side = np.full(n, np.nan, dtype=np.float64)
    last_age = None
    last_side = np.nan
    for i in range(n):
        if not np.isfinite(rsi[i]):
            continue
        if rsi[i] >= 70.0 or rsi[i] <= 30.0:
            last_age = 0
            last_side = 1.0 if rsi[i] >= 70.0 else -1.0
        elif last_age is not None:
            last_age += 1
        if last_age is not None:
            age[i], side[i] = float(last_age), last_side
    ema12 = _sma_seeded_ema(c, 12)
    ema26 = _sma_seeded_ema(c, 26)
    macd_raw = ema12 - ema26
    first = np.flatnonzero(np.isfinite(macd_raw))
    signal = np.full(n, np.nan, dtype=np.float64)
    if len(first):
        start = int(first[0])
        signal[start:] = _sma_seeded_ema(macd_raw[start:], 9)
    line = np.full(n, np.nan, dtype=np.float64)
    hist = np.full(n, np.nan, dtype=np.float64)
    valid_line = np.isfinite(macd_raw) & np.isfinite(atr)
    valid_hist = np.isfinite(macd_raw) & np.isfinite(signal) & np.isfinite(atr)
    line[valid_line] = macd_raw[valid_line] / atr[valid_line]
    hist[valid_hist] = (macd_raw[valid_hist] - signal[valid_hist]) / atr[valid_hist]
    stoch = np.full(n, np.nan, dtype=np.float64)
    if n >= 14:
        highest = sliding_window_view(h, 14).max(axis=1)
        lowest = sliding_window_view(l, 14).min(axis=1)
        span = highest - lowest
        valid = span > 0
        tail = stoch[13:]
        tail[valid] = (c[13:][valid] - lowest[valid]) / span[valid]
    return dict(zip(FIELDS, (age.astype(np.float32), side.astype(np.float32),
                             line.astype(np.float32), hist.astype(np.float32),
                             stoch.astype(np.float32)), strict=True))
