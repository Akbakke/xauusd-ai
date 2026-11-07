"""Independent full-prefix HLC reference for five parked MTF generic scalars.

The caller supplies causal, closed native bars and binds their clock/cache bytes.
This source-only oracle does no file or model work. Values are cache-order float32;
NaN denotes an honestly unavailable formula (warmup or zero denominator).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

GENERIC5_CACHE_NAMES = (
    'rsi14_z48',
    'macd_line_atr_change_3',
    'macd_line_atr_change_5',
    'range_z20',
    'mid_ema200_distance_change_1_atr',
)
CACHE_COLUMNS = (230, 231, 232, 234, 238)
assert len(GENERIC5_CACHE_NAMES) == len(CACHE_COLUMNS) == 5


def _need(condition: object, reason: str) -> None:
    if not bool(condition):
        raise ValueError('GENERIC5_ORACLE_' + reason)


def _sma_seeded_smooth(source: np.ndarray, period: int, *, wilder: bool) -> np.ndarray:
    """Independent C-filter recurrence after an exact float64 SMA seed."""
    n = len(source)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return result
    seed = float(np.mean(source[:period], dtype=np.float64))
    result[period - 1] = seed
    if n > period:
        alpha = 1.0 / period if wilder else 2.0 / (period + 1.0)
        result[period:], _ = lfilter(
            [alpha], [1.0, -(1.0 - alpha)], source[period:],
            zi=[(1.0 - alpha) * seed],
        )
    return result


def _rolling_z(source: np.ndarray, window: int) -> np.ndarray:
    """Population z-score for every trailing window, bounded scratch memory."""
    n = len(source)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(source, window)
    for start in range(0, len(windows), 65536):
        stop = min(start + 65536, len(windows))
        sample = windows[start:stop]
        with np.errstate(invalid='ignore'):
            mean = np.mean(sample, axis=1)
            std = np.std(sample, axis=1, ddof=0)
        row = np.arange(start + window - 1, stop + window - 1)
        valid = np.isfinite(std) & (std > 0)
        result[row[valid]] = (source[row[valid]] - mean[valid]) / std[valid]
    return result


def generic5_from_closed_hlc(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    positions: np.ndarray | None = None,
) -> np.ndarray:
    """Calculate all history before optional sorted, unique observed positions.

    Source price units cancel in every output. RSI and range z-scores use
    population standard deviation; 3/5 are native-bar changes, and the EMA200
    gap change uses the current bar's Wilder ATR14 as denominator.
    """
    raw = (high, low, close)
    _need(all(isinstance(x, np.ndarray) and x.dtype == np.float64
              and x.ndim == 1 for x in raw), 'FLOAT64_HLC')
    n = len(close)
    _need(n > 0 and all(len(x) == n and np.isfinite(x).all()
                        and np.all(x > 0) for x in raw), 'FINITE_POSITIVE_HLC')
    _need(bool(np.all(high >= close) and np.all(low <= close)
               and np.all(high >= low)), 'GEOMETRY')
    if positions is not None:
        _need(isinstance(positions, np.ndarray) and positions.dtype == np.int64
              and positions.ndim == 1 and len(positions) > 0
              and positions[0] >= 0 and positions[-1] < n
              and np.all(np.diff(positions) > 0), 'POSITIONS')

    width = high - low
    true_range = width.copy()
    if n > 1:
        true_range[1:] = np.maximum.reduce((
            true_range[1:], np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ))
    atr14 = _sma_seeded_smooth(true_range, 14, wilder=True)
    positive_atr = np.isfinite(atr14) & (atr14 > 0)

    change = np.diff(close)
    gain = _sma_seeded_smooth(np.maximum(change, 0), 14, wilder=True)
    loss = _sma_seeded_smooth(np.maximum(-change, 0), 14, wilder=True)
    rsi = np.full(n, np.nan, dtype=np.float64)
    if n > 14:
        g, ell = gain[13:], loss[13:]
        both_zero = (g == 0) & (ell == 0)
        only_loss_zero = (ell == 0) & ~both_zero
        ratio = np.divide(g, ell, out=np.zeros(len(g)), where=ell > 0)
        rsi[14:] = np.where(both_zero, 50.0,
                            np.where(only_loss_zero, 100.0,
                                     100.0 - 100.0 / (1.0 + ratio)))
    rsi_z48 = _rolling_z(rsi, 48)

    ema12 = _sma_seeded_smooth(close, 12, wilder=False)
    ema26 = _sma_seeded_smooth(close, 26, wilder=False)
    macd = np.divide(ema12 - ema26, atr14,
                     out=np.full(n, np.nan), where=positive_atr)
    macd_change3 = np.full(n, np.nan, dtype=np.float64)
    macd_change5 = np.full(n, np.nan, dtype=np.float64)
    if n > 3:
        macd_change3[3:] = macd[3:] - macd[:-3]
    if n > 5:
        macd_change5[5:] = macd[5:] - macd[:-5]

    range_z20 = _rolling_z(width, 20)
    mid = (high + low) * 0.5
    gap200 = mid - _sma_seeded_smooth(mid, 200, wilder=False)
    gap_change = np.full(n, np.nan, dtype=np.float64)
    if n > 219:
        valid = positive_atr[219:]
        gap_change[219:] = np.divide(
            gap200[219:] - gap200[218:-1], atr14[219:],
            out=np.full(n - 219, np.nan), where=valid,
        )

    result = np.column_stack((rsi_z48, macd_change3, macd_change5,
                              range_z20, gap_change)).astype(np.float32)
    _need(not np.isinf(result).any(), 'FLOAT32_INFINITY')
    return result if positions is None else result[positions]
