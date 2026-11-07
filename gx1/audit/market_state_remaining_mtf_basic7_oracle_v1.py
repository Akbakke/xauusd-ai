"""Independent closed-TF HLC oracle for seven basic_v1 MTF cache fields.

Only cache columns 194, 195, 197–201 are calculated. Input is already-closed,
ordered native-TF high/low/close with an availability clock. The caller owns
verified source rows, TF resampling, TRAIN selection and cache alignment.
No production indicator function, data file or model is imported here.
"""
from __future__ import annotations

import math

import numpy as np

CACHE_COLUMNS = (194, 195, 197, 198, 199, 200, 201)
FIELD_NAMES = (
    "_v1_pk_sigma20", "_v1_bb10_bandwidth_change_3", "_v1_range_z",
    "_v1_kurt_r", "_v1_tema20_change_3_atr", "_v1_bb_squeeze_20_2",
    "_v1_kama30_change_5_atr",
)
WARMUP_ROWS = (19, 12, 47, 48, 60, 118, 35)
UNITS = (
    "log_range_volatility", "price_relative_bandwidth_change", "range_zscore",
    "excess_kurtosis", "ATR_multiples", "bandwidth_relative_to_100_bar_mean",
    "ATR_multiples",
)


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("MTF_BASIC7_ORACLE_" + reason)


def _inputs(high, low, close, available_ns, train_end_exclusive_ns):
    h, l, c, clock = map(np.asarray, (high, low, close, available_ns))
    n = len(c) if c.ndim == 1 else 0
    _require(n > 0 and h.shape == l.shape == c.shape == clock.shape == (n,),
             "SHAPE")
    _require(all(x.dtype.kind == "f" and x.dtype.itemsize >= 4 for x in (h, l, c))
             and clock.dtype == np.dtype("int64"), "DTYPE")
    h, l, c = (x.astype(np.float64) for x in (h, l, c))
    _require(bool(np.isfinite(h).all() and np.isfinite(l).all()
                  and np.isfinite(c).all())
             and bool(np.all(l > 0) and np.all(h >= l)
                      and np.all(h >= c) and np.all(l <= c)),
             "HLC_GEOMETRY")
    _require(bool(np.all(clock > 0) and np.all(clock[1:] > clock[:-1])),
             "AVAILABLE_CLOCK")
    if train_end_exclusive_ns is not None:
        _require(type(train_end_exclusive_ns) is int
                 and train_end_exclusive_ns > 0
                 and bool(np.all(clock < train_end_exclusive_ns)),
                 "TRAIN_ONLY_CUTOFF")
    return h, l, c


def _mean_std(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    mean = np.full(len(values), np.nan)
    std = np.full(len(values), np.nan)
    for t in range(window - 1, len(values)):
        block = values[t - window + 1:t + 1]
        if not np.isfinite(block).all():
            continue
        avg = math.fsum(float(x) for x in block) / window
        mean[t] = avg
        std[t] = math.sqrt(math.fsum((float(x) - avg) ** 2
                                    for x in block) / window)
    return mean, std


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) < span:
        return out
    first = int(valid[0])
    if not np.isfinite(values[first:]).all():
        raise ValueError("MTF_BASIC7_ORACLE_EMA_GAP")
    seed_at = first + span - 1
    if seed_at >= len(values):
        return out
    current = math.fsum(float(x) for x in values[first:seed_at + 1]) / span
    out[seed_at] = current
    alpha = 2.0 / (span + 1.0)
    for t in range(seed_at + 1, len(values)):
        current += alpha * (float(values[t]) - current)
        out[t] = current
    return out


def _atr14(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    tr = high - low
    for t in range(1, len(close)):
        tr[t] = max(tr[t], abs(high[t] - close[t - 1]),
                    abs(low[t] - close[t - 1]))
    out = np.full(len(close), np.nan)
    if len(close) >= 14:
        out[13] = math.fsum(float(x) for x in tr[:14]) / 14.0
        for t in range(14, len(close)):
            out[t] = (13.0 * out[t - 1] + tr[t]) / 14.0
    return out


def _kama30(close: np.ndarray) -> np.ndarray:
    out = np.full(len(close), np.nan)
    previous = float(close[29]) if len(close) >= 30 else math.nan
    fast = 2.0 / 3.0
    slow = 2.0 / 31.0
    for t in range(30, len(close)):
        volatility = math.fsum(abs(float(close[i] - close[i - 1]))
                               for i in range(t - 29, t + 1))
        change = abs(float(close[t] - close[t - 30]))
        efficiency = 1.0 if volatility == 0.0 else change / volatility
        alpha = (efficiency * (fast - slow) + slow) ** 2
        previous += alpha * (float(close[t]) - previous)
        out[t] = previous
    return out


def independent_closed_tf_basic7_oracle(
    high, low, close, *, available_ns, train_end_exclusive_ns=None,
) -> dict:
    """Evaluate seven formulas on causal complete windows, with honest NaNs."""
    h, l, c = _inputs(high, low, close, available_ns,
                      train_end_exclusive_ns)
    n = len(c)
    result = np.full((n, len(FIELD_NAMES)), np.nan, dtype=np.float64)
    atr = _atr14(h, l, c)

    log_range_squared = np.log(h / l) ** 2
    mean_pk, _ = _mean_std(log_range_squared, 20)
    result[:, 0] = np.sqrt(mean_pk) / math.sqrt(4.0 * math.log(2.0))

    mean10, std10 = _mean_std(c, 10)
    width10 = np.full(n, np.nan)
    valid10 = np.isfinite(mean10) & (mean10 > 0)
    width10[valid10] = 4.0 * std10[valid10] / mean10[valid10]
    result[3:, 1] = width10[3:] - width10[:-3]

    candle_range = h - l
    range_mean, range_std = _mean_std(candle_range, 48)
    valid_range = np.isfinite(range_std) & (range_std > 0)
    result[valid_range, 2] = (
        candle_range[valid_range] - range_mean[valid_range]
    ) / range_std[valid_range]

    returns = np.full(n, np.nan)
    returns[1:] = (c[1:] - c[:-1]) / c[:-1]
    size = 48
    factor1 = size * (size + 1) / ((size - 1) * (size - 2) * (size - 3))
    factor2 = 3.0 * (size - 1) ** 2 / ((size - 2) * (size - 3))
    for t in range(size, n):
        block = returns[t - size + 1:t + 1]
        mean = math.fsum(float(x) for x in block) / size
        deviation = [float(x) - mean for x in block]
        maximum = max(abs(x) for x in deviation)
        if maximum == 0:
            continue
        normalized = [x / maximum for x in deviation]
        sum2 = math.fsum(x * x for x in normalized)
        sum4 = math.fsum(x ** 4 for x in normalized)
        sample_variance = sum2 / (size - 1)
        result[t, 3] = factor1 * sum4 / sample_variance**2 - factor2

    ema1 = _ema(c, 20)
    ema2 = _ema(ema1, 20)
    ema3 = _ema(ema2, 20)
    tema = 3.0 * ema1 - 3.0 * ema2 + ema3
    valid_tema = np.isfinite(tema[3:]) & np.isfinite(tema[:-3]) & (atr[3:] > 0)
    tema_rows = np.flatnonzero(valid_tema) + 3
    result[tema_rows, 4] = (tema[tema_rows] - tema[tema_rows - 3]) / atr[tema_rows]

    mean20, std20 = _mean_std(c, 20)
    width20 = np.full(n, np.nan)
    valid20 = np.isfinite(mean20) & (mean20 > 0)
    width20[valid20] = 4.0 * std20[valid20] / mean20[valid20]
    width_baseline, _ = _mean_std(width20, 100)
    valid_squeeze = np.isfinite(width20) & np.isfinite(width_baseline) & (width_baseline > 0)
    result[valid_squeeze, 5] = width20[valid_squeeze] / width_baseline[valid_squeeze] - 1.0

    kama = _kama30(c)
    valid_kama = np.isfinite(kama[5:]) & np.isfinite(kama[:-5]) & (atr[5:] > 0)
    kama_rows = np.flatnonzero(valid_kama) + 5
    result[kama_rows, 6] = (kama[kama_rows] - kama[kama_rows - 5]) / atr[kama_rows]

    first_available = np.arange(n)[:, None] >= np.asarray(WARMUP_ROWS)[None, :]
    _require(bool(np.isfinite(result[~first_available]).sum() == 0),
             "PREMATURE_OUTPUT")
    available = np.isfinite(result)
    return {
        "cache_columns": CACHE_COLUMNS, "field_names": FIELD_NAMES,
        "units": UNITS, "warmup_rows": WARMUP_ROWS,
        "values": result.astype(np.float32),
        "available": available,
        "undefined_after_warmup": first_available & ~available,
    }
