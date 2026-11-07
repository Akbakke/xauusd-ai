"""Independent closed-OHLC reference for the 21 parked MTF candle fields.

This module computes the stored float32 feature values from native float64
OHLC bars. It does not call the feature producer, read data, or authorize a run.
A future audit must bind the cache manifest, closed-bar clock and source bytes.
"""
from __future__ import annotations

import numpy as np

CANDLE_CACHE_NAMES = (
    'mtf_candle_raw_body_signed_range',
    'mtf_candle_raw_upper_wick_share',
    'mtf_candle_raw_lower_wick_share',
    'mtf_candle_raw_open_position_previous_range',
    'mtf_candle_raw_close_change_local_geometry',
    'mtf_candle_raw_high_change_local_geometry',
    'mtf_candle_raw_low_change_local_geometry',
    'mtf_candle_raw_body_change_local_geometry',
    'mtf_candle_raw_open_above_previous_high_local_geometry',
    'mtf_candle_raw_open_below_previous_low_local_geometry',
    'mtf_candle_raw_body_overlap_previous_local_geometry',
    'mtf_candle_raw_body_contains_previous_flag',
    'mtf_candle_raw_body_contained_by_previous_flag',
    'mtf_candle_raw_range_contains_previous_flag',
    'mtf_candle_raw_range_contained_by_previous_flag',
    'mtf_candle_raw_bull_body_covers_previous_bear_body_event',
    'mtf_candle_raw_bear_body_covers_previous_bull_body_event',
    'mtf_candle_raw_high_rejection_previous_high_event',
    'mtf_candle_raw_low_rejection_previous_low_event',
    'mtf_candle_raw_observed_body_direction_duration_bars',
    'mtf_candle_raw_observed_range_relation_duration_bars',
)
CACHE_COLUMNS = tuple(range(29, 50))
assert len(CANDLE_CACHE_NAMES) == len(CACHE_COLUMNS) == 21


def _need(condition: object, reason: str) -> None:
    if not bool(condition):
        raise ValueError('CANDLE_ORACLE_' + reason)


def _runs(values: np.ndarray) -> np.ndarray:
    """Observed-bar run lengths, including zero-direction dojis as a state."""
    n = len(values)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    starts = np.empty(n, dtype=np.int64)
    starts[0] = 0
    starts[1:] = np.where(values[1:] != values[:-1], np.arange(1, n), 0)
    np.maximum.accumulate(starts, out=starts)
    return np.arange(n, dtype=np.int64) - starts + 1


def candle21_from_closed_ohlc(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    positions: np.ndarray | None = None,
) -> np.ndarray:
    """Return cache-ordered float32 values; the caller owns as-of clocks.

    All history before a selected position is required because two raw age
    fields count observed bars without a cap. The first bar has three finite
    whole-bar shares and body-direction age 1; other relational fields are NaN.
    """
    raw = (open_, high, low, close)
    _need(all(isinstance(x, np.ndarray) and x.dtype == np.float64
              and x.ndim == 1 for x in raw), 'FLOAT64_OHLC')
    n = len(close)
    _need(n > 0 and all(len(x) == n and np.isfinite(x).all()
                        and np.all(x > 0) for x in raw), 'FINITE_POSITIVE_OHLC')
    _need(bool(np.all(high >= np.maximum(open_, close))
               and np.all(low <= np.minimum(open_, close))
               and np.all(high >= low)), 'GEOMETRY')
    if positions is not None:
        _need(isinstance(positions, np.ndarray) and positions.dtype == np.int64
              and positions.ndim == 1 and len(positions) > 0
              and positions[0] >= 0 and positions[-1] < n
              and np.all(np.diff(positions) > 0), 'POSITIONS')

    width = high - low
    signed_body = close - open_
    body = np.abs(signed_body)
    body_top = np.maximum(open_, close)
    body_bottom = np.minimum(open_, close)
    out = np.full((n, 21), np.nan, dtype=np.float64)
    for j, numerator in enumerate((signed_body, high - body_top, body_bottom - low)):
        out[:, j] = np.divide(numerator, width, out=np.zeros(n), where=width > 0)
    out[:, 19] = _runs(np.sign(signed_body)).astype(np.float64)

    if n > 1:
        current = slice(1, None)
        previous = slice(None, -1)
        prev_width = width[previous]
        prev_body = body[previous]
        # The scale is a causal two-bar geometry envelope in price units.
        scale = np.maximum(width[current], prev_width)
        for value in (
            np.abs(open_[current] - close[previous]),
            np.abs(close[current] - close[previous]),
            np.abs(high[current] - high[previous]),
            np.abs(low[current] - low[previous]),
            np.abs(width[current] - prev_width),
            np.abs(body[current] - prev_body),
        ):
            np.maximum(scale, value, out=scale)
        price = np.ones(n - 1, dtype=np.float64)
        for value in (open_[current], high[current], low[current], close[current],
                      open_[previous], high[previous], low[previous], close[previous]):
            np.maximum(price, value, out=price)
        np.maximum(scale, price * np.finfo(np.float64).eps, out=scale)
        _need(bool(np.isfinite(scale).all() and np.all(scale > 0)), 'LOCAL_SCALE')

        out[current, 3] = np.divide(
            open_[current] - low[previous], prev_width,
            out=np.zeros(n - 1), where=prev_width > 0)
        for j, numerator in enumerate((
            close[current] - close[previous],
            high[current] - high[previous],
            low[current] - low[previous],
            body[current] - prev_body,
            np.maximum(open_[current] - high[previous], 0),
            np.maximum(low[previous] - open_[current], 0),
        ), start=4):
            out[current, j] = numerator / scale

        overlap = np.maximum(
            np.minimum(body_top[current], body_top[previous])
            - np.maximum(body_bottom[current], body_bottom[previous]), 0)
        out[current, 10] = overlap / scale
        equal_body = ((body_bottom[current] == body_bottom[previous])
                      & (body_top[current] == body_top[previous]))
        contains_body = ((body_bottom[current] <= body_bottom[previous])
                         & (body_top[current] >= body_top[previous]) & ~equal_body)
        contained_body = ((body_bottom[current] >= body_bottom[previous])
                          & (body_top[current] <= body_top[previous]) & ~equal_body)
        out[current, 11] = contains_body
        out[current, 12] = contained_body

        equal_range = ((high[current] == high[previous])
                       & (low[current] == low[previous]))
        contains_range = ((high[current] >= high[previous])
                          & (low[current] <= low[previous]) & ~equal_range)
        contained_range = ((high[current] <= high[previous])
                           & (low[current] >= low[previous]) & ~equal_range)
        out[current, 13] = contains_range
        out[current, 14] = contained_range
        out[current, 15] = contains_body & (signed_body[current] > 0) & (signed_body[previous] < 0)
        out[current, 16] = contains_body & (signed_body[current] < 0) & (signed_body[previous] > 0)
        out[current, 17] = (high[current] > high[previous]) & (close[current] <= high[previous])
        out[current, 18] = (low[current] < low[previous]) & (close[current] >= low[previous])
        relation = np.where(equal_range, 2, np.where(contains_range, 1,
                            np.where(contained_range, -1, 0))).astype(np.int8)
        out[current, 20] = _runs(relation).astype(np.float64)

    _need(bool(np.isfinite(out[0, [0, 1, 2, 19]]).all()
               and np.isnan(out[0, [*range(3, 19), 20]]).all()
               and (n == 1 or np.isfinite(out[1:]).all())), 'AVAILABILITY')
    result = out.astype(np.float32)
    _need(not np.isinf(result).any(), 'FLOAT32_INFINITY')
    return result if positions is None else result[positions]
