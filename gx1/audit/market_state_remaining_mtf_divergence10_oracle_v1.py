"""Independent causal reference for the ten MTF confirmed-pivot divergences.

Inputs are closed HLC, separately qualified raw Wilder RSI14, and positive
Wilder ATR14 on the same native lane. This oracle neither computes those two
inputs nor reads cache/data; the audit caller binds their source and clock.
"""
from __future__ import annotations

import numpy as np

DIVERGENCE10_CACHE_NAMES = (
    'bear_divergence_event',
    'bull_divergence_event',
    'bear_divergence_strength',
    'bull_divergence_strength',
    'divergence_event_age_bars',
    'divergence_last_event_side',
    'hidden_bear_divergence_event',
    'hidden_bull_divergence_event',
    'hidden_bear_divergence_strength',
    'hidden_bull_divergence_strength',
)
CACHE_COLUMNS = tuple(range(113, 123))
SWING_LOOKBACK = 3
PIVOT_TIE_TOLERANCE = 1e-12
RSI_MIDLINE = 50.0
assert len(DIVERGENCE10_CACHE_NAMES) == len(CACHE_COLUMNS) == 10


def _need(condition: object, reason: str) -> None:
    if not bool(condition):
        raise ValueError('DIVERGENCE10_ORACLE_' + reason)


def _valid_suffix(values: np.ndarray) -> bool:
    valid = np.isfinite(values)
    return not valid.any() or bool(valid[np.argmax(valid):].all())


def _pivot_masks(high: np.ndarray, low: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(high)
    upper = np.zeros(n, dtype=bool)
    lower = np.zeros(n, dtype=bool)
    width = 2 * SWING_LOOKBACK + 1
    if n >= width:
        high_windows = np.lib.stride_tricks.sliding_window_view(high, width)
        low_windows = np.lib.stride_tricks.sliding_window_view(low, width)
        center = slice(SWING_LOOKBACK, n - SWING_LOOKBACK)
        upper[center] = high[center] >= (high_windows.max(axis=1) - PIVOT_TIE_TOLERANCE)
        lower[center] = low[center] <= (low_windows.min(axis=1) + PIVOT_TIE_TOLERANCE)
    return upper, lower


def _confirmed_pair(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Search confirmation times, independent of the producer's carry loop."""
    n = len(mask)
    pivots = np.flatnonzero(mask)
    recent = np.full(n, -1, dtype=np.int64)
    older = np.full(n, -1, dtype=np.int64)
    arriving = np.zeros(n, dtype=bool)
    if len(pivots):
        confirmations = pivots + SWING_LOOKBACK
        counts = np.searchsorted(confirmations, np.arange(n), side='right')
        has_recent = counts >= 1
        has_older = counts >= 2
        recent[has_recent] = pivots[counts[has_recent] - 1]
        older[has_older] = pivots[counts[has_older] - 2]
        arriving[confirmations] = True
    return recent, older, arriving


def divergence10_from_closed_hlc_rsi_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    rsi14_raw: np.ndarray,
    atr14_positive: np.ndarray,
    *,
    positions: np.ndarray | None = None,
) -> np.ndarray:
    """Return cache-order float32 values after the full causal prefix.

    Unavailable pair comparisons stay NaN. Defined off-event flags and strength
    are zero. Age and held side remain NaN until the first regular event after
    both high/low pair streams are defined. Hidden events do not reset them.
    """
    raw = (high, low, close, rsi14_raw, atr14_positive)
    _need(all(isinstance(x, np.ndarray) and x.dtype == np.float64
              and x.ndim == 1 for x in raw), 'FLOAT64_INPUTS')
    n = len(close)
    _need(n > 0 and all(len(x) == n for x in raw), 'LENGTH')
    _need(bool(np.isfinite(high).all() and np.isfinite(low).all()
               and np.isfinite(close).all() and np.all(low > 0)
               and np.all(low <= close) and np.all(close <= high)), 'HLC_GEOMETRY')
    _need(bool(not np.isinf(rsi14_raw).any() and _valid_suffix(rsi14_raw)
               and np.all((rsi14_raw[np.isfinite(rsi14_raw)] >= 0)
                          & (rsi14_raw[np.isfinite(rsi14_raw)] <= 100))), 'RSI_DOMAIN')
    _need(bool(not np.isinf(atr14_positive).any() and _valid_suffix(atr14_positive)
               and np.all(atr14_positive[np.isfinite(atr14_positive)] > 0)), 'ATR_DOMAIN')
    if positions is not None:
        _need(isinstance(positions, np.ndarray) and positions.dtype == np.int64
              and positions.ndim == 1 and len(positions) > 0
              and positions[0] >= 0 and positions[-1] < n
              and np.all(np.diff(positions) > 0), 'POSITIONS')

    high_pivots, low_pivots = _pivot_masks(high, low)
    new_high, old_high, high_arrival = _confirmed_pair(high_pivots)
    new_low, old_low, low_arrival = _confirmed_pair(low_pivots)
    result = np.full((n, 10), np.nan, dtype=np.float64)
    rsi_valid = np.isfinite(rsi14_raw)
    bear_defined = (old_high >= 0) & rsi_valid[np.maximum(old_high, 0)]
    bull_defined = (old_low >= 0) & rsi_valid[np.maximum(old_low, 0)]
    for flag, strength, defined in ((0, 2, bear_defined), (1, 3, bull_defined),
                                    (6, 8, bear_defined), (7, 9, bull_defined)):
        result[defined, flag] = 0.0
        result[defined, strength] = 0.0

    high_ready = high_arrival & bear_defined
    low_ready = low_arrival & bull_defined
    high_latest = np.maximum(new_high, 0)
    high_older = np.maximum(old_high, 0)
    low_latest = np.maximum(new_low, 0)
    low_older = np.maximum(old_low, 0)
    bear = (high_ready & (high[high_latest] > high[high_older])
            & (rsi14_raw[high_latest] < rsi14_raw[high_older]))
    bull = (low_ready & (low[low_latest] < low[low_older])
            & (rsi14_raw[low_latest] > rsi14_raw[low_older]))
    hidden_bear = (high_ready & (high[high_latest] < high[high_older])
                   & (rsi14_raw[high_latest] > rsi14_raw[high_older]))
    hidden_bull = (low_ready & (low[low_latest] > low[low_older])
                   & (rsi14_raw[low_latest] < rsi14_raw[low_older]))

    for event, flag, strength, newer, older, price, price_sign, rsi_sign in (
        (bear, 0, 2, new_high, old_high, high, 1, -1),
        (bull, 1, 3, new_low, old_low, low, -1, 1),
        (hidden_bear, 6, 8, new_high, old_high, high, -1, 1),
        (hidden_bull, 7, 9, new_low, old_low, low, 1, -1),
    ):
        rows = np.flatnonzero(event)
        result[rows, flag] = 1.0
        if len(rows):
            current = newer[rows]
            previous = older[rows]
            rsi_delta = rsi_sign * (rsi14_raw[current] - rsi14_raw[previous])
            price_delta = price_sign * (price[current] - price[previous])
            _need(bool(np.isfinite(atr14_positive[current]).all()
                       and np.all(atr14_positive[current] > 0)), 'EVENT_ATR')
            result[rows, strength] = (rsi_delta / RSI_MIDLINE) * (
                price_delta / atr14_positive[current])

    both_defined = bear_defined & bull_defined
    regular_event = (bear | bull) & both_defined
    event_row = np.where(regular_event, np.arange(n), -1)
    np.maximum.accumulate(event_row, out=event_row)
    observed = both_defined & (event_row >= 0)
    rows = np.flatnonzero(observed)
    result[rows, 4] = rows - event_row[rows]
    event_side = np.where(bear & bull, 0.0, np.where(bear, 1.0, -1.0))
    result[rows, 5] = event_side[event_row[rows]]
    result = result.astype(np.float32)
    _need(not np.isinf(result).any(), 'FLOAT32_INFINITY')
    return result if positions is None else result[positions]
