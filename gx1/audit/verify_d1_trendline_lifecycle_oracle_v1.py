"""Independent D1 trendline oracle from causal OHLC and ATR arrays.

This is a reference implementation for source qualification, not a producer,
model input, fit, or permission to read market data. It deliberately does not
import the trendline registry or consume its line IDs, populations, or state.
The caller owns immutable source, TRAIN constants, capped admission, and cache
comparison. The full prefix must be replayed before any selected witness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np


NAMES = (
    "geomline_above_active_count", "geomline_above_dist_atr",
    "geomline_above_slope_atr_per_bar", "geomline_above_touch_count",
    "geomline_above_age_bars", "geomline_above_last_touch_age_bars",
    "geomline_above_max_dev_atr", "geomline_below_active_count",
    "geomline_below_dist_atr", "geomline_below_slope_atr_per_bar",
    "geomline_below_touch_count", "geomline_below_age_bars",
    "geomline_below_last_touch_age_bars", "geomline_below_max_dev_atr",
    "geomline_touch_above", "geomline_touch_below", "geomline_break_up",
    "geomline_break_down", "geomline_break_line_touch_count",
    "geomline_break_line_age_bars", "geomline_retest_hold_up",
    "geomline_retest_fail_up", "geomline_retest_hold_down",
    "geomline_retest_fail_down", "geomline_bars_since_break_signed",
    "geomchan_active", "geomchan_width_atr", "geomchan_pos_0_1",
    "geomchan_slope_atr_per_bar", "geomchan_converging",
    "geomchan_apex_proximity",
)
LIFECYCLE_NAMES = (
    "geomline_above_active_count", "geomline_above_touch_count",
    "geomline_above_max_dev_atr", "geomline_below_active_count",
    "geomline_below_touch_count", "geomline_below_max_dev_atr",
    "geomline_touch_above", "geomline_touch_below", "geomline_break_up",
    "geomline_break_down", "geomline_break_line_touch_count",
    "geomline_break_line_age_bars", "geomline_retest_hold_up",
    "geomline_retest_fail_up", "geomline_retest_hold_down",
    "geomline_retest_fail_down", "geomline_bars_since_break_signed",
    "geomchan_active", "geomchan_converging",
)
PIVOT_RADIUS = 3
PIVOT_TIE = 1e-12
RETEST_BARS = 2 * PIVOT_RADIUS + 1
SUPPORT, RESISTANCE = 1, -1


@dataclass
class _Line:
    ordinal: int
    side: int
    first: int
    first_price: float
    second: int
    second_price: float
    touch_bars: set[int]
    last_touch: int
    max_deviation: float
    mode: str = "active"
    in_band_before: bool = False
    broken_at: int = -1
    break_direction: int = 0
    slope: float = field(init=False)
    intercept: float = field(init=False)

    def __post_init__(self) -> None:
        self.slope = (self.second_price - self.first_price) / float(self.second - self.first)
        self.intercept = self.first_price - self.slope * float(self.first)

    def at(self, bar: int) -> float:
        return self.intercept + self.slope * float(bar)

    @property
    def touches(self) -> int:
        return len(self.touch_bars)


@dataclass
class _Candidate:
    ordinal: int
    side: int
    first: int
    first_price: float
    second: int
    second_price: float
    slope: float = field(init=False)
    intercept: float = field(init=False)

    def __post_init__(self) -> None:
        self.slope = (self.second_price - self.first_price) / float(self.second - self.first)
        self.intercept = self.first_price - self.slope * float(self.first)

    def at(self, bar: int) -> float:
        return self.intercept + self.slope * float(bar)


@dataclass
class _Book:
    pivots: dict[int, list[tuple[int, float]]] = field(
        default_factory=lambda: {SUPPORT: [], RESISTANCE: []})
    pending: dict[int, list[_Candidate]] = field(
        default_factory=lambda: {SUPPORT: [], RESISTANCE: []})
    lines: list[_Line] = field(default_factory=list)
    next_ordinal: int = 0
    last_break: int = -1
    last_break_direction: int = 0
    confirmed_pivots: int = 0
    promotions: int = 0
    first_breaks: int = 0
    held_retests: int = 0
    failed_retests: int = 0


def _require(ok: bool, name: str) -> None:
    if not ok:
        raise ValueError("D1_TRENDLINE_ORACLE_" + name)


def confirmed_center(high: Sequence[float], low: Sequence[float], t: int,
                     radius: int = PIVOT_RADIUS) -> tuple[bool, bool]:
    """Return independently checked low/high pivots known at bar t."""
    _require(radius > 0 and t >= 0 and len(high) == len(low) and t < len(high), "PIVOT_DOMAIN")
    if t < 2 * radius:
        return False, False
    center = t - radius
    lo = min(float(low[i]) for i in range(center - radius, center + radius + 1))
    hi = max(float(high[i]) for i in range(center - radius, center + radius + 1))
    return (float(low[center]) <= lo + PIVOT_TIE,
            float(high[center]) >= hi - PIVOT_TIE)


def _prune(book: _Book, t: int, seq_len: int) -> None:
    for side in (SUPPORT, RESISTANCE):
        book.pivots[side] = [pivot for pivot in book.pivots[side]
                             if t - pivot[0] < seq_len]
        book.pending[side] = [candidate for candidate in book.pending[side]
                              if t - candidate.first < seq_len]


def _pivot(book: _Book, side: int, center: int, price: float, pivot_atr: float,
           t: int, band: float, events: np.ndarray) -> None:
    touch_index = 15 if side == SUPPORT else 14
    for line in book.lines:
        if line.mode != "active" or line.side != side:
            continue
        deviation = abs(price - line.at(center)) / pivot_atr
        if deviation > band:
            continue
        line.max_deviation = max(line.max_deviation, deviation)
        if center not in line.touch_bars:
            line.touch_bars.add(center)
            line.last_touch = max(line.last_touch, center)
            events[touch_index] = 1.0
    for candidate in book.pending[side]:
        deviation = abs(price - candidate.at(center)) / pivot_atr
        if deviation <= band:
            book.lines.append(_Line(
                ordinal=candidate.ordinal, side=side,
                first=candidate.first, first_price=candidate.first_price,
                second=candidate.second, second_price=candidate.second_price,
                touch_bars={candidate.first, candidate.second, center},
                last_touch=center, max_deviation=deviation))
            book.promotions += 1
            events[touch_index] = 1.0
    # Every candidate gets exactly one third-pivot validation opportunity.
    book.pending[side] = []
    for old_bar, old_price in book.pivots[side]:
        book.pending[side].append(_Candidate(
            book.next_ordinal, side, old_bar, old_price, center, price))
        book.next_ordinal += 1
    book.pivots[side].append((center, price))
    book.confirmed_pivots += 1


def _advance_lines(book: _Book, t: int, high: float, low: float, close: float,
                   atr: float, band: float, seq_len: int, events: np.ndarray) -> None:
    for side in (SUPPORT, RESISTANCE):
        if side == SUPPORT:
            book.pending[side] = [c for c in book.pending[side] if close >= c.at(t)]
        else:
            book.pending[side] = [c for c in book.pending[side] if close <= c.at(t)]
    margin = band * atr
    kept: list[_Line] = []
    newly_broken: list[_Line] = []
    for line in book.lines:
        if line.mode == "active" and t - line.last_touch >= seq_len:
            continue
        level = line.at(t)
        if line.mode == "active":
            if line.side == SUPPORT and close < level - margin:
                direction = -1
            elif line.side == RESISTANCE and close > level + margin:
                direction = 1
            else:
                direction = 0
            if direction:
                line.mode = "broken"
                line.broken_at = t
                line.break_direction = direction
                events[17 if direction < 0 else 16] = 1.0
                book.last_break = t
                book.last_break_direction = direction
                book.first_breaks += 1
                newly_broken.append(line)
                kept.append(line)
                continue
            extreme = low if line.side == SUPPORT else high
            intact = close >= level if line.side == SUPPORT else close <= level
            in_band = abs(extreme - level) <= margin
            if in_band and intact and not line.in_band_before:
                line.touch_bars.add(t)
                line.last_touch = t
                events[15 if line.side == SUPPORT else 14] = 1.0
            line.in_band_before = in_band
            kept.append(line)
            continue
        if t - line.broken_at > RETEST_BARS:
            continue
        entered = ((high >= level - margin) if line.break_direction < 0
                   else (low <= level + margin))
        if not entered:
            kept.append(line)
            continue
        held = ((close < level) if line.break_direction < 0 else (close > level))
        event_index = (22 if held else 23) if line.break_direction < 0 else (20 if held else 21)
        events[event_index] = 1.0
        if not held:
            book.failed_retests += 1
            continue
        book.held_retests += 1
        line.side = -line.side
        line.mode = "active"
        line.broken_at = -1
        line.break_direction = 0
        line.touch_bars.add(t)
        line.last_touch = t
        line.in_band_before = True
        kept.append(line)
    if newly_broken:
        chosen = max(newly_broken, key=lambda line: (line.touches, -line.ordinal))
        events[18] = float(chosen.touches)
        events[19] = float(t - (chosen.first + PIVOT_RADIUS))
    book.lines = kept


def _channel(book: _Book, above: _Line | None, above_distance: float,
             below: _Line | None, below_distance: float, t: int, close: float,
             atr: float, band: float, seq_len: int) -> tuple[float, ...]:
    if (above is not None and below is not None
            and below.side == SUPPORT and above.side == RESISTANCE
            and below.slope > above.slope):
        upper = above.at(t)
        lower = below.at(t)
        width = upper - lower
        if width > 0:
            upper_intercept = above.first_price - above.slope * float(above.first)
            lower_intercept = below.first_price - below.slope * float(below.first)
            apex = (upper_intercept - lower_intercept) / (below.slope - above.slope)
            proximity = min(max(1.0 - (apex - t) / seq_len, 0.0), 1.0)
            position = min(max((close - lower) / width, 0.0), 1.0)
            return (1.0, width / atr, position,
                    (above.slope + below.slope) / (2.0 * atr), 1.0, proximity)
    slots = [(above_distance, 0, above), (below_distance, 1, below)]
    for _, _, line in sorted((s for s in slots if s[2] is not None),
                             key=lambda item: (item[0], item[1])):
        offset = line.first_price - line.slope * float(line.first)
        opposite = book.pivots[-line.side]
        projected_offsets = [price - line.slope * float(bar) for bar, price in opposite]
        if line.side == SUPPORT:
            options = [v for v in projected_offsets if v > offset]
            if not options:
                continue
            far = max(options)
            members = sum(far - v <= band * atr for v in options)
            lower_offset = offset
        else:
            options = [v for v in projected_offsets if v < offset]
            if not options:
                continue
            far = min(options)
            members = sum(v - far <= band * atr for v in options)
            lower_offset = far
        width = abs(far - offset)
        if members < 2 or width <= 0:
            continue
        lower = lower_offset + line.slope * t
        position = min(max((close - lower) / width, 0.0), 1.0)
        return (1.0, width / atr, position, line.slope / atr, 0.0, 0.0)
    return (0.0,) * 6


def _emit(book: _Book, t: int, close: float, atr: float,
          band: float, seq_len: int) -> np.ndarray:
    row = np.zeros(31, dtype=np.float64)
    upper: _Line | None = None
    lower: _Line | None = None
    upper_key = (math.inf, math.inf)
    lower_key = (math.inf, math.inf)
    for line in book.lines:
        if line.mode != "active":
            continue
        level = line.at(t)
        if level >= close:
            row[0] += 1
            key = (level - close, line.ordinal)
            if key < upper_key:
                upper_key, upper = key, line
        else:
            row[7] += 1
            key = (close - level, line.ordinal)
            if key < lower_key:
                lower_key, lower = key, line
    for base, key, line in ((0, upper_key, upper), (7, lower_key, lower)):
        if line is None:
            continue
        row[base+1] = key[0] / atr
        row[base+2] = line.slope / atr
        row[base+3] = line.touches
        row[base+4] = t - (line.first + PIVOT_RADIUS)
        row[base+5] = t - line.last_touch
        row[base+6] = line.max_deviation
    row[24] = (book.last_break_direction * (t - book.last_break)
               if book.last_break >= 0 else math.nan)
    row[25:31] = _channel(book, upper, upper_key[0], lower,
                          lower_key[0], t, close, atr, band, seq_len)
    return row


def oracle_rows(high: Sequence[float], low: Sequence[float], close: Sequence[float],
                atr: Sequence[float], *, seq_len: int, band_atr: float,
                witness_positions: Sequence[int]) -> tuple[np.ndarray, dict[str, int]]:
    """Replay a complete D1 prefix and return 31 raw values at named bars.

    ``witness_positions`` are positions in the full D1 prefix, not cache rows.
    Source/cutoff/hash validation belongs to the caller's capped audit plan.
    """
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    a = np.asarray(atr, dtype=np.float64)
    _require(h.ndim == l.ndim == c.ndim == a.ndim == 1
             and len(h) == len(l) == len(c) == len(a) > 0, "SHAPE")
    _require(np.all(np.isfinite(h)) and np.all(np.isfinite(l))
             and np.all(np.isfinite(c)) and np.all(h > 0)
             and np.all(l > 0) and np.all(c > 0)
             and np.all(h >= c) and np.all(l <= c) and np.all(h >= l), "PRICE")
    finite_atr = np.isfinite(a)
    _require(not np.isinf(a).any() and np.all(a[finite_atr] > 0)
             and (not finite_atr.any() or finite_atr[np.flatnonzero(finite_atr)[0]:].all()),
             "ATR")
    _require(isinstance(seq_len, int) and seq_len > 2 * PIVOT_RADIUS + 2
             and math.isfinite(band_atr) and band_atr > 0, "CONSTANTS")
    positions = [int(v) for v in witness_positions]
    _require(bool(positions) and positions == sorted(set(positions))
             and positions[0] >= 0 and positions[-1] < len(h), "WITNESSES")
    position_set = set(positions)
    book = _Book()
    output: dict[int, np.ndarray] = {}
    for t in range(positions[-1] + 1):
        _prune(book, t, seq_len)
        events = np.zeros(31, dtype=np.float64)
        if t >= 2 * PIVOT_RADIUS:
            center = t - PIVOT_RADIUS
            if math.isfinite(a[center]):
                is_low, is_high = confirmed_center(h, l, t)
                if is_low:
                    _pivot(book, SUPPORT, center, float(l[center]),
                           float(a[center]), t, band_atr, events)
                if is_high:
                    _pivot(book, RESISTANCE, center, float(h[center]),
                           float(a[center]), t, band_atr, events)
        if math.isfinite(a[t]):
            _advance_lines(book, t, float(h[t]), float(l[t]), float(c[t]),
                           float(a[t]), band_atr, seq_len, events)
        if t in position_set:
            row = np.full(31, math.nan, dtype=np.float64)
            if t >= 2 * PIVOT_RADIUS + 2 and math.isfinite(a[t]):
                row = _emit(book, t, float(c[t]), float(a[t]), band_atr, seq_len)
                row[14:24] = events[14:24]
            output[t] = row
    return np.stack([output[t] for t in positions]), {
        "confirmed_pivots": book.confirmed_pivots,
        "promotions": book.promotions,
        "first_breaks": book.first_breaks,
        "held_retests": book.held_retests,
        "failed_retests": book.failed_retests,
    }
