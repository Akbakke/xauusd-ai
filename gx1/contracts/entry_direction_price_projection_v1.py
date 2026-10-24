"""Pure endpoint price labels on an already qualified dynamic M1 source.

No reader, range-index construction, path query, costs, fit or admission lives
here. The caller owns canonical source/clock/SHA/pointer validation and TRAIN
scope. An observed zero price move remains distinct from missing supervision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gx1.contracts.entry_causal_m1_outcomes_v1 import (
    CausalM1QuoteSource, DynamicM1PrefixIndex, _SURFACE_COLUMNS,
    _entry_notional_pnl_bps, _require_surface, _utc_index,
)
from gx1.contracts.entry_direction_snapshot_data_v1 import HORIZONS, MAX_BLOCK_ROWS

SCHEMA_VERSION = "gx1_direction_price_projection_v1"
_MINUTE_NS = 60 * 10**9
_BAR_NS = 300 * 10**9


def _require(value, reason):
    if not value:
        raise RuntimeError("DIRECTION_PRICE_PROJECTION_" + reason)


def project_direction_price_endpoints(
    index, *, fill_surface, parent_rows, bar_start_ns, dynamic_entry_m1_row,
):
    """Project fixed1/5/15 endpoints without prices at or beyond the split cutoff.

    The qualified index is reused, not rebuilt or exhaustively revalidated on
    every block. Parent IDs address the original complete fill surface; passed
    clocks/pointers must match it exactly. Only selected pre-cutoff entry and
    endpoint quotes are inspected. Source row offsets stay explicit.
    """
    _require(isinstance(index, DynamicM1PrefixIndex)
        and isinstance(index.source, CausalM1QuoteSource), "QUALIFIED_INDEX")
    offset = index.source_row_offset
    _require(type(offset) is int and offset >= 0, "SOURCE_OFFSET")
    clock = index.source.times
    _require(isinstance(clock, pd.DatetimeIndex) and str(clock.tz) == "UTC"
        and len(clock) > 0 and clock.unit == "ns", "SOURCE_CLOCK_HEADER")
    cutoff = index.split_end_exclusive
    _require(isinstance(cutoff, pd.Timestamp) and str(cutoff.tz) == "UTC"
        and cutoff.floor("min") == cutoff, "CUTOFF")
    clock_ns, cutoff_ns = clock.asi8, int(cutoff.value)
    _require(int(clock_ns[-1]) < cutoff_ns
        and offset <= np.iinfo(np.int64).max - len(clock), "SOURCE_PREFIX")
    prices = index.source.values
    _require(all(name in prices and isinstance(prices[name], np.ndarray)
        and prices[name].dtype == np.float64 and prices[name].shape == (len(clock),)
        and not prices[name].flags.writeable for name in ("bid_open", "ask_open")),
        "SOURCE_PRICE_HEADERS")
    _require(isinstance(parent_rows, np.ndarray) and parent_rows.dtype == np.int64
        and parent_rows.ndim == 1 and 0 < len(parent_rows) <= MAX_BLOCK_ROWS
        and np.all(parent_rows >= 0) and np.all(np.diff(parent_rows) > 0), "PARENT_ROWS")
    size = len(parent_rows)
    _require(isinstance(fill_surface, pd.DataFrame)
        and tuple(fill_surface.columns) == _SURFACE_COLUMNS
        and int(parent_rows[-1]) < len(fill_surface), "FILL_SURFACE")
    _require(isinstance(bar_start_ns, np.ndarray) and bar_start_ns.dtype == np.int64
        and bar_start_ns.shape == (size,) and np.all(bar_start_ns >= 0)
        and np.all(bar_start_ns % _BAR_NS == 0) and np.all(np.diff(bar_start_ns) > 0)
        and np.all(bar_start_ns <= np.iinfo(np.int64).max - _BAR_NS - 15 * _MINUTE_NS),
        "BAR_CLOCK")
    _require(isinstance(dynamic_entry_m1_row, np.ndarray)
        and dynamic_entry_m1_row.dtype == np.int64
        and dynamic_entry_m1_row.shape == (size,), "POINTER_HEADER")
    selected = fill_surface.iloc[parent_rows]
    actual_bars = _utc_index(selected["time"], name="PRICE_M5", seconds=300).asi8
    decisions = bar_start_ns + _BAR_NS
    actual_decisions = _utc_index(selected["entry_decision_at"],
        name="PRICE_DECISION", seconds=60).asi8
    stored = selected["entry_m1_row"].to_numpy()
    _require(stored.dtype == np.int64 and np.array_equal(stored, dynamic_entry_m1_row),
        "POINTER_MISMATCH")
    _require(np.array_equal(actual_bars, bar_start_ns)
        and np.array_equal(actual_decisions, decisions), "FILL_CLOCK_MISMATCH")
    _require(selected["entry_fill_bound"].to_numpy().dtype == np.bool_, "FILL_MASK")

    # Never validate or index entry prices for a decision at/after the cutoff.
    entry_ok = np.zeros(size, np.bool_)
    local_entry = np.full(size, -1, np.int64)
    before = np.flatnonzero(decisions < cutoff_ns)
    if len(before):
        checked = _require_surface(selected.iloc[before])
        declared = checked["entry_fill_bound"].to_numpy(np.bool_)
        pointer = checked["entry_m1_row"].to_numpy(np.int64)
        _require(np.array_equal(pointer, dynamic_entry_m1_row[before]), "FILL_POINTER_PRECISION")
        local = pointer - offset
        within = (local >= 0) & (local < len(clock))
        _require(not np.any(declared & ~within), "FILL_SOURCE_ROW_MISMATCH")
        positions = np.flatnonzero(declared)
        if len(positions):
            source_rows = local[positions]
            _require(np.array_equal(clock_ns[source_rows], decisions[before[positions]]),
                "FILL_SOURCE_ROW_MISMATCH")
            _require(np.array_equal(prices["bid_open"][source_rows],
                checked["entry_bid"].to_numpy(np.float64)[positions])
                and np.array_equal(prices["ask_open"][source_rows],
                checked["entry_ask"].to_numpy(np.float64)[positions]), "FILL_QUOTE_MISMATCH")
        entry_ok[before] = declared
        local_entry[before] = np.where(declared, local, -1)

    horizons = np.asarray(HORIZONS, np.int64)
    times = decisions[:, None] + horizons[None, :] * _MINUTE_NS
    elapsed = np.broadcast_to(horizons * 60, (size, 3)).copy()
    allowed = entry_ok[:, None] & (times < cutoff_ns)
    observed = np.zeros((size, 3), np.bool_)
    terminal_rows = np.full((size, 3), -1, np.int64)
    move = np.zeros((size, 3), np.float64)
    returns = np.zeros_like(move)
    signs = np.zeros((size, 3), np.int8)
    gross = np.full((size * 3, 2), np.nan, np.float64)
    candidates = np.flatnonzero(allowed.reshape(-1))
    if len(candidates):
        target_times = times.reshape(-1)[candidates]
        end = np.searchsorted(clock_ns, target_times, side="left")
        present = np.flatnonzero(end < len(clock))
        exact = present[clock_ns[end[present]] == target_times[present]]
        output = candidates[exact]
        if len(output):
            ends = end[exact]
            starts = local_entry[output // 3]
            bid0, ask0 = prices["bid_open"][starts], prices["ask_open"][starts]
            bidT, askT = prices["bid_open"][ends], prices["ask_open"][ends]
            _require(all(np.isfinite(v).all() and np.all(v > 0)
                for v in (bid0, ask0, bidT, askT))
                and np.all(ask0 >= bid0) and np.all(askT >= bidT), "ENDPOINT_QUOTES")
            mid0, midT = (bid0 + ask0) / 2.0, (bidT + askT) / 2.0
            change = midT - mid0
            price_return = 10000.0 * change / mid0
            _require(np.isfinite(mid0).all() and np.isfinite(midT).all()
                and np.isfinite(change).all() and np.isfinite(price_return).all(), "FINITE_PRICE_LABEL")
            long_gross, short_gross = _entry_notional_pnl_bps(
                entry_bid=bid0, entry_ask=ask0, exit_bid=bidT, exit_ask=askT)
            _require(np.isfinite(long_gross).all() and np.isfinite(short_gross).all(), "FINITE_GROSS")
            observed.reshape(-1)[output] = True
            terminal_rows.reshape(-1)[output] = ends + offset
            move.reshape(-1)[output] = change
            returns.reshape(-1)[output] = price_return
            signs.reshape(-1)[output] = np.sign(change).astype(np.int8)
            gross[output] = np.column_stack((long_gross, short_gross))

    observations = {"endpoint_kind": "terminal_open_inclusive", "side_order": ("long", "short"),
        "query_entry_rows": np.repeat(parent_rows, 3),
        "horizon_minutes": np.tile(horizons, size),
        "terminal_time_ns": times.reshape(-1), "elapsed_wall_seconds": elapsed.reshape(-1),
        "terminal_gross_return_bps": gross,
        "terminal_gross_observed": np.repeat(observed.reshape(-1, 1), 2, axis=1)}
    result = {"schema_version": SCHEMA_VERSION, "horizons_minutes": HORIZONS,
        "cutoff_ns_exclusive": cutoff_ns, "source_row_offset": offset,
        "parent_rows": parent_rows.copy(), "bar_start_ns": bar_start_ns.copy(),
        "dynamic_entry_m1_row": dynamic_entry_m1_row.copy(), "decision_ns": decisions,
        "terminal_time_ns": times, "elapsed_wall_seconds": elapsed,
        "terminal_source_row": terminal_rows, "observed": observed,
        "mid_move": move, "mid_return_bps": returns, "sign": signs,
        "terminal_observations": observations}
    for group in (result, observations):
        for value in group.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
    return result
