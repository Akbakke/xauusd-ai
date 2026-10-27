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
CURVE_SCHEMA_VERSION = "gx1_direction_price_projection_minute_curve_v1"
DENSE_HORIZONS_MINUTES = tuple(range(1, 61))
REPORT_HORIZONS_MINUTES = (1, 5, 10, 15, 30, 60)
_MINUTE_NS = 60 * 10**9
_BAR_NS = 300 * 10**9


def _require(value, reason):
    if not value:
        raise RuntimeError("DIRECTION_PRICE_PROJECTION_" + reason)


def project_direction_price_endpoints(
    index, *, fill_surface, parent_rows, bar_start_ns, dynamic_entry_m1_row,
    horizons_minutes=HORIZONS,
):
    """Project explicit elapsed-minute endpoints before the strict split cutoff.

    Every horizon uses the same M5 decision and opening midpoint. The legacy
    default remains 1/5/15; callers may explicitly select ordered minutes 1..60.
    Nonlegacy selections have a separate schema, even if they have three columns.
    Midpoint bps, sign and observation masks are targets, not action/FLAT labels.

    The qualified index is reused, not rebuilt or exhaustively revalidated on
    every block. Parent IDs address the original complete fill surface; passed
    clocks/pointers must match it exactly. Only selected pre-cutoff entry and
    endpoint quotes are inspected. Source row offsets stay explicit.
    """
    _require(type(horizons_minutes) is tuple and 0 < len(horizons_minutes) <= 60
        and all(type(h) is int and 1 <= h <= 60 for h in horizons_minutes)
        and all(a < b for a, b in zip(horizons_minutes, horizons_minutes[1:])),
        "HORIZON_DOMAIN")
    width = len(horizons_minutes)
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
        and np.all(bar_start_ns <= np.iinfo(np.int64).max - _BAR_NS - max(horizons_minutes) * _MINUTE_NS),
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

    horizons = np.asarray(horizons_minutes, np.int64)
    times = decisions[:, None] + horizons[None, :] * _MINUTE_NS
    elapsed = np.broadcast_to(horizons * 60, (size, width)).copy()
    allowed = entry_ok[:, None] & (times < cutoff_ns)
    observed = np.zeros((size, width), np.bool_)
    terminal_rows = np.full((size, width), -1, np.int64)
    move = np.zeros((size, width), np.float64)
    returns = np.zeros_like(move)
    signs = np.zeros((size, width), np.int8)
    gross = np.full((size * width, 2), np.nan, np.float64)
    candidates = np.flatnonzero(allowed.reshape(-1))
    if len(candidates):
        target_times = times.reshape(-1)[candidates]
        end = np.searchsorted(clock_ns, target_times, side="left")
        present = np.flatnonzero(end < len(clock))
        exact = present[clock_ns[end[present]] == target_times[present]]
        output = candidates[exact]
        if len(output):
            ends = end[exact]
            starts = local_entry[output // width]
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
        "query_entry_rows": np.repeat(parent_rows, width),
        "horizon_minutes": np.tile(horizons, size),
        "terminal_time_ns": times.reshape(-1), "elapsed_wall_seconds": elapsed.reshape(-1),
        "terminal_gross_return_bps": gross,
        "terminal_gross_observed": np.repeat(observed.reshape(-1, 1), 2, axis=1)}
    result = {"schema_version": SCHEMA_VERSION if horizons_minutes == HORIZONS else CURVE_SCHEMA_VERSION,
        "horizons_minutes": horizons_minutes,
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


def compare_direction_native_midpoint(saved, *, entry_quotes, terminal_quotes, cutoff_ns):
    """Audit canonical native opens against the existing arithmetic midpoint.

    This pure comparison never changes the projector or its targets. The caller
    owns file identity, source offsets and quote gathering. Quotes carry actual
    source clocks/rows and an independent observation mask; absent terminal
    quotes use time/price zero and source row -1. Native ``open`` is not proof of
    a simultaneous executable quote at the candle's nominal start timestamp.
    """
    members = {
        "parent_rows", "bar_start_ns", "dynamic_entry_m1_row", "decision_ns",
        "terminal_time_ns", "elapsed_wall_seconds", "terminal_source_row",
        "observed", "mid_move", "mid_return_bps", "sign",
    }
    _require(isinstance(saved, dict) and set(saved) == members, "MIDPOINT_SAVED_MEMBERS")
    parents = saved["parent_rows"]
    _require(isinstance(parents, np.ndarray) and parents.dtype == np.int64
        and parents.ndim == 1 and len(parents) > 0, "MIDPOINT_PARENT_HEADER")
    size, shape = len(parents), (len(parents), len(HORIZONS))
    for name in members:
        dtype = (np.bool_ if name == "observed" else np.int8 if name == "sign"
            else np.float64 if name in ("mid_move", "mid_return_bps") else np.int64)
        expected_shape = (size,) if name in (
            "parent_rows", "bar_start_ns", "dynamic_entry_m1_row", "decision_ns") else shape
        value = saved[name]
        _require(isinstance(value, np.ndarray) and value.dtype == dtype
            and value.shape == expected_shape, "MIDPOINT_SAVED_HEADER:" + name)
    bars, decisions, observed = saved["bar_start_ns"], saved["decision_ns"], saved["observed"]
    _require(type(cutoff_ns) is int and 0 < cutoff_ns <= np.iinfo(np.int64).max,
        "MIDPOINT_CUTOFF")
    _require(np.array_equal(parents, np.arange(size, dtype=np.int64))
        and np.all(bars >= 0) and np.all(np.diff(bars) > 0)
        and np.all(bars % _BAR_NS == 0)
        and np.all(bars <= np.iinfo(np.int64).max - _BAR_NS - max(HORIZONS) * _MINUTE_NS)
        and np.array_equal(decisions, bars + _BAR_NS)
        and np.all(decisions < cutoff_ns), "MIDPOINT_PARENT_CLOCK")
    expected_terminal = decisions[:, None] + np.asarray(HORIZONS, np.int64) * _MINUTE_NS
    _require(np.array_equal(saved["terminal_time_ns"], expected_terminal)
        and np.array_equal(saved["elapsed_wall_seconds"],
            np.broadcast_to(np.asarray(HORIZONS, np.int64) * 60, shape)), "MIDPOINT_TERMINAL_CLOCK")
    _require(np.all(saved["dynamic_entry_m1_row"] >= 0)
        and np.all(np.diff(saved["dynamic_entry_m1_row"]) > 0)
        and np.all(saved["terminal_source_row"][observed] >= 0)
        and np.all(saved["terminal_source_row"][~observed] == -1)
        and np.all(expected_terminal[observed] < cutoff_ns), "MIDPOINT_SAVED_SOURCE_OR_CUTOFF")
    _require(np.isin(saved["sign"], (-1, 0, 1)).all()
        and all(np.isfinite(saved[name]).all() and np.all(saved[name][~observed] == 0)
            for name in ("mid_move", "mid_return_bps", "sign")), "MIDPOINT_SAVED_STORAGE")
    quote_members = {"time_ns", "source_row", "observed", "open", "bid_open", "ask_open"}
    for label, quotes, expected_shape in (
            ("ENTRY", entry_quotes, (size,)), ("TERMINAL", terminal_quotes, shape)):
        _require(isinstance(quotes, dict) and set(quotes) == quote_members,
            "MIDPOINT_" + label + "_MEMBERS")
        for name in quote_members:
            dtype = np.bool_ if name == "observed" else np.int64 if name in (
                "time_ns", "source_row") else np.float64
            _require(isinstance(quotes[name], np.ndarray) and quotes[name].dtype == dtype
                and quotes[name].shape == expected_shape, "MIDPOINT_" + label + "_HEADER:" + name)
        seen = quotes["observed"]
        _require(np.all(quotes["time_ns"][~seen] == 0)
            and np.all(quotes["source_row"][~seen] == -1)
            and all(np.isfinite(quotes[name][seen]).all() and np.all(quotes[name][seen] > 0)
                and np.all(quotes[name][~seen] == 0) for name in ("open", "bid_open", "ask_open"))
            and np.all(quotes["ask_open"][seen] >= quotes["bid_open"][seen]),
            "MIDPOINT_" + label + "_PRICE_OR_STORAGE")
    _require(entry_quotes["observed"].all()
        and np.array_equal(entry_quotes["time_ns"], decisions)
        and np.array_equal(entry_quotes["source_row"], saved["dynamic_entry_m1_row"]),
        "MIDPOINT_ENTRY_JOIN")
    _require(np.array_equal(terminal_quotes["observed"], observed)
        and np.array_equal(terminal_quotes["time_ns"][observed], expected_terminal[observed])
        and np.array_equal(terminal_quotes["source_row"], saved["terminal_source_row"])
        and np.all(terminal_quotes["source_row"][observed]
            > np.broadcast_to(entry_quotes["source_row"][:, None], shape)[observed]),
        "MIDPOINT_TERMINAL_JOIN")

    # First reproduce the legacy calculation using independently gathered quotes.
    legacy_move, legacy_bps = np.zeros(shape, np.float64), np.zeros(shape, np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        entry_mid = (entry_quotes["bid_open"] + entry_quotes["ask_open"]) / 2.0
        terminal_mid = (terminal_quotes["bid_open"] + terminal_quotes["ask_open"]) / 2.0
        start = np.broadcast_to(entry_mid[:, None], shape)[observed]
        change = terminal_mid[observed] - start
        legacy_move[observed], legacy_bps[observed] = change, 10000.0 * change / start
    _require(np.isfinite(entry_mid).all() and np.isfinite(terminal_mid[observed]).all()
        and np.isfinite(legacy_move).all() and np.isfinite(legacy_bps).all(), "MIDPOINT_LEGACY_FINITE")
    legacy_sign = np.sign(legacy_move).astype(np.int8)
    reproduced = {"observed": terminal_quotes["observed"], "mid_move": legacy_move,
        "mid_return_bps": legacy_bps, "sign": legacy_sign}
    _require(all(value.tobytes() == saved[name].tobytes() for name, value in reproduced.items()),
        "MIDPOINT_LEGACY_BIT_PARITY")

    native_move, native_bps = np.zeros(shape, np.float64), np.zeros(shape, np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        start = np.broadcast_to(entry_quotes["open"][:, None], shape)[observed]
        change = terminal_quotes["open"][observed] - start
        native_move[observed], native_bps[observed] = change, 10000.0 * change / start
    _require(np.isfinite(native_move).all() and np.isfinite(native_bps).all(), "MIDPOINT_NATIVE_FINITE")
    native_sign = np.sign(native_move).astype(np.int8)

    def difference_summary(native, legacy):
        count = len(native)
        if not count:
            return {"count": 0, "bit_identical": 0, "numerically_equal": 0,
                "mean_signed_difference": None, "mean_absolute_difference": None,
                "maximum_absolute_difference": None}
        with np.errstate(over="ignore", invalid="ignore"):
            difference = native - legacy
            mean, mean_abs, maximum = float(difference.mean()), float(np.abs(difference).mean()), float(np.abs(difference).max())
        _require(np.isfinite((mean, mean_abs, maximum)).all(), "MIDPOINT_DIFFERENCE_FINITE")
        return {"count": count, "bit_identical": int(np.count_nonzero(
            native.view(np.uint64) == legacy.view(np.uint64))),
            "numerically_equal": int(np.count_nonzero(native == legacy)),
            "mean_signed_difference": mean, "mean_absolute_difference": mean_abs,
            "maximum_absolute_difference": maximum}

    reports = []
    for j, horizon in enumerate(HORIZONS):
        seen = observed[:, j]
        old, new = legacy_sign[seen, j], native_sign[seen, j]
        confusion = np.bincount((old.astype(np.int64) + 1) * 3 + new + 1,
            minlength=9).reshape(3, 3)
        reports.append({"horizon_minutes": horizon, "parents": size,
            "observed": int(seen.sum()), "mature_missing": int(np.count_nonzero(
                ~seen & (expected_terminal[:, j] < cutoff_ns))),
            "cutoff_censored": int(np.count_nonzero(expected_terminal[:, j] >= cutoff_ns)),
            "entry_price_difference": difference_summary(entry_quotes["open"][seen], entry_mid[seen]),
            "terminal_price_difference": difference_summary(terminal_quotes["open"][seen, j], terminal_mid[seen, j]),
            "return_bps_difference": difference_summary(native_bps[seen, j], legacy_bps[seen, j]),
            "sign_confusion_legacy_by_native": confusion.tolist(),
            "nonzero_sign_flips": int(np.count_nonzero(old * new == -1)),
            "legacy_zero_to_native_nonzero": int(np.count_nonzero((old == 0) & (new != 0))),
            "legacy_nonzero_to_native_zero": int(np.count_nonzero((old != 0) & (new == 0)))})
    arrays = {"observed": observed.copy(), "native_mid_move": native_move,
        "native_mid_return_bps": native_bps, "native_sign": native_sign}
    for value in arrays.values():
        value.setflags(write=False)
    return {"schema_version": "gx1_direction_native_midpoint_parity_v1",
        "horizons_minutes": HORIZONS, "parent_rows": size, "cutoff_ns_exclusive": cutoff_ns,
        "legacy_bit_parity": {name: True for name in reproduced},
        "sign_order": (-1, 0, 1), "per_horizon": reports, "arrays": arrays}
