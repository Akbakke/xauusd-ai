"""Research-only first positive executable quote event on causal M1 paths.

The first favorable event is separate from the existing first global MFE.
Callers explicitly choose same-side quote movement or gross quote markout;
only the latter includes spread. Fees, slippage and financing are excluded.
No trade or native-action authority is produced. The caller owns source
identity and split-specific admission.
"""
from __future__ import annotations

import math

import numpy as np

from gx1.contracts.entry_causal_m1_outcomes_v1 import (
    DYNAMIC_M1_PREFIX_QUERY_CHUNK_ROWS,
    DYNAMIC_M1_PREFIX_TERMINAL_ENDPOINT,
    DynamicM1PrefixIndex,
    query_dynamic_m1_prefix,
)

SCHEMA_VERSION = "gx1_first_positive_quote_target_research_v1"
HURDLE_SCHEMA_VERSION = "gx1_first_favorable_quote_hurdle_target_research_v1"
_EVENT_BASES = ("same_side_quote_move", "gross_quote_markout")


def _require(condition, reason):
    if not condition:
        raise RuntimeError("FIRST_POSITIVE_QUOTE_TARGET_" + reason)


def _require_favorable_bps(value):
    _require(not isinstance(value, bool) and isinstance(value, (int, float)), "FAVORABLE_BPS")
    try:
        value = float(value)
    except (OverflowError, ValueError):
        raise RuntimeError("FIRST_POSITIVE_QUOTE_TARGET_FAVORABLE_BPS") from None
    _require(math.isfinite(value) and value >= 0., "FAVORABLE_BPS")
    return value


def _crosses_favorable_bps(values, reference, favorable_bps, *, maximum):
    # Preserve the original strict price comparison exactly at zero. For a
    # nonzero hurdle, compare return in bps directly, avoiding rounding a price
    # barrier before deciding whether an exact threshold counts as a crossing.
    if favorable_bps == 0.:
        return values > reference if maximum else values < reference
    _require(np.all(reference > 0.), "NONPOSITIVE_ENTRY_REFERENCE")
    with np.errstate(over="ignore"):
        bps = ((values - reference) if maximum else (reference - values)) / reference * 10_000.
    return bps > favorable_bps


def first_positive_quote_target_contract(*, favorable_bps=0.):
    """Return the target semantics; this does not grant execution authority."""
    favorable_bps = _require_favorable_bps(favorable_bps)
    value = {
        "schema_version": SCHEMA_VERSION,
        "role": "research_target_only_no_native_action_or_exit_policy",
        "entry": "existing_exact_causal_m5_decision_m1_bid_ask_fill",
        "event_bases": {
            "same_side_quote_move": "long bid above entry bid; short ask below entry ask",
            "gross_quote_markout": "long bid above entry ask; short ask below entry bid",
        },
        "event_basis": "explicit_required_no_fitted_or_selected_threshold",
        "event_unit": "first_qualifying_nominal_m1_bucket_or_terminal_open",
        "entry_bar_order": "unknown_relative_to_actual_entry_quote_without_subminute_timestamps",
        "costs": "same_side_quote_move_excludes_spread; gross_quote_markout_includes_spread; neither_includes_commission_slippage_or_financing",
        "event_at_open": "first_recorded_same_side_quote_in_nominal_m1_bucket",
        "event_inside_bar": "nominal_m1_bucket_only_order_unknown",
        "quote_time": "subminute_quote_time_unavailable_even_for_recorded_open",
        "pre_event_adverse": "entry_notional_bps_bounds_under_nominal_open_entry_convention",
        "same_bar_high_low": "no_invented_order",
        "unknown_gap": "reuse_existing_dynamic_path_censoring",
        "entry_bar_censoring": "outcome_dependent_do_not_treat_discarded_hits_as_random_missing",
        "event_occurrence_bounds": "later_closed_bar_confirms_any_nominal_hit_is_possible_under_entry_quote_assumption",
        "absent_event": "right_censored_no_event_only_if_path_observed",
        "magnitude_targets": "unchanged_existing_terminal_gross_and_signed_mfe",
        "horizon_role": "observation_only_not_holding_time_or_stop",
        "future_targets_as_input": False,
        "test_or_control_authorized": False,
    }
    if favorable_bps:
        value.update({
            "schema_version": HURDLE_SCHEMA_VERSION,
            "event_basis": "explicit_required_fixed_strict_greater_than_favorable_bps",
            "favorable_bps": favorable_bps,
            "event_threshold": "signed_side_quote_return_bps_from_selected_entry_reference_strictly_above_favorable_bps",
            "pre_event_adverse": "entry_reference_bps_bounds_under_nominal_open_entry_convention_not_distance_from_hurdle",
        })
    return value


def _first_strict_crossing(range_index, starts, ends, barrier, *, maximum, favorable_bps=0.):
    """Locate the earliest qualifying bar via existing range-extreme queries."""
    _require(range_index.maximum is maximum and starts.ndim == ends.ndim == barrier.ndim == 1
             and starts.shape == ends.shape == barrier.shape
             and np.all(starts >= 0) and np.all(ends > starts)
             and np.isfinite(barrier).all(), "RANGE_INPUT")
    extreme = range_index.query(starts, ends)
    _require(np.all(extreme >= starts) and np.all(extreme < ends), "RANGE_EXTREME")
    values = range_index.values[extreme]
    qualifies = _crosses_favorable_bps(values, barrier, favorable_bps, maximum=maximum)
    result = np.full(len(starts), -1, dtype=np.int64)
    positions = np.flatnonzero(qualifies)
    if not len(positions):
        return result
    begin = starts[positions]
    lo = begin.copy()
    hi = ends[positions] - 1
    levels = barrier[positions]
    while np.any(lo < hi):
        active = np.flatnonzero(lo < hi)
        middle = lo[active] + (hi[active] - lo[active]) // 2
        extreme = range_index.query(begin[active], middle + 1)
        _require(np.all(extreme >= begin[active]) and np.all(extreme <= middle),
                 "PREFIX_EXTREME")
        values = range_index.values[extreme]
        earlier = _crosses_favorable_bps(values, levels[active], favorable_bps, maximum=maximum)
        hi[active] = np.where(earlier, middle, hi[active])
        lo[active] = np.where(earlier, lo[active], middle + 1)
    result[positions] = lo
    return result


def query_first_positive_quote(
    index: DynamicM1PrefixIndex, *, fill_surface, query_entry_rows, horizon_minutes,
    event_basis: str, endpoint_kind: str = DYNAMIC_M1_PREFIX_TERMINAL_ENDPOINT,
    query_chunk_rows: int = DYNAMIC_M1_PREFIX_QUERY_CHUNK_ROWS, favorable_bps: float = 0.,
):
    """Return first-event masks and pre-event adverse bounds for each side.

    Input binding, cutoff, gap, endpoint and row validation are owned by the
    existing dynamic-prefix query. Both outputs use precisely its path mask;
    the returned original terminal/MFE labels remain available unchanged.
    A no-event result is meaningful only when first_event_path_observed is true.
    All labels use the existing nominal-open convention; an entry-bar event
    cannot establish that the extreme followed the actual entry quote. Event
    occurrence bounds assume the nominal entry quote is the entry price and
    describe the recorded source; they are not execution guarantees.

    The event time is a nominal M1 bucket, not an exact quote timestamp; even
    recorded opens can be delayed within their bucket. A caller may supply a
    fixed nonnegative favorable_bps hurdle; the event requires strictly more
    than that side-specific return, while adverse bps remain measured from the
    entry reference. Zero exactly preserves the original event. The hurdle is
    an observation label, not a stop or a holding rule. No raw source is
    opened here and the function does not materialize one row per horizon.
    """
    _require(type(event_basis) is str and event_basis in _EVENT_BASES, "EVENT_BASIS")
    favorable_bps = _require_favorable_bps(favorable_bps)
    original = query_dynamic_m1_prefix(
        index, fill_surface=fill_surface, query_entry_rows=query_entry_rows,
        horizon_minutes=horizon_minutes, endpoint_kind=endpoint_kind,
        query_chunk_rows=query_chunk_rows,
    )
    entries = original["query_entry_rows"]
    n = len(entries)
    path_observed = original["path_observed"]
    _require(path_observed.shape == (n, 2)
             and np.array_equal(path_observed[:, 0], path_observed[:, 1]),
             "PATH_MASK")
    shape = (n, 2)
    nominal_happened = np.zeros(shape, dtype=np.bool_)
    event_lower = np.zeros(shape, dtype=np.bool_)
    event_upper = np.zeros(shape, dtype=np.bool_)
    entry_bar_ambiguous = np.zeros(shape, dtype=np.bool_)
    at_open = np.zeros(shape, dtype=np.bool_)
    source_rows = np.full(shape, -1, dtype=np.int64)
    nominal_bar_start_ns = np.full(shape, -1, dtype=np.int64)
    at_nominal_terminal_open = np.zeros(shape, dtype=np.bool_)
    lower_bps = np.full(shape, np.nan, dtype=np.float64)
    upper_bps = np.full(shape, np.nan, dtype=np.float64)
    clock = index.source.times.asi8
    quotes = index.source.values

    for left in range(0, n, query_chunk_rows):
        right = min(left + query_chunk_rows, n)
        active = np.arange(left, right, dtype=np.int64)
        active = active[path_observed[active, 0]]
        if not len(active):
            continue
        selected = fill_surface.iloc[entries[active]]
        starts = selected["entry_m1_row"].to_numpy(np.int64) - index.source_row_offset
        target = original["terminal_time_ns"][active]
        ends = np.searchsorted(clock, target, side="left")
        _require(np.all(starts >= 0) and np.all(ends > starts)
                 and np.all(ends <= len(clock)), "OBSERVED_RANGE")
        bid_entry = quotes["bid_open"][starts]
        ask_entry = quotes["ask_open"][starts]
        long_barrier = bid_entry if event_basis == "same_side_quote_move" else ask_entry
        short_barrier = ask_entry if event_basis == "same_side_quote_move" else bid_entry
        for side, event_name, adverse_name, opening_name, barrier, maximum in (
            (0, "bid_high", "bid_low", "bid_open", long_barrier, True),
            (1, "ask_low", "ask_high", "ask_open", short_barrier, False),
        ):
            hit = _first_strict_crossing(
                index._ranges[event_name], starts, ends, barrier, maximum=maximum,
                favorable_bps=favorable_bps,
            )
            if endpoint_kind == DYNAMIC_M1_PREFIX_TERMINAL_ENDPOINT:
                _require(np.all(ends < len(clock))
                         and np.array_equal(clock[ends], target), "TERMINAL_CLOCK")
                terminal_open = quotes[opening_name][ends]
                later_hit = _crosses_favorable_bps(
                    terminal_open, barrier, favorable_bps, maximum=maximum,
                )
                hit = np.where((hit < 0) & later_hit, ends, hit)
            has_event = hit >= 0
            # A full bar after the entry bucket is chronologically later than
            # an entry quote recorded in that first bucket. A hit only in the
            # entry bucket or at nominal terminal open is merely possible.
            confirmed = has_event & (hit > starts) & (hit < ends)
            uncertain_first = np.flatnonzero(
                has_event & (hit == starts) & (ends > starts + 1)
            )
            if len(uncertain_first):
                later = _first_strict_crossing(
                    index._ranges[event_name], starts[uncertain_first] + 1,
                    ends[uncertain_first], barrier[uncertain_first], maximum=maximum,
                    favorable_bps=favorable_bps,
                )
                confirmed[uncertain_first] = later >= 0
            event_lower[active, side] = confirmed
            event_upper[active, side] = has_event
            _require(np.all(~confirmed | has_event), "EVENT_BOUNDS")
            if not has_event.any():
                continue
            local = np.flatnonzero(has_event)
            output_rows = active[local]
            event_row = hit[local]
            start = starts[local]
            end = ends[local]
            boundary = barrier[local]
            opened = quotes[opening_name][event_row]
            exact_open = _crosses_favorable_bps(opened, boundary, favorable_bps, maximum=maximum)
            _require(np.all((event_row != end) | exact_open), "TERMINAL_EVENT")
            previous = index._ranges[adverse_name].query(start, event_row)
            running = quotes[opening_name][start].copy()
            previous_valid = previous >= 0
            if maximum:
                running[previous_valid] = np.minimum(
                    running[previous_valid], quotes[adverse_name][previous[previous_valid]],
                )
                lower_price = np.minimum(running, opened)
                upper_price = lower_price.copy()
                inside = ~exact_open
                upper_price[inside] = np.minimum(
                    upper_price[inside], quotes[adverse_name][event_row[inside]],
                )
                lower = np.maximum(0., (boundary - lower_price) / boundary * 10_000.)
                upper = np.maximum(0., (boundary - upper_price) / boundary * 10_000.)
            else:
                running[previous_valid] = np.maximum(
                    running[previous_valid], quotes[adverse_name][previous[previous_valid]],
                )
                lower_price = np.maximum(running, opened)
                upper_price = lower_price.copy()
                inside = ~exact_open
                upper_price[inside] = np.maximum(
                    upper_price[inside], quotes[adverse_name][event_row[inside]],
                )
                lower = np.maximum(0., (lower_price - boundary) / boundary * 10_000.)
                upper = np.maximum(0., (upper_price - boundary) / boundary * 10_000.)
            _require(np.isfinite(lower).all() and np.isfinite(upper).all()
                     and np.all(lower >= 0) and np.all(upper >= lower), "ADVERSE_BOUNDS")
            start_ns = clock[event_row]
            _require(np.all(start_ns <= target[local])
                     and np.all((event_row != end) | (start_ns == target[local])),
                     "EVENT_AFTER_HORIZON")
            nominal_happened[output_rows, side] = True
            entry_bar_ambiguous[output_rows, side] = event_row == start
            at_open[output_rows, side] = exact_open
            source_rows[output_rows, side] = event_row + index.source_row_offset
            nominal_bar_start_ns[output_rows, side] = start_ns
            at_nominal_terminal_open[output_rows, side] = event_row == end
            lower_bps[output_rows, side] = lower
            upper_bps[output_rows, side] = upper

    result = {
        "schema_version": HURDLE_SCHEMA_VERSION if favorable_bps else SCHEMA_VERSION,
        "contract": first_positive_quote_target_contract(favorable_bps=favorable_bps),
        "event_basis": event_basis,
        "original_dynamic_prefix": original,
        "first_event_path_observed": path_observed.copy(),
        "first_event_nominal_happened": nominal_happened,
        "first_event_occurrence_lower": event_lower,
        "first_event_occurrence_upper": event_upper,
        "first_event_entry_bar_order_unknown": entry_bar_ambiguous,
        "first_event_at_open": at_open,
        "first_event_source_row": source_rows,
        "first_event_nominal_bar_start_ns": nominal_bar_start_ns,
        "first_event_at_nominal_terminal_open": at_nominal_terminal_open,
        "nominal_pre_event_adverse_lower_bps": lower_bps,
        "nominal_pre_event_adverse_upper_bps": upper_bps,
    }
    if favorable_bps:
        result["favorable_bps"] = favorable_bps
    for group in (original, result):
        for value in group.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
    return result
