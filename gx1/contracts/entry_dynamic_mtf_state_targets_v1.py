"""Causal local XAU state targets; no forecast-origin price and no trade rule.

The sole numerical movement choice is the operator's candidate 0.5 ATR(14).
One rolling native-TF duration defines quietness, not a maximum state duration.
OHLC can establish an ending direction without establishing intraminute order.
Wide-bar interior endpoints remain unknown; no extrema order is invented.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType

import numpy as np
import pandas as pd

from gx1.contracts.entry_exit_feature_base_v1 import EXIT_DECISION_BAR_SECONDS
from gx1.contracts.entry_gc_mtf_history_v1 import TEST_BOUNDARY_NS
from gx1.features.htf_features import (
    MULTI_TF_TIMEFRAMES, MULTI_TF_SHIFT, MULTI_TF_RESAMPLE_ORIGIN_CONTRACT,
    _resample_ohlc_for_model_native_scalars, _load_verified_cache_npy,
)
from gx1.features.technical_indicators_v1 import (
    EMA50_200_ATR_PERIOD, technical_indicator_contract_metadata, wilder_atr,
)
from gx1.utils.artifact_primitives_v1 import canonical_json_sha256, sha256_file

SCHEMA_VERSION = "gx1_dynamic_mtf_state_targets_v1"
TIMEFRAMES = tuple(MULTI_TF_TIMEFRAMES)
STATE_NAMES = ("DOWN", "FLAT", "UP")
DOWN, FLAT, UP = range(len(STATE_NAMES))
UNKNOWN = -1
MINIMUM_MOVE_ATR = 0.5
MINUTE_NS = EXIT_DECISION_BAR_SECONDS * 10**9
LOCAL_WINDOW_M1_BARS = MappingProxyType({
    tf: int(MULTI_TF_SHIFT[tf].value // MINUTE_NS) for tf in TIMEFRAMES
})
LEAD_MINUTES = tuple(range(1, int(MULTI_TF_SHIFT["D1"].value // MINUTE_NS) + 1))
ATR_UNAVAILABLE = 1
LOCAL_SPAN_INCOMPLETE = 2
UNKNOWN_SOURCE_GAP = 4
INTRAMINUTE_ORDER_UNKNOWN = 8
DIRECTION_UNRESOLVED = 16
QUERY_UNAVAILABLE = 32
IMMATURE = 64
SERIES_ARRAY_DTYPES = {
    "validity_ns": "int64", "source_row": "int64",
    "state": "int8", "observed": "bool", "path_observed": "bool",
    "maturity_ns": "int64", "threshold": "float64", "local_range": "float64",
    "reason": "uint8", "atr_available_ns": "int64",
    "parent_rows": "int64", "issue_ns": "int64", "issue_source_index": "int64",
}
_VECTOR_ARRAYS = frozenset(("validity_ns", "source_row", "parent_rows", "issue_ns", "issue_source_index"))


def _require(ok, reason):
    if not ok:
        raise RuntimeError("DYNAMIC_MTF_STATE_" + reason)


def _seal(value):
    return {**value, "contract_sha256": canonical_json_sha256(value)}


def _ref(value):
    _require(isinstance(value, Mapping) and set(value) == {"path", "sha256"}, "REF_KEYS")
    path, digest = value["path"], value["sha256"]
    _require(isinstance(path, str) and Path(path).is_absolute() and ".." not in Path(path).parts
             and str(Path(path)) == path and "TEST" not in Path(path).parts
             and isinstance(digest, str) and len(digest) == 64
             and all(c in "0123456789abcdef" for c in digest), "REF")
    return dict(value)


def _cutoff(value):
    _require(type(value) is int and 0 < value <= TEST_BOUNDARY_NS and value % MINUTE_NS == 0, "CUTOFF")
    return value


def default_trial_lead_minutes():
    """This trial covers one D1 duration; the state process has no duration cap."""
    return LEAD_MINUTES


def extended_dense_lead_minutes(last_minute):
    """A declared longer one-minute query grid, never a trade-duration rule.

    Refuse a grid whose JSON array alone exceeds the existing 32 MiB private
    publication ceiling before constructing it. Native memory, output storage,
    source coverage and runtime still need separate source-bound qualification.
    """
    _require(type(last_minute) is int and last_minute > LEAD_MINUTES[-1], "EXTENDED_HORIZON")
    _require(last_minute <= 33554432, "EXTENDED_GRID_METADATA_BUDGET")
    # The existing report's accumulated FP32 coherence tolerance must remain
    # below the maximum possible probability difference; otherwise it is vacuous.
    _require(last_minute * len(STATE_NAMES) * np.finfo(np.float32).eps < 1.0, "EXTENDED_COHERENCE_BOUND")
    digits, start = 0, 1
    while start <= last_minute:
        stop = min(last_minute, start * 10 - 1)
        digits += (stop - start + 1) * len(str(start))
        start *= 10
    json_bytes = digits + last_minute + 1  # numbers, commas and two brackets
    _require(json_bytes <= 33554432, "EXTENDED_GRID_METADATA_BUDGET")
    return tuple(range(1, last_minute + 1))


def dynamic_mtf_state_target_contract():
    return _seal({
        "schema_version": SCHEMA_VERSION,
        "timeframes": list(TIMEFRAMES), "state_names": list(STATE_NAMES), "unknown_storage": UNKNOWN,
        "minimum_move_atr": MINIMUM_MOVE_ATR, "atr_period": EMA50_200_ATR_PERIOD,
        "technical_indicator_owner": technical_indicator_contract_metadata(),
        "tf_span_ns": {tf: int(MULTI_TF_SHIFT[tf].value) for tf in TIMEFRAMES},
        "local_window_m1_bars": dict(LOCAL_WINDOW_M1_BARS),
        "local_window_clock": "observed_active_M1_bars; known_closures_do_not_advance_window",
        "local_range": "last_N_observed_M1_high_low_and_previous_close; preserve_known_closure_reopen_jump",
        "tf_resample_origin": dict(MULTI_TF_RESAMPLE_ORIGIN_CONTRACT),
        "price_basis": "native_XAU_mid_OHLC_not_averaged_bid_ask_extrema",
        "source_resolution_seconds": EXIT_DECISION_BAR_SECONDS,
        "validity": "closed_M1_bar_start_plus_one_minute",
        "atr_availability": "last_completed_native_TF_bar_close_at_or_before_state_validity",
        "atr_gap_rule": "native_Wilder_series_over_observed_TF_bars_no_M1_gap_reset",
        "direction": "causal_extrema_directional_change_at_ge_threshold_no_issue_price",
        "uncertain_extrema": "wide_bar_interior_endpoint_resets_direction_unknown_no_guessed_extremum",
        "flat": "N_observed_M1_bars_since_source_start_or_last_unknown_gap_and_full_local_range_lt_threshold",
        "opposing_moves": "never_net_to_flat; observed_M1_state_series_retains_every_supported_change",
        "intraminute": "range_ge_threshold_order_unknown; close_equals_high_proves_UP_exact_peak; close_equals_low_proves_DOWN_exact_trough; interior_unknown_even_if_near_extreme",
        "intraminute_window": "ambiguity_ages_out_only_after_N_subsequent_observed_M1_bars",
        "unknown_reset": "unknown_gap_resets_direction_seed_at_observed_open; unresolved_extrema_at_observed_close",
        "post_gap": "new_confirmed_direction_can_be_known_early; flat_and_path_require_N_observed_bars_including_first_post_gap_bar",
        "maturity": "validity_time_no_retrospective_pivot_timestamp",
        "closure": "only_exact_same_source_clock_external_schedule; otherwise_all_gaps_unknown",
        "query": "exact_future_wall_minute_validity_lookup; missing_or_maturity_ge_cutoff_masked",
        "inputs": "future_state_threshold_observability_and_maturity_are_labels_never_issue_inputs",
        "trial_lead_minutes": list(LEAD_MINUTES),
        "trial_horizon_role": "computational_query_domain_not_state_duration_or_holding_rule",
        "reason_bits": {
            "atr_unavailable": ATR_UNAVAILABLE, "local_span_incomplete": LOCAL_SPAN_INCOMPLETE,
            "unknown_source_gap": UNKNOWN_SOURCE_GAP, "intraminute_order_unknown": INTRAMINUTE_ORDER_UNKNOWN,
            "direction_unresolved": DIRECTION_UNRESOLVED, "query_unavailable": QUERY_UNAVAILABLE,
            "immature": IMMATURE,
        },
        "reason_semantics": "diagnostic_flags; observed_is_state_mask_and_path_observed_is_local_path_mask",
    })


def require_dynamic_mtf_state_target_contract(value):
    expected = dynamic_mtf_state_target_contract()
    _require(isinstance(value, Mapping) and dict(value) == expected, "TARGET_CONTRACT")
    return expected


def _clock(frame, seconds):
    _require(isinstance(frame, pd.DataFrame) and "time" in frame, "FRAME_CLOCK")
    times = pd.DatetimeIndex(pd.to_datetime(frame["time"], utc=True, errors="coerce")).as_unit("ns")
    _require(len(times) > 0 and not times.hasnans and times.is_unique and times.is_monotonic_increasing
             and np.all(times.asi8 % (seconds * 10**9) == 0), "CLOCK")
    return times


def _ohlc(frame):
    _require(all(name in frame for name in ("open", "high", "low", "close")), "OHLC_COLUMNS")
    values = {name: frame[name].to_numpy(dtype=np.float64, copy=True) for name in ("open", "high", "low", "close")}
    _require(all(np.isfinite(a).all() and np.all(a > 0) for a in values.values()), "OHLC_FINITE")
    _require(np.all(values["high"] >= np.maximum(values["open"], values["close"]))
             and np.all(values["low"] <= np.minimum(values["open"], values["close"])), "OHLC_GEOMETRY")
    return values


def build_causal_tf_atr(m5_ohlc, *, cutoff_ns):
    """Reuse native M5 -> closed five-TF OHLC and canonical Wilder ATR14.

    This function never infers that an absent M1 minute is a market closure.
    ATR follows its existing observed-TF-bar definition independently of the
    state path's stricter gap mask.
    """
    cutoff_ns = _cutoff(cutoff_ns)
    return _build_input_tf_atr(m5_ohlc, cutoff_ns=cutoff_ns)


def _build_input_tf_atr(m5_ohlc, *, cutoff_ns):
    """Pure input ATR math; callers own admission, with no target queries or I/O."""
    _require(type(cutoff_ns) is int and cutoff_ns > 0 and cutoff_ns % MINUTE_NS == 0, "CUTOFF")
    times = _clock(m5_ohlc, int(MULTI_TF_SHIFT["M5"].total_seconds()))
    keep = times.asi8 + int(MULTI_TF_SHIFT["M5"].value) < cutoff_ns
    frame = m5_ohlc.loc[keep, ["open", "high", "low", "close"]].copy()
    frame.index = times[keep]
    _ohlc(frame)
    surfaces = {}
    for tf in TIMEFRAMES:
        bars = _resample_ohlc_for_model_native_scalars(frame, tf)
        available = bars.index.asi8 + int(MULTI_TF_SHIFT[tf].value)
        bars = bars.loc[available < cutoff_ns]
        available = available[available < cutoff_ns].astype(np.int64)
        atr = wilder_atr(bars.high, bars.low, bars.close, EMA50_200_ATR_PERIOD).to_numpy(np.float64)
        surfaces[tf] = {"available_ns": available, "atr": atr}
    return {"schema_version": SCHEMA_VERSION + "_causal_atr", "cutoff_ns": cutoff_ns,
            "definition_sha256": dynamic_mtf_state_target_contract()["contract_sha256"], "surfaces": surfaces}


def _known_closures(times, source_ref, closure_authority, exact_schedule):
    """Reuse the exact closure authorities; never infer a calendar from gaps."""
    gaps = np.flatnonzero(np.diff(times.asi8) != MINUTE_NS)
    if closure_authority is None and exact_schedule is None:
        return set(), None
    _require(closure_authority is not None and exact_schedule is not None, "CLOSURE_PAIR")
    from gx1.contracts.unified_exit_market_closure_authority_v1 import (
        m1_clock_sha256, require_market_closure_authority, require_exact_market_schedule,
    )
    schedule = require_exact_market_schedule(exact_schedule)
    _require(schedule["source_method"] in (
        "externally_sourced_exact_xau_utc_closure_intervals_v1", "observed_m1_clock_all_gaps_unknown_v1"),
        "INFERRED_CLOSURE_FORBIDDEN")
    authority = require_market_closure_authority(closure_authority,
        expected_m1_source_sha256=source_ref["sha256"], expected_m1_clock_sha256=m1_clock_sha256(times))
    _require(authority["m1_row_count"] == len(times) and len(authority["intervals"]) == len(gaps)
             and authority["schedule_sha256"] == schedule["schedule_sha256"], "CLOSURE_CLOCK")
    intervals = {(pd.Timestamp(x["start_utc"]).value, pd.Timestamp(x["end_utc_exclusive"]).value)
                 for x in schedule["intervals"]}
    known = set()
    for row, item in zip(gaps, authority["intervals"]):
        interval = (int(times.asi8[row]) + MINUTE_NS, int(times.asi8[row + 1]))
        _require(item["gap_after_m1_row"] == int(row)
                 and pd.Timestamp(item["previous_bar_start_utc"]).value == int(times.asi8[row])
                 and pd.Timestamp(item["next_bar_start_utc"]).value == int(times.asi8[row + 1])
                 and pd.Timestamp(item["closure_start_utc"]).value == interval[0]
                 and pd.Timestamp(item["closure_end_utc_exclusive"]).value == interval[1]
                 and bool(item["successor_across_gap_allowed"]) == (interval in intervals), "CLOSURE_INTERVAL")
        if item["successor_across_gap_allowed"]:
            known.add(int(row) + 1)
    return known, {"authority_sha256": authority["artifact_sha256"], "schedule_sha256": schedule["schedule_sha256"]}


def _step_direction(direction, anchor_low, anchor_high, high, low, close, threshold):
    """Conservative ending-state/extremum proof for one completed OHLC bar.

    For a wide bar, arbitrary extra reversals are compatible with its OHLC.
    Only closing exactly at H or L supplies the sufficient endpoint proof:
    after the opposite extreme, the final move reaches the threshold and
    ends at the global extreme. Its final direction and anchor are exact;
    the intraminute event count/order remains unknown. An interior close is
    masked, even when it lies less than one threshold from an extreme.
    Narrow bars cannot contain both threshold-sized opposite transitions.
    """
    ambiguous = high - low >= threshold
    if ambiguous:
        if close == high:
            return UP, high, high, True
        if close == low:
            return DOWN, low, low, True
        return UNKNOWN, close, close, True
    if direction == UP:
        lo, hi = max(anchor_low, high), max(anchor_high, high)
        if lo - low >= threshold:
            return DOWN, low, low, False
        if hi - low < threshold:
            return UP, lo, hi, False
        return UNKNOWN, close, close, True
    if direction == DOWN:
        lo, hi = min(anchor_low, low), min(anchor_high, low)
        if high - hi >= threshold:
            return UP, high, high, False
        if high - lo < threshold:
            return DOWN, lo, hi, False
        return UNKNOWN, close, close, True
    up, down = high - anchor_low >= threshold, anchor_high - low >= threshold
    if up and not down:
        return UP, high, high, False
    if down and not up:
        return DOWN, low, low, False
    if up and down:
        return UNKNOWN, close, close, True
    return UNKNOWN, min(anchor_low, low), max(anchor_high, high), False


@dataclass(frozen=True)
class DynamicMtfStateSeries:
    arrays: Mapping
    metadata: Mapping


def _series(arrays, metadata):
    for value in arrays.values():
        value.setflags(write=False)
    result = DynamicMtfStateSeries(MappingProxyType(dict(arrays)), MappingProxyType(dict(metadata)))
    require_dynamic_mtf_state_series(result)
    return result


def build_dynamic_mtf_state_series(m1_ohlc, *, tf_atr, source_ref, atr_source_ref, cutoff_ns,
                                   source_row_offset=0, closure_authority=None, exact_schedule=None):
    """Build one causal [source-validity,TF] tape, never a parent×lead cache."""
    cutoff_ns = _cutoff(cutoff_ns)
    source_ref, atr_source_ref = _ref(source_ref), _ref(atr_source_ref)
    _require(type(source_row_offset) is int and source_row_offset >= 0, "SOURCE_ROW_OFFSET")
    full_times = _clock(m1_ohlc, EXIT_DECISION_BAR_SECONDS)
    known, closure_binding = _known_closures(full_times, source_ref, closure_authority, exact_schedule)
    retained = int(np.searchsorted(full_times.asi8, cutoff_ns - MINUTE_NS, side="left"))
    _require(retained > 0, "EMPTY_PREFIX")
    times = full_times[:retained]
    values = _ohlc(m1_ohlc.iloc[:retained])
    validity = times.asi8 + MINUTE_NS
    n, width = retained, len(TIMEFRAMES)
    _require(tf_atr["schema_version"] == SCHEMA_VERSION + "_causal_atr"
             and tf_atr["definition_sha256"] == dynamic_mtf_state_target_contract()["contract_sha256"]
             and tf_atr["cutoff_ns"] == cutoff_ns and set(tf_atr["surfaces"]) == set(TIMEFRAMES), "ATR_BINDING")
    unknown_gap = np.zeros(n, dtype=bool)
    for row in np.flatnonzero(np.diff(times.asi8) != MINUTE_NS) + 1:
        unknown_gap[row] = int(row) not in known
    observed_row = np.arange(n, dtype=np.int64)
    last_unknown_gap_row = np.maximum.accumulate(np.where(unknown_gap, observed_row, -1))
    previous_close = np.r_[values["open"][0], values["close"][:-1]]
    previous_close[unknown_gap] = values["open"][unknown_gap]
    local_high, local_low = np.maximum(values["high"], previous_close), np.minimum(values["low"], previous_close)
    arrays = {
        "validity_ns": validity.astype(np.int64), "source_row": np.arange(source_row_offset, source_row_offset + n, dtype=np.int64),
        "state": np.full((n, width), UNKNOWN, np.int8), "observed": np.zeros((n, width), bool),
        "path_observed": np.zeros((n, width), bool), "maturity_ns": np.broadcast_to(validity[:, None], (n, width)).copy(),
        "threshold": np.zeros((n, width), np.float64), "local_range": np.zeros((n, width), np.float64),
        "reason": np.zeros((n, width), np.uint8), "atr_available_ns": np.zeros((n, width), np.int64),
        "parent_rows": np.empty(0, np.int64), "issue_ns": np.empty(0, np.int64), "issue_source_index": np.empty(0, np.int64),
    }
    atr_bindings = {}
    for col, tf in enumerate(TIMEFRAMES):
        surface = tf_atr["surfaces"][tf]
        at, atr = surface["available_ns"], surface["atr"]
        _require(isinstance(at, np.ndarray) and at.dtype == np.int64 and at.ndim == 1
                 and np.all(np.diff(at) > 0) and np.all(at < cutoff_ns)
                 and isinstance(atr, np.ndarray) and atr.dtype == np.float64 and atr.shape == at.shape
                 and np.all(np.isnan(atr) | (np.isfinite(atr) & (atr >= 0))), "ATR_SURFACE:" + tf)
        pos = np.searchsorted(at, validity, side="right") - 1
        safe = np.maximum(pos, 0)
        good = ((pos >= 0) & np.isfinite(atr[safe]) & (atr[safe] > 0)
                if len(at) else np.zeros(n, dtype=bool))
        theta = np.where(good, MINIMUM_MOVE_ATR * atr[safe], 0.0) if len(at) else np.zeros(n, dtype=np.float64)
        arrays["threshold"][:, col] = theta
        arrays["atr_available_ns"][:, col] = np.where(good, at[safe], 0) if len(at) else 0
        window = LOCAL_WINDOW_M1_BARS[tf]
        # A genuine closed M1 bar contributes one observation, irrespective of
        # elapsed time. Known pauses preserve the preceding bars and reopen jump.
        activity = (pd.Series(local_high).rolling(window, min_periods=1).max()
                    - pd.Series(local_low).rolling(window, min_periods=1).min()).to_numpy(np.float64)
        arrays["local_range"][:, col] = activity
        full_span = observed_row + 1 >= window
        # The first post-gap bar is observation 1; N new bars restore support.
        gap_in_span = ((last_unknown_gap_row >= 0)
                       & (observed_row - last_unknown_gap_row + 1 < window))
        complete = full_span & ~gap_in_span
        reason = arrays["reason"][:, col]
        reason[~good] |= ATR_UNAVAILABLE
        reason[~full_span] |= LOCAL_SPAN_INCOMPLETE
        reason[gap_in_span] |= UNKNOWN_SOURCE_GAP
        direction, lo, hi, last_ambiguous = UNKNOWN, float(values["open"][0]), float(values["open"][0]), -1
        for row in range(n):
            if unknown_gap[row] or not good[row]:
                direction = UNKNOWN
                lo = hi = float(values["open"][row])
            if not good[row]:
                continue
            direction, lo, hi, ambiguous = _step_direction(
                direction, lo, hi, float(values["high"][row]), float(values["low"][row]),
                float(values["close"][row]), float(theta[row]))
            if ambiguous:
                last_ambiguous = row
            order_known = last_ambiguous < 0 or row - last_ambiguous >= window
            arrays["path_observed"][row, col] = bool(complete[row] and order_known)
            if not order_known:
                reason[row] |= INTRAMINUTE_ORDER_UNKNOWN
            if complete[row] and activity[row] < theta[row]:
                arrays["state"][row, col], arrays["observed"][row, col] = FLAT, True
            elif direction in (DOWN, UP):
                arrays["state"][row, col], arrays["observed"][row, col] = direction, True
            else:
                reason[row] |= DIRECTION_UNRESOLVED
        atr_bindings[tf] = {
            "rows": len(at), "available_ns_sha256": hashlib.sha256(at.astype("<i8", copy=False).tobytes()).hexdigest(),
            "atr_sha256": hashlib.sha256(atr.astype("<f8", copy=False).tobytes()).hexdigest(),
        }
    metadata = {
        "schema_version": SCHEMA_VERSION + "_series", "definition": dynamic_mtf_state_target_contract(),
        "source_refs": {"m1": source_ref, "m5_atr": atr_source_ref}, "cutoff_ns": cutoff_ns,
        "supplied_source_rows": len(full_times), "source_row_offset": source_row_offset,
        "supplied_clock_sha256": hashlib.sha256(full_times.asi8.astype("<i8", copy=False).tobytes()).hexdigest(),
        "rows": n, "parents": 0, "atr_surfaces": atr_bindings, "closure_binding": closure_binding,
        "gap_count_scope": "computed_only_by_separately_bound_TRAIN_audit_not_global_later_aggregation",
        "target_rule_locked": False, "test_data_used": False,
    }
    return _series(arrays, metadata)


def require_dynamic_mtf_state_series(series):
    """Full validation at build/load boundaries, never once per training batch."""
    _require(isinstance(series, DynamicMtfStateSeries), "SERIES_TYPE")
    a, meta = series.arrays, series.metadata
    require_dynamic_mtf_state_target_contract(meta["definition"])
    _cutoff(meta["cutoff_ns"])
    _require(meta["schema_version"] == SCHEMA_VERSION + "_series" and meta["test_data_used"] is False
             and set(meta["source_refs"]) == {"m1", "m5_atr"}, "SERIES_METADATA")
    for item in meta["source_refs"].values():
        _ref(item)
    _require(set(a) == set(SERIES_ARRAY_DTYPES), "ARRAY_KEYS")
    n, p = meta["rows"], meta["parents"]
    _require(type(n) is int and n > 0 and type(p) is int and p >= 0, "ROW_COUNTS")
    for name, dtype in SERIES_ARRAY_DTYPES.items():
        size = p if name in ("parent_rows", "issue_ns", "issue_source_index") else n
        shape = (size,) if name in _VECTOR_ARRAYS else (size, len(TIMEFRAMES))
        _require(isinstance(a[name], np.ndarray) and a[name].dtype.name == dtype and a[name].shape == shape
                 and not a[name].flags.writeable, "ARRAY_HEADER:" + name)
    _require(np.all(np.diff(a["validity_ns"]) > 0) and np.all(a["validity_ns"] % MINUTE_NS == 0)
             and np.all(a["validity_ns"] < meta["cutoff_ns"])
             and np.all(np.diff(a["source_row"]) == 1), "SERIES_CLOCK")
    _require(np.array_equal(a["maturity_ns"], np.broadcast_to(a["validity_ns"][:, None], a["state"].shape)), "MATURITY")
    _require(np.isin(a["state"][a["observed"]], (DOWN, FLAT, UP)).all()
             and np.all(a["state"][~a["observed"]] == UNKNOWN), "STATE_MASK")
    _require(np.isfinite(a["threshold"]).all() and np.all(a["threshold"] >= 0)
             and np.isfinite(a["local_range"]).all() and np.all(a["local_range"] >= 0)
             and np.all(a["threshold"][a["observed"]] > 0)
             and np.all(a["atr_available_ns"] <= a["maturity_ns"]), "SCALE_CAUSALITY")
    flat = a["observed"] & (a["state"] == FLAT)
    _require(np.all(a["local_range"][flat] < a["threshold"][flat])
             and np.all((a["reason"][flat] & (LOCAL_SPAN_INCOMPLETE | UNKNOWN_SOURCE_GAP)) == 0), "FALSE_FLAT")
    if p:
        _require(np.all(np.diff(a["parent_rows"]) > 0) and np.all(a["parent_rows"] >= 0)
                 and np.all(np.diff(a["issue_ns"]) > 0)
                 and np.all(a["issue_ns"] % int(MULTI_TF_SHIFT["M5"].value) == 0)
                 and np.all((a["issue_source_index"] >= -1) & (a["issue_source_index"] < n)), "PARENTS")
        valid = a["issue_source_index"] >= 0
        _require(np.array_equal(a["validity_ns"][a["issue_source_index"][valid]], a["issue_ns"][valid]), "PARENT_LOOKUP")
    return series


def bind_dynamic_mtf_state_parents(series, *, parent_rows, issue_ns):
    _require(isinstance(series, DynamicMtfStateSeries), "SERIES_TYPE")
    for value in (parent_rows, issue_ns):
        _require(isinstance(value, np.ndarray) and value.dtype == np.int64 and value.ndim == 1, "PARENT_AXIS")
    _require(parent_rows.shape == issue_ns.shape and len(parent_rows) > 0, "PARENT_SHAPE")
    axis = series.arrays["validity_ns"]
    pos = np.searchsorted(axis, issue_ns)
    good = pos < len(axis)
    good[good] &= axis[pos[good]] == issue_ns[good]
    arrays = {**series.arrays, "parent_rows": parent_rows.copy(), "issue_ns": issue_ns.copy(),
              "issue_source_index": np.where(good, pos, -1).astype(np.int64)}
    return _series(arrays, {**series.metadata, "parents": len(parent_rows)})


def _historical_current_at_bound_positions(series, *, positions, issue_ns):
    """Gather current-only event-time evidence; no future-label arrays are returned."""
    arrays = series.arrays
    current = arrays["issue_source_index"][positions]
    axis = arrays["validity_ns"]
    _require(np.all((current >= -1) & (current < len(axis))), "CURRENT_SOURCE_INDEX")
    present = current >= 0
    _require(np.array_equal(axis[current[present]], issue_ns[present]), "CURRENT_SOURCE_VALIDITY")
    shape = (len(current), len(TIMEFRAMES))
    observed = np.zeros(shape, dtype=bool)
    path_observed = np.zeros(shape, dtype=bool)
    state = np.full(shape, UNKNOWN, dtype=np.int8)
    if np.any(present):
        selected = current[present]
        clock = issue_ns[present, None]
        _require(np.all(arrays["maturity_ns"][selected] <= clock), "CURRENT_MATURITY_CLOCK")
        _require(np.all(arrays["atr_available_ns"][selected] <= clock), "CURRENT_ATR_CLOCK")
        observed[present] = arrays["observed"][selected]
        path_observed[present] = arrays["path_observed"][selected]
        state[present] = arrays["state"][selected]
    state[~observed] = UNKNOWN
    return {"current_state": state, "current_observed": observed,
            "current_path_observed": path_observed}


def query_historical_dynamic_mtf_current_state(series, *, parent_rows, issue_ns, cutoff_ns):
    """Current-only input at bound historical event times, not ONLINE delivery proof.

    The series was source-bound and validated at its build/load boundary. This
    accessor returns no future targets, thresholds, prices or future masks. Its
    masks express historical event-time observability only: physical receipt,
    processing and restart parity must be established independently before use
    in a real-time forecast. State and local-path masks are independent.
    """
    _require(isinstance(series, DynamicMtfStateSeries), "SERIES_TYPE")
    cutoff_ns = _cutoff(cutoff_ns)
    _require(cutoff_ns <= series.metadata["cutoff_ns"], "QUERY_CUTOFF")
    _require(isinstance(parent_rows, np.ndarray) and parent_rows.dtype == np.int64
             and isinstance(issue_ns, np.ndarray) and issue_ns.dtype == np.int64
             and parent_rows.ndim == 1 and parent_rows.shape == issue_ns.shape and len(parent_rows) > 0
             and np.all(issue_ns < cutoff_ns) and np.all(issue_ns >= 0), "QUERY_PARENTS")
    stored = series.arrays["parent_rows"]
    _require(len(stored) > 0, "PARENT_BINDING_REQUIRED")
    positions = np.searchsorted(stored, parent_rows)
    _require(np.all(positions < len(stored)) and np.array_equal(stored[positions], parent_rows)
             and np.array_equal(series.arrays["issue_ns"][positions], issue_ns), "QUERY_PARENT_IDENTITY")
    result = _historical_current_at_bound_positions(series, positions=positions, issue_ns=issue_ns)
    return {"parent_rows": parent_rows.copy(), "issue_ns": issue_ns.copy(), **result,
            "clock_basis": "historical_event_time_only_not_online_delivery"}


def query_historical_dynamic_mtf_confirmed_run_age(series, *, parent_rows, issue_ns, cutoff_ns):
    """Preparation-only input: confirmed M1 state-run length at bound M5 issues.

    A run starts at two consecutive observed, path-observed M1 states whose
    classes differ. Its first row has count 1; this is an inclusive row count,
    not an exact intraminute transition time or the unknowable true duration.
    Unknown state/path, a missing M1 minute (including a documented closure),
    or a left-truncated run makes age unknown until another observed change.
    No future target is read and no native model consumes this candidate yet.
    Call once for a declared prefix, not once per model minibatch.
    """
    current = query_historical_dynamic_mtf_current_state(
        series, parent_rows=parent_rows, issue_ns=issue_ns, cutoff_ns=cutoff_ns)
    arrays = series.arrays
    axis = arrays["validity_ns"]
    # Slice every state/clock channel before access: later target rows cannot
    # affect a current feature even when the source-bound tape contains them.
    stop = int(np.searchsorted(axis, int(issue_ns.max()), side="right"))
    age = np.zeros((len(parent_rows), len(TIMEFRAMES)), dtype=np.int64)
    age_observed = np.zeros_like(age, dtype=bool)
    if stop:
        clock = axis[:stop]
        _require(np.all(arrays["maturity_ns"][:stop] <= clock[:, None]), "AGE_MATURITY_CLOCK")
        _require(np.all(arrays["atr_available_ns"][:stop] <= clock[:, None]), "AGE_ATR_CLOCK")
        positions = np.searchsorted(clock, issue_ns)
        present = positions < stop
        present[present] &= clock[positions[present]] == issue_ns[present]
        rows = np.arange(stop, dtype=np.int64)
        consecutive = np.r_[False, np.diff(clock) == MINUTE_NS]
        for tf in range(len(TIMEFRAMES)):
            state = arrays["state"][:stop, tf]
            eligible = arrays["observed"][:stop, tf] & arrays["path_observed"][:stop, tf]
            linked = consecutive & eligible & np.r_[False, eligible[:-1]]
            change = linked & np.r_[False, state[1:] != state[:-1]]
            last_change = np.maximum.accumulate(np.where(change, rows, -1))
            last_break = np.maximum.accumulate(np.where(~linked, rows, -1))
            known = eligible & (last_change > last_break)
            selected = positions[present]
            age_observed[present, tf] = known[selected]
            age[present, tf] = np.where(known[selected], rows[selected] - last_change[selected] + 1, 0)
    return {**current, "current_confirmed_run_age_minutes": age,
            "current_confirmed_run_age_observed": age_observed,
            "age_definition": "consecutive_observed_path_M1_rows_including_confirmed_change_row_v1"}


def query_dynamic_mtf_states(series, *, parent_rows, issue_ns, cutoff_ns, lead_minutes,
                             include_first_flat_activity=False):
    """Gather only the requested batch; all future prices/scales remain labels."""
    _require(isinstance(series, DynamicMtfStateSeries), "SERIES_TYPE")
    _require(type(include_first_flat_activity) is bool, "FIRST_ACTIVITY_MODE")
    cutoff_ns = _cutoff(cutoff_ns)
    _require(cutoff_ns <= series.metadata["cutoff_ns"], "QUERY_CUTOFF")
    leads = np.asarray(lead_minutes)
    _require(leads.dtype.kind in "iu" and leads.ndim == 1 and len(leads) > 0
             and np.all(leads > 0) and np.all(np.diff(leads.astype(np.int64)) > 0), "QUERY_LEADS")
    _require(isinstance(parent_rows, np.ndarray) and parent_rows.dtype == np.int64
             and isinstance(issue_ns, np.ndarray) and issue_ns.dtype == np.int64
             and parent_rows.ndim == 1 and parent_rows.shape == issue_ns.shape and len(parent_rows) > 0
             and np.all(issue_ns < cutoff_ns) and np.all(issue_ns >= 0)
             and np.all(issue_ns <= np.iinfo(np.int64).max - int(leads[-1]) * MINUTE_NS), "QUERY_PARENTS")
    stored = series.arrays["parent_rows"]
    _require(len(stored) > 0, "PARENT_BINDING_REQUIRED")
    positions = np.searchsorted(stored, parent_rows)
    _require(np.all(positions < len(stored)) and np.array_equal(stored[positions], parent_rows)
             and np.array_equal(series.arrays["issue_ns"][positions], issue_ns), "QUERY_PARENT_IDENTITY")
    wanted = issue_ns[:, None] + leads.astype(np.int64)[None, :] * MINUTE_NS
    axis = series.arrays["validity_ns"]
    rows = np.searchsorted(axis, wanted)
    safe = np.minimum(rows, len(axis) - 1)
    exists = (rows < len(axis)) & (axis[safe] == wanted)
    in_scope = wanted < cutoff_ns
    result = {"parent_rows": parent_rows.copy(), "issue_ns": issue_ns.copy(),
              "validity_ns": wanted, "lead_minutes": leads.astype(np.int64)}
    # Only gather exact, mature label rows. Masking after an eager gather can
    # touch the next fold at the cutoff or when a missing minute precedes it.
    available = exists & in_scope
    selected = rows[available]
    shape = (len(issue_ns), len(TIMEFRAMES), len(leads))
    for key, missing in (("state", UNKNOWN), ("observed", False),
                         ("path_observed", False), ("reason", QUERY_UNAVAILABLE)):
        values = np.full(shape, missing, dtype=SERIES_ARRAY_DTYPES[key])
        values.transpose(0, 2, 1)[available] = series.arrays[key][selected]
        result[key] = values
    result["state"][~result["observed"]] = UNKNOWN
    result["maturity_ns"] = np.broadcast_to(wanted[:, None, :], shape).copy()
    np.copyto(result["reason"], np.uint8(IMMATURE), where=~in_scope[:, None, :])
    if include_first_flat_activity:
        # A complete local range proves FLAT/non-FLAT without asserting the
        # unknown intraminute ordering or supplying any future value to forward.
        first_present = available[:, 0]
        first_rows = rows[first_present, 0]
        threshold = series.arrays["threshold"][first_rows]
        local_range = series.arrays["local_range"][first_rows]
        excluded = ATR_UNAVAILABLE | LOCAL_SPAN_INCOMPLETE | UNKNOWN_SOURCE_GAP | QUERY_UNAVAILABLE | IMMATURE
        eligible = (((result["reason"][first_present, :, 0] & excluded) == 0)
                    & np.isfinite(threshold) & (threshold > 0) & np.isfinite(local_range))
        activity_observed = np.zeros(shape[:2], dtype=bool)
        is_flat = np.zeros(shape[:2], dtype=bool)
        activity_observed[first_present] = eligible
        is_flat[first_present] = eligible & (local_range < threshold)
        first_state = result["state"][:, :, 0]
        known = result["observed"][:, :, 0]
        _require(np.all(~is_flat | (known & (first_state == FLAT)))
                 and np.all(~(activity_observed & known) | ((first_state == FLAT) == is_flat)),
                 "FIRST_ACTIVITY_STATE_CONSISTENCY")
        result["first_flat_activity_observed"] = activity_observed
        result["first_flat_activity_is_flat"] = is_flat
    result.update(_historical_current_at_bound_positions(series, positions=positions, issue_ns=issue_ns))
    return result


def _artifact_arrays(manifest, *, directory):
    _require(set(manifest["arrays"]) == set(SERIES_ARRAY_DTYPES), "ARTIFACT_INVENTORY")
    n, p = manifest["metadata"]["rows"], manifest["metadata"]["parents"]
    for name, dtype in SERIES_ARRAY_DTYPES.items():
        item = manifest["arrays"][name]
        _require(set(item) == {"path", "sha256", "size_bytes", "dtype", "shape"}, "ARTIFACT_DESCRIPTOR")
        _ref({k: item[k] for k in ("path", "sha256")})
        size = p if name in ("parent_rows", "issue_ns", "issue_source_index") else n
        shape = [size] if name in _VECTOR_ARRAYS else [size, len(TIMEFRAMES)]
        _require(item["path"] == str(directory / (name + ".npy")) and item["dtype"] == dtype
                 and item["shape"] == shape and type(item["size_bytes"]) is int and item["size_bytes"] > 0,
                 "ARTIFACT_HEADER:" + name)


def read_dynamic_mtf_state_series_metadata(binding, *, expected_source_refs, expected_cutoff_ns):
    """Verify small immutable metadata, no state arrays or market data reads."""
    from gx1.contracts.entry_gc_mtf_history_v1 import _read_gc_mtf_history_artifact_json
    binding = _ref(binding)
    path = Path(binding["path"])
    _require(path.name == "SERIES.json", "ARTIFACT_NAME")
    manifest = _read_gc_mtf_history_artifact_json(binding)
    _require(set(manifest) == {"schema_version", "metadata", "arrays"}
             and manifest["schema_version"] == SCHEMA_VERSION + "_artifact", "ARTIFACT_SCHEMA")
    meta = manifest["metadata"]
    require_dynamic_mtf_state_target_contract(meta["definition"])
    _require(meta["source_refs"] == {key: _ref(value) for key, value in expected_source_refs.items()}
             and set(expected_source_refs) == {"m1", "m5_atr"}
             and meta["cutoff_ns"] == _cutoff(expected_cutoff_ns)
             and meta["test_data_used"] is False, "ARTIFACT_BINDING")
    _artifact_arrays(manifest, directory=path.parent)
    return manifest


def _load_arrays(manifest, directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    arrays = {}
    try:
        for name, item in manifest["arrays"].items():
            arrays[name] = _load_verified_cache_npy(fd, name + ".npy",
                expected_sha256=item["sha256"], expected_size_bytes=item["size_bytes"], label=SCHEMA_VERSION + "/" + name)
    finally:
        os.close(fd)
    return _series(arrays, manifest["metadata"])


def load_dynamic_mtf_state_series(binding, *, expected_source_refs, expected_cutoff_ns):
    manifest = read_dynamic_mtf_state_series_metadata(binding,
        expected_source_refs=expected_source_refs, expected_cutoff_ns=expected_cutoff_ns)
    return _load_arrays(manifest, Path(binding["path"]).parent)


def save_dynamic_mtf_state_series(series, output_dir):
    """Atomic no-replace directory publication using existing immutable owners.

    Failed staging is retained for the retention owner; this API never deletes
    old or partially written data and never overwrites an existing artifact.
    """
    from gx1.contracts.immutable_event_authority_v1 import (
        _publish_bytes, _publish_file_noreplace, _fsync_directory,
    )
    require_dynamic_mtf_state_series(series)
    output_dir = Path(output_dir)
    _require(output_dir.is_absolute() and not output_dir.exists()
             and not any(p.is_symlink() for p in (output_dir, *output_dir.parents))
             and output_dir.parent.is_dir(), "OUTPUT_DIRECTORY")
    stage = Path(tempfile.mkdtemp(prefix="." + output_dir.name + ".staging.", dir=output_dir.parent))
    descriptors = {}
    for name, values in series.arrays.items():
        path = stage / (name + ".npy")
        with path.open("xb") as stream:
            np.save(stream, values, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        path.chmod(0o444)
        descriptors[name] = {"path": str(output_dir / path.name), "sha256": sha256_file(path),
                             "size_bytes": path.stat().st_size, "dtype": values.dtype.name, "shape": list(values.shape)}
    manifest = {"schema_version": SCHEMA_VERSION + "_artifact", "metadata": dict(series.metadata), "arrays": descriptors}
    _artifact_arrays(manifest, directory=output_dir)
    verified = _load_arrays(manifest, stage)
    del verified
    body = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    _publish_bytes(stage / "SERIES.json", body)
    (stage / "SERIES.json").chmod(0o444)
    _fsync_directory(stage)
    _publish_file_noreplace(stage, output_dir)
    _fsync_directory(output_dir.parent)
    return {"path": str(output_dir / "SERIES.json"), "sha256": hashlib.sha256(body).hexdigest()}
