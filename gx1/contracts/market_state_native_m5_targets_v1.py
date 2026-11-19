"""Pure direction targets on native closed M5 prices and observed higher bars."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from gx1.features.htf_features import (
    MULTI_TF_RESAMPLE_ORIGIN_CONTRACT, MULTI_TF_SHIFT, MULTI_TF_TIMEFRAMES,
    _resample_ohlc_for_model_native_scalars, multi_tf_resample,
)
from gx1.features.technical_indicators_v1 import EMA50_200_ATR_PERIOD, wilder_atr

TIMEFRAMES = tuple(MULTI_TF_TIMEFRAMES)
TICK_NS = 300_000_000_000
NATIVE_K = tuple(int(MULTI_TF_SHIFT[tf].value // TICK_NS) for tf in TIMEFRAMES)
MOVE_ATR = 0.5
DOWN, FLAT, UP, ACTIVE_UNKNOWN, UNKNOWN = 1, 2, 4, 5, 7


def _need(condition, reason):
    if not bool(condition):
        raise ValueError('NATIVE_M5_TARGET_' + reason)


def contract():
    value = {
        'schema_version': 'gx1_market_state_native_m5_targets_v1',
        'role': 'candidate_targets_no_training_or_trading_authority',
        'timeframes': list(TIMEFRAMES), 'native_query_k': list(NATIVE_K),
        'source_bar_seconds': 300, 'issue': 'native_bar_start_plus_300_seconds',
        'cutoff': 'exclusive_source_close_and_target_validity',
        'source_columns': ['open', 'high', 'low', 'close'],
        'tf_resample_origin': dict(MULTI_TF_RESAMPLE_ORIGIN_CONTRACT),
        'ATR': 'canonical_Wilder_ATR14_on_observed_native_M5_aggregates',
        'ATR_partial_bins': 'as_is_actual_M5_counts_reported_no_complete_market_coverage_assertion',
        'move_ATR': MOVE_ATR, 'local_observed_M5_bars': list(NATIVE_K),
        'S': 'local_range_and_conservative_ending_direction_at_exact_future_M5_close',
        'S_range': 'rolling_observed_high_low_including_previous_close_except_unknown_gap',
        'S_flat': 'complete_local_span_and_range_lt_half_causal_TF_ATR',
        'S_activity': 'complete_range_ge_threshold_or_conservative_direction_proof',
        'S_side': 'OHLC_ending_direction_no_intrabar_order_inference',
        'S_gap': 'every_non_300s_gap_resets_direction_and_local_span_first_new_bar_counts_one',
        'F': 'exact_future_M5_close_minus_issue_close_scaled_by_TF_ATR_at_issue',
        'F_side': 'absolute_change_ge_half_ATR_else_FLAT_NET',
        'F_gap': 'reported_endpoints_do_not_claim_intermediate_path',
        'unknown_numeric_values': 'NaN_and_false_eligibility',
        'maturity': 'validity_strictly_before_explicit_fold_upper',
        'allowed_bits': {'DOWN': DOWN, 'FLAT_OR_FLAT_NET': FLAT, 'UP': UP,
                         'ACTIVE_SIDE_UNKNOWN': ACTIVE_UNKNOWN, 'UNKNOWN': UNKNOWN},
        'activity_mask': 'eligible_and_mature_and_bits_not_7',
        'side_mask': 'eligible_and_mature_and_bits_in_1_4',
    }
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def _owned(array):
    return np.frombuffer(np.ascontiguousarray(array).tobytes(),
                         dtype=array.dtype).reshape(array.shape)


def _ending_direction(side, low_anchor, high_anchor, high, low, close, threshold):
    # Resolution-independent OHLC proof retained from the earlier state algorithm.
    # A wide bar with an interior close cannot prove its final directional leg.
    if high - low >= threshold:
        if close == high:
            return UP, high, high
        if close == low:
            return DOWN, low, low
        return UNKNOWN, close, close
    if side == UP:
        lo, hi = max(low_anchor, high), max(high_anchor, high)
        if lo - low >= threshold:
            return DOWN, low, low
        if hi - low < threshold:
            return UP, lo, hi
        return UNKNOWN, close, close
    if side == DOWN:
        lo, hi = min(low_anchor, low), min(high_anchor, low)
        if high - hi >= threshold:
            return UP, high, high
        if high - lo < threshold:
            return DOWN, lo, hi
        return UNKNOWN, close, close
    up, down = high - low_anchor >= threshold, high_anchor - low >= threshold
    if up and not down:
        return UP, high, high
    if down and not up:
        return DOWN, low, low
    if up and down:
        return UNKNOWN, close, close
    return UNKNOWN, min(low_anchor, low), max(high_anchor, high)


@dataclass(frozen=True)
class NativeM5Targets:
    issue_ns: np.ndarray
    source_row: np.ndarray
    close: np.ndarray
    atr: np.ndarray
    atr_available_ns: np.ndarray
    tf_bar_m5_counts: np.ndarray
    source_gap: np.ndarray
    s_allowed: np.ndarray
    s_span_complete: np.ndarray
    s_local_range: np.ndarray
    source_ref: MappingProxyType
    cutoff_ns: int

    def _query(self, issues, query_k, fold_upper_ns):
        _need(isinstance(issues, np.ndarray) and issues.dtype == np.int64
              and issues.ndim == 1 and len(issues) > 0
              and isinstance(query_k, np.ndarray) and query_k.dtype == np.int64
              and query_k.shape == (len(issues), len(TIMEFRAMES))
              and np.all((query_k >= 1) & (query_k <= np.asarray(NATIVE_K)))
              and type(fold_upper_ns) is int and 0 < fold_upper_ns <= self.cutoff_ns,
              'QUERY')
        at = np.searchsorted(self.issue_ns, issues)
        _need(np.all(at < len(self.issue_ns))
              and np.array_equal(self.issue_ns[at], issues)
              and np.all(issues < fold_upper_ns)
              and np.all(issues <= np.iinfo(np.int64).max - max(NATIVE_K) * TICK_NS),
              'ISSUE_AXIS')
        validity = issues[:, None] + query_k * TICK_NS
        mature = validity < fold_upper_ns
        endpoint = np.full(query_k.shape, -1, np.int64)
        wanted = validity[mature]
        positions = np.searchsorted(self.issue_ns, wanted)
        present = positions < len(self.issue_ns)
        present[present] &= self.issue_ns[positions[present]] == wanted[present]
        endpoint.flat[np.flatnonzero(mature)[present]] = positions[present]
        return at, validity, mature, endpoint

    def gather_state(self, issue_ns, query_k, *, fold_upper_ns):
        _, validity, mature, endpoint = self._query(issue_ns, query_k, fold_upper_ns)
        allowed = np.full(query_k.shape, UNKNOWN, np.uint8)
        present = endpoint >= 0
        columns = np.broadcast_to(np.arange(len(TIMEFRAMES)), query_k.shape)
        allowed[present] = self.s_allowed[endpoint[present], columns[present]]
        return {
            'issue_ns': issue_ns.copy(), 'query_k': query_k.copy(),
            'validity_ns': validity, 'source_row': endpoint,
            'allowed': allowed, 'eligible': present & (allowed != UNKNOWN),
            'mature': mature,
        }

    def gather_displacement(self, issue_ns, *, fold_upper_ns):
        query = np.broadcast_to(np.asarray(NATIVE_K, np.int64),
                                (len(issue_ns), len(TIMEFRAMES))).copy()
        at, validity, mature, endpoint = self._query(issue_ns, query, fold_upper_ns)
        present = endpoint >= 0
        scale = self.atr[at]
        eligible = present & np.isfinite(scale) & (scale > 0)
        delta = np.full(query.shape, np.nan, np.float64)
        origin = np.broadcast_to(self.close[at, None], query.shape)
        delta[eligible] = self.close[endpoint[eligible]] - origin[eligible]
        magnitude = np.full(query.shape, np.nan, np.float64)
        magnitude[eligible] = np.abs(delta[eligible]) / scale[eligible]
        allowed = np.full(query.shape, UNKNOWN, np.uint8)
        allowed[eligible] = FLAT
        active = eligible & (magnitude >= MOVE_ATR)
        allowed[active & (delta < 0)] = DOWN
        allowed[active & (delta > 0)] = UP
        gaps = np.zeros(query.shape, np.bool_)
        row_distance = endpoint - at[:, None]
        gaps[present] = row_distance[present] != query[present]
        _need(np.isfinite(delta[eligible]).all() and np.isfinite(magnitude[eligible]).all(),
              'DISPLACEMENT_NUMERIC')
        return {
            'issue_ns': issue_ns.copy(), 'query_k': query, 'validity_ns': validity,
            'source_row': endpoint, 'allowed': allowed, 'eligible': eligible,
            'mature': mature, 'endpoint_present': present, 'gap_between': gaps,
            'net_change': delta, 'abs_change_over_atr': magnitude,
        }


def build_native_m5_targets(bar_start_ns, ohlc, *, source_ref, cutoff_ns):
    """Build source-clock targets without IO, historical selections or fitting."""
    _need(isinstance(bar_start_ns, np.ndarray) and bar_start_ns.dtype == np.int64
          and bar_start_ns.ndim == 1 and len(bar_start_ns) > 0
          and isinstance(ohlc, np.ndarray) and ohlc.dtype == np.float64
          and ohlc.shape == (len(bar_start_ns), 4)
          and np.all(bar_start_ns >= 0) and np.all(bar_start_ns % TICK_NS == 0)
          and np.all(np.diff(bar_start_ns) > 0)
          and type(cutoff_ns) is int and 0 < cutoff_ns <= np.iinfo(np.int64).max
          and cutoff_ns % TICK_NS == 0
          and np.all(bar_start_ns < cutoff_ns - TICK_NS), 'SOURCE_CLOCK')
    _need(np.isfinite(ohlc).all() and np.all(ohlc > 0)
          and np.all(ohlc[:, 1] >= np.maximum(ohlc[:, 0], ohlc[:, 3]))
          and np.all(ohlc[:, 2] <= np.minimum(ohlc[:, 0], ohlc[:, 3]))
          and np.all(ohlc[:, 1] >= ohlc[:, 2]), 'OHLC')
    _need(type(source_ref) is dict and set(source_ref) == {'path', 'sha256'}
          and isinstance(source_ref['path'], str) and Path(source_ref['path']).is_absolute()
          and '..' not in Path(source_ref['path']).parts
          and isinstance(source_ref['sha256'], str) and len(source_ref['sha256']) == 64
          and all(c in '0123456789abcdef' for c in source_ref['sha256']), 'SOURCE_REF')
    frame = pd.DataFrame(ohlc.copy(), columns=('open', 'high', 'low', 'close'),
                         index=pd.to_datetime(bar_start_ns, utc=True))
    issue = bar_start_ns + TICK_NS
    n = len(issue)
    atr = np.full((n, len(TIMEFRAMES)), np.nan, np.float64)
    available_at = np.zeros((n, len(TIMEFRAMES)), np.int64)
    counts = np.zeros((n, len(TIMEFRAMES)), np.int64)
    for j, tf in enumerate(TIMEFRAMES):
        bars = _resample_ohlc_for_model_native_scalars(frame, tf)
        available = bars.index.asi8 + int(MULTI_TF_SHIFT[tf].value)
        observed = multi_tf_resample(frame, tf).size().reindex(bars.index).to_numpy(np.int64)
        values = wilder_atr(bars.high, bars.low, bars.close, EMA50_200_ATR_PERIOD).to_numpy(np.float64)
        at = np.searchsorted(available, issue, side='right') - 1
        known = at >= 0
        atr[known, j] = values[at[known]]
        available_at[known, j] = available[at[known]]
        counts[known, j] = observed[at[known]]
    _need(np.all(available_at <= issue[:, None]), 'ATR_AVAILABILITY')
    gap = np.r_[False, np.diff(issue) != TICK_NS]
    rows = np.arange(n, dtype=np.int64)
    last_reset = np.maximum.accumulate(np.where(gap, rows, 0))
    previous_close = np.r_[ohlc[0, 0], ohlc[:-1, 3]]
    previous_close[gap] = ohlc[gap, 0]
    high = np.maximum(ohlc[:, 1], previous_close)
    low = np.minimum(ohlc[:, 2], previous_close)
    allowed = np.full(atr.shape, UNKNOWN, np.uint8)
    span_complete = np.zeros(atr.shape, np.bool_)
    local_range = np.zeros(atr.shape, np.float64)
    for j, window in enumerate(NATIVE_K):
        local_range[:, j] = (
            pd.Series(high).rolling(window, min_periods=1).max()
            - pd.Series(low).rolling(window, min_periods=1).min()).to_numpy(np.float64)
        complete = rows - last_reset + 1 >= window
        span_complete[:, j] = complete
        side, lo, hi = UNKNOWN, float(ohlc[0, 0]), float(ohlc[0, 0])
        for row in range(n):
            scale = atr[row, j]
            if gap[row] or not np.isfinite(scale) or scale <= 0:
                side, lo, hi = UNKNOWN, float(ohlc[row, 0]), float(ohlc[row, 0])
            if not np.isfinite(scale) or scale <= 0:
                continue
            threshold = MOVE_ATR * scale
            side, lo, hi = _ending_direction(
                side, lo, hi, float(ohlc[row, 1]), float(ohlc[row, 2]),
                float(ohlc[row, 3]), float(threshold))
            if complete[row] and local_range[row, j] < threshold:
                allowed[row, j] = FLAT
            elif side in (DOWN, UP):
                allowed[row, j] = side
            elif complete[row] and local_range[row, j] >= threshold:
                allowed[row, j] = ACTIVE_UNKNOWN
    _need(np.isfinite(local_range).all(), 'LOCAL_RANGE')
    arrays = (issue, rows, ohlc[:, 3], atr, available_at, counts,
              gap, allowed, span_complete, local_range)
    return NativeM5Targets(*(_owned(value) for value in arrays),
                           MappingProxyType(dict(source_ref)), cutoff_ns)
