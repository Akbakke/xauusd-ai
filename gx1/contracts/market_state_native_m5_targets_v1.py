"""Native M5 and higher bar directions from consecutive observed opens/closes."""
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

TIMEFRAMES = tuple(MULTI_TF_TIMEFRAMES)
TICK_NS = 300_000_000_000


def _need(condition, reason):
    if not bool(condition):
        raise ValueError('NATIVE_M5_TARGET_' + reason)


def contract():
    value = {
        'schema_version': 'gx1_market_state_native_m5_bar_directions_v2',
        'role': 'bar_direction_targets_no_training_or_trading_authority',
        'timeframes': list(TIMEFRAMES), 'source_bar_seconds': 300,
        'source_columns': ['open', 'high', 'low', 'close'],
        'issue': 'native_M5_bar_start_plus_300_seconds',
        'cutoff': 'exclusive_source_M5_close_and_native_TF_bar_availability',
        'tf_resample_origin': dict(MULTI_TF_RESAMPLE_ORIGIN_CONTRACT),
        'tf_duration_ns': {tf: int(MULTI_TF_SHIFT[tf].value) for tf in TIMEFRAMES},
        'observed_TF_bar': 'canonical_nonempty_native_M5_aggregate_available_at_label_plus_TF_duration',
        'open_sign': 'sign(current_observed_TF_open_minus_previous_observed_TF_open)',
        'close_sign': 'sign(current_observed_TF_close_minus_previous_observed_TF_close)',
        'sign_values': [-1, 0, 1],
        'equality': 'known_zero_when_previous_observed_TF_bar_exists',
        'mixed_open_close': 'two_separate_known_signs',
        'observed_previous': 'all_observed_TF_bars_except_first_source_prefix_bar',
        'previous_unknown_storage': 'NaN_previous_prices_and_zero_signs_with_false_mask',
        'source_gap': 'reported_observed_M5_interruption_or_missing_TF_bins_comparisons_remain_known',
        'partial_TF_bins': 'observed_M5_count_reported_first_open_last_close_no_missing_values_inferred',
        'future_target': 'next_observed_TF_bar_with_available_ns_strictly_after_M5_issue',
        'target_maturity': 'next_actual_TF_available_ns_strictly_before_fold_upper',
        'open_availability': 'first_actual_M5_member_close_of_observed_TF_bar',
        'open_known_at_issue': 'target_open_available_ns_le_M5_issue_may_precede_target_close',
        'forecast_interpretation': 'close_sign_predictive_score_separate_from_already_observed_open_sign',
        'gathered_metadata': 'future_time_index_eligibility_and_labels_are_target_only_never_features',
        'future_eligibility': 'target_present_and_mature_and_observed_previous',
        'immature_target_storage': 'target_row_minus_one_zero_signs_and_false_observed_previous',
    }
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def _owned(array):
    return np.frombuffer(np.ascontiguousarray(array).tobytes(),
                         dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class NativeTFBarDirections:
    bar_start_ns: np.ndarray
    available_ns: np.ndarray
    open_available_ns: np.ndarray
    ohlc: np.ndarray
    previous_open: np.ndarray
    previous_close: np.ndarray
    open_sign: np.ndarray
    close_sign: np.ndarray
    observed_previous: np.ndarray
    m5_count: np.ndarray
    source_gap: np.ndarray


@dataclass(frozen=True)
class NativeM5Targets:
    issue_ns: np.ndarray
    source_row: np.ndarray
    ohlc: np.ndarray
    source_gap: np.ndarray
    bars_by_timeframe: MappingProxyType
    source_ref: MappingProxyType
    cutoff_ns: int

    def gather_next_bar(self, issue_ns, *, fold_upper_ns):
        """Gather each TF's next actual close, checking maturity before labels."""
        _need(isinstance(issue_ns, np.ndarray) and issue_ns.dtype == np.int64
              and issue_ns.ndim == 1 and len(issue_ns) > 0
              and type(fold_upper_ns) is int and 0 < fold_upper_ns <= self.cutoff_ns,
              'QUERY')
        at = np.searchsorted(self.issue_ns, issue_ns)
        _need(np.all(at < len(self.issue_ns))
              and np.array_equal(self.issue_ns[at], issue_ns)
              and np.all(issue_ns < fold_upper_ns), 'ISSUE_AXIS')
        shape = (len(issue_ns), len(TIMEFRAMES))
        target_row = np.full(shape, -1, np.int64)
        validity = np.full(shape, -1, np.int64)
        target_present = np.zeros(shape, np.bool_)
        mature = np.zeros(shape, np.bool_)
        open_known = np.zeros(shape, np.bool_)
        previous = np.zeros(shape, np.bool_)
        open_sign = np.zeros(shape, np.int8)
        close_sign = np.zeros(shape, np.int8)
        count = np.zeros(shape, np.int64)
        gap = np.zeros(shape, np.bool_)
        for j, tf in enumerate(TIMEFRAMES):
            bars = self.bars_by_timeframe[tf]
            next_row = np.searchsorted(bars.available_ns, issue_ns, side='right')
            present = next_row < len(bars.available_ns)
            target_present[:, j] = present
            validity[present, j] = bars.available_ns[next_row[present]]
            open_known[present, j] = bars.open_available_ns[next_row[present]] <= issue_ns[present]
            ready = present & (validity[:, j] < fold_upper_ns)
            mature[:, j] = ready
            selected = next_row[ready]
            target_row[ready, j] = selected
            previous[ready, j] = bars.observed_previous[selected]
            open_sign[ready, j] = bars.open_sign[selected]
            close_sign[ready, j] = bars.close_sign[selected]
            count[ready, j] = bars.m5_count[selected]
            gap[ready, j] = bars.source_gap[selected]
        return {
            'issue_ns': issue_ns.copy(), 'target_row': target_row,
            'validity_ns': validity, 'target_present': target_present,
            'observed_previous': previous, 'mature': mature,
            'eligible': mature & previous, 'open_known_at_issue': open_known,
            'open_sign': open_sign, 'close_sign': close_sign,
            'm5_count': count, 'source_gap': gap,
        }


def _bar_directions(frame, source_gap_frame, tf, cutoff_ns):
    bars = _resample_ohlc_for_model_native_scalars(frame, tf)
    duration = int(MULTI_TF_SHIFT[tf].value)
    available = bars.index.asi8 + duration
    keep = available < cutoff_ns
    bars = bars.loc[keep]
    available = available[keep].astype(np.int64, copy=False)
    values = bars.loc[:, ['open', 'high', 'low', 'close']].to_numpy(np.float64)
    source_available = pd.Series(frame.index.asi8 + TICK_NS, index=frame.index)
    open_available = multi_tf_resample(source_available, tf).first().reindex(
        bars.index).to_numpy(np.int64)
    counts = multi_tf_resample(frame, tf).size().reindex(bars.index).to_numpy(np.int64)
    gaps = multi_tf_resample(source_gap_frame, tf).max().reindex(bars.index)
    gap = gaps['source_gap'].to_numpy(np.bool_)
    if len(available) > 1:
        gap[1:] |= np.diff(available) != duration
    previous = np.arange(len(bars)) > 0
    previous_open = np.full(len(bars), np.nan, np.float64)
    previous_close = np.full(len(bars), np.nan, np.float64)
    previous_open[1:] = values[:-1, 0]
    previous_close[1:] = values[:-1, 3]
    open_sign = np.zeros(len(bars), np.int8)
    close_sign = np.zeros(len(bars), np.int8)
    open_sign[1:] = np.sign(values[1:, 0] - previous_open[1:]).astype(np.int8)
    close_sign[1:] = np.sign(values[1:, 3] - previous_close[1:]).astype(np.int8)
    arrays = (bars.index.asi8.astype(np.int64), available, open_available, values,
              previous_open, previous_close, open_sign, close_sign,
              previous.astype(np.bool_), counts, gap)
    return NativeTFBarDirections(*(_owned(value) for value in arrays))


def build_native_m5_targets(bar_start_ns, ohlc, *, source_ref, cutoff_ns):
    """Build immutable observed bar comparisons from supplied native M5 arrays."""
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
    gap = np.r_[False, np.diff(issue) != TICK_NS]
    gap_frame = pd.DataFrame({'source_gap': gap}, index=frame.index)
    bars = {tf: _bar_directions(frame, gap_frame, tf, cutoff_ns) for tf in TIMEFRAMES}
    return NativeM5Targets(
        _owned(issue), _owned(np.arange(len(issue), dtype=np.int64)),
        _owned(ohlc), _owned(gap), MappingProxyType(bars),
        MappingProxyType(dict(source_ref)), cutoff_ns)
