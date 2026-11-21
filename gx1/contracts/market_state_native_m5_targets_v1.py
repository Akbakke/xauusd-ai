"""Native M5 and higher bar directions from consecutive observed opens/closes."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from gx1.features.technical_indicators_v1 import EMA50_200_ATR_PERIOD, wilder_atr

from gx1.features.htf_features import (
    MULTI_TF_RESAMPLE_ORIGIN_CONTRACT, MULTI_TF_RESAMPLE_ORIGIN_OFFSET,
    MULTI_TF_SHIFT, MULTI_TF_TIMEFRAMES,
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


def noise_margin_contract(margin_atr):
    """Explicit net-change neutral zone; neither a path nor whole-market FLAT label."""
    _need(type(margin_atr) in (int, float) and np.isfinite(margin_atr) and margin_atr > 0,
          'NOISE_MARGIN_ATR')
    value = {
        'schema_version': 'gx1_native_m5_bar_net_change_noise_margin_v1',
        'raw_target_contract_sha256': contract()['contract_sha256'],
        'margin_atr': float(margin_atr), 'timeframes': list(TIMEFRAMES),
        'delta': 'current_observed_TF_open_or_close_minus_previous_observed_same_price',
        'atr_owner': 'gx1.features.technical_indicators_v1.wilder_atr',
        'atr_period': EMA50_200_ATR_PERIOD, 'atr_source': 'entire_stored_observed_TF_bar_prefix',
        'atr_asof': 'last_TF_available_ns_le_issue_ns_never_target_or_future_ATR',
        'margin_price': 'explicit_margin_atr_times_issue_asof_ATR14',
        'neutral_net_change': 'abs(delta)_strictly_less_than_margin_price_or_exact_zero',
        'boundary': 'abs(delta)_equal_to_margin_price_keeps_nonzero_delta_sign',
        'unknown_ATR': 'nonzero_delta_unknown_exact_zero_with_known_endpoints_known',
        'known': 'raw_eligible_endpoints_and_exact_zero_or_finite_nonnegative_margin',
        'open_eligible': 'noise_open_known_and_not_raw_open_known_at_issue',
        'close_eligible': 'noise_close_known',
        'gaps': 'raw_source_gap_facts_and_observed_previous_comparisons_retained',
        'meaning': 'neutral_zone_for_net_price_change_not_quiet_path_or_whole_market_FLAT',
        'usage': 'producer_only_once_per_whole_cohort_then_persist_before_model_batches',
        'metadata_as_model_input': False,
    }
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def noise_margin_dataset_contract(margin_atr):
    """Bind training labels separately from unchanged observed bar facts."""
    margin = noise_margin_contract(margin_atr)
    value = {
        'schema_version': 'gx1_native_m5_bar_noise_margin_dataset_v1',
        'raw_target_contract_sha256': contract()['contract_sha256'],
        'noise_margin_contract': margin,
        'timeframes': list(TIMEFRAMES), 'class_order': [-1, 0, 1],
        'class_names': ['DOWN', 'NEUTRAL_NET', 'UP'],
        'tasks': {
            task: {'sign': 'noise_' + task + '_sign',
                   'known': 'noise_' + task + '_known',
                   'eligible': 'noise_' + task + '_eligible'}
            for task in ('open', 'close')},
        'required_array_fields': [
            'issue_ns', 'target_row', 'validity_ns', 'target_present',
            'observed_previous', 'mature', 'eligible', 'open_known_at_issue',
            'open_sign', 'close_sign', 'm5_count', 'source_gap',
            'noise_margin_price', 'noise_atr14_asof',
            'noise_atr_asof_row', 'noise_atr_asof_available_ns',
            'noise_open_delta', 'noise_close_delta',
            'noise_open_sign', 'noise_close_sign',
            'noise_open_known', 'noise_close_known',
            'noise_open_eligible', 'noise_close_eligible'],
        'required_projection_metadata_fields': ['noise_margin_contract'],
        'raw_fact_fields_preserved': True,
        'future_maturity': 'target_present_and_observed_previous_and_validity_ns_strictly_before_fold_upper_ns',
        'known_open': 'open_known_at_issue_is_observed_context_never_predictive_credit',
        'projection': 'immutable_whole_cohort_once_before_model_batches',
        'target_arrays_and_metadata_as_model_input': False,
        'path_or_range_flat_semantics_changed': False,
    }
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def require_noise_margin_dataset_contract(value):
    """Require the exact owner-declared fields, masks, semantics and identity."""
    _need(type(value) is dict and type(value.get('noise_margin_contract')) is dict,
          'NOISE_DATASET_CONTRACT')
    expected = noise_margin_dataset_contract(value['noise_margin_contract'].get('margin_atr'))
    raw = json.dumps({key: part for key, part in value.items() if key != 'contract_sha256'},
                     sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    _need(value == expected and hashlib.sha256(raw).hexdigest() == expected['contract_sha256'],
          'NOISE_DATASET_CONTRACT')
    return expected



def fully_future_bar_contract():
    """Opt-in reference for an entirely unstarted native TF bar, not a FLAT rule."""
    value = {
        'schema_version': 'gx1_native_m5_fully_future_bar_reference_v1',
        'raw_target_contract_sha256': contract()['contract_sha256'],
        'timeframes': list(TIMEFRAMES),
        'tf_duration_ns': {tf: int(MULTI_TF_SHIFT[tf].value) for tf in TIMEFRAMES},
        'tf_origin_ns': {tf: int(MULTI_TF_RESAMPLE_ORIGIN_OFFSET[tf].value) for tf in TIMEFRAMES},
        'selection': 'first_observed_TF_bar_with_nominal_start_ns_ge_issue_ns',
        'empty_bins': 'never_invented_or_filled',
        'maturity': 'selected_available_ns_strictly_before_fold_upper_ns',
        'immature': 'time_metadata_only_no_price_sign_or_coverage_labels',
        'raw_comparisons': 'unchanged_previous_observed_TF_open_and_close',
        'complete': 'all_nominal_M5_member_starts_present_exactly_once',
        'source_gap': 'raw_gap_reported_separately_not_used_as_completeness_mask',
        'atr_owner': 'gx1.features.technical_indicators_v1.wilder_atr',
        'atr_period': EMA50_200_ATR_PERIOD,
        'atr_asof': 'last_observed_TF_available_ns_le_issue_ns_only',
        'state_labels': 'none_raw_range_is_not_a_direction_or_a_FLAT_label',
        'target_arrays_and_metadata_as_model_input': False,
    }
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def fully_future_state_contract():
    """Pure partial-observation interface; deliberately grants no label recipe."""
    value = {
        'schema_version': 'gx1_native_m5_fully_future_state_dataset_v1',
        'qualification': 'observation_algebra_only_no_qualified_state_label_generator',
        'future_bar_contract_sha256': fully_future_bar_contract()['contract_sha256'],
        'timeframes': list(TIMEFRAMES),
        'tf_duration_ns': {tf: int(MULTI_TF_SHIFT[tf].value) for tf in TIMEFRAMES},
        'tf_origin_ns': {tf: int(MULTI_TF_RESAMPLE_ORIGIN_OFFSET[tf].value) for tf in TIMEFRAMES},
        'class_order': ['DOWN', 'FLAT', 'UP'],
        'observation_fields': ['activity_known', 'activity', 'side_known', 'side_up'],
        'activity_known': 'requires_present_mature_nominally_complete_fully_future_bar',
        'side_known': 'requires_known_active_and_separately_certified_direction',
        'unknown_storage': 'false_boolean_payload_is_not_an_observed_class',
        'local_window': 'unbound_requires_separate_explicit_label_contract',
        'activity_rule': 'unbound_requires_separate_explicit_label_contract',
        'side_certificate': 'unbound_requires_separate_explicit_label_contract',
        'production_training_admitted': False,
        'target_arrays_and_metadata_as_model_input': False,
    }
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def require_fully_future_state_contract(value):
    """Reject old raw/net/M1 identities and arbitrary self-hashed state recipes."""
    _need(type(value) is dict and value == fully_future_state_contract(),
          'FULLY_FUTURE_STATE_CONTRACT')
    return fully_future_state_contract()

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


    def gather_fully_future_bar(self, issue_ns, *, fold_upper_ns):
        """Producer projection of next unstarted TF bar with strict source proof.

        Membership uses the source clock, independently of OHLC and raw gap flags.
        Only mature rows read selected OHLC/signs. ATR is strictly issue-as-of.
        No local-state rule is inferred from range, net sign or extrema ordering.
        """
        _need(isinstance(issue_ns, np.ndarray) and issue_ns.dtype == np.int64
              and issue_ns.ndim == 1 and len(issue_ns) > 0
              and type(fold_upper_ns) is int and 0 < fold_upper_ns <= self.cutoff_ns,
              'QUERY')
        at = np.searchsorted(self.issue_ns, issue_ns)
        _need(np.all(at < len(self.issue_ns))
              and np.array_equal(self.issue_ns[at], issue_ns)
              and np.all(issue_ns < fold_upper_ns), 'ISSUE_AXIS')
        shape = (len(issue_ns), len(TIMEFRAMES))
        out = {'issue_ns': issue_ns.copy(),
               'fully_future_bar_contract': fully_future_bar_contract()}
        for name in ('target_row', 'target_start_ns', 'validity_ns',
                     'open_available_ns', 'atr_asof_row', 'atr_asof_available_ns'):
            out[name] = np.full(shape, -1, np.int64)
        for name in ('target_present', 'mature', 'nominal_complete',
                     'observed_previous', 'eligible', 'open_known_at_issue', 'source_gap'):
            out[name] = np.zeros(shape, np.bool_)
        out['m5_count'] = np.zeros(shape, np.int64)
        for name in ('open_sign', 'close_sign'):
            out[name] = np.zeros(shape, np.int8)
        out['target_ohlc'] = np.full((*shape, 4), np.nan, np.float64)
        out['target_range'] = np.full(shape, np.nan, np.float64)
        out['atr14_asof'] = np.full(shape, np.nan, np.float64)
        source_starts = self.issue_ns - TICK_NS
        for j, tf in enumerate(TIMEFRAMES):
            bars = self.bars_by_timeframe[tf]
            duration = int(MULTI_TF_SHIFT[tf].value)
            selected = np.searchsorted(bars.bar_start_ns, issue_ns, side='left')
            present = selected < len(bars.bar_start_ns)
            rows = selected[present]
            out['target_present'][:, j] = present
            out['target_start_ns'][present, j] = bars.bar_start_ns[rows]
            out['validity_ns'][present, j] = bars.available_ns[rows]
            out['open_available_ns'][present, j] = bars.open_available_ns[rows]
            ready = present & (out['validity_ns'][:, j] < fold_upper_ns)
            out['mature'][:, j] = ready
            rows = selected[ready]
            out['target_row'][ready, j] = rows
            out['observed_previous'][ready, j] = bars.observed_previous[rows]
            out['eligible'][ready, j] = bars.observed_previous[rows]
            out['open_sign'][ready, j] = bars.open_sign[rows]
            out['close_sign'][ready, j] = bars.close_sign[rows]
            out['m5_count'][ready, j] = bars.m5_count[rows]
            out['source_gap'][ready, j] = bars.source_gap[rows]
            out['target_ohlc'][ready, j] = bars.ohlc[rows]
            out['target_range'][ready, j] = bars.ohlc[rows, 1] - bars.ohlc[rows, 2]
            # Unique aligned source starts let count plus aligned endpoints prove
            # every expected member: n integer grid points in an n-point interval.
            lo = np.searchsorted(source_starts, bars.bar_start_ns[rows], side='left')
            hi = np.searchsorted(source_starts, bars.available_ns[rows], side='left')
            member_count = hi - lo
            complete = member_count == duration // TICK_NS
            nonempty = member_count > 0
            if nonempty.any():
                idx = np.flatnonzero(nonempty)
                complete[idx] &= (source_starts[lo[idx]] == bars.bar_start_ns[rows[idx]])
                complete[idx] &= (source_starts[hi[idx] - 1]
                                  == bars.available_ns[rows[idx]] - TICK_NS)
            out['nominal_complete'][ready, j] = complete
            # Compute only the maximum closed prefix needed by this cohort.
            # Each returned ATR still joins its own issue-as-of closed TF row.
            closed = np.searchsorted(bars.available_ns, issue_ns, side='right') - 1
            observed = closed >= 0
            if observed.any():
                prefix = bars.ohlc[:int(closed[observed].max()) + 1]
                frame = pd.DataFrame(prefix, columns=('open', 'high', 'low', 'close'), copy=False)
                full_atr = wilder_atr(frame['high'], frame['low'], frame['close'],
                                     EMA50_200_ATR_PERIOD).to_numpy(np.float64)
                out['atr_asof_row'][observed, j] = closed[observed]
                out['atr_asof_available_ns'][observed, j] = bars.available_ns[closed[observed]]
                out['atr14_asof'][observed, j] = full_atr[closed[observed]]
        return out


    def gather_next_bar_noise_margin(self, issue_ns, *, fold_upper_ns, margin_atr):
        """Producer-only whole-cohort augmentation; persist before model batching.

        No mutable ATR cache or default-build state is added. This one producer
        call computes each TF's full-prefix canonical ATR14 once and joins only
        the last already closed TF bar. Every original gather key is untouched.
        """
        margin_contract = noise_margin_contract(margin_atr)
        out = self.gather_next_bar(issue_ns, fold_upper_ns=fold_upper_ns)
        shape = out['target_row'].shape
        atr = np.full(shape, np.nan, np.float64)
        margin = np.full(shape, np.nan, np.float64)
        asof_row = np.full(shape, -1, np.int64)
        asof_available = np.full(shape, -1, np.int64)
        delta = {task: np.full(shape, np.nan, np.float64) for task in ('open', 'close')}
        signs = {task: np.zeros(shape, np.int8) for task in ('open', 'close')}
        known = {task: np.zeros(shape, np.bool_) for task in ('open', 'close')}
        for j, tf in enumerate(TIMEFRAMES):
            bars = self.bars_by_timeframe[tf]
            frame = pd.DataFrame(bars.ohlc, columns=('open', 'high', 'low', 'close'), copy=False)
            full_atr = wilder_atr(frame['high'], frame['low'], frame['close'], EMA50_200_ATR_PERIOD).to_numpy(np.float64)
            closed = np.searchsorted(bars.available_ns, issue_ns, side='right') - 1
            observed = closed >= 0
            asof_row[observed, j] = closed[observed]
            asof_available[observed, j] = bars.available_ns[closed[observed]]
            atr[observed, j] = full_atr[closed[observed]]
            with np.errstate(over='ignore', invalid='ignore'):
                raw_margin = atr[:, j] * margin_contract['margin_atr']
            margin_known = np.isfinite(raw_margin) & (raw_margin >= 0)
            margin[margin_known, j] = raw_margin[margin_known]
            endpoints = out['eligible'][:, j]
            selected = out['target_row'][endpoints, j]
            for task, column, previous in (
                    ('open', 0, bars.previous_open), ('close', 3, bars.previous_close)):
                delta[task][endpoints, j] = bars.ohlc[selected, column] - previous[selected]
                values = delta[task][:, j]
                zero = values == 0
                known[task][:, j] = endpoints & np.isfinite(values) & (zero | margin_known)
                directional = known[task][:, j] & ~zero & (np.abs(values) >= margin[:, j])
                signs[task][directional, j] = np.sign(values[directional]).astype(np.int8)
        out.update({
            'noise_margin_contract': margin_contract,
            'noise_margin_price': margin, 'noise_atr14_asof': atr,
            'noise_atr_asof_row': asof_row, 'noise_atr_asof_available_ns': asof_available,
            'noise_open_delta': delta['open'], 'noise_close_delta': delta['close'],
            'noise_open_sign': signs['open'], 'noise_close_sign': signs['close'],
            'noise_open_known': known['open'], 'noise_close_known': known['close'],
            'noise_open_eligible': known['open'] & ~out['open_known_at_issue'],
            'noise_close_eligible': known['close'],
        })
        return out


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
