"""Streaming evidence for the explicit dynamic MTF market-state forecast.

No threshold selection, model calls, trade decisions or target reconstruction.
The caller supplies the exact mature-observed mask from the objective owner.
"""
from __future__ import annotations

import io
import math

import numpy as np

from gx1.contracts.entry_dynamic_mtf_state_targets_v1 import STATE_NAMES, TIMEFRAMES

SCHEMA_VERSION = 'gx1_dynamic_mtf_state_report_v1'


def _require(condition, code):
    if not condition:
        raise RuntimeError('[DYNAMIC_STATE_REPORT_' + code + ']')


def dynamic_state_prediction_layout(rows, lead_count):
    """Exact on-disk array shapes/dtypes for one dynamic-state measurement."""
    _require(type(rows) is int and rows > 0 and type(lead_count) is int and lead_count > 0, 'OUTPUT_LAYOUT')
    return {'probabilities': ((rows, len(TIMEFRAMES), lead_count, len(STATE_NAMES)), np.float32),
            'sequence_nll': ((rows, len(TIMEFRAMES)), np.float32), 'parent_rows': ((rows,), np.int64),
            'issue_ns': ((rows,), np.int64), 'bar_start_ns': ((rows,), np.int64)}


def dynamic_state_prediction_file_bytes(rows, lead_count):
    """Logical NPY bytes including headers; filesystem/VHD overhead is extra."""
    result = {}
    for name, (shape, dtype) in dynamic_state_prediction_layout(rows, lead_count).items():
        dtype = np.dtype(dtype)
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {'descr': np.lib.format.dtype_to_descr(dtype),
                                                       'fortran_order': False, 'shape': shape})
        result[name] = len(header.getvalue()) + math.prod(shape) * dtype.itemsize
    return result


def _grid(value):
    leads = np.asarray(value)
    _require(leads.ndim == 1 and len(leads) > 0 and np.issubdtype(leads.dtype, np.integer)
             and np.all(leads > 0) and np.all(np.diff(leads) > 0), 'QUERY_GRID')
    return leads.astype(np.int64, copy=True)


def _labels(state, eligible, leads):
    state, eligible = np.asarray(state), np.asarray(eligible)
    _require(state.ndim == 3 and state.shape[1:] == (len(TIMEFRAMES), len(leads))
             and eligible.shape == state.shape and eligible.dtype == np.bool_
             and np.issubdtype(state.dtype, np.integer), 'LABEL_SHAPES')
    _require(np.all((state[eligible] >= 0) & (state[eligible] < len(STATE_NAMES))), 'ELIGIBLE_CLASS')
    return state, eligible


def _ratio(numerator, denominator, infinite=None):
    numerator, denominator = np.asarray(numerator), np.asarray(denominator)
    out = np.empty(denominator.shape, dtype=object)
    for index in np.ndindex(denominator.shape):
        out[index] = (None if denominator[index] == 0 or (infinite is not None and infinite[index] > 0)
                      else float(numerator[index] / denominator[index]))
    return out.tolist()


def dense_train_query_counts(series, *, prefix_rows, cutoff_ns, lead_minutes):
    """Exact TRAIN-only dense query counts without materializing parent-by-lead arrays.

    One source validity row is weighted by the number of TRAIN issues that can
    query it. Consecutive source pairs are weighted separately; missing minutes
    are never bridged. Only the prefix before cutoff contributes label/reason
    values. Caller must load and source-bind the immutable series beforehand.
    """
    from gx1.contracts.entry_dynamic_mtf_state_targets_v1 import (
        DynamicMtfStateSeries, MINUTE_NS, TEST_BOUNDARY_NS, UNKNOWN, QUERY_UNAVAILABLE, IMMATURE,
        dynamic_mtf_state_target_contract, require_dynamic_mtf_state_target_contract)
    _require(isinstance(series, DynamicMtfStateSeries), 'TRAIN_SERIES')
    _require(series.metadata.get('test_data_used') is False, 'TRAIN_TEST_CLOSED')
    require_dynamic_mtf_state_target_contract(series.metadata['definition'])
    _require(type(prefix_rows) is int and prefix_rows > 0 and type(cutoff_ns) is int
             and 0 < cutoff_ns <= min(series.metadata['cutoff_ns'], TEST_BOUNDARY_NS)
             and cutoff_ns % MINUTE_NS == 0, 'TRAIN_BOUNDARY')
    leads = _grid(lead_minutes)
    length = len(leads)
    _require(np.array_equal(leads, np.arange(1, length + 1, dtype=np.int64))
             and length * MINUTE_NS <= np.iinfo(np.int64).max
             and prefix_rows * length <= np.iinfo(np.int64).max, 'TRAIN_DENSE_GRID')
    a = series.arrays
    issues = a['issue_ns']
    parents = a['parent_rows']
    _require(len(issues) == len(parents) == series.metadata['parents'] and prefix_rows <= len(issues)
             and np.all(np.diff(issues) > 0) and np.all(np.diff(parents) > 0)
             and np.all(issues >= 0) and np.all(issues % MINUTE_NS == 0)
             and int(np.searchsorted(issues, cutoff_ns, side='left')) == prefix_rows, 'TRAIN_EXACT_PREFIX')
    issues = issues[:prefix_rows]
    axis = a['validity_ns']
    end = int(np.searchsorted(axis, cutoff_ns, side='left'))
    validity = axis[:end]
    _require(end > 0 and np.all(validity >= 0) and np.all(np.diff(validity) > 0)
             and np.all(validity % MINUTE_NS == 0), 'TRAIN_SOURCE_CLOCK')
    state, observed = a['state'][:end], a['observed'][:end]
    path, reason = a['path_observed'][:end], a['reason'][:end]
    _require(state.shape == observed.shape == path.shape == reason.shape == (end, len(TIMEFRAMES))
             and np.array_equal(a['maturity_ns'][:end], np.broadcast_to(validity[:, None], state.shape))
             and np.all(state[~observed] == UNKNOWN) and np.isin(state[observed], range(len(STATE_NAMES))).all()
             and not np.any(reason & (QUERY_UNAVAILABLE | IMMATURE)), 'TRAIN_STATE_TAPE')
    # A source validity v can be queried only by issue t with v-H <= t <= v-1.
    weights = (np.searchsorted(issues, validity - MINUTE_NS, side='right')
               - np.searchsorted(issues, validity - length * MINUTE_NS, side='left')).astype(np.int64)
    _require(np.all(weights >= 0), 'TRAIN_QUERY_MULTIPLICITY')
    mature = int(np.clip((cutoff_ns - 1 - issues) // MINUTE_NS, 0, length).sum(dtype=np.int64))
    total = prefix_rows * length
    missing = mature - int(weights.sum(dtype=np.int64))
    immature = total - mature
    _require(missing >= 0 and immature >= 0, 'TRAIN_QUERY_PARTITION')
    classes = len(STATE_NAMES)
    counts = np.zeros((len(TIMEFRAMES), classes), np.int64)
    transitions = np.zeros((len(TIMEFRAMES), classes, classes), np.int64)
    path_counts = np.zeros(len(TIMEFRAMES), np.int64)
    bits = dynamic_mtf_state_target_contract()['reason_bits']
    reasons = {name: np.zeros(len(TIMEFRAMES), np.int64) for name in bits}
    excluded = {name: np.zeros(len(TIMEFRAMES), np.int64) for name in bits}
    if length > 1:
        pair_weights = (np.searchsorted(issues, validity[:-1] - MINUTE_NS, side='right')
            - np.searchsorted(issues, validity[:-1] - (length - 1) * MINUTE_NS, side='left')).astype(np.int64)
        adjacent = np.diff(validity) == MINUTE_NS
        _require(np.all(pair_weights >= 0), 'TRAIN_PAIR_MULTIPLICITY')
    for tf in range(len(TIMEFRAMES)):
        path_counts[tf] = weights[path[:, tf]].sum(dtype=np.int64)
        for k in range(classes):
            counts[tf, k] = weights[observed[:, tf] & (state[:, tf] == k)].sum(dtype=np.int64)
        if length > 1:
            pairs = adjacent & observed[:-1, tf] & observed[1:, tf]
            for left in range(classes):
                for right in range(classes):
                    use = pairs & (state[:-1, tf] == left) & (state[1:, tf] == right)
                    transitions[tf, left, right] = pair_weights[use].sum(dtype=np.int64)
        for name, bit in bits.items():
            hit = (reason[:, tf] & bit) != 0
            reasons[name][tf] = weights[hit].sum(dtype=np.int64)
            excluded[name][tf] = weights[hit & ~observed[:, tf]].sum(dtype=np.int64)
    for name, bit in bits.items():
        addition = missing if bit == int(QUERY_UNAVAILABLE) else immature if bit == int(IMMATURE) else 0
        reasons[name] += addition
        excluded[name] += addition
    current = a['issue_source_index'][:prefix_rows]
    _require(np.all((current >= -1) & (current < end)), 'TRAIN_CURRENT_INDEX')
    present = current >= 0
    _require(np.array_equal(validity[current[present]], issues[present]), 'TRAIN_CURRENT_CLOCK')
    safe = np.maximum(current, 0)
    current_known = observed[safe] & present[:, None]
    current_paths = path[safe] & present[:, None]
    current_states = state[safe]
    current_counts = np.stack([np.count_nonzero(current_known & (current_states == k), axis=0)
                               for k in range(classes)], axis=1)
    return {'schema_version': 'gx1_dynamic_state_train_query_counts_v1', 'status': 'COUNTS_ONLY_NOT_ADMISSION',
            'prefix_rows': prefix_rows, 'cutoff_ns': cutoff_ns, 'lead_count': length,
            'class_counts': counts.tolist(), 'transition_counts': transitions.tolist(),
            'eligible_query_cells_per_tf': counts.sum(axis=1).tolist(),
            'path_observed_query_cells_per_tf': path_counts.tolist(),
            'current_state_counts_per_tf': current_counts.tolist(),
            'current_path_observed_per_tf': current_paths.sum(axis=0).tolist(),
            'missing_source_query_cells_per_tf': missing, 'immature_query_cells_per_tf': immature,
            'diagnostic_reason_query_counts': {k: v.tolist() for k, v in reasons.items()},
            'excluded_reason_query_counts': {k: v.tolist() for k, v in excluded.items()},
            'later_label_metrics_inspected': False, 'test_data_used': False}


class StatePriorAccumulator:
    """Empirical class prior on the exact TRAIN target population; no smoothing."""
    def __init__(self, *, lead_minutes):
        self.leads = _grid(lead_minutes)
        self.counts = np.zeros((len(TIMEFRAMES), len(STATE_NAMES)), np.int64)
        self.parents = 0
        self.transition_counts = np.zeros((len(TIMEFRAMES), len(STATE_NAMES), len(STATE_NAMES)), np.int64)

    def update(self, state, eligible):
        state, eligible = _labels(state, eligible, self.leads)
        self.parents += state.shape[0]
        for k in range(len(STATE_NAMES)):
            self.counts[:, k] += np.count_nonzero(eligible & (state == k), axis=(0, 2))

        pair_observed = eligible[:, :, :-1] & eligible[:, :, 1:]
        for tf in range(len(TIMEFRAMES)):
            m = pair_observed[:, tf]
            code = len(STATE_NAMES) * state[:, tf, :-1][m].astype(np.int64) + state[:, tf, 1:][m]
            self.transition_counts[tf] += np.bincount(code, minlength=len(STATE_NAMES)**2).reshape(len(STATE_NAMES), len(STATE_NAMES))

    def finalize(self):
        _require(np.all(self.transition_counts.sum(axis=-1) > 0), 'TRAIN_TRANSITION_ROWS_REQUIRED')
        _require(self.parents > 0 and np.all(self.counts > 0), 'TRAIN_ALL_CLASSES_REQUIRED_PER_TF')
        return {'schema_version': SCHEMA_VERSION, 'timeframes': list(TIMEFRAMES),
                'class_names': list(STATE_NAMES), 'lead_minutes': self.leads.tolist(),
                'parents': self.parents, 'counts': self.counts.tolist(),
                'probabilities': (self.counts / self.counts.sum(axis=1, keepdims=True)).tolist(),
                'transition_counts': self.transition_counts.tolist(),
                'transition_probabilities': (self.transition_counts / self.transition_counts.sum(axis=-1, keepdims=True)).tolist(),
                'scope': 'All mature observed TRAIN query cells and adjacent observed pairs; empirical class and transition frequencies per TF; no VAL fit or smoothing.'}


def dense_train_prior_from_counts(counts, *, lead_minutes):
    """Turn exact TRAIN counts into the existing unsmoothed prior, never an admission."""
    leads = _grid(lead_minutes)
    _require(isinstance(counts, dict) and counts.get('schema_version') == 'gx1_dynamic_state_train_query_counts_v1'
             and counts.get('status') == 'COUNTS_ONLY_NOT_ADMISSION'
             and counts.get('later_label_metrics_inspected') is False and counts.get('test_data_used') is False
             and type(counts.get('prefix_rows')) is int and counts['prefix_rows'] > 0
             and type(counts.get('lead_count')) is int and counts['lead_count'] == len(leads)
             and np.array_equal(leads, np.arange(1, len(leads) + 1, dtype=np.int64)), 'TRAIN_COUNT_IDENTITY')
    parents = counts['prefix_rows']
    _require(parents * len(leads) <= np.iinfo(np.int64).max, 'TRAIN_COUNT_OVERFLOW')
    def integer_array(value, shape):
        array = np.asarray(value, dtype=object)
        _require(array.shape == shape and all(type(x) is int and 0 <= x <= np.iinfo(np.int64).max
                                               for x in array.flat), 'TRAIN_COUNT_ARRAY')
        return np.asarray(value, dtype=np.int64)
    classes = integer_array(counts.get('class_counts'), (len(TIMEFRAMES), len(STATE_NAMES)))
    transitions = integer_array(counts.get('transition_counts'), (len(TIMEFRAMES), len(STATE_NAMES), len(STATE_NAMES)))
    eligible = integer_array(counts.get('eligible_query_cells_per_tf'), (len(TIMEFRAMES),))
    _require(np.array_equal(classes.sum(axis=1, dtype=object), eligible.astype(object))
             and np.all(eligible <= parents * len(leads))
             and np.all(transitions.sum(axis=2, dtype=object) <= classes.astype(object))
             and np.all(transitions.sum(axis=1, dtype=object) <= classes.astype(object)), 'TRAIN_COUNT_CONSISTENCY')
    prior = StatePriorAccumulator(lead_minutes=leads)
    prior.counts[:] = classes
    prior.transition_counts[:] = transitions
    prior.parents = parents
    return prior.finalize()


class _MatchedTransitionEvidence:
    """Compare strict modal transition forecasts on identical observed endpoints."""
    def __init__(self, transition, markov_grid, *, model_receives_current_state):
        self.model_receives_current_state = model_receives_current_state
        f, k, leads = len(TIMEFRAMES), len(STATE_NAMES), markov_grid.shape[2]
        # TRAIN Markov is frozen: compute its joint decisions once per TF/case,
        # rather than allocating a second batch x TF x horizon x 3 x 3 tensor.
        self.markov_pair_code = np.zeros((f, k + 1, max(leads - 1, 0)), np.uint8)
        self.markov_pair_unique = np.zeros_like(self.markov_pair_code, dtype=bool)
        for tf in range(f):
            for case in range(k + 1):
                joint = (markov_grid[tf, case, :-1, :, None] * transition[tf][None, :, :]).reshape(-1, k * k)
                self.markov_pair_code[tf, case] = joint.argmax(axis=-1)
                self.markov_pair_unique[tf, case] = (joint == joint.max(axis=-1, keepdims=True)).sum(axis=-1) == 1
        shape = (f, k, k)
        self.groups = {group: {name: np.zeros(shape, np.int64) for name in
                       ('truth', 'model_forecasts', 'model_exact_hits', 'markov_forecasts', 'markov_exact_hits')}
                       for group in ('current_to_first', 'future_adjacent')}
        self.ties = {group: {name: np.zeros(len(TIMEFRAMES), np.int64) for name in ('model', 'markov')}
                     for group in self.groups}

    def _accumulate(self, group, eligible, truth_code, model_code, model_unique, markov_code, markov_unique):
        for tf in range(len(TIMEFRAMES)):
            mask = eligible[:, tf]
            true = truth_code[:, tf]
            model = model_code[:, tf]
            markov = markov_code[:, tf]
            mu, bu = model_unique[:, tf], markov_unique[:, tf]
            for name, values, selection in (
                ('truth', true, mask),
                ('model_forecasts', model, mask & mu),
                ('model_exact_hits', true, mask & mu & (model == true)),
                ('markov_forecasts', markov, mask & bu),
                ('markov_exact_hits', true, mask & bu & (markov == true))):
                self.groups[group][name][tf] += np.bincount(values[selection], minlength=9).reshape(3, 3)
            self.ties[group]['model'][tf] += int(np.count_nonzero(mask & ~mu))
            self.ties[group]['markov'][tf] += int(np.count_nonzero(mask & ~bu))

    def update(self, p, model_joint, markov_p, state, eligible, current, current_observed):
        k = len(STATE_NAMES)
        first_model, first_markov = p[:, :, 0], markov_p[:, :, 0]
        true_code = k * current.astype(np.int64) + state[:, :, 0].astype(np.int64)
        model_code = k * current.astype(np.int64) + first_model.argmax(axis=-1)
        markov_code = k * current.astype(np.int64) + first_markov.argmax(axis=-1)
        model_unique = (first_model == first_model.max(axis=-1, keepdims=True)).sum(axis=-1) == 1
        markov_unique = (first_markov == first_markov.max(axis=-1, keepdims=True)).sum(axis=-1) == 1
        self._accumulate('current_to_first', eligible[:, :, 0] & current_observed,
                         true_code, model_code, model_unique, markov_code, markov_unique)
        if state.shape[2] <= 1:
            return
        truth = k * state[:, :, :-1].astype(np.int64) + state[:, :, 1:].astype(np.int64)
        mask = eligible[:, :, :-1] & eligible[:, :, 1:]
        model_flat = model_joint.reshape(*model_joint.shape[:3], k * k)
        case = (np.where(current_observed, current, k) if self.model_receives_current_state
                else np.full(current.shape, k, dtype=np.int64))
        markov_code = self.markov_pair_code[np.arange(len(TIMEFRAMES))[None, :], case]
        markov_unique = self.markov_pair_unique[np.arange(len(TIMEFRAMES))[None, :], case]
        self._accumulate('future_adjacent', mask, truth,
            model_flat.argmax(axis=-1), (model_flat == model_flat.max(axis=-1, keepdims=True)).sum(axis=-1) == 1,
            markov_code, markov_unique)

    def finalize(self):
        off_diagonal = ~np.eye(len(STATE_NAMES), dtype=bool)
        groups = {}
        for name, values in self.groups.items():
            result = {key + '_by_from_to': value.tolist() for key, value in values.items()}
            truth = values['truth'][:, off_diagonal].sum(axis=1)
            result['observed_pairs'] = values['truth'].sum(axis=(1, 2)).tolist()
            result['true_changes'] = truth.tolist()
            for forecast in ('model', 'markov'):
                hits = values[forecast + '_exact_hits'][:, off_diagonal].sum(axis=1)
                predictions = values[forecast + '_forecasts'][:, off_diagonal].sum(axis=1)
                result[forecast + '_tied_pairs'] = self.ties[name][forecast].tolist()
                result[forecast + '_exact_change_hits'] = hits.tolist()
                result[forecast + '_predicted_changes'] = predictions.tolist()
                result[forecast + '_change_recall'] = _ratio(hits, truth)
                result[forecast + '_change_precision'] = _ratio(hits, predictions)
            groups[name] = result
        return {'model_receives_current_state': self.model_receives_current_state,
                'markov_uses_current_state_when_observed': self.model_receives_current_state,
                'baseline': 'frozen_TRAIN_transition_and_prior; current_state_only_if_explicitly_supplied_to_model',
                'current_to_first_scope': 'requires observed current state; both state forecasts are compared with that state',
                'groups': groups,
                'limits': ['Both use identical observed target endpoints; missing states are excluded.',
                           'Tied forecasts make no modal prediction and count as misses in change recall.',
                           'Strict hits require the same from-state and to-state at the same minute.',
                           'Future pair decisions use the joint distribution, not differences between modal marginals.',
                           'The baseline does not consume model indicators or the independent current-path mask.',
                           'Parent/lead pairs overlap and are not independent market episodes or evidence of economic edge.',
                           'Current state is historical event-time evidence, not proof of physical ONLINE receipt.']}


class StateScoreAccumulator:
    def __init__(self, prior, *, lead_minutes, model_receives_current_state=False):
        _require(type(model_receives_current_state) is bool, 'MATCHED_CURRENT_INPUT_MODE')
        self.model_receives_current_state = model_receives_current_state
        self.leads = _grid(lead_minutes)
        _require(prior['schema_version'] == SCHEMA_VERSION and prior['timeframes'] == list(TIMEFRAMES)
                 and prior['class_names'] == list(STATE_NAMES)
                 and prior['lead_minutes'] == self.leads.tolist(), 'PRIOR_IDENTITY')
        counts = np.asarray(prior['counts'])
        _require(np.issubdtype(counts.dtype, np.integer) and np.all(counts > 0)
                 and np.all(counts <= np.iinfo(np.int64).max), 'PRIOR_COUNT_DTYPE')
        counts = counts.astype(np.int64, copy=False)
        self.prior = np.asarray(prior['probabilities'], np.float64)
        _require(counts.shape == self.prior.shape == (len(TIMEFRAMES), len(STATE_NAMES))
                 and np.all(counts > 0)
                 and np.array_equal(self.prior, counts / counts.sum(axis=1, keepdims=True)), 'PRIOR_COUNTS')
        transition_counts = np.asarray(prior['transition_counts'])
        _require(np.issubdtype(transition_counts.dtype, np.integer) and np.all(transition_counts >= 0)
                 and np.all(transition_counts <= np.iinfo(np.int64).max), 'MARKOV_COUNT_DTYPE')
        transition_counts = transition_counts.astype(np.int64, copy=False)
        transition = np.asarray(prior['transition_probabilities'], np.float64)
        _require(transition_counts.shape == transition.shape == (len(TIMEFRAMES), len(STATE_NAMES), len(STATE_NAMES))
                 and np.all(transition_counts >= 0) and np.all(transition_counts.sum(axis=-1) > 0)
                 and np.array_equal(transition, transition_counts / transition_counts.sum(axis=-1, keepdims=True)), 'MARKOV_COUNTS')
        _require(np.array_equal(self.leads, np.arange(1, len(self.leads) + 1)), 'DENSE_MINUTE_MARKOV_GRID')
        # First three initial cases are causal observed current states. The fourth is the TRAIN class prior.
        cases = np.concatenate((np.broadcast_to(np.eye(len(STATE_NAMES)), (len(TIMEFRAMES), len(STATE_NAMES), len(STATE_NAMES))), self.prior[:, None, :]), axis=1)
        self.markov_grid = np.empty((len(TIMEFRAMES), len(STATE_NAMES) + 1, len(self.leads), len(STATE_NAMES)), np.float64)
        for lead in range(len(self.leads)):
            cases = np.einsum('fkc,fcd->fkd', cases, transition)
            self.markov_grid[:, :, lead] = cases
        shape = (len(TIMEFRAMES), len(self.leads))
        self.matched_transitions = _MatchedTransitionEvidence(transition, self.markov_grid,
            model_receives_current_state=model_receives_current_state)
        self.matched_state_counts = {name: np.zeros(shape, np.int64) for name in ('correct', 'tied', 'logloss_infinite')}
        self.matched_state_sums = {name: np.zeros(shape, np.float64) for name in ('logloss', 'brier')}
        self.parents = 0
        self.cells = {name: np.zeros(shape, np.int64) for name in
                      ('observed', 'correct', 'tied', 'prior_correct', 'prior_tied',
                       'persistence_observed', 'persistence_correct', 'logloss_infinite', 'markov_logloss_infinite', 'markov_correct', 'markov_tied')}
        self.sums = {name: np.zeros(shape, np.float64) for name in
                     ('logloss', 'brier', 'prior_logloss', 'prior_brier', 'persistence_brier', 'markov_logloss', 'markov_brier')}
        self.confusion = np.zeros((len(TIMEFRAMES), len(STATE_NAMES), len(STATE_NAMES)), np.int64)
        self.class_counts = np.zeros((len(TIMEFRAMES), len(STATE_NAMES)), np.int64)
        self.probability_sums = np.zeros_like(self.class_counts, dtype=np.float64)
        self.sequence_nll = np.zeros(len(TIMEFRAMES), np.float64)
        self.pairs = {name: np.zeros(len(TIMEFRAMES), np.int64) for name in
                      ('observed', 'correct', 'tied', 'changes', 'predicted_changes',
                       'change_hits', 'false_change_alerts', 'missed_changes', 'logloss_infinite')}
        self.pair_sums = {name: np.zeros(len(TIMEFRAMES), np.float64) for name in
                         ('logloss', 'brier', 'change_brier')}
        self.timing = {name: np.zeros(len(TIMEFRAMES), np.int64) for name in
                       ('complete_windows', 'true_first_change', 'predicted_first_change', 'paired_first_changes')}
        self.timing_error_sum = np.zeros(len(TIMEFRAMES), np.float64)
        self.timing_absolute_error_sum = np.zeros(len(TIMEFRAMES), np.float64)
        self.modal_events = {name: np.zeros((len(TIMEFRAMES), len(STATE_NAMES)), np.int64) for name in
                             ('true_events_by_new_state', 'modal_forecasts_by_new_state',
                              'strict_exact_time_and_pair_hits')}
        self.modal_pair_support = {name: np.zeros(len(TIMEFRAMES), np.int64) for name in
                                   ('observed_state_endpoint_pairs', 'observable_adjacent_pairs',
                                    'tied_modal_pairs_with_observed_states')}
        self.prefix_timing = {name: np.zeros(len(TIMEFRAMES), np.int64) for name in
                              ('windows_with_observed_current_state', 'windows_with_initial_prefix',
                               'windows_with_true_first_change', 'windows_with_predicted_first_change',
                               'windows_with_both_first_changes', 'first_change_same_new_state',
                               'first_change_same_new_state_and_minute',
                               'windows_prefix_stopped_by_unknown_state', 'windows_prefix_stopped_by_modal_tie')}
        self.prefix_length_histogram = np.zeros((len(TIMEFRAMES), len(self.leads) + 1), np.int64)
        self.prefix_signed_error_sum = np.zeros(len(TIMEFRAMES), np.int64)
        self.prefix_absolute_error_sum = np.zeros(len(TIMEFRAMES), np.int64)

    def update(self, probabilities, *, state, eligible, current_state, current_observed,
               sequence_nll, transition_probabilities):
        state, eligible = _labels(state, eligible, self.leads)
        p = np.asarray(probabilities)
        c, co = np.asarray(current_state), np.asarray(current_observed)
        nll = np.asarray(sequence_nll, np.float64)
        b, f, q = state.shape
        _require(p.shape == (b, f, q, len(STATE_NAMES)) and np.issubdtype(p.dtype, np.floating)
                 and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)), 'PROBABILITIES')
        # Finished softmax rows incur class-count rounding; the chain scan accumulates over its grid.
        tolerance = len(STATE_NAMES) * np.finfo(np.float32).eps
        chain_tolerance = q * tolerance
        _require(np.all(np.abs(p.sum(axis=-1) - 1) <= tolerance), 'PROBABILITY_SUM')
        _require(c.shape == co.shape == nll.shape == (b, f) and co.dtype == np.bool_
                 and np.issubdtype(c.dtype, np.integer)
                 and np.all((c[co] >= 0) & (c[co] < len(STATE_NAMES)))
                 and np.isfinite(nll).all() and np.all(nll >= -chain_tolerance), 'CURRENT_OR_NLL')
        transitions = np.asarray(transition_probabilities)
        _require(transitions.shape == (b, f, max(q - 1, 0), len(STATE_NAMES), len(STATE_NAMES))
                 and np.issubdtype(transitions.dtype, np.floating)
                 and np.isfinite(transitions).all() and np.all((transitions >= 0) & (transitions <= 1))
                 and np.all(np.abs(transitions.sum(axis=-1) - 1) <= tolerance), 'TRANSITIONS')
        if q > 1:
            implied = np.einsum('bfqc,bfqcd->bfqd', p[:, :, :-1].astype(np.float64), transitions.astype(np.float64))
            _require(np.all(np.abs(implied - p[:, :, 1:]) <= chain_tolerance), 'JOINT_MARGINAL_COHERENCE')
        self.parents += b
        self.sequence_nll += nll.sum(axis=0)
        self.cells['observed'] += eligible.sum(axis=0)
        y = np.where(eligible, state, 0)
        selected = np.take_along_axis(p, y[..., None], axis=-1)[..., 0].astype(np.float64)
        positive = eligible & (selected > 0)
        logs = np.zeros_like(selected)
        np.log(selected, out=logs, where=positive)
        self.sums['logloss'] -= np.where(positive, logs, 0).sum(axis=0)
        self.cells['logloss_infinite'] += (eligible & ~positive).sum(axis=0)
        pf = p.astype(np.float64)
        self.sums['brier'] += np.where(eligible, np.square(pf).sum(axis=-1) - 2 * selected + 1, 0).sum(axis=0)
        pred = p.argmax(axis=-1)
        unique = (p == p.max(axis=-1, keepdims=True)).sum(axis=-1) == 1
        self.cells['correct'] += (eligible & unique & (pred == state)).sum(axis=0)
        self.cells['tied'] += (eligible & ~unique).sum(axis=0)
        prior_p = np.broadcast_to(self.prior[None, :, None, :], p.shape)
        prior_y = np.take_along_axis(prior_p, y[..., None], axis=-1)[..., 0]
        self.sums['prior_logloss'] -= np.where(eligible, np.log(prior_y), 0).sum(axis=0)
        self.sums['prior_brier'] += np.where(eligible, np.square(prior_p).sum(axis=-1) - 2 * prior_y + 1, 0).sum(axis=0)
        prior_argmax = self.prior.argmax(axis=-1)[None, :, None]
        prior_unique = ((self.prior == self.prior.max(axis=-1, keepdims=True)).sum(axis=-1) == 1)[None, :, None]
        self.cells['prior_correct'] += (eligible & prior_unique & (state == prior_argmax)).sum(axis=0)
        self.cells['prior_tied'] += (eligible & ~prior_unique).sum(axis=0)
        persist_mask = eligible & co[..., None]
        persist_correct = persist_mask & (state == c[..., None])
        self.cells['persistence_observed'] += persist_mask.sum(axis=0)
        self.cells['persistence_correct'] += persist_correct.sum(axis=0)
        self.sums['persistence_brier'] += (2 * (persist_mask & ~persist_correct)).sum(axis=0)
        case = np.where(co, c, len(STATE_NAMES))
        markov_p = self.markov_grid[np.arange(f)[None, :], case]
        markov_y = np.take_along_axis(markov_p, y[..., None], axis=-1)[..., 0]
        markov_positive = eligible & (markov_y > 0)
        markov_log = np.zeros_like(markov_y)
        np.log(markov_y, out=markov_log, where=markov_positive)
        self.sums['markov_logloss'] -= np.where(markov_positive, markov_log, 0).sum(axis=0)
        self.cells['markov_logloss_infinite'] += (eligible & ~markov_positive).sum(axis=0)
        self.sums['markov_brier'] += np.where(eligible, np.square(markov_p).sum(axis=-1) - 2 * markov_y + 1, 0).sum(axis=0)
        markov_unique = (markov_p == markov_p.max(axis=-1, keepdims=True)).sum(axis=-1) == 1
        self.cells['markov_correct'] += (eligible & markov_unique & (markov_p.argmax(axis=-1) == state)).sum(axis=0)
        self.cells['markov_tied'] += (eligible & ~markov_unique).sum(axis=0)
        # For the legacy model the informed Markov above has extra state input.
        # The matched reference uses only the TRAIN prior when the model did not receive that input.
        matched_p = (markov_p if self.model_receives_current_state else
            np.broadcast_to(self.markov_grid[None, :, len(STATE_NAMES), :, :], p.shape))
        matched_y = np.take_along_axis(matched_p, y[..., None], axis=-1)[..., 0]
        matched_positive = eligible & (matched_y > 0)
        matched_log = np.zeros_like(matched_y)
        np.log(matched_y, out=matched_log, where=matched_positive)
        self.matched_state_sums['logloss'] -= np.where(matched_positive, matched_log, 0).sum(axis=0)
        self.matched_state_counts['logloss_infinite'] += (eligible & ~matched_positive).sum(axis=0)
        self.matched_state_sums['brier'] += np.where(eligible, np.square(matched_p).sum(axis=-1) - 2 * matched_y + 1, 0).sum(axis=0)
        matched_unique = (matched_p == matched_p.max(axis=-1, keepdims=True)).sum(axis=-1) == 1
        self.matched_state_counts['correct'] += (eligible & matched_unique & (matched_p.argmax(axis=-1) == state)).sum(axis=0)
        self.matched_state_counts['tied'] += (eligible & ~matched_unique).sum(axis=0)
        for k in range(len(STATE_NAMES)):
            self.class_counts[:, k] += np.count_nonzero(eligible & (state == k), axis=(0, 2))
            self.probability_sums[:, k] += np.where(eligible, pf[..., k], 0).sum(axis=(0, 2))
        for tf in range(f):
            mask = eligible[:, tf] & unique[:, tf]
            code = len(STATE_NAMES) * state[:, tf][mask] + pred[:, tf][mask]
            self.confusion[tf] += np.bincount(code, minlength=len(STATE_NAMES)**2).reshape(len(STATE_NAMES), len(STATE_NAMES))
        if q > 1:
            joint = pf[:, :, :-1, :, None] * transitions.astype(np.float64)
            pair_mask = eligible[:, :, :-1] & eligible[:, :, 1:]
            truth_pair = np.where(pair_mask, len(STATE_NAMES) * state[:, :, :-1] + state[:, :, 1:], 0)
            flat = joint.reshape(b, f, q - 1, len(STATE_NAMES)**2)
            selected_pair = np.take_along_axis(flat, truth_pair[..., None], axis=-1)[..., 0]
            finite_pair = pair_mask & (selected_pair > 0)
            pair_log = np.zeros_like(selected_pair)
            np.log(selected_pair, out=pair_log, where=finite_pair)
            self.pair_sums['logloss'] -= np.where(finite_pair, pair_log, 0).sum(axis=(0, 2))
            self.pairs['logloss_infinite'] += (pair_mask & ~finite_pair).sum(axis=(0, 2))
            self.pair_sums['brier'] += np.where(pair_mask, np.square(flat).sum(axis=-1) - 2 * selected_pair + 1, 0).sum(axis=(0, 2))
            argmax = flat.argmax(axis=-1)
            pair_unique = (flat == flat.max(axis=-1, keepdims=True)).sum(axis=-1) == 1
            true_change = state[:, :, :-1] != state[:, :, 1:]
            pred_change = argmax // len(STATE_NAMES) != argmax % len(STATE_NAMES)
            off_diagonal = ~np.eye(len(STATE_NAMES), dtype=bool)
            change_p = joint[..., off_diagonal].sum(axis=-1)
            self.pair_sums['change_brier'] += np.where(pair_mask, np.square(change_p - true_change), 0).sum(axis=(0, 2))
            cases = {'observed': pair_mask, 'correct': pair_mask & pair_unique & (argmax == truth_pair),
                     'tied': pair_mask & ~pair_unique, 'changes': pair_mask & true_change,
                     'predicted_changes': pair_mask & pair_unique & pred_change,
                     'change_hits': pair_mask & pair_unique & pred_change & true_change,
                     'false_change_alerts': pair_mask & pair_unique & pred_change & ~true_change,
                     'missed_changes': pair_mask & true_change & ~(pair_unique & pred_change)}
            for name, mask in cases.items():
                self.pairs[name] += mask.sum(axis=(0, 2))
        self.matched_transitions.update(p, joint if q > 1 else None, matched_p, state, eligible, c, co)
        # Timing is a limited diagnostic of the first modal change, not a claim of full event matching.
        complete = co & eligible.all(axis=-1) & unique.all(axis=-1)
        true_change = np.concatenate(((state[:, :, :1] != c[..., None]), state[:, :, 1:] != state[:, :, :-1]), axis=-1)
        pred_change = np.concatenate(((pred[:, :, :1] != c[..., None]), pred[:, :, 1:] != pred[:, :, :-1]), axis=-1)
        true_any, pred_any = true_change.any(axis=-1), pred_change.any(axis=-1)
        paired = complete & true_any & pred_any
        delta = self.leads[pred_change.argmax(axis=-1)] - self.leads[true_change.argmax(axis=-1)]
        for name, mask in {'complete_windows': complete, 'true_first_change': complete & true_any,
                           'predicted_first_change': complete & pred_any, 'paired_first_changes': paired}.items():
            self.timing[name] += mask.sum(axis=0)
        self.timing_error_sum += np.where(paired, delta, 0).sum(axis=0)
        self.timing_absolute_error_sum += np.where(paired, np.abs(delta), 0).sum(axis=0)
        self._update_observable_modal_events(state, eligible, pred, unique, c, co)

    def _update_observable_modal_events(self, state, observed, pred, unique, current, current_observed):
        # Same observable endpoint/prefix rules as SAVED_STATE_EVENTS_001.
        # These modal-marginal events do not replace joint-pair argmax metrics.
        prev_true = np.concatenate((current[..., None], state[:, :, :-1]), axis=2)
        prev_pred = np.concatenate((current[..., None], pred[:, :, :-1]), axis=2)
        previous_observed = np.concatenate((current_observed[..., None], observed[:, :, :-1]), axis=2)
        previous_unique = np.concatenate((np.ones_like(current_observed[..., None]), unique[:, :, :-1]), axis=2)
        state_pairs = observed & previous_observed
        eligible = state_pairs & unique & previous_unique
        actual = eligible & (state != prev_true)
        forecast = eligible & (pred != prev_pred)
        hit = actual & forecast & (pred == state) & (prev_pred == prev_true)
        self.modal_pair_support['observed_state_endpoint_pairs'] += state_pairs.sum(axis=(0, 2))
        self.modal_pair_support['observable_adjacent_pairs'] += eligible.sum(axis=(0, 2))
        self.modal_pair_support['tied_modal_pairs_with_observed_states'] += (state_pairs & ~eligible).sum(axis=(0, 2))
        for k in range(len(STATE_NAMES)):
            self.modal_events['true_events_by_new_state'][:, k] += (actual & (state == k)).sum(axis=(0, 2))
            self.modal_events['modal_forecasts_by_new_state'][:, k] += (forecast & (pred == k)).sum(axis=(0, 2))
            self.modal_events['strict_exact_time_and_pair_hits'][:, k] += (hit & (state == k)).sum(axis=(0, 2))

        prefix = np.logical_and.accumulate(observed & unique, axis=2) & current_observed[..., None]
        true_change, pred_change = prefix & (state != prev_true), prefix & (pred != prev_pred)
        lengths = prefix.sum(axis=2)
        true_any, pred_any = true_change.any(axis=2), pred_change.any(axis=2)
        true_at, pred_at = true_change.argmax(axis=2), pred_change.argmax(axis=2)
        bi, fi = np.arange(state.shape[0])[:, None], np.arange(state.shape[1])[None, :]
        both = true_any & pred_any
        same = both & (state[bi, fi, true_at] == pred[bi, fi, pred_at])
        delta = self.leads[pred_at] - self.leads[true_at]
        stopped = current_observed & (lengths < len(self.leads))
        stop_at = np.minimum(lengths, len(self.leads) - 1)
        stop_observed, stop_unique = observed[bi, fi, stop_at], unique[bi, fi, stop_at]
        masks = {
            'windows_with_observed_current_state': current_observed,
            'windows_with_initial_prefix': lengths > 0,
            'windows_with_true_first_change': true_any,
            'windows_with_predicted_first_change': pred_any,
            'windows_with_both_first_changes': both,
            'first_change_same_new_state': same,
            'first_change_same_new_state_and_minute': same & (delta == 0),
            'windows_prefix_stopped_by_unknown_state': stopped & ~stop_observed,
            'windows_prefix_stopped_by_modal_tie': stopped & stop_observed & ~stop_unique}
        for name, mask in masks.items():
            self.prefix_timing[name] += mask.sum(axis=0)
        self.prefix_signed_error_sum += np.where(same, delta, 0).sum(axis=0)
        self.prefix_absolute_error_sum += np.where(same, np.abs(delta), 0).sum(axis=0)
        for tf in range(len(TIMEFRAMES)):
            self.prefix_length_histogram[tf] += np.bincount(lengths[:, tf], minlength=len(self.leads) + 1)

    def finalize(self):
        counts = self.cells['observed']
        total = counts.sum(axis=1)
        per_lead = {k: v.tolist() for k, v in self.cells.items()}
        per_lead.update({name: _ratio(values, counts, self.cells[name + '_infinite'] if name in ('logloss', 'markov_logloss') else None)
                         for name, values in self.sums.items() if name != 'persistence_brier'})
        per_lead['persistence_brier'] = _ratio(self.sums['persistence_brier'], self.cells['persistence_observed'])
        per_tf = {'observed': total.tolist(), 'unobserved': (self.parents * len(self.leads) - total).tolist(),
                  'class_counts': self.class_counts.tolist(), 'confusion_true_pred_unique_only': self.confusion.tolist(),
                  'sequence_nll_per_observed_cell': _ratio(self.sequence_nll, total),
                  'mean_probability': _ratio(self.probability_sums, np.broadcast_to(total[:, None], self.probability_sums.shape)),
                  'observed_class_frequency': _ratio(self.class_counts, np.broadcast_to(total[:, None], self.class_counts.shape))}
        for name, values in self.cells.items():
            if name != 'observed': per_tf[name] = values.sum(axis=1).tolist()
        for name, values in self.sums.items():
            denom = self.cells['persistence_observed'].sum(axis=1) if name == 'persistence_brier' else total
            inf = self.cells[name + '_infinite'].sum(axis=1) if name in ('logloss', 'markov_logloss') else None
            per_tf[name] = _ratio(values.sum(axis=1), denom, inf)
        per_tf['correct_fraction_all_observed'] = _ratio(self.cells['correct'].sum(axis=1), total)
        per_tf['prior_correct_fraction_all_observed'] = _ratio(self.cells['prior_correct'].sum(axis=1), total)
        per_tf['markov_correct_fraction_all_observed'] = _ratio(self.cells['markov_correct'].sum(axis=1), total)
        per_tf['persistence_accuracy'] = _ratio(self.cells['persistence_correct'].sum(axis=1), self.cells['persistence_observed'].sum(axis=1))
        recall_denominator = self.class_counts
        recalls = _ratio(np.diagonal(self.confusion, axis1=1, axis2=2), recall_denominator)
        per_tf['class_recall'] = recalls
        per_tf['balanced_accuracy'] = [None if any(x is None for x in row) else float(sum(row) / len(row)) for row in recalls]
        matched_state = {'observed': counts.tolist(),
            **{name: value.tolist() for name, value in self.matched_state_counts.items()}}
        matched_state.update({name: _ratio(values, counts,
            self.matched_state_counts['logloss_infinite'] if name == 'logloss' else None)
            for name, values in self.matched_state_sums.items()})
        matched_state['information'] = ('causal_observed_current_state_else_TRAIN_prior' if self.model_receives_current_state
                                        else 'TRAIN_prior_only_no_explicit_current_state')
        pairs = {name: value.tolist() for name, value in self.pairs.items()}
        for name, values in self.pair_sums.items():
            pairs[name] = _ratio(values, self.pairs['observed'], self.pairs['logloss_infinite'] if name == 'logloss' else None)
        timing = {name: value.tolist() for name, value in self.timing.items()}
        timing['signed_error_minutes'] = _ratio(self.timing_error_sum, self.timing['paired_first_changes'])
        timing['absolute_error_minutes'] = _ratio(self.timing_absolute_error_sum, self.timing['paired_first_changes'])
        events = {name: values.tolist() for name, values in self.modal_events.items()}
        events.update({name: values.tolist() for name, values in self.modal_pair_support.items()})
        events['candidate_adjacent_pairs'] = [self.parents * len(self.leads)] * len(TIMEFRAMES)
        events['unobservable_state_endpoint_pairs'] = (
            self.parents * len(self.leads) - self.modal_pair_support['observed_state_endpoint_pairs']).tolist()
        hits = self.modal_events['strict_exact_time_and_pair_hits']
        events['missed_true_events_by_new_state'] = (self.modal_events['true_events_by_new_state'] - hits).tolist()
        events['unmatched_modal_forecasts_by_new_state'] = (self.modal_events['modal_forecasts_by_new_state'] - hits).tolist()
        events['definition'] = ('Adjacent observed endpoints, including causal current-to-lead1; both modal endpoints must be unique. '
                                'Strict hits match original state, new state and exact minute. Counts are per TF and new DOWN/FLAT/UP state.')
        events['limits'] = [
            'Unknown state endpoints and tied modal endpoints are excluded; no interpolation through gaps.',
            'True-event counts use the same observed-and-unique comparison opportunities as forecasts.',
            'Modal marginal changes differ from joint-pair argmax decisions in adjacent_pair_forecast.',
            'Overlapping parent/lead opportunities are dependent, not distinct market events or economic edge.']
        prefix_timing = {name: value.tolist() for name, value in self.prefix_timing.items()}
        pt = self.prefix_timing
        prefix_timing.update({
            'windows_without_observed_current_state': (self.parents - pt['windows_with_observed_current_state']).tolist(),
            'windows_without_initial_prefix': (self.parents - pt['windows_with_initial_prefix']).tolist(),
            'windows_with_full_initial_prefix': self.prefix_length_histogram[:, -1].tolist(),
            'first_changes_different_new_state': (pt['windows_with_both_first_changes'] - pt['first_change_same_new_state']).tolist(),
            'true_first_change_without_predicted_change': (pt['windows_with_true_first_change'] - pt['windows_with_both_first_changes']).tolist(),
            'predicted_first_change_without_true_change': (pt['windows_with_predicted_first_change'] - pt['windows_with_both_first_changes']).tolist(),
            'true_first_changes_without_same_new_state_forecast': (pt['windows_with_true_first_change'] - pt['first_change_same_new_state']).tolist(),
            'predicted_first_changes_without_same_new_state_truth': (pt['windows_with_predicted_first_change'] - pt['first_change_same_new_state']).tolist(),
            'windows_with_neither_first_change': (pt['windows_with_initial_prefix'] - pt['windows_with_true_first_change']
                - pt['windows_with_predicted_first_change'] + pt['windows_with_both_first_changes']).tolist(),
            'absolute_error_sum_minutes': self.prefix_absolute_error_sum.tolist(),
            'signed_error_sum_minutes': self.prefix_signed_error_sum.tolist(),
            'absolute_error_mean_minutes': _ratio(self.prefix_absolute_error_sum, pt['first_change_same_new_state']),
            'signed_error_mean_minutes': _ratio(self.prefix_signed_error_sum, pt['first_change_same_new_state']),
            'initial_prefix_median_minutes': [None if not self.parents else int(np.searchsorted(np.cumsum(row), (self.parents + 1) // 2))
                                              for row in self.prefix_length_histogram],
            'definition': 'First change inside the contiguous prefix from a known current state; stop at the first unknown state or modal tie.',
            'limits': [
                'Timing errors condition on both first changes having the same new state; positive signed error means forecast later.',
                'No event inside a censored prefix does not establish absence later in the forecast window.',
                'Prefix median is the lower empirical median over all parent windows, including zero-length prefixes.',
                'Overlapping parent/lead opportunities are dependent, not distinct events, full trajectory matching or economic edge.']})
        return {'schema_version': SCHEMA_VERSION, 'parents': self.parents, 'timeframes': list(TIMEFRAMES),
                'class_names': list(STATE_NAMES), 'lead_minutes': self.leads.tolist(),
                'per_timeframe': per_tf, 'per_lead': per_lead, 'adjacent_pair_forecast': pairs,
                'matched_markov_state_baseline': matched_state,
                'matched_markov_transition_comparison': self.matched_transitions.finalize(),
                'first_modal_change_timing_complete_windows_only': timing,
                'observable_modal_state_events': events,
                'first_modal_change_within_initial_observed_prefix': prefix_timing,
                'limits': ['Probabilities are frozen at issue time; pair forecasts use model marginals, never future target conditioning.',
                           'Markov reference uses TRAIN-only adjacent transition frequencies and causal current state, or TRAIN class prior when current state is unknown.',
                           'Unknown/mature masks come from the target/loss owners; unknown is not FLAT.',
                           'Tied modal predictions remain unclassified and are counted separately.',
                           'A null logloss with positive infinite_count means infinite loss, not missing evidence.',
                           'Calibration reported here is calibration in the large only, not full conditional calibration.',
                           'Timing matches only the first modal change in fully observed windows; all observed adjacent pairs are scored separately.',
                           'Overlapping parents, future minutes and TF are dependent; these descriptive metrics are not independent-sample confidence intervals or trading edge.']}


class FirstFlatActivityAccumulator:
    """Score complete first-query FLAT/non-FLAT labels, including unknown direction.

    The provided Markov grid is the same frozen TRAIN prior and causal-current
    information used by StateScoreAccumulator. No selection, smoothing or trade.
    """
    def __init__(self, first_markov_grid):
        grid = np.asarray(first_markov_grid, dtype=np.float64)
        f, k = len(TIMEFRAMES), len(STATE_NAMES)
        tolerance = k * np.finfo(np.float32).eps
        _require(grid.shape == (f, k + 1, k) and np.isfinite(grid).all()
                 and np.all((grid >= 0) & (grid <= 1))
                 and np.all(np.abs(grid.sum(axis=-1) - 1) <= tolerance), 'ACTIVITY_MARKOV_GRID')
        self.grid = grid.copy()
        self.counts = {key: np.zeros(f, np.int64) for key in
                       ('observed', 'flat', 'nonflat', 'direction_unknown')}
        self.metrics = {name: {key: np.zeros(f, np.int64 if key in ('infinite', 'tied', 'correct', 'flat_hits', 'predicted_flat') else np.float64)
                               for key in ('infinite', 'tied', 'correct', 'flat_hits', 'predicted_flat', 'logloss', 'brier')}
                        for name in ('model', 'markov')}

    def update(self, model_flat, *, activity_observed, is_flat, state, state_observed,
               current_state, current_observed):
        p, mask, truth, state, known, current, co = map(np.asarray, (
            model_flat, activity_observed, is_flat, state, state_observed, current_state, current_observed))
        f, k = len(TIMEFRAMES), len(STATE_NAMES)
        _require(p.ndim == 2 and p.shape[1] == f and all(a.shape == p.shape for a in
                 (mask, truth, state, known, current, co)) and mask.dtype == truth.dtype == known.dtype == co.dtype == np.bool_
                 and np.issubdtype(state.dtype, np.integer) and np.issubdtype(current.dtype, np.integer)
                 and np.issubdtype(p.dtype, np.floating) and np.isfinite(p).all()
                 and np.all((p >= 0) & (p <= 1)), 'ACTIVITY_INPUT')
        flat = STATE_NAMES.index('FLAT')
        _require(np.all(~truth | mask) and np.all((state[known] >= 0) & (state[known] < k))
                 and np.all(state[~known] == -1) and np.all((current[co] >= 0) & (current[co] < k))
                 and np.all(current[~co] == -1)
                 and np.all(~truth | (known & (state == flat)))
                 and np.all(~(mask & known) | ((state == flat) == truth)), 'ACTIVITY_LABEL_CONSISTENCY')
        case = np.where(co, current, k)
        markov = self.grid[np.arange(f)[None, :], case, flat]
        for name, event in (('observed', mask), ('flat', mask & truth),
                            ('nonflat', mask & ~truth), ('direction_unknown', mask & ~known)):
            self.counts[name] += event.sum(axis=0)
        for name, prob in (('model', p.astype(np.float64)), ('markov', markov)):
            metrics = self.metrics[name]
            selected = np.where(truth, prob, 1.0 - prob)
            positive = mask & (selected > 0)
            logs = np.zeros_like(selected)
            np.log(selected, out=logs, where=positive)
            metrics['logloss'] -= np.where(positive, logs, 0).sum(axis=0)
            metrics['infinite'] += (mask & ~positive).sum(axis=0)
            metrics['brier'] += np.where(mask, (prob - truth.astype(np.float64)) ** 2, 0).sum(axis=0)
            tied, predicted_flat = prob == 0.5, prob > 0.5
            metrics['tied'] += (mask & tied).sum(axis=0)
            metrics['correct'] += (mask & ~tied & (predicted_flat == truth)).sum(axis=0)
            metrics['predicted_flat'] += (mask & predicted_flat).sum(axis=0)
            metrics['flat_hits'] += (mask & truth & predicted_flat).sum(axis=0)

    def finalize(self):
        counts = {key: value.tolist() for key, value in self.counts.items()}
        output = {'timeframes': list(TIMEFRAMES), 'counts': counts,
                  'scope': 'complete_first_query_local_windows_only; unknown_direction_remains_unknown',
                  'baseline': 'frozen_TRAIN_Markov_given_same_causal_current_state_or_TRAIN_prior_fallback',
                  'mask_selection_bias_not_excluded': True}
        for name, metrics in self.metrics.items():
            output[name] = {'logloss': _ratio(metrics['logloss'], self.counts['observed'], metrics['infinite']),
                            'logloss_infinite': metrics['infinite'].tolist(),
                            'brier': _ratio(metrics['brier'], self.counts['observed']),
                            'correct': metrics['correct'].tolist(), 'tied': metrics['tied'].tolist(),
                            'predicted_flat': metrics['predicted_flat'].tolist(),
                            'flat_hits': metrics['flat_hits'].tolist(),
                            'flat_recall': _ratio(metrics['flat_hits'], self.counts['flat']),
                            'flat_precision': _ratio(metrics['flat_hits'], metrics['predicted_flat'])}
        return output
