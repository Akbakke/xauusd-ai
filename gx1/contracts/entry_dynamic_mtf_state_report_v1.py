"""Streaming evidence for the explicit dynamic MTF market-state forecast.

No threshold selection, model calls, trade decisions or target reconstruction.
The caller supplies the exact mature-observed mask from the objective owner.
"""
from __future__ import annotations

import numpy as np

from gx1.contracts.entry_dynamic_mtf_state_targets_v1 import STATE_NAMES, TIMEFRAMES

SCHEMA_VERSION = 'gx1_dynamic_mtf_state_report_v1'


def _require(condition, code):
    if not condition:
        raise RuntimeError('[DYNAMIC_STATE_REPORT_' + code + ']')


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


class StateScoreAccumulator:
    def __init__(self, prior, *, lead_minutes):
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
        pairs = {name: value.tolist() for name, value in self.pairs.items()}
        for name, values in self.pair_sums.items():
            pairs[name] = _ratio(values, self.pairs['observed'], self.pairs['logloss_infinite'] if name == 'logloss' else None)
        timing = {name: value.tolist() for name, value in self.timing.items()}
        timing['signed_error_minutes'] = _ratio(self.timing_error_sum, self.timing['paired_first_changes'])
        timing['absolute_error_minutes'] = _ratio(self.timing_absolute_error_sum, self.timing['paired_first_changes'])
        return {'schema_version': SCHEMA_VERSION, 'parents': self.parents, 'timeframes': list(TIMEFRAMES),
                'class_names': list(STATE_NAMES), 'lead_minutes': self.leads.tolist(),
                'per_timeframe': per_tf, 'per_lead': per_lead, 'adjacent_pair_forecast': pairs,
                'first_modal_change_timing_complete_windows_only': timing,
                'limits': ['Probabilities are frozen at issue time; pair forecasts use model marginals, never future target conditioning.',
                           'Markov reference uses TRAIN-only adjacent transition frequencies and causal current state, or TRAIN class prior when current state is unknown.',
                           'Unknown/mature masks come from the target/loss owners; unknown is not FLAT.',
                           'Tied modal predictions remain unclassified and are counted separately.',
                           'A null logloss with positive infinite_count means infinite loss, not missing evidence.',
                           'Calibration reported here is calibration in the large only, not full conditional calibration.',
                           'Timing matches only the first modal change in fully observed windows; all observed adjacent pairs are scored separately.',
                           'Overlapping parents, future minutes and TF are dependent; these descriptive metrics are not independent-sample confidence intervals or trading edge.']}
