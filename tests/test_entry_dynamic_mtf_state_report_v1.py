"""Small probability identities and streaming evidence checks; no model calls."""
import json
import numpy as np
import pytest
from gx1.contracts.entry_dynamic_mtf_state_report_v1 import StatePriorAccumulator, StateScoreAccumulator
from gx1.contracts.entry_dynamic_mtf_state_targets_v1 import TIMEFRAMES


def _data():
    # Every class and every transition occur, independently of the TF repetition.
    paths = np.array([[a, b, c] for a in range(3) for b in range(3) for c in range(3)], np.int8)
    state = np.repeat(paths[:, None, :], len(TIMEFRAMES), axis=1)
    eligible = np.ones_like(state, bool)
    return state, eligible


def _prior(state, eligible):
    prior = StatePriorAccumulator(lead_minutes=[1, 2, 3])
    prior.update(state, eligible)
    return prior.finalize()


def _update(acc, state, eligible, p):
    acc.update(p, state=state, eligible=eligible,
               current_state=state[:, :, 0], current_observed=np.ones(state.shape[:2], bool),
               sequence_nll=np.full(state.shape[:2], 3 * np.log(3)),
               transition_probabilities=np.broadcast_to(p[:, :, 1:, None, :], (*state.shape[:2], 2, 3, 3)))


def test_uniform_joint_scores_and_ties_are_honest():
    state, eligible = _data()
    acc = StateScoreAccumulator(_prior(state, eligible), lead_minutes=[1, 2, 3])
    _update(acc, state, eligible, np.full((*state.shape, 3), 1 / 3))
    report = acc.finalize()
    json.dumps(report, allow_nan=False)
    tf = report['per_timeframe']
    assert tf['observed'] == [81] * len(TIMEFRAMES)
    assert tf['tied'] == [81] * len(TIMEFRAMES)
    assert tf['correct'] == [0] * len(TIMEFRAMES)
    assert tf['logloss'] == pytest.approx([np.log(3)] * len(TIMEFRAMES))
    assert tf['brier'] == pytest.approx([2 / 3] * len(TIMEFRAMES))
    assert tf['markov_logloss'] == pytest.approx(tf['logloss'])
    assert report['adjacent_pair_forecast']['logloss'] == pytest.approx([np.log(9)] * len(TIMEFRAMES))
    assert report['adjacent_pair_forecast']['observed'] == [54] * len(TIMEFRAMES)


def test_streaming_equals_one_batch_and_unknown_is_not_flat():
    state, eligible = _data()
    prior = _prior(state, eligible)
    state = state.copy(); eligible = eligible.copy()
    eligible[1:4, :, 1] = False; state[~eligible] = -1
    p = np.broadcast_to(np.array([0.2, 0.3, 0.5]), (*state.shape, 3)).copy()
    full = StateScoreAccumulator(prior, lead_minutes=[1, 2, 3])
    split = StateScoreAccumulator(prior, lead_minutes=[1, 2, 3])
    _update(full, state, eligible, p)
    for left, right in ((0, 9), (9, 27)):
        _update(split, state[left:right], eligible[left:right], p[left:right])
    a, b = full.finalize(), split.finalize()
    assert a['per_timeframe']['class_counts'] == b['per_timeframe']['class_counts']
    assert a['per_timeframe']['observed'] == [78] * len(TIMEFRAMES)
    assert a['per_timeframe']['logloss'] == pytest.approx(b['per_timeframe']['logloss'])
    assert a['adjacent_pair_forecast']['observed'] == [48] * len(TIMEFRAMES)
    assert a['adjacent_pair_forecast']['logloss'] == pytest.approx(b['adjacent_pair_forecast']['logloss'])
    assert a['observable_modal_state_events'] == b['observable_modal_state_events']
    assert a['first_modal_change_within_initial_observed_prefix'] == b['first_modal_change_within_initial_observed_prefix']
    assert a['matched_markov_transition_comparison'] == b['matched_markov_transition_comparison']
    for key in ('observed', 'correct', 'tied', 'logloss_infinite'):
        assert a['matched_markov_state_baseline'][key] == b['matched_markov_state_baseline'][key]
    for key in ('logloss', 'brier'):
        assert np.asarray(a['matched_markov_state_baseline'][key]) == pytest.approx(np.asarray(b['matched_markov_state_baseline'][key]))


def test_zero_probability_is_infinite_not_silently_clipped():
    state, eligible = _data()
    acc = StateScoreAccumulator(_prior(state, eligible), lead_minutes=[1, 2, 3])
    p = np.zeros((*state.shape, 3)); p[..., 0] = 1
    _update(acc, state, eligible, p)
    report = acc.finalize()
    assert report['per_timeframe']['logloss'] == [None] * len(TIMEFRAMES)
    assert report['per_timeframe']['logloss_infinite'] == [54] * len(TIMEFRAMES)
    json.dumps(report, allow_nan=False)


def test_prior_does_not_fabricate_missing_classes():
    state = np.ones((2, len(TIMEFRAMES), 3), np.int8)
    prior = StatePriorAccumulator(lead_minutes=[1, 2, 3])
    prior.update(state, np.ones_like(state, bool))
    with pytest.raises(RuntimeError, match='TRAIN_.*REQUIRED'):
        prior.finalize()


def test_marginal_and_transition_mismatch_rejected_before_accumulation():
    state, eligible = _data()
    acc = StateScoreAccumulator(_prior(state, eligible), lead_minutes=[1, 2, 3])
    p = np.broadcast_to(np.array([0.2, 0.3, 0.5]), (*state.shape, 3)).copy()
    with pytest.raises(RuntimeError, match='JOINT_MARGINAL_COHERENCE'):
        acc.update(p, state=state, eligible=eligible,
                   current_state=state[:, :, 0], current_observed=np.ones(state.shape[:2], bool),
                   sequence_nll=np.ones(state.shape[:2]),
                   transition_probabilities=np.full((*state.shape[:2], 2, 3, 3), 1 / 3))
    assert acc.parents == 0


@pytest.mark.parametrize('field', ['counts', 'transition_counts'])
def test_fractional_prior_counts_rejected(field):
    state, eligible = _data()
    prior = _prior(state, eligible)
    prior[field] = (np.asarray(prior[field], np.float64) + 0.5).tolist()
    with pytest.raises(RuntimeError, match='COUNT_DTYPE'):
        StateScoreAccumulator(prior, lead_minutes=[1, 2, 3])


def test_probability_above_one_rejected_before_accumulation():
    state, eligible = _data()
    acc = StateScoreAccumulator(_prior(state, eligible), lead_minutes=[1, 2, 3])
    p = np.zeros((*state.shape, 3)); p[..., 0] = 1 + np.finfo(np.float32).eps
    with pytest.raises(RuntimeError, match='PROBABILITIES'):
        _update(acc, state, eligible, p)
    assert acc.parents == 0


def _event_fixture(states, predictions, current, *, current_observed=None, ties=()):
    states, predictions = np.atleast_2d(np.asarray(states, np.int8)), np.atleast_2d(np.asarray(predictions, np.int8))
    assert states.shape == predictions.shape
    state = np.repeat(states[:, None, :], len(TIMEFRAMES), axis=1)
    pred = np.repeat(predictions[:, None, :], len(TIMEFRAMES), axis=1)
    eligible = state >= 0
    batch, _, length = state.shape
    leads = list(range(1, length + 1))
    # Genuine synthetic prior accumulation, with all classes/transition rows supported.
    training, _ = _data()
    training = np.tile(training, (1, 1, (length + 2) // 3))[:, :, :length]
    prior_acc = StatePriorAccumulator(lead_minutes=leads)
    prior_acc.update(training, np.ones_like(training, bool))
    prior = prior_acc.finalize()
    p = np.full((*state.shape, 3), 0.1)
    np.put_along_axis(p, pred[..., None], 0.8, axis=-1)
    for parent, lead_index in ties:
        p[parent, :, lead_index] = [0.4, 0.4, 0.2]
    c = np.repeat(np.asarray(current, np.int8).reshape(batch, 1), len(TIMEFRAMES), axis=1)
    known = np.ones(batch, bool) if current_observed is None else np.asarray(current_observed, bool)
    co = np.repeat(known.reshape(batch, 1), len(TIMEFRAMES), axis=1)
    selected = np.take_along_axis(p, np.where(eligible, state, 0)[..., None], axis=-1)[..., 0]
    kwargs = dict(probabilities=p, state=state, eligible=eligible, current_state=c, current_observed=co,
                  sequence_nll=-np.where(eligible, np.log(selected), 0).sum(axis=-1),
                  transition_probabilities=np.broadcast_to(p[:, :, 1:, None, :], (*state.shape[:2], length - 1, 3, 3)))
    return prior, kwargs


def _event_report(*args, **kwargs):
    prior, batch = _event_fixture(*args, **kwargs)
    acc = StateScoreAccumulator(prior, lead_minutes=prior['lead_minutes'])
    acc.update(**batch)
    report = acc.finalize()
    json.dumps(report, allow_nan=False)
    return report


def test_modal_reversals_count_both_directions_and_causal_current_to_lead1():
    report = _event_report([[2, 0, 2, 1, 2]], [[2, 0, 2, 1, 2]], [0])
    events = report['observable_modal_state_events']
    assert events['candidate_adjacent_pairs'] == [5] * len(TIMEFRAMES)
    assert events['observable_adjacent_pairs'] == [5] * len(TIMEFRAMES)
    assert report['adjacent_pair_forecast']['observed'] == [4] * len(TIMEFRAMES)
    for name in ('true_events_by_new_state', 'modal_forecasts_by_new_state', 'strict_exact_time_and_pair_hits'):
        assert events[name] == [[1, 1, 3]] * len(TIMEFRAMES)
    assert events['missed_true_events_by_new_state'] == [[0, 0, 0]] * len(TIMEFRAMES)
    assert events['unmatched_modal_forecasts_by_new_state'] == [[0, 0, 0]] * len(TIMEFRAMES)
    timing = report['first_modal_change_within_initial_observed_prefix']
    assert timing['first_change_same_new_state_and_minute'] == [1] * len(TIMEFRAMES)
    assert timing['absolute_error_mean_minutes'] == [0.0] * len(TIMEFRAMES)


def test_modal_hits_require_original_state_new_state_and_same_minute():
    report = _event_report([[2, 0, 0, 1, 1]], [[1, 0, 0, 2, 1]], [0])
    events = report['observable_modal_state_events']
    # Lead2 reaches DOWN in both paths, but from different original states.
    assert events['true_events_by_new_state'] == [[1, 1, 1]] * len(TIMEFRAMES)
    assert events['modal_forecasts_by_new_state'] == [[1, 2, 1]] * len(TIMEFRAMES)
    assert events['strict_exact_time_and_pair_hits'] == [[0, 0, 0]] * len(TIMEFRAMES)
    assert events['missed_true_events_by_new_state'] == events['true_events_by_new_state']
    assert events['unmatched_modal_forecasts_by_new_state'] == events['modal_forecasts_by_new_state']
    timing = report['first_modal_change_within_initial_observed_prefix']
    assert timing['first_changes_different_new_state'] == [1] * len(TIMEFRAMES)
    assert timing['absolute_error_mean_minutes'] == [None] * len(TIMEFRAMES)


def test_modal_events_do_not_bridge_gaps_but_resume_on_observed_adjacent_pairs():
    # Same hand-checked witness as the saved-only review.
    report = _event_report([[0, 1, -1, 2, 0]], [[0, 1, 0, 2, 2]], [0])
    events = report['observable_modal_state_events']
    assert events['observable_adjacent_pairs'] == [3] * len(TIMEFRAMES)
    assert events['unobservable_state_endpoint_pairs'] == [2] * len(TIMEFRAMES)
    assert events['true_events_by_new_state'] == [[1, 1, 0]] * len(TIMEFRAMES)
    assert events['modal_forecasts_by_new_state'] == [[0, 1, 0]] * len(TIMEFRAMES)
    assert events['strict_exact_time_and_pair_hits'] == [[0, 1, 0]] * len(TIMEFRAMES)
    assert events['missed_true_events_by_new_state'] == [[1, 0, 0]] * len(TIMEFRAMES)
    timing = report['first_modal_change_within_initial_observed_prefix']
    assert timing['windows_with_initial_prefix'] == [1] * len(TIMEFRAMES)
    assert timing['initial_prefix_median_minutes'] == [2] * len(TIMEFRAMES)
    assert timing['windows_prefix_stopped_by_unknown_state'] == [1] * len(TIMEFRAMES)
    assert timing['first_change_same_new_state_and_minute'] == [1] * len(TIMEFRAMES)
    assert report['first_modal_change_timing_complete_windows_only']['complete_windows'] == [0] * len(TIMEFRAMES)


def test_modal_tie_excludes_both_touching_pairs_and_stops_initial_prefix():
    report = _event_report([[0, 2, 0, 2, 0]], [[0, 2, 0, 2, 0]], [0], ties=((0, 1),))
    events = report['observable_modal_state_events']
    assert events['observed_state_endpoint_pairs'] == [5] * len(TIMEFRAMES)
    assert events['tied_modal_pairs_with_observed_states'] == [2] * len(TIMEFRAMES)
    assert events['observable_adjacent_pairs'] == [3] * len(TIMEFRAMES)
    assert events['true_events_by_new_state'] == [[1, 0, 1]] * len(TIMEFRAMES)
    assert events['strict_exact_time_and_pair_hits'] == [[1, 0, 1]] * len(TIMEFRAMES)
    timing = report['first_modal_change_within_initial_observed_prefix']
    assert timing['initial_prefix_median_minutes'] == [1] * len(TIMEFRAMES)
    assert timing['windows_prefix_stopped_by_modal_tie'] == [1] * len(TIMEFRAMES)
    assert timing['windows_with_true_first_change'] == [0] * len(TIMEFRAMES)
    assert timing['windows_with_predicted_first_change'] == [0] * len(TIMEFRAMES)
    assert timing['absolute_error_mean_minutes'] == [None] * len(TIMEFRAMES)


def test_unknown_current_masks_only_start_pair_and_all_initial_prefix_timing():
    report = _event_report([[2, 0, 2, 1, 0]], [[2, 0, 2, 1, 0]], [-1], current_observed=[False])
    events = report['observable_modal_state_events']
    assert events['observable_adjacent_pairs'] == [4] * len(TIMEFRAMES)
    assert events['unobservable_state_endpoint_pairs'] == [1] * len(TIMEFRAMES)
    assert events['strict_exact_time_and_pair_hits'] == [[2, 1, 1]] * len(TIMEFRAMES)
    timing = report['first_modal_change_within_initial_observed_prefix']
    assert timing['windows_without_observed_current_state'] == [1] * len(TIMEFRAMES)
    assert timing['windows_without_initial_prefix'] == [1] * len(TIMEFRAMES)
    assert timing['windows_with_initial_prefix'] == [0] * len(TIMEFRAMES)
    assert timing['windows_prefix_stopped_by_unknown_state'] == [0] * len(TIMEFRAMES)
    assert timing['initial_prefix_median_minutes'] == [0] * len(TIMEFRAMES)
    assert timing['signed_error_mean_minutes'] == [None] * len(TIMEFRAMES)


def test_first_prefix_timing_conditions_errors_on_matching_new_state_and_reports_misses():
    states = [[0, 2, 2, -1, 0], [0, 0, 2, -1, 0], [0, 2, 2, -1, 0],
              [0, 2, 2, -1, 0], [0, 0, 0, -1, 0]]
    preds = [[0, 0, 2, 0, 0], [0, 2, 2, 0, 0], [0, 1, 1, 0, 0],
             [0, 0, 0, 0, 0], [0, 1, 1, 0, 0]]
    report = _event_report(states, preds, [0] * 5)
    timing = report['first_modal_change_within_initial_observed_prefix']
    for name, expected in {
        'windows_with_initial_prefix': 5, 'windows_with_true_first_change': 4,
        'windows_with_predicted_first_change': 4, 'windows_with_both_first_changes': 3,
        'first_change_same_new_state': 2, 'first_change_same_new_state_and_minute': 0,
        'first_changes_different_new_state': 1, 'true_first_change_without_predicted_change': 1,
        'predicted_first_change_without_true_change': 1,
        'true_first_changes_without_same_new_state_forecast': 2,
        'predicted_first_changes_without_same_new_state_truth': 2,
        'windows_prefix_stopped_by_unknown_state': 5, 'initial_prefix_median_minutes': 3,
        'absolute_error_sum_minutes': 2, 'signed_error_sum_minutes': 0,
        'absolute_error_mean_minutes': 1.0, 'signed_error_mean_minutes': 0.0,
    }.items():
        assert timing[name] == [expected] * len(TIMEFRAMES), name
    assert report['first_modal_change_timing_complete_windows_only']['complete_windows'] == [0] * len(TIMEFRAMES)


def test_additive_event_accumulation_preserves_every_legacy_report_field(monkeypatch):
    prior, batch = _event_fixture([[2, 0, -1, 1, 2], [0, 1, 2, 0, 1]],
                                  [[1, 0, 0, 2, 2], [0, 1, 0, 0, 1]], [0, -1],
                                  current_observed=[True, False], ties=((1, 2),))
    ordinary = StateScoreAccumulator(prior, lead_minutes=prior['lead_minutes'])
    without_addition = StateScoreAccumulator(prior, lead_minutes=prior['lead_minutes'])
    monkeypatch.setattr(without_addition, '_update_observable_modal_events', lambda *args: None)
    ordinary.update(**batch)
    without_addition.update(**batch)
    actual, legacy = ordinary.finalize(), without_addition.finalize()
    for name in ('observable_modal_state_events', 'first_modal_change_within_initial_observed_prefix'):
        actual.pop(name)
        legacy.pop(name)
    assert actual == legacy
