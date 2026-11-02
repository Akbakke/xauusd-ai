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
