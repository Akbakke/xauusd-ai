"""Matched state/switch reference: synthetic probabilities only; no model calls."""
from copy import deepcopy
from types import SimpleNamespace
import json

import numpy as np
import pytest

from gx1.contracts.entry_dynamic_mtf_state_report_v1 import StateScoreAccumulator
from gx1.contracts.entry_dynamic_mtf_state_targets_v1 import TIMEFRAMES
from gx1.scripts.run_entry_native_direction_prefix_fit_window_v1 import _state_report_current_input


def _prior(leads=(1, 2)):
    counts = np.tile(np.array([3, 2, 1], np.int64), (len(TIMEFRAMES), 1))
    transitions = np.tile(np.array([[8, 1, 1], [1, 8, 1], [1, 1, 8]], np.int64), (len(TIMEFRAMES), 1, 1))
    return {'schema_version':'gx1_dynamic_mtf_state_report_v1','timeframes':list(TIMEFRAMES),
        'class_names':['DOWN','FLAT','UP'],'lead_minutes':list(leads),'parents':6,
        'counts':counts.tolist(),'probabilities':(counts/counts.sum(axis=-1,keepdims=True)).tolist(),
        'transition_counts':transitions.tolist(),
        'transition_probabilities':(transitions/transitions.sum(axis=-1,keepdims=True)).tolist()}


def _run(*, model_receives_current_state, ties=False, missing=False, leads=(1, 2)):
    first = np.array([1/3, 1/3, 1/3] if ties else [.9, .05, .05], np.float64)
    second = np.array([1/3, 1/3, 1/3] if ties else [.05, .05, .9], np.float64)
    ps = np.stack([first] + ([second] if len(leads) == 2 else []))
    p = np.broadcast_to(ps[None, None, :, :], (1, 5, len(leads), 3)).copy()
    state = np.broadcast_to(np.array([0] + ([2] if len(leads) == 2 else []), np.int8)[None, None, :],
                            (1, 5, len(leads))).copy()
    eligible = np.ones_like(state, bool)
    current = np.full((1, 5), 2, np.int8)
    current_observed = np.ones_like(current, bool)
    if missing:
        state[:, :, 0] = -1; eligible[:, :, 0] = False; current[:] = -1; current_observed[:] = False
    if len(leads) == 2:
        trans = np.broadcast_to(second, (1, 5, 1, 3, 3)).copy()
    else:
        trans = np.empty((1, 5, 0, 3, 3), np.float64)
    score = StateScoreAccumulator(_prior(leads), lead_minutes=leads,
                                  model_receives_current_state=model_receives_current_state)
    score.update(p, state=state, eligible=eligible, current_state=current, current_observed=current_observed,
                 sequence_nll=np.zeros((1, 5)), transition_probabilities=trans)
    report = score.finalize(); json.dumps(report, allow_nan=False)
    return report


def test_baseline_never_receives_explicit_current_state_missing_from_model():
    no_current = _run(model_receives_current_state=False)
    with_current = _run(model_receives_current_state=True)
    a = no_current['matched_markov_transition_comparison']
    b = with_current['matched_markov_transition_comparison']
    assert a['markov_uses_current_state_when_observed'] is False
    assert b['markov_uses_current_state_when_observed'] is True
    for report in (a, b):
        first = report['groups']['current_to_first']; later = report['groups']['future_adjacent']
        assert first['true_changes'] == later['true_changes'] == [1] * 5
        assert first['truth_by_from_to'][0][2][0] == 1  # UP -> DOWN
        assert later['truth_by_from_to'][0][0][2] == 1  # DOWN -> UP
        assert first['model_exact_change_hits'] == later['model_exact_change_hits'] == [1] * 5
    assert a['groups']['current_to_first']['markov_exact_change_hits'] == [1] * 5
    assert b['groups']['current_to_first']['markov_exact_change_hits'] == [0] * 5
    assert a['groups']['future_adjacent']['markov_exact_change_hits'] == [0] * 5
    assert b['groups']['future_adjacent']['markov_exact_change_hits'] == [0] * 5
    assert no_current['matched_markov_state_baseline']['correct'][0][0] == 1
    assert with_current['matched_markov_state_baseline']['correct'][0][0] == 0
    # Additive report: the existing informed Markov fields are unchanged.
    for old in ('per_timeframe', 'per_lead', 'adjacent_pair_forecast', 'observable_modal_state_events'):
        assert no_current[old] == with_current[old]


def test_ties_are_reported_and_do_not_hide_true_changes():
    r = _run(model_receives_current_state=True, ties=True)['matched_markov_transition_comparison']
    for group in ('current_to_first', 'future_adjacent'):
        x = r['groups'][group]
        assert x['true_changes'] == [1] * 5 and x['model_tied_pairs'] == [1] * 5
        assert x['model_exact_change_hits'] == [0] * 5
        assert x['model_change_recall'] == [0.] * 5 and x['model_change_precision'] == [None] * 5


def test_unknown_state_is_not_bridged_or_turned_into_flat():
    r = _run(model_receives_current_state=True, missing=True)['matched_markov_transition_comparison']
    assert r['groups']['current_to_first']['observed_pairs'] == [0] * 5
    assert r['groups']['future_adjacent']['observed_pairs'] == [0] * 5
    assert r['groups']['future_adjacent']['model_change_recall'] == [None] * 5


def test_single_lead_has_no_future_adjacent_pair():
    r = _run(model_receives_current_state=False, leads=(1,))['matched_markov_transition_comparison']
    assert r['groups']['current_to_first']['observed_pairs'] == [1] * 5
    assert r['groups']['future_adjacent']['observed_pairs'] == [0] * 5


def test_model_and_view_must_agree_on_actual_current_input():
    assert _state_report_current_input(SimpleNamespace(current_state_input=False), {}) is False
    assert _state_report_current_input(SimpleNamespace(current_state_input=True),
        {'current_state_first_loss_weight': 1.0}) is True
    with pytest.raises(RuntimeError, match='STATE_REPORT_CURRENT_INPUT_ALIGNMENT'):
        _state_report_current_input(SimpleNamespace(current_state_input=False),
            {'current_state_first_loss_weight': 1.0})
    with pytest.raises(RuntimeError, match='STATE_REPORT_CURRENT_INPUT_ALIGNMENT'):
        _state_report_current_input(SimpleNamespace(current_state_input=True), {})
    with pytest.raises(RuntimeError, match='MATCHED_CURRENT_INPUT_MODE'):
        StateScoreAccumulator(_prior(), lead_minutes=(1, 2), model_receives_current_state=1)


def test_wrong_direction_is_not_counted_as_a_correct_change():
    first = np.array([.05, .9, .05])  # model predicts FLAT rather than DOWN
    second = np.array([.05, .05, .9])
    p = np.broadcast_to(np.stack((first, second))[None, None], (1, 5, 2, 3)).copy()
    state = np.broadcast_to(np.array([0, 2], np.int8)[None, None], (1, 5, 2)).copy()
    trans = np.broadcast_to(second, (1, 5, 1, 3, 3)).copy()
    score = StateScoreAccumulator(_prior(), lead_minutes=(1, 2), model_receives_current_state=True)
    score.update(p, state=state, eligible=np.ones_like(state, bool),
                 current_state=np.full((1, 5), 2, np.int8), current_observed=np.ones((1, 5), bool),
                 sequence_nll=np.zeros((1, 5)), transition_probabilities=trans)
    for group in score.finalize()['matched_markov_transition_comparison']['groups'].values():
        assert group['true_changes'] == group['model_predicted_changes'] == [1] * 5
        assert group['model_exact_change_hits'] == [0] * 5
        assert group['model_change_recall'] == group['model_change_precision'] == [0.] * 5


def test_unobserved_current_uses_same_prior_fallback_in_both_modes():
    explicit = _run(model_receives_current_state=True, missing=True)['matched_markov_state_baseline']
    no_input = _run(model_receives_current_state=False, missing=True)['matched_markov_state_baseline']
    for key in ('observed', 'correct', 'tied', 'logloss_infinite', 'logloss', 'brier'):
        assert explicit[key] == no_input[key]


@pytest.mark.parametrize('use_current', [False, True])
def test_matched_markov_matches_independent_loop_for_multiple_cases(use_current):
    rng = np.random.default_rng(19)
    batch, tfs, q = 4, 5, 4
    prior = _prior(tuple(range(1, q + 1)))
    tc = np.asarray(prior['transition_counts'], np.int64)
    tc[0, 2] = [5, 5, 0]  # a genuine tied baseline decision
    prior['transition_counts'] = tc.tolist()
    transition = tc / tc.sum(axis=-1, keepdims=True)
    prior['transition_probabilities'] = transition.tolist()
    initial = rng.uniform(.1, 1., (batch, tfs, 3)); initial /= initial.sum(axis=-1, keepdims=True)
    conditional = rng.uniform(.1, 1., (batch, tfs, q-1, 3, 3)); conditional /= conditional.sum(axis=-1, keepdims=True)
    probs = np.empty((batch, tfs, q, 3)); probs[:, :, 0] = initial
    for j in range(1, q): probs[:, :, j] = np.einsum('bfc,bfcd->bfd', probs[:, :, j-1], conditional[:, :, j-1])
    state = rng.integers(0, 3, (batch, tfs, q), dtype=np.int8)
    eligible = np.ones_like(state, bool); eligible[0, 1, 1] = False; state[~eligible] = -1
    current = rng.integers(0, 3, (batch, tfs), dtype=np.int8)
    current_observed = np.ones_like(current, bool); current_observed[1, 2] = False; current[~current_observed] = -1
    score = StateScoreAccumulator(prior, lead_minutes=tuple(range(1, q+1)), model_receives_current_state=use_current)
    score.update(probs, state=state, eligible=eligible, current_state=current, current_observed=current_observed,
                 sequence_nll=np.zeros((batch, tfs)), transition_probabilities=conditional)
    report = score.finalize()
    matrices = {group: {name:np.zeros((tfs, 3, 3), np.int64) for name in
                       ('truth', 'markov_forecasts', 'markov_exact_hits')}
                for group in ('current_to_first', 'future_adjacent')}
    ties = {group:np.zeros(tfs, np.int64) for group in matrices}
    observed = np.zeros((tfs, q), np.int64); correct = np.zeros_like(observed); tied = np.zeros_like(observed)
    logsum = np.zeros((tfs, q)); briersum = np.zeros_like(logsum)
    initial_prior = np.asarray(prior['probabilities'])
    for b in range(batch):
        for tf in range(tfs):
            distribution = (np.eye(3)[current[b, tf]] if use_current and current_observed[b, tf]
                            else initial_prior[tf].copy())
            for j in range(q):
                previous = distribution.copy()
                distribution = previous @ transition[tf]
                if eligible[b, tf, j]:
                    actual = int(state[b, tf, j]); observed[tf, j] += 1
                    selected = distribution[actual]
                    assert selected > 0
                    logsum[tf, j] -= np.log(selected)
                    briersum[tf, j] += np.square(distribution).sum() - 2*selected + 1
                    unique = np.count_nonzero(distribution == distribution.max()) == 1
                    tied[tf, j] += not unique
                    correct[tf, j] += unique and distribution.argmax() == actual
                if j == 0:
                    if not (current_observed[b, tf] and eligible[b, tf, j]): continue
                    group = 'current_to_first'; origin = int(current[b, tf]); destination = int(state[b, tf, j])
                    predicted = origin*3 + int(distribution.argmax())
                    unique = np.count_nonzero(distribution == distribution.max()) == 1
                else:
                    if not (eligible[b, tf, j-1] and eligible[b, tf, j]): continue
                    group = 'future_adjacent'; origin = int(state[b, tf, j-1]); destination = int(state[b, tf, j])
                    joint = (previous[:, None] * transition[tf]).reshape(-1)
                    predicted = int(joint.argmax()); unique = np.count_nonzero(joint == joint.max()) == 1
                matrices[group]['truth'][tf, origin, destination] += 1
                if unique:
                    matrices[group]['markov_forecasts'][tf, predicted//3, predicted%3] += 1
                    if predicted == origin*3 + destination:
                        matrices[group]['markov_exact_hits'][tf, origin, destination] += 1
                else:
                    ties[group][tf] += 1
    evidence = report['matched_markov_transition_comparison']['groups']
    for group in matrices:
        for key, array in matrices[group].items():
            assert evidence[group][key + '_by_from_to'] == array.tolist()
        assert evidence[group]['markov_tied_pairs'] == ties[group].tolist()
    baseline = report['matched_markov_state_baseline']
    assert baseline['observed'] == observed.tolist()
    assert baseline['correct'] == correct.tolist() and baseline['tied'] == tied.tolist()
    assert np.asarray(baseline['logloss']) == pytest.approx(logsum / observed)
    assert np.asarray(baseline['brier']) == pytest.approx(briersum / observed)


def test_up_down_flat_up_all_count_as_separate_transitions():
    probs = np.array([[.9,.05,.05], [.05,.9,.05], [.05,.05,.9]])
    p = np.broadcast_to(probs[None,None], (1,5,3,3)).copy()
    trans = np.broadcast_to(probs[None,None,1:,None,:], (1,5,2,3,3)).copy()
    state = np.broadcast_to(np.array([0,1,2],np.int8)[None,None], (1,5,3)).copy()
    score = StateScoreAccumulator(_prior((1,2,3)), lead_minutes=(1,2,3), model_receives_current_state=True)
    score.update(p, state=state, eligible=np.ones_like(state,bool),
        current_state=np.full((1,5),2,np.int8), current_observed=np.ones((1,5),bool),
        sequence_nll=np.zeros((1,5)), transition_probabilities=trans)
    groups = score.finalize()['matched_markov_transition_comparison']['groups']
    first = groups['current_to_first']; later = groups['future_adjacent']
    assert first['model_exact_hits_by_from_to'][0][2][0] == 1  # UP -> DOWN
    assert later['model_exact_hits_by_from_to'][0][0][1] == 1  # DOWN -> FLAT
    assert later['model_exact_hits_by_from_to'][0][1][2] == 1  # FLAT -> UP
    assert first['true_changes'] == [1]*5 and later['true_changes'] == [2]*5
