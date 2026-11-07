"""Pure target masks and paired score checks for the tabular diagnostic."""
from copy import deepcopy

import numpy as np
import pytest

from gx1.contracts import market_state_five_endpoint_target_binding_v1 as binding
from gx1.contracts import market_state_tabular_target_score_v1 as adapter


def _bound():
    official, next_boundary = binding.FOLD_BOUNDARIES_NS[:2]
    effective = official - 1440 * binding.MINUTE_NS
    issues = np.array([binding.TRAIN_START_NS + binding.M5_NS,
                       effective - binding.M5_NS, effective,
                       official - binding.M5_NS, official + binding.M5_NS,
                       next_boundary - binding.M5_NS], dtype=np.int64)
    parents = np.arange(10, 70, 10, dtype=np.int64)
    allowed = np.array([[1, 2, 4, 5, 7], [4, 1, 2, 5, 7],
                        [7, 7, 7, 7, 7], [1, 1, 1, 1, 1],
                        [4, 2, 1, 5, 7], [1, 4, 2, 5, 7]], dtype=np.uint8)
    m1_source = np.full((len(issues), 5), 100, dtype=np.int64)
    m1_source[allowed == 7] = -1
    return binding.BoundFiveEndpointTargets(
        parents, issues, allowed, m1_source, np.arange(len(issues), dtype=np.int64), 0)


def test_binary_masks_keep_active_unknown_side_and_unknown_out():
    allowed = np.array([[1, 2, 4, 5, 7], [4, 1, 2, 5, 7]], dtype=np.uint8)
    eligible = np.array([[True] * 4 + [False], [True] * 4 + [False]], dtype=np.bool_)
    mature = np.array([[True] * 5, [True, False, True, True, True]], dtype=np.bool_)
    value = adapter.binary_fit_targets(allowed, eligible, mature)
    assert value['activity_mask'].tolist() == [[True] * 4 + [False],
                                               [True, False, True, True, False]]
    assert value['activity_label'].tolist() == [[1, 0, 1, 1, 0], [1, 0, 0, 1, 0]]
    assert value['certified_side_mask'].tolist() == [[True, False, True, False, False],
                                                     [True, False, False, False, False]]
    assert value['side_up_label'].tolist() == [[0, 0, 1, 0, 0], [1, 0, 0, 0, 0]]
    assert value['certified_side_counts']['D1'] == 0
    assert not value['activity_label'].flags.writeable
    assert not value['certified_side_mask'].flags.writeable


def test_mature_fold_prior_uses_only_short_fit_and_zero_side_support():
    bound = _bound()
    prepared = adapter.mature_fold_training_targets(bound, fold_id=1)
    assert set(prepared) == {'cohort', 'binary', 'prior'}
    assert prepared['cohort']['joined_fit_rows'] == 2
    assert prepared['cohort']['score_parent_rows'].tolist() == [50, 60]
    assert prepared['prior']['fit_rows'] == 2
    assert prepared['prior']['counts']['D1']['certified_side'] == 0
    assert prepared['prior']['probabilities'][4] == [0.25, 0.5, 0.25]
    assert prepared['binary']['certified_side_counts']['H4'] == 0
    assert 'allowed' not in prepared and 'score_truth' not in prepared
    assert 'allowed' not in prepared['cohort']
    with pytest.raises(ValueError, match='OOF_FOLD'):
        adapter.mature_fold_training_targets(bound, fold_id=6)


def test_binary_probability_composition_and_invalid_zero_active():
    active = np.full((2, 5), 0.8, dtype=np.float64)
    up = np.full((2, 5), 0.25, dtype=np.float32)
    combined = adapter.combine_binary_probabilities(active, up)
    assert combined.shape == (2, 5, 3)
    assert np.allclose(combined[0, 0], [0.6, 0.2, 0.2])
    assert np.allclose(combined.sum(axis=-1), 1.0)
    active[0, 0] = 0.0
    with pytest.raises(ValueError, match='BINARY_PROBABILITIES'):
        adapter.combine_binary_probabilities(active, up)


def test_paired_score_uses_original_score_axes_and_same_maturity_masks():
    bound = _bound()
    prepared = adapter.mature_fold_training_targets(bound, fold_id=1)
    cohort, record = prepared['cohort'], prepared['prior']
    probabilities = np.broadcast_to(np.asarray(record['probabilities'], dtype=np.float64),
                                    (len(cohort['score_parent_rows']), 5, 3)).copy()
    scored = adapter.paired_mature_fold_scores(
        bound, fold_id=1, prediction_parent_rows=cohort['score_parent_rows'],
        prediction_decision_ns=cohort['score_decision_ns'],
        probabilities=probabilities, prior_record=record)
    assert scored['score_rows'] == 2
    assert scored['score_parent_rows_sha256'] == cohort['score_parent_rows_sha256']
    assert scored['model'] == scored['fit_prior']
    for tf in ('M5', 'M15', 'H1', 'H4'):
        assert scored['model']['per_tf'][tf]['activity']['count'] == 1
    assert scored['model']['per_tf']['D1']['activity']['count'] == 0
    wrong = deepcopy(record)
    wrong['probabilities'][0][0] += 0.01
    with pytest.raises(ValueError, match='FIT_PRIOR_IDENTITY'):
        adapter.paired_mature_fold_scores(
            bound, fold_id=1, prediction_parent_rows=cohort['score_parent_rows'],
            prediction_decision_ns=cohort['score_decision_ns'],
            probabilities=probabilities, prior_record=wrong)
    with pytest.raises(ValueError, match='SCORE_AXES'):
        adapter.paired_mature_fold_scores(
            bound, fold_id=1, prediction_parent_rows=cohort['score_parent_rows'][::-1].copy(),
            prediction_decision_ns=cohort['score_decision_ns'],
            probabilities=probabilities, prior_record=record)
