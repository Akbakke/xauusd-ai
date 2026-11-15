"""Fixed S path clocks, masks, interval changes and causal reference."""
import numpy as np
import pytest

from gx1.contracts.market_state_fold2_multiminute_path_v1 import (
    QUERY_MINUTES, fixed_query_grid, gather_fixed_path,
    observed_transition_intervals, same_tf_persistence_reference,
    score_fixed_path,
)
from gx1.contracts.market_state_minute_query_candidate_v1 import (
    MinuteQueryTargetTape, NATIVE_MINUTES,
)
from gx1.contracts.market_state_time_query_target_v1 import MINUTE_NS


def _synthetic_path():
    axis_minute = np.unique(QUERY_MINUTES)
    bits = np.full((len(axis_minute), 5), 7, np.uint8)
    for minute, value in zip((1, 2, 3, 4, 5), (4, 4, 1, 4, 4)):
        bits[np.searchsorted(axis_minute, minute), 0] = value
    for minute, value in zip((1, 4, 8, 12, 15), (4, 7, 1, 1, 4)):
        bits[np.searchsorted(axis_minute, minute), 1] = value
    for j in (2, 3, 4):
        bits[np.searchsorted(axis_minute, QUERY_MINUTES[:, j]), j] = 2
    tape = MinuteQueryTargetTape(
        axis_minute.astype(np.int64) * MINUTE_NS,
        np.arange(len(axis_minute), dtype=np.int64), bits,
        np.ones_like(bits, dtype=np.bool_),
        np.zeros(len(axis_minute), np.uint8))
    return gather_fixed_path(tape, np.array([0], np.int64),
                             fold_upper_ns=1000 * MINUTE_NS)


def test_fixed_landmarks_and_exact_maturity_masks():
    assert QUERY_MINUTES.shape == (5, 5)
    np.testing.assert_array_equal(QUERY_MINUTES[0], 1)
    np.testing.assert_array_equal(QUERY_MINUTES[-1], NATIVE_MINUTES)
    assert (np.diff(QUERY_MINUTES, axis=0) > 0).all()
    issue = np.array([0, 5 * MINUTE_NS], np.int64)
    query = fixed_query_grid(issue)
    np.testing.assert_array_equal(query[0], query[1])
    query[0, 0, 0] = 5
    assert fixed_query_grid(issue)[0, 0, 0] == 1
    with pytest.raises(ValueError, match='PATH_ISSUES'):
        fixed_query_grid(np.array([MINUTE_NS], np.int64))
    path = _synthetic_path()
    assert path['allowed'][0, 0, 0] == 4
    assert path['mature'][0, -1, -1] == 0
    assert path['allowed'][0, -1, -1] == 7
    assert path['eligible'][0, -1, -1] == 0
    assert path['eligible'][0, 1, 1] == 0  # certified unknown stays unknown
    np.testing.assert_array_equal(path['validity_ns'],
                                  path['issue_ns'][:, None, None]
                                  + path['query_minutes'] * MINUTE_NS)


def test_interval_changes_are_lower_bounds_and_unknowns_remain_visible():
    intervals = observed_transition_intervals(_synthetic_path())['intervals_by_tf']
    m5 = intervals['M5']
    assert [(x['from_minute'], x['through_minute']) for x in m5] == [
        (1, 2), (2, 3), (3, 4), (4, 5)]
    assert [x['proven_at_least_one_change'] for x in m5] == [0, 1, 1, 0]
    assert [x['equal_endpoints_path_unresolved'] for x in m5] == [1, 0, 0, 1]
    assert intervals['M15'][0]['unresolved_pairs'] == 1
    assert intervals['M15'][1]['unresolved_pairs'] == 1
    assert intervals['D1'][-1]['unresolved_pairs'] == 1


def test_marginal_scores_and_same_tf_reference_use_identical_times():
    path = _synthetic_path()
    raw = np.zeros(path['allowed'].shape + (2,), np.float32)
    score = score_fixed_path(raw, path)
    assert len(score['per_checkpoint']) == 5
    assert score['joint_path_probability_claimed'] is False
    m5 = score['transition_prediction_diagnostic']['per_tf']['M5']
    assert [x['missed_change'] for x in m5] == [0, 1, 1, 0]
    assert [x['false_change'] for x in m5] == [0, 0, 0, 0]
    assert score['transition_observations']['intervals_by_tf']['M5'][1][
        'proven_at_least_one_change'] == 1
    duplicated = {key: np.repeat(value, 2, axis=0) if isinstance(value, np.ndarray)
                  else value for key, value in path.items()}
    duplicated['allowed'][1, 0, 0] = 1
    current = np.full((2, 5), 0, np.uint8)
    current[:, 0] = [4, 1]
    ref = same_tf_persistence_reference(duplicated, current)
    first = ref['per_tf']['M5'][0]
    assert first['known_future_side_rows'] == 2
    assert first['same_tf_current_sign_auc'] == 1.0
    assert first['constant_rank_auc'] == 0.5
    assert ref['per_tf']['D1'][-1]['mature_rows'] == 0
    current[:, 0] = 0
    assert same_tf_persistence_reference(duplicated, current)[
        'per_tf']['M5'][0]['same_tf_current_sign_auc'] == 0.5


def test_identical_five_checkpoint_support_does_not_mix_cohorts():
    from gx1.audit.measure_market_state_fold2_multiminute_path_v1 import (
        _same_rows_known_side,
    )
    path = _synthetic_path()
    now = np.zeros((1, 5), np.uint8)
    now[0, 0] = 4
    got = _same_rows_known_side(path, now)
    assert got['M5']['all_five_known_side_rows'] == 1
    assert got['M5']['first_last_side_flip_rows'] == 0
    assert got['M5']['at_least_one_sampled_adjacent_side_flip_rows'] == 1
    assert got['M5']['same_tf_current_sign_auc_by_checkpoint'] == [None] * 5
    assert got['M15']['all_five_known_side_rows'] == 0
