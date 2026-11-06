import numpy as np
import pytest

from gx1.scripts.report_dynamic_m5_october_target_alignment_v1 import (
    day_bootstrap, joint_counts,
)


def test_joint_counts_keep_unknown_state_and_entry_margins_separate():
    state = np.array([0, 1, 2, -1, 2, 0, -1, 1], np.int8)
    observed = np.array([True, True, True, False, True, True, False, True])
    side = np.array([1, 0, 1, 0, -1, -1, -1, 1], np.int8)
    out = joint_counts(state, observed, side)
    assert out['table'] == {
        'DOWN': {'SHORT': 0, 'LONG': 1},
        'FLAT': {'SHORT': 1, 'LONG': 1},
        'UP': {'SHORT': 0, 'LONG': 1},
    }
    assert out['common_observed'] == 4
    assert out['state_unknown_entry_known'] == 1
    assert out['entry_unknown_state_known'] == 2
    assert out['both_unknown'] == 1
    assert out['entry_known'] == 5
    assert out['state_known'] == 6


def test_utc_day_bootstrap_has_fixed_direction_and_no_row_independence_claim():
    state = np.array([2, 0, 2, 0, 2, 0], np.int8)
    side = np.array([1, 0, 1, 0, 1, 0], np.int8)
    days = np.array([10, 10, 11, 11, 12, 12], np.int64)
    out = day_bootstrap(state, side, days)
    assert out['long_better_rate_by_state'] == {'DOWN': 0.0, 'UP': 1.0}
    assert out['up_minus_down_long_better_rate'] == 1.0
    assert out['up_minus_down_utc_day_95_interval'] == [1.0, 1.0]
    assert out['outcome_utc_days'] == 3
    assert out['bootstrap']['valid_draws'] == 10_000
    assert out['bootstrap']['invalid_draws'] == 0


def test_utc_day_bootstrap_counts_draws_missing_a_class():
    state = np.array([2, 2, 0, 0], np.int8)
    side = np.array([1, 1, 0, 0], np.int8)
    days = np.array([10, 10, 11, 11], np.int64)
    out = day_bootstrap(state, side, days)
    assert 0 < out['bootstrap']['invalid_draws'] < 10_000
    assert out['bootstrap']['valid_draws'] + out['bootstrap']['invalid_draws'] == 10_000
    assert out['up_minus_down_utc_day_95_interval'] == [1.0, 1.0]


def test_joint_counts_rejects_unmasked_unknown_state():
    with pytest.raises(RuntimeError, match='JOINT_DOMAIN'):
        joint_counts(np.array([-1], np.int8), np.array([True]), np.array([1], np.int8))
