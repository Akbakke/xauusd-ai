import numpy as np

from gx1.scripts.report_dynamic_m5_october_age_pair_v1 import _auc, _auc_day_comparison, _day_interval, _losses


def test_paired_day_interval_keeps_whole_days_and_exact_constant_difference():
    day = np.array([1, 1, 2, 3, 3, 3], dtype=np.int64)
    delta = np.full(6, -0.125, dtype=np.float64)
    result = _day_interval(delta, day)
    assert result['days'] == 3
    assert np.allclose(result['row'], [-0.125, -0.125])
    assert np.allclose(result['equal_day'], [-0.125, -0.125])
    assert _day_interval(delta[:2], day[:2])['row'] is None


def test_probability_loss_never_clips_zero_and_direction_uses_nonflat_only():
    p = np.array([[0.8, 0.1, 0.1], [0.0, 0.5, 0.5], [0.1, 0.1, 0.8]], dtype=np.float64)
    truth = np.array([0, 0, 2], dtype=np.int64)
    loss = _losses(p, truth)
    assert np.isinf(loss['logloss'][1])
    assert np.isclose(loss['brier'][0], 0.06)
    assert _auc(p, truth, np.array([True, False, True])) == 1.0


def test_direction_day_bootstrap_matches_identical_arms():
    truth = np.array([0, 2, 0, 2], dtype=np.int64)
    day = np.array([1, 1, 2, 2], dtype=np.int64)
    p = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8],
                  [0.7, 0.1, 0.2], [0.2, 0.1, 0.7]], dtype=np.float64)
    result = _auc_day_comparison({name: p for name in ('candidate', 'control', 'prior', 'markov')},
                                 truth, day)
    assert result['nonflat_days'] == 2
    assert result['equal_day_auc']['candidate'] == 1.0
    assert result['candidate_minus_reference_intervals']['control']['row'] == [0.0, 0.0]
    assert result['candidate_minus_reference_intervals']['prior']['equal_day'] == [0.0, 0.0]
