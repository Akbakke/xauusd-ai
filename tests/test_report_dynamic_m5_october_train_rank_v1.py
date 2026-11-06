import numpy as np
import pytest

from gx1.scripts.report_dynamic_m5_october_train_rank_v1 import auc_with_day_interval


def test_day_block_auc_keeps_exact_score_ties():
    y = np.array([0, 1, 0, 1, 0, 1], np.int8)
    score = np.full(6, 0.5, np.float64)
    days = np.array([10, 10, 11, 11, 12, 12], np.int64)
    result = auc_with_day_interval(y, score, days)
    assert result['row_auc'] == 0.5
    assert result['minus_half_utc_day_95_interval'] == [0.0, 0.0]
    assert result['bootstrap']['valid_draws'] == 10_000


def test_day_block_auc_orientation_is_up_positive():
    y = np.array([0, 1, 0, 1, 0, 1], np.int8)
    score = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7], np.float64)
    days = np.array([10, 10, 11, 11, 12, 12], np.int64)
    result = auc_with_day_interval(y, score, days)
    assert result['row_auc'] == 1.0
    assert result['minus_half_utc_day_95_interval'] == [0.5, 0.5]
    assert result['outcome_utc_days'] == 3


def test_day_block_auc_requires_both_directions():
    with pytest.raises(RuntimeError, match='AUC_DOMAIN'):
        auc_with_day_interval(np.array([1, 1], np.int8),
                              np.array([0.1, 0.2], np.float64),
                              np.array([10, 11], np.int64))
