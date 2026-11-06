import numpy as np

from gx1.scripts.report_dynamic_m5_october_entry_direction_join_v1 import direction_score, paired_auc


def test_issued_direction_score_is_conditional_nonflat_rank_only():
    p = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]], dtype=np.float64)
    assert np.allclose(direction_score(p), [1 / 9, 8 / 9])


def test_identical_saved_ranks_have_zero_paired_day_difference():
    y = np.array([0, 1, 0, 1], dtype=np.int8)
    days = np.array([1, 1, 2, 2], dtype=np.int64)
    rank = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float64)
    report = paired_auc(y, days, {'candidate': rank, 'control': rank})
    assert report['auc']['candidate']['row'] == 1.0
    assert report['auc']['candidate']['equal_day'] == 1.0
    assert report['candidate_minus_control_and_half_utc_day_intervals']['candidate_minus_control']['row'] == [0.0, 0.0]
