import numpy as np

from gx1.audit.measure_market_state_fold2_h1_sign_rank_v1 import cell, rank_auc


def test_auc_orientation_and_ties():
    assert rank_auc(np.array([0, 1], dtype=np.uint8), np.array([-1, 1])) == 1.0
    assert rank_auc(np.array([1, 0], dtype=np.uint8), np.array([-1, 1])) == 0.0
    assert rank_auc(np.array([0, 1], dtype=np.uint8), np.array([0, 0])) == 0.5
    assert rank_auc(np.array([1, 1], dtype=np.uint8), np.array([1, 0])) is None


def test_unknown_current_is_counted_as_zero_on_same_future_rows():
    now = np.array([[0, 0, 4, 0, 0], [1, 0, 0, 0, 0], [4, 0, 1, 0, 0]], dtype=np.uint8)
    truth = {'allowed': np.array([[4], [1], [1]], dtype=np.uint8),
             'eligible': np.ones((3, 1), dtype=bool),
             'mature': np.ones((3, 1), dtype=bool)}
    out = cell(truth, now, 0)
    assert out['known_future_side_rows'] == 3
    assert out['h1_current_direction_known_rows'] == 2
    assert out['m5_current_direction_known_rows'] == 2
    assert out['h1_sign_auc'] == 1.0
    assert out['m5_sign_control_auc'] == 0.5
