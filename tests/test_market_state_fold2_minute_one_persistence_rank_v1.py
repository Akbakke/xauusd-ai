"""Causal sign reference includes unknown current state on declared rows."""
import numpy as np

from gx1.audit.measure_market_state_fold2_minute_one_persistence_rank_v1 import cell


def test_unknown_current_state_stays_in_rank_population():
    truth = {
        "allowed": np.array([[1, 7, 7, 7, 7],
                             [4, 7, 7, 7, 7],
                             [1, 7, 7, 7, 7],
                             [4, 7, 7, 7, 7]], np.uint8),
        "eligible": np.ones((4, 5), np.bool_),
        "mature": np.ones((4, 5), np.bool_),
    }
    now = np.array([[1, 7, 4, 7, 7], [4, 7, 1, 7, 7],
                    [5, 7, 5, 7, 7], [0, 7, 0, 7, 7]], np.uint8)
    got = cell(truth, now, 0)
    assert got["known_future_side_rows"] == 4
    assert got["current_certified_direction_rows"] == 2
    assert got["current_active_side_unknown_rows"] == 1
    assert got["current_missing_rows"] == 1
    assert got["same_tf_current_sign_auc"] == 0.875
    assert got["h1_current_sign_control_auc"] == 0.125
    one_class = cell(truth, now, 0, np.array([True, False, True, False]))
    assert one_class["known_future_side_rows"] == 2
    assert one_class["same_tf_current_sign_auc"] is None
