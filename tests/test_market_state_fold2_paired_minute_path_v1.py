"""The paired minute-path reference uses the same certified rows."""
import numpy as np

from gx1.audit.measure_market_state_fold2_paired_minute_path_v1 import paired_cell


def test_paired_auc_and_change_count_keep_unknown_population():
    early_bits = np.array([1, 4, 1, 2, 5], np.uint8)
    end_bits = np.array([4, 1, 1, 4, 4], np.uint8)
    early = {
        "allowed": np.tile(early_bits[:, None], (1, 5)),
        "eligible": np.ones((5, 5), np.bool_),
        "mature": np.ones((5, 5), np.bool_)}
    early["mature"][4, 0] = False
    endpoint = {
        "allowed": np.tile(end_bits[:, None], (1, 5)),
        "eligible": np.ones((5, 5), np.bool_),
        "mature": np.ones((5, 5), np.bool_)}
    now = np.zeros((5, 5), np.uint8)
    now[:, 0] = [1, 4, 1, 2, 5]
    got = paired_cell(early, endpoint, now, 0)
    assert got["rows"] == 5 and got["early_eligible"] == 4
    assert got["early_allowed_counts"]["5"] == 1
    assert got["paired_singleton_rows"] == 4
    assert got["paired_certified_state_change_rows"] == 3
    assert got["paired_known_side_rows"] == 3
    assert got["paired_future_side_flip_rows"] == 2
    assert got["minute_one_same_tf_sign_auc"] == 1.0
    assert got["endpoint_same_tf_sign_auc"] == 0.25
