import numpy as np

from gx1.audit.score_market_state_ta_integrated_turn_v1 import native_pairs, _phase, MINUTE_NS


def test_native_turn_uses_previous_mature_endpoint_and_current_forecast():
    minutes = np.array([0, 5, 10, 20, 25], dtype=np.int64)
    issues = (1735689600000000000 + minutes * MINUTE_NS).astype(np.int64)
    allowed = np.full((5, 5), 7, dtype=np.uint8)
    allowed[:, 0] = [1, 4, 1, 4, 1]
    eligible = np.ones((5, 5), dtype=bool)
    mature = np.ones((5, 5), dtype=bool)
    mature[2, 0] = False
    raw = np.zeros((5, 5, 2), dtype=np.float32)
    raw[1, 0, 1] = 4.0
    raw[4, 0, 1] = -4.0
    data = {'issue_ns': issues, 'state_allowed': allowed,
            'state_eligible': eligible, 'state_mature': mature,
            'state_raw': raw}
    current, previous, turns = native_pairs(data, 0, 5)
    assert current.tolist() == [1, 4]
    assert previous.tolist() == [0, 3]
    assert turns.tolist() == [True, True]
    score, probability = _phase(data, current, previous, turns, 0)
    assert score['known_pair_count'] == score['actual_switch_count'] == 2
    assert score['predicted_switch_count'] == 2
    assert score['true_positive_switch_count'] == 2
    assert (probability > .9).all()
