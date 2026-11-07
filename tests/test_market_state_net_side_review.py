import numpy as np
import pytest

from gx1.audit.review_market_state_net_side_v1 import summarize


def test_separates_net_sign_from_certified_state_side():
    sign = np.array([-1, 1, 0, -1, 1, 1, 0], dtype=np.int8)
    bits = np.array([1, 1, 4, 4, 5, 2, 7], dtype=np.uint8)
    crossings = np.array([0, 1, 1, 2, 0, 1, 0], dtype=np.int32)
    out = summarize(sign, bits, crossings)
    assert out == {
        'n': 7, 'net_down_zero_up': [2, 2, 3],
        'certified_side': 4, 'certified_same_net': 1,
        'certified_opposite_net': 2, 'certified_zero_net': 1,
        'start_price_close_crossing_rows': 4,
        'certified_opposite_with_start_crossing': 2,
        'active_side_unknown': 1, 'flat': 1, 'all_unknown': 1,
    }


def test_rejects_invalid_sign_and_mask_domain():
    with pytest.raises(RuntimeError, match='SUMMARY_DOMAIN'):
        summarize(np.array([2], dtype=np.int8), np.array([5], dtype=np.uint8),
                  np.array([0], dtype=np.int32))
    with pytest.raises(RuntimeError, match='SUMMARY_DOMAIN'):
        summarize(np.array([1], dtype=np.int8), np.array([3], dtype=np.uint8),
                  np.array([0], dtype=np.int32))
