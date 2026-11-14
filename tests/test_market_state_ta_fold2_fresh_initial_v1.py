import numpy as np

from gx1.audit.observe_market_state_ta_fold2_fresh_initial_v1 import (
    auc_with_ties, direction_readout,
)


def test_auc_uses_midrank_ties_and_requires_both_sides():
    score = np.array([-1., 0., 0., 2.])
    labels = np.array([False, True, False, True])
    assert auc_with_ties(score, labels, np) == 0.875
    assert auc_with_ties(score, np.zeros(4, dtype=bool), np) is None


def test_direction_readout_keeps_known_mature_sides_and_months():
    raw = np.zeros((4, 5, 2), dtype=np.float32)
    raw[:, :, 1] = np.array([-2., 2., 5., -5.])[:, None]
    allowed = np.tile(np.array([1, 4, 2, 4], dtype=np.uint8)[:, None], (1, 5))
    truth = dict(allowed=allowed, eligible=np.ones((4, 5), dtype=bool),
                 mature=np.ones((4, 5), dtype=bool))
    truth['mature'][3, 0] = False
    issues = np.array(['2025-01-01', '2025-01-02', '2025-02-01', '2025-02-02'],
                      dtype='datetime64[ns]').astype(np.int64)
    out = direction_readout(raw, truth, issues, np)
    m5 = out['all']['M5']
    assert m5['known_side_n'] == 2
    assert m5['auc'] == 1.
    assert m5['accuracy_at_zero'] == 1.
    assert m5['always_up_accuracy'] == .5
    assert sum(b['count'] for b in m5['calibration']) == 2
    assert out['2025-01']['M5']['known_side_n'] == 2
    assert out['2025-02']['M5']['known_side_n'] == 0
    assert out['all']['M15']['known_side_n'] == 3
