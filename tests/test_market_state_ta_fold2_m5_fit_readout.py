"""Independent paired-readout controls; no model or fit execution."""
import numpy as np
import pytest

from gx1.audit.score_market_state_ta_fold2_m5_fit_v1 import (
    paired_later_f_m5, score_capture, weighted_auc,
)


def _capture():
    counts = (1365, 1365, 1366)
    months = ('2025-01-01', '2025-02-01', '2025-03-01')
    issues = np.concatenate([
        np.datetime64(month, 'ns').astype(np.int64)
        + np.arange(count, dtype=np.int64) * 60 * 10**9
        for month, count in zip(months, counts)
    ])
    parents = np.arange(4096, dtype=np.int64)
    allowed = np.where(parents[:, None] % 2 == 0, 1, 4) * np.ones((1, 5), dtype=np.int64)
    raw = np.zeros((4096, 5, 2), dtype=np.float32)
    raw[:, :, 1] = np.linspace(-1, 1, 4096, dtype=np.float32)[:, None]
    return {'parents': parents, 'issues': issues,
            **{task + '_' + key: value.copy()
               for task in ('state', 'future')
               for key, value in (('raw', raw), ('allowed', allowed),
                                  ('eligible', np.ones((4096, 5), dtype=bool)),
                                  ('mature', np.ones((4096, 5), dtype=bool)))}}


def test_weighted_auc_gives_half_credit_for_tied_scores():
    scores = np.array([-1., 0., 0., 1.])
    up = np.array([False, False, True, True])
    weights = np.ones((2, 4), dtype=np.float64)
    assert weighted_auc(scores, up, weights).tolist() == [.875, .875]


def test_identical_later_capture_has_zero_paired_gain_and_log_loss():
    capture = _capture()
    score = score_capture(capture)
    assert score['future']['all']['M5']['known_side_n'] == 4096
    assert 0 < score['future']['all']['M5']['log_loss'] < 1
    paired = paired_later_f_m5(capture, capture)
    assert paired['development_ranking_signal_qualified'] is False
    assert all(paired['schemes'][scheme]['months'][month]
               ['final_minus_initial_auc']['lower95'] == 0
               for scheme in ('UTC_DAY', 'UTC_MONDAY_WEEK')
               for month in ('all', '2025-01', '2025-02', '2025-03'))
    altered = _capture()
    altered['future_allowed'][0, 0] = 4
    with pytest.raises(ValueError, match='FOLD2_M5_READOUT_PAIRED_AXES_TARGETS'):
        paired_later_f_m5(capture, altered)
