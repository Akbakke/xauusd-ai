import numpy as np
import pytest

from gx1.audit.measure_ta_m5_structure_candidate_v1 import (
    DOWN, TIE, UNKNOWN, M5_NS, measure_candidate,
)


CLOSE = np.array([
    0, 1, 2, 3, 2, 1, 0, -1, 0, 1, 2, 1, 0, -1, -2, -1,
    0, 1, 2, 1, 0, -1, 0, 1, 2, 3, 2, 1, 0, -1, 0, 1,
    2, 1, 0, -1, 0, 1, 2, 1, 0,
], dtype=np.float64)
START_NS = int(np.datetime64("2024-05-01T00:00:00", "ns").astype(np.int64))


def _case(close=CLOSE, time=None, issues=(11, 13)):
    if time is None:
        time = START_NS + np.arange(len(close), dtype=np.int64) * M5_NS
    return measure_candidate(time, close + 0.1, close - 0.1, close,
                             np.ones(len(close)), np.asarray(issues, dtype=np.int64),
                             fold_target_end_ns=int(time[-1] + M5_NS))


def test_frozen_levels_and_next_confirmed_pair_are_separate():
    result = _case()
    assert result["frozen_high_idx"].tolist() == [3, 10]
    assert result["frozen_low_idx"].tolist() == [7, 7]
    assert result["next_high_idx"].tolist() == [18, 18]
    assert result["next_low_idx"].tolist() == [14, 14]
    assert result["next_high_confirm_row"].tolist() == [21, 21]
    assert result["next_low_confirm_row"].tolist() == [17, 17]
    assert result["pair_class"].tolist() == [DOWN, TIE]
    assert result["both_levels_hit"].tolist() == [False, True]
    assert result["high_hit"].tolist() == [False, True]
    assert result["low_hit"].tolist() == [True, True]


def test_unconfirmed_future_and_source_gap_are_censored():
    full = _case(issues=(11,))
    prefix = _case(close=CLOSE[:21], issues=(11,))
    assert prefix["frozen_high_idx"][0] == full["frozen_high_idx"][0]
    assert prefix["frozen_low_idx"][0] == full["frozen_low_idx"][0]
    assert not prefix["high_observed"][0]
    assert prefix["low_observed"][0]
    assert prefix["pair_class"][0] == UNKNOWN
    time = START_NS + np.arange(len(CLOSE), dtype=np.int64) * M5_NS
    time[16:] += M5_NS
    gap = _case(time=time, issues=(11,))
    assert not gap["high_observed"][0] and not gap["low_observed"][0]
    assert gap["pair_class"][0] == UNKNOWN


def test_gap_between_frozen_confirmation_and_issue_masks_that_level():
    time = START_NS + np.arange(len(CLOSE), dtype=np.int64) * M5_NS
    time[11:] += M5_NS
    result = _case(time=time, issues=(11,))
    assert result["frozen_low_idx"][0] == 7
    assert not result["low_observed"][0]
    assert result["pair_class"][0] == UNKNOWN


def test_bar_at_exclusive_fold_end_is_rejected():
    time = START_NS + np.arange(len(CLOSE), dtype=np.int64) * M5_NS
    with pytest.raises(ValueError, match="TA_M5_STRUCTURE_CLOCK"):
        measure_candidate(time, CLOSE + 0.1, CLOSE - 0.1, CLOSE,
                          np.ones(len(CLOSE)), np.array([11], dtype=np.int64),
                          fold_target_end_ns=int(time[-1]))


def test_test_period_is_rejected_before_source_replay():
    time = START_NS + np.arange(len(CLOSE), dtype=np.int64) * M5_NS
    time[-1] = int(np.datetime64("2026-07-01T00:00:00", "ns").astype(np.int64))
    with pytest.raises(ValueError, match="TA_M5_STRUCTURE_CLOCK"):
        _case(time=time, issues=(11,))


def test_same_bar_high_and_low_keep_hits_but_no_direction_order():
    high, low = CLOSE + 0.1, CLOSE - 0.1
    high[18], low[18] = 5.0, -5.0
    time = START_NS + np.arange(len(CLOSE), dtype=np.int64) * M5_NS
    result = measure_candidate(time, high, low, CLOSE, np.ones(len(CLOSE)),
                               np.array([17], dtype=np.int64),
                               fold_target_end_ns=int(time[-1] + M5_NS))
    assert result["same_pivot_bar"][0]
    assert result["both_levels_hit"][0]
    assert result["pair_class"][0] == UNKNOWN


def test_no_level_hit_is_observed_without_becoming_flat():
    close = CLOSE.copy()
    close[7] = -3.0
    result = _case(close=close, issues=(11,))
    assert result["high_observed"][0] and result["low_observed"][0]
    assert result["no_frozen_level_hit"][0]
    assert not result["high_hit"][0] and not result["low_hit"][0]
    assert result["pair_class"][0] != UNKNOWN
