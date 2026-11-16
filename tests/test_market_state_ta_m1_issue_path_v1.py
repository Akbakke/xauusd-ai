"""The M1 path distinguishes order, retains gaps and reads no future row."""
import numpy as np
import pytest

from gx1.contracts.market_state_ta_m1_issue_path_v1 import (
    M1IssuePathTape, MINUTE_NS, M5_NS,
)

START = np.datetime64('2024-05-01T00:00:00', 'ns').astype(np.int64)


def _bars(closes):
    close = 2300.0 + np.asarray(closes, dtype=np.float64)
    opens = np.r_[2300.0, close[:-1]]
    return np.stack((opens, np.maximum(opens, close),
                     np.minimum(opens, close), close), axis=1)


def test_equal_m5_ohlc_with_opposite_m1_order_remains_distinct():
    up_first = _bars([-1.2, -0.8, 1.2, 0.4, 0.0])
    down_first = _bars([1.2, 0.8, -1.2, -0.4, 0.0])
    for row in (up_first, down_first):
        assert [row[0, 0], row[:, 1].max(), row[:, 2].min(), row[-1, 3]] == [2300., 2301.2, 2298.8, 2300.]
    axis = START + np.arange(5, dtype=np.int64) * MINUTE_NS
    issue = np.array([START + M5_NS], dtype=np.int64)
    first = M1IssuePathTape(axis, up_first).gather(issue)
    second = M1IssuePathTape(axis, down_first).gather(issue)
    assert first['full_path'].tolist() == second['full_path'].tolist() == [True]
    assert not np.array_equal(first['mid_ohlc_bps'], second['mid_ohlc_bps'])
    assert first['mid_ohlc_bps'][0, -1, -1] == second['mid_ohlc_bps'][0, -1, -1] == 0


def test_missing_minute_has_mask_and_zero_without_dropping_issue():
    axis = START + np.array([0, 1, 3, 4], dtype=np.int64) * MINUTE_NS
    got = M1IssuePathTape(axis, _bars([0.1, 0.2, 0.3, 0.4])).gather(
        np.array([START + M5_NS], dtype=np.int64))
    assert got['observed'].tolist() == [[True, True, False, True, True]]
    assert got['full_path'].tolist() == [False]
    assert got['last_bar_present'].tolist() == [True]
    assert got['source_index'].tolist() == [[0, 1, -1, 2, 3]]
    assert np.array_equal(got['mid_ohlc_bps'][0, 2], np.zeros(4, np.float32))


def test_future_change_cannot_change_current_path_and_bad_clock_fails():
    axis = START + np.arange(7, dtype=np.int64) * MINUTE_NS
    first = _bars([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    second = first.copy()
    second[5:] += 10.0
    issue = np.array([START + M5_NS], dtype=np.int64)
    a = M1IssuePathTape(axis, first).gather(issue)
    b = M1IssuePathTape(axis, second).gather(issue)
    assert np.array_equal(a['mid_ohlc_bps'], b['mid_ohlc_bps'])
    assert np.array_equal(a['observed'], b['observed'])
    with pytest.raises(ValueError, match='ISSUE_CLOCK'):
        M1IssuePathTape(axis, first).gather(np.array([START + M5_NS + MINUTE_NS], np.int64))
