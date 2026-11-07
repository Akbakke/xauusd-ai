import numpy as np

from gx1.audit.measure_market_state_decision_return_gc_v1 import exact_future_close_move


def test_exact_future_close_is_decision_anchored_and_masks_missing_endpoints():
    tick = 300_000_000_000
    issues = np.array([0, 1, 2, 4], np.int64) * tick
    close = np.array([100., 101., 102., 104.], np.float64)
    present, move, gap = exact_future_close_move(issues, close, 5)
    assert present.tolist() == [True, True, False, False]
    assert move.tolist() == [1., 1., 0., 0.]
    assert not gap.any()
    present, move, gap = exact_future_close_move(issues, close, 10)
    assert present.tolist() == [True, False, True, False]
    assert move.tolist() == [2., 0., 2., 0.]
    assert gap.tolist() == [False, False, True, False]


def test_source_row_mapping_excludes_warmup_and_terminal_bar():
    from gx1.audit import build_full_train_market_state_endpoint_candidate_v1 as base

    tick = base.M5_NS
    bars = np.array([0, 1, 2, 3], np.int64) * tick
    keep, issues, rows = base.bound_m5_issue_clock(bars, tick, 4 * tick, np)
    bounded_close = np.array([90., 100., 101., 200.], np.float64)[keep]
    assert rows.tolist() == [1, 2]
    assert issues.tolist() == [2 * tick, 3 * tick]
    present, move, gap = exact_future_close_move(issues, bounded_close[rows], 5)
    assert present.tolist() == [True, False]
    assert move.tolist() == [1., 0.]
    assert not gap.any()
