from __future__ import annotations

import numpy as np
import pytest

from gx1.audit import verify_single_threshold_market_state_candidate_v1 as worker
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target


def test_one_boundary_keeps_direction_proof_separate_from_activity() -> None:
    tape = {
        "threshold": np.asarray([[1], [1], [1], [1], [1], [1]], np.float64),
        "local_range": np.asarray([[0.2], [1.2], [1.2], [1.2], [0.8], [1.2]], np.float64),
        "reason": np.asarray([[0], [0], [0], [0], [target.LOCAL_SPAN_INCOMPLETE],
                              [target.LOCAL_SPAN_INCOMPLETE]], np.uint8),
        "state": np.asarray([[target.FLAT], [target.DOWN], [target.UP],
                             [target.UNKNOWN], [target.UNKNOWN], [target.UNKNOWN]], np.int8),
        "observed": np.asarray([[True], [True], [True], [False], [False], [False]]),
    }
    early = np.asarray([False, False, False, False, False, True])
    masks, complete, counts = worker.classify_one_factor(
        tape, early, np.arange(6), 0, target, np)
    assert masks.tolist() == [worker.FLAT, worker.DOWN, worker.UP,
                              worker.ACTIVE, worker.ALL, worker.ACTIVE]
    assert complete.tolist() == [True, True, True, True, False, False]
    assert counts == {"flat": 1, "down": 1, "up": 1,
                      "partial_complete": 1, "partial_early": 1}


def test_old_gray_only_may_become_flat() -> None:
    old = np.asarray([2, 1, 4, 5, 7, 7], np.uint8)
    new = np.asarray([2, 1, 4, 5, 2, 7], np.uint8)
    month = {"middle_unknown": 1, "missing_exact_endpoint": 1,
             "other_unknown": 0}
    assert worker._compare_old(old, new, month, np) == {
        "old_gray_to_flat": 1, "old_other_unknown_stays_unknown": 1,
        "old_nonunknown_unchanged": 4}
    wrong = new.copy()
    wrong[1] = worker.UP
    with pytest.raises(RuntimeError, match="OLD_NONUNKNOWN_CHANGED"):
        worker._compare_old(old, wrong, month, np)
    wrong = new.copy()
    wrong[5] = worker.ACTIVE
    with pytest.raises(RuntimeError, match="OLD_GRAY_PARITY"):
        worker._compare_old(old, wrong, month, np)


def test_calibration_accepts_zero_range_but_no_invalid_ratio() -> None:
    summary = worker.ratio_summary(np.asarray([0.0, 1.0, 2.0, 3.0]), np)
    assert summary == {"count": 4, "median": 1.5, "p10": 0.30000000000000004,
                       "p90": 2.7}
    spreads = worker.ratio_summary(np.asarray([0.0, 1.0, 2.0, 3.0]), np,
                                   spread_units=True)
    assert spreads["share_threshold_le_two_spreads"] == 0.75
    assert worker.ratio_summary(np.asarray([], np.float64), np)["count"] == 0
    with pytest.raises(RuntimeError, match="CALIBRATION_DATA"):
        worker.ratio_summary(np.asarray([float("nan")]), np)
