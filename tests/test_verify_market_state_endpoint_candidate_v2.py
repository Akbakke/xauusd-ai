"""Focused synthetic invariants for the proposed endpoint mask verifier."""
import numpy as np
import pytest

from gx1.audit.verify_market_state_endpoint_candidate_v2 import (
    ACTIVE, ALL, DOWN, FLAT, classify_rows, exact_positions, verify_partition,
)
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target


def test_boundaries_and_post_gap_side_stays_partial():
    span = np.array([0.49, 0.5, 0.99, 1.0, 1.2, 0.1, 1.2])[:, None]
    reasons = np.array([0, 0, 0, 0, 0, target.LOCAL_SPAN_INCOMPLETE,
                        target.UNKNOWN_SOURCE_GAP])[:, None]
    a05 = {"threshold": np.full((7, 1), 0.5), "local_range": span,
           "reason": reasons}
    a10 = {"threshold": np.full((7, 1), 1.0), "local_range": span.copy(),
           "reason": reasons, "state": np.array([target.FLAT, target.FLAT,
               target.FLAT, target.DOWN, target.UNKNOWN, target.FLAT,
               target.UP])[:, None],
           "observed": np.array([True, True, True, True, False, True,
                                 True])[:, None]}
    early = np.array([False, False, False, False, False, False, True])
    allowed, counts = classify_rows(a05, a10, early, np.arange(7), 0, target, np)
    assert allowed.tolist() == [FLAT, ALL, ALL, DOWN, ACTIVE, ALL, ACTIVE]
    assert counts == {"flat": 1, "down": 1, "up": 0,
                      "partial_complete": 1, "partial_early": 1,
                      "middle_unknown": 2}


def test_exact_utc_endpoint_and_strict_maturity():
    minute = 60_000_000_000
    axis = np.array([5, 10, 15]) * minute
    pos, present = exact_positions(axis, np.array([0, 1, 10]) * minute,
                                   5, 15 * minute, np)
    assert pos.tolist() == [0, -1, -1]
    assert present.tolist() == [True, False, False]


def test_saved_month_tf_divergence_fails():
    counts = {"flat": 1, "down": 1, "up": 0, "partial_complete": 1,
              "partial_early": 1, "middle_unknown": 2}
    expected = {"flat": 1, "down_up": 1, "partial_complete": 1,
                "partial_early": 1, "middle_unknown": 2}
    verify_partition(counts, expected)
    with pytest.raises(RuntimeError, match="SAVED_25_PARTITION_DIVERGENCE"):
        verify_partition(counts, {**expected, "partial_early": 2})
