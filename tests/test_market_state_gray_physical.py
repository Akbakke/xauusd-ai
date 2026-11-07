"""Focused source-only geometry and masking checks for the gray-zone audit."""
import numpy as np
import pytest

from gx1.audit import verify_market_state_gray_physical_v1 as audit

M = audit.base.MINUTE_NS
OFFSET = audit.SOURCE_ROW_OFFSET


def test_known_closure_preserves_window_but_unknown_gap_rejects_it():
    validity = np.array([M, 2*M, 10*M, 11*M], np.int64)
    known = audit.unknown_gap_mask(validity, {2})
    unknown = audit.unknown_gap_mask(validity, set())
    assert not known.any()
    assert unknown.tolist() == [False, False, True, False]
    issue = np.array([validity[2] - 5*M], np.int64)
    rows = np.array([2], np.int64)
    source = rows + OFFSET
    assert audit.require_endpoint_and_window(issue, rows, source, validity, 5, 2, known).tolist() == [1]
    with pytest.raises(RuntimeError, match="GRAY_PHYSICAL_UNKNOWN_GAP_WINDOW"):
        audit.require_endpoint_and_window(issue, rows, source, validity, 5, 2, unknown)
    # A full post-gap window may begin at the first new observed bar.
    assert audit.require_endpoint_and_window(
        np.array([validity[3] - 5*M]), np.array([3]),
        np.array([OFFSET + 3]), validity, 5, 2, unknown).tolist() == [2]


def test_reopening_jump_in_range_and_zero_range_are_not_fabricated_net():
    opened = np.array([10., 11., 15.])
    high = np.array([10., 12., 16.])
    low = np.array([10., 11., 15.])
    close = np.array([10., 12., 16.])
    known = np.zeros(3, bool)
    unknown = np.array([False, False, True])
    assert audit.raw_window_range(opened, high, low, close, known, 1)[2] == 4.
    assert audit.raw_window_range(opened, high, low, close, unknown, 1)[2] == 1.
    constant = np.array([10., 10.])
    assert audit.raw_window_range(constant, constant, constant, constant,
                                  np.zeros(2, bool), 2)[1] == 0.
    crossings, run = audit.close_diagnostics(
        constant, np.array([0]), np.array([1]), np.array([10.]),
        np.array([0.]), 2)
    assert crossings.tolist() == [0] and run.tolist() == [0]
    summary = audit.summarize(np.array([0.]), np.array([np.nan]),
                              np.array([np.nan]), crossings, run,
                              np.array([2.]))
    assert summary["directional_efficiency_valid"] == 0
    assert summary["absolute_net_over_local_range"]["median"] is None
    assert summary["net_down_zero_up"] == [0, 1, 0]


def test_exact_ratio_and_endpoint_mismatch_fail_closed():
    audit.require_ratio_parity(np.array([2.]), np.array([4.]), np.array([.5]))
    with pytest.raises(RuntimeError, match="GRAY_PHYSICAL_RANGE_ATR_PARITY"):
        audit.require_ratio_parity(np.array([2.]), np.array([4.]), np.array([.6]))
    validity = np.array([M, 2*M, 3*M], np.int64)
    rows = np.array([2], np.int64)
    with pytest.raises(RuntimeError, match="GRAY_PHYSICAL_ENDPOINT"):
        audit.require_endpoint_and_window(np.array([0]), rows,
                                          np.array([OFFSET + 2]), validity,
                                          2, 2, np.zeros(3, bool))
    with pytest.raises(RuntimeError, match="GRAY_PHYSICAL_ENDPOINT"):
        audit.require_endpoint_and_window(np.array([M]), rows,
                                          np.array([OFFSET + 1]), validity,
                                          2, 2, np.zeros(3, bool))


def test_close_crossings_ignore_equal_and_terminal_count_stops_at_opposite():
    close = np.array([10.5, 9.5, 10., 10.25])
    crossings, run = audit.close_diagnostics(
        close, np.array([0]), np.array([3]), np.array([10.]),
        np.array([.25]), 4)
    assert crossings.tolist() == [2]
    assert run.tolist() == [1]


def test_summary_rejects_misaligned_report_arrays():
    with pytest.raises(RuntimeError, match="GRAY_PHYSICAL_SUMMARY_SHAPE"):
        audit.summarize(np.array([0.]), np.array([], dtype=float),
                        np.array([np.nan]), np.array([0]),
                        np.array([0]), np.array([1.]))
