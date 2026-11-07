"""Synthetic exact-clock and maturity checks for a caller-selected query."""
import numpy as np
import pytest

from gx1.contracts.market_state_time_query_target_v1 import (
    MINUTE_NS, NATIVE_K, TimeQueryTargetTape,
)


def test_native_endpoint_queries_match_selected_synthetic_mask_and_rows():
    minutes = np.array([5, 15, 60, 240, 1440], dtype=np.int64)
    bits = np.full((len(minutes), 5), 7, dtype=np.uint8)
    selected_bits = np.array([5, 1, 4, 2, 5], dtype=np.uint8)
    bits[np.arange(5), np.arange(5)] = selected_bits
    source = np.arange(900, 905, dtype=np.int64)
    tape = TimeQueryTargetTape(minutes * MINUTE_NS, source, bits)
    result = tape.gather(np.array([0], dtype=np.int64), np.asarray(NATIVE_K, dtype=np.int64)[None, :],
                         fold_upper_ns=1441 * MINUTE_NS)
    assert np.array_equal(result['validity_ns'][0], minutes * MINUTE_NS)
    assert np.array_equal(result['allowed'][0], selected_bits)
    assert np.array_equal(result['m1_source_row'][0], source)
    assert result['mature'].all() and result['eligible'].all()


def test_active_unknown_side_is_informative_but_all_unknown_is_not():
    bits = np.array([[5, 1, 2, 4, 7]], dtype=np.uint8)
    tape = TimeQueryTargetTape(np.array([5 * MINUTE_NS], dtype=np.int64),
                               np.array([10], dtype=np.int64), bits)
    result = tape.gather(np.array([0], dtype=np.int64),
                         np.ones((1, 5), dtype=np.int64),
                         fold_upper_ns=10 * MINUTE_NS)
    assert np.array_equal(result['allowed'][0], bits[0])
    assert np.array_equal(result['eligible'][0], [True, True, True, True, False])
    assert result['m1_source_row'][0, 4] == 10


def test_immature_precedes_source_lookup_and_mature_missing_stays_unknown():
    axis = np.array([5, 60], dtype=np.int64) * MINUTE_NS
    bits = np.full((2, 5), 4, dtype=np.uint8)
    tape = TimeQueryTargetTape(axis, np.array([10, 11], dtype=np.int64), bits)
    result = tape.gather(np.array([0], dtype=np.int64),
                         np.array([[1, 2, 12, 1, 1]], dtype=np.int64),
                         fold_upper_ns=30 * MINUTE_NS)
    assert result['allowed'][0, 0] == 4 and result['m1_source_row'][0, 0] == 10
    assert result['mature'][0, 1] and result['allowed'][0, 1] == 7
    assert result['m1_source_row'][0, 1] == -1
    assert not result['mature'][0, 2]
    assert result['allowed'][0, 2] == 7 and result['m1_source_row'][0, 2] == -1
    assert not result['eligible'][0, 1] and not result['eligible'][0, 2]


def test_tape_is_copied_and_query_bounds_are_per_timeframe():
    axis = np.array([5 * MINUTE_NS], dtype=np.int64)
    source = np.array([10], dtype=np.int64)
    bits = np.full((1, 5), 2, dtype=np.uint8)
    tape = TimeQueryTargetTape(axis, source, bits)
    axis[0] = 0; source[0] = 99; bits[0, 0] = 1
    result = tape.gather(np.array([0], dtype=np.int64),
                         np.ones((1, 5), dtype=np.int64),
                         fold_upper_ns=10 * MINUTE_NS)
    assert result['allowed'][0, 0] == 2 and result['m1_source_row'][0, 0] == 10
    with pytest.raises(ValueError):
        tape.allowed_bits.flags.writeable = True
    bad = np.ones((1, 5), dtype=np.int64); bad[0, 0] = 2
    with pytest.raises(ValueError, match='QUERY_DOMAIN'):
        tape.gather(np.array([0], dtype=np.int64), bad,
                    fold_upper_ns=10 * MINUTE_NS)


def test_two_issues_keep_timeframe_and_source_rows_aligned():
    bits = np.array([[1, 2, 4, 5, 7], [4, 5, 2, 1, 7]], dtype=np.uint8)
    tape = TimeQueryTargetTape(np.array([5, 10], dtype=np.int64) * MINUTE_NS,
                               np.array([100, 101], dtype=np.int64), bits)
    issues = np.array([0, 5], dtype=np.int64) * MINUTE_NS
    result = tape.gather(issues, np.ones((2, 5), dtype=np.int64),
                         fold_upper_ns=15 * MINUTE_NS)
    assert np.array_equal(result['allowed'], bits)
    assert np.array_equal(result['m1_source_row'],
                          np.array([[100] * 5, [101] * 5], dtype=np.int64))
