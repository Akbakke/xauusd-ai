"""Focused synthetic checks for local side provenance, without market reads."""
import numpy as np
import pytest

from gx1.audit.verify_m5_state_side_locality_v1 import (
    closure_window_flags, local_side, replay_side_entry, summarize,
)
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target


def test_cold_local_side_preserves_known_reopen_but_resets_unknown_gap():
    open_ = np.array([100., 101., 101., 101., 101., 101.])
    high = open_.copy()
    low = open_.copy()
    close = open_.copy()
    theta = np.ones(6)
    known = np.zeros(6, bool)
    side, after_unknown = local_side(
        5, open_, high, low, close, theta, known, target)
    assert side == target.UP and not after_unknown
    unknown = known.copy()
    unknown[1] = True
    side, after_unknown = local_side(
        5, open_, high, low, close, theta, unknown, target)
    assert side == target.UNKNOWN and after_unknown


def test_documented_closure_immediately_before_local_window_is_reported():
    minute = 60_000_000_000
    axis = np.array([0, 10, 11, 12, 13, 14], dtype=np.int64) * minute
    unknown = np.zeros(6, bool)
    assert closure_window_flags(5, axis, unknown) == (False, True)
    unknown[1] = True
    assert closure_window_flags(5, axis, unknown) == (False, False)
    axis[3] += minute
    with pytest.raises(RuntimeError, match="CLOSURE_WINDOW"):
        closure_window_flags(5, axis, unknown)


def test_full_replay_keeps_side_entry_through_a_flat_observation():
    open_ = np.array([100., 101., 101., 101.])
    high = np.array([101., 101., 102., 102.])
    low = np.array([100., 101., 101., 102.])
    close = np.array([101., 101., 102., 102.])
    theta = np.ones(4)
    unknown = np.zeros(4, bool)
    flat = np.array([False, True, False, False])
    side, entry, last_flat = replay_side_entry(
        open_, high, low, close, theta, unknown, flat, target)
    assert np.array_equal(side, np.full(4, target.UP))
    assert np.array_equal(entry, np.zeros(4))
    assert last_flat[2] > entry[2]
    unknown[2] = True
    side, entry, _ = replay_side_entry(
        open_, high, low, close, theta, unknown, flat, target)
    assert entry[2] == 2
    assert side[2] == target.UP


def test_summary_separates_same_unknown_and_opposite():
    mask = np.array([True, True, True, False])
    local = np.array([target.UP, target.UNKNOWN, target.DOWN, target.UNKNOWN])
    full = np.array([target.UP, target.UP, target.UP, target.UNKNOWN])
    age = np.array([1, 2, 3, 0])
    age_utc = age.astype(float) * 2
    old = np.array([False, True, True, False])
    issue = np.array([True, False, False, False])
    flat = np.array([False, True, False, False])
    closure = np.array([False, False, True, False])
    closure_before = np.array([True, False, False, False])
    reset = np.array([False, True, False, False])
    result = summarize(mask, local, full, age, age_utc,
                       old, issue, flat, closure, closure_before, reset)
    assert (result["endpoints"], result["local_same"],
            result["local_unknown"], result["local_opposite"]) == (3, 1, 1, 1)
    assert result["side_entry_older_than_local_window"] == 2
    assert result["known_closure_within_window"] == 1
    assert result["known_closure_before_first_window_bar"] == 1
    assert result["known_closure_touching_window"] == 2
    assert result["first_window_bar_after_unknown_gap"] == 1
    with pytest.raises(RuntimeError, match="SUMMARY_CLASS"):
        summarize(mask, local, np.array([1, 1, 1, -1]), age, age_utc,
                  old, issue, flat, closure, closure_before, reset)
