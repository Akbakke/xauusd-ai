"""Synthetic exact-endpoint and cutoff checks; no historical artifacts."""
import numpy as np
import pytest

from gx1.audit.dynamic_mtf_october_target_only import endpoint_support
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target


M = target.MINUTE_NS
BASE = 1_700_000_000_000_000_000 // M * M


def _report(offsets, issue_offsets, *, cutoff, labels=None, masks=None, flags=None):
    axis = BASE + np.asarray(offsets, np.int64) * M
    issue = BASE + np.asarray(issue_offsets, np.int64) * M
    n = len(axis)
    state = np.full(n, target.UNKNOWN, np.int8) if labels is None else np.asarray(labels, np.int8)
    observed = state != target.UNKNOWN if masks is None else np.asarray(masks, bool)
    reason = np.zeros(n, np.uint8) if flags is None else np.asarray(flags, np.uint8)
    return endpoint_support(axis, issue, state, observed, reason, BASE + cutoff * M, tf='M5', lead=5)


def test_cutoff_equality_is_immature_and_never_read_as_endpoint():
    r = _report(range(6), [0, 1], cutoff=6,
                labels=[target.UNKNOWN] * 5 + [target.UP])
    assert r['parents'] == 2 and r['immature'] == 1
    assert r['exact_mature_m1_endpoint'] == 1
    assert r['complete_window_endpoint_counts']['UP'] == 1
    assert r['complete_window_endpoint_counts']['DOWN'] == 0


def test_missing_exact_m1_endpoint_is_distinct_from_unknown_class():
    r = _report([0, 1, 2, 3, 4, 6], [0], cutoff=7)
    assert r['missing_mature_m1_endpoint'] == 1
    assert r['exact_mature_m1_endpoint'] == 0
    assert r['complete_local_windows'] == 0


def test_incomplete_local_window_is_counted_but_not_classified():
    r = _report(range(6), [0], cutoff=7,
                flags=[0, 0, 0, 0, 0, target.LOCAL_SPAN_INCOMPLETE])
    assert r['exact_mature_m1_endpoint'] == 1
    assert r['incomplete_local_window_or_atr'] == 1
    assert r['complete_local_windows'] == 0


def test_endpoint_mask_and_class_mismatch_fails_closed():
    with pytest.raises(RuntimeError, match='ENDPOINT_CLASS_MASK'):
        _report(range(6), [0], cutoff=7,
                labels=[target.UNKNOWN] * 5 + [target.UP], masks=[False] * 6)
