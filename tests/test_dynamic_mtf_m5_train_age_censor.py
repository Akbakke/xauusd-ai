"""Small synthetic first-gap event cases; no market or target artifacts."""
import numpy as np
import pytest

from gx1.audit.dynamic_mtf_m5_train_age_censor import _endpoint_source_check, first_gap_m5_support
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target


M = target.MINUTE_NS
BASE = (1_700_000_000_000_000_000 // M) * M


def _case(offsets, states, *, observed=None, path=None, cutoff=7):
    axis = BASE + np.asarray(offsets, dtype=np.int64) * M
    state = np.asarray(states, dtype=np.int8)
    observed = np.ones(len(axis), dtype=bool) if observed is None else np.asarray(observed, dtype=bool)
    path = np.ones(len(axis), dtype=bool) if path is None else np.asarray(path, dtype=bool)
    return first_gap_m5_support(axis, np.asarray([BASE], np.int64),
                                np.asarray([0], np.int64), state, observed, path,
                                BASE + cutoff * M)


def test_first_exit_and_return_are_counted_at_exact_m1_closes():
    r = _case(range(6), [target.DOWN, target.DOWN, target.UP, target.UP, target.DOWN, target.DOWN])
    assert r['known_start_state_and_path'] == 1
    assert r['full_five_minute_followup'] == 1
    assert r['events']['first_exit']['event_lead_counts']['2'] == 1
    assert r['events']['first_exit_then_return']['event_lead_counts']['4'] == 1
    assert r['events']['first_exit']['at_risk_parent_minutes'] == 2
    assert r['events']['first_exit_then_return']['at_risk_parent_minutes'] == 4
    assert not any(r['events'][name]['right_censored_before_horizon'] for name in r['events'])


def test_full_observed_flat_path_confirms_both_events_absent():
    r = _case(range(6), [target.FLAT] * 6)
    assert r['full_five_minute_followup'] == 1
    assert all(r['events'][name]['confirmed_absent_full_horizon'] == 1 for name in r['events'])
    assert all(r['events'][name]['event_parent_issues'] == 0 for name in r['events'])


def test_first_missing_minute_censors_return_but_preserves_earlier_exit():
    r = _case([0, 1, 2, 4, 5], [target.DOWN, target.DOWN, target.UP, target.DOWN, target.DOWN])
    assert r['first_unobservable_future_reason']['missing_m1'] == 1
    assert r['consecutive_observed_future_minutes']['2'] == 1
    assert r['events']['first_exit']['event_parent_issues'] == 1
    assert r['events']['first_exit_then_return']['right_censored_before_horizon'] == 1
    assert r['events']['first_exit_then_return']['confirmed_absent_full_horizon'] == 0


def test_unknown_path_censors_before_a_potential_exit():
    r = _case(range(6), [target.DOWN, target.DOWN, target.UP, target.DOWN, target.DOWN, target.DOWN],
              path=[True, True, False, True, True, True])
    assert r['first_unobservable_future_reason']['unknown_path'] == 1
    assert r['events']['first_exit']['right_censored_before_horizon'] == 1
    assert r['events']['first_exit_then_return']['right_censored_before_horizon'] == 1


def test_cutoff_equality_never_confirms_absent_at_plus_five():
    r = _case(range(5), [target.DOWN] * 5, cutoff=5)
    assert r['first_unobservable_future_reason']['cutoff'] == 1
    assert r['consecutive_observed_future_minutes']['4'] == 1
    assert r['events']['first_exit']['right_censored_before_horizon'] == 1
    assert r['events']['first_exit']['confirmed_absent_full_horizon'] == 0


def test_unknown_issue_start_is_excluded_from_risk_set():
    r = _case(range(6), [target.DOWN] * 6,
              path=[False, True, True, True, True, True])
    assert r['known_start_state_and_path'] == 0
    assert r['unknown_start_reasons']['unknown_path'] == 1
    assert all(r['events'][name]['at_risk_parent_minutes'] == 0 for name in r['events'])


def test_saved_m5_endpoint_reconciliation_rejects_changed_counts():
    axis = BASE + np.arange(6, dtype=np.int64) * M
    arrays = {'issue_ns': np.asarray([BASE], np.int64), 'validity_ns': axis,
              'state': np.full((6, 5), target.DOWN, np.int8),
              'observed': np.ones((6, 5), dtype=bool),
              'reason': np.zeros((6, 5), np.uint8)}
    expected = {'parents': 1, 'immature': 0, 'missing_mature_m1_endpoint': 0,
                'exact_mature_m1_endpoint': 1, 'all_known_endpoint_classes': 1,
                'incomplete_local_window_or_atr': 0, 'complete_local_windows': 1,
                'complete_window_endpoint_counts': {'UNKNOWN': 0, 'DOWN': 1, 'FLAT': 0, 'UP': 0}}
    review = {'native_leads': [dict(timeframe='M5', lead_minutes=5, **expected)]}
    assert _endpoint_source_check(arrays, review, BASE + 7 * M) == expected
    review['native_leads'][0]['complete_window_endpoint_counts']['UP'] = 1
    with pytest.raises(RuntimeError, match='PRESERVED_ENDPOINT_COUNTS'):
        _endpoint_source_check(arrays, review, BASE + 7 * M)
