"""Synthetic checks for midpoint evidence, censoring, and read-only preflight."""
import hashlib
import json
import subprocess

import numpy as np
import pytest

from gx1.audit import verify_market_state_two_checkpoint_v1 as audit


def test_midpoint_missed_path_lower_bounds_and_gap_censor():
    minutes = np.array([*range(12), *range(13, 21)], dtype=np.int64)
    axis = minutes * audit.MINUTE_NS
    issues = np.array([0, 5, 10], dtype=np.int64) * audit.MINUTE_NS
    bits = np.full(len(axis), 2, np.uint8)
    by_minute = {int(v): i for i, v in enumerate(minutes)}
    bits[by_minute[1]] = 5  # ACTIVE between two FLAT checkpoints.
    bits[by_minute[7]] = 4  # Certified opposite side between two DOWN checkpoints.
    bits[by_minute[8]] = bits[by_minute[10]] = 1
    bits[by_minute[11]] = 7  # Unknown label before missing minute 12.
    path = np.zeros(len(axis), bool)
    path[[by_minute[v] for v in (7, 8, 10)]] = True
    gap_kind = np.zeros(len(axis), np.uint8)
    gap_kind[by_minute[13]] = 2
    got = audit.measure_tf(issues, axis, bits, path, gap_kind, 5,
                           int(axis[-1] + audit.MINUTE_NS))
    all_ = got['all']
    assert got['midpoint_minutes'] == 3
    assert got['midpoint_exact_M1'] == got['endpoint_exact_M1'] == 3
    assert all_['midpoint_immature'] == all_['endpoint_immature'] == 0
    assert all_['midpoint_missing_exact_M1'] == all_['endpoint_missing_exact_M1'] == 0
    assert all_['issues'] == 3
    assert all_['both_flat'] == 2
    assert all_['missed_activity_observed_lower_bound'] == 1
    assert all_['both_same_singleton_side'] == 1
    assert all_['missed_opposite_path_confirmed_lower_bound'] == 1
    assert all_['activity_observable_both_flat'] == 1
    assert all_['side_observable_both_same_singleton'] == 1
    assert all_['missed_activity_on_activity_observable'] == 1
    assert all_['missed_opposite_on_side_observable'] == 1
    assert all_['full_M1_clock'] == 2
    assert all_['intervals_with_missing_M1'] == 1
    assert all_['missing_M1_minutes'] == 1
    assert all_['intervals_with_unknown_label'] == 1
    assert all_['intervals_crossing_unknown_source_gap'] == 1
    assert sum(m['issues'] for m in got['months'].values()) == 3


def test_uncertified_opposite_and_missing_midpoint_do_not_become_evidence():
    axis = np.array([0, 1, 2, 4, 5], np.int64) * audit.MINUTE_NS
    issues = np.array([0], np.int64)
    bits = np.array([1, 4, 4, 1, 1], np.uint8)
    path = np.zeros(len(axis), bool)
    gap_kind = np.array([0, 0, 0, 1, 0], np.uint8)
    got = audit.measure_tf(issues, axis, bits, path, gap_kind, 5,
                           6 * audit.MINUTE_NS)
    all_ = got['all']
    assert got['midpoint_minutes'] == 3
    assert got['midpoint_exact_M1'] == 0
    assert got['endpoint_exact_M1'] == 1
    assert all_['midpoint_immature'] == all_['endpoint_immature'] == 0
    assert all_['midpoint_missing_exact_M1'] == 1
    assert all_['endpoint_missing_exact_M1'] == 0
    assert all_['midpoint_bits']['7'] == 1
    assert all_['both_same_singleton_side'] == 0
    assert all_['missed_opposite_path_confirmed_lower_bound'] == 0
    assert all_['intervals_crossing_documented_closure'] == 1
    assert all_['missing_M1_minutes'] == 1


def test_nonpath_opposite_not_promoted_even_with_same_checkpoint_side():
    axis = np.arange(6, dtype=np.int64) * audit.MINUTE_NS
    bits = np.array([1, 4, 4, 1, 1, 1], np.uint8)
    path = np.zeros(6, bool)
    got = audit.measure_tf(np.array([0], np.int64), axis, bits, path,
                           np.zeros(6, np.uint8), 5, 6 * audit.MINUTE_NS)
    assert got['all']['both_same_singleton_side'] == 1
    assert got['all']['side_observable_both_same_singleton'] == 0
    assert got['all']['intervals_with_unconfirmed_singleton_side'] == 1
    assert got['all']['missed_opposite_path_confirmed_lower_bound'] == 0
    path[1] = True
    got = audit.measure_tf(np.array([0], np.int64), axis, bits, path,
                           np.zeros(6, np.uint8), 5, 6 * audit.MINUTE_NS)
    assert got['all']['missed_opposite_path_confirmed_lower_bound'] == 1
    assert got['all']['side_observable_both_same_singleton'] == 0
    path[2:] = True
    got = audit.measure_tf(np.array([0], np.int64), axis, bits, path,
                           np.zeros(6, np.uint8), 5, 6 * audit.MINUTE_NS)
    assert got['all']['side_observable_both_same_singleton'] == 1
    assert got['all']['missed_opposite_on_side_observable'] == 1


def test_preflight_checks_small_refs_without_target_payload(monkeypatch, tmp_path):
    plan_path, out_dir = tmp_path / 'PLAN.json', tmp_path / 'RUN'
    monkeypatch.setattr(audit, 'PLAN_PATH', plan_path)
    monkeypatch.setattr(audit, 'OUT_ROOT', out_dir)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    monkeypatch.setattr(audit.duration, '_clean_head', lambda: head)
    called = []
    monkeypatch.setattr(audit.duration.endpoint, '_small_reference_chain',
                        lambda: called.append('small'))
    monkeypatch.setattr(audit.duration.binding, '_checked_bytes',
                        lambda *args: pytest.fail('preflight read target payload'))
    owners = {name: hashlib.sha256((audit.REPO / name).read_bytes()).hexdigest()
              for name in audit.SOURCE_PATHS}
    plan = {
        'schema_version': 'gx1_market_state_two_checkpoint_plan_v1',
        'stage': audit.STAGE, 'status': 'PREPARED_NOT_ADMITTED',
        'output_root': str(out_dir), 'max_invocations': 1,
        'input_refs': audit.INPUT_REFS, 'source_owners': owners,
        'source_base_head': head, 'execution': audit.EXECUTION,
        'limits': audit.LIMITS, 'metric_contract': audit.CONTRACT,
        'expected_m1_rows': 591682, 'expected_issue_rows': 118706,
        'numba_version': audit.duration.NUMBA_VERSION,
        'command_template': audit.planned_command('{plan_sha256}'),
        'read_only': True, 'test_data_used': False,
    }
    plan_path.write_text(json.dumps(plan))
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    seen, ref, got_head, policy_ref = audit.preflight(plan_path, digest, out_dir)
    assert seen == plan and ref['sha256'] == digest and got_head == head
    assert policy_ref is None and called == ['small'] and not out_dir.exists()


def test_postclaim_failure_is_terminal_and_one_use(monkeypatch, tmp_path):
    out_dir = tmp_path / 'RUN'
    plan_ref = {'path': 'plan', 'sha256': 'a' * 64}
    policy_ref = {'path': 'policy', 'sha256': 'b' * 64}
    monkeypatch.setattr(audit, 'preflight', lambda *args, **kwargs:
                        ({'source_owners': {}}, plan_ref, 'head', policy_ref))
    monkeypatch.setattr(audit.duration.endpoint, '_checked_small_json',
                        lambda *args: {})
    monkeypatch.setattr(audit.duration, '_temporary_tapes', lambda *args:
                        (_ for _ in ()).throw(RuntimeError('synthetic blocker')))
    with pytest.raises(RuntimeError, match='synthetic blocker'):
        audit.run(audit.PLAN_PATH, 'a' * 64, out_dir)
    assert (out_dir / 'CLAIM.json').exists()
    assert not (out_dir / 'RESULT.json').exists()
    failed = json.loads((out_dir / 'FAILED.json').read_text())
    terminal = json.loads((out_dir / 'TERMINAL.json').read_text())
    assert failed['error'] == 'synthetic blocker'
    assert terminal['status'] == 'FAILED' and terminal['exit_code'] == 1
    assert terminal['failure']['path'] == str(out_dir / 'FAILED.json')
    assert terminal['no_relaunch_under_this_plan'] is True


def test_cutoff_precedes_exact_future_source_in_checkpoint_reasons():
    axis = np.arange(21, dtype=np.int64) * audit.MINUTE_NS
    issues = np.array([0, 5, 10, 15], np.int64) * audit.MINUTE_NS
    bits = np.full(len(axis), 2, np.uint8)
    got = audit.measure_tf(issues, axis, bits, np.zeros(len(axis), bool),
                           np.zeros(len(axis), np.uint8), 5,
                           15 * audit.MINUTE_NS)
    all_ = got['all']
    assert all_['midpoint_exact_M1'] == 3
    assert all_['midpoint_immature'] == 1
    assert all_['midpoint_missing_exact_M1'] == 0
    assert all_['endpoint_exact_M1'] == 2
    assert all_['endpoint_immature'] == 2
    assert all_['endpoint_missing_exact_M1'] == 0
    assert all_['full_M1_clock'] == 2
    assert all_['intervals_with_missing_M1'] == 0
    assert list(got['months'].values())[0]['endpoint_immature'] == 2
