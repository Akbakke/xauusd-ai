"""Synthetic checks for the unfitted exact-clock persistence reference."""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gx1.audit import measure_market_state_causal_persistence_reference_v1 as audit
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as target


def _clock_fixture():
    t = np.int64(1730000000000000000)
    leads = np.asarray(audit.LEAD_MINUTES, np.int64)
    issues = np.sort(t - leads * target.MINUTE_NS)
    m1 = np.full((5, 5), -1, dtype=np.int64)
    allowed = np.full((5, 5), 7, dtype=np.uint8)
    for j, lead in enumerate(leads):
        at = int(np.searchsorted(issues, t - lead * target.MINUTE_NS))
        m1[at, j] = 42
        allowed[at, j] = (1, 2, 4, 5, 7)[j]
    return issues, m1, allowed, np.asarray([t], dtype=np.int64)


def test_exact_predecessor_and_same_m1_source_row_abstain_explicitly():
    issues, m1, allowed, query = _clock_fixture()
    bits, reasons = audit._current_states(issues, m1, allowed, query)
    assert bits.tolist() == [[1, 2, 4, 5, 0]]
    assert reasons.tolist() == [[0, 0, 0, 0, 5]]
    # Removing one exact TF predecessor is not repaired by a nearest row.
    keep = issues != query[0] - 15 * target.MINUTE_NS
    bits, reasons = audit._current_states(issues[keep], m1[keep], allowed[keep], query)
    assert bits[0, 1] == 0 and reasons[0, 1] == 3
    # Without the independent M5 endpoint row, no TF obtains a forecast.
    keep = issues != query[0] - target.M5_NS
    bits, reasons = audit._current_states(issues[keep], m1[keep], allowed[keep], query)
    assert bits.tolist() == [[0] * 5]
    assert reasons.tolist() == [[1] * 5]


def test_missing_physical_row_abstains_and_mismatch_fails_closed():
    issues, m1, allowed, query = _clock_fixture()
    h1_at = int(np.searchsorted(issues, query[0] - 60 * target.MINUTE_NS))
    missing = m1.copy()
    missing[h1_at, 2] = -1
    bits, reasons = audit._current_states(issues, missing, allowed, query)
    assert bits[0, 2] == 0 and reasons[0, 2] == 4
    mismatched = m1.copy()
    mismatched[h1_at, 2] = 43
    with pytest.raises(RuntimeError, match='PHYSICAL_M1_ROW_MISMATCH:H1'):
        audit._current_states(issues, mismatched, allowed, query)


def test_persistence_brier_has_exact_support_and_infinite_log_on_wrong_side():
    n = 4
    now = np.zeros((n, 5), dtype=np.uint8)
    now[:, 0] = [1, 4, 2, 5]
    reasons = np.ones((n, 5), dtype=np.uint8)
    reasons[:, 0] = 0
    allowed = np.full((n, 5), 7, dtype=np.uint8)
    allowed[:, 0] = [4, 1, 2, 2]
    truth = {'allowed': allowed, 'eligible': allowed != 7,
             'mature': np.ones((n, 5), dtype=np.bool_)}
    probabilities = np.tile(np.asarray([0.3, 0.2, 0.5]), (5, 1))
    issues = np.asarray([1730000000000000000 + i * target.M5_NS for i in range(n)],
                        dtype=np.int64)
    scored = audit._score_rows(now, reasons, truth, probabilities, issues)
    m5 = scored['M5']
    assert m5['activity']['count'] == 4
    assert m5['activity']['wrong_count'] == 1
    assert m5['activity']['persistence_brier_sum'] == 1
    assert m5['conditional_side']['count'] == 2
    assert m5['conditional_side']['wrong_count'] == 2
    assert m5['conditional_side']['persistence_brier_sum'] == 2
    assert m5['conditional_side']['persistence_log_mean'] == 'Infinity'
    assert m5['three_class']['count'] == 3
    assert m5['three_class']['wrong_count'] == 2
    assert m5['three_class']['persistence_brier_sum'] == 4
    assert m5['three_class']['persistence_zero_truth_probability_rows'] == 2
    assert m5['current_bits']['5'] == 1  # activity only; never guessed a side
    assert scored['D1']['activity']['count'] == 0
    assert scored['D1']['no_forecast']['NO_M5_ANCHOR'] == n
    assert m5['activity']['prior_brier_mean'] is not None
    assert m5['conditional_side']['prior_log_mean'] is not None


def _plan(tmp_path, monkeypatch):
    out = tmp_path / 'out'
    path = tmp_path / 'plan.json'
    monkeypatch.setattr(audit, 'OUT_ROOT', out)
    monkeypatch.setattr(audit, 'PLAN_PATH', path)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    refs = {name: {'path': source, 'sha256': digest}
            for name, (source, digest) in target.REFS.items()}
    plan = {'schema_version': audit.SCHEMA, 'stage': audit.STAGE,
            'status': 'PREPARED_NOT_ADMITTED', 'source_base_head': head,
            'output_root': str(out), 'fold_ids': [1, 2, 3, 4, 5],
            'reference_rule': audit.RULE, 'max_invocations': 1,
            'input_refs': refs, 'prior_result': audit.PRIOR_RESULT,
            'prior_terminal': audit.PRIOR_TERMINAL,
            'limits': audit.LIMITS, 'execution': audit.EXECUTION,
            'model_forwards_authorized': 0, 'optimizer_steps_authorized': 0,
            'training_authorized': False, 'baseline_estimation_authorized': False,
            'test_data_used': False,
            'source_owners': {name: audit._sha(audit.REPO / name)
                              for name in audit.SOURCE_PATHS}}
    path.write_text(json.dumps(plan))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return plan, path, sha, out, head


def test_read_only_preflight_has_no_claim_or_payload_decode(tmp_path, monkeypatch):
    plan, path, sha, out, head = _plan(tmp_path, monkeypatch)
    monkeypatch.setattr(audit.prior, '_clean_head', lambda: head)
    monkeypatch.setattr(audit, '_receipt_preflight', lambda: {})
    report = audit.preflight_read_only(path, sha, out)
    assert report['status'] == 'READY_NO_CLAIM_NO_PAYLOAD'
    assert report['claim_created'] is False and not out.exists()
    assert report['model_forwards'] == report['optimizer_steps'] == 0
    plan['baseline_estimation_authorized'] = True
    with pytest.raises(RuntimeError, match='PLAN_SCOPE'):
        audit._qualify_plan(plan, {'path': str(path), 'sha256': sha}, head)


def _publication_setup(tmp_path, monkeypatch):
    plan, path, sha, out, head = _plan(tmp_path, monkeypatch)
    ref = {'path': str(path), 'sha256': sha}
    policy_ref = {'path': str(audit.POLICY), 'sha256': audit._sha(audit.POLICY)}
    monkeypatch.setattr(audit, '_preflight', lambda *_: (plan, ref, policy_ref, head, {}))
    monkeypatch.setattr(audit.prior, '_clean_head', lambda: head)
    return path, sha, out


def test_one_use_success_and_failure_terminals(tmp_path, monkeypatch):
    path, sha, out = _publication_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(audit, '_measure', lambda *_: {
        'model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False})
    report = audit.run(path, sha, out)
    assert {p.name for p in out.iterdir()} == {'CLAIM.json', 'RESULT.json', 'TERMINAL.json'}
    terminal = json.loads((out / 'TERMINAL.json').read_text())
    assert terminal['result'] == report['result']
    assert terminal['no_relaunch_under_this_plan'] is True
    with pytest.raises(FileExistsError):
        audit.run(path, sha, out)

    second = tmp_path / 'second'
    monkeypatch.setattr(audit, 'OUT_ROOT', second)
    monkeypatch.setattr(audit, '_measure', lambda *_: (_ for _ in ()).throw(RuntimeError('synthetic stop')))
    with pytest.raises(RuntimeError, match='synthetic stop'):
        audit.run(path, sha, second)
    assert {p.name for p in second.iterdir()} == {'CLAIM.json', 'FAILED.json', 'TERMINAL.json'}
    failed = json.loads((second / 'TERMINAL.json').read_text())
    assert failed['status'] == 'FAILED' and failed['no_relaunch_under_this_plan'] is True


def test_admitted_cpu_policy_and_lock_gate_fail_closed(tmp_path, monkeypatch):
    from gx1.contracts import gx1_capped_execution_v1 as capped
    plan, path, sha, out, head = _plan(tmp_path, monkeypatch)
    policy_path = tmp_path / 'policy.json'
    monkeypatch.setattr(audit, 'POLICY', policy_path)
    monkeypatch.setattr(audit.prior, '_clean_head', lambda: head)
    monkeypatch.setattr(audit, '_receipt_preflight', lambda: {})
    cap = {'memory_max': audit.LIMITS['memory_max'],
           'memory_high': audit.LIMITS['memory_high'],
           'swap': audit.LIMITS['swap'], 'pids': 64,
           'cgroup_path': '/sys/fs/cgroup/synthetic.scope'}
    monkeypatch.setattr(capped, 'require_capped_cpu_audit_execution', lambda: cap)
    monkeypatch.setattr(capped, 'require_capped_lock_ancestry',
                        lambda: {'scope': '/synthetic.scope'})
    gate = {'decision': 'ADMITTED',
            'plan': {'path': str(path), 'sha256': sha},
            'execution_authorized': True, 'data_reads_authorized': True,
            'deterministic_reference_authorized': True,
            'baseline_estimation_authorized': False,
            'claimed_invocations': 0, 'completed_invocations': 0,
            'max_invocations': 1, 'model_forwards_authorized': 0,
            'optimizer_steps_authorized': 0, 'training_authorized': False,
            'test_authorized': False, 'relaunch_allowed': False,
            'output_root': str(out), 'memory_max': '2G',
            'swap_max': '512M', 'wall_seconds': 600}
    policy = {'training_enabled': False, 'native_training_enabled': False,
              'native_launch_authorized': False, 'full_val_allowed': False,
              audit.STAGE: gate}
    policy_path.write_text(json.dumps(policy))
    assert audit._preflight(path, sha, out)[0] == plan
    gate['deterministic_reference_authorized'] = False
    policy_path.write_text(json.dumps(policy))
    with pytest.raises(RuntimeError, match='POLICY_GATE'):
        audit._preflight(path, sha, out)
    gate['deterministic_reference_authorized'] = True
    policy_path.write_text(json.dumps(policy))
    cap['memory_max'] -= 1
    with pytest.raises(RuntimeError, match='CAP_LOCK'):
        audit._preflight(path, sha, out)
    assert not out.exists()
