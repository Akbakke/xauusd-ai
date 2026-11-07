"""Focused candidate-tape schema, preflight and one-use receipt checks."""
import hashlib
import io
import json
import subprocess

import numpy as np
import pytest

from gx1.audit import build_market_state_time_query_tape_v1 as audit


def _tiny_tape(monkeypatch):
    monkeypatch.setattr(audit.duration.base, 'M1_ROWS', 4)
    axis = np.array([0, 1, 2, 4], np.int64) * audit.duration.MINUTE_NS
    rows = np.arange(5193205, 5193209, dtype=np.int64)
    bits = np.array([[2, 5, 1, 4, 7], [5, 2, 4, 1, 5],
                     [1, 4, 2, 5, 7], [4, 1, 5, 2, 5]], np.uint8)
    path = np.isin(bits, (1, 2, 4))
    gaps = np.array([0, 0, 0, 1], np.uint8)
    return axis, rows, bits, path, gaps


def test_candidate_tape_schema_roundtrip_has_only_target_axes_and_masks(monkeypatch):
    arrays = _tiny_tape(monkeypatch)
    payload, report = audit.encode_tape(*arrays)
    with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
        assert set(saved.files) == audit.TAPE_KEYS
        for key, expected in zip(('validity_ns', 'm1_source_row', 'allowed_set',
                                  'path_observed', 'gap_kind'), arrays):
            assert np.array_equal(saved[key], expected)
    assert report['m1_rows'] == 4
    assert report['allowed_counts']['M5']['2'] == 1
    assert report['allowed_counts']['M5']['5'] == 1
    assert report['documented_closure_gaps'] == 1


def test_invalid_mask_or_path_never_publishes(monkeypatch):
    axis, rows, bits, path, gaps = _tiny_tape(monkeypatch)
    bits[0, 0] = 3
    with pytest.raises(RuntimeError, match='TAPE_SCHEMA'):
        audit.encode_tape(axis, rows, bits, path, gaps)
    bits[0, 0] = 2
    path[0, 1] = True  # ACTIVE with unknown side cannot be path-confirmed singleton.
    with pytest.raises(RuntimeError, match='TAPE_SCHEMA'):
        audit.encode_tape(axis, rows, bits, path, gaps)


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
        'schema_version': audit.SCHEMA,
        'stage': audit.STAGE, 'status': 'PREPARED_NOT_ADMITTED',
        'output_root': str(out_dir), 'max_invocations': 1,
        'input_refs': audit.INPUT_REFS, 'source_owners': owners,
        'source_base_head': head, 'execution': audit.EXECUTION,
        'limits': audit.LIMITS, 'metric_contract': audit.CONTRACT,
        'numba_version': audit.duration.NUMBA_VERSION,
        'expected_m1_rows': 591682, 'expected_issue_rows': 118706,
        'command_template': audit.planned_command('{plan_sha256}'),
        'read_only': True, 'test_data_used': False,
    }
    plan_path.write_text(json.dumps(plan))
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    seen, ref, got_head, policy_ref = audit.preflight(plan_path, digest, out_dir)
    assert seen == plan and ref['sha256'] == digest and got_head == head
    assert policy_ref is None and called == ['small'] and not out_dir.exists()


def test_parity_failure_after_claim_is_terminal_without_tape(monkeypatch, tmp_path):
    out_dir = tmp_path / 'RUN'
    plan_ref = {'path': 'plan', 'sha256': 'a' * 64}
    policy_ref = {'path': 'policy', 'sha256': 'b' * 64}
    monkeypatch.setattr(audit, 'preflight', lambda *args, **kwargs:
                        ({'source_owners': {}}, plan_ref, 'head', policy_ref))
    monkeypatch.setattr(audit.duration.endpoint, '_checked_small_json',
                        lambda *args: {})
    arrays = {'validity_ns': np.array([0], np.int64),
              'source_row': np.array([5193205], np.int64)}
    monkeypatch.setattr(audit.duration, '_temporary_tapes', lambda *args:
                        (None, np.array([0], np.int64), np.array([0], np.int64),
                         {}, arrays, {}, np.array([0], np.uint8), {}))
    monkeypatch.setattr(audit.duration, '_selected_full_mask', lambda: {})
    monkeypatch.setattr(audit.duration, '_classify_all_m1', lambda *args:
                        (np.zeros((1, 5), np.uint8),
                         np.zeros((1, 5), bool), {}))
    monkeypatch.setattr(audit.duration, '_selected_endpoint_parity',
                        lambda *args: (_ for _ in ()).throw(
                            RuntimeError('synthetic endpoint mismatch')))
    with pytest.raises(RuntimeError, match='synthetic endpoint mismatch'):
        audit.run(audit.PLAN_PATH, 'a' * 64, out_dir)
    assert (out_dir / 'CLAIM.json').exists()
    assert not (out_dir / audit.TAPE_NAME).exists()
    assert not (out_dir / 'RESULT.json').exists()
    failed = json.loads((out_dir / 'FAILED.json').read_text())
    terminal = json.loads((out_dir / 'TERMINAL.json').read_text())
    assert failed['error'] == 'synthetic endpoint mismatch'
    assert terminal['status'] == 'FAILED' and terminal['exit_code'] == 1
    assert terminal['no_relaunch_under_this_plan'] is True
