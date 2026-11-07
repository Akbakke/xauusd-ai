from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

from gx1.audit import fit_market_state_mature_fold1_units_v1 as owner


def _report_fixture():
    _, original, profile = owner._original_receipts()
    sources = {'current_plan': {'path': '/tmp/new-plan', 'sha256': 'a' * 64},
               'current_source_head': 'b' * 40}
    report = copy.deepcopy(original)
    report['fit_allowed'] = True
    report['blockers'] = []
    report['unobserved_units_profile'] = profile
    report['declared_unit_support'] = original['blockers']
    report['proof']['source_bindings'] = sources
    return report, original, profile, sources


def test_existing_complete_raw_report_and_sealed_profile_match_exactly():
    result, old, profile = owner._original_receipts()
    assert result['cohorts'][0]['joined_fit_rows'] == owner.FIT_ROWS
    assert len(old['blockers']) == 153
    assert old['fit_allowed'] is False
    assert profile['contract_sha256'] == owner.PROFILE_SHA256
    assert profile['support_evidence'] == owner.INPUT_REFS['original_fold1_report']


def test_new_qualification_must_preserve_physical_proof_and_all_153_fields():
    report, original, profile, sources = _report_fixture()
    owner._validate_new_report(report, original, profile, sources)
    bad = copy.deepcopy(report)
    bad['declared_unit_support'] = bad['declared_unit_support'][:-1]
    with pytest.raises(RuntimeError, match='EXACT_REQUALIFICATION'):
        owner._validate_new_report(bad, original, profile, sources)
    bad = copy.deepcopy(report)
    bad['proof']['gc']['D1']['rows'] += 1
    with pytest.raises(RuntimeError, match='EXACT_REQUALIFICATION'):
        owner._validate_new_report(bad, original, profile, sources)
    bad = copy.deepcopy(report)
    bad['proof']['source_bindings'] = original['proof']['source_bindings']
    with pytest.raises(RuntimeError, match='EXACT_REQUALIFICATION'):
        owner._validate_new_report(bad, original, profile, sources)


def test_policy_is_one_use_train_only_and_cannot_read_test():
    ref = {'path': str(owner.PLAN_PATH), 'sha256': 'a' * 64}
    gate = {
        'status': 'ADMITTED', 'decision': 'ADMITTED', 'plan': ref,
        'execution_authorized': True, 'data_reads_authorized': True,
        'normalization_fit_authorized': True,
        'claimed_invocations': 0, 'completed_invocations': 0,
        'max_invocations': 1, 'relaunch_allowed': False,
        'memory_max': '4G', 'swap_max': '512M', 'wall_seconds': 2400,
        'test_data_used': False, 'checkpoint_reads_authorized': False,
        'broker_access_allowed': False, 'model_forwards_authorized': 0,
        'optimizer_steps_authorized': 0, 'training_authorized': False,
        'test_authorized': False, 'output_root': str(owner.OUT_ROOT),
    }
    policy = {
        'training_enabled': False, 'native_training_enabled': False,
        'native_launch_authorized': False, 'full_val_allowed': False,
        owner.STAGE: gate,
    }
    owner._policy_gate(policy, ref)
    changed = copy.deepcopy(policy)
    changed[owner.STAGE]['claimed_invocations'] = 1
    with pytest.raises(RuntimeError, match='POLICY_GATE'):
        owner._policy_gate(changed, ref)
    changed = copy.deepcopy(policy)
    changed[owner.STAGE]['test_authorized'] = True
    with pytest.raises(RuntimeError, match='POLICY_GATE'):
        owner._policy_gate(changed, ref)


def test_cli_rejects_duplicate_flags_and_nonabsolute_paths(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['fold1', '--plan', str(owner.PLAN_PATH),
        '--plan', str(owner.PLAN_PATH), '--plan-sha256', 'a' * 64,
        '--out-dir', str(owner.OUT_ROOT), '--preflight'])
    with pytest.raises(SystemExit) as exc:
        owner.main()
    assert exc.value.code == 2
    monkeypatch.setattr(sys, 'argv', ['fold1', '--plan', 'relative-plan.json',
        '--plan-sha256', 'a' * 64, '--out-dir', str(owner.OUT_ROOT),
        '--preflight'])
    with pytest.raises(SystemExit) as exc:
        owner.main()
    assert exc.value.code == 2


def test_read_only_preflight_cli_does_not_call_run(monkeypatch, capsys):
    expected = {'status': 'READY_NO_CLAIM_NO_PAYLOAD', 'claim_created': False}
    monkeypatch.setattr(owner, 'preflight_read_only',
                        lambda plan_path, plan_sha256, out_dir: expected)
    monkeypatch.setattr(owner, 'run',
                        lambda *args: (_ for _ in ()).throw(AssertionError('RUN')))
    monkeypatch.setattr(sys, 'argv', ['fold1', '--plan', str(owner.PLAN_PATH),
        '--plan-sha256', 'a' * 64, '--out-dir', str(owner.OUT_ROOT),
        '--preflight'])
    assert owner.main() == 0
    assert 'READY_NO_CLAIM_NO_PAYLOAD' in capsys.readouterr().out


def test_tiny_success_and_failure_receipts_are_hash_bound(tmp_path, monkeypatch):
    import hashlib
    import json

    complete_root = tmp_path / 'complete'
    complete_root.mkdir()
    monkeypatch.setattr(owner, '_published_bytes', 0)
    claim = owner._write(complete_root, 'CLAIM.json', {
        'schema_version': 'gx1_market_state_mature_fold1_normalization_claim_v1',
        'invocation': 1, 'test_data_used': False})
    result = owner._write(complete_root, 'RESULT.json', {
        'schema_version': 'gx1_market_state_mature_fold1_normalization_result_v1',
        'status': 'COMPLETE_FOLD1_MATURE_UNITS_PREFIX_NORMALIZATION_NO_MODEL',
        'claim': claim, 'statistics_fits': 1, 'test_data_used': False})
    terminal = owner._write(complete_root, 'TERMINAL.json', {
        'schema_version': 'gx1_market_state_mature_fold1_normalization_terminal_v1',
        'status': 'COMPLETE', 'exit_code': 0, 'result': result,
        'no_relaunch_under_this_plan': True})
    for ref in (claim, result, terminal):
        assert hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest() == ref['sha256']
    assert json.loads((complete_root / 'TERMINAL.json').read_text())['result'] == result

    failed_root = tmp_path / 'failed'
    monkeypatch.setattr(owner, '_preflight', lambda *args, **kwargs: (
        {}, {}, 'c' * 40, {}, {'path': '/tmp/policy', 'sha256': 'd' * 64}))
    monkeypatch.setattr(owner, '_original_receipts',
                        lambda: (_ for _ in ()).throw(RuntimeError('synthetic_failure')))
    with pytest.raises(RuntimeError, match='synthetic_failure'):
        owner.run(owner.PLAN_PATH, 'e' * 64, failed_root)
    assert sorted(path.name for path in failed_root.iterdir()) == [
        'CLAIM.json', 'FAILED.json', 'TERMINAL.json']
    failure = json.loads((failed_root / 'FAILED.json').read_text())
    receipt = json.loads((failed_root / 'TERMINAL.json').read_text())
    assert receipt['status'] == 'FAILED'
    assert receipt['exit_code'] == 1
    assert receipt['failure']['sha256'] == hashlib.sha256(
        (failed_root / 'FAILED.json').read_bytes()).hexdigest()
    assert failure['no_relaunch_under_this_plan'] is True
    assert receipt['no_relaunch_under_this_plan'] is True
    assert not (failed_root / 'RESULT.json').exists()
