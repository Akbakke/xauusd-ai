"""One-use shared-M5 fit scope checks; no physical TRAIN pass is run."""
import hashlib
import json

import pytest

from gx1.audit import run_market_state_ta_fold2_shared_m5_fit_v1 as owner


def test_claim_and_terminal_are_exclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(owner, 'ROOT', tmp_path)
    claim = owner.publish('CLAIM.json', {'stage': owner.STAGE, 'test_data_used': False})
    assert claim['sha256'] == hashlib.sha256((tmp_path / 'CLAIM.json').read_bytes()).hexdigest()
    terminal = {'status': 'FAILED_CONSUMED', 'exit_code': 1, 'claim': claim,
                'test_data_used': False}
    owner.publish('TERMINAL.json', terminal)
    assert json.loads((tmp_path / 'TERMINAL.json').read_text()) == terminal
    with pytest.raises(FileExistsError):
        owner.publish('CLAIM.json', {})
    with pytest.raises(FileExistsError):
        owner.publish('TERMINAL.json', terminal)


def test_wrong_plan_hash_rejected_before_receipts_or_model(tmp_path, monkeypatch):
    monkeypatch.setattr(owner, 'ROOT', tmp_path)
    plan = tmp_path / 'PLAN.json'
    plan.write_text('{}\n')
    with pytest.raises(RuntimeError, match='FOLD2_SHARED_M5_FIT_PLAN_BINDING'):
        owner.preflight(plan, '0' * 64, require_admission=False)


def test_success_result_terminal_hash_binding(tmp_path, monkeypatch):
    monkeypatch.setattr(owner, 'ROOT', tmp_path)
    result = owner.publish('RESULT.json', {'status': 'COMPLETE_SHARED_M5_FIT_AND_READOUT',
                                           'test_data_used': False})
    owner.publish('TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0,
                                    'result': result, 'test_data_used': False})
    saved = json.loads((tmp_path / 'TERMINAL.json').read_text())
    assert saved['result']['sha256'] == hashlib.sha256((tmp_path / 'RESULT.json').read_bytes()).hexdigest()
