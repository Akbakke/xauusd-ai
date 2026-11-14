"""One-use baseline binding checks; no model, data or GPU execution."""
import hashlib
import json

import pytest

from gx1.audit import observe_market_state_ta_fold2_shared_m5_initial_v1 as owner


def test_claim_and_result_publication_are_exclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(owner, 'ROOT', tmp_path)
    ref = owner.publish('CLAIM.json', {'stage': owner.STAGE, 'test_data_used': False})
    assert ref['sha256'] == hashlib.sha256((tmp_path / 'CLAIM.json').read_bytes()).hexdigest()
    assert json.loads((tmp_path / 'CLAIM.json').read_text())['test_data_used'] is False
    with pytest.raises(FileExistsError):
        owner.publish('CLAIM.json', {'stage': owner.STAGE})


def test_wrong_plan_hash_rejected_before_receipts_or_model(tmp_path, monkeypatch):
    monkeypatch.setattr(owner, 'ROOT', tmp_path)
    plan = tmp_path / 'PLAN.json'
    plan.write_text('{}\n')
    with pytest.raises(RuntimeError, match='FOLD2_SHARED_M5_INITIAL_PLAN_BINDING'):
        owner.preflight(plan, '0' * 64, require_admission=False)


@pytest.mark.parametrize('success', [False, True])
def test_terminal_success_and_failure_publication_is_immutable(tmp_path, monkeypatch, success):
    monkeypatch.setattr(owner, 'ROOT', tmp_path)
    claim = owner.publish('CLAIM.json', {'stage': owner.STAGE, 'test_data_used': False})
    terminal = {'status': 'COMPLETE' if success else 'FAILED_CONSUMED',
                'exit_code': 0 if success else 1, 'claim': claim,
                'test_data_used': False}
    if success:
        terminal['result'] = owner.publish('RESULT.json', {
            'status': 'COMPLETE_ZERO_UPDATE_SHARED_M5_INITIAL',
            'optimizer_updates': 0, 'test_data_used': False})
    else:
        terminal['error'] = 'RuntimeError:synthetic'
    owner.publish('TERMINAL.json', terminal)
    saved = json.loads((tmp_path / 'TERMINAL.json').read_text())
    assert saved == terminal
    assert (tmp_path / 'RESULT.json').exists() is success
    with pytest.raises(FileExistsError):
        owner.publish('TERMINAL.json', terminal)
