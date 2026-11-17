"""Cold fit admission binds consumed input and initial receipts before CLAIM."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gx1.audit import run_market_state_ta_m5_decision_fit_v1 as fit


def _closed_policy(monkeypatch):
    policy = json.loads((fit.REPO / 'NEXT_RUN_POLICY.json').read_text())
    policy[fit.COLD_STAGE] = {'status': 'PREPARED_NOT_ADMITTED',
        'plan': fit.ref(fit.COLD_PLAN), 'remaining_invocations': 0,
        'test_data_used': False, 'test_authorized': False,
        'online_authorized': False, 'broker_access_allowed': False,
        'execution_authorized': False, 'training_authorized': False}
    monkeypatch.setattr(fit, 'json', SimpleNamespace(loads=lambda _text: policy))
    monkeypatch.setattr(fit, 'source_clean', lambda _plan: True)
    return policy


def test_cold_plan_reuses_exact_preflight_and_prefix_only_stats(monkeypatch):
    _closed_policy(monkeypatch)
    plan, stats = fit.inspect_cold(fit.ref(fit.COLD_PLAN)['sha256'], admitted=False)
    assert plan['inputs']['initial_result']['sha256'] == (
        '1b536acf1f56e3d24687a17b1eae258eb49b08b085d14c71d7093a499e608f9a')
    assert plan['inputs']['raw_result']['sha256'] == (
        '26fea62e0ccf1f1af93c7882c9f13c9cb354cdab4cdd50334d6fc63cca6c2b50')
    assert plan['initial_model_state_sha256'] == (
        '803214ff3b5b5724cb1c1488e5ae29c1c17d479ca8db23d51301a45a6094629e')
    assert stats['sha256'] == plan['fit_statistics_sha256']
    assert (plan['fit_rows'], plan['excluded_gap_rows'], plan['score_rows']) == (29574, 275, 17489)
    assert plan['model_forwards'] == 4037 and plan['max_optimizer_updates'] == 1849
    assert not fit.COLD_ROOT.exists()


def test_cold_plan_rejects_mutated_initial_receipt_before_claim(monkeypatch):
    _closed_policy(monkeypatch)
    real_read = fit.read
    def altered(binding):
        value = real_read(binding)
        if binding['path'] == str(fit.COLD_INITIAL_ROOT / 'RESULT.json'):
            value = copy.deepcopy(value)
            value['initial_model_state_sha256'] = '0' * 64
        return value
    monkeypatch.setattr(fit, 'read', altered)
    with pytest.raises(RuntimeError, match='COLD_RAW_AND_INITIAL_RECEIPTS'):
        fit.inspect_cold(fit.ref(fit.COLD_PLAN)['sha256'], admitted=False)
    assert not fit.COLD_ROOT.exists()


def test_cold_fit_requires_boot_after_663(monkeypatch):
    policy = _closed_policy(monkeypatch)
    for key in ('training_enabled', 'native_training_enabled',
                'full_epoch_training_allowed', 'native_launch_authorized'):
        policy[key] = True
    policy[fit.COLD_STAGE].update(status='ADMITTED_ONE_USE',
        plan=fit.ref(fit.COLD_PLAN), remaining_invocations=1,
        execution_authorized=True, training_authorized=True,
        relaunch_allowed=False, max_optimizer_updates=1849, model_forwards=4037)
    monkeypatch.setattr(fit, 'physical_windows_boot_id', lambda: 663)
    with pytest.raises(RuntimeError, match='COLD_ONE_USE_ADMISSION'):
        fit.inspect_cold(fit.ref(fit.COLD_PLAN)['sha256'], admitted=True)
    assert not fit.COLD_ROOT.exists()


def test_cold_cli_preflight_never_runs_model_or_claim(monkeypatch, capsys):
    plan = json.loads(fit.COLD_PLAN.read_text())
    monkeypatch.setattr(fit, 'inspect_cold',
                        lambda plan_sha, *, admitted: (plan, {}) if not admitted else None)
    monkeypatch.setattr(fit, 'physical_windows_boot_id', lambda: 664)
    monkeypatch.setattr(fit, 'run_cold', lambda *_: pytest.fail('cold fit started'))
    monkeypatch.setattr(sys, 'argv', ['fit', '--plan', str(fit.COLD_PLAN),
        '--plan-sha256', fit.ref(fit.COLD_PLAN)['sha256'], '--device', 'cuda',
        '--out-dir', str(fit.COLD_ROOT), '--preflight'])
    fit.main()
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'PREFLIGHT_PASS'
    assert report['physical_windows_boot_id'] == 664
    assert report['fresh_physical_boot'] is True
    assert report['claim_created'] is False and report['model_forwards'] == 0
    assert not fit.COLD_ROOT.exists()


def test_warm_cli_rejects_preflight(monkeypatch):
    monkeypatch.setattr(fit, 'run', lambda *_: pytest.fail('warm fit started'))
    monkeypatch.setattr(sys, 'argv', ['fit', '--plan', str(fit.PLAN),
        '--plan-sha256', fit.ref(fit.PLAN)['sha256'], '--device', 'cuda',
        '--out-dir', str(fit.ROOT), '--preflight'])
    with pytest.raises(RuntimeError, match='CLI_SCOPE'):
        fit.main()


def test_wrapper_matches_only_exact_cold_preflight_and_both_fit_paths():
    script = (fit.REPO / 'scripts/gx1_capped_run.sh').read_text()
    begin = script.index('matches_bound_market_state_ta_m5_decision_fit() {')
    end = script.index('\n}', begin) + 2
    function = script[begin:end]
    shell = ('REPO_ROOT=/home/andre2/src/GX1_CURRENT\n'
        'CANONICAL_TRAINER_PYTHON=/home/andre2/src/GX1_CURRENT/.venv/bin/python\n'
        'TA_M5_DECISION_FIT_TRAINER_MODULE=gx1.audit.run_market_state_ta_m5_decision_fit_v1\n'
        + function + '\n' + 'matches_bound_market_state_ta_m5_decision_fit "$@"')
    def matches(plan, root, extra=()):
        args = [str(fit.REPO / '.venv/bin/python'), '-m',
            'gx1.audit.run_market_state_ta_m5_decision_fit_v1', '--plan', str(plan),
            '--plan-sha256', fit.ref(plan)['sha256'], '--device', 'cuda',
            '--out-dir', str(root), *extra]
        return subprocess.run(['bash', '-c', shell, 'bash', str(plan), str(root), *args],
            cwd=fit.REPO, capture_output=True).returncode == 0
    assert matches(fit.PLAN, fit.ROOT)
    assert matches(fit.COLD_PLAN, fit.COLD_ROOT)
    assert matches(fit.COLD_PLAN, fit.COLD_ROOT, ('--preflight',))
    assert not matches(fit.PLAN, fit.ROOT, ('--preflight',))
    assert not matches(fit.COLD_PLAN, fit.COLD_ROOT, ('--unexpected',))
    assert not matches(fit.COLD_PLAN, fit.ROOT, ('--preflight',))
