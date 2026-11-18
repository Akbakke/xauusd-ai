"""The new M5 fit can only consume the sealed EMA view and fresh initial."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gx1.audit import run_market_state_ta_m5_decision_fit_v1 as fit


def test_ema_plan_seals_current_fit_and_initial_receipts():
    plan = fit.build_ema_plan()
    assert plan['status'] == 'PREPARED_NOT_ADMITTED'
    assert (plan['fit_rows'], plan['excluded_gap_rows'], plan['score_rows']) == (
        29574, 275, 17489)
    assert (plan['batch_visits'], plan['initial_score_forwards'],
            plan['final_score_forwards'], plan['model_forwards']) == (1849, 1094, 1094, 4037)
    assert plan['fit_statistics_sha256'] == fit.EMA_FIT_STATS_SHA
    assert plan['initial_model_state_sha256'] == fit.EMA_INITIAL_SHA
    assert plan['core_contract_sha256'] == fit.EMA_CORE_SHA
    assert plan['decision_contract_sha256'] == fit.EMA_DECISION_SHA
    assert plan['inputs']['initial_result']['sha256'] == (
        '8fe95ca667c1ba814a6c0b9785218891cf9bae5e4e0d2fb01cbddbb2a3972adb')
    assert plan['inputs']['prior_result']['sha256'] == (
        '55a9cdd4a54cf612434f6e3ce8a650b49f8514fc04b26f7f0decfceafde28267')


def test_ema_plan_rejects_altered_initial_state_before_claim(monkeypatch, tmp_path):
    monkeypatch.setattr(fit, 'EMA_ROOT', tmp_path / 'no-claim')
    original = fit.read
    def altered(binding):
        value = original(binding)
        if binding['path'] == str(fit.EMA_INITIAL_ROOT / 'RESULT.json'):
            value = copy.deepcopy(value)
            value['initial_model_state_sha256'] = '0' * 64
        return value
    monkeypatch.setattr(fit, 'read', altered)
    with pytest.raises(RuntimeError, match='EMA_INITIAL_RECEIPTS'):
        fit.build_ema_plan()
    assert not fit.EMA_ROOT.exists()


def _isolated_plan(monkeypatch, tmp_path):
    monkeypatch.setattr(fit, 'EMA_ROOT', tmp_path / 'no-claim')
    plan = fit.build_ema_plan()
    path = tmp_path / 'TA_M5_EMA_DELTA_JOINT_FIT_20261118_001.json'
    path.write_text(json.dumps(plan, sort_keys=True) + '\n')
    monkeypatch.setattr(fit, 'EMA_PLAN', path)
    monkeypatch.setattr(fit, 'source_clean', lambda _plan, _sources: True)
    policy = json.loads((fit.REPO / 'NEXT_RUN_POLICY.json').read_text())
    policy[fit.EMA_STAGE] = {'status': 'PREPARED_NOT_ADMITTED',
        'plan': fit.ref(path), 'remaining_invocations': 0,
        'test_data_used': False, 'test_authorized': False,
        'online_authorized': False, 'broker_access_allowed': False,
        'execution_authorized': False, 'training_authorized': False}
    monkeypatch.setattr(fit, 'json', SimpleNamespace(loads=lambda _text: policy,
        dumps=json.dumps))
    return plan, path, policy


def test_ema_preflight_closed_and_forged_admission_needs_new_boot(monkeypatch, tmp_path):
    plan, path, policy = _isolated_plan(monkeypatch, tmp_path)
    actual, stats = fit.inspect_ema(fit.ref(path)['sha256'], admitted=False)
    assert actual == plan and stats['sha256'] == fit.EMA_FIT_STATS_SHA
    assert not fit.EMA_ROOT.exists()
    for key in ('training_enabled', 'native_training_enabled',
                'full_epoch_training_allowed', 'native_launch_authorized'):
        policy[key] = True
    policy[fit.EMA_STAGE].update(status='ADMITTED_ONE_USE',
        remaining_invocations=1, max_invocations=1,
        claimed_invocations=0, completed_invocations=0,
        execution_authorized=True, training_authorized=True,
        relaunch_allowed=False, max_optimizer_updates=1849,
        model_forwards=4037)
    monkeypatch.setattr(fit, 'physical_windows_boot_id', lambda: 664)
    with pytest.raises(RuntimeError, match='EMA_ONE_USE_ADMISSION'):
        fit.inspect_ema(fit.ref(path)['sha256'], admitted=True)
    monkeypatch.setattr(fit, 'physical_windows_boot_id', lambda: 665)
    accepted, admitted_stats = fit.inspect_ema(fit.ref(path)['sha256'], admitted=True)
    assert accepted == plan and admitted_stats['sha256'] == fit.EMA_FIT_STATS_SHA
    assert not fit.EMA_ROOT.exists()


def test_ema_cli_preflight_never_claims_or_constructs_model(monkeypatch, tmp_path, capsys):
    plan, path, _ = _isolated_plan(monkeypatch, tmp_path)
    monkeypatch.setattr(fit, 'inspect_ema', lambda _sha, *, admitted: (plan, {}))
    monkeypatch.setattr(fit, 'physical_windows_boot_id', lambda: 665)
    monkeypatch.setattr(fit, 'run_ema', lambda _sha: pytest.fail('fit started'))
    monkeypatch.setattr(sys, 'argv', ['fit', '--plan', str(path),
        '--plan-sha256', fit.ref(path)['sha256'], '--device', 'cuda',
        '--out-dir', str(fit.EMA_ROOT), '--preflight'])
    fit.main()
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'PREFLIGHT_PASS'
    assert report['fresh_physical_boot'] is True
    assert report['claim_created'] is False
    assert report['model_forwards'] == report['optimizer_steps'] == 0
    assert not fit.EMA_ROOT.exists()


def test_wrapper_accepts_only_exact_ema_plan_and_output(tmp_path):
    script = (fit.REPO / 'scripts/gx1_capped_run.sh').read_text()
    begin = script.index('matches_bound_market_state_ta_m5_decision_fit() {')
    end = script.index('\n}', begin) + 2
    function = script[begin:end]
    repo = tmp_path / 'repo'
    path = repo / 'configs/research/TA_M5_EMA_DELTA_JOINT_FIT_20261118_001.json'
    path.parent.mkdir(parents=True)
    path.write_text('{"scope":"test-only"}\n')
    python = str(tmp_path / 'trainer-python')
    output = '/home/andre2/GX1_RUNS/TA_M5_EMA_DELTA_JOINT_FIT_20261118_001'
    shell = (f'REPO_ROOT={repo}\nCANONICAL_TRAINER_PYTHON={python}\n'
        'TA_M5_DECISION_FIT_TRAINER_MODULE=gx1.audit.run_market_state_ta_m5_decision_fit_v1\n'
        + function + '\n' + 'matches_bound_market_state_ta_m5_decision_fit "$@"')
    digest = fit.ref(path)['sha256']
    def matches(root, sha=digest, extra=()):
        args = [python, '-m', 'gx1.audit.run_market_state_ta_m5_decision_fit_v1',
            '--plan', str(path), '--plan-sha256', sha, '--device', 'cuda',
            '--out-dir', root, *extra]
        return subprocess.run(['bash', '-c', shell, 'bash', str(path), root, *args],
            capture_output=True).returncode == 0
    assert matches(output)
    assert matches(output, extra=('--preflight',))
    assert not matches(output, sha='0' * 64)
    assert not matches(output + '_OTHER')
    assert not matches(output, extra=('--unexpected',))
