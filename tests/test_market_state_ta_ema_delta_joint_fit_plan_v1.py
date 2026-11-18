"""Retired EMA launch refuses M1 provenance before model or input work."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gx1.audit import run_market_state_ta_m5_decision_fit_v1 as fit


@pytest.mark.parametrize('return_stats', (False, True))
def test_retired_ema_plan_refuses_before_input_reads(monkeypatch, return_stats):
    def unexpected_io(*args, **kwargs):
        pytest.fail('retired EMA plan opened an input')

    monkeypatch.setattr(fit, 'ref', unexpected_io)
    monkeypatch.setattr(fit, 'read', unexpected_io)
    monkeypatch.setattr(fit, 'load_targets', unexpected_io)
    with pytest.raises(RuntimeError, match='DIRECTION_M1_PROVENANCE_RETIRED'):
        fit.build_ema_plan(return_stats=return_stats)


def test_retired_ema_cli_preflight_refuses_before_model_or_claim(monkeypatch, tmp_path):
    root = tmp_path / 'no-claim'
    path = tmp_path / 'retired-ema-plan.json'
    path.write_text(json.dumps({'source_base_commit': 'retired-fixture'}) + '\n')
    monkeypatch.setattr(fit, 'EMA_ROOT', root)
    monkeypatch.setattr(fit, 'EMA_PLAN', path)
    original_read = fit.read

    def fixture_only(binding):
        assert binding['path'] == str(path)
        return original_read(binding)

    def unexpected_model_or_claim(*args, **kwargs):
        pytest.fail('retired EMA fit constructed a model or created a claim')

    monkeypatch.setattr(fit, 'read', fixture_only)
    monkeypatch.setattr(fit, 'load_targets', unexpected_model_or_claim)
    monkeypatch.setattr(fit, 'publish', unexpected_model_or_claim)
    monkeypatch.setattr(fit, 'run_ema', unexpected_model_or_claim)
    monkeypatch.setattr(sys, 'argv', ['fit', '--plan', str(path),
        '--plan-sha256', fit.ref(path)['sha256'], '--device', 'cuda',
        '--out-dir', str(root), '--preflight'])

    with pytest.raises(RuntimeError, match='DIRECTION_M1_PROVENANCE_RETIRED'):
        fit.main()

    assert not root.exists()


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
