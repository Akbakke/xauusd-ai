from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE = 'gx1.scripts.validate_entry_model_native_technical_checkpoint_v1'
FLAG = '--market-state-ta-direction-cache-plan'


def test_exact_250w_selector_rejects_changed_scope(tmp_path):
    source = (REPO / 'scripts/gx1_capped_run.sh').read_text()
    name = 'matches_bound_market_state_ta_direction_cache'
    body = source.split(name + '() {\n', 1)[1].split('\n}\n', 1)[0]
    shell = name + '() {\n' + body + '\n}\n' + name + ' "$@"'
    plan = tmp_path / 'plan.json'
    plan.write_text('{"synthetic":true}')
    output = str(tmp_path / 'output')
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    python = '/canonical/python'
    target = [python, '-m', MODULE, FLAG, str(plan), FLAG + '-sha256',
              digest, '--device', 'cuda', '--out-dir', output]
    env = {**os.environ, 'CANONICAL_TRAINER_PYTHON': python,
           'TECHNICAL_VALIDATION_PRODUCER_MODULE': MODULE}

    def invoke(args, expected_plan=str(plan)):
        return subprocess.run(['bash', '-c', shell, 'selector', expected_plan, output,
                               *args], env=env, capture_output=True).returncode

    assert invoke(target) == 0
    for index, replacement in ((2, 'other.module'), (6, '0' * 64),
                               (8, 'cpu'), (10, output + '-other')):
        changed = target.copy()
        changed[index] = replacement
        assert invoke(changed) != 0
    assert invoke(target + ['--device', 'cuda']) != 0
    link = tmp_path / 'link.json'
    link.symlink_to(plan)
    changed = target.copy()
    changed[4] = str(link)
    assert invoke(changed, str(link)) != 0
    plan.write_text('{"synthetic":false}')
    assert invoke(target) != 0


def test_dispatch_preflight_and_run_are_distinct(monkeypatch, tmp_path):
    from gx1.audit import cache_market_state_ta_direction_v1 as wrapper
    from gx1.scripts import validate_entry_model_native_technical_checkpoint_v1 as validator

    calls = []
    monkeypatch.setattr(wrapper, 'preflight_read_only',
                        lambda **kwargs: calls.append(('preflight', kwargs)) or {'status': 'SYNTHETIC'})
    monkeypatch.setattr(wrapper, 'run',
                        lambda **kwargs: calls.append(('run', kwargs)) or {'status': 'SYNTHETIC'})
    args = [FLAG, str(tmp_path / 'plan.json'), FLAG + '-sha256', 'a' * 64,
            '--device', 'cuda', '--out-dir', str(tmp_path / 'output')]
    assert validator.main(args + ['--preflight']) == 0
    assert validator.main(args) == 0
    assert [name for name, _ in calls] == ['preflight', 'run']
    assert calls[1][1]['device'] == 'cuda'
    with pytest.raises(SystemExit):
        validator.main(args + ['--device', 'cuda'])
    with pytest.raises(SystemExit):
        validator.main(args[:4] + ['--device', 'cpu', '--out-dir', str(tmp_path)])

