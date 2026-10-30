import copy
import hashlib
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from gx1.audit import direction_gc_q1_freshness_v1 as audit

REPO = Path(__file__).resolve().parents[1]
STEP = 300_000_000_000


def test_previous_rows_are_exact_and_missing_clocks_are_unchanged():
    clocks = np.array([2, 3, 5, 6, 9], dtype=np.int64) * STEP
    rows = np.array([1, 2, 3, 4], dtype=np.int64)
    prior, valid = audit.previous_rows(clocks, rows)
    assert prior.tolist() == [0, 2, 2, 4]
    assert valid.tolist() == [True, False, True, False]
    with pytest.raises(RuntimeError, match='DECISION_CLOCK'):
        audit.previous_rows(clocks[::-1].copy(), rows)
    with pytest.raises(RuntimeError, match='ROWS'):
        audit.previous_rows(clocks, np.array([len(clocks)], dtype=np.int64))


def test_source_transition_accepts_only_added_q1_scope():
    key = 'scripts/gx1_capped_run.sh'
    historical = subprocess.check_output(['git', 'show', audit.FIT_COMMIT + ':' + key], cwd=REPO)
    current = (REPO / key).read_bytes()
    old = {'path': '/repo/' + key, 'sha256': hashlib.sha256(historical).hexdigest()}
    new = {'path': old['path'], 'sha256': hashlib.sha256(current).hexdigest()}
    owners = {key: old, **{f'owner_{i}': {'path': f'/repo/{i}', 'sha256': '0' * 64} for i in range(63)}}
    launch = {'source_owners': owners}; plan = {'sources': {key: new}}
    frozen = audit.fit_source_transition(launch, plan, historical, current)
    assert frozen['source_owners'][key] == new and launch['source_owners'][key] == old
    tampered = current + b'\n# unrelated change\n'
    bad = copy.deepcopy(plan); bad['sources'][key]['sha256'] = hashlib.sha256(tampered).hexdigest()
    with pytest.raises(RuntimeError, match='FIT_RUNNER_UNRELATED_CHANGE'):
        audit.fit_source_transition(launch, bad, historical, tampered)
    with pytest.raises(RuntimeError, match='FIT_RUNNER_HASH'):
        audit.fit_source_transition(launch, plan, historical + b'changed', current)


def test_exact_power_selector_rejects_changed_or_extra_arguments(tmp_path):
    source = (REPO / 'scripts/gx1_capped_run.sh').read_text()
    body = source.split('matches_bound_gc_q1_freshness() {\n', 1)[1].split('\n}\n', 1)[0]
    shell = 'matches_bound_gc_q1_freshness() {\n' + body + '\n}\nmatches_bound_gc_q1_freshness "$@"'
    plan = tmp_path / 'plan.json'; plan.write_text('{"synthetic":true}')
    output = str(tmp_path / 'output'); digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    python = '/canonical/python'; module = 'gx1.scripts.validate_entry_model_native_technical_checkpoint_v1'
    target = [python, '-m', module, '--direction-gc-q1-freshness-plan', str(plan),
              '--direction-gc-q1-freshness-plan-sha256', digest, '--device', 'cuda', '--out-dir', output]
    env = {**os.environ, 'CANONICAL_TRAINER_PYTHON': python, 'TECHNICAL_VALIDATION_PRODUCER_MODULE': module}
    def invoke(args, expected_plan=str(plan)):
        return subprocess.run(['bash', '-c', shell, 'selector', expected_plan, output, *args], env=env, capture_output=True).returncode
    assert invoke(target) == 0
    changed = target.copy(); changed[6] = '0' * 64; assert invoke(changed) != 0
    changed = target.copy(); changed[2] = 'unrelated.module'; assert invoke(changed) != 0
    changed = target.copy(); changed[-1] += '-other'; assert invoke(changed) != 0
    assert invoke(target + ['--device', 'cuda']) != 0
    link = tmp_path / 'link.json'; link.symlink_to(plan); changed = target.copy(); changed[4] = str(link)
    assert invoke(changed, str(link)) != 0
    plan.write_text('{"synthetic":false}'); assert invoke(target) != 0


def test_q1_dispatch_requires_cuda_and_exact_flags(monkeypatch, tmp_path):
    from gx1.scripts import validate_entry_model_native_technical_checkpoint_v1 as validator
    calls = []
    monkeypatch.setattr(audit, 'run', lambda **kwargs: calls.append(kwargs) or {'status': 'SYNTHETIC'})
    args = ['--direction-gc-q1-freshness-plan', str(tmp_path / 'plan.json'),
            '--direction-gc-q1-freshness-plan-sha256', 'a' * 64, '--device', 'cuda', '--out-dir', str(tmp_path)]
    assert validator.main(args) == 0 and len(calls) == 1 and calls[0]['device'] == 'cuda'
    with pytest.raises(SystemExit): validator.main(args + ['--device', 'cuda'])
    with pytest.raises(SystemExit): validator.main(args[:4] + ['--device', 'cpu', '--out-dir', str(tmp_path)])
