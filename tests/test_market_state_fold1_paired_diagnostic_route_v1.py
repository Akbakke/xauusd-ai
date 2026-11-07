from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE = 'gx1.scripts.validate_entry_model_native_technical_checkpoint_v1'
FLAG = '--market-state-fold1-paired-diagnostic-plan'


def test_exact_250w_selector_rejects_changed_scope(tmp_path):
    source = (REPO / 'scripts/gx1_capped_run.sh').read_text()
    name = 'matches_bound_market_state_fold1_paired_diagnostic'
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
    from gx1.audit import run_market_state_fold1_paired_diagnostic_v1 as wrapper
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


def test_post_root_fsync_failure_gets_terminal(monkeypatch, tmp_path):
    from gx1.audit import run_market_state_fold1_paired_diagnostic_v1 as wrapper
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import immutable_event_authority_v1 as publication

    output = tmp_path / 'output'
    monkeypatch.setattr(wrapper, 'OUT_ROOT', output)
    monkeypatch.setattr(wrapper, '_qualify', lambda *_: (
        {}, {'path': str(tmp_path / 'plan'), 'sha256': 'a' * 64},
        {'path': str(tmp_path / 'policy'), 'sha256': 'b' * 64}, 'c' * 40, {}))
    monkeypatch.setattr(cap, 'require_guarded_cuda_producer_execution', lambda: {
        'cgroup_path': '/sys/fs/cgroup/test', 'memory_max': 12 * 1024**3,
        'memory_high': 12 * 1024**3, 'swap': 512 * 1024**2})
    monkeypatch.setattr(cap, 'require_capped_lock_ancestry',
                        lambda: {'scope': '/test'})
    monkeypatch.setenv('GX1_TRAINER_GPU_MAX_POWER_LIMIT_W', '250')
    monkeypatch.setenv('GX1_TRAINER_GPU_MAX_POWER_DRAW_W', '170')
    original_fsync = publication._fsync_directory
    calls = 0

    def fail_once(path):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError('synthetic fsync failure')
        return original_fsync(path)

    monkeypatch.setattr(publication, '_fsync_directory', fail_once)
    with pytest.raises(OSError, match='synthetic fsync failure'):
        wrapper.run(tmp_path / 'plan', 'a' * 64, 'cuda', output)
    assert (output / 'FAILED.json').is_file()
    assert (output / 'TERMINAL.json').is_file()
    assert not (output / 'CLAIM.json').exists()
