"""Real guard shell behavior with fake scope and telemetry owners; no GPU use."""
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import time

import pytest
from tests.test_gx1_capped_run_contract import TRAINER_GUARD, _guard_env, _fake_nvidia_smi


def _fixture(tmp_path, *, target=200, telemetry_limit=200, owner_fault=''):
    repo = tmp_path / 'repo'; scripts = repo / 'scripts'; scripts.mkdir(parents=True)
    guard = scripts / TRAINER_GUARD.name; shutil.copyfile(TRAINER_GUARD, guard)
    python = repo / '.venv/bin/python'; python.parent.mkdir(parents=True)
    calls = tmp_path / 'owner-calls'
    python.write_text('#!/bin/bash\nset -eu\n'
        + 'printf "%s\\n" "$4" >> ' + shlex.quote(str(calls)) + '\n'
        + 'case "$4" in\n'
        + ('claim) exit 75;;\n' if owner_fault == 'claim' else '')
        + ('heartbeat) exit 75;;\n' if owner_fault == 'heartbeat' else '')
        + ('close) exit 75;;\n' if owner_fault == 'close' else '')
        + '*) exit 0;;\nesac\n')
    python.chmod(0o755)
    telemetry = _fake_nvidia_smi(tmp_path, f'50, 60, 150, {telemetry_limit}, 1000')
    env = _guard_env(device='cuda', nvidia_smi_path=telemetry,
                     max_power_limit_w=target, max_power_draw_w=target+10)
    env['GX1_POWER_BENCHMARK_SCOPE_JSON'] = str(tmp_path / 'synthetic-scope.json')
    env['GX1_POWER_BENCHMARK_SCOPE_SHA256'] = 'a' * 64
    return guard, env, calls


@pytest.mark.parametrize('target', [160, 200])
def test_matched_scope_closes_after_normal_child_exit(tmp_path, target):
    guard, env, calls = _fixture(tmp_path, target=target, telemetry_limit=target)
    result = subprocess.run(['bash', str(guard), '/bin/sleep', '2'], env=env,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    actions = calls.read_text().splitlines()
    assert actions[0] == 'claim' and 'heartbeat' in actions and actions[-1] == 'close'


def test_scope_claim_failure_never_launches_or_closes_another_claim(tmp_path):
    guard, env, calls = _fixture(tmp_path, owner_fault='claim')
    marker = tmp_path / 'child-started'
    result = subprocess.run(['bash', str(guard), '/usr/bin/touch', str(marker)],
                            env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 75 and not marker.exists()
    assert calls.read_text().splitlines() == ['claim']


@pytest.mark.parametrize('observed', [160, 201])
def test_exact_treatment_required_before_starting_child(tmp_path, observed):
    guard, env, calls = _fixture(tmp_path, telemetry_limit=observed)
    marker = tmp_path / 'child-started'
    result = subprocess.run(['bash', str(guard), '/usr/bin/touch', str(marker)],
                            env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 75 and not marker.exists()
    assert calls.read_text().splitlines() == ['claim', 'close']


def test_lost_scope_heartbeat_stops_child_then_requests_close(tmp_path):
    guard, env, calls = _fixture(tmp_path, owner_fault='heartbeat')
    pid_file = tmp_path / 'child.pid'
    child = 'echo $$ > ' + shlex.quote(str(pid_file)) + '; exec /bin/sleep 60'
    result = subprocess.run(['bash', str(guard), '/bin/bash', '-c', child],
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 75
    assert 'power_benchmark_authorization_lost' in result.stderr
    assert calls.read_text().splitlines() == ['claim', 'heartbeat', 'close']
    with pytest.raises(ProcessLookupError): os.kill(int(pid_file.read_text()), 0)


def test_sigterm_closes_scope_and_terminates_owned_child(tmp_path):
    guard, env, calls = _fixture(tmp_path)
    pid_file = tmp_path / 'child.pid'
    command = 'echo $$ > ' + shlex.quote(str(pid_file)) + '; exec /bin/sleep 60'
    process = subprocess.Popen(['bash', str(guard), '/bin/bash', '-c', command],
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline: time.sleep(0.02)
        assert pid_file.exists()
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 130, stderr
        assert calls.read_text().splitlines()[-1] == 'close'
        with pytest.raises(ProcessLookupError): os.kill(int(pid_file.read_text()), 0)
    finally:
        if process.poll() is None:
            process.kill(); process.communicate(timeout=5)


def test_failed_close_is_not_reported_as_success(tmp_path):
    guard, env, calls = _fixture(tmp_path, owner_fault='close')
    result = subprocess.run(['bash', str(guard), '/bin/true'], env=env,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert calls.read_text().splitlines()[-1] == 'close'
