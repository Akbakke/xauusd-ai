"""Execute the exact final FIT score selector without launching the CUDA job."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE = 'gx1.audit.score_ta_ema_final_fit_v1'
OUTPUT = '/home/andre2/GX1_RUNS/TA_M5_EMA_FINAL_FIT_SCORE_20261118_001'
PLAN_NAME = 'TA_M5_EMA_FINAL_FIT_SCORE_20261118_001.json'


def gate(tmp_path, *, changes=None, extra=(), mem='20G', swap='512M', symlink=False):
    source = (REPO / 'scripts/gx1_capped_run.sh').read_text()
    name = 'matches_bound_ta_ema_final_fit_score() {'
    start = source.index(name)
    matcher = source[start:source.index('\n}\n', start) + 2]
    branch = '      if [[ "${3:-}" == "$TA_EMA_FINAL_FIT_SCORE_PRODUCER_MODULE" ]]; then\n'
    start = source.index(branch)
    end = source.index('      if [[ "${3:-}" == "$TA_CURRENT_NUMERIC_ROUTES_PRODUCER_MODULE" ]]; then', start)
    selector = source[start:end]
    repo = tmp_path / 'repo'
    plan = repo / 'configs/research' / PLAN_NAME
    plan.parent.mkdir(parents=True)
    if symlink:
        actual = tmp_path / 'actual.json'
        actual.write_text('{"test":true}\n')
        plan.symlink_to(actual)
    else:
        plan.write_text('{"test":true}\n')
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    target = ['/canonical/python', '-m', MODULE, '--plan', str(plan),
              '--plan-sha256', digest, '--device', 'cuda', '--out-dir', OUTPUT]
    for index, value in (changes or {}).items():
        target[index] = value
    script = (matcher + '\ninvoke() {\n'
        'NUMERICAL_THREAD_COUNT=1\nTRAINER_GPU_MAX_POWER_LIMIT_W=160\n'
        'TRAINER_MAX_WALL_SECONDS=3600\nTRAINER_MODEL_MAX_WALL_SECONDS=3600\n'
        'TRAINER_GPU_MAX_POWER_DRAW_W=170\nTRAINER_GPU_MAX_CORE_TEMP_C=65\n'
        'TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n' + selector
        + 'printf "%s %s %s %s %s %s %s\\n" "$TRAINER_GPU_MAX_POWER_LIMIT_W" '
        '"$TRAINER_MAX_WALL_SECONDS" "$TRAINER_MODEL_MAX_WALL_SECONDS" '
        '"$TRAINER_GPU_MAX_POWER_DRAW_W" "$TRAINER_GPU_MAX_CORE_TEMP_C" '
        '"$TRAINER_GPU_MAX_MEMORY_TEMP_C" "$NUMERICAL_THREAD_COUNT"\n'
        '}\ninvoke "$@"\n')
    env = {**os.environ, 'REPO_ROOT': str(repo),
           'CANONICAL_TRAINER_PYTHON': '/canonical/python',
           'TA_EMA_FINAL_FIT_SCORE_PRODUCER_MODULE': MODULE,
           'MEM': mem, 'SWAP': swap}
    return subprocess.run(['bash', '-c', script, 'fit-score-gate', *target, *extra],
                          env=env, capture_output=True, text=True, timeout=10)


@pytest.mark.parametrize('extra', [(), ('--preflight',)])
def test_exact_frozen_fit_score_route_keeps_safety_caps(tmp_path, extra):
    result = gate(tmp_path, extra=extra)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '250 3600 3600 170 65 80 8'


@pytest.mark.parametrize('changes,extra,mem,swap,symlink', [
    ({3: '--plan-other'}, (), '20G', '512M', False),
    ({4: '/tmp/other.json'}, (), '20G', '512M', False),
    ({6: '0' * 64}, (), '20G', '512M', False),
    ({8: 'cpu'}, (), '20G', '512M', False),
    ({10: OUTPUT + '_OTHER'}, (), '20G', '512M', False),
    ({}, ('--device', 'cuda'), '20G', '512M', False),
    ({}, ('--preflight', '--preflight'), '20G', '512M', False),
    ({}, (), '4G', '512M', False),
    ({}, (), '20G', '0', False),
    ({}, (), '20G', '512M', True),
])
def test_route_rejects_mismatch_before_launch(tmp_path, changes, extra, mem, swap, symlink):
    result = gate(tmp_path, changes=changes, extra=extra, mem=mem, swap=swap,
                  symlink=symlink)
    assert result.returncode == 75
    assert result.stdout == ''
    assert 'not exactly bound' in result.stderr


def test_route_stays_on_cuda_producer_allowlist():
    source = (REPO / 'scripts/gx1_capped_run.sh').read_text()
    assert '&& [[ "${3:-}" != "$TA_EMA_FINAL_FIT_SCORE_PRODUCER_MODULE" ]]' in source
    assert 'TRAINER_EXECUTION_MODE=cuda_producer' in source
    assert '&& "$3" == "$TA_EMA_FINAL_FIT_SCORE_PRODUCER_MODULE"' in source


def test_full_wrapper_rejects_forged_sha_before_lock_or_cuda_guard():
    plan = REPO / 'configs/research' / PLAN_NAME
    argv = ['bash', str(REPO / 'scripts/gx1_capped_run.sh'),
            '--class', 'producer', '--mem', '20G', '--swap', '512M',
            '--cuda-producer', '--', str(REPO / '.venv/bin/python'),
            '-m', MODULE, '--plan', str(plan), '--plan-sha256', '0' * 64,
            '--device', 'cuda', '--out-dir', OUTPUT, '--preflight']
    result = subprocess.run(argv, cwd=REPO, capture_output=True, text=True,
                            timeout=10)
    assert result.returncode == 75
    assert 'final EMA FIT score producer is not exactly bound' in result.stderr
    assert 'capped_run_scope_verified' not in result.stderr
