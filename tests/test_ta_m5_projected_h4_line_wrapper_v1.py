"""Exact capped route for the one read-only projected H4 line audit."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE = 'gx1.audit.measure_ta_m5_projected_h4_line_v1'
PLAN_NAME = 'TA_M5_PROJECTED_H4_LINE_20261118_001.json'
OUTPUT = '/home/andre2/GX1_RUNS/TA_M5_PROJECTED_H4_LINE_20261118_001'


def match(tmp_path, *, change=None, extra=(), symlink=False):
    source = (REPO / 'scripts/gx1_capped_run.sh').read_text()
    head = 'matches_bound_ta_m5_projected_h4_line() {'
    start = source.index(head)
    matcher = source[start:source.index('\n}\n', start) + 2]
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
    argv = ['/canonical/python', '-m', MODULE, '--plan', str(plan),
            '--plan-sha256', digest, '--out-dir', OUTPUT]
    if change is not None:
        index, value = change
        argv[index] = value
    env = {**os.environ, 'REPO_ROOT': str(repo),
           'CANONICAL_TRAINER_PYTHON': '/canonical/python',
           'TA_M5_PROJECTED_H4_LINE_AUDIT_MODULE': MODULE}
    return subprocess.run(['bash', '-c', matcher + '\n'
                           'matches_bound_ta_m5_projected_h4_line "$@"',
                           'h4-line-match', *argv, *extra],
                          env=env, capture_output=True, text=True, timeout=10)


@pytest.mark.parametrize('extra', [(), ('--preflight',)])
def test_exact_bound_route_accepts_only_fixed_plan_and_output(tmp_path, extra):
    assert match(tmp_path, extra=extra).returncode == 0


@pytest.mark.parametrize('change,extra,symlink', [
    ((2, 'gx1.audit.unrelated'), (), False),
    ((4, '/tmp/unbound.json'), (), False),
    ((6, '0' * 64), (), False),
    ((8, OUTPUT + '_OTHER'), (), False),
    (None, ('--train',), False),
    (None, ('--preflight', '--preflight'), False),
    (None, (), True),
])
def test_route_rejects_unbound_or_symlinked_plan(tmp_path, change, extra, symlink):
    assert match(tmp_path, change=change, extra=extra, symlink=symlink).returncode != 0


def test_audit_route_has_no_cuda_or_training_and_hard_wall_cap():
    source = (REPO / 'scripts/gx1_capped_run.sh').read_text()
    assert 'TA_M5_PROJECTED_H4_LINE_AUDIT_MODULE=gx1.audit.measure_ta_m5_projected_h4_line_v1' in source
    assert '[[ "$JOB_CLASS" != audit || "$ATTENDED_SMOKE" != false' in source
    assert '"$CUDA_PRODUCER_GUARD" != false || "$MEM" != 4G || "$SWAP" != 512M' in source
    assert 'H4_LINE_AUDIT_GUARD=true' in source
    assert '/usr/bin/timeout --signal=TERM --kill-after=10s 1800s "$@"' in source


def test_full_wrapper_rejects_forged_plan_before_scope():
    plan = REPO / 'configs/research' / PLAN_NAME
    argv = ['bash', str(REPO / 'scripts/gx1_capped_run.sh'),
            '--class', 'audit', '--mem', '4G', '--swap', '512M', '--',
            str(REPO / '.venv/bin/python'), '-m', MODULE,
            '--plan', str(plan), '--plan-sha256', '0' * 64,
            '--out-dir', OUTPUT, '--preflight']
    result = subprocess.run(argv, cwd=REPO, capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == 75
    assert 'projected H4 line audit is not exactly bound' in result.stderr
    assert 'capped_run_scope_verified' not in result.stderr
