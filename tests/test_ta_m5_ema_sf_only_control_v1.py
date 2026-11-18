"""Exact one-use EMA S/F-control launch boundaries."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gx1.audit import run_ta_m5_ema_sf_only_control_v1 as control

REPO = Path('/home/andre2/src/GX1_CURRENT')
MODULE = 'gx1.audit.run_ta_m5_ema_sf_only_control_v1'
OUTPUT = '/home/andre2/GX1_RUNS/TA_M5_EMA_SF_ONLY_CONTROL_20261118_001'


def test_capped_route_accepts_only_bound_plan_and_preflight(tmp_path):
    script = (REPO / 'scripts/gx1_capped_run.sh').read_text()
    begin = script.index('matches_bound_ta_m5_ema_sf_only_control() {')
    end = script.index('\n}', begin) + 2
    function = script[begin:end]
    repo = tmp_path / 'repo'
    plan = repo / 'configs/research/TA_M5_EMA_SF_ONLY_CONTROL_20261118_001.json'
    plan.parent.mkdir(parents=True)
    plan.write_text('{"scope":"route-fixture"}\n')
    python = str(tmp_path / 'trainer-python')
    shell = (f'REPO_ROOT={repo}\nCANONICAL_TRAINER_PYTHON={python}\n'
             f'TA_M5_EMA_SF_ONLY_CONTROL_TRAINER_MODULE={MODULE}\n'
             + function + '\n' + 'matches_bound_ta_m5_ema_sf_only_control "$@"')
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()

    def matches(*, sha=digest, out=OUTPUT, module=MODULE, device='cuda', extra=()):
        args = [python, '-m', module, '--plan', str(plan), '--plan-sha256', sha,
                '--device', device, '--out-dir', out, *extra]
        return subprocess.run(['bash', '-c', shell, 'bash', *args],
                              capture_output=True).returncode == 0

    assert matches()
    assert matches(extra=('--preflight',))
    assert not matches(sha='0' * 64)
    assert not matches(out=OUTPUT + '_OTHER')
    assert not matches(module=MODULE + '_OTHER')
    assert not matches(device='cpu')
    assert not matches(extra=('--unexpected',))
    real = tmp_path / 'real-plan.json'
    plan.replace(real)
    plan.symlink_to(real)
    assert not matches()


def test_capped_route_retains_guard_and_single_fit_cap():
    script = (REPO / 'scripts/gx1_capped_run.sh').read_text()
    begin = script.index('  if [[ "${3:-}" == "$TA_M5_EMA_SF_ONLY_CONTROL_TRAINER_MODULE" ]]; then')
    end = script.index('\n  fi', begin) + 5
    branch = script[begin:end]
    assert 'matches_bound_ta_m5_ema_sf_only_control "$@"' in branch
    assert 'TRAINER_MAX_WALL_SECONDS=10800' in branch
    assert 'TRAINER_MODEL_MAX_WALL_SECONDS=10800' in branch
    assert 'TRAINER_GPU_MAX_POWER_LIMIT_W=300' in branch
    assert 'TRAINER_GPU_MAX_POWER_DRAW_W=310' in branch
    assert 'TRAINER_GPU_MAX_CORE_TEMP_C=85' in branch
    assert f'TRAINER_OUT_BUNDLE_DIR={OUTPUT}' in branch


def test_fixed_h480_side_panel_excludes_incomplete_and_preserves_long_reference():
    n = 4
    capture = {'parent_rows': np.arange(n, dtype=np.int64),
               'native_F_raw': np.zeros((n, 5, 2), dtype=np.float32)}
    capture['native_F_raw'][:, 3, 1] = [1, -1, -1, 0]
    path = np.zeros((n, 6), dtype=bool)
    path[[0, 1, 3], 5] = True
    net = np.full((n, 6, 2), np.nan, dtype=np.float32)
    mae = np.full_like(net, np.nan)
    net[0, 5] = [10, 20]
    net[1, 5] = [-5, 5]
    net[3, 5] = [3, 4]
    mae[[0, 1, 3], 5] = 0
    report = control.h480_sign_panel(capture, {
        'path_observed': path, 'net_bps': net,
        'pre_mfe_mae_upper_bps': mae})
    assert report['complete_path_rows'] == 3
    assert report['F_H4_sign_short_count'] == 1
    assert report['F_H4_sign_quality_bps_mean'] == pytest.approx(6)
    assert report['always_LONG_quality_bps_mean'] == pytest.approx(8/3)
    assert report['F_H4_sign_minus_always_LONG_bps_mean'] == pytest.approx(10/3)
    assert report['role'] == 'overlapping_h480_markout_not_trading_PnL'
    assert capture['h480_complete'].tolist() == [True, True, False, True]


def test_retired_control_plan_refuses_before_input_reads(monkeypatch):
    def unexpected_io(*args, **kwargs):
        pytest.fail('retired control opened an input')

    monkeypatch.setattr(control, 'ref', unexpected_io)
    monkeypatch.setattr(control, 'read', unexpected_io)
    monkeypatch.setattr(control, 'initial_core', unexpected_io)
    with pytest.raises(RuntimeError, match='DIRECTION_M1_PROVENANCE_RETIRED'):
        control.build_plan()


def test_retired_cli_preflight_refuses_before_model_or_claim(monkeypatch, tmp_path):
    root = tmp_path / 'no-claim'
    path = tmp_path / 'retired-plan.json'
    path.write_text(json.dumps({'source_base_commit': 'retired-fixture'}) + '\n')
    monkeypatch.setattr(control, 'ROOT', root)
    monkeypatch.setattr(control, 'PLAN', path)
    original_read = control.read

    def fixture_only(binding):
        assert binding['path'] == str(path)
        return original_read(binding)

    def unexpected_model_or_claim(*args, **kwargs):
        pytest.fail('retired control constructed a model or created a claim')

    monkeypatch.setattr(control, 'read', fixture_only)
    monkeypatch.setattr(control, 'initial_core', unexpected_model_or_claim)
    monkeypatch.setattr(control.decision, 'publish', unexpected_model_or_claim)
    monkeypatch.setattr(control, 'run', unexpected_model_or_claim)
    monkeypatch.setattr(sys, 'argv', ['control', '--plan', str(path),
        '--plan-sha256', control.ref(path)['sha256'], '--device', 'cuda',
        '--out-dir', str(root), '--preflight'])

    with pytest.raises(RuntimeError, match='DIRECTION_M1_PROVENANCE_RETIRED'):
        control.main()

    assert not root.exists()
