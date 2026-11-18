"""Exact one-use EMA S/F-control launch boundaries."""
from __future__ import annotations

import copy
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


def _isolated_plan(monkeypatch, tmp_path):
    root = tmp_path / 'no-claim'
    monkeypatch.setattr(control, 'ROOT', root)
    plan = control.build_plan()
    assert (plan['fit_rows'], plan['later_rows'], plan['batch_visits'],
            plan['fit_score_forwards'], plan['later_score_forwards'],
            plan['model_forwards']) == (29574, 17489, 1849, 1849, 1094, 4792)
    assert plan['inputs'] == control.successor_input_refs()
    assert plan['sf_objective_sha256']
    assert plan['source_owners']['scripts/gx1_capped_run.sh'] == hashlib.sha256(
        (REPO / 'scripts/gx1_capped_run.sh').read_bytes()).hexdigest()
    plan_path = tmp_path / 'TA_M5_EMA_SF_ONLY_CONTROL_20261118_001.json'
    plan_path.write_text(json.dumps(plan, sort_keys=True) + '\n')
    monkeypatch.setattr(control, 'PLAN', plan_path)
    monkeypatch.setattr(control, 'build_plan', lambda *, source_base_commit=None: plan)
    monkeypatch.setattr(control, 'source_clean', lambda _plan: True)
    policy = json.loads((REPO / 'NEXT_RUN_POLICY.json').read_text())
    policy[control.STAGE] = {'status': 'PREPARED_NOT_ADMITTED',
        'plan': control.ref(plan_path), 'output_root': str(root),
        'fit_rows': 29574, 'later_rows': 17489,
        'max_optimizer_updates': 1849, 'model_forwards': 4792,
        'max_invocations': 1, 'remaining_invocations': 0,
        'claimed_invocations': 0, 'completed_invocations': 0,
        'execution_authorized': False, 'training_authorized': False,
        'test_authorized': False, 'online_authorized': False,
        'broker_access_allowed': False, 'test_data_used': False,
        'relaunch_allowed': False}
    original_read = control.read
    def read(ref):
        if ref['path'] == str(REPO / 'NEXT_RUN_POLICY.json'):
            return policy
        return original_read(ref)
    monkeypatch.setattr(control, 'read', read)
    return plan, plan_path, root, policy


def test_prepared_admission_requires_fresh_boot_and_unclaimed_root(monkeypatch, tmp_path):
    plan, path, root, policy = _isolated_plan(monkeypatch, tmp_path)
    actual, _ = control.inspect(control.ref(path)['sha256'], admitted=False)
    assert actual == plan and not root.exists()
    for key in ('training_enabled', 'native_training_enabled',
                'full_epoch_training_allowed', 'native_launch_authorized'):
        policy[key] = True
    policy[control.STAGE].update(status='ADMITTED_ONE_USE',
        execution_authorized=True, training_authorized=True,
        remaining_invocations=1)
    monkeypatch.setattr(control.paired, 'physical_windows_boot_id', lambda: 665)
    with pytest.raises(RuntimeError, match='ONE_USE_ADMISSION'):
        control.inspect(control.ref(path)['sha256'], admitted=True)
    monkeypatch.setattr(control.paired, 'physical_windows_boot_id', lambda: 666)
    accepted, _ = control.inspect(control.ref(path)['sha256'], admitted=True)
    assert accepted == plan and not root.exists()
    root.mkdir()
    with pytest.raises(RuntimeError, match='PLAN_OUTPUT'):
        control.inspect(control.ref(path)['sha256'], admitted=True)


def test_preflight_does_not_claim_or_forward(monkeypatch, tmp_path, capsys):
    plan, path, root, _ = _isolated_plan(monkeypatch, tmp_path)
    monkeypatch.setattr(control, 'inspect', lambda _sha, *, admitted: (plan, control.ref(path)))
    monkeypatch.setattr(control, 'successor_input_refs', lambda: {'fit_targets': {}})
    monkeypatch.setattr(control, 'prepare_ema', lambda *, device, inputs: {'release': lambda: None})
    monkeypatch.setattr(control.decision, 'load_targets', lambda *_args: object())
    monkeypatch.setattr(control.decision, 'target_slice', lambda *_args: object())
    monkeypatch.setattr(control, 'initial_core', lambda *_args, **_kwargs: (None, plan['original_initial_core_sha256']))
    monkeypatch.setattr(control.paired, 'physical_windows_boot_id', lambda: 665)
    monkeypatch.setattr(control, 'run', lambda _sha: pytest.fail('trainer started'))
    monkeypatch.setattr(sys, 'argv', ['control', '--plan', str(path),
        '--plan-sha256', control.ref(path)['sha256'], '--device', 'cuda',
        '--out-dir', str(root), '--preflight'])
    control.main()
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'PREFLIGHT_PASS'
    assert report['claim_created'] is False
    assert report['model_forwards'] == report['optimizer_updates'] == 0
    assert report['output_root_absent'] is True and not root.exists()


@pytest.mark.parametrize('binding', ('source', 'input'))
def test_inspect_rejects_forged_binding_before_claim(monkeypatch, tmp_path, binding):
    plan, path, root, policy = _isolated_plan(monkeypatch, tmp_path)
    forged = copy.deepcopy(plan)
    if binding == 'source':
        forged['source_owners']['scripts/gx1_capped_run.sh'] = '0' * 64
    else:
        forged['inputs']['fit_targets']['sha256'] = '0' * 64
    path.write_text(json.dumps(forged, sort_keys=True) + '\n')
    policy[control.STAGE]['plan'] = control.ref(path)
    with pytest.raises(RuntimeError, match='FIXED_SCOPE_SOURCE'):
        control.inspect(control.ref(path)['sha256'], admitted=False)
    assert not root.exists()
