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



def test_cache_paths_never_cross_original_and_equal_loss_roots():
    from gx1.audit import cache_market_state_ta_direction_v1 as wrapper
    for stage, root, dedicated in ((wrapper.STAGE, wrapper.ROOT, False),
                                  (wrapper.FINAL_STAGE, wrapper.FINAL_ROOT, True)):
        assert wrapper._cache_scope(root / 'PLAN.json', root) == (stage, root, dedicated)
        other = wrapper.FINAL_ROOT if root == wrapper.ROOT else wrapper.ROOT
        with pytest.raises(RuntimeError, match='EXACT_CACHE_PATHS'):
            wrapper._cache_scope(root / 'PLAN.json', other)
        with pytest.raises(RuntimeError, match='EXACT_CACHE_PATHS'):
            wrapper._cache_scope(root / 'other.json', root)


def test_success_failure_and_partial_outputs_use_selected_root(tmp_path):
    import json
    import numpy as np
    from gx1.audit import cache_market_state_ta_direction_v1 as wrapper
    from gx1.contracts.immutable_event_authority_v1 import ImmutableEventAuthorityError
    root = tmp_path / 'new-final'
    root.mkdir()
    for name in ('RESULT.json', 'FAILED.json', 'TERMINAL.json'):
        value = {'synthetic': True, 'artifact': name}
        ref = wrapper._publish(name, value, root=root)
        assert ref['path'] == str(root / name)
        assert json.loads((root / name).read_text()) == value
        with pytest.raises(ImmutableEventAuthorityError, match='already exists'):
            wrapper._publish(name, value, root=root)
    arrays = {'parents': np.arange(4), 'representation': np.zeros((4, 5, 128), np.float32)}
    for partial, rows in ((False, 4), (True, 2)):
        ref = wrapper._save_cache('fit', arrays, rows, partial=partial, root=root)
        assert Path(ref['artifact']['path']).parent == root
        assert ref['partial'] is partial
        with np.load(ref['artifact']['path'], allow_pickle=False) as cache:
            np.testing.assert_array_equal(cache['parents'], np.arange(rows))
            assert cache['representation'].shape == (rows, 5, 128)


def test_cached_scores_must_match_full_native_measurement():
    import copy
    from gx1.audit import cache_market_state_ta_direction_v1 as wrapper
    from gx1.contracts.market_state_time_query_target_v1 import TIMEFRAMES
    scores = {task: {tf: {'conditional_side': {
        'count': 10, 'brier_sum': 2.4, 'log_sum': 6.8}} for tf in TIMEFRAMES}
        for task in ('state', 'future')}
    baseline = {'scores': {'model': copy.deepcopy(scores)}}
    assert wrapper._side_parity(scores, baseline)['future']['M5']['brier']['absolute_difference'] == 0
    baseline['scores']['model']['future']['M5']['conditional_side']['brier_sum'] += .1
    with pytest.raises(RuntimeError, match='SIDE_PARITY:future:M5:brier_sum'):
        wrapper._side_parity(scores, baseline)
