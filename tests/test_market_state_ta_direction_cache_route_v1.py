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


def reversed_plan():
    from gx1.audit import cache_market_state_ta_direction_v1 as wrapper
    ref = lambda path: {'path': str(path), 'sha256': 'a' * 64}
    return dict(schema_version=wrapper.REVERSED_SCHEMA, stage=wrapper.REVERSED_STAGE,
        source_repo=str(wrapper.REPO), source_base_commit='a' * 40,
        worker=ref(wrapper.WORKER), dispatch=ref(wrapper.REPO / wrapper.DISPATCH),
        cap_runner=ref(wrapper.REPO / wrapper.CAP_RUNNER), runtime_source_override={},
        inputs=wrapper.REVERSED_REFS, **wrapper.REVERSED_BOUNDS)


def test_reversed_scope_keeps_old_roots_and_requires_exact_train_only_plan():
    from gx1.audit import cache_market_state_ta_direction_v1 as wrapper
    plan = reversed_plan()
    path = wrapper.REVERSED_ROOT / 'PLAN.json'
    assert wrapper._reversed_plan(plan, path, wrapper.REVERSED_ROOT) == plan
    for key, value in [('fit_rows', 4096), ('score_rows', 192), ('max_backbone_encodings', 2959),
                       ('max_checkpoint_loads', 2), ('max_optimizer_updates', 1),
                       ('training_enabled', True), ('batch_size', 16.0), ('device', 'cpu')]:
        with pytest.raises(RuntimeError, match='REVERSED_EXACT_PLAN'):
            wrapper._reversed_plan({**plan, key: value}, path, wrapper.REVERSED_ROOT)
    for bad_path, bad_output in ((wrapper.ROOT / 'PLAN.json', wrapper.REVERSED_ROOT),
                                 (path, wrapper.FINAL_ROOT), (path, wrapper.REVERSED_ROOT / 'other')):
        with pytest.raises(RuntimeError, match='REVERSED_EXACT_PLAN'):
            wrapper._reversed_plan(plan, bad_path, bad_output)
    with pytest.raises(RuntimeError, match='EXACT_CACHE_PATHS'):
        wrapper._cache_scope(path, wrapper.REVERSED_ROOT)


def test_reversed_dispatch_does_not_enter_consumed_cache_paths(monkeypatch):
    from gx1.audit import cache_market_state_ta_direction_v1 as wrapper
    calls = []
    monkeypatch.setattr(wrapper, '_inspect_reversed', lambda *args, **kwargs:
        calls.append(('inspect', args, kwargs)) or dict(plan_ref={'path': 'bound', 'sha256': 'a' * 64},
            view={'contract_sha256': 'b' * 64}, current_sources={str(i): 'c' * 64 for i in range(85)}))
    monkeypatch.setattr(wrapper, '_run_reversed',
        lambda *args, **kwargs: calls.append(('run', args, kwargs)) or {'status': 'synthetic'})
    monkeypatch.setattr(wrapper, '_inspect', lambda *args, **kwargs: pytest.fail('old consumed scope entered'))
    path, root = wrapper.REVERSED_ROOT / 'PLAN.json', wrapper.REVERSED_ROOT
    assert wrapper.preflight_read_only(path, 'a' * 64, root)['backbone_encodings'] == 0
    assert wrapper.run(path, 'a' * 64, root, device='cuda') == {'status': 'synthetic'}
    assert calls[0][2] == {'execute': False} and calls[1][2] == {'device': 'cuda'}


def test_reversed_source_bridge_changes_only_cap_and_verifies_all_other_owners(monkeypatch):
    from gx1.audit import cache_market_state_ta_direction_v1 as wrapper
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
    names = [wrapper.CAP_RUNNER] + ['gx1/fixture_' + str(i) + '.py' for i in range(76)]
    names += ['.venv/fixture_' + str(i) + '.py' for i in range(8)]
    raw = b'unchanged historical source'
    old_sha, cap_sha = hashlib.sha256(raw).hexdigest(), hashlib.sha256(b'new exact cap case').hexdigest()
    owners = {name: {'path': str(wrapper.REPO / name), 'sha256': old_sha} for name in names}
    cap = {'path': str(wrapper.REPO / wrapper.CAP_RUNNER), 'sha256': cap_sha}
    plan = dict(cap_runner=cap, runtime_source_override={'original_ref': owners[wrapper.CAP_RUNNER], 'current_ref': cap})
    live = {ref['path']: ref['sha256'] for ref in owners.values()}; live[cap['path']] = cap_sha
    seen = []
    def verify(ref, **kwargs):
        seen.append(ref['path'])
        if live[ref['path']] != ref['sha256']:
            raise RuntimeError('PHYSICAL_OWNER_MISMATCH')
        return ref
    monkeypatch.setattr(fit, 'required_source_paths', lambda **kwargs: set(names))
    monkeypatch.setattr(wrapper.subprocess, 'check_output', lambda *args, **kwargs: raw)
    monkeypatch.setattr(wrapper.campaign, 'require_binding', verify)
    actual = wrapper._reversed_sources(plan, {'source_owners': owners})
    assert len(actual) == len(set(seen)) == 85 and actual[wrapper.CAP_RUNNER] == cap_sha
    assert all(actual[name] == old_sha for name in names if name != wrapper.CAP_RUNNER)
    live[owners[names[1]]['path']] = cap_sha
    with pytest.raises(RuntimeError, match='PHYSICAL_OWNER_MISMATCH'):
        wrapper._reversed_sources(plan, {'source_owners': owners})
    with pytest.raises(RuntimeError, match='REVERSED_CAP_ONLY_BRIDGE'):
        wrapper._reversed_sources({**plan, 'runtime_source_override': {}}, {'source_owners': owners})


def parity_arrays():
    import numpy as np
    n = 6
    captured = dict(parents=np.arange(n, dtype=np.int64), issues=np.arange(n, dtype=np.int64),
                    query_native=np.tile(np.array([1, 3, 12, 48, 288], np.int64), (n, 1)))
    baseline = dict(parents=captured['parents'].copy(), issues=captured['issues'].copy())
    for task in ('state', 'future'):
        captured[task + '_activity_side_logits'] = np.zeros((n, 5, 2), np.float32)
        baseline[task + '_raw'] = np.zeros((n, 5, 2), np.float32)
        baseline[task + '_query'] = captured['query_native'].copy()
        for key, dtype in (('allowed', np.uint8), ('eligible', bool), ('mature', bool)):
            captured[task + '_' + key] = np.ones((n, 5), dtype)
            baseline[task + '_' + key] = np.ones((n, 5), dtype)
    return captured, baseline


def test_reversed_parity_is_exact_m5_including_tail_not_other_sampled_s_queries():
    import numpy as np
    from gx1.audit import cache_market_state_ta_direction_v1 as wrapper
    captured, baseline = parity_arrays()
    baseline['state_query'][:, 1:] = 1
    baseline['state_raw'][:, 1:] = 7.
    baseline['state_allowed'][:, 1:] = 4
    value = wrapper._reversed_batch_parity(captured, baseline, 0, 6, np=np)
    assert value['M5_raw_bitexact'] and value['exact_M5_axes_masks']
    captured['state_activity_side_logits'][5, 0, 1] = np.nextafter(np.float32(0), np.float32(1))
    value = wrapper._reversed_batch_parity(captured, baseline, 0, 6, np=np)
    assert value['M5_raw_bitexact'] is False and value['max_abs_M5_raw_error']['state'] > 0
    captured['state_activity_side_logits'][5, 0, 1] = 0
    captured['future_mature'][5, 0] = False
    assert wrapper._reversed_batch_parity(captured, baseline, 0, 6, np=np)['exact_M5_axes_masks'] is False


def test_reversed_partial_preserves_offending_batch_before_failure(tmp_path, monkeypatch):
    import numpy as np
    from gx1.audit import cache_market_state_ta_direction_v1 as wrapper
    monkeypatch.setattr(wrapper, 'REVERSED_ROOT', tmp_path)
    arrays = dict(parents=np.arange(22, dtype=np.int64),
                  state_representation=np.ones((22, 128), np.float32))
    artifact = wrapper._save_reversed(arrays, 6, partial=True)
    assert Path(artifact['path']).name == 'FIT_REPRESENTATIONS_PARTIAL.npz'
    with np.load(artifact['path'], allow_pickle=False) as saved:
        assert saved['state_representation'].shape == (6, 128)
        np.testing.assert_array_equal(saved['parents'], np.arange(6))
