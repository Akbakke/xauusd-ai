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



@pytest.mark.parametrize('worker_name', ['_run_reversed', '_run_fixed_score'])
@pytest.mark.parametrize('phase', ['before_provider', 'already_released', 'prepared'])
def test_reversed_finalizer_uses_page_release_for_module_provider_only(phase, worker_name):
    """Exercise the production finally block without constructing a model."""
    import ast
    from types import ModuleType, SimpleNamespace
    tree = ast.parse((REPO / 'gx1/audit/cache_market_state_ta_direction_v1.py').read_text())
    worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == worker_name)
    final = next(node.finalbody for node in worker.body if isinstance(node, ast.Try) and node.finalbody)
    cleanup = ast.fix_missing_locations(ast.Module(body=final, type_ignores=[]))
    provider = None if phase == 'before_provider' else ModuleType('mature_prefix_input_provider')
    prepared = {'synthetic_pages': object()} if phase == 'prepared' else None
    releases, signals = [], []
    saved_alarm = object()
    namespace = dict(provider=provider, prepared=prepared, previous_alarm=saved_alarm,
        base=SimpleNamespace(_release_pages=lambda owner, pages: releases.append((owner, pages))),
        signal=SimpleNamespace(ITIMER_REAL=0, SIGALRM=14,
            setitimer=lambda *args: signals.append(('timer', args)),
            signal=lambda *args: signals.append(('handler', args))))
    exec(compile(cleanup, '<reversed-cache-finalizer>', 'exec'), namespace)
    assert releases == ([(provider, prepared)] if phase == 'prepared' else [])
    assert signals == [('timer', (0, 0.)), ('handler', (14, saved_alarm))]


def fixed_score_plan():
    from gx1.audit import cache_market_state_ta_direction_v1 as wrapper
    ref = lambda path: {'path': str(path), 'sha256': 'a' * 64}
    return dict(schema_version=wrapper.SCORE_SCHEMA, stage=wrapper.SCORE_STAGE,
        source_repo=str(wrapper.REPO), source_base_commit='a' * 40,
        worker=ref(wrapper.WORKER), dispatch=ref(wrapper.REPO / wrapper.DISPATCH),
        cap_runner=ref(wrapper.REPO / wrapper.CAP_RUNNER), runtime_source_override={},
        inputs=wrapper.SCORE_REFS, **wrapper.SCORE_BOUNDS)


def test_fixed_score_plan_rejects_fits_counts_extra_baselines_and_consumed_roots():
    from gx1.audit import cache_market_state_ta_direction_v1 as wrapper
    plan = fixed_score_plan()
    path, root = wrapper.SCORE_ROOT / 'PLAN.json', wrapper.SCORE_ROOT
    assert wrapper._fixed_score_plan(plan, path, root) == plan
    for key, value in [('fit_rows', 4096), ('score_rows', 192), ('reused_score_rows', 0),
                       ('max_backbone_encodings', 1110), ('fresh_baseline_forwards', 0),
                       ('max_missing_score_encodings', 1110), ('max_model_constructions', 2),
                       ('max_checkpoint_loads', 1), ('head_tensor_imports', 0),
                       ('max_optimizer_updates', 1), ('max_backwards', 1),
                       ('training_enabled', True), ('batch_size', 16.0), ('device', 'cpu')]:
        with pytest.raises(RuntimeError, match='FIXED_SCORE_EXACT_PLAN'):
            wrapper._fixed_score_plan({**plan, key: value}, path, root)
    for old in (wrapper.ROOT, wrapper.FINAL_ROOT, wrapper.REVERSED_ROOT):
        with pytest.raises(RuntimeError, match='FIXED_SCORE_EXACT_PLAN'):
            wrapper._fixed_score_plan(plan, old / 'PLAN.json', root)
        with pytest.raises(RuntimeError, match='FIXED_SCORE_EXACT_PLAN'):
            wrapper._fixed_score_plan(plan, path, old)
    with pytest.raises(RuntimeError, match='FIXED_SCORE_EXACT_PLAN'):
        wrapper._fixed_score_plan({**plan, 'inputs': wrapper.REVERSED_REFS}, path, root)
    with pytest.raises(RuntimeError, match='FIXED_SCORE_SOURCE_BASE'):
        wrapper._fixed_score_plan({**plan, 'source_base_commit': 'main'}, path, root)


def test_fixed_score_dispatch_never_enters_consumed_cache_paths(monkeypatch):
    from gx1.audit import cache_market_state_ta_direction_v1 as wrapper
    calls = []
    monkeypatch.setattr(wrapper, '_inspect_fixed_score', lambda *args, **kwargs:
        calls.append(('inspect', args, kwargs)) or dict(plan_ref={'path': 'bound', 'sha256': 'a' * 64},
            view={'contract_sha256': 'b' * 64}, current_sources={str(i): 'c' * 64 for i in range(85)}))
    monkeypatch.setattr(wrapper, '_run_fixed_score',
        lambda *args, **kwargs: calls.append(('run', args, kwargs)) or {'status': 'synthetic'})
    for name in ('_inspect', '_inspect_reversed', '_run_reversed'):
        monkeypatch.setattr(wrapper, name, lambda *args, **kwargs: pytest.fail('consumed scope entered'))
    path, root = wrapper.SCORE_ROOT / 'PLAN.json', wrapper.SCORE_ROOT
    assert wrapper.preflight_read_only(path, 'a' * 64, root)['backbone_encodings'] == 0
    assert wrapper.run(path, 'a' * 64, root, device='cuda') == {'status': 'synthetic'}
    assert calls[0][2] == {'execute': False} and calls[1][2] == {'device': 'cuda'}


def test_fixed_score_reuses_exact_twelve_batches_and_scores_tail_nine_once():
    from gx1.audit import cache_market_state_ta_direction_v1 as wrapper
    reused, missing = wrapper._fixed_score_layout()
    assert reused == list(range(0, 64)) + list(range(8352, 8416)) + list(range(17680, 17744))
    new_rows = [row for batch in missing for row in range(16 * batch, min(16 * batch + 16, 17753))]
    assert len(missing) == 1098 and missing[-1] == 1109
    assert new_rows[-9:] == list(range(17744, 17753))
    assert len(new_rows) == 17561 and not set(reused).intersection(new_rows)
    assert sorted(reused + new_rows) == list(range(17753))
    assert 2 * (1849 + 12) * 2 == wrapper.SCORE_BOUNDS['max_cached_head_calls']
    assert 2 * len(missing) == wrapper.SCORE_BOUNDS['max_reference_linear_calls']


def test_fixed_score_four_coefficients_are_exact_locked_fp32_rounding():
    import numpy as np
    import torch
    from gx1.audit import cache_market_state_ta_direction_v1 as wrapper
    archive, lock = {}, {'beta_fp64_sha256': {}}
    for task, offset in (('state', 0.), ('future', 1.)):
        beta = np.linspace(-.13, .17, 129, dtype=np.float64) + offset
        archive[task + '_beta_fp64'] = beta
        archive[task + '_beta_fp32'] = beta.astype(np.float32)
        lock['beta_fp64_sha256'][task] = hashlib.sha256(beta.tobytes()).hexdigest()
    tensors = wrapper._fixed_coefficients(archive, lock, np=np, torch=torch)
    assert len(tensors) == 4
    for task, name in (('state', 'state_model.m5_side_head'), ('future', 'future_m5_side_head')):
        assert tensors[name + '.weight'].shape == (1, 128)
        assert tensors[name + '.bias'].shape == (1,)
        assert tensors[name + '.weight'].dtype == torch.float32
        np.testing.assert_array_equal(tensors[name + '.weight'].numpy()[0], archive[task + '_beta_fp32'][1:])
        np.testing.assert_array_equal(tensors[name + '.bias'].numpy(), archive[task + '_beta_fp32'][:1])
    archive['state_beta_fp32'][0] = np.nextafter(archive['state_beta_fp32'][0], np.float32(1))
    with pytest.raises(RuntimeError, match='FIXED_SCORE_COEFFICIENT_LOCK:state'):
        wrapper._fixed_coefficients(archive, lock, np=np, torch=torch)
    archive['state_beta_fp32'] = archive['state_beta_fp64'].astype(np.float32)
    lock['beta_fp64_sha256']['future'] = '0' * 64
    with pytest.raises(RuntimeError, match='FIXED_SCORE_COEFFICIENT_LOCK:future'):
        wrapper._fixed_coefficients(archive, lock, np=np, torch=torch)


def fixed_score_values():
    import copy
    value = lambda b, log: dict(count=10, up_count=6, brier=b, log=log, mean_p=.55)
    cell = dict(CANDIDATE=value(.20, .60), INITIAL=value(.21, .61),
                PRIOR=value(.22, .62), R1=value(.19, .59))
    return {panel: {task: copy.deepcopy(cell) for task in ('state', 'future')}
            for panel in ('TRAIN', 'LATER_FULL')}


def test_fixed_score_requires_both_primary_references_in_all_four_cells():
    from gx1.audit import cache_market_state_ta_direction_v1 as wrapper
    scores = fixed_score_values()
    result = wrapper._fixed_score_decision(scores)
    assert result['decision'] == 'PASS_ALL_FOUR_INITIAL_PRIOR_CELLS'
    assert result['R1_diagnostic_only'] and not result['candidate_selection']
    assert all(not result['cells'][p][t]['R1']['passes'] for p in scores for t in scores[p])
    for panel in scores:
        for task in scores[panel]:
            for reference, margin in (('INITIAL', 1e-6), ('PRIOR', 1e-8)):
                for metric in ('brier', 'log'):
                    candidate = scores[panel][task]['CANDIDATE']
                    saved = candidate[metric]
                    candidate[metric] = scores[panel][task][reference][metric] - margin / 2
                    assert wrapper._fixed_score_decision(scores)['decision'] == 'FAIL_INITIAL_PRIOR_COMPARISON'
                    candidate[metric] = saved
    scores['LATER_FULL']['future']['PRIOR']['up_count'] = 5
    with pytest.raises(RuntimeError, match='FIXED_SCORE_COMPARISON_OBSERVATIONS'):
        wrapper._fixed_score_decision(scores)


def candidate_weights_fixture():
    import copy
    import torch
    from gx1.audit import cache_market_state_ta_direction_v1 as wrapper
    from gx1.contracts import entry_native_direction_research_state_v1 as old
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    model = torch.nn.Linear(2, 1)
    model.register_buffer('persistent_probe', torch.tensor([3.]))
    model.register_buffer('nonpersistent_probe', torch.tensor([4.]), persistent=False)
    saved = dict(schema_version=wrapper.SCORE_WEIGHTS_SCHEMA, canonical_resume_authority=False,
        optimizer_state_included=False, model_state=copy.deepcopy(model.state_dict()),
        buffer_state=copy.deepcopy(dict(model.named_buffers())))
    saved['model_state_sha256'] = canonical_model_state_sha256(saved['model_state'])
    saved['buffer_state_sha256'] = old._tree_sha(saved['buffer_state'])
    saved['weights_sha256'] = old._tree_sha(saved)
    return model, saved


def test_fixed_score_roundtrip_actually_decodes_and_restores_all_buffers(tmp_path):
    import torch
    from gx1.audit import cache_market_state_ta_direction_v1 as wrapper
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    model, expected = candidate_weights_fixture()
    path = tmp_path / 'CANDIDATE_WEIGHTS.pt'
    torch.save(expected, path)
    ref = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    counts = dict(resume_state_load_calls=0, resume_state_load_returns=0)
    saved = common._load_state(ref, torch=torch, maximum=65536, counters=counts)
    assert counts == dict(resume_state_load_calls=1, resume_state_load_returns=1)
    assert saved is not expected and saved['model_state']['weight'] is not expected['model_state']['weight']
    with torch.no_grad():
        model.weight.fill_(8.)
        for value in model.buffers(): value.fill_(9.)
    wrapper._require_candidate_weights(saved, expected=expected, model=model, torch=torch)
    assert all(torch.equal(value, expected['model_state'][name]) for name, value in model.state_dict().items())
    assert all(torch.equal(value, expected['buffer_state'][name]) for name, value in model.named_buffers())


@pytest.mark.parametrize('field', ['model_state_sha256', 'buffer_state_sha256'])
def test_fixed_score_roundtrip_rejects_false_tensor_or_buffer_hash(field):
    import copy
    import torch
    from gx1.audit import cache_market_state_ta_direction_v1 as wrapper
    from gx1.contracts import entry_native_direction_research_state_v1 as old
    model, expected = candidate_weights_fixture()
    saved = copy.deepcopy(expected)
    saved[field] = '0' * 64
    saved['weights_sha256'] = old._tree_sha({k: v for k, v in saved.items() if k != 'weights_sha256'})
    # Exercise the tensor-digest check independently of the enclosing tree seal.
    expected['weights_sha256'] = saved['weights_sha256']
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(RuntimeError, match='FIXED_SCORE_ROUNDTRIP_DIGEST'):
        wrapper._require_candidate_weights(saved, expected=expected, model=model, torch=torch)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def shrink_plan():
    from gx1.audit import cache_market_state_ta_direction_v1 as wrapper
    ref = lambda path: {'path': str(path), 'sha256': 'a' * 64}
    return dict(schema_version=wrapper.SHRINK_SCHEMA, stage=wrapper.SHRINK_STAGE,
        source_repo=str(wrapper.REPO), source_base_commit='a' * 40,
        worker=ref(wrapper.WORKER), dispatch=ref(wrapper.REPO / wrapper.DISPATCH),
        cap_runner=ref(wrapper.REPO / wrapper.CAP_RUNNER), runtime_source_override={},
        inputs=wrapper.SHRINK_REFS, **wrapper.SHRINK_BOUNDS)


def shrink_inputs():
    import torch
    coefficients = {}
    for prefix, bias, offset in (('state_model.m5_side_head', .19012347, 0.),
                                 ('future_m5_side_head', -.27098765, .173)):
        coefficients[prefix + '.weight'] = (torch.linspace(-.37, .91, 128, dtype=torch.float32) + offset).reshape(1, 128)
        coefficients[prefix + '.bias'] = torch.tensor([bias], dtype=torch.float32)
    lock = {'slopes': {'state': {'slope': .371234567890123}, 'future': {'slope': .819876543210987}}}
    priors = {'tasks': {'state': {'raw_logits': [[0., 1.23456789012345]]},
                        'future': {'raw_logits': [[0., -.876543210987654]]}}}
    return coefficients, lock, priors


def test_shrink_plan_is_exact_and_cannot_expand_encodings_or_updates():
    from gx1.audit import cache_market_state_ta_direction_v1 as wrapper
    plan = shrink_plan()
    path, root = wrapper.SHRINK_ROOT / 'PLAN.json', wrapper.SHRINK_ROOT
    assert wrapper.SHRINK_STAGE == 'market_state_ta_m5_locked_shrink_native_20261111_001'
    assert wrapper._shrink_plan(plan, path, root) == plan
    for key, value in (('max_backbone_encodings', plan['max_backbone_encodings'] + 1),
                       ('max_optimizer_updates', 1), ('training_enabled', True),
                       ('inputs', wrapper.SCORE_REFS), ('stage', wrapper.SCORE_STAGE)):
        with pytest.raises(RuntimeError):
            wrapper._shrink_plan({**plan, key: value}, path, root)
    for old in (wrapper.ROOT, wrapper.FINAL_ROOT, wrapper.REVERSED_ROOT, wrapper.SCORE_ROOT):
        with pytest.raises(RuntimeError):
            wrapper._shrink_plan(plan, old / 'PLAN.json', root)
        with pytest.raises(RuntimeError):
            wrapper._shrink_plan(plan, path, old)


def test_shrink_dispatch_keeps_preflight_and_run_out_of_consumed_routes(monkeypatch):
    from gx1.audit import cache_market_state_ta_direction_v1 as wrapper
    calls = []
    monkeypatch.setattr(wrapper, '_inspect_shrink_score', lambda *args, **kwargs:
        calls.append(('inspect', args, kwargs)) or dict(plan_ref={'path': 'bound', 'sha256': 'a' * 64},
            view={'contract_sha256': 'b' * 64}, current_sources={str(i): 'c' * 64 for i in range(85)}))
    monkeypatch.setattr(wrapper, '_run_shrink_score',
        lambda *args, **kwargs: calls.append(('run', args, kwargs)) or {'status': 'synthetic'})
    for name in ('_inspect', '_inspect_reversed', '_run_reversed', '_inspect_fixed_score', '_run_fixed_score'):
        monkeypatch.setattr(wrapper, name, lambda *args, **kwargs: pytest.fail('consumed route entered'))
    path, root = wrapper.SHRINK_ROOT / 'PLAN.json', wrapper.SHRINK_ROOT
    assert wrapper.preflight_read_only(path, 'a' * 64, root)['backbone_encodings'] == 0
    assert wrapper.run(path, 'a' * 64, root, device='cuda') == {'status': 'synthetic'}
    assert calls == [('inspect', (path, 'a' * 64, root), {'execute': False}),
                     ('run', (path, 'a' * 64, root), {'device': 'cuda'})]


def test_shrink_folds_prior_centered_fp32_source_once_without_mutating_it():
    import numpy as np
    import torch
    from gx1.audit import cache_market_state_ta_direction_v1 as wrapper
    original, lock, priors = shrink_inputs()
    before = {key: value.clone() for key, value in original.items()}
    got = wrapper._shrink_coefficients(original, lock, priors, torch=torch)
    assert set(got) == set(original) and len(got) == 4
    for task, prefix in (('state', 'state_model.m5_side_head'), ('future', 'future_m5_side_head')):
        slope = lock['slopes'][task]['slope']
        prior = priors['tasks'][task]['raw_logits'][0][1]
        weight = before[prefix + '.weight'].numpy().astype(np.float64)
        bias = before[prefix + '.bias'].numpy().astype(np.float64)
        expected = {prefix + '.weight': (slope * weight).astype(np.float32),
                    prefix + '.bias': (prior + slope * (bias - prior)).astype(np.float32)}
        for key, value in expected.items():
            assert got[key].dtype == torch.float32 and got[key].shape == before[key].shape
            np.testing.assert_array_equal(got[key].detach().cpu().numpy().view(np.uint32), value.view(np.uint32))
    assert all(torch.equal(original[key], value) for key, value in before.items())
    identity = wrapper._shrink_coefficients(original,
        {'slopes': {task: {'slope': 1.} for task in ('state', 'future')}}, priors, torch=torch)
    assert all(torch.equal(identity[key], value) for key, value in before.items())


def test_shrink_rejects_zero_expansion_negative_and_nonfinite_slopes():
    import copy
    import torch
    from gx1.audit import cache_market_state_ta_direction_v1 as wrapper
    original, lock, priors = shrink_inputs()
    before = {key: value.clone() for key, value in original.items()}
    for task in ('state', 'future'):
        for slope in (0., -.1, 1.000001, float('nan'), float('inf'), float('-inf')):
            invalid = copy.deepcopy(lock)
            invalid['slopes'][task]['slope'] = slope
            with pytest.raises(RuntimeError):
                wrapper._shrink_coefficients(original, invalid, priors, torch=torch)
    assert all(torch.equal(original[key], value) for key, value in before.items())
