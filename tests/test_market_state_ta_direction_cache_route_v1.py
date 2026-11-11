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



def _synthetic_fusion_backbone(torch):
    """Small synthetic layout only; never constructs the native model."""
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._specialist_names = ('a', 'b', 'c')
            self._mtf_specialist_names = ('a', 'b', 'c')
            self._mtf_feature_gate_width = 5
            self.tf_gate_logits = torch.nn.Parameter(torch.zeros(2))
            self.register_buffer('fixture_buffer', torch.ones(1))
            self.fuse = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(.2))
            self.specialist_out = torch.nn.Linear(4, 4)
            self.cross_tf_out = torch.nn.Linear(4, 4)
            self.family_tf_cooperation_out = torch.nn.Linear(4, 4)
            self.entry_q_joint_norm = torch.nn.LayerNorm(16)
            self.entry_q_joint_in = torch.nn.Linear(16, 4)
            self.fault = None

        def forward(self, x):
            base = self.fuse(x)
            if self.fault == 'duplicate':
                self.fuse(x)
            specialist = (torch.zeros_like(base) if self.fault == 'missing'
                          else self.specialist_out(base))
            local = base + specialist
            mtf = self.cross_tf_out(local)
            cooperation = self.family_tf_cooperation_out(local)
            joint = torch.cat((local, local + mtf + cooperation, mtf, x), dim=1)
            hidden = torch.nn.functional.gelu(self.entry_q_joint_in(self.entry_q_joint_norm(joint)))
            output = dict(entry_q_joint_hidden=hidden,
                specialist_gate=torch.softmax(hidden[:, :3], dim=1),
                tf_gate=torch.softmax(hidden[:, :2] + self.tf_gate_logits, dim=1),
                family_tf_cooperation_gate=torch.softmax(hidden[:, :1].expand(-1, 6), dim=1),
                family_tf_feature_gate=hidden[:, :1, None].expand(-1, 2, 5))
            if self.fault == 'nonfinite':
                output['tf_gate'] = output['tf_gate'] * float('nan')
            elif self.fault == 'dtype':
                output['entry_q_joint_hidden'] = hidden.double()
            elif self.fault == 'shape':
                output['entry_q_joint_hidden'] = hidden[:, :1]
            elif self.fault == 'keys':
                output['extra'] = hidden
            return output
    return Backbone()


def _fusion_hook_inventory(model):
    return {id(m): (tuple(m._forward_hooks), tuple(m._forward_pre_hooks))
            for m in model.modules()}


def test_passive_fusion_preserves_outputs_rng_parameters_modes_and_graph():
    import torch
    from gx1.audit.cache_market_state_ta_direction_v1 import observe_ta_fusion_once
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        model = _synthetic_fusion_backbone(torch).train()
        model.specialist_out.eval()
        model.specialist_out.weight.requires_grad_(False)
        x = torch.arange(8, dtype=torch.float32).reshape(2, 4).requires_grad_()
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        parameters = [(id(p), p.requires_grad) for p in model.parameters()]
        modes = [m.training for m in model.modules()]
        existing = model.fuse.register_forward_hook(lambda *args: None)
        hooks = _fusion_hook_inventory(model)
        rng = torch.get_rng_state().clone()
        expected = model(x)
        expected_rng = torch.get_rng_state().clone()
        expected_gradient, = torch.autograd.grad(expected['entry_q_joint_hidden'].sum(), x)
        torch.set_rng_state(rng)
        # 57 float32 scalars per row, two rows: exact successful byte boundary.
        with observe_ta_fusion_once(model, batch_rows=2, max_bytes=456, torch=torch) as captured:
            actual = model(x)
        assert len(captured) == 10 and sum(t.numel() * t.element_size() for t in captured.values()) == 456
        for name in expected:
            assert torch.equal(actual[name], expected[name])
            assert torch.equal(captured[name], actual[name])
            assert not captured[name].requires_grad and captured[name].grad_fn is None
            assert captured[name].data_ptr() != actual[name].data_ptr()
        gradient, = torch.autograd.grad(actual['entry_q_joint_hidden'].sum(), x)
        assert torch.equal(gradient, expected_gradient)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
        assert [(id(p), p.requires_grad) for p in model.parameters()] == parameters
        assert [m.training for m in model.modules()] == modes
        assert _fusion_hook_inventory(model) == hooks
        existing.remove()


@pytest.mark.parametrize('fault,error', [
    ('duplicate', 'FUSION_DUPLICATE:fuse'), ('missing', 'FUSION_INCOMPLETE'),
    ('nonfinite', 'FUSION_NONFINITE:tf_gate'), ('dtype', 'FUSION_TENSOR:entry_q_joint_hidden'),
    ('shape', 'FUSION_TENSOR:entry_q_joint_hidden'), ('keys', 'FUSION_OUTPUT_KEYS'),
])
def test_passive_fusion_rejects_incomplete_or_invalid_capture_and_removes_hooks(fault, error):
    import torch
    from gx1.audit.cache_market_state_ta_direction_v1 import observe_ta_fusion_once
    model = _synthetic_fusion_backbone(torch)
    model.fault = fault
    hooks = _fusion_hook_inventory(model)
    with pytest.raises(RuntimeError, match=error):
        with observe_ta_fusion_once(model, batch_rows=2, max_bytes=456, torch=torch) as captured:
            model(torch.ones(2, 4))
    assert captured == {} and _fusion_hook_inventory(model) == hooks


@pytest.mark.parametrize('rows,limit,error', [
    (0, 456, 'FUSION_BOUNDS'), (True, 456, 'FUSION_BOUNDS'),
    (2, True, 'FUSION_BOUNDS'), (2, 455, 'FUSION_BYTE_LIMIT'),
])
def test_passive_fusion_rejects_bounds_before_registering_hooks(rows, limit, error):
    import torch
    from gx1.audit.cache_market_state_ta_direction_v1 import observe_ta_fusion_once
    model = _synthetic_fusion_backbone(torch)
    hooks = _fusion_hook_inventory(model)
    with pytest.raises(RuntimeError, match=error):
        with observe_ta_fusion_once(model, batch_rows=rows, max_bytes=limit, torch=torch):
            pytest.fail('invalid bound admitted')
    assert _fusion_hook_inventory(model) == hooks


def test_passive_fusion_removes_hooks_after_registration_or_user_failure(monkeypatch):
    import torch
    from gx1.audit.cache_market_state_ta_direction_v1 import observe_ta_fusion_once
    model = _synthetic_fusion_backbone(torch)
    hooks = _fusion_hook_inventory(model)
    def reject(*args, **kwargs):
        raise RuntimeError('registration failed')
    with monkeypatch.context() as patch:
        patch.setattr(model.specialist_out, 'register_forward_hook', reject)
        with pytest.raises(RuntimeError, match='registration failed'):
            with observe_ta_fusion_once(model, batch_rows=2, max_bytes=456, torch=torch):
                pytest.fail('failed registration admitted')
    assert _fusion_hook_inventory(model) == hooks
    with pytest.raises(ValueError, match='caller failed'):
        with observe_ta_fusion_once(model, batch_rows=2, max_bytes=456, torch=torch) as captured:
            model(torch.ones(2, 4))
            raise ValueError('caller failed')
    assert captured == {} and _fusion_hook_inventory(model) == hooks


def test_passive_fusion_rejects_wrong_batch_and_swallowed_duplicate():
    import torch
    from gx1.audit.cache_market_state_ta_direction_v1 import observe_ta_fusion_once
    model = _synthetic_fusion_backbone(torch)
    hooks = _fusion_hook_inventory(model)
    with pytest.raises(RuntimeError, match='FUSION_TENSOR:fuse'):
        with observe_ta_fusion_once(model, batch_rows=2, max_bytes=456, torch=torch):
            model(torch.ones(3, 4))
    assert _fusion_hook_inventory(model) == hooks
    with pytest.raises(RuntimeError, match='FUSION_INCOMPLETE'):
        with observe_ta_fusion_once(model, batch_rows=2, max_bytes=456, torch=torch) as captured:
            model(torch.ones(2, 4))
            with pytest.raises(RuntimeError, match='FUSION_DUPLICATE:fuse'):
                model(torch.ones(2, 4))
    assert captured == {} and _fusion_hook_inventory(model) == hooks



def fusion_plan():
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    ref=lambda path: {'path':str(path),'sha256':'a'*64}
    return dict(schema_version=w.FUSION_SCHEMA,stage=w.FUSION_STAGE,source_repo=str(w.REPO),
        source_base_commit='b'*40,worker=ref(w.WORKER),dispatch=ref(w.REPO/w.DISPATCH),
        cap_runner=ref(w.REPO/w.CAP_RUNNER),runtime_source_override={},inputs=w.FUSION_REFS,**w.FUSION_BOUNDS)


def test_fusion_exact_plan_and_fixed_complete_batch_selection():
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    plan=fusion_plan();path=w.FUSION_ROOT/'PLAN.json'
    assert w._fusion_plan(plan,path,w.FUSION_ROOT)==plan
    for key,value in [('fit_rows',512),('batches_per_partition',32),('max_backbone_encodings',34),
                      ('training_enabled',True),('scalar_fits',1),('max_cache_bytes',2**30),
                      ('max_candidate_writes',1),('batch_size',16.),('batch_selection','random')]:
        with pytest.raises(RuntimeError,match='FUSION_EXACT_PLAN'):
            w._fusion_plan({**plan,key:value},path,w.FUSION_ROOT)
    for wrong_path,wrong_root in ((w.SHRINK_ROOT/'PLAN.json',w.FUSION_ROOT),(path,w.SHRINK_ROOT)):
        with pytest.raises(RuntimeError,match='FUSION_EXACT_PLAN'):
            w._fusion_plan(plan,wrong_path,wrong_root)
    for rows in (29574,17753):
        indices=w._fusion_batch_indices(rows)
        assert len(indices)==len(set(indices))==16
        assert indices==tuple(i*(rows//16-1)//15 for i in range(16))
        assert indices[0]==0 and (indices[-1]+1)*16<=rows
    with pytest.raises(RuntimeError,match='FUSION_POPULATION'):
        w._fusion_batch_indices(512)


def test_fusion_dispatch_keeps_preflight_and_run_separate(monkeypatch):
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    calls=[]
    monkeypatch.setattr(w,'_inspect_fusion_observation',lambda *args,**kw:
        calls.append(('inspect',kw)) or dict(plan_ref={'path':'bound','sha256':'a'*64},
            view={'contract_sha256':'b'*64},current_sources={str(i):'c'*64 for i in range(85)}))
    monkeypatch.setattr(w,'_run_fusion_observation',lambda *args,**kw:
        calls.append(('run',kw)) or {'synthetic':True})
    monkeypatch.setattr(w,'_inspect_shrink_score',lambda *a,**k:pytest.fail('consumed shrink entered'))
    path=w.FUSION_ROOT/'PLAN.json'
    report=w.preflight_read_only(path,'a'*64,w.FUSION_ROOT)
    assert report['archive_decodes']==report['backbone_encodings']==report['model_constructions']==0
    assert w.run(path,'a'*64,w.FUSION_ROOT,device='cuda')=={'synthetic':True}
    assert calls==[('inspect',{'execute':False}),('run',{'device':'cuda'})]


@pytest.mark.parametrize('mode',['fusion','retained','temporal'])
def test_fusion_preflight_checks_actual_candidate_schema_without_array_decode(monkeypatch,mode):
    import copy
    import numpy as np
    import torch
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    retained,temporal = mode=='retained',mode=='temporal'
    plan=temporal_boundary_plan() if temporal else retained_mtf_plan() if retained else fusion_plan();refs=plan['inputs']
    root=w.TEMPORAL_ROOT if temporal else w.MTF_RETAINED_ROOT if retained else w.FUSION_ROOT;path=root/'PLAN.json';seen=[]
    policy_ref={'path':str(w.REPO/'NEXT_RUN_POLICY.json'),'sha256':'d'*64}
    candidate=dict(status='COMPLETE',source_unchanged=True,evaluation_state_unchanged=True,
        plan=refs['candidate_plan'],candidate_model_state_sha256=w.FUSION_BOUNDS['candidate_model_state_sha256'],
        artifacts={'folded_coefficients':refs['folded_coefficients'],'fresh_baseline':refs['candidate_baseline']},
        fresh_candidate_baseline={'rows':16,'artifact':refs['candidate_baseline']},
        counts={'head_tensor_imports':4,'scalar_fits':0},comparison={'decision':'PASS_ALL_FOUR_INITIAL_PRIOR_CELLS'},
        test_data_used=False)
    docs={str(path):plan,refs['candidate_plan']['path']:{'inputs':refs},
        refs['candidate_result']['path']:candidate,
        refs['candidate_review']['path']:dict(status='PASS_INDEPENDENT_RETAINED_LOGIT_ARITHMETIC',
            result=refs['candidate_result'],plan=refs['candidate_plan'],actual_row0_outputs_bitexact_cached_heads=True,
            folded_coefficients_exact_from_original_FP32_and_locked_slopes=True,primary_comparison_pass=True,test_data_used=False),
        refs['candidate_terminal']['path']:dict(status='COMPLETE',exit_code=0,result=refs['candidate_result'],
            plan=refs['candidate_plan'],no_relaunch_under_this_plan=True,test_data_used=False),
        policy_ref['path']:{k:False for k in ('training_enabled','native_training_enabled',
                                             'native_launch_authorized','full_epoch_training_allowed')}}
    monkeypatch.setattr(w,'_read',lambda ref:copy.deepcopy(docs[ref['path']]))
    monkeypatch.setattr(w,'_ref',lambda path:policy_ref)
    monkeypatch.setattr(w,'_git',lambda *args:'b'*40)
    monkeypatch.setattr(w.subprocess,'run',lambda *a,**k:None)
    monkeypatch.setattr(w.campaign,'require_binding',lambda ref,**kw:seen.append(ref['path']) or ref)
    monkeypatch.setattr(w,'_qualify_reversed_origin',lambda p,b:
        seen.append('same_reversed_qualifier') or {'view':{'contract_sha256':'c'*64},'current_sources':{}})
    monkeypatch.setattr(w,'_qualify_fusion_trained_panels',lambda r:
        seen.append('trained_panel_qualifier') if r==w.MTF_RETAINED_REFS else pytest.fail('wrong trained refs'))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('preflight decoded arrays'))
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('preflight decoded checkpoint'))
    checked=w._inspect_fusion_observation(path,'a'*64,root,execute=False,retained=retained,temporal=temporal)
    assert checked['plan']==plan and 'same_reversed_qualifier' in seen
    assert ('trained_panel_qualifier' in seen) is (retained or temporal)
    assert refs['folded_coefficients']['path'] in seen and refs['candidate_baseline']['path'] in seen
    for mutation in [('candidate_model_state_sha256','0'*64),('evaluation_state_unchanged',False)]:
        bad=copy.deepcopy(candidate);bad[mutation[0]]=mutation[1]
        docs[refs['candidate_result']['path']]=bad
        with pytest.raises(RuntimeError,match='FUSION_QUALIFIED_CANDIDATE'):
            w._inspect_fusion_observation(path,'a'*64,root,execute=False,retained=retained,temporal=temporal)


def test_fusion_private_arrays_and_success_failure_terminals(tmp_path,monkeypatch):
    import json
    import numpy as np
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    arrays={'parents':np.arange(16,dtype=np.int64),'capture_fuse':np.ones((16,4),np.float32)}
    assert w._fusion_exact_arrays(arrays,{k:v.copy() for k,v in arrays.items()})
    changed={k:v.copy() for k,v in arrays.items()};changed['parents'][0]=17
    assert not w._fusion_exact_arrays(arrays,changed)
    for complete in (True,False):
        root=tmp_path/str(complete);root.mkdir();monkeypatch.setattr(w,'FUSION_ROOT',root)
        counts={'published_npz_bytes':0}
        name='TRAIN_FUSION.npz' if complete else 'TRAIN_FUSION_PARTIAL.npz'
        artifact=w._save_fusion_arrays(name,arrays,counts,np=np)
        with np.load(artifact['path'],allow_pickle=False) as z:
            assert w._fusion_exact_arrays({k:z[k] for k in z.files},arrays)
        assert counts['published_npz_bytes']==Path(artifact['path']).stat().st_size
        refs=w._publish_fusion_outcome({'path':'synthetic','sha256':'a'*64},
            {'source_unchanged':True,'counts':counts,'artifacts':{'TRAIN':artifact}},complete=complete)
        result=json.loads(Path(refs['result']['path']).read_text())
        terminal=json.loads(Path(refs['terminal']['path']).read_text())
        assert result['status']==('COMPLETE' if complete else 'FAILED_CONSUMED')
        assert result['predictive_value_claimed'] is result['training_enabled'] is False
        assert terminal['no_relaunch_under_this_plan'] and terminal['exit_code']==(0 if complete else 1)
        assert terminal['result']==refs['result']
        counts['published_npz_bytes']=w.FUSION_BOUNDS['max_cache_bytes']-1
        with pytest.raises(RuntimeError,match='FUSION_OUTPUT_BYTES'):
            w._save_fusion_arrays('LATER_FUSION_PARTIAL.npz',arrays,counts,np=np)
        assert not (root/'LATER_FUSION_PARTIAL.npz').exists()


def initial_fusion_plan():
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    return dict(fusion_plan(),stage=w.FUSION_INITIAL_STAGE,inputs=w.FUSION_INITIAL_REFS,**{
        k:v for k,v in w.FUSION_INITIAL_BOUNDS.items()})


def test_initial_fusion_exact_plan_has_no_import_or_extra_forward_scope():
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    plan=initial_fusion_plan();path=w.FUSION_INITIAL_ROOT/'PLAN.json'
    assert w._fusion_plan(plan,path,w.FUSION_INITIAL_ROOT,initial=True)==plan
    assert plan['head_tensor_imports']==0 and plan['max_backbone_encodings']==33
    assert plan['candidate_model_state_sha256']=='ae772fd755cbfa07b1b7bcca8bccd2c8a8889806989a289ddb8267972c281b37'
    for key,value in [('head_tensor_imports',4),('max_checkpoint_loads',2),('max_backbone_encodings',34),
                      ('candidate_model_state_sha256',w.FUSION_BOUNDS['candidate_model_state_sha256']),
                      ('max_optimizer_updates',1),('max_candidate_writes',1),('batch_size',16.)]:
        with pytest.raises(RuntimeError,match='FUSION_EXACT_PLAN'):
            w._fusion_plan({**plan,key:value},path,w.FUSION_INITIAL_ROOT,initial=True)
    for p,r,initial in ((path,w.FUSION_INITIAL_ROOT,False),
                        (w.FUSION_ROOT/'PLAN.json',w.FUSION_ROOT,True),
                        (path,w.FUSION_ROOT,True)):
        with pytest.raises(RuntimeError,match='FUSION_EXACT_PLAN'):
            w._fusion_plan(plan,p,r,initial=initial)


def test_initial_fusion_dispatch_reuses_observer_route(monkeypatch):
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    calls=[]
    monkeypatch.setattr(w,'_inspect_fusion_observation',lambda *a,**kw:
        calls.append(('inspect',kw)) or dict(plan_ref={},view={'contract_sha256':'a'*64},current_sources={}))
    monkeypatch.setattr(w,'_run_fusion_observation',lambda *a,**kw:calls.append(('run',kw)) or {'synthetic':True})
    monkeypatch.setattr(w,'_inspect_shrink_score',lambda *a,**k:pytest.fail('wrong route'))
    root=w.FUSION_INITIAL_ROOT;path=root/'PLAN.json'
    report=w.preflight_read_only(path,'a'*64,root)
    assert report['archive_decodes']==report['backbone_encodings']==report['model_constructions']==0
    assert report['candidate_model_state_sha256']==w.FUSION_INITIAL_BOUNDS['candidate_model_state_sha256']
    assert w.run(path,'a'*64,root,device='cuda')=={'synthetic':True}
    assert calls==[('inspect',{'execute':False,'initial':True}),('run',{'device':'cuda','initial':True})]


def test_initial_fusion_metadata_preflight_binds_baseline_and_matched_axes(tmp_path,monkeypatch):
    import copy
    import numpy as np
    import torch
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    monkeypatch.setattr(w,'FUSION_INITIAL_ROOT',tmp_path)
    plan=initial_fusion_plan();refs=plan['inputs'];path=tmp_path/'PLAN.json';seen=[]
    pr={'path':str(path),'sha256':'a'*64};policy_ref={'path':str(w.REPO/'NEXT_RUN_POLICY.json'),'sha256':'b'*64}
    recipe=dict(input_refs={old:refs[new] for new,old in (
        ('initial_checkpoint','original_initial_checkpoint'),('initial_baseline','initial_later_capture'),
        ('initial_measurement','initial_later_measurement'))},
        expected_initial_model_state_sha256=plan['candidate_model_state_sha256'],
        expected_model_contract_sha256='c'*64,expected_view_sha256='d'*64)
    baseline=dict(schema_version='gx1_market_state_ta_sf_fold1_evaluation_v1',
        evaluated_checkpoint=refs['initial_checkpoint'],evaluated_model_state_sha256=plan['candidate_model_state_sha256'],
        capture=refs['initial_baseline'],optimizer_updates=0,fold_id=1,test_data_used=False,full_population_evaluated=False,
        fixed_subset_positions=[i for start in (0,8352,17680) for i in range(start,start+64)],
        measurement=dict(rows=192,batches=12,partition='score',model_contract_sha256='c'*64,feature_view_sha256='d'*64))
    trained_plan={'path':str(w.FUSION_ROOT/'PLAN.json'),'sha256':'e'*64}
    trained=dict(status='COMPLETE',source_unchanged=True,evaluation_state_unchanged=True,source_commit='f'*40,
        candidate_model_state_sha256=w.FUSION_BOUNDS['candidate_model_state_sha256'],plan=trained_plan,test_data_used=False,
        artifacts={'TRAIN':refs['trained_train'],'LATER':refs['trained_later']},
        batch_indices={p:list(w._fusion_batch_indices(n)) for p,n in (('TRAIN',29574),('LATER',17753))},
        counts=dict(model_constructions=1,checkpoint_loads=1,backbone_encodings=33,forward_returns=33,
            instrumented_forwards=32,archive_decodes=2,head_tensor_imports=4,candidate_writes=0,optimizer_constructions=0,
            optimizer_updates=0,backwards=0,checkpoint_writes=0,resume_state_load_calls=1,resume_state_load_returns=1,
            scalar_fits=0,sampled_rows=512),passivity=dict(uninstrumented_matches_bound_baseline=True,
                instrumented_matches_uninstrumented=True,instrumented_matches_bound_baseline=True))
    terminal=dict(status='COMPLETE',result=refs['trained_result'],plan=trained_plan,exit_code=0,
        source_unchanged=True,no_relaunch_under_this_plan=True,test_data_used=False)
    review=dict(status='COMPLETE',schema_version='gx1_ta_fusion_observation_numeric_review_v1',
        result=refs['trained_result'],terminal=refs['trained_terminal'],source_commit='f'*40,test_data_used=False,
        partitions={p:dict(artifact=refs['trained_'+p.lower()],rows=256,unique_parents=256) for p in ('TRAIN','LATER')})
    gate=dict(status='ADMITTED',plan=pr,max_invocations=1,remaining_invocations=1,claimed_invocations=0,
        completed_invocations=0,relaunch_allowed=False,execution_authorized=True,data_reads_authorized=True,
        checkpoint_reads_authorized=True,backbone_encodings_authorized=33,head_tensor_imports_authorized=0,
        candidate_weights_writes_authorized=0,backward_calls_authorized=0,optimizer_steps_authorized=0,
        training_authorized=False,**w.FUSION_INITIAL_BOUNDS)
    policy={k:False for k in ('training_enabled','native_training_enabled','native_launch_authorized','full_epoch_training_allowed')}
    policy[w.FUSION_INITIAL_STAGE]=gate
    docs={str(path):plan,refs['initial_measurement']['path']:baseline,refs['trained_result']['path']:trained,
        refs['trained_terminal']['path']:terminal,refs['trained_review']['path']:review,policy_ref['path']:policy}
    monkeypatch.setattr(w,'_read',lambda ref:copy.deepcopy(docs[ref['path']]))
    monkeypatch.setattr(w,'_ref',lambda p:policy_ref)
    monkeypatch.setattr(w,'_git',lambda *a:'work/gx1-current' if a[0]=='branch' else 'b'*40)
    monkeypatch.setattr(w.subprocess,'run',lambda *a,**k:None)
    monkeypatch.setattr(w.campaign,'require_binding',lambda ref,**kw:seen.append(ref['path']) or ref)
    monkeypatch.setattr(w.campaign,'require_clean_source',lambda *a,**k:None)
    monkeypatch.setattr(w,'_qualify_reversed_origin',lambda p,b:
        seen.append('original_qualifier') or dict(recipe=recipe,view={'contract_sha256':'d'*64},current_sources={}))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('metadata preflight decoded arrays'))
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('metadata preflight decoded checkpoint'))
    checked=w._inspect_fusion_observation(path,'a'*64,tmp_path,execute=False,initial=True)
    assert checked['plan']==plan and 'original_qualifier' in seen
    assert all(refs[k]['path'] in seen for k in ('initial_checkpoint','initial_baseline','trained_train','trained_later'))
    assert not (tmp_path/'CLAIM.json').exists()
    assert w._inspect_fusion_observation(path,'a'*64,tmp_path,execute=True,initial=True)['plan']==plan
    for kind,mutate,reason in [
        ('initial_measurement',lambda x:x.update(evaluated_model_state_sha256='0'*64),'BASELINE'),
        ('initial_measurement',lambda x:x.update(optimizer_updates=1),'BASELINE'),
        ('initial_measurement',lambda x:x['fixed_subset_positions'].__setitem__(0,1),'BASELINE'),
        ('trained_result',lambda x:x['batch_indices']['TRAIN'].__setitem__(0,1),'MATCHED_TRAINED'),
        ('trained_result',lambda x:x['passivity'].update(instrumented_matches_uninstrumented=False),'MATCHED_TRAINED'),
        ('trained_terminal',lambda x:x.update(status='FAILED_CONSUMED'),'MATCHED_TRAINED'),
        ('trained_review',lambda x:x['partitions']['LATER'].update(unique_parents=255),'MATCHED_TRAINED')]:
        key=refs[kind]['path'];original=docs[key];bad=copy.deepcopy(original);mutate(bad);docs[key]=bad
        with pytest.raises(RuntimeError,match='FUSION_INITIAL_'+reason):
            w._inspect_fusion_observation(path,'a'*64,tmp_path,execute=False,initial=True)
        docs[key]=original
    gate['head_tensor_imports_authorized']=4
    with pytest.raises(RuntimeError,match='FUSION_ADMISSION'):
        w._inspect_fusion_observation(path,'a'*64,tmp_path,execute=True,initial=True)


def _synthetic_fusion_checkpoint(monkeypatch, *, initial):
    import copy
    import torch
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    from gx1.contracts import entry_native_direction_research_state_v1 as old
    from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as model_sha
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__();self.linear=torch.nn.Linear(2,2)
            self.register_buffer('dynamic',torch.ones(2),persistent=False)
        def require_input_normalization_state(self): pass
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.state_model=torch.nn.Module();self.state_model.backbone=Backbone()
            self.state_model.feature_view_identity={'contract_sha256':'a'*64}
            self.research_objective={'contract_sha256':'b'*64}
        @property
        def backbone(self): return self.state_model.backbone
    model=Model();pairs=list(model.named_parameters());order=torch.arange(4,dtype=torch.int64)
    numerical={'seed':42,'optimizer':{'kwargs':{'lr':1e-4,'betas':[.9,.999]}}}
    monkeypatch.setattr(state,'_parameters',lambda m:(list(m.named_parameters()),list(m.named_parameters()),[]))
    monkeypatch.setattr(state,'_numerical',lambda n,m:copy.deepcopy(n))
    monkeypatch.setattr(state,'_require_model_binding',lambda m:None)
    monkeypatch.setattr(state,'build_fold1_fit_optimizer',lambda *a,**k:pytest.fail('optimizer constructed'))
    monkeypatch.setattr(state,'restore_fold1_fit_state',lambda *a,**k:pytest.fail('full state restored'))
    monkeypatch.setattr(trainer,'_restore_attended_session_rng_state',lambda *a,**k:pytest.fail('RNG restored'))
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('fixture must not deserialize'))
    binding=dict(schema_version=state.TA_BINDING_SCHEMA,numerical_recipe=numerical,
        feature_view_identity=model.state_model.feature_view_identity,research_objective=model.research_objective,
        active_parameter_names=[n for n,_ in pairs],inactive_parameter_names=[],
        parameter_inventory=[dict(name=n,shape=list(p.shape),dtype=str(p.dtype)) for n,p in pairs],
        optimizer_group_parameter_names=[[n for n,_ in pairs]],ema_created=False,scheduler_created=False,target_model_created=False)
    binding['contract_sha256']=old._tree_sha(binding)
    rng=copy.deepcopy(trainer._attended_session_rng_state(device=torch.device('cpu')))
    rng['torch_cuda']=[torch.zeros(8,dtype=torch.uint8)]
    ck=dict(model_state=copy.deepcopy(model.state_dict()),buffer_state=copy.deepcopy(dict(model.named_buffers())),
        optimizer_state={'state':{},'param_groups':[dict(lr=1e-4,betas=(.9,.999),params=list(range(len(pairs))))]},
        rng_state=rng,module_training={n:m.training for n,m in model.named_modules()},parent_order=order.clone())
    digest=model_sha(ck['model_state'])
    refs={'initial_checkpoint':{'path':'synthetic.pt','sha256':'c'*64},'reverse_recipe':{'path':'synthetic.json','sha256':'d'*64}}
    recipe=dict(numerical_recipe=numerical,expected_view_sha256='a'*64,expected_initial_model_state_sha256=digest,
        input_refs={'original_initial_checkpoint':refs['initial_checkpoint']},batch_indices=[0],batch_order_sha256='e'*64)
    checked=dict(recipe=recipe,plan={'inputs':refs},origin={'original_result':{
        'source_commit':'f'*40,'parent_order_sha256':old._tree_sha(order)}},result={'parent_order_sha256':old._tree_sha(order)})
    if initial:
        monkeypatch.setattr(w,'FUSION_INITIAL_BOUNDS',{**w.FUSION_INITIAL_BOUNDS,'candidate_model_state_sha256':digest})
        ck.update(schema_version=state.TA_STATE_SCHEMA,source_commit='f'*40,fold_id=1,feature_view_sha256='a'*64,
            binding=binding,cursor=0,optimizer_updates=0,skipped_batches=0)
    else:
        monkeypatch.setattr(w,'REVERSED_MODEL',digest)
        ck.update(schema_version='gx1_ta_m5_full_reverse_order_evidence_v1',arm='REVERSED',recipe=refs['reverse_recipe'],
            source_commit=w.REVERSED_SOURCE,original_checkpoint=refs['initial_checkpoint'],canonical_resume_authority=False,
            parent_cursor=w.FIT_ROWS,physical_optimizer_updates=1849,pending_gradients=0,parameter_binding=binding,
            original_numerical_recipe_reference=numerical,test_data_used=False,
            method_state=dict(batch_indices=[0],batch_order_sha256='e'*64,full_order_sha256=old._tree_sha(order),
                original_fit_batch_starts=[0],pass_index=0,all_original_rows_once=True,tail_policy='original_tail_last',
                loss_and_optimizer_unchanged=True))
        ck['optimizer_state']['state'][0]=dict(step=torch.tensor(1.),exp_avg=torch.zeros_like(pairs[0][1]),
            exp_avg_sq=torch.zeros_like(pairs[0][1]))
    def seal(value):
        value['model_state_sha256']=model_sha(value['model_state'])
        for key in ('buffer_state','optimizer_state','rng_state'): value[key+'_sha256']=old._tree_sha(value[key])
        key='contract_sha256' if initial else 'evidence_sha256'
        value[key]=old._tree_sha({k:v for k,v in value.items() if k!=key})
    seal(ck)
    with torch.no_grad():
        for p in model.parameters(): p.add_(1.)
        model.backbone.dynamic.add_(2.)
    return model,ck,checked,order,seal


@pytest.mark.parametrize('initial',[False,True])
def test_fusion_weight_loader_preserves_rng_and_never_restores_optimizer(monkeypatch,initial):
    import torch
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    from gx1.contracts import entry_native_direction_research_state_v1 as old
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    model,ck,checked,order,_=_synthetic_fusion_checkpoint(monkeypatch,initial=initial)
    before=old._tree_sha(trainer._attended_session_rng_state(device=torch.device('cpu')))
    saved=old._tree_sha(ck);ids={n:id(p) for n,p in model.named_parameters()}
    result=w._load_reversed_weights(model,ck,checked,order,torch=torch,initial=initial)
    assert result['model_state_sha256']==ck['model_state_sha256']
    assert old._tree_sha(dict(model.named_buffers()))==ck['buffer_state_sha256']
    assert result['optimizer_state_restored'] is result['cuda_rng_restored'] is False
    assert old._tree_sha(trainer._attended_session_rng_state(device=torch.device('cpu')))==before
    assert old._tree_sha(ck)==saved and {n:id(p) for n,p in model.named_parameters()}==ids
    assert not any(m.training for m in model.modules()) and not any(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize('change,reason',[
    ('cursor','INITIAL_STATE_HEADER'),('source','INITIAL_STATE_HEADER'),('extra','INITIAL_STATE_HEADER'),
    ('seal','INITIAL_STATE_SEAL'),('order','INITIAL_STATE_ORDER'),('binding','PARAMETER_BINDING'),
    ('history','OPTIMIZER_ORDER'),('mode','MODULE_MODES'),('buffer','REVERSED_BUFFERS_TENSOR')])
def test_initial_fusion_loader_rejects_malformed_state_before_mutation(monkeypatch,change,reason):
    import torch
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    from gx1.contracts import entry_native_direction_research_state_v1 as old
    model,ck,checked,order,seal=_synthetic_fusion_checkpoint(monkeypatch,initial=True)
    before=old._tree_sha(dict(model=model.state_dict(),buffers=dict(model.named_buffers())))
    if change=='cursor': ck['cursor']=False
    elif change=='source': ck['source_commit']='0'*40
    elif change=='extra': ck['unexpected']=True
    elif change=='order': ck['parent_order']=ck['parent_order'].flip(0)
    elif change=='binding': ck['binding']['active_parameter_names']=[]
    elif change=='history': ck['optimizer_state']['state'][0]={}
    elif change=='mode': ck['module_training'].pop('')
    elif change=='buffer': ck['buffer_state']['state_model.backbone.dynamic'].fill_(float('nan'))
    seal(ck)
    if change=='seal': ck['contract_sha256']='0'*64
    with pytest.raises(RuntimeError,match=reason):
        w._load_reversed_weights(model,ck,checked,order,torch=torch,initial=True)
    assert old._tree_sha(dict(model=model.state_dict(),buffers=dict(model.named_buffers())))==before
    assert all(p.requires_grad for p in model.parameters()) and model.training


def test_initial_fusion_explicit_root_preserves_original_outputs(tmp_path,monkeypatch):
    import json
    import numpy as np
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    original=tmp_path/'trained';original.mkdir();(original/'RESULT.json').write_text('sealed')
    monkeypatch.setattr(w,'FUSION_ROOT',original)
    for complete in (True,False):
        initial=tmp_path/str(complete);initial.mkdir();counts={'published_npz_bytes':0}
        arrays={k:np.arange(16,dtype=np.int64) for k in w.FUSION_MATCH_KEYS}
        assert w._fusion_exact_arrays(arrays,{k:v.copy() for k,v in arrays.items()})
        for key in w.FUSION_MATCH_KEYS:
            changed={k:v.copy() for k,v in arrays.items()};changed[key][0]+=1
            assert not w._fusion_exact_arrays(arrays,changed)
        name='TRAIN_FUSION.npz' if complete else 'TRAIN_FUSION_PARTIAL.npz'
        artifact=w._save_fusion_arrays(name,arrays,counts,np=np,root=initial)
        published=w._publish_fusion_outcome({'path':str(initial/'PLAN.json'),'sha256':'a'*64},
            dict(source_unchanged=True,counts=counts,artifacts={'TRAIN':artifact}),complete=complete,root=initial)
        terminal=json.loads(Path(published['terminal']['path']).read_text())
        assert terminal['exit_code']==(0 if complete else 1) and terminal['no_relaunch_under_this_plan']
        assert Path(artifact['path']).parent==initial and (original/'RESULT.json').read_text()=='sealed'


def retained_mtf_plan():
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    return dict(fusion_plan(),stage=w.MTF_RETAINED_STAGE,inputs=w.MTF_RETAINED_REFS,**w.MTF_RETAINED_BOUNDS)


def test_retained_mtf_exact_scope_and_dispatch(monkeypatch):
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    plan=retained_mtf_plan();root=w.MTF_RETAINED_ROOT;path=root/'PLAN.json'
    assert w._fusion_plan(plan,path,root,retained=True)==plan
    for key,value in [('max_backbone_encodings',1),('max_joint_calls',65),('mean_vector_estimates',3),
                      ('predictor_fits',1),('head_tensor_imports',0),('mean_arithmetic','float32'),
                      ('baseline_parity','allclose'),('max_checkpoint_writes',1)]:
        with pytest.raises(RuntimeError,match='FUSION_EXACT_PLAN'):
            w._fusion_plan({**plan,key:value},path,root,retained=True)
    with pytest.raises(RuntimeError,match='FUSION_EXACT_PLAN'):
        w._fusion_plan(plan,w.FUSION_ROOT/'PLAN.json',w.FUSION_ROOT,retained=True)
    with pytest.raises(RuntimeError,match='FUSION_MODE'):
        w._fusion_plan(plan,path,root,retained=True,initial=True)
    calls=[]
    monkeypatch.setattr(w,'_inspect_fusion_observation',lambda *a,**kw:
        calls.append(('inspect',kw)) or dict(plan_ref={},view={'contract_sha256':'a'*64},current_sources={}))
    monkeypatch.setattr(w,'_run_fusion_observation',lambda *a,**kw:calls.append(('run',kw)) or {'synthetic':True})
    report=w.preflight_read_only(path,'a'*64,root)
    assert report['archive_decodes']==report['backbone_encodings']==report['model_constructions']==report['joint_calls']==0
    assert w.run(path,'a'*64,root,device='cuda')=={'synthetic':True}
    assert calls==[('inspect',{'execute':False,'retained':True}),('run',{'device':'cuda','retained':True})]


def test_retained_mtf_means_are_fp64_then_fp32_and_preserve_other_blocks():
    import numpy as np
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    source=np.zeros((256,512),np.float32)
    pattern=np.tile(np.asarray([1e8,1.,-1e8,1.],np.float32),64)
    source[:,128:256]=pattern[:,None];source[:,256:384]=pattern[:,None]
    source[0,0]=np.float32(-0.);source[:,384:]=np.arange(128,dtype=np.float32)
    original=source.copy();means=w._retained_mtf_means(source,np=np)
    assert all(v.dtype==np.float32 and np.array_equal(v,np.full(128,.5,np.float32)) for v in means.values())
    altered=w._retained_mtf_intervention(source[:16],means,np=np)
    assert source.tobytes()==original.tobytes()
    assert altered[:,:128].tobytes()==source[:16,:128].tobytes()
    assert altered[:,384:].tobytes()==source[:16,384:].tobytes()
    assert np.array_equal(altered[:,128:256],source[:16,:128]+means['mean_mtf_residual'])
    assert np.array_equal(altered[:,256:384],np.broadcast_to(means['mean_mtf_pool'],(16,128)))
    with pytest.raises(RuntimeError,match='RETAINED_TRAIN_MEAN_SOURCE'):
        w._retained_mtf_means(source[:16],np=np)
    with pytest.raises(RuntimeError,match='RETAINED_INTERVENTION_INPUT'):
        w._retained_mtf_intervention(source[:16].astype(np.float64),means,np=np)


def test_retained_scores_keep_zero_known_groups_and_pool_by_observations():
    import numpy as np
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    arrays={}
    for task in ('state','future'):
        arrays[task+'_allowed']=np.ones((256,5),np.uint8);arrays[task+'_allowed'][0,0]=4
        arrays[task+'_eligible']=np.zeros((256,5),np.bool_);arrays[task+'_mature']=np.ones((256,5),np.bool_)
        arrays[task+'_eligible'][0,0]=True;arrays[task+'_eligible'][16:32,0]=True
        arrays['original_'+task+'_activity_side_logits']=np.zeros((256,5,2),np.float32)
        arrays['intervention_'+task+'_activity_side_logits']=np.ones((256,5,2),np.float32)
    scores=w._retained_conditional_scores(arrays,np=np)
    for task in ('state','future'):
        s=scores[task]['M5'];p=1/(1+np.exp(-1.))
        assert s['role']=='PRIMARY_M5' and s['original']['count']==17 and s['original']['up_count']==1
        assert s['original']['brier']==.25
        assert s['intervention']['brier']==pytest.approx(((p-1)**2+16*p*p)/17)
        assert len(s['groups'])==16 and [g['group'] for g in s['groups']]==list(range(16))
        assert s['groups'][2]['original']==dict(count=0,up_count=0,brier=None,log=None,mean_p=None)
        assert s['groups'][2]['delta']=={'brier':None,'log':None}
        assert s['delta']['brier']==pytest.approx(sum(g['original']['count']*g['delta']['brier']
            for g in s['groups'] if g['original']['count'])/17)
        assert scores[task]['D1']['role']=='DESCRIPTIVE_SECONDARY' and scores[task]['D1']['delta']['log'] is None


def _retained_test_counts():
    return dict(backbone_encodings=0,joint_calls=0,joint_linear_calls=0,query_projection_calls=0,
        native_head_calls=0,m5_head_calls=0,state_decoder_calls=0,state_decoder_returns=0,
        future_decoder_calls=0,future_decoder_returns=0,archive_decodes=1,mean_vector_estimates=0,
        predictor_fits=0,published_npz_bytes=0,sampled_rows=0)


def _retained_test_model(monkeypatch):
    import torch
    from gx1.contracts import market_state_time_query_native_F_batch_v1 as batch
    monkeypatch.setattr(batch,'_ta_only',lambda m:True)
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__();self.entry_q_joint_norm=torch.nn.LayerNorm(512);self.entry_q_joint_in=torch.nn.Linear(512,128)
        def forward(self,*a,**k): raise AssertionError('toy backbone must never execute')
    class State(torch.nn.Module):
        def __init__(self):
            super().__init__();self.backbone=Backbone();self._dedicated_m5_side=True
            self.query_projection=torch.nn.Sequential(torch.nn.Linear(128,128),torch.nn.GELU())
            self.query_head=torch.nn.Linear(128,2);self.m5_side_head=torch.nn.Linear(128,1)
            self.register_buffer('query_native_k',torch.tensor([1,3,12,48,288],dtype=torch.int64))
        def decode_raw_query_grid(self,h,q):
            assert q.shape==(16,1,5) and torch.equal(q[:,0],self.query_native_k.expand(16,-1))
            rep=self.query_projection(h[:,None,None,:].expand(-1,1,5,-1))
            out=self.query_head(rep).clone();out[...,0,1]=self.m5_side_head(rep[...,0,:]).squeeze(-1);return out
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.state_model=State();self.future_head=torch.nn.Linear(128,2)
            self.future_m5_side_head=torch.nn.Linear(128,1);self.fail_future=False
        @property
        def backbone(self):return self.state_model.backbone
        def decode_native_F_raw(self,h):
            if self.fail_future:raise RuntimeError('synthetic future failure')
            rep=self.state_model.query_projection(h[:,None,:].expand(-1,5,-1))
            out=self.future_head(rep).clone();out[:,0,1]=self.future_m5_side_head(rep[:,0,:]).squeeze(-1);return out
    model=Model().eval()
    for p in model.parameters():p.requires_grad_(False)
    return model


def test_retained_decoder_hooks_count_pure_calls_reject_backbone_and_cleanup(monkeypatch):
    import torch
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    model=_retained_test_model(monkeypatch);counts=_retained_test_counts();inventory=_fusion_hook_inventory(model)
    source=torch.arange(16*512,dtype=torch.float32).reshape(16,512)/1000
    query=model.state_model.query_native_k.expand(16,-1)
    rng=torch.get_rng_state().clone();state={k:v.clone() for k,v in model.state_dict().items()}
    with w._retained_decoder_hooks(model,counts):
        one=w._decode_retained_joint(model,source,query,counts,torch=torch)
        two=w._decode_retained_joint(model,source,query,counts,torch=torch)
    assert w._fusion_exact_arrays(one,two) and torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
    assert counts['backbone_encodings']==0 and counts['joint_calls']==counts['joint_linear_calls']==2
    assert counts['state_decoder_calls']==counts['future_decoder_calls']==2
    assert counts['query_projection_calls']==counts['native_head_calls']==counts['m5_head_calls']==4
    assert _fusion_hook_inventory(model)==inventory
    for failure in ('backbone','future','limit'):
        counts=_retained_test_counts();model.fail_future=failure=='future'
        if failure=='limit':counts['joint_calls']=64
        with pytest.raises(RuntimeError,match={'backbone':'BACKBONE_CALL_FORBIDDEN','future':'synthetic future failure','limit':'CALL_LIMIT'}[failure]):
            with w._retained_decoder_hooks(model,counts):
                if failure=='backbone':model.backbone(source)
                else:w._decode_retained_joint(model,source,query,counts,torch=torch)
        assert _fusion_hook_inventory(model)==inventory


@pytest.mark.parametrize('mismatch',[False,True])
def test_retained_panels_lock_means_before_later_and_stop_on_bit_mismatch(tmp_path,monkeypatch,mismatch):
    import json
    import numpy as np
    import torch
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    from gx1.scripts.run_market_state_time_query_fold1_fit_window_v1 import _ta_m5_capture_layout
    model=_retained_test_model(monkeypatch);inventory=_fusion_hook_inventory(model);panels={}
    reference_counts=_retained_test_counts()
    with w._retained_decoder_hooks(model,reference_counts):
        for index,panel in enumerate(('TRAIN','LATER')):
            arrays={k:np.zeros(shape,dtype) for k,(shape,dtype) in _ta_m5_capture_layout(256).items()}
            arrays['parents']=np.arange(256,dtype=np.int64)+index*1000
            arrays['issues']=arrays['parents']+10000
            arrays['query_native']=np.broadcast_to(np.array([1,3,12,48,288],np.int64),(256,5)).copy()
            source=(np.arange(256*512,dtype=np.float32).reshape(256,512)%101)/100+np.float32(index/10)
            arrays['capture_entry_q_joint_source']=source;parts={}
            for left in range(0,256,16):
                decoded=w._decode_retained_joint(model,torch.from_numpy(source[left:left+16]),
                    torch.from_numpy(arrays['query_native'][left:left+16]),reference_counts,torch=torch)
                for k,v in decoded.items():parts.setdefault(k,[]).append(v)
            arrays.update({k:np.concatenate(v) for k,v in parts.items()})
            for task in ('state','future'):
                arrays[task+'_allowed'][:]=1;arrays[task+'_eligible'][:]=True;arrays[task+'_mature'][:]=True
            panels[panel]=arrays
    if mismatch:
        a=panels['TRAIN']['state_activity_side_logits'];a[16,0,1]=np.nextafter(a[16,0,1],np.float32(np.inf))
    class Archive:
        def __init__(self,value):self.value=value;self.files=list(value)
        def __getitem__(self,k):return self.value[k]
        def __enter__(self):return self
        def __exit__(self,*a):return False
    opened=[]
    def load(path,**kw):
        panel='TRAIN' if path=='synthetic_train' else 'LATER';opened.append(panel)
        if panel=='LATER':
            receipt=json.loads((tmp_path/'TRAIN_MTF_MEANS.json').read_text())
            assert receipt['later_archive_opened'] is False and receipt['mean_vector_estimates']==2
            assert (tmp_path/'TRAIN_MTF_MEANS.npz').exists()
        return Archive(panels[panel])
    monkeypatch.setattr(np,'load',load)
    refs={'trained_train':{'path':'synthetic_train','sha256':'a'*64},
          'trained_later':{'path':'synthetic_later','sha256':'b'*64},'trained_result':{'path':'synthetic_result','sha256':'c'*64}}
    checked={'plan_ref':{'path':str(tmp_path/'PLAN.json'),'sha256':'d'*64},'plan':{'inputs':refs}}
    counts=_retained_test_counts();artifacts={};passivity={};rng=torch.get_rng_state().clone()
    if mismatch:
        with pytest.raises(RuntimeError,match='RETAINED_BASELINE_PARITY:state_activity_side_logits'):
            w._run_retained_mtf_panels(model,checked,counts,artifacts,passivity,np=np,torch=torch)
        assert opened==['TRAIN'] and counts['sampled_rows']==16 and counts['joint_calls']==3
        assert 'baseline_parity_failure' in artifacts and 'TRAIN_partial' in artifacts
        assert json.loads((tmp_path/'BASELINE_PARITY_FAILURE.json').read_text())['tolerance_fallback'] is False
    else:
        report=w._run_retained_mtf_panels(model,checked,counts,artifacts,passivity,np=np,torch=torch)
        assert opened==['TRAIN','LATER'] and counts['sampled_rows']==512 and counts['archive_decodes']==3
        assert counts['joint_calls']==counts['state_decoder_calls']==counts['future_decoder_calls']==64
        assert counts['native_head_calls']==counts['query_projection_calls']==counts['m5_head_calls']==128
        assert counts['mean_vector_estimates']==2 and counts['predictor_fits']==0 and all(passivity.values())
        assert report['baseline_tolerance_fallback'] is report['model_change_recommended'] is False
        assert report['conditional_scores']['LATER']['state']['M5']['original']['count']==256
        assert all((tmp_path/(p+'_INTERVENTION.npz')).exists() for p in ('TRAIN','LATER'))
    assert counts['backbone_encodings']==0 and _fusion_hook_inventory(model)==inventory
    assert torch.equal(rng,torch.get_rng_state())


def temporal_boundary_plan():
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    return dict(fusion_plan(),stage=w.TEMPORAL_STAGE,inputs=w.TEMPORAL_REFS,**w.TEMPORAL_BOUNDS)


def test_temporal_exact_plan_bounds_and_metadata_only_dispatch(monkeypatch):
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    plan=temporal_boundary_plan();root=w.TEMPORAL_ROOT;path=root/'PLAN.json'
    assert w._fusion_plan(plan,path,root,temporal=True)==plan
    for key,value in [('max_backbone_encodings',32),('batch_equal_length_mtf_eval',True),
                      ('temporal_lengths',[16,64,96,96,251]),('temporal_lengths',[16.,64,96,96,252]),
                      ('max_temporal_family_encoder_calls',1281),
                      ('max_temporal_family_axis_calls',33),('max_cache_bytes',128*1024**2),
                      ('head_tensor_imports',0),('max_optimizer_updates',1)]:
        with pytest.raises(RuntimeError,match='FUSION_EXACT_PLAN'):
            w._fusion_plan({**plan,key:value},path,root,temporal=True)
    with pytest.raises(RuntimeError,match='FUSION_EXACT_PLAN'):
        w._fusion_plan(plan,w.FUSION_ROOT/'PLAN.json',w.FUSION_ROOT,temporal=True)
    with pytest.raises(RuntimeError,match='FUSION_MODE'):
        w._fusion_plan(plan,path,root,temporal=True,retained=True)
    calls=[]
    monkeypatch.setattr(w,'_inspect_fusion_observation',lambda *a,**kw:
        calls.append(('inspect',kw)) or dict(plan_ref={},view={'contract_sha256':'a'*64},current_sources={}))
    monkeypatch.setattr(w,'_run_fusion_observation',lambda *a,**kw:calls.append(('run',kw)) or {'synthetic':True})
    report=w.preflight_read_only(path,'a'*64,root)
    assert report['archive_decodes']==report['backbone_encodings']==report['model_constructions']==0
    assert w.run(path,'a'*64,root,device='cuda')=={'synthetic':True}
    assert calls==[('inspect',{'execute':False,'temporal':True}),('run',{'device':'cuda','temporal':True})]


def _temporal_test_counts():
    return dict(temporal_family_encoder_calls=0,temporal_family_axis_calls=0,
        temporal_timeframe_axis_calls=0,temporal_cross_tf_calls=0,temporal_observed_forwards=0)


def _synthetic_temporal_backbone(torch):
    """Small actual TransformerEncoder parents; no native model or real data."""
    from types import SimpleNamespace
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._mtf_specialist_names=tuple('f'+str(i) for i in range(8))
            self.cfg=SimpleNamespace(d_model=4,m5_seq_len=16,m15_seq_len=64,h1_seq_len=96,h4_seq_len=96,d1_seq_len=252)
            def encoder():
                return torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(
                    d_model=4,nhead=2,dim_feedforward=8,dropout=.2,batch_first=True,activation='gelu',norm_first=True),
                    num_layers=1,norm=torch.nn.LayerNorm(4,elementwise_affine=False),enable_nested_tensor=False)
            self.mtf_family_encoder=torch.nn.ModuleDict({name:encoder() for name in self._mtf_specialist_names})
            self.family_axis_attn=encoder();self.timeframe_axis_attn=encoder();self.cross_tf_attn=encoder()
            self.family_tf_token_identity=torch.nn.Parameter(torch.arange(160).reshape(1,5,8,4).float()/301)
            self.fault=None;self.record={}

        def forward(self,x):
            batch=len(x);grids=[];self.record={}
            for tf,length in enumerate((16,64,96,96,252)):
                vectors=[]
                order=list(range(8))
                if self.fault=='order' and tf==0:order[:2]=[1,0]
                for family in order:
                    value=x[:,None,:]+torch.arange(length).float()[None,:,None]/19+family/11
                    if tf==family==0:
                        if self.fault=='shape':value=value[:,:-1]
                        elif self.fault=='dtype':value=value.double()
                        elif self.fault=='nonfinite':value[0,0,0]=float('nan')
                    output=self.mtf_family_encoder[self._mtf_specialist_names[family]](value,src_key_padding_mask=None)
                    self.record[tf,family]=(value.detach().clone(),output.detach().clone())
                    vectors.append(output.mean(dim=1)+output[:,-1,:])
                grids.append(torch.stack(vectors,dim=1))
            grid=torch.stack(grids,dim=1)+self.family_tf_token_identity
            family_input=grid.permute(0,2,1,3).reshape(batch*8,5,4)
            if self.fault=='family_parity':family_input[0,0,0]+=1e-3
            family_output=self.family_axis_attn(family_input,src_key_padding_mask=None)
            grid=family_output.reshape(batch,8,5,4).permute(0,2,1,3)
            self.record['family_axis_output']=grid.detach().clone()
            timeframe_input=grid.reshape(batch*5,8,4)
            if self.fault=='timeframe_parity':timeframe_input[0,0,0]+=1e-3
            grid=self.timeframe_axis_attn(timeframe_input).reshape(batch,5,8,4)
            self.record['timeframe_axis_output']=grid.detach().clone()
            tokens=grid.mean(dim=2)
            self.record['cross_tf_input']=tokens.detach().clone()
            output=tokens if self.fault=='missing' else self.cross_tf_attn(tokens)
            self.record['cross_tf_output']=output.detach().clone()
            return output
    return Backbone().eval().requires_grad_(False)


def test_temporal_parent_hooks_preserve_fp32_output_rng_state_and_fastpath_children():
    import torch
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71);model=_synthetic_temporal_backbone(torch)
        x=torch.arange(8).reshape(2,4).float()/7
        state={k:v.detach().clone() for k,v in model.state_dict().items()}
        parameters=[(id(p),p.requires_grad,p.grad) for p in model.parameters()]
        modes=[m.training for m in model.modules()]
        hooks=_fusion_hook_inventory(model);rng=torch.get_rng_state().clone();counts=_temporal_test_counts()
        with torch.inference_mode():
            expected=model(x)
            with w.observe_ta_temporal_once(model,batch_rows=2,max_bytes=9280,counts=counts,torch=torch) as captured:
                actual=model(x)
                # Only parent encoders are touched; layer hook inventory is unchanged.
                for m in model.modules():
                    if not isinstance(m,torch.nn.TransformerEncoder):
                        assert (tuple(m._forward_hooks),tuple(m._forward_pre_hooks))==hooks[id(m)]
        assert torch.equal(actual.view(torch.int32),expected.view(torch.int32))
        assert len(captured)==10 and sum(t.numel()*t.element_size() for t in captured.values())==9280
        assert counts==dict(temporal_family_encoder_calls=40,temporal_family_axis_calls=1,
            temporal_timeframe_axis_calls=1,temporal_cross_tf_calls=1,temporal_observed_forwards=1)
        for tf in range(5):
            for family in range(8):
                for index,label in enumerate(('input','output')):
                    value=model.record[tf,family][index];wide=value.double();prefix='temporal_'+label
                    assert torch.equal(captured[prefix+'_mean'][:,tf,family],value.mean(dim=1))
                    assert torch.equal(captured[prefix+'_last'][:,tf,family],value[:,-1])
                    moments=torch.stack((wide.square().mean((1,2)).sqrt(),
                        (wide-wide.mean(1,keepdim=True)).square().mean((1,2)).sqrt()),dim=1).float()
                    assert torch.equal(captured[prefix+'_moments'][:,tf,family],moments)
        for key in ('family_axis_output','timeframe_axis_output','cross_tf_input','cross_tf_output'):
            assert torch.equal(captured[key],model.record[key])
            assert captured[key].data_ptr()!=model.record[key].data_ptr()
        assert all(not t.requires_grad and t.grad_fn is None for t in captured.values())
        assert torch.equal(torch.get_rng_state(),rng) and _fusion_hook_inventory(model)==hooks
        assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
        assert [(id(p),p.requires_grad,p.grad) for p in model.parameters()]==parameters
        assert [m.training for m in model.modules()]==modes


def test_temporal_readout_uses_native_fp32_mean_not_fp64_mean(monkeypatch):
    import torch
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    model=_synthetic_temporal_backbone(torch)
    def cancellation(src,**kwargs):
        result=torch.zeros_like(src)
        result[:,0,:]=2**24;result[:,1,:]=1;result[:,2,:]=-(2**24)
        return result
    monkeypatch.setattr(model.mtf_family_encoder['f0'],'forward',cancellation)
    with torch.inference_mode(),w.observe_ta_temporal_once(
            model,batch_rows=2,max_bytes=9280,counts=_temporal_test_counts(),torch=torch) as captured:
        model(torch.ones(2,4))
    source=model.record[0,0][1]
    assert not torch.equal(source.mean(1),source.double().mean(1).float())
    assert torch.equal(captured['temporal_output_mean'][:,0,0],source.mean(1))


@pytest.mark.parametrize('fault,error',[
    ('order','TEMPORAL_ROUTE_ORDER'),('shape','TEMPORAL_TENSOR:family_input'),
    ('dtype','TEMPORAL_TENSOR:family_input'),('nonfinite','TEMPORAL_NONFINITE:family_input'),
    ('family_parity','TEMPORAL_BIT_PARITY:family_axis_input'),
    ('timeframe_parity','TEMPORAL_BIT_PARITY:timeframe_axis_input'),('missing','TEMPORAL_INCOMPLETE')])
def test_temporal_rejects_wrong_order_shape_values_or_axis_parity_and_cleans_hooks(fault,error):
    import torch
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    model=_synthetic_temporal_backbone(torch);model.fault=fault;hooks=_fusion_hook_inventory(model)
    with pytest.raises(RuntimeError,match=error):
        with torch.inference_mode(),w.observe_ta_temporal_once(
                model,batch_rows=2,max_bytes=9280,counts=_temporal_test_counts(),torch=torch) as captured:
            model(torch.ones(2,4))
    assert captured=={} and _fusion_hook_inventory(model)==hooks


def test_temporal_rejects_merged_mode_lengths_bytes_and_call_limit_before_hooks():
    import torch
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import _BATCH_EQUAL_LENGTH_MTF_EVAL
    model=_synthetic_temporal_backbone(torch);hooks=_fusion_hook_inventory(model)
    for fault,error in [('merged','TEMPORAL_BATCH_MODE'),('length','TEMPORAL_LAYOUT'),
                        ('bytes','TEMPORAL_BYTE_LIMIT'),('counter','TEMPORAL_COUNTERS')]:
        counts=_temporal_test_counts();limit=9279 if fault=='bytes' else 9280
        if fault=='counter':counts['temporal_family_encoder_calls']=1280
        if fault=='length':model.cfg.h1_seq_len=97
        token=_BATCH_EQUAL_LENGTH_MTF_EVAL.set(fault=='merged')
        try:
            with pytest.raises(RuntimeError,match=error):
                with w.observe_ta_temporal_once(model,batch_rows=2,max_bytes=limit,counts=counts,torch=torch):
                    pytest.fail('invalid temporal layout admitted')
        finally:
            _BATCH_EQUAL_LENGTH_MTF_EVAL.reset(token);model.cfg.h1_seq_len=96
        assert _fusion_hook_inventory(model)==hooks


def test_temporal_cleans_partial_registration_caller_and_swallowed_hook_errors(monkeypatch):
    import torch
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    model=_synthetic_temporal_backbone(torch);hooks=_fusion_hook_inventory(model)
    def reject(*args,**kwargs):raise RuntimeError('registration failed')
    with monkeypatch.context() as patch:
        patch.setattr(model.mtf_family_encoder['f1'],'register_forward_hook',reject)
        with pytest.raises(RuntimeError,match='registration failed'):
            with w.observe_ta_temporal_once(model,batch_rows=2,max_bytes=9280,counts=_temporal_test_counts(),torch=torch):
                pytest.fail('partial registration admitted')
    assert _fusion_hook_inventory(model)==hooks
    for swallowed in (False,True):
        with pytest.raises(RuntimeError,match='TEMPORAL_INCOMPLETE' if swallowed else 'caller failed'):
            with torch.inference_mode(),w.observe_ta_temporal_once(
                    model,batch_rows=2,max_bytes=9280,counts=_temporal_test_counts(),torch=torch) as captured:
                model(torch.ones(2,4))
                if swallowed:
                    with pytest.raises(RuntimeError,match='TEMPORAL_ROUTE_ORDER'):model(torch.ones(2,4))
                else:raise RuntimeError('caller failed')
        assert captured=={} and _fusion_hook_inventory(model)==hooks


def test_temporal_npz_aggregate_cap_is_explicit_and_does_not_expand_old_route(tmp_path,monkeypatch):
    import numpy as np
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    root=tmp_path/'temporal';root.mkdir();monkeypatch.setattr(w,'TEMPORAL_ROOT',root)
    arrays={'parents':np.arange(2,dtype=np.int64),'boundary_temporal_input_mean':np.zeros((2,5,8,4),np.float32)}
    counts={'published_npz_bytes':w.FUSION_BOUNDS['max_cache_bytes']}
    with pytest.raises(RuntimeError,match='FUSION_OUTPUT_BYTES'):
        w._save_fusion_arrays('TRAIN_TEMPORAL_BOUNDARY.npz',arrays,counts,np=np,root=root)
    for name in ('TRAIN_TEMPORAL_BOUNDARY.npz','LATER_TEMPORAL_BOUNDARY_PARTIAL.npz'):
        ref=w._save_fusion_arrays(name,arrays,counts,np=np,root=root,max_bytes=64*1024**2)
        with np.load(ref['path'],allow_pickle=False) as z:
            assert w._fusion_exact_arrays({k:z[k] for k in z.files},arrays)
    with pytest.raises(RuntimeError,match='FUSION_TEMPORAL_BYTE_ROOT'):
        w._save_fusion_arrays('TRAIN_FUSION.npz',arrays,counts,np=np,root=tmp_path,max_bytes=64*1024**2)
    counts['published_npz_bytes']=64*1024**2
    with pytest.raises(RuntimeError,match='FUSION_OUTPUT_BYTES'):
        w._save_fusion_arrays('LATER_TEMPORAL_BOUNDARY.npz',arrays,counts,np=np,root=root,max_bytes=64*1024**2)
    # Exact 512-row tensor budget leaves room for axes, NPZ headers and row0 baseline.
    tensor_bytes=4*512*(250*128+160)
    assert tensor_bytes==65_863_680 and 64*1024**2-tensor_bytes>1024**2


def test_mixing_gradient_exact_plan_and_dispatch(monkeypatch):
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    plan=dict(fusion_plan(),stage=w.MIXING_GRADIENT_STAGE,inputs=w.MIXING_GRADIENT_REFS,
        gradient_helper={'path':str(w.MIXING_GRADIENT_HELPER),'sha256':'a'*64},**w.MIXING_GRADIENT_BOUNDS)
    root=w.MIXING_GRADIENT_ROOT;path=root/'PLAN.json'
    assert w._fusion_plan(plan,path,root,gradient=True)==plan
    for key,value in [('max_backbone_encodings',18),('max_autograd_grad_calls',65),
            ('max_backwards',0),('max_optimizer_updates',1),('gradient_fit_authorized',True),
            ('feature_partition','score'),('batch_equal_length_mtf_eval',True),
            ('parameter_groups',['cross_tf_attn']),('closure_relative_tolerance',1e-3),
            ('runtime_mode','train'),('head_tensor_imports',0)]:
        with pytest.raises(RuntimeError,match='FUSION_EXACT_PLAN'):
            w._fusion_plan({**plan,key:value},path,root,gradient=True)
    with pytest.raises(RuntimeError,match='FUSION_EXACT_PLAN'):
        w._fusion_plan(plan,w.FUSION_ROOT/'PLAN.json',w.FUSION_ROOT,gradient=True)
    with pytest.raises(RuntimeError,match='FUSION_MODE'):
        w._fusion_plan(plan,path,root,gradient=True,temporal=True)
    calls=[]
    monkeypatch.setattr(w,'_inspect_fusion_observation',lambda *a,**kw:
        calls.append(('inspect',kw)) or dict(plan_ref={},view={'contract_sha256':'a'*64},current_sources={}))
    monkeypatch.setattr(w,'_run_mixing_gradient_observation',lambda *a,**kw:calls.append(('run',kw)) or {'synthetic':True})
    report=w.preflight_read_only(path,'a'*64,root)
    assert report['archive_decodes']==report['backbone_encodings']==report['model_constructions']==report['autograd_grad_calls']==0
    assert w.run(path,'a'*64,root,device='cuda')=={'synthetic':True}
    assert calls==[('inspect',{'execute':False,'gradient':True}),('run',{'device':'cuda'})]


def test_mixing_gradient_geometry_distinguishes_conflict_and_zero():
    import torch
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    make=lambda values:torch.tensor(values,dtype=torch.float64)
    v=dict(state_m5_side=make([1.,0.]),future_m5_side=make([0.,0.]),
        rest=make([-2.,1.]),total=make([-1.,1.]))
    m=w._mixing_vector_summary(v,torch=torch)
    assert m['combined_m5_dot_rest']==-2 and m['combined_m5_dot_total']==-1
    assert m['cosine']['state_m5_side']['future_m5_side'] is None
    assert m['combined_m5_norm']==1
    with pytest.raises(RuntimeError,match='GRADIENT_VECTORS'):
        w._mixing_vector_summary({**v,'rest':make([float('nan'),0.])},torch=torch)


@pytest.mark.parametrize('complete',[True,False])
def test_mixing_gradient_json_bound_covers_actual_success_and_failure_bytes(tmp_path,complete):
    import json
    from gx1.audit import cache_market_state_ta_direction_v1 as w
    (tmp_path/'CLAIM.json').write_text('{}\n')
    payload=dict(source_unchanged=True,diagnostic={'x':'long enough to expose pretty-print and wrapper overhead'})
    with pytest.raises(RuntimeError,match='GRADIENT_JSON_BOUND'):
        w._publish_fusion_outcome({},payload,complete=complete,root=tmp_path,json_bytes_bound=128)
    assert not (tmp_path/'RESULT.json').exists() and not (tmp_path/'TERMINAL.json').exists()
    w._publish_fusion_outcome({},payload,complete=complete,root=tmp_path,json_bytes_bound=4096)
    actual=sum((tmp_path/n).stat().st_size for n in ('CLAIM.json','RESULT.json','TERMINAL.json'))
    assert 128<actual<=4096
    assert json.loads((tmp_path/'RESULT.json').read_text())['status']==('COMPLETE' if complete else 'FAILED_CONSUMED')
