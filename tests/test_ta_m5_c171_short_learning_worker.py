"""Synthetic worker boundaries only: no GX1 model, checkpoint, data or fit."""
from copy import deepcopy
import ast
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gx1.audit import trace_market_state_ta_m5_original_cuda_prefix_v1 as worker
from gx1.contracts import market_state_ta_m5_short_learning_v1 as short
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K

SHA = '1' * 64


def recipe(short_learning=True):
    value = dict(run_scope=short.SCOPE if short_learning else worker.scope.TA_M5_ORIGINAL_CUDA_PREFIX_SCOPE,
        max_private_updates=256, max_backwards=256, max_forwards=288 if short_learning else 256,
        max_model_seconds=1800 if short_learning else 720,
        max_trace_bytes=64 * 1024**2, max_checkpoint_loads=2, max_full_state_restores=1,
        max_checkpoint_writes=2 if short_learning else 0,
        numerical_recipe=dict(grad_accum_steps=1, grad_clip_norm=1.0,
                              future_weight=1.0, m5_equal_observation=True),
        original_source_commit='a' * 40, trace_worker={'path': '/fixture/worker', 'sha256': SHA},
        input_refs={name: {'path': '/fixture/' + name, 'sha256': SHA}
                    for name in ('original_initial_checkpoint', 'original_p4096_checkpoint')},
        max_published_json_bytes=1_000_000, expected_view_sha256=SHA,
        expected_model_contract_sha256=SHA,
        expected_loss_contract_sha256=objective.contract(ta_only=True, dedicated_m5_side=True,
                                                         m5_equal_observation=True)['contract_sha256'],
        expected_initial_model_state_sha256=short.MODEL_SHA256)
    if short_learning:
        value.update(max_training_forwards=256, max_evaluation_forwards=32,
            panel_batch_indices=short.PANEL_BATCH_INDICES,
            optimizer_initialization='fresh_adamw_empty_state',
            candidate_inputs={'reversed_evidence': {'path': '/fixture/R1', 'sha256': SHA}})
    return value


def tree_sha(value):
    def encode(item):
        if isinstance(item, torch.Tensor):
            return [str(item.dtype), list(item.shape), item.detach().cpu().numpy().tobytes().hex()]
        if isinstance(item, dict):
            return {str(k): encode(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [encode(v) for v in item]
        return item
    return hashlib.sha256(json.dumps(encode(value), sort_keys=True).encode()).hexdigest()


class SyntheticModules:
    """No forward implementation: the panel receives already synthetic logits."""
    def __init__(self):
        self.training = True
        self._forward_hooks, self._forward_pre_hooks = {}, {}
        self.child = SimpleNamespace(training=False, _forward_hooks={}, _forward_pre_hooks={})
        self.parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))

    def named_modules(self):
        return iter([('', self), ('child', self.child)])

    def named_parameters(self):
        return iter([('parameter', self.parameter)])

    def eval(self):
        for _, module in self.named_modules():
            module.training = False
        return self


def synthetic_batch(left):
    parents = np.arange(left, left + 16, dtype=np.int64)
    issues = parents * np.int64(300_000_000_000)
    state_query = np.ones((16, 5), dtype=np.int64)
    future_query = np.broadcast_to(np.asarray(NATIVE_K, np.int64), (16, 5)).copy()
    truths = {}
    for task, query in (('state', state_query), ('future', future_query)):
        allowed = np.broadcast_to(np.asarray([1, 2, 4, 5, 7, 4, 1, 2], np.uint8)[:, None], (8, 5))
        allowed = np.concatenate((allowed, allowed), axis=0).copy()
        eligible, mature = np.ones((16, 5), bool), np.ones((16, 5), bool)
        eligible[0, :] = False
        mature[2, :] = False
        if task == 'future':
            allowed[:, 0] = 7  # A zero-support cell must stay absent, not become zero loss.
        truth = dict(parent_rows=parents.copy(), issue_ns=issues.copy(),
                     allowed=allowed, eligible=eligible, mature=mature)
        if task == 'state':
            truth['query_k'] = query.copy()
        truths[task] = truth
    state_raw = torch.arange(160, dtype=torch.float32).reshape(16, 5, 2) / 100 - .8
    future_raw = state_raw.flip(0).clone()
    original = objective.combined_loss(state_raw=state_raw, future_raw=future_raw,
        state_truth=truths['state'], future_truth=truths['future'],
        state_query_k=torch.from_numpy(state_query), future_weight=1.0,
        ta_only=True, dedicated_m5_side=True, m5_equal_observation=True)
    report = dict(loss=original['loss'], skip_update=original['skip_update'],
        state_activity_side_logits=state_raw, future_activity_side_logits=future_raw,
        state_query_k=state_query, future_native_k=np.asarray(NATIVE_K, np.int64),
        model_contract_sha256=SHA, loss_contract_sha256=original['contract_sha256'])
    for task in truths:
        for key in ('allowed', 'mature'):
            report[task + '_' + key] = truths[task][key].copy()
    return report, parents, issues, truths, original


@pytest.fixture
def panel(monkeypatch):
    model = SyntheticModules()
    value = recipe()
    counters = {key: 0 for key in ('forwards', 'evaluation_forwards', 'forward_returns',
        'extra_target_gathers', 'loss_component_owner_calls')}
    seen, gathered = [], {}
    def capture():
        return dict(model_state_sha256=tree_sha(model.parameter),
            model_state=model.parameter.detach().clone(), optimizer_state={'state': {}},
            module_training={name: module.training for name, module in model.named_modules()},
            rng_state=torch.get_rng_state().clone())
    def evaluate(left):
        assert not torch.is_grad_enabled()
        assert not any(module.training for _, module in model.named_modules())
        seen.append(left)
        report, parents, issues, truth, _ = synthetic_batch(left)
        gathered[left] = truth
        return report, parents, issues
    def gather(parents, issues):
        return gathered[int(parents[0])]
    # The real scorer still computes every proper score; only the irrelevant prior fixture is synthetic.
    monkeypatch.setattr(objective, '_endpoint_prior_arrays',
        lambda *args, **kwargs: {task: np.zeros((5, 2), np.float32) for task in ('state', 'future')})
    arguments = dict(recipe=value, counters=counters, evaluate=evaluate,
        gather_truth=gather, capture=capture, tree_sha=tree_sha,
        priors={'contract_sha256': SHA}, clock_guard=lambda: None, np=np, torch=torch)
    return SimpleNamespace(model=model, args=arguments, counters=counters,
                           seen=seen, capture=capture, before=tree_sha(capture()))


def test_legacy_and_short_limits_are_separate():
    worker._require_execution_bounds(recipe(False), short_learning=False)
    worker._require_execution_bounds(recipe(), short_learning=True)
    with pytest.raises(RuntimeError, match='ADMITTED_BOUNDS'):
        worker._require_execution_bounds(recipe(), short_learning=False)


@pytest.mark.parametrize('key,value', [
    ('max_forwards', 289), ('max_private_updates', 257), ('max_evaluation_forwards', 33),
    ('optimizer_initialization', 'restore_R1_moments'),
    ('panel_batch_indices', [0] * 16), ('max_checkpoint_writes', 3),
])
def test_short_budget_and_initialization_fail_closed(key, value):
    changed = recipe()
    changed[key] = value
    with pytest.raises(RuntimeError, match='SHORT_ADMITTED_BOUNDS'):
        worker._require_execution_bounds(changed, short_learning=True)


def test_existing_loss_components_preserve_masks_normalization_and_exact_total():
    report, _, _, truth, original = synthetic_batch(0)
    before = report['loss'].clone()
    result = worker._short_loss_components(report, truth, torch=torch)
    assert result['total'] == float(original['loss']) == float(before)
    for task in ('state', 'future'):
        assert result[task + '_per_tf_activity_plus_side_before_divisor'] == {
            key: float(value) for key, value in original[task]['per_tf_mean_loss'].items()}
    report['loss'] = report['loss'] + 1
    with pytest.raises(RuntimeError, match='SHORT_ORIGINAL_LOSS_PARITY'):
        worker._short_loss_components(report, truth, torch=torch)


def test_fixed_panel_success_preserves_complete_state_and_zero_support(panel):
    measured, arrays, components, digest = worker._short_train_panel(panel.model, **panel.args)
    assert panel.seen == [16 * index for index in short.PANEL_BATCH_INDICES]
    assert arrays['parents'].shape == (256,) and arrays['future_raw'].shape == (256, 5, 2)
    assert np.array_equal(arrays['parents'], np.concatenate([np.arange(left, left + 16) for left in panel.seen]))
    assert measured['rows'] == 256 and measured['batches'] == 16
    assert measured['scores']['model']['future']['M5']['conditional_side']['count'] == 0
    assert 'conditional_side' not in measured['means']['model']['future']['M5']
    assert len(components) == 16 and digest == tree_sha(panel.model.parameter)
    assert tree_sha(panel.capture()) == panel.before
    assert panel.counters == dict(forwards=16, evaluation_forwards=16, forward_returns=16,
                                 extra_target_gathers=32, loss_component_owner_calls=16)
    second, axes, _, _ = worker._short_train_panel(panel.model, **panel.args)
    assert second == measured and np.array_equal(arrays['state_query'], axes['state_query'])
    assert panel.counters['forwards'] == 32


def test_panel_failed_forward_preserves_modes_state_and_partial_counts(panel):
    real = panel.args['evaluate']
    def fail(left):
        if len(panel.seen) == 2:
            raise ValueError('synthetic forward failed')
        return real(left)
    panel.args['evaluate'] = fail
    with pytest.raises(ValueError, match='synthetic forward failed'):
        worker._short_train_panel(panel.model, **panel.args)
    assert tree_sha(panel.capture()) == panel.before
    assert panel.counters['forwards'] == 3 and panel.counters['forward_returns'] == 2
    assert panel.counters['loss_component_owner_calls'] == 2


@pytest.mark.parametrize('change', ['parameters', 'rng', 'trainability', 'hooks'])
def test_panel_detects_state_mutation_and_restores_original_modes(panel, change):
    real = panel.args['evaluate']
    def mutate(left):
        report = real(left)
        if change == 'parameters':
            panel.model.parameter.add_(1)
        elif change == 'rng':
            torch.rand(1)
        elif change == 'trainability':
            panel.model.parameter.requires_grad_(False)
        else:
            panel.model._forward_hooks[3] = object()
        return report
    panel.args['evaluate'] = mutate
    with pytest.raises(RuntimeError, match='SHORT_PANEL_CHANGED_STATE'):
        worker._short_train_panel(panel.model, **panel.args)
    assert panel.model.training is True and panel.model.child.training is False


def test_panel_rejects_target_mismatch_and_budget_before_extra_call(panel):
    gather = panel.args['gather_truth']
    def wrong(parents, issues):
        truth = gather(parents, issues)
        truth['state']['allowed'][0, 0] = 4
        return truth
    panel.args['gather_truth'] = wrong
    with pytest.raises(RuntimeError, match='SHORT_PANEL_TARGET_PARITY'):
        worker._short_train_panel(panel.model, **panel.args)
    assert tree_sha(panel.capture()) == panel.before
    panel.counters['forwards'] = 288
    before = len(panel.seen)
    with pytest.raises(RuntimeError, match='SHORT_PANEL_FORWARD_BUDGET'):
        worker._short_train_panel(panel.model, **panel.args)
    assert len(panel.seen) == before


@pytest.mark.parametrize('short_learning', [False, True])
def test_pre_model_failure_publishes_correct_route_and_no_final_authority(monkeypatch, tmp_path, short_learning):
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    value = recipe(short_learning)
    ref = dict(path=str(short.RECIPE_PATH if short_learning else worker.scope.TA_M5_ORIGINAL_CUDA_PREFIX_RECIPE_PATH), sha256=SHA)
    window = dict(recipe=ref, output_directory=str(tmp_path / 'output'),
                  campaign_cursor_path=str(tmp_path / 'cursor.json'), max_invocation_seconds=3600)
    plan = dict(source_commit='b' * 40, plan_sha256=SHA)
    origin = dict(original_recipe={'numerical_recipe': value['numerical_recipe']})
    monkeypatch.setattr(cap, 'require_guarded_cuda_trainer_execution', lambda: {})
    monkeypatch.setattr(authority, 'require_native_direction_research_context', lambda *a: (window, plan, {}, {}, value))
    monkeypatch.setattr(short, 'read_origin', lambda r: origin)
    monkeypatch.setattr(worker.scope, 'read_ta_m5_original_cuda_prefix_origin', lambda r: origin)
    monkeypatch.setattr(short, 'ROOT', tmp_path)
    monkeypatch.setattr(worker.scope, 'TA_M5_ORIGINAL_CUDA_PREFIX_ROOT', tmp_path)
    def stop(*args):
        raise RuntimeError('synthetic source rejection before model')
    monkeypatch.setattr(worker, '_bookend', stop)
    published, cursors = {}, []
    def publish(path, payload, *args):
        published[Path(path).name] = deepcopy(payload)
        return dict(path=str(path), sha256=SHA)
    monkeypatch.setattr(common, '_json', publish)
    def cursor(**kwargs):
        cursors.append(kwargs)
        return kwargs
    monkeypatch.setattr(authority, 'build_research_cursor', cursor)
    monkeypatch.setattr(authority, 'build_research_progress', lambda **kwargs: kwargs)
    monkeypatch.setattr(authority, 'require_research_progress', lambda *a, **kw: None)
    with pytest.raises(RuntimeError, match='synthetic source rejection before model'):
        worker.run_original_cuda_prefix(policy_path=tmp_path / 'window.json', policy_file_sha256=SHA,
            progress_path=tmp_path / 'progress.json', short_learning=short_learning)
    result = published['RESULT.json']
    assert result['outcome'] == 'FAILED' and result['final_state'] is None
    assert result['counts']['model_constructions'] == result['counts']['optimizer_updates'] == 0
    assert cursors[0]['full_state'] is None and published['TERMINAL.json']['exit_code'] == 1
    if short_learning:
        assert result['schema_version'] == short.RESULT_SCHEMA
        assert result['initial_restore_exact'] is result['evaluation_state_unchanged'] is False
        assert result['measurement_refs'] == {} and result['initial_model_state_sha256'] is None
        assert 'first_step_exact' not in result and 'first_step_reference' not in result
    else:
        assert result['schema_version'] == worker.scope.TA_M5_ORIGINAL_CUDA_PREFIX_RESULT_SCHEMA
        assert result['first_step_reference'] == dict(loss=worker.FIRST_LOSS, gradient_norm=worker.FIRST_NORM)
        assert 'optimizer_initialization' not in result


def test_no_second_training_loop_and_legacy_first_step_guard_is_retained():
    tree = ast.parse(inspect.getsource(worker.run_original_cuda_prefix))
    native_loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
                    and ast.unparse(node.iter) == 'range(0, ROWS, 16)']
    assert len(native_loops) == 1
    checks = [node for node in ast.walk(native_loops[0]) if isinstance(node, ast.If)
              and ast.unparse(node.test) == 'left == 0 and (not short_learning)']
    assert len(checks) == 1
    assert 'ORIGINAL_FIRST_STEP' in ast.unparse(checks[0])
