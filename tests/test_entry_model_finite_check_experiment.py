from __future__ import annotations

import ast
import gc
import inspect
import weakref

import pytest
import torch

from gx1.contracts import entry_training_precision_v1 as policy
from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as model
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

POLICIES = (
    policy.EXPERIMENTAL_FP32_3090_FINITE_SINGLE,
    policy.EXPERIMENTAL_FP32_3090_FINITE_GROUPED,
)


@pytest.mark.parametrize('mode', ['single', 'grouped'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), complex(1, float('inf'))])
def test_every_nonfinite_is_rejected_before_scope_returns(mode, value):
    returned = False
    with pytest.raises(RuntimeError, match='NONFINITE: first contains NaN/Inf'):
        with model.model_finite_check_scope(mode):
            model._assert_finite('finite', torch.ones(3))
            model._assert_finite('first', torch.tensor(value))
            model._assert_finite('second', torch.tensor(float('nan')))
        returned = True
    assert not returned
    assert model._MODEL_FINITE_CHECK_SCOPE.get() is None


@pytest.mark.parametrize('mode', ['single', 'grouped'])
def test_empty_integer_boolean_and_complex_finite_inputs_are_valid(mode):
    with model.model_finite_check_scope(mode):
        for t in [torch.empty(0), torch.tensor([1, 2]), torch.tensor([True, False]), torch.tensor([1+2j])]:
            model._assert_finite('finite', t)
    assert model._MODEL_FINITE_CHECK_SCOPE.get() is None


def test_grouped_scope_keeps_only_detached_scalar_predicates():
    with model.model_finite_check_scope('grouped'):
        t = torch.ones(4, requires_grad=True)
        ref = weakref.ref(t)
        model._assert_finite('activation', t)
        del t
        gc.collect()
        assert ref() is None
        queue = model._MODEL_FINITE_CHECK_SCOPE.get().pending
        assert len(queue) == 1
        flag = queue[0][1]
        assert flag.shape == torch.Size([]) and flag.dtype == torch.bool
        assert not flag.requires_grad and flag.grad_fn is None


def test_nested_scope_rejects_earlier_outer_failure_before_entering_inner():
    entered = False
    with pytest.raises(RuntimeError, match='NONFINITE: outer'):
        with model.model_finite_check_scope('grouped'):
            model._assert_finite('outer', torch.tensor(float('nan')))
            with model.model_finite_check_scope('grouped'):
                entered = True
    assert not entered
    assert model._MODEL_FINITE_CHECK_SCOPE.get() is None


def test_inner_failure_restores_outer_scope_without_leaking_predicates():
    with model.model_finite_check_scope('grouped'):
        outer = model._MODEL_FINITE_CHECK_SCOPE.get()
        model._assert_finite('outer_finite', torch.ones(1))
        with pytest.raises(RuntimeError, match='NONFINITE: inner'):
            with model.model_finite_check_scope('grouped'):
                model._assert_finite('inner', torch.tensor(float('inf')))
        assert model._MODEL_FINITE_CHECK_SCOPE.get() is outer
        assert outer.pending == []
        model._assert_finite('outer_still_finite', torch.ones(1))
    assert model._MODEL_FINITE_CHECK_SCOPE.get() is None


@pytest.mark.parametrize('invalid', [False, True])
def test_downstream_exception_preserves_prior_nonfinite_or_original_error(invalid):
    original = RuntimeError('downstream')
    with pytest.raises(RuntimeError) as caught:
        with model.model_finite_check_scope('grouped'):
            model._assert_finite('earlier', torch.tensor(float('nan') if invalid else 1.))
            raise original
    if invalid:
        assert str(caught.value) == 'NONFINITE: earlier contains NaN/Inf'
    else:
        assert caught.value is original
    assert model._MODEL_FINITE_CHECK_SCOPE.get() is None


@pytest.mark.parametrize('selected', POLICIES)
def test_policy_is_fp32_batch8_with_unchanged_hardware_geometry(selected):
    assert policy.require_training_precision_policy(selected, device_type='cuda', execution_tier='canonical', profile='smoke', batch_size=8) == selected
    m = policy.training_precision_metadata(selected, device_type='cuda')
    for key in ['parameter_dtype', 'loss_reduction_dtype', 'optimizer_state_dtype', 'ema_dtype']:
        assert m[key] == 'float32'
    for key in ['autocast', 'tf32', 'compile', 'gradient_scaler']:
        assert m[key] is False
    assert m['cuda_memory_fraction'] == .45
    assert m['required_cuda_compute_capability'] == [8, 6]
    assert m['model_finite_check_mode'] == policy.model_finite_check_mode(selected)
    assert policy.numerical_thread_count(selected) == 8
    assert policy.unified_exit_chunk_rows(selected, batch_size=8) == 8
    assert policy.candidate_checkpoint_interval(selected) == 64
    assert policy.candidate_validation_checkpoint_interval(selected) == 64


@pytest.mark.parametrize('selected', POLICIES)
@pytest.mark.parametrize('change', [{'device_type':'cpu'}, {'profile':'candidate'}, {'execution_tier':'attended_only'}, {'batch_size':10}, {'batch_size':True}])
def test_policy_rejects_scope_expansion(selected, change):
    kwargs = dict(device_type='cuda', execution_tier='canonical', profile='smoke', batch_size=8)
    kwargs.update(change)
    with pytest.raises(policy.TrainingPrecisionPolicyError):
        policy.require_training_precision_policy(selected, **kwargs)


@pytest.mark.parametrize('selected', POLICIES)
@pytest.mark.parametrize('change', [{'epochs':2}, {'grad_accum_steps':2}, {'subsample_rows':513}, {'subsample_rows':0}])
def test_policy_rejects_unbounded_training(selected, change):
    kwargs = dict(epochs=1, grad_accum_steps=1, subsample_rows=512)
    policy.require_local_precision_benchmark_geometry(selected, **kwargs)
    kwargs.update(change)
    with pytest.raises(policy.TrainingPrecisionPolicyError):
        policy.require_local_precision_benchmark_geometry(selected, **kwargs)


@pytest.mark.parametrize('selected', sorted(policy.TRAINING_PRECISION_POLICIES - set(POLICIES)))
def test_other_policies_keep_immediate_original_finite_checks(monkeypatch, selected):
    monkeypatch.setattr(trainer, '_TRAINING_PRECISION_POLICY', selected)
    with trainer._training_model_finite_check_context():
        assert model._MODEL_FINITE_CHECK_SCOPE.get() is None
        with pytest.raises(RuntimeError, match='NONFINITE: unchanged'):
            model._assert_finite('unchanged', torch.tensor(float('nan')))


@pytest.mark.parametrize('selected', POLICIES)
def test_actual_entry_forward_owner_flushes_before_exposing_output(monkeypatch, selected):
    monkeypatch.setattr(trainer, '_TRAINING_PRECISION_POLICY', selected)
    class InvalidForward(torch.nn.Module):
        def forward(self):
            model._assert_finite('inside_actual_forward', torch.tensor(float('nan')))
            return {'entry_action_q_bps': torch.zeros(1, 3)}
    with pytest.raises(RuntimeError, match='NONFINITE: inside_actual_forward'):
        trainer._model_forward_fp32(InvalidForward())
    assert model._MODEL_FINITE_CHECK_SCOPE.get() is None
    assert not torch.is_autocast_enabled()


def test_all_actual_entry_and_exit_model_calls_are_inside_finite_scope():
    tree = ast.parse(inspect.getsource(trainer))
    found = set()
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.With):
                continue
            contexts = [item.context_expr.func.id for item in node.items if isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Name)]
            if '_training_autocast_context' in contexts:
                assert '_training_model_finite_check_context' in contexts
                found.add(fn.name)
    assert found == {'_model_forward_fp32', '_forward_unified_exit_episode_pack', '_forward_unified_exit_episode_batch', '_unified_exit_influence_forward'}


@pytest.mark.parametrize('capability,accepted', [((8,6),True), ((8,0),False), ((9,0),False)])
def test_local_finite_check_capability_does_not_expand_to_other_devices(monkeypatch, capability, accepted):
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda _: capability)
    if accepted:
        trainer._require_local_fp32_finite_check_capability()
    else:
        with pytest.raises(RuntimeError, match='LOCAL_FP32_FINITE_CHECK_CAPABILITY_REQUIRED'):
            trainer._require_local_fp32_finite_check_capability()
