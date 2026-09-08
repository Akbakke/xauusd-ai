from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch

from gx1.contracts import entry_training_precision_v1 as policy
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from gx1.models.entry_v10 import training_kernel_profile as kp


POLICY = policy.EXPERIMENTAL_FP32_3090_KERNEL_PROFILE


def test_actual_cpu_profiler_captures_only_warmed_ninth_update(tmp_path):
    output = tmp_path / 'profile'
    with kp._KernelProfileSession(output, include_cuda=False):
        for step in range(1, 65):
            with kp.kernel_profile_range('update_' + str(step)):
                value = torch.ones(4, 4)
                _ = value @ value
            kp.kernel_profile_step()
    assert kp._ACTIVE_PROFILE.get() is None
    session = json.loads((output / 'session.json').read_text())
    assert session['terminal_success'] is True
    assert session['completed_optimizer_steps'] == 64
    assert session['trace_count'] == 1 and session['trace_optimizer_step'] == 9
    report = json.loads((output / 'operators.json').read_text())
    assert report['cuda_timing_available'] is False
    assert report['status'] == 'CPU_ONLY_CAPTURED'
    assert report['profiled_optimizer_steps'] == [9]
    trace = Path(report['trace']['path'])
    assert hashlib.sha256(trace.read_bytes()).hexdigest() == report['trace']['sha256']
    events = json.loads(trace.read_text())['traceEvents']
    labels = {event.get('name') for event in events if event.get('name', '').startswith('GX1/update_')}
    assert labels == {'GX1/update_9'}
    assert any(row['operator'] == 'aten::mm' and row['count'] > 0 for row in report['operators'])


def test_actual_cpu_profiler_preserves_optimizer_result_and_torch_rng(tmp_path):
    def run(profiled):
        torch.manual_seed(984)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        context = kp._KernelProfileSession(tmp_path / 'numeric_profile', include_cuda=False) if profiled else __import__('contextlib').nullcontext()
        with context:
            for _ in range(12):
                loss = model(torch.randn(4, 3)).square().mean()
                loss.backward()
                trainer._optimizer_step_with_finite_gradients(model=model, optimizer=optimizer)
                kp.kernel_profile_step()
        return model.state_dict(), optimizer.state_dict(), torch.get_rng_state()
    reference, ref_optimizer, ref_rng = run(False)
    actual, actual_optimizer, actual_rng = run(True)
    for key in reference:
        assert torch.equal(reference[key], actual[key]), key
    assert ref_optimizer['param_groups'] == actual_optimizer['param_groups']
    for index, state in ref_optimizer['state'].items():
        for key, tensor in state.items():
            assert torch.equal(tensor, actual_optimizer['state'][index][key])
    assert torch.equal(ref_rng, actual_rng)


def test_profiler_exception_cleanup_does_not_mask_training_error(tmp_path):
    original = RuntimeError('training failed')
    output = tmp_path / 'failed_profile'
    with pytest.raises(RuntimeError) as caught:
        with kp._KernelProfileSession(output, include_cuda=False):
            kp.kernel_profile_step()
            raise original
    assert caught.value is original
    assert kp._ACTIVE_PROFILE.get() is None
    assert json.loads((output / 'session.json').read_text())['terminal_success'] is False


def test_missing_active_step_is_not_reported_as_success(tmp_path):
    with pytest.raises(RuntimeError, match='REQUIRED_TRACE_MISSING'):
        with kp._KernelProfileSession(tmp_path / 'incomplete', include_cuda=False):
            kp.kernel_profile_step()
    assert kp._ACTIVE_PROFILE.get() is None


def test_profile_output_is_exclusive(tmp_path):
    output = tmp_path / 'existing'
    output.mkdir()
    with pytest.raises(FileExistsError):
        with kp._KernelProfileSession(output, include_cuda=False):
            pytest.fail('must reject before profiling')
    assert kp._ACTIVE_PROFILE.get() is None


@pytest.mark.parametrize('selected', sorted(policy.TRAINING_PRECISION_POLICIES - {POLICY}))
def test_other_policies_do_not_start_profiler_or_change_call_signature(monkeypatch, selected):
    def unexpected(*args, **kwargs):
        pytest.fail('default policy started profiler')
    monkeypatch.setattr(kp, '_KernelProfileSession', unexpected)
    def function(value, *, kernel_profile_output_dir=None):
        with kp.kernel_profile_range('unused'):
            kp.kernel_profile_step()
        return value
    wrapped = kp.profile_training_epoch(policy=lambda: selected)(function)
    assert inspect.signature(wrapped) == inspect.signature(function)
    assert wrapped(42) == 42
    with pytest.raises(RuntimeError, match='POLICY_OUTPUT_MISMATCH'):
        wrapped(42, kernel_profile_output_dir=Path('/unused'))


def test_profiler_policy_requires_declared_output_before_any_work():
    @kp.profile_training_epoch(policy=lambda: POLICY)
    def function():
        pytest.fail('profile started without bound output')
    with pytest.raises(RuntimeError, match='POLICY_OUTPUT_MISMATCH'):
        function()


def test_actual_train_epoch_step_call_follows_real_optimizer_update():
    tree = ast.parse(inspect.getsource(trainer.train_epoch))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    steps = [n for n in calls if n.func.id == 'kernel_profile_step']
    updates = [n for n in calls if n.func.id == '_optimizer_step_with_finite_gradients']
    assert len(steps) == len(updates) == 1
    assert updates[0].lineno < steps[0].lineno
    branch = next(n for n in ast.walk(fn) if isinstance(n, ast.If) and ast.unparse(n.test) == '_accum_count >= _accum_steps')
    assert steps[0] in list(ast.walk(branch)) and updates[0] in list(ast.walk(branch))
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'profile_training_epoch' for n in fn.decorator_list)


def test_smoke_call_derives_profile_directory_from_declared_bundle(tmp_path):
    tree = ast.parse(inspect.getsource(trainer))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'train_epoch']
    keywords = [k for call in calls for k in call.keywords if k.arg == 'kernel_profile_output_dir']
    assert len(keywords) == 1
    expression = compile(ast.Expression(keywords[0].value), '<actual-profile-output>', 'eval')
    bundle = tmp_path / 'declared_bundle'
    namespace = dict(Path=Path, _resolve_train_out_bundle_dir=lambda path, _: path, out_bundle_dir=bundle,
                     gx1_data_override='', precision_policy=POLICY, EXPERIMENTAL_FP32_3090_KERNEL_PROFILE=POLICY)
    assert eval(expression, namespace) == tmp_path / '.declared_bundle.kernel_profile'
    namespace['precision_policy'] = policy.DETERMINISTIC_FP32
    assert eval(expression, namespace) is None


def test_kernel_profile_has_original_fp32_and_resource_geometry():
    assert policy.require_training_precision_policy(POLICY, device_type='cuda', execution_tier='canonical', profile='smoke', batch_size=8) == POLICY
    metadata = policy.training_precision_metadata(POLICY, device_type='cuda')
    assert metadata['kernel_profiling'] is True
    assert metadata['throughput_qualification_allowed'] is False
    assert metadata['profiled_optimizer_step'] == 9
    assert metadata['cuda_memory_fraction'] == .45
    assert metadata['required_cuda_compute_capability'] == [8,6]
    assert policy.model_finite_check_mode(POLICY) is None
    assert policy.numerical_thread_count(POLICY) == 8
    assert policy.unified_exit_chunk_rows(POLICY, batch_size=8) == 8
    assert policy.candidate_checkpoint_interval(POLICY) == 64
    for key in ['autocast', 'tf32', 'compile', 'gradient_scaler']:
        assert metadata[key] is False
    for key in ['parameter_dtype', 'loss_reduction_dtype', 'optimizer_state_dtype', 'ema_dtype']:
        assert metadata[key] == 'float32'


@pytest.mark.parametrize('change', [{'profile':'candidate'}, {'device_type':'cpu'}, {'execution_tier':'attended_only'}, {'batch_size':10}, {'batch_size':True}])
def test_kernel_profile_rejects_other_execution_surfaces(change):
    kwargs = dict(device_type='cuda', execution_tier='canonical', profile='smoke', batch_size=8)
    kwargs.update(change)
    with pytest.raises(policy.TrainingPrecisionPolicyError):
        policy.require_training_precision_policy(POLICY, **kwargs)


@pytest.mark.parametrize('change', [{'epochs':2}, {'grad_accum_steps':2}, {'subsample_rows':511}, {'subsample_rows':513}, {'subsample_rows':True}])
def test_kernel_profile_requires_exact_bounded_geometry(change):
    kwargs = dict(epochs=1, grad_accum_steps=1, subsample_rows=512)
    policy.require_local_precision_benchmark_geometry(POLICY, **kwargs)
    kwargs.update(change)
    with pytest.raises(policy.TrainingPrecisionPolicyError):
        policy.require_local_precision_benchmark_geometry(POLICY, **kwargs)


def test_profiler_implementation_is_in_canonical_recipe_source_closure():
    from gx1.contracts.entry_model_native_train_launch_v1 import recipe_source_binding_paths
    repo = Path(trainer.__file__).resolve().parents[3]
    paths = recipe_source_binding_paths(repo=repo, wrapper_path=repo / 'gx1/scripts/run_entry_model_native_pretest_technical_train_v1.py')
    assert paths['python:gx1/models/entry_v10/training_kernel_profile.py'] == Path(kp.__file__).resolve()
