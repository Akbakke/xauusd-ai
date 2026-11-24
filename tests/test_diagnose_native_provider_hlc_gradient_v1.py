"""Synthetic fixed-panel gradient accounting; no genuine payload or native model."""
import ast
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from gx1.audit import diagnose_native_provider_hlc_gradient_v1 as m

TOLERANCES = {
    'component_gradient_absolute_L2_floor': 1e-12,
    'component_gradient_global_relative_L2_max': 1e-5,
    'dot_identity_relative_error_max': 1e-5,
    'no_post_failure_relaxation': True,
    'saved_FINAL_logits': {'atol': 1e-6, 'rtol': 1e-5,
        'source_owner': 'THROUGHPUT_EVAL_TOLERANCE; candidate fixed gate, not prequalified CPU/autograd parity'},
    'sign_margin_relative_to_normS_times_normTotal': 1e-6,
}


def vector(values):
    return [torch.tensor(values, dtype=torch.float64)]


def test_accumulation_converts_each_complete_gradient_to_float64_before_addition():
    accumulator = [None]
    for value in (1e8, 1., -1e8):
        result = m.accumulate_gradients(accumulator, (torch.tensor([value], dtype=torch.float32),))
        assert result is accumulator
    assert accumulator[0].dtype == torch.float64
    assert accumulator[0].device.type == 'cpu'
    assert accumulator[0].item() == 1.


def test_accumulation_keeps_ordered_axis_and_None_as_mathematical_zero_only():
    accumulator = [None, None, None]
    first = torch.tensor([1., 2.], dtype=torch.float32, requires_grad=True)
    m.accumulate_gradients(accumulator, (first, None, torch.tensor(3., dtype=torch.float32)))
    m.accumulate_gradients(accumulator, (None, None, torch.tensor(4., dtype=torch.float32)))
    assert accumulator[1] is None
    assert torch.equal(accumulator[0], torch.tensor([1., 2.], dtype=torch.float64))
    assert accumulator[2].item() == 7.
    assert all(value is None or not value.requires_grad for value in accumulator)
    assert first.grad is None
    first.data[0] = 9.
    assert accumulator[0][0].item() == 1.


@pytest.mark.parametrize('bad', ['axis', 'shape', 'nan', 'bad_accumulator_dtype'])
def test_accumulation_rejects_invalid_axis_shape_or_nonfinite_values(bad):
    accumulator = [torch.zeros(2, dtype=torch.float64)]
    gradients = (torch.ones(2, dtype=torch.float32),)
    if bad == 'axis': gradients = ()
    elif bad == 'shape': gradients = (torch.ones(3, dtype=torch.float32),)
    elif bad == 'nan': gradients = (torch.tensor([float('nan'), 1.]),)
    else: accumulator = [torch.zeros(2, dtype=torch.float32)]
    with pytest.raises(ValueError): m.accumulate_gradients(accumulator, gradients)


def test_native_microchunk_scaling_preserves_original_full_B96_denominator():
    full_D = 12
    canonical_chunk_losses = [torch.tensor(2., dtype=torch.float64), torch.tensor(5., dtype=torch.float64)]
    chunk_D = [10, 2]
    total = sum(m.scaled_canonical_loss(loss, count, full_D)
                for loss, count in zip(canonical_chunk_losses, chunk_D))
    assert total.item() == pytest.approx((2. * 10 + 5. * 2) / 12, abs=1e-15)
    assert total.item() != sum(loss.item() for loss in canonical_chunk_losses) / 2
    zero = m.scaled_canonical_loss(torch.tensor(0., dtype=torch.float64), 0, full_D)
    assert zero.item() == 0


@pytest.mark.parametrize('chunk,full', [(True, 12), (10, True), (-1, 12), (13, 12), (1, 0)])
def test_native_scaling_rejects_invalid_denominator_contract(chunk, full):
    with pytest.raises(ValueError): m.scaled_canonical_loss(torch.tensor(1.), chunk, full)


def test_owner_loads_only_explicit_CPU_weights_only_without_optimizer_or_RNG_restore():
    tree = ast.parse(Path(m.__file__).read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    loads = [node for node in calls if isinstance(node.func, ast.Attribute)
             and isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch' and node.func.attr == 'load']
    assert loads
    for call in loads:
        options = {kw.arg: kw.value for kw in call.keywords}
        assert ast.literal_eval(options['map_location']) == 'cpu'
        assert ast.literal_eval(options['weights_only']) is True
    forbidden = {'strict_restore', 'restore_training_rng', 'set_rng_state', 'AdamW', 'backward', 'add_safe_globals'}
    assert not any(isinstance(call.func, ast.Attribute) and call.func.attr in forbidden for call in calls)
    assert not any(isinstance(call.func, ast.Name) and call.func.id in forbidden for call in calls)


class TinyState:
    def __init__(self):
        self.training = False
        self.parameter = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.float32))
    def state_dict(self):
        return {'parameter': self.parameter.detach()}
    def parameters(self):
        return iter((self.parameter,))


def test_measurement_bookend_does_not_change_state_RNG_or_parameter_grad(monkeypatch):
    model = TinyState(); state = m.core.state_sha(model.state_dict()); rng = m.core.shared_gradient_rng()
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    m.measurement_bookend(model, state, rng)
    assert m.core.state_sha(model.state_dict()) == state
    assert m.core.state_sha(m.core.shared_gradient_rng()) == m.core.state_sha(rng)
    assert model.parameter.grad is None


@pytest.mark.parametrize('bad', ['state', 'RNG', 'python_rng', 'numpy_rng', 'grad', 'training', 'CUDA'])
def test_measurement_bookend_rejects_changed_state_RNG_or_allocated_grad(monkeypatch, bad):
    model = TinyState(); state = m.core.state_sha(model.state_dict()); rng = m.core.shared_gradient_rng()
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    if bad == 'state': model.parameter.data[0] = 3.
    elif bad == 'RNG': rng['torch_cpu'][0] ^= 1
    elif bad == 'python_rng': rng['python'] = ('changed',)
    elif bad == 'numpy_rng': rng['numpy'][1][0] ^= 1
    elif bad == 'training': model.training = True
    elif bad == 'CUDA': monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    else: model.parameter.grad = torch.ones_like(model.parameter)
    with pytest.raises(ValueError): m.measurement_bookend(model, state, rng)


def test_gradient_report_uses_full_panel_sum_before_dot_not_sum_of_chunk_dots():
    # Each per-chunk side/total dot is +1; the full-B96 sums have dot -4.
    sums = {name: [None] for name in ('total', 'activity', 'side', 'auxiliary')}
    for total, activity, side in (([1., -3.], [0., -3.], [1., 0.]),
                                  ([-3., 1.], [-3., 0.], [0., 1.])):
        for name, values in (('total', total), ('activity', activity), ('side', side)):
            m.accumulate_gradients(sums[name], vector(values))
    report = m.gradient_report(*(sums[key] for key in ('total', 'activity', 'side', 'auxiliary')),
                               tolerances=TOLERANCES)
    assert report['gSide_dot_gTotal'] == -4.
    assert report['gSide_dot_gActivity'] == -6.
    assert report['gSide_squared_L2'] == pytest.approx(2.)
    assert report['local_verdict'] == 'LOCAL_CONFLICT'
    assert report['component_gradient_contract_verified'] is True
    assert report['actual_auxiliary_shared_gradient_zero'] is True
    assert report['component_error_L2'] == 0.
    assert report['dot_identity_absolute_error'] < 1e-14


@pytest.mark.parametrize('total,activity,side,verdict', [
    ([2., 0.], [1., 0.], [1., 0.], 'NOT_LOCAL_CONFLICT'),
    ([0., 0.], [-1., 0.], [1., 0.], 'INDETERMINATE'),
    ([1., 0.], [1., 0.], [0., 0.], 'INDETERMINATE'),
    ([1e-8, 1.], [-1. + 1e-8, 1.], [1., 0.], 'INDETERMINATE'),
])
def test_gradient_report_respects_fixed_sign_margin(total, activity, side, verdict):
    report = m.gradient_report(vector(total), vector(activity), vector(side), [None], tolerances=TOLERANCES)
    assert report['local_verdict'] == verdict
    assert report['L2']['actual_auxiliary'] == 0.
    assert report['component_error_relative_L2'] <= 1e-5


@pytest.mark.parametrize('bad', ['mismatch', 'auxiliary', 'axis', 'tolerance'])
def test_gradient_report_rejects_broken_component_or_auxiliary_proofs(bad):
    total, activity, side, auxiliary = vector([2.]), vector([1.]), vector([1.]), [None]
    tolerances = copy.deepcopy(TOLERANCES)
    if bad == 'mismatch': total = vector([3.])
    elif bad == 'auxiliary': auxiliary = vector([1e-30]); total = vector([2.])
    elif bad == 'axis': total = []
    else: tolerances['component_gradient_global_relative_L2_max'] = .1
    with pytest.raises(ValueError):
        m.gradient_report(total, activity, side, auxiliary, tolerances=tolerances)


def completed_counts():
    return {'full_feature_archive_loads': 1, 'feature_arrays_decoded': 27,
            'target_archive_loads': 1, 'target_arrays_decoded': 14,
            'capture_archive_loads': 1, 'capture_arrays_decoded': 3,
            'checkpoint_deserializations': 1, 'model_constructions': 1, 'model_state_install_calls': 1,
            'parity_forwards': 96, 'gradient_forwards': 96, 'NN_forwards': 192, 'autograd_calls': 384,
            'backward_calls': 0, 'optimizer_constructions': 0, 'optimizer_restores': 0,
            'optimizer_updates': 0, 'checkpoint_RNG_restore_calls': 0, 'fits': 0,
            'prior_refits': 0, 'new_source_HTTP_calls': 0}


def test_exact_completed_counts_include_all_forwards_and_derivatives_but_no_updates():
    assert m.validate_counts(completed_counts(), {'NN_forwards_max': 192, 'autograd_calls_max': 384}) is True


@pytest.mark.parametrize('bad', ['parity_forwards', 'gradient_forwards', 'autograd_calls',
    'NN_forwards', 'optimizer_updates', 'checkpoint_RNG_restore_calls', 'bool', 'extra'])
def test_count_proof_rejects_missing_overrun_training_or_boolean_counts(bad):
    counts = completed_counts()
    if bad == 'bool': counts['checkpoint_deserializations'] = True
    elif bad == 'extra': counts['unbudgeted_action'] = 0
    else: counts[bad] += 1
    with pytest.raises(ValueError, match='EXACT_ACTUAL_COUNTS'):
        m.validate_counts(counts, {'NN_forwards_max': 192, 'autograd_calls_max': 384})


def test_safe_FINAL_loader_authenticates_binding_and_full_proof_with_CPU_weights_only(monkeypatch):
    inputs = {key: {'path': '/synthetic/' + key + ('.pt' if key == 'final_checkpoint' else '.json'),
                    'sha256': 'a' * 64}
              for key in ('final_checkpoint', 'original_epoch_plan', 'completed_model_plan')}
    metadata = {key: {'synthetic': key} for key in ('final_receipt', 'original_epoch_plan', 'trajectory')}
    plan = {'inputs': inputs, 'metadata': metadata}; cp = {'synthetic': 'FINAL'}
    calls = []
    monkeypatch.setattr(m, 'reference', lambda value: calls.append(('reference', value)))
    def load(path, **kwargs):
        calls.append(('load', path, kwargs)); return cp
    def validate(*args): calls.append(('validate', args))
    monkeypatch.setattr(torch, 'load', load)
    monkeypatch.setattr(m.readout, 'validate_final', validate)
    assert m.safe_load_final(plan) is cp
    assert calls == [('reference', inputs['final_checkpoint']),
        ('load', inputs['final_checkpoint']['path'], {'map_location': 'cpu', 'weights_only': True}),
        ('validate', (cp, metadata['final_receipt'], metadata['original_epoch_plan'],
                      inputs['original_epoch_plan'], metadata['trajectory'], inputs['completed_model_plan']))]


def lifecycle(monkeypatch, tmp_path):
    monkeypatch.setattr(m, 'ROOT', tmp_path / 'OUT')
    monkeypatch.setattr(m, 'PLAN', tmp_path / 'PLAN.json')
    monkeypatch.setattr(m, 'POLICY', tmp_path / 'POLICY.json')
    m.PLAN.write_text('{"synthetic":true}'); m.POLICY.write_text('{"training_enabled":false}')
    planref, pref = m.ref(m.PLAN), m.ref(m.POLICY)
    plan = {'inputs': {role: {'path': str(tmp_path / filename), 'sha256': 'a' * 64}
                      for role, filename in {'feature_archive': 'FEATURES.npz', 'capture': 'CAPTURE.npz',
                        'target_arrays': 'TARGETS.npz', 'final_checkpoint': 'FINAL.pt'}.items()}}
    monkeypatch.setattr(m, 'inspect', lambda *a, **k: (plan, planref, pref))
    monkeypatch.setattr(m, 'checked_inputs', lambda plan: None)
    monkeypatch.setattr(m, 'source_clean', lambda plan: True)
    monkeypatch.setattr(m, 'require_capped_cpu_audit_execution', lambda: {
        'memory_max': 4*1024**3, 'memory_high': 4*1024**3, 'swap': 512*1024**2,
        'numerical_threads': 1, 'pids': 64, 'cgroup_path': '/sys/fs/cgroup/synthetic'})
    monkeypatch.setattr(m, 'require_capped_lock_ancestry', lambda: {'scope': '/synthetic'})
    hooks, timers = [], []
    monkeypatch.setattr(sys, 'addaudithook', hooks.append)
    monkeypatch.setattr(m.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(m.signal, 'setitimer', lambda *a: timers.append(a))
    monkeypatch.setattr(m, 'calculate', lambda *a, **k: {'synthetic': True})
    return plan, planref, pref, hooks, timers


def test_preflight_exact_CLI_has_zero_decodes_derivatives_and_no_claim(monkeypatch, tmp_path, capsys):
    _, planref, _, hooks, _ = lifecycle(monkeypatch, tmp_path)
    def forbidden(*a, **k): raise AssertionError('NO_PAYLOAD_OR_CALCULATE_IN_PREFLIGHT')
    monkeypatch.setattr(m, 'calculate', forbidden); monkeypatch.setattr(torch, 'load', forbidden)
    monkeypatch.setattr(np, 'load', forbidden)
    monkeypatch.setattr(sys, 'argv', ['audit', '--plan-sha256', planref['sha256'], '--preflight'])
    m.main(); result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'PREFLIGHT_PROVIDER_HLC_FROZEN_GRADIENT_PASS'
    assert result['plan'] == planref and result['claim_created'] is False
    assert all(result[key] == 0 for key in ('checkpoint_deserializations', 'NPZ_arrays_decoded',
                                           'models', 'NN_forwards', 'autograd_calls'))
    assert result['CUDA_initialized'] is False and hooks == [] and not m.ROOT.exists()
    monkeypatch.setattr(sys, 'argv', sys.argv + ['--preflight'])
    with pytest.raises(ValueError, match='EXACT_CLI'): m.main()


@pytest.mark.parametrize('failure', [None, 'calculate', 'bookend'])
def test_one_use_success_and_consumed_failure_preserve_claim_and_progress(monkeypatch, tmp_path, failure):
    _, planref, pref, _, timers = lifecycle(monkeypatch, tmp_path)
    def calculate(*args, **kwargs):
        kwargs['progress']['phase'] = 'SYNTHETIC_DERIVATIVE_BOUNDARY'
        (m.ROOT / 'PRESERVED.json').write_text('{"preserved":true}')
        if failure == 'calculate': raise ValueError('synthetic_calculate')
        return {'synthetic': True}
    monkeypatch.setattr(m, 'calculate', calculate)
    if failure == 'bookend': monkeypatch.setattr(m, 'source_clean', lambda plan: False)
    if failure:
        with pytest.raises(ValueError): m.run(planref['sha256'])
    else:
        outputs = m.run(planref['sha256'])
        assert outputs['result'] == m.ref(m.ROOT / 'RESULT.json')
        result = json.loads((m.ROOT / 'RESULT.json').read_text())
        assert result['status'] == 'COMPLETE_CONDITIONAL_PROVIDER_HLC_FROZEN_GRADIENT'
        assert result['test_data_used'] is result['economic_edge_proven'] is False
    terminal = json.loads((m.ROOT / 'TERMINAL.json').read_text())
    claim = json.loads((m.ROOT / 'CLAIM.json').read_text())
    assert terminal['status'] == ('FAILED_CONSUMED' if failure else 'COMPLETE')
    assert terminal['exit_code'] == int(bool(failure)) and terminal['plan'] == planref
    assert terminal['remaining_invocations'] == 0 and terminal['relaunch_allowed'] is False
    assert claim['policy_at_claim'] == pref and claim['training_authorized'] is False
    assert (m.ROOT / 'RESULT.json').exists() is (failure is None)
    assert (m.ROOT / 'PRESERVED.json').is_file()
    if failure: assert terminal['progress']['phase'] == 'SYNTHETIC_DERIVATIVE_BOUNDARY'
    assert timers[-1] == (m.signal.ITIMER_REAL, 0)
    # A claimed root cannot be overwritten by a second invocation.
    with pytest.raises(FileExistsError): m.run(planref['sha256'])


def test_runtime_audit_allows_only_exact_one_CP_three_NPZ_and_no_network(monkeypatch, tmp_path):
    plan, planref, _, hooks, _ = lifecycle(monkeypatch, tmp_path)
    m.run(planref['sha256']); assert len(hooks) == 1
    for binding in plan['inputs'].values(): hooks[0]('open', (binding['path'],))
    for filename in ('unbound.pt', 'INITIAL.pt', 'foreign.npz', 'other.parquet', 'other.pkl'):
        with pytest.raises(RuntimeError, match='PAYLOAD_FORBIDDEN'):
            hooks[0]('open', (str(tmp_path / filename),))
    for event in ('socket.connect', 'socket.getaddrinfo', 'urllib.Request'):
        with pytest.raises(RuntimeError, match='NETWORK_FORBIDDEN'): hooks[0](event, ())


@pytest.mark.parametrize('admitted', [False, True])
def test_exact_one_use_gate_keeps_training_TEST_and_relaunch_closed(admitted):
    binding = {'path': '/synthetic/PLAN.json', 'sha256': 'a' * 64}
    gate = m.gate_expected(binding, admitted)
    assert gate['plan'] == binding and gate['execution_authorized'] is admitted
    assert gate['remaining_invocations'] == int(admitted)
    assert gate['training_authorized'] is gate['test_authorized'] is gate['relaunch_allowed'] is False


def test_inspect_rejects_every_global_flag_scope_training_and_existing_root(monkeypatch, tmp_path):
    planfile, policyfile = tmp_path / 'PLAN.json', tmp_path / 'POLICY.json'
    monkeypatch.setattr(m, 'PLAN', planfile); monkeypatch.setattr(m, 'POLICY', policyfile)
    monkeypatch.setattr(m, 'ROOT', tmp_path / 'OUT'); monkeypatch.setattr(m, 'REPO', Path.cwd())
    plan = {'source_base_commit': '1' * 40}; planfile.write_text(json.dumps(plan)); planref = m.ref(planfile)
    policy = dict.fromkeys(m.saved.FLAGS, False); policy[m.STAGE] = m.gate_expected(planref, False)
    monkeypatch.setattr(m, 'build_plan', lambda **kwargs: plan)
    monkeypatch.setattr(m, 'source_clean', lambda value: True)
    def inspect():
        policyfile.write_text(json.dumps(policy)); return m.inspect(planref['sha256'], admitted=False)
    assert inspect()[0] == plan and not m.ROOT.exists()
    for key in m.saved.FLAGS:
        policy[key] = True
        with pytest.raises(ValueError, match='ONE_USE_ALL_TRAINING_CLOSED'): inspect()
        policy[key] = False
    policy[m.STAGE]['training_authorized'] = True
    with pytest.raises(ValueError, match='ONE_USE_ALL_TRAINING_CLOSED'): inspect()
    policy[m.STAGE]['training_authorized'] = False; m.ROOT.mkdir()
    with pytest.raises(ValueError, match='FRESH_ROOT'): inspect()


class SyntheticBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.25, dtype=torch.float32))


class SyntheticModel(torch.nn.Module):
    """One scalar shared parameter, detached auxiliary heads, no native network."""
    def __init__(self):
        super().__init__()
        self.backbone = SyntheticBackbone()
    def forward(self, **features):
        z = self.backbone.weight * features['x']
        return {'activity_logits': torch.cat((z, z.detach().expand(-1, 4)), dim=1),
                'side_logits': torch.cat((-2*z, (-2*z).detach().expand(-1, 4)), dim=1)}


def synthetic_calculation(monkeypatch):
    from types import SimpleNamespace
    import weakref
    class Context(dict): pass
    n = 96
    issues = np.arange(n, dtype=np.int64) * 300_000_000_000 + int(m.train.START)
    rows = np.arange(n, dtype=np.int64)
    x = np.linspace(.1, .3, n, dtype=np.float32).reshape(n, 1)
    features = {key: torch.from_numpy(x.copy()) for key in ('x', *('dummy'+str(j) for j in range(8)))}
    truth = {'issue_ns': issues, 'activity': np.zeros((n, 5), np.float32),
             'side_up': np.zeros((n, 5), np.float32), 'activity_known': np.ones((n, 5), bool),
             'side_known': np.ones((n, 5), bool)}
    packet = {'features': features, 'truth': truth}
    capture = {'issue_ns': issues.copy(), 'activity_logits': np.repeat(x*.25, 5, axis=1),
               'side_logits': np.repeat(-2*(x*.25), 5, axis=1)}
    state = {'backbone.weight': torch.tensor(.25, dtype=torch.float32)}
    plan = {'metadata': {'adapter_metadata': {}, 'original_epoch_plan': {'metadata': {'architecture': {}}},
                        'final_receipt': {'proof': {'model_state': m.core.state_sha(state)}}},
            'inputs': {'capture': {'path': '/synthetic/CAPTURE.npz', 'sha256': 'a'*64}},
            'panel': {'issue_ns_sha256': m.core.digest(issues), 'source_rows_sha256': m.core.digest(rows)},
            'TRAIN_issue_ns_sha256': m.core.digest(issues), 'capture_array_headers': {},
            'budget': {'packet_feature_bytes': sum(t.numel()*t.element_size() for t in features.values()),
                       'capture_ndarray_bytes_decoded': sum(a.nbytes for a in capture.values()),
                       'NN_forwards_max': 192, 'autograd_calls_max': 384},
            'tolerances': copy.deepcopy(TOLERANCES), 'cache_scope': {'synthetic': True}, 'limits': []}
    released, models = [], []
    def load_context(_):
        ctx = Context(provider=SimpleNamespace(bar_start_ns=np.array([int(m.train.START)], np.int64)),
                      source_rows=rows, normalization={}, prior={}, input_binding={}, target_contract={})
        released.append(weakref.ref(ctx)); return ctx
    def factory(**kwargs):
        assert kwargs['device'] == torch.device('cpu') and kwargs['backbone_gradient_policy'] == 'M5_primary'
        assert released[0]() is None  # The full context must be gone before model construction.
        model = SyntheticModel(); models.append(model); return model
    def slice_truth(value, take, **kwargs):
        return {key: array[take].copy() for key, array in value.items()}
    def canonical_loss(*, model_output, truth, **kwargs):
        terms = []
        for output, label, mask in (('activity_logits', 'activity', 'activity_known'),
                                    ('side_logits', 'side_up', 'side_known')):
            labels = torch.from_numpy(truth[label]); known = torch.from_numpy(truth[mask])
            bce = torch.nn.functional.binary_cross_entropy_with_logits(model_output[output], labels, reduction='none')
            terms.append(torch.where(known, bce, torch.zeros_like(bce)).sum())
        count = int(truth['activity_known'].sum())
        return {'loss': sum(terms)/count, 'count': count}
    monkeypatch.setattr(m.adapter, 'load_context', load_context)
    monkeypatch.setattr(m.adapter, 'packet', lambda context, take: packet)
    monkeypatch.setattr(m.train, 'slice_provider_hlc_truth', slice_truth)
    monkeypatch.setattr(m.saved, 'load_fields', lambda binding, fields, headers: {key: capture[key] for key in fields})
    monkeypatch.setattr(m.model_owner, 'build_native_future_state_model', factory)
    monkeypatch.setattr(m, 'safe_load_final', lambda plan: {'model_state': state})
    monkeypatch.setattr(m.objective, 'native_future_state_loss', canonical_loss)
    monkeypatch.setattr(torch, 'set_num_threads', lambda count: None)
    monkeypatch.setattr(torch, 'get_num_threads', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    return plan, capture, models


def test_full96_synthetic_path_preserves_Dfull_components_F64_counts_and_state(monkeypatch):
    plan, _, models = synthetic_calculation(monkeypatch); progress = {}
    report = m.calculate(plan, progress=progress)
    assert report['known_activity_denominator_full_B96'] == 480
    assert report['per_TF_support'] == {'activity': [96]*5, 'side': [96]*5}
    assert report['counts'] == completed_counts()
    assert report['logit_parity']['passed'] is True and report['logit_parity']['rows'] == 96
    assert report['gradient_proof']['component_gradient_contract_verified'] is True
    assert report['gradient_proof']['actual_auxiliary_shared_gradient_zero'] is True
    assert report['parameter_axis'] == {'names': ['weight'], 'tensors': 1,
        'aggregation_dtype': 'float64', 'None_is_mathematical_zero_only': True}
    assert report['model_state_and_local_RNG_bookend_passed'] is True
    assert report['all_parameter_grad_fields_None'] is True
    assert report['actual_training_cause'] == report['selected_causal_fix'] == 'UNKNOWN'
    assert report['later_generalization'] == report['economic_edge'] == 'UNKNOWN'
    assert report['all20_approved'] is False
    assert len(models) == 1 and models[0].backbone.weight.item() == .25
    assert models[0].backbone.weight.grad is None
    assert progress['phase'] == 'COMPLETE_FROZEN_FULL_B96_GRADIENT'


@pytest.mark.parametrize('nonfinite', [False, True])
def test_saved_parity_failure_precedes_every_derivative_and_cannot_relax_tolerance(monkeypatch, tmp_path, nonfinite):
    plan, capture, models = synthetic_calculation(monkeypatch)
    observed = capture['side_logits'][95].astype(np.float64).copy()
    if nonfinite: capture['side_logits'][95, 4] = np.nan
    else: capture['side_logits'][95, 4] += .01
    expected = capture['side_logits'][95].astype(np.float64)
    absolute_delta = np.abs(observed - expected)
    expected_values = expected[:4].tolist() + ['nan'] if nonfinite else expected.tolist()
    delta_values = [0., 0., 0., 0., 'nan'] if nonfinite else absolute_delta.tolist()
    max_delta = 'nan' if nonfinite else float(absolute_delta.max())
    failure = {'index': 95, 'issue_ns': int(capture['issue_ns'][95]), 'head': 'side_logits',
        'timeframes': ['M5', 'M15', 'H1', 'H4', 'D1'],
        'observed_logits': observed.tolist(), 'expected_logits': expected_values,
        'absolute_delta': delta_values, 'max_absolute_delta': max_delta, 'nonfinite_values': nonfinite,
        'fixed_tolerance': TOLERANCES['saved_FINAL_logits'], 'passed_mask': [True, True, True, True, False]}
    calculate_owner = m.calculate
    lifecycle_plan, planref, _, _, _ = lifecycle(monkeypatch, tmp_path)
    plan['inputs'].update(lifecycle_plan['inputs'])
    monkeypatch.setattr(m, 'inspect', lambda *a, **k: (plan, planref, m.ref(m.POLICY)))
    seen_progress = []
    def calculate(bound_plan, *, progress):
        seen_progress.append(progress)
        return calculate_owner(bound_plan, progress=progress)
    def forbidden(*args, **kwargs): raise AssertionError('NO_DERIVATIVE_BEFORE_FULL96_PARITY')
    monkeypatch.setattr(m, 'calculate', calculate)
    monkeypatch.setattr(torch.autograd, 'grad', forbidden)
    with pytest.raises(ValueError, match='SAVED_FINAL_LOGIT_PARITY:side_logits'):
        m.run(planref['sha256'])
    progress = seen_progress[0]
    terminal = json.loads((m.ROOT / 'TERMINAL.json').read_text())
    assert progress['parity_failure'] == failure
    assert json.loads(json.dumps(progress, allow_nan=False)) == progress
    assert terminal['progress'] == progress
    assert terminal['progress']['parity_failure'] == failure
    assert terminal['status'] == 'FAILED_CONSUMED' and terminal['result'] is None
    assert not (m.ROOT / 'RESULT.json').exists()
    assert progress['counts']['parity_forwards'] == 96
    assert progress['counts']['autograd_calls'] == progress['counts']['gradient_forwards'] == 0
    assert models[0].backbone.weight.grad is None
    assert plan['tolerances'] == TOLERANCES


@pytest.mark.parametrize('budget_field', ['NN_forwards_max', 'autograd_calls_max'])
def test_runtime_forward_and_derivative_caps_fail_without_shrinking_panel(monkeypatch, budget_field):
    plan, _, models = synthetic_calculation(monkeypatch)
    plan['budget'][budget_field] = 1
    progress = {}
    with pytest.raises(ValueError, match='FORWARD_BUDGET|AUTOGRAD_BUDGET'):
        m.calculate(plan, progress=progress)
    assert models[0].backbone.weight.grad is None
    assert progress['counts']['optimizer_updates'] == progress['counts']['backward_calls'] == 0
    assert plan['panel']['issue_ns_sha256'] == m.core.digest(
        np.arange(96, dtype=np.int64)*300_000_000_000 + int(m.train.START))


def test_first_parity_appends_authority_preserving_old_functions_constants_and_main_tail():
    import subprocess
    anchor = '3231f1d5961e1c8eeb4d14e2b0e17fa48dc9b15b'
    before = subprocess.check_output(['git', 'show', anchor + ':' + m.SELF], cwd=m.REPO, text=True)
    after = Path(m.__file__).read_text()
    old_tree, new_tree = ast.parse(before), ast.parse(after)
    old_functions = {node.name: node for node in old_tree.body if isinstance(node, ast.FunctionDef)}
    new_functions = {node.name: node for node in new_tree.body if isinstance(node, ast.FunctionDef)}
    for name, node in old_functions.items():
        if name != 'main':
            assert ast.get_source_segment(before, node) == ast.get_source_segment(after, new_functions[name])
    assignments = lambda tree: [node for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))]
    for node in assignments(old_tree):
        assert ast.dump(node, include_attributes=False) in [ast.dump(current, include_attributes=False)
                                                          for current in assignments(new_tree)]
    original_main, current_main = old_functions['main'], new_functions['main']
    dispatch = current_main.body[0]
    assert isinstance(dispatch, ast.If)
    expected_guard = ast.parse("sys.argv[1:2] == ['--first-parity-plan-sha256']", mode='eval').body
    assert ast.dump(dispatch.test, include_attributes=False) == ast.dump(expected_guard, include_attributes=False)
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == 'main_first_parity' for node in ast.walk(dispatch))
    assert ast.dump(ast.Module(body=current_main.body[1:], type_ignores=[]), include_attributes=False) == \
           ast.dump(ast.Module(body=original_main.body, type_ignores=[]), include_attributes=False)


FIRST_PARITY_ROUTES = [
    {'name': 'CPU_NO_GRAD_EQUAL_TF', 'device': 'cpu', 'batch_rows': 1, 'model_training': False,
     'grad_enabled': False, 'grouped': True, 'equal_tf': True, 'capture': False},
    {'name': 'CPU_NO_GRAD_UNMERGED', 'device': 'cpu', 'batch_rows': 1, 'model_training': False,
     'grad_enabled': False, 'grouped': True, 'equal_tf': False, 'capture': False},
    {'name': 'CPU_GRAD_ENABLED_UNMERGED', 'device': 'cpu', 'batch_rows': 1, 'model_training': False,
     'grad_enabled': True, 'grouped': True, 'equal_tf': False, 'capture': False},
]


def first_parity_counts():
    counts = completed_counts()
    counts.update(parity_forwards=3, gradient_forwards=0, NN_forwards=3, autograd_calls=0,
                  canonical_loss_calls=0, hidden_capture_calls=0,
                  grad_enabled_forward_calls=1, no_grad_forward_calls=2)
    return counts


def test_first_parity_comparison_preserves_fixed_tolerance_and_zero_reference_semantics():
    reference = np.array([[0., 0., 1., -1., .001]], np.float32)
    candidate = np.array([[0., 1., 1.+1e-7, -1.+2e-5, 0.]], np.float32)
    result = m.first_parity_compare(reference, candidate, tolerance=TOLERANCES['saved_FINAL_logits'])
    delta = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))[0]
    assert result['absolute_delta'] == delta.tolist()
    assert result['max_absolute_delta'] == float(delta.max())
    assert result['reference_values'] == reference[0].astype(np.float64).tolist()
    assert result['candidate_values'] == candidate[0].astype(np.float64).tolist()
    assert result['reference_zero_mask'] == [True, True, False, False, False]
    assert result['relative_error'][:2] == [None, None]
    assert result['relative_error'][2:] == (delta[2:] / np.abs(reference[0, 2:].astype(np.float64))).tolist()
    assert result['zero_reference_relative_status'] == ['ZERO_OVER_ZERO', 'INFINITE', None, None, None]
    assert result['passed_mask'] == [True, False, True, False, False]
    assert result['all_within_original_tolerance'] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('bad', ['shape', 'dtype', 'nan', 'inf', 'tolerance'])
def test_first_parity_comparison_rejects_nonfinite_or_unlocked_contract(bad):
    reference, candidate = np.ones((1, 5), np.float32), np.ones((1, 5), np.float32)
    tolerance = copy.deepcopy(TOLERANCES['saved_FINAL_logits'])
    if bad == 'shape': candidate = candidate[0]
    elif bad == 'dtype': candidate = candidate.astype(np.float64)
    elif bad == 'nan': candidate[0, 0] = np.nan
    elif bad == 'inf': reference[0, 0] = np.inf
    else: tolerance['atol'] = .1
    with pytest.raises(ValueError): m.first_parity_compare(reference, candidate, tolerance=tolerance)


def test_first_parity_completed_counts_require_three_forwards_zero_derivatives():
    assert m.first_parity_validate_counts(first_parity_counts(), {'NN_forwards_max': 3, 'autograd_calls_max': 0}) is True


@pytest.mark.parametrize('bad', ['NN_forwards', 'autograd_calls', 'canonical_loss_calls',
                                 'hidden_capture_calls', 'grad_enabled_forward_calls', 'bool'])
def test_first_parity_count_proof_rejects_unbudgeted_actions_or_wrong_modes(bad):
    counts = first_parity_counts()
    if bad == 'bool': counts['checkpoint_deserializations'] = True
    else: counts[bad] += 1
    with pytest.raises(ValueError):
        m.first_parity_validate_counts(counts, {'NN_forwards_max': 3, 'autograd_calls_max': 0})


def first_parity_lifecycle(monkeypatch, tmp_path):
    plan, _, pref, hooks, timers = lifecycle(monkeypatch, tmp_path)
    monkeypatch.setattr(m, 'PARITY_ROOT', tmp_path / 'PARITY_OUT')
    monkeypatch.setattr(m, 'PARITY_PLAN', tmp_path / 'PARITY_PLAN.json')
    m.PARITY_PLAN.write_text('{"synthetic_first_parity":true}')
    binding = m.ref(m.PARITY_PLAN)
    monkeypatch.setattr(m, 'inspect_first_parity', lambda *a, **k: (plan, binding, pref))
    monkeypatch.setattr(m, 'calculate_first_parity', lambda *a, **k: {
        'all_routes_within_saved_original_tolerance': False, 'synthetic': True})
    return plan, binding, pref, hooks, timers


def test_first_parity_preflight_dispatch_has_no_payload_forward_or_claim(monkeypatch, tmp_path, capsys):
    _, binding, _, hooks, _ = first_parity_lifecycle(monkeypatch, tmp_path)
    def forbidden(*a, **k): raise AssertionError('NO_FIRST_PARITY_PAYLOAD_IN_PREFLIGHT')
    monkeypatch.setattr(m, 'calculate_first_parity', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden); monkeypatch.setattr(np, 'load', forbidden)
    monkeypatch.setattr(sys, 'argv', ['audit', '--first-parity-plan-sha256', binding['sha256'], '--preflight'])
    m.main(); report = json.loads(capsys.readouterr().out)
    assert report['plan'] == binding and report['claim_created'] is False
    assert report['NN_forwards'] == report['autograd_calls'] == 0
    assert report['checkpoint_deserializations'] == report['NPZ_arrays_decoded'] == report['models'] == 0
    assert report['CUDA_initialized'] is False and hooks == []
    assert not m.PARITY_ROOT.exists() and not m.ROOT.exists()
    monkeypatch.setattr(sys, 'argv', sys.argv + ['--preflight'])
    with pytest.raises(ValueError, match='EXACT_CLI'): m.main()


@pytest.mark.parametrize('failure', [False, True])
def test_first_parity_mismatch_completion_and_partial_failure_are_consumed(monkeypatch, tmp_path, failure):
    _, binding, pref, _, timers = first_parity_lifecycle(monkeypatch, tmp_path)
    completed = {'name': 'CPU_NO_GRAD_EQUAL_TF', 'finite_mismatch_is_recorded': True}
    def calculate(plan, *, progress):
        progress.update(routes=[completed], active_route='CPU_NO_GRAD_UNMERGED')
        if failure: raise ValueError('SYNTHETIC_SECOND_ROUTE_FAILURE')
        return {'all_routes_within_saved_original_tolerance': False, 'routes': [completed]}
    monkeypatch.setattr(m, 'calculate_first_parity', calculate)
    if failure:
        with pytest.raises(ValueError, match='SYNTHETIC_SECOND_ROUTE_FAILURE'):
            m.run_first_parity(binding['sha256'])
    else:
        outputs = m.run_first_parity(binding['sha256'])
        assert outputs['result'] == m.ref(m.PARITY_ROOT / 'RESULT.json')
        result = json.loads((m.PARITY_ROOT / 'RESULT.json').read_text())
        assert result['report']['all_routes_within_saved_original_tolerance'] is False
        assert result['report']['routes'] == [completed]
        assert result['test_data_used'] is result['economic_edge_proven'] is False
    terminal = json.loads((m.PARITY_ROOT / 'TERMINAL.json').read_text())
    claim = json.loads((m.PARITY_ROOT / 'CLAIM.json').read_text())
    assert terminal['status'] == ('FAILED_CONSUMED' if failure else 'COMPLETE')
    assert terminal['stage'] == claim['stage'] == m.PARITY_STAGE
    assert terminal['exit_code'] == int(failure) and terminal['remaining_invocations'] == 0
    assert terminal['relaunch_allowed'] is claim['training_authorized'] is False
    assert claim['policy_at_claim'] == pref and terminal['plan'] == binding
    assert (m.PARITY_ROOT / 'RESULT.json').exists() is (not failure)
    assert not m.ROOT.exists()  # Dedicated new-stage outputs never consume the old output root.
    if failure:
        assert terminal['progress']['routes'] == [completed]
        assert terminal['progress']['active_route'] == 'CPU_NO_GRAD_UNMERGED'
        json.dumps(terminal, allow_nan=False)
    assert timers[-1] == (m.signal.ITIMER_REAL, 0)
    with pytest.raises(FileExistsError): m.run_first_parity(binding['sha256'])


@pytest.mark.parametrize('admitted', [False, True])
def test_first_parity_exact_gate_cannot_authorize_gradient_training_or_relaunch(admitted):
    binding = {'path': '/synthetic/FIRST_PARITY_PLAN.json', 'sha256': 'b'*64}
    gate = m.gate_first_parity_expected(binding, admitted)
    assert gate['plan'] == binding and gate['execution_authorized'] is admitted
    assert gate['remaining_invocations'] == int(admitted)
    assert gate['relaunch_allowed'] is gate['training_authorized'] is gate['test_authorized'] is False


def test_first_parity_runtime_whitelist_and_network_guard_reuse_exact_four_payload_roles(monkeypatch, tmp_path):
    plan, binding, _, hooks, _ = first_parity_lifecycle(monkeypatch, tmp_path)
    m.run_first_parity(binding['sha256']); assert len(hooks) == 1
    for value in plan['inputs'].values(): hooks[0]('open', (value['path'],))
    for filename in ('UNBOUND.pt', 'OTHER_FEATURES.npz', 'foreign.parquet', 'other.pkl'):
        with pytest.raises(RuntimeError, match='PAYLOAD_FORBIDDEN'):
            hooks[0]('open', (str(tmp_path / filename),))
    for event in ('socket.connect', 'socket.getaddrinfo', 'urllib.Request'):
        with pytest.raises(RuntimeError, match='NETWORK_FORBIDDEN'): hooks[0](event, ())


def first_parity_calculation(monkeypatch, *, fail_route=None, corrupt=None):
    from contextlib import nullcontext
    from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as body
    plan, capture, models = synthetic_calculation(monkeypatch)
    plan.update(routes=copy.deepcopy(FIRST_PARITY_ROUTES),
                fixed_original_tolerance=copy.deepcopy(TOLERANCES['saved_FINAL_logits']))
    plan['budget'].update(NN_forwards_max=3, autograd_calls_max=0)
    packet_owner = m.adapter.packet
    def packet(context, take):
        value = packet_owner(context, take)
        for j, tf in enumerate(('m5', 'm15', 'h1', 'h4', 'd1')):
            value['features']['seq_' + tf] = value['features'].pop('dummy' + str(j))
        return value
    monkeypatch.setattr(m.adapter, 'packet', packet)
    calls, tokens = [], []
    def helper(model, features, *, grouped, equal_tf, capture):
        assert body._BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False
        assert body._MODEL_FINITE_CHECK_SCOPE.get() is None
        assert len(model._forward_pre_hooks) == 1
        assert all(not child._forward_hooks and not child._forward_pre_hooks
                   for child in model.modules() if child is not model)
        index = len(calls)
        calls.append({'grouped': grouped, 'equal_tf': equal_tf, 'capture': capture,
                      'grad_enabled': torch.is_grad_enabled(), 'model_training': model.training,
                      'feature_ids': {key: id(value) for key, value in features.items()}})
        if index == fail_route: raise ValueError('SYNTHETIC_HELPER_ROUTE_FAILURE')
        with body.model_finite_check_scope('grouped'), (body.batch_equal_length_mtf_eval_scope()
                                                       if equal_tf else nullcontext()):
            output = model(**features)
            output = {key: value + index*.001 for key, value in output.items()}
            groups = {}
            for tf in ('M5', 'M15', 'H1', 'H4', 'D1'):
                groups.setdefault(str(features['seq_' + tf.lower()].shape[1]), []).append(tf)
            activation = {'batch_eval_active': equal_tf and not torch.is_grad_enabled(),
                'model_training': model.training, 'grad_enabled': torch.is_grad_enabled(),
                'scope_enabled': body._BATCH_EQUAL_LENGTH_MTF_EVAL.get(),
                'eligible_equal_length_TF_groups': groups}
        if index == 0:
            if corrupt == 'state': model.backbone.weight.data.add_(1.)
            elif corrupt == 'grad': model.backbone.weight.grad = torch.ones_like(model.backbone.weight)
            elif corrupt == 'equal_scope':
                tokens.append((body._BATCH_EQUAL_LENGTH_MTF_EVAL, body._BATCH_EQUAL_LENGTH_MTF_EVAL.set(True)))
            elif corrupt == 'finite_scope':
                tokens.append((body._MODEL_FINITE_CHECK_SCOPE, body._MODEL_FINITE_CHECK_SCOPE.set(object())))
        return output, None, activation
    def forbidden(*a, **k): raise AssertionError('NO_FIRST_PARITY_DERIVATIVE_OR_LOSS')
    monkeypatch.setattr(m.core, 'future_cuda_forward', helper)
    monkeypatch.setattr(torch.autograd, 'grad', forbidden)
    monkeypatch.setattr(m.objective, 'native_future_state_loss', forbidden)
    if corrupt == 'RNG':
        original_rng = m.core.shared_gradient_rng
        seen = []
        def rng():
            value = original_rng()
            if seen: value['torch_cpu'][0] ^= 1
            seen.append(True); return value
        monkeypatch.setattr(m.core, 'shared_gradient_rng', rng)
    return plan, capture, models, calls, tokens


@pytest.mark.parametrize('ambient_grad', [False, True])
def test_first_parity_three_exact_routes_complete_all_finite_mismatches_without_derivatives(monkeypatch, ambient_grad):
    from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as body
    plan, capture, models, calls, _ = first_parity_calculation(monkeypatch)
    original = {key: capture[key][:1].copy() for key in ('activity_logits', 'side_logits')}
    for key in original: capture[key][0] += .1
    progress = {}
    with torch.set_grad_enabled(ambient_grad):
        report = m.calculate_first_parity(plan, progress=progress)
        assert torch.is_grad_enabled() is ambient_grad
    assert report['counts'] == first_parity_counts()
    assert len(calls) == len(report['routes']) == 3
    assert report['all_routes_within_saved_original_tolerance'] is False
    assert report['measurement_completed'] is True and report['full96_gradient_qualification'] is False
    assert report['selected_issue_ns'] == int(capture['issue_ns'][0]) and report['selected_source_row'] == 0
    assert report['retained_B96_packet_unmutated'] is True
    assert len(report['first_row_feature_proof']) == 9
    assert report['original_B96_known_activity_denominator_identity_only'] == 480
    for index, (call, record, route) in enumerate(zip(calls, report['routes'], FIRST_PARITY_ROUTES)):
        assert record['name'] == route['name'] and record['requested'] == route
        assert {key: call[key] for key in ('grouped', 'equal_tf', 'capture', 'grad_enabled', 'model_training')} == \
               {key: route[key] for key in ('grouped', 'equal_tf', 'capture', 'grad_enabled', 'model_training')}
        assert call['feature_ids'] == calls[0]['feature_ids']
        assert record['activation']['batch_eval_active'] is (index == 0)
        assert record['activation']['scope_enabled'] is route['equal_tf']
        for head, base in original.items():
            values = base + np.float32(index*.001)
            assert record['logits'][head] == {'values': values[0].astype(np.float64).tolist(),
                'shape': [1, 5], 'dtype': np.dtype('float32').str, 'sha256': m.core.digest(values)}
            assert record['vs_saved_FINAL'][head] == m.first_parity_compare(
                capture[head][:1], values, tolerance=TOLERANCES['saved_FINAL_logits'])
            assert record['vs_saved_FINAL'][head]['all_within_original_tolerance'] is False
    assert [(pair['reference_route'], pair['candidate_route']) for pair in report['pairwise']] == [
        ('CPU_NO_GRAD_EQUAL_TF', 'CPU_NO_GRAD_UNMERGED'),
        ('CPU_NO_GRAD_UNMERGED', 'CPU_GRAD_ENABLED_UNMERGED'),
        ('CPU_NO_GRAD_EQUAL_TF', 'CPU_GRAD_ENABLED_UNMERGED')]
    by_name = {record['name']: record for record in report['routes']}
    for pair in report['pairwise']:
        for head in original:
            left = np.array([by_name[pair['reference_route']]['logits'][head]['values']], np.float32)
            right = np.array([by_name[pair['candidate_route']]['logits'][head]['values']], np.float32)
            assert pair['per_head'][head] == m.first_parity_compare(left, right, tolerance=TOLERANCES['saved_FINAL_logits'])
    assert progress['routes'] == report['routes'] and progress['pairwise'] == report['pairwise']
    assert progress['phase'] == 'COMPLETE_FIRST_PARITY_MEASUREMENT'
    assert report['context_bookends_passed'] is report['ambient_grad_mode_restored'] is True
    assert report['model_state_and_local_RNG_bookend_passed'] is True
    assert body._BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False and body._MODEL_FINITE_CHECK_SCOPE.get() is None
    assert len(models) == 1 and models[0].backbone.weight.item() == .25 and models[0].backbone.weight.grad is None
    assert not models[0]._forward_pre_hooks
    assert report['actual_training_cause'] == report['selected_causal_fix'] == 'UNKNOWN'
    json.dumps(report, allow_nan=False)


def test_first_parity_helper_failure_preserves_completed_route_and_restores_context(monkeypatch):
    from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as body
    plan, _, models, calls, _ = first_parity_calculation(monkeypatch, fail_route=1)
    progress = {}; ambient = torch.is_grad_enabled()
    with pytest.raises(ValueError, match='SYNTHETIC_HELPER_ROUTE_FAILURE'):
        m.calculate_first_parity(plan, progress=progress)
    assert len(calls) == 2 and len(progress['routes']) == 1
    assert progress['routes'][0]['name'] == 'CPU_NO_GRAD_EQUAL_TF'
    assert progress['active_route'] == 'CPU_NO_GRAD_UNMERGED'
    assert progress['counts']['NN_forwards'] == 1 and progress['counts']['autograd_calls'] == 0
    assert torch.is_grad_enabled() is ambient
    assert body._BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False and body._MODEL_FINITE_CHECK_SCOPE.get() is None
    assert not models[0]._forward_pre_hooks and models[0].backbone.weight.grad is None
    json.dumps(progress, allow_nan=False)


@pytest.mark.parametrize('corrupt', ['equal_scope', 'finite_scope', 'state', 'RNG', 'grad'])
def test_first_parity_rejects_context_state_RNG_or_grad_mutation(monkeypatch, corrupt):
    plan, _, _, _, tokens = first_parity_calculation(monkeypatch, corrupt=corrupt)
    try:
        with pytest.raises(ValueError, match='CONTEXT_RESTORE|MODEL_STATE_MUTATION|LOCAL_RNG_MUTATION|MODE_GRAD_CUDA'):
            m.calculate_first_parity(plan, progress={})
    finally:
        for variable, token in reversed(tokens): variable.reset(token)


def test_first_parity_inspection_keeps_old_gradient_consumed_and_all_training_flags_closed(monkeypatch, tmp_path):
    planfile, policyfile = tmp_path / 'FIRST_PARITY_PLAN.json', tmp_path / 'POLICY.json'
    monkeypatch.setattr(m, 'PARITY_PLAN', planfile); monkeypatch.setattr(m, 'POLICY', policyfile)
    monkeypatch.setattr(m, 'PARITY_ROOT', tmp_path / 'PARITY_OUT'); monkeypatch.setattr(m, 'REPO', Path.cwd())
    old_binding = {'path': '/synthetic/CONSUMED_GRADIENT_PLAN.json', 'sha256': 'c'*64}
    plan = {'source_base_commit': '1'*40, 'inputs': {'failed_gradient_plan': old_binding}}
    planfile.write_text(json.dumps(plan)); binding = m.ref(planfile)
    policy = dict.fromkeys(m.saved.FLAGS, False)
    policy[m.STAGE] = {'plan': old_binding, 'execution_authorized': False, 'remaining_invocations': 0,
        'relaunch_allowed': False, 'training_authorized': False, 'test_authorized': False}
    policy[m.PARITY_STAGE] = m.gate_first_parity_expected(binding, False)
    monkeypatch.setattr(m, 'build_first_parity_plan', lambda **k: plan)
    monkeypatch.setattr(m, 'source_clean', lambda value: True)
    def inspect():
        policyfile.write_text(json.dumps(policy)); return m.inspect_first_parity(binding['sha256'], admitted=False)
    assert inspect()[0] == plan and not m.PARITY_ROOT.exists()
    for key in m.saved.FLAGS:
        policy[key] = True
        with pytest.raises(ValueError, match='ONE_USE_OLD_GRADIENT_CLOSED'): inspect()
        policy[key] = False
    for key in ('execution_authorized', 'training_authorized', 'test_authorized', 'relaunch_allowed'):
        policy[m.STAGE][key] = True
        with pytest.raises(ValueError, match='ONE_USE_OLD_GRADIENT_CLOSED'): inspect()
        policy[m.STAGE][key] = False
    policy[m.STAGE]['remaining_invocations'] = 1
    with pytest.raises(ValueError, match='ONE_USE_OLD_GRADIENT_CLOSED'): inspect()
    policy[m.STAGE]['remaining_invocations'] = 0
    policy[m.PARITY_STAGE]['training_authorized'] = True
    with pytest.raises(ValueError, match='ONE_USE_OLD_GRADIENT_CLOSED'): inspect()
    policy[m.PARITY_STAGE]['training_authorized'] = False; m.PARITY_ROOT.mkdir()
    with pytest.raises(ValueError, match='FRESH_ROOT'): inspect()
