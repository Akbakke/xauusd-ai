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


def test_batch_geometry_addition_preserves_old_authorities_and_first_parity_parser_tail():
    import subprocess
    anchor = '1eae0a9710da279a9a8b81a7cc1147ec86126083'
    before = subprocess.check_output(['git', 'show', anchor + ':' + m.SELF], cwd=m.REPO, text=True)
    after = cpu_conditional_strip_source(Path(m.__file__).read_text())
    old_tree, new_tree = ast.parse(before), ast.parse(after)
    old_functions = {node.name: node for node in old_tree.body if isinstance(node, ast.FunctionDef)}
    new_functions = {node.name: node for node in new_tree.body if isinstance(node, ast.FunctionDef)}
    for name, node in old_functions.items():
        if name != 'main_first_parity':
            assert ast.get_source_segment(before, node) == ast.get_source_segment(after, new_functions[name])
    assignment_nodes = lambda tree: [node for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))]
    for node in assignment_nodes(old_tree):
        assert ast.dump(node, include_attributes=False) in [ast.dump(current, include_attributes=False)
                                                          for current in assignment_nodes(new_tree)]
    old_main, current_main = old_functions['main_first_parity'], new_functions['main_first_parity']
    dispatch = current_main.body[0]
    assert isinstance(dispatch, ast.If)
    expected_guard = ast.parse("sys.argv[3:4] == ['--batch-geometry']", mode='eval').body
    assert ast.dump(dispatch.test, include_attributes=False) == ast.dump(expected_guard, include_attributes=False)
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == 'main_batch_geometry' for node in ast.walk(dispatch))
    assert ast.dump(ast.Module(body=current_main.body[1:], type_ignores=[]), include_attributes=False) == \
           ast.dump(ast.Module(body=old_main.body, type_ignores=[]), include_attributes=False)
    old_test = subprocess.check_output(['git', 'show', anchor + ':' + m.TEST], cwd=m.REPO)
    assert Path(__file__).read_bytes().startswith(old_test)


def batch_geometry_lifecycle(monkeypatch, tmp_path):
    plan, _, pref, hooks, timers = first_parity_lifecycle(monkeypatch, tmp_path)
    monkeypatch.setattr(m, 'BATCH_ROOT', tmp_path / 'BATCH_OUT')
    monkeypatch.setattr(m, 'BATCH_PLAN', tmp_path / 'BATCH_PLAN.json')
    m.BATCH_PLAN.write_text('{"synthetic_batch_geometry":true}')
    binding = m.ref(m.BATCH_PLAN)
    monkeypatch.setattr(m, 'inspect_batch_geometry', lambda *a, **k: (plan, binding, pref))
    report = {'finite_parity_measurement': True, 'parity_pass': False}
    monkeypatch.setattr(m, 'calculate_batch_geometry', lambda *a, **k: report)
    return plan, binding, pref, hooks, timers, report


def test_batch_geometry_exact_opt_in_preflight_has_no_payload_or_forward(monkeypatch, tmp_path, capsys):
    _, binding, _, hooks, _, _ = batch_geometry_lifecycle(monkeypatch, tmp_path)
    def forbidden(*a, **k): raise AssertionError('NO_BATCH_GEOMETRY_PAYLOAD_IN_PREFLIGHT')
    monkeypatch.setattr(m, 'calculate_batch_geometry', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden); monkeypatch.setattr(np, 'load', forbidden)
    monkeypatch.setattr(sys, 'argv', ['audit', '--first-parity-plan-sha256', binding['sha256'], '--batch-geometry', '--preflight'])
    m.main(); report = json.loads(capsys.readouterr().out)
    assert report['plan'] == binding and report['claim_created'] is False
    assert all(report[key] == 0 for key in ('NN_forwards', 'autograd_calls', 'checkpoint_deserializations', 'NPZ_arrays_decoded', 'models'))
    assert report['CUDA_initialized'] is False and hooks == []
    assert not any(root.exists() for root in (m.BATCH_ROOT, m.PARITY_ROOT, m.ROOT))
    monkeypatch.setattr(sys, 'argv', sys.argv + ['--batch-geometry'])
    with pytest.raises(ValueError, match='EXACT_CLI'): m.main()
    monkeypatch.setattr(sys, 'argv', ['audit', '--first-parity-plan-sha256', binding['sha256'], '--preflight', '--batch-geometry'])
    with pytest.raises(SystemExit): m.main()


@pytest.mark.parametrize('failure', [False, True])
def test_batch_geometry_measurement_or_partial_failure_consumes_only_new_root(monkeypatch, tmp_path, failure):
    _, binding, pref, _, timers, report = batch_geometry_lifecycle(monkeypatch, tmp_path)
    def calculate(plan, *, progress):
        progress.update(phase='B256_REFERENCE_AUTHENTICATED', first256_reference_verified=True,
                        counts={'NN_forwards': 0, 'autograd_calls': 0})
        if failure: raise ValueError('SYNTHETIC_B256_FORWARD_FAILURE')
        return report
    monkeypatch.setattr(m, 'calculate_batch_geometry', calculate)
    if failure:
        with pytest.raises(ValueError, match='SYNTHETIC_B256_FORWARD_FAILURE'):
            m.run_batch_geometry(binding['sha256'])
    else:
        outputs = m.run_batch_geometry(binding['sha256'])
        assert outputs['result'] == m.ref(m.BATCH_ROOT / 'RESULT.json')
        result = json.loads((m.BATCH_ROOT / 'RESULT.json').read_text())
        assert result['report'] == report and result['test_data_used'] is result['economic_edge_proven'] is False
    terminal = json.loads((m.BATCH_ROOT / 'TERMINAL.json').read_text())
    claim = json.loads((m.BATCH_ROOT / 'CLAIM.json').read_text())
    assert terminal['status'] == ('FAILED_CONSUMED' if failure else 'COMPLETE')
    assert terminal['stage'] == claim['stage'] == m.BATCH_STAGE
    assert terminal['exit_code'] == int(failure) and terminal['remaining_invocations'] == 0
    assert terminal['relaunch_allowed'] is claim['training_authorized'] is False
    assert claim['policy_at_claim'] == pref and terminal['plan'] == binding
    assert (m.BATCH_ROOT / 'RESULT.json').exists() is (not failure)
    assert not m.PARITY_ROOT.exists() and not m.ROOT.exists()
    if failure:
        assert terminal['progress']['phase'] == 'B256_REFERENCE_AUTHENTICATED'
        assert terminal['progress']['first256_reference_verified'] is True
        assert terminal['progress']['counts']['NN_forwards'] == terminal['progress']['counts']['autograd_calls'] == 0
    assert timers[-1] == (m.signal.ITIMER_REAL, 0)
    with pytest.raises(FileExistsError): m.run_batch_geometry(binding['sha256'])


@pytest.mark.parametrize('admitted', [False, True])
def test_batch_geometry_one_use_gate_never_authorizes_gradients_training_or_relaunch(admitted):
    binding = {'path': '/synthetic/B256_PLAN.json', 'sha256': 'd'*64}
    gate = m.gate_batch_geometry_expected(binding, admitted)
    assert gate['plan'] == binding and gate['execution_authorized'] is admitted
    assert gate['remaining_invocations'] == int(admitted)
    assert gate['relaunch_allowed'] is gate['training_authorized'] is gate['test_authorized'] is False


def batch_geometry_counts():
    counts = first_parity_counts()
    counts.update(NN_forwards=1, parity_forwards=1, grad_enabled_forward_calls=0, no_grad_forward_calls=1)
    return counts


def test_batch_geometry_comparison_retains_all_256_rows_and_original_zero_reference_rules():
    reference = np.ones((256, 5), np.float32); reference[0, :2] = 0.
    candidate = reference.copy(); candidate[0, 1] = 1.; candidate[95, 2] += 2e-5; candidate[255, 4] += 1e-7
    result = m.batch_geometry_compare(reference, candidate, tolerance=TOLERANCES['saved_FINAL_logits'])
    delta = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    assert result['rows'] == len(result['per_row']) == 256
    assert result['max_absolute_delta'] == float(delta.max())
    assert result['per_TF_max_absolute_delta'] == delta.max(axis=0).tolist()
    assert result['passed_cells'] == 1278 and result['failed_cells'] == 2
    assert result['all_within_original_tolerance'] is False
    for row in (0, 95, 96, 255):
        assert result['per_row'][row] == m.first_parity_compare(
            reference[row:row+1], candidate[row:row+1], tolerance=TOLERANCES['saved_FINAL_logits'])
    assert result['per_row'][0]['zero_reference_relative_status'] == ['ZERO_OVER_ZERO', 'INFINITE', None, None, None]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('bad', ['shape', 'dtype', 'nan', 'tolerance'])
def test_batch_geometry_comparison_rejects_invalid_domain_or_changed_tolerance(bad):
    reference, candidate = np.zeros((256, 5), np.float32), np.zeros((256, 5), np.float32)
    tolerance = copy.deepcopy(TOLERANCES['saved_FINAL_logits'])
    if bad == 'shape': candidate = candidate[:255]
    elif bad == 'dtype': reference = reference.astype(np.float64)
    elif bad == 'nan': candidate[255, 4] = np.nan
    else: tolerance['rtol'] = .1
    with pytest.raises(ValueError): m.batch_geometry_compare(reference, candidate, tolerance=tolerance)


def test_batch_geometry_count_proof_requires_one_no_grad_forward_and_zero_other_actions():
    assert m.batch_geometry_validate_counts(batch_geometry_counts(), {'NN_forwards_max': 1, 'autograd_calls_max': 0}) is True
    for field in ('NN_forwards', 'grad_enabled_forward_calls', 'autograd_calls', 'canonical_loss_calls',
                  'hidden_capture_calls', 'optimizer_updates'):
        counts = batch_geometry_counts(); counts[field] += 1
        with pytest.raises(ValueError):
            m.batch_geometry_validate_counts(counts, {'NN_forwards_max': 1, 'autograd_calls_max': 0})
    counts = batch_geometry_counts(); counts['checkpoint_deserializations'] = True
    with pytest.raises(ValueError):
        m.batch_geometry_validate_counts(counts, {'NN_forwards_max': 1, 'autograd_calls_max': 0})


def batch_geometry_calculation(monkeypatch, *, corrupt=None):
    from types import SimpleNamespace
    import weakref
    from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as body
    class Context(dict): pass
    class BatchModel(SyntheticModel):
        def forward(self, **features):
            return super().forward(x=features['seq_x'][:, 0, :])
    n = 256
    issues = np.arange(n, dtype=np.int64)*300_000_000_000 + int(m.train.START)
    rows = np.arange(n, dtype=np.int64)
    x = np.linspace(.1, .3, n, dtype=np.float32).reshape(n, 1)
    features = {'seq_x': torch.from_numpy(np.repeat(x[:, None, :], 96, axis=1)),
                'snap_x': torch.from_numpy(x.copy()), 'ctx_cont': torch.from_numpy(x.copy()),
                'ctx_cat': torch.zeros((n, 1), dtype=torch.int64)}
    windows = {'m5': 16, 'm15': 64, 'h1': 96, 'h4': 96, 'd1': 252}
    features.update({'seq_'+tf: torch.from_numpy(np.repeat(x[:, None, :], window, axis=1))
                     for tf, window in windows.items()})
    truth = {'issue_ns': issues, 'activity': np.zeros((n, 5), np.float32),
             'side_up': np.zeros((n, 5), np.float32), 'activity_known': np.ones((n, 5), bool),
             'side_known': np.ones((n, 5), bool)}
    packet = {'features': features, 'truth': truth}
    capture = {'issue_ns': issues.copy(), 'activity_logits': np.repeat(x*.25, 5, axis=1),
               'side_logits': np.repeat(-2*(x*.25), 5, axis=1)}
    def array_record(array):
        return {'values': array[0].astype(np.float64).tolist(), 'shape': list(array.shape),
                'dtype': array.dtype.str, 'sha256': m.core.digest(array)}
    row0_proof = {key: {'shape': list(value[:1].shape), 'dtype': str(value.dtype),
        'sha256': m.core.digest(value[:1].numpy())} for key, value in features.items()}
    activation = {'batch_eval_active': True, 'model_training': False, 'grad_enabled': False,
                  'scope_enabled': True, 'eligible_equal_length_TF_groups': {
                      '16': ['M5'], '64': ['M15'], '96': ['H1', 'H4'], '252': ['D1']}}
    route = {'name': 'CPU_NO_GRAD_EQUAL_TF', 'requested': copy.deepcopy(FIRST_PARITY_ROUTES[0]),
             'activation': copy.deepcopy(activation),
             'logits': {head: array_record(capture[head][:1]) for head in ('activity_logits', 'side_logits')}}
    state = {'backbone.weight': torch.tensor(.25, dtype=torch.float32)}
    plan = {'metadata': {'adapter_metadata': {}, 'original_epoch_plan': {'metadata': {'architecture': {}}},
                        'final_receipt': {'proof': {'model_state': m.core.state_sha(state)}}},
            'inputs': {'capture': {'path': '/synthetic/B256_CAPTURE.npz', 'sha256': 'a'*64}},
            'panel': {'rows': 256, 'original_first96_issue_ns_sha256': m.core.digest(issues[:96]),
                      'original_first96_source_rows_sha256': m.core.digest(rows[:96])},
            'TRAIN_issue_ns_sha256': m.core.digest(issues), 'capture_array_headers': {},
            'budget': {'packet_feature_bytes': sum(t.numel()*t.element_size() for t in features.values()),
                       'capture_ndarray_bytes_decoded': sum(a.nbytes for a in capture.values()),
                       'NN_forwards_max': 1, 'autograd_calls_max': 0},
            'fixed_original_tolerance': copy.deepcopy(TOLERANCES['saved_FINAL_logits']),
            'prior_CPU_B1': {'route': route, 'selected_issue_ns': int(issues[0]),
                             'selected_source_row': int(rows[0]), 'feature_proof': row0_proof},
            'original_score_batch': {'cursor': 1, 'rows': 256, 'rows_done': 256,
                 'known_activity_cells': 1280, 'activation': copy.deepcopy(activation),
                 'output_sha256': {head: m.core.digest(capture[head]) for head in ('activity_logits', 'side_logits')}},
            'cache_scope': {'synthetic': True}, 'limits': [],
            'route': {'name': 'CPU_B256_NO_GRAD_EQUAL_TF', 'device': 'cpu', 'batch_rows': 256,
                      'model_training': False, 'grad_enabled': False, 'grouped': True, 'equal_tf': True, 'capture': False}}
    released, models, calls = [], [], []
    def load_context(_):
        context = Context(provider=SimpleNamespace(bar_start_ns=np.array([int(m.train.START)], np.int64)),
                          source_rows=rows, normalization={}, prior={}, input_binding={}, target_contract={})
        released.append(weakref.ref(context)); return context
    def factory(**kwargs):
        assert released[0]() is None
        assert kwargs['device'] == torch.device('cpu') and kwargs['backbone_gradient_policy'] == 'M5_primary'
        model = BatchModel(); models.append(model); return model
    def helper(model, bound_features, *, grouped, equal_tf, capture):
        assert grouped is equal_tf is True and capture is False
        assert torch.is_grad_enabled() is False and model.training is False
        assert all(value.shape[0] == 256 for value in bound_features.values())
        assert len(model._forward_pre_hooks) == 1
        assert all(not child._forward_hooks and not child._forward_pre_hooks
                   for child in model.modules() if child is not model)
        calls.append({'features': bound_features, 'grouped': grouped, 'equal_tf': equal_tf, 'capture': capture})
        with body.model_finite_check_scope('grouped'), body.batch_equal_length_mtf_eval_scope():
            output = {head: value + .01 for head, value in model(**bound_features).items()}
        if corrupt == 'helper': raise ValueError('SYNTHETIC_B256_HELPER_EXCEPTION')
        if corrupt == 'packet': bound_features['seq_x'][0, 0, 0] += 1.
        return output, None, copy.deepcopy(activation)
    def forbidden(*a, **k): raise AssertionError('NO_B256_DERIVATIVE_OR_LOSS')
    monkeypatch.setattr(m.adapter, 'load_context', load_context)
    monkeypatch.setattr(m.adapter, 'packet', lambda context, take: packet)
    monkeypatch.setattr(m.saved, 'load_fields', lambda binding, fields, headers: {key: capture[key] for key in fields})
    monkeypatch.setattr(m.model_owner, 'build_native_future_state_model', factory)
    monkeypatch.setattr(m, 'safe_load_final', lambda plan: {'model_state': state})
    monkeypatch.setattr(m.core, 'future_cuda_forward', helper)
    monkeypatch.setattr(torch.autograd, 'grad', forbidden)
    monkeypatch.setattr(m.objective, 'native_future_state_loss', forbidden)
    monkeypatch.setattr(torch, 'set_num_threads', lambda count: None)
    monkeypatch.setattr(torch, 'get_num_threads', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    if corrupt == 'capture': capture['side_logits'][128, 4] += .1
    elif corrupt == 'prior_logit': plan['prior_CPU_B1']['route']['logits']['side_logits']['sha256'] = 'f'*64
    elif corrupt == 'first96_clock': plan['panel']['original_first96_issue_ns_sha256'] = 'f'*64
    elif corrupt == 'row0_feature': plan['prior_CPU_B1']['feature_proof']['seq_x']['sha256'] = 'f'*64
    return plan, capture, packet, models, calls


def test_batch_geometry_authenticates_one_full256_packet_and_records_all_finite_mismatches(monkeypatch):
    from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as body
    plan, capture, packet, models, calls = batch_geometry_calculation(monkeypatch)
    progress = {}; ambient = torch.is_grad_enabled()
    report = m.calculate_batch_geometry(plan, progress=progress)
    assert len(models) == len(calls) == 1
    assert report['counts'] == batch_geometry_counts()
    assert report['all_256_within_saved_original_tolerance'] is False
    assert report['measurement_completed'] is report['stop_parity_series_after_this_comparison'] is True
    assert report['full96_gradient_qualification'] is False
    assert report['reference_capture256_SHA_authenticated_before_forward'] is True
    assert progress['reference_capture256_SHA_authenticated'] is True
    assert report['actual_activation'] == plan['original_score_batch']['activation']
    assert report['input_identity']['first256_issue_ns_sha256'] == m.core.digest(packet['truth']['issue_ns'])
    assert report['input_identity']['first256_source_rows_sha256'] == m.core.digest(np.arange(256, dtype=np.int64))
    assert report['input_identity']['packet_sha256'] == m.core.state_sha(packet)
    assert report['row0_feature_proof'] == plan['prior_CPU_B1']['feature_proof']
    for head in ('activity_logits', 'side_logits'):
        actual = capture[head] + np.float32(.01)
        assert report['outputs'][head] == {'values': actual.astype(np.float64).tolist(), 'shape': [256, 5],
            'dtype': np.dtype('float32').str, 'sha256': m.core.digest(actual)}
        assert report['saved_GPU_reference'][head] == {'values': capture[head].astype(np.float64).tolist(), 'shape': [256, 5],
            'dtype': np.dtype('float32').str, 'sha256': m.core.digest(capture[head])}
        assert report['vs_saved_GPU'][head] == m.batch_geometry_compare(capture[head], actual, tolerance=TOLERANCES['saved_FINAL_logits'])
        assert report['vs_saved_GPU'][head]['all_within_original_tolerance'] is False
        assert report['row0_vs_prior_CPU_B1'][head] == m.first_parity_compare(
            capture[head][:1], actual[:1], tolerance=TOLERANCES['saved_FINAL_logits'])
        assert len(report['vs_saved_GPU'][head]['per_row']) == 256
    assert torch.is_grad_enabled() is ambient
    assert body._BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False and body._MODEL_FINITE_CHECK_SCOPE.get() is None
    assert models[0].backbone.weight.grad is None and models[0].backbone.weight.item() == .25
    assert not models[0]._forward_pre_hooks
    assert report['actual_training_cause'] == report['selected_causal_fix'] == 'UNKNOWN'
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('corrupt', ['capture', 'first96_clock', 'row0_feature'])
def test_batch_geometry_rejects_reference_or_input_identity_before_any_forward(monkeypatch, corrupt):
    plan, _, _, models, calls = batch_geometry_calculation(monkeypatch, corrupt=corrupt)
    progress = {}
    with pytest.raises(ValueError): m.calculate_batch_geometry(plan, progress=progress)
    assert calls == models == [] and progress['counts']['NN_forwards'] == 0
    assert progress['counts']['autograd_calls'] == 0
    assert progress['counts']['model_constructions'] == progress['counts']['checkpoint_deserializations'] == 0


@pytest.mark.parametrize('corrupt', ['helper', 'packet', 'prior_logit'])
def test_batch_geometry_failure_preserves_pre_forward_proofs_and_removes_counter(monkeypatch, corrupt):
    from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as body
    plan, _, _, models, calls = batch_geometry_calculation(monkeypatch, corrupt=corrupt)
    progress = {}; ambient = torch.is_grad_enabled()
    with pytest.raises(ValueError): m.calculate_batch_geometry(plan, progress=progress)
    assert len(calls) == len(models) == 1
    assert progress['counts']['NN_forwards'] == 1 and progress['counts']['autograd_calls'] == 0
    assert progress['reference_capture256_SHA_authenticated'] is True
    assert len(progress['input_identity']['first256_issue_ns_sha256']) == 64
    assert len(progress['saved_GPU_reference']['activity_logits']['values']) == 256
    assert not models[0]._forward_pre_hooks and models[0].backbone.weight.grad is None
    assert torch.is_grad_enabled() is ambient
    assert body._BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False and body._MODEL_FINITE_CHECK_SCOPE.get() is None
    json.dumps(progress, allow_nan=False)


def test_batch_geometry_inspection_rejects_cross_route_SHA_and_reopening_consumed_plans(monkeypatch, tmp_path):
    planfile, policyfile, oldfile = tmp_path / 'B256_PLAN.json', tmp_path / 'POLICY.json', tmp_path / 'OLD_PARITY_PLAN.json'
    monkeypatch.setattr(m, 'BATCH_PLAN', planfile); monkeypatch.setattr(m, 'POLICY', policyfile)
    monkeypatch.setattr(m, 'PARITY_PLAN', oldfile)
    monkeypatch.setattr(m, 'BATCH_ROOT', tmp_path / 'BATCH_OUT'); monkeypatch.setattr(m, 'REPO', Path.cwd())
    gradient_binding = {'path': '/synthetic/CONSUMED_GRADIENT.json', 'sha256': 'c'*64}
    oldfile.write_text('{"consumed_first_parity":true}'); first_binding = m.ref(oldfile)
    plan = {'source_base_commit': '1'*40, 'inputs': {'failed_gradient_plan': gradient_binding,
                                                   'first_parity_plan': first_binding}}
    planfile.write_text(json.dumps(plan)); binding = m.ref(planfile)
    policy = dict.fromkeys(m.saved.FLAGS, False)
    for stage, old_binding in ((m.STAGE, gradient_binding), (m.PARITY_STAGE, first_binding)):
        policy[stage] = {'plan': old_binding, 'execution_authorized': False, 'remaining_invocations': 0,
                        'relaunch_allowed': False, 'training_authorized': False, 'test_authorized': False}
    policy[m.BATCH_STAGE] = m.gate_batch_geometry_expected(binding, False)
    monkeypatch.setattr(m, 'build_batch_geometry_plan', lambda **k: plan)
    monkeypatch.setattr(m, 'source_clean', lambda value: True)
    def forbidden(*a, **k): raise AssertionError('OLD_BUILDER_MUST_NOT_BE_REACHED_WITH_NEW_SHA')
    monkeypatch.setattr(m, 'build_first_parity_plan', forbidden)
    def inspect():
        policyfile.write_text(json.dumps(policy)); return m.inspect_batch_geometry(binding['sha256'], admitted=False)
    assert inspect()[0] == plan and not m.BATCH_ROOT.exists()
    with pytest.raises(ValueError): m.inspect_batch_geometry(first_binding['sha256'], admitted=False)
    with pytest.raises(ValueError): m.inspect_first_parity(binding['sha256'], admitted=False)
    for key in m.saved.FLAGS:
        policy[key] = True
        with pytest.raises(ValueError, match='ONE_USE_FLAGS'): inspect()
        policy[key] = False
    for stage in (m.STAGE, m.PARITY_STAGE):
        policy[stage]['remaining_invocations'] = 1
        with pytest.raises(ValueError, match='OLD_PLANS_CLOSED'): inspect()
        policy[stage]['remaining_invocations'] = 0
        for key in ('execution_authorized', 'training_authorized', 'test_authorized', 'relaunch_allowed'):
            policy[stage][key] = True
            with pytest.raises(ValueError, match='OLD_PLANS_CLOSED'): inspect()
            policy[stage][key] = False
    policy[m.BATCH_STAGE]['training_authorized'] = True
    with pytest.raises(ValueError, match='ONE_USE_FLAGS'): inspect()
    policy[m.BATCH_STAGE]['training_authorized'] = False; m.BATCH_ROOT.mkdir()
    with pytest.raises(ValueError, match='FRESH_ROOT'): inspect()


def test_batch_geometry_builder_binds_descriptive_pointer_separately_from_actual_score_fields():
    tree = ast.parse(Path(m.__file__).read_text())
    builder = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == 'build_batch_geometry_plan')
    statements = [node for node in ast.walk(builder) if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
        and node.value.func.id == 'need' and len(node.value.args) == 2
        and isinstance(node.value.args[1], ast.Constant)
        and node.value.args[1].value == 'BATCH_GEOMETRY_ORIGINAL_SCORE_BATCH']
    assert len(statements) == 1
    code = compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(statements[0])], type_ignores=[])),
                   '<actual_B256_original_score_batch_gate>', 'exec')
    # Actual saved GPU batch shape, including partial_NLL but no JSON_pointer.
    first = {'activation': {'batch_eval_active': True, 'eligible_equal_length_TF_groups': {
                '16': ['M5'], '252': ['D1'], '64': ['M15'], '96': ['H1', 'H4']},
                'grad_enabled': False, 'model_training': False, 'scope_enabled': True},
        'cursor': 1, 'known_activity_cells': 1280,
        'output_sha256': {'activity_logits': '5746312468812960e402dd008a3788b8334d36fdda968a7777c7b5be6a6d2c67',
                          'side_logits': '38c62b343741395250d198cb353973e3d4851e06da21cee1d4c4df1f6d3d285e'},
        'partial_NLL': 0.5790096521377563, 'rows': 256, 'rows_done': 256}
    pointer = '/report/whole_FINAL_score/batches/0'
    d = {'original_score_batch': {'JSON_pointer': pointer,
        **{key: copy.deepcopy(first[key]) for key in ('activation', 'cursor', 'known_activity_cells',
                                                      'output_sha256', 'rows', 'rows_done')}}}
    bindings = {key: {'path': '/synthetic/' + key + '.json', 'sha256': 'a'*64}
                for key in ('completed_model_plan', 'capture', 'final_receipt')}
    whole_output_sha = {'activity_logits': 'b'*64, 'side_logits': 'c'*64}
    base = {'inputs': bindings, 'metadata': {'final_receipt': {'proof': {'model_state': 'd'*64}}},
            'capture_array_headers': {key: {'sha256': value} for key, value in whole_output_sha.items()}}
    score = {'batch_rows': 256, 'state_sha256_before_and_after': 'd'*64,
             'output_sha256': whole_output_sha, 'batches': [first]}
    original = {'plan': copy.deepcopy(bindings['completed_model_plan']),
        'report': {'final_logits': copy.deepcopy(bindings['capture']),
                   'final_checkpoint_receipt': copy.deepcopy(bindings['final_receipt']),
                   'whole_FINAL_score': score}}
    environment = {'need': m.need, 'CAPTURE': m.CAPTURE, 'd': d, 'first': first,
                   'score': score, 'base': base, 'original': original}
    exec(code, environment)
    d['original_score_batch']['JSON_pointer'] = '/report/whole_FINAL_score/batches/1'
    with pytest.raises(ValueError, match='BATCH_GEOMETRY_ORIGINAL_SCORE_BATCH'): exec(code, environment)
    d['original_score_batch']['JSON_pointer'] = pointer
    first['output_sha256']['activity_logits'] = '0'*64
    with pytest.raises(ValueError, match='BATCH_GEOMETRY_ORIGINAL_SCORE_BATCH'): exec(code, environment)


# CPU-conditional gradient tests; all other earlier cases retain their source.
def cpu_conditional_strip_source(source):
    begin = '# CPU-conditional B96 gradients; saved-CUDA gate remains failed.\n'
    end = '# End CPU-conditional B96 gradients.\n\n'
    dispatch = "    if sys.argv[3:4]==['--cpu-conditional-gradient']:\n        return main_cpu_conditional()\n\n"
    assert source.count(begin) == source.count(end) == 1
    first, tail = source.split(begin)
    _, last = tail.split(end)
    assert last.count(dispatch) == 1
    return first + last.replace(dispatch, '', 1)


def test_cpu_conditional_source_addition_strips_to_exact_frozen_prior_owner():
    import subprocess
    before = subprocess.check_output(['git', 'show', 'f9c825571a0ba291364527f1754818fe8e0ab6a7:' + m.SELF], cwd=m.REPO, text=True)
    after = Path(m.__file__).read_text()
    assert cpu_conditional_strip_source(after) == before


def cpu_conditional_lifecycle(monkeypatch, tmp_path):
    plan, _, pref, hooks, timers, _ = batch_geometry_lifecycle(monkeypatch, tmp_path)
    monkeypatch.setattr(m, 'CPU_ROOT', tmp_path / 'CPU_OUT')
    monkeypatch.setattr(m, 'CPU_PLAN', tmp_path / 'CPU_PLAN.json')
    m.CPU_PLAN.write_text('{"synthetic_cpu_conditional":true}')
    binding = m.ref(m.CPU_PLAN)
    monkeypatch.setattr(m, 'inspect_cpu_conditional', lambda *a, **k: (plan, binding, pref))
    monkeypatch.setattr(m, 'calculate_cpu_conditional', lambda *a, **k: {'original_saved_CUDA_gate': 'FAILED'})
    return plan, binding, pref, hooks, timers


def test_cpu_conditional_exact_opt_in_preflight_has_zero_derivatives_and_no_baseline(monkeypatch, tmp_path, capsys):
    _, binding, _, hooks, _ = cpu_conditional_lifecycle(monkeypatch, tmp_path)
    def forbidden(*a, **k): raise AssertionError('NO_CPU_CONDITIONAL_PAYLOAD_IN_PREFLIGHT')
    monkeypatch.setattr(m, 'calculate_cpu_conditional', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden); monkeypatch.setattr(np, 'load', forbidden)
    monkeypatch.setattr(sys, 'argv', ['audit', '--first-parity-plan-sha256', binding['sha256'], '--cpu-conditional-gradient', '--preflight'])
    m.main(); result = json.loads(capsys.readouterr().out)
    assert result['plan'] == binding and result['claim_created'] is False
    assert all(result[key] == 0 for key in ('NN_forwards', 'autograd_calls', 'checkpoint_deserializations', 'NPZ_arrays_decoded', 'models'))
    assert result['CUDA_initialized'] is False and hooks == []
    assert not any(root.exists() for root in (m.CPU_ROOT, m.BATCH_ROOT, m.PARITY_ROOT, m.ROOT))
    monkeypatch.setattr(sys, 'argv', sys.argv + ['--cpu-conditional-gradient'])
    with pytest.raises(ValueError, match='EXACT_CLI'): m.main()
    monkeypatch.setattr(sys, 'argv', ['audit', '--first-parity-plan-sha256', binding['sha256'], '--preflight', '--cpu-conditional-gradient'])
    with pytest.raises(SystemExit): m.main()


@pytest.mark.parametrize('failure', [False, True])
def test_cpu_conditional_lifecycle_retains_baseline_and_failed_GPU_gate_without_relaunch(monkeypatch, tmp_path, failure):
    _, binding, pref, _, timers = cpu_conditional_lifecycle(monkeypatch, tmp_path)
    def calculate(plan, *, progress):
        baseline = m.cpu_conditional_publish('CPU_BASELINE.json', {
            'status': 'CPU_BASELINE_COMPLETE_BEFORE_DERIVATIVES', 'counts': {'NN_forwards': 96, 'autograd_calls': 0}})
        progress.update(cpu_baseline=baseline, second_pass_rows=[{'index': 3, 'paired_gate': False}],
                        counts={'NN_forwards': 100, 'autograd_calls': 12})
        if failure: raise ValueError('SYNTHETIC_LATE_CPU_PAIR_FAILURE')
        return {'cpu_baseline': baseline, 'original_saved_CUDA_gate': 'FAILED',
                'full96_gradient_qualification': False, 'actual_training_cause': 'UNKNOWN'}
    monkeypatch.setattr(m, 'calculate_cpu_conditional', calculate)
    if failure:
        with pytest.raises(ValueError, match='SYNTHETIC_LATE_CPU_PAIR_FAILURE'):
            m.run_cpu_conditional(binding['sha256'])
    else:
        outputs = m.run_cpu_conditional(binding['sha256'])
        assert outputs['result'] == m.ref(m.CPU_ROOT / 'RESULT.json')
        result = json.loads((m.CPU_ROOT / 'RESULT.json').read_text())
        assert result['report']['original_saved_CUDA_gate'] == 'FAILED'
        assert result['report']['full96_gradient_qualification'] is False
        assert result['test_data_used'] is result['economic_edge_proven'] is False
    terminal = json.loads((m.CPU_ROOT / 'TERMINAL.json').read_text())
    claim = json.loads((m.CPU_ROOT / 'CLAIM.json').read_text())
    baseline = json.loads((m.CPU_ROOT / 'CPU_BASELINE.json').read_text())
    assert baseline['counts'] == {'NN_forwards': 96, 'autograd_calls': 0}
    assert terminal['status'] == ('FAILED_CONSUMED' if failure else 'COMPLETE')
    assert terminal['stage'] == claim['stage'] == m.CPU_STAGE
    assert terminal['exit_code'] == int(failure) and terminal['remaining_invocations'] == 0
    assert terminal['relaunch_allowed'] is claim['training_authorized'] is False
    assert claim['policy_at_claim'] == pref and terminal['plan'] == binding
    assert (m.CPU_ROOT / 'RESULT.json').exists() is (not failure)
    assert not any(root.exists() for root in (m.BATCH_ROOT, m.PARITY_ROOT, m.ROOT))
    if failure:
        assert terminal['result'] is None
        assert terminal['progress']['cpu_baseline'] == m.ref(m.CPU_ROOT / 'CPU_BASELINE.json')
        assert terminal['progress']['counts'] == {'NN_forwards': 100, 'autograd_calls': 12}
    assert timers[-1] == (m.signal.ITIMER_REAL, 0)
    with pytest.raises(FileExistsError): m.run_cpu_conditional(binding['sha256'])


@pytest.mark.parametrize('admitted', [False, True])
def test_cpu_conditional_one_use_gate_keeps_all_training_and_retry_closed(admitted):
    binding = {'path': '/synthetic/CPU_CONDITIONAL.json', 'sha256': 'e'*64}
    gate = m.gate_cpu_conditional_expected(binding, admitted)
    assert gate['plan'] == binding and gate['execution_authorized'] is admitted
    assert gate['remaining_invocations'] == int(admitted)
    assert gate['relaunch_allowed'] is gate['training_authorized'] is gate['test_authorized'] is False


def test_cpu_conditional_panel_comparison_retains_96_rows_and_fixed_zero_reference_rules():
    reference = np.ones((96, 5), np.float32); reference[0, 0] = 0.
    candidate = reference.copy(); candidate[0, 0] = 1.; candidate[95, 4] += 2e-5
    result = m.cpu_conditional_compare(reference, candidate, tolerance=TOLERANCES['saved_FINAL_logits'])
    delta = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    assert result['rows'] == len(result['per_row']) == 96
    assert result['passed_cells'] == 478 and result['failed_cells'] == 2
    assert result['all_within_original_tolerance'] is False
    assert result['max_absolute_delta'] == float(delta.max())
    assert result['per_TF_max_absolute_delta'] == delta.max(axis=0).tolist()
    assert result['per_row'][0]['zero_reference_relative_status'][0] == 'INFINITE'
    assert result['per_row'][95] == m.first_parity_compare(reference[95:], candidate[95:], tolerance=TOLERANCES['saved_FINAL_logits'])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('bad', ['shape', 'dtype', 'nan', 'tolerance'])
def test_cpu_conditional_panel_comparison_rejects_changed_domain_or_tolerance(bad):
    reference, candidate = np.zeros((96, 5), np.float32), np.zeros((96, 5), np.float32)
    tolerance = copy.deepcopy(TOLERANCES['saved_FINAL_logits'])
    if bad == 'shape': candidate = candidate[:95]
    elif bad == 'dtype': reference = reference.astype(np.float64)
    elif bad == 'nan': candidate[95, 4] = np.nan
    else: tolerance['atol'] = .1
    with pytest.raises(ValueError): m.cpu_conditional_compare(reference, candidate, tolerance=tolerance)


def cpu_conditional_calculation(monkeypatch, tmp_path, *, mismatch_row=None, wrong_anchor=False):
    from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as body
    plan, capture, models = synthetic_calculation(monkeypatch)
    packet = m.adapter.packet(None, np.arange(96, dtype=np.int64))
    for j, tf in enumerate(('m5', 'm15', 'h1', 'h4', 'd1')):
        packet['features']['seq_' + tf] = packet['features'].pop('dummy' + str(j))
    def record(array):
        return {'values': array.astype(np.float64).tolist(), 'shape': list(array.shape),
                'dtype': array.dtype.str, 'sha256': m.core.digest(array)}
    baseline = {head: capture[head].copy() for head in ('activity_logits', 'side_logits')}
    row0 = {key: {'shape': list(value[:1].shape), 'dtype': str(value.dtype),
                 'sha256': m.core.digest(value[:1].numpy())} for key, value in packet['features'].items()}
    anchor = {'name': 'CPU_GRAD_ENABLED_UNMERGED', 'requested': copy.deepcopy(FIRST_PARITY_ROUTES[2]),
              'activation': {'batch_eval_active': False, 'model_training': False, 'grad_enabled': True,
                  'scope_enabled': False, 'eligible_equal_length_TF_groups': {'1': ['M5', 'M15', 'H1', 'H4', 'D1']}},
              'logits': {head: {**record(array[:1]), 'values': array[0].astype(np.float64).tolist()}
                         for head, array in baseline.items()}}
    if wrong_anchor:
        bad = np.array([anchor['logits']['side_logits']['values']], np.float32); bad[0, 4] += .1
        anchor['logits']['side_logits'] = {**record(bad), 'values': bad[0].astype(np.float64).tolist()}
    plan.update(prior_CPU_anchor={'route': anchor, 'selected_issue_ns': int(capture['issue_ns'][0]),
                                 'selected_source_row': 0, 'feature_proof': row0},
                fixed_original_tolerance=copy.deepcopy(TOLERANCES['saved_FINAL_logits']),
                route=copy.deepcopy(m.CPU_ROUTE))
    plan['budget'].update(NN_forwards_max=192, autograd_calls_max=384)
    for head in baseline: capture[head][:] -= np.float32(.01)
    monkeypatch.setattr(m, 'CPU_ROOT', tmp_path / 'CPU_CLAIMED')
    monkeypatch.setattr(m, 'CPU_PLAN', tmp_path / 'CPU_PLAN.json')
    monkeypatch.setattr(m, 'POLICY', tmp_path / 'POLICY.json')
    m.CPU_ROOT.mkdir()
    m.CPU_PLAN.write_text(json.dumps(plan, sort_keys=True, allow_nan=False))
    plan_ref = m.ref(m.CPU_PLAN)
    policy = dict.fromkeys(m.saved.FLAGS, False)
    policy[m.CPU_STAGE] = m.gate_cpu_conditional_expected(plan_ref, True)
    m.POLICY.write_text(json.dumps(policy, sort_keys=True, allow_nan=False))
    policy_ref = m.ref(m.POLICY)
    claim = {'stage': m.CPU_STAGE, 'plan': plan_ref, 'policy_at_claim': policy_ref,
             'remaining_invocations': 0, 'relaunch_allowed': False,
             'training_authorized': False, 'test_data_used': False}
    claim_path = m.CPU_ROOT / 'CLAIM.json'
    claim_path.write_text(json.dumps(claim, sort_keys=True, allow_nan=False))
    execution_refs = {'plan': plan_ref, 'claim': m.ref(claim_path), 'policy_at_claim': policy_ref}
    calls, publications, grad_rows, paired_gates, scales, proof_calls = [], [], [], [], [], []
    baseline_ref = {'path': str(m.CPU_ROOT / 'CPU_BASELINE.json'), 'sha256': 'f'*64}
    original_compare, original_grad = m.first_parity_compare, torch.autograd.grad
    original_scaling, original_report = m.scaled_canonical_loss, m.gradient_report
    def helper(model, features, *, grouped, equal_tf, capture):
        assert grouped is True and equal_tf is capture is False
        assert torch.is_grad_enabled() is True and model.training is False
        assert all(value.shape[0] == 1 for value in features.values())
        assert body._BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False and body._MODEL_FINITE_CHECK_SCOPE.get() is None
        assert len(model._forward_pre_hooks) == 1
        assert all(not child._forward_hooks and not child._forward_pre_hooks
                   for child in model.modules() if child is not model)
        index = len(calls); calls.append(index)
        with body.model_finite_check_scope('grouped'):
            output = model(**features)
        if index >= 96 and index-96 == mismatch_row:
            output['side_logits'] = output['side_logits'] + .02
        return output, None, copy.deepcopy(anchor['activation'])
    def compare(reference, candidate, *, tolerance):
        result = original_compare(reference, candidate, tolerance=tolerance)
        if len(calls) > 96:
            head = 'activity_logits' if candidate[0, 0] > 0 else 'side_logits'
            paired_gates.append((len(calls)-97, head, result['all_within_original_tolerance']))
        return result
    def publish(name, artifact):
        assert name == 'CPU_BASELINE.json' and len(calls) == 96 and grad_rows == []
        assert artifact['status'] == 'CPU_BASELINE_COMPLETE_BEFORE_DERIVATIVES'
        publications.append(copy.deepcopy(artifact)); return baseline_ref.copy()
    def gradient(*args, **kwargs):
        row = len(calls)-97
        assert len(publications) == 1 and 0 <= row < 96
        assert {head for index, head, passed in paired_gates if index == row and passed} == {'activity_logits', 'side_logits'}
        grad_rows.append(row)
        return original_grad(*args, **kwargs)
    def scale(loss, chunk_denominator, full_denominator):
        assert chunk_denominator == 5 and full_denominator == 480
        scales.append((chunk_denominator, full_denominator))
        return original_scaling(loss, chunk_denominator, full_denominator)
    def proof(*vectors, **kwargs):
        assert len(vectors) == 4
        assert all(value is None or value.dtype == torch.float64 for vector in vectors for value in vector)
        proof_calls.append(True); return original_report(*vectors, **kwargs)
    monkeypatch.setattr(m.core, 'future_cuda_forward', helper)
    monkeypatch.setattr(m, 'first_parity_compare', compare)
    monkeypatch.setattr(m, 'cpu_conditional_publish', publish)
    monkeypatch.setattr(torch.autograd, 'grad', gradient)
    monkeypatch.setattr(m, 'scaled_canonical_loss', scale)
    monkeypatch.setattr(m, 'gradient_report', proof)
    trace = {'calls': calls, 'publications': publications, 'grad_rows': grad_rows, 'paired_gates': paired_gates,
             'scales': scales, 'proof_calls': proof_calls, 'baseline_ref': baseline_ref, 'execution_refs': execution_refs}
    return plan, capture, baseline, models, trace


def cpu_conditional_counts():
    counts = completed_counts()
    del counts['parity_forwards']
    counts.update(baseline_forwards=96, gradient_forwards=96, NN_forwards=192, autograd_calls=384,
                  canonical_loss_calls=96, independent_BCE_calls=192, baseline_artifact_writes=1,
                  grad_enabled_forward_calls=192, no_grad_forward_calls=0)
    return counts


def test_cpu_conditional_count_proof_requires_complete_paired_measurement_and_zero_updates():
    assert m.cpu_conditional_validate_counts(cpu_conditional_counts(), {'NN_forwards_max': 192, 'autograd_calls_max': 384}) is True
    for field in ('baseline_forwards', 'gradient_forwards', 'autograd_calls', 'baseline_artifact_writes',
                  'no_grad_forward_calls', 'optimizer_updates'):
        counts = cpu_conditional_counts(); counts[field] += 1
        with pytest.raises(ValueError):
            m.cpu_conditional_validate_counts(counts, {'NN_forwards_max': 192, 'autograd_calls_max': 384})
    counts = cpu_conditional_counts(); counts['checkpoint_deserializations'] = True
    with pytest.raises(ValueError):
        m.cpu_conditional_validate_counts(counts, {'NN_forwards_max': 192, 'autograd_calls_max': 384})


def test_cpu_conditional_baseline_precedes_every_derivative_and_reuses_original_Dfull_and_F64(monkeypatch, tmp_path):
    from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as body
    plan, capture, baseline, models, trace = cpu_conditional_calculation(monkeypatch, tmp_path)
    refs = trace['execution_refs']
    progress = {'execution_refs': refs}; ambient = torch.is_grad_enabled()
    report = m.calculate_cpu_conditional(plan, progress=progress)
    assert report['counts'] == cpu_conditional_counts()
    assert len(trace['calls']) == 192 and trace['grad_rows'] == [row for row in range(96) for _ in range(4)]
    assert trace['scales'] == [(5, 480)]*96 and trace['proof_calls'] == [True]
    assert len(trace['publications']) == 1
    artifact = trace['publications'][0]
    assert {role: artifact[role] for role in refs} == refs
    assert artifact['counts']['NN_forwards'] == artifact['counts']['baseline_forwards'] == 96
    assert artifact['counts']['autograd_calls'] == artifact['autograd_calls_before_publication'] == 0
    assert artifact['bookends_passed'] is True and artifact['original_saved_CUDA_gate'] == 'FAILED'
    assert report['cpu_baseline'] == progress['cpu_baseline'] == trace['baseline_ref']
    assert report['baseline_counts']['NN_forwards'] == 96 and report['baseline_counts']['autograd_calls'] == 0
    assert report['known_activity_denominator_full_B96'] == 480
    assert report['original_saved_CUDA_gate'] == 'FAILED'
    assert report['saved_CUDA_comparison_is_informational'] is True
    assert report['full96_gradient_qualification'] is False
    assert report['CPU_baseline_published_before_derivatives'] is report['CPU_conditional_local_interaction_only'] is True
    for head in ('activity_logits', 'side_logits'):
        expected = {'values': baseline[head].astype(np.float64).tolist(), 'shape': [96, 5],
                    'dtype': np.dtype('float32').str, 'sha256': m.core.digest(baseline[head])}
        assert artifact['outputs'][head] == report['cpu_baseline_outputs'][head] == expected
        assert report['cpu_replay_outputs'][head] == expected
        assert report['cpu_replay_comparison'][head]['all_within_original_tolerance'] is True
        assert report['cpu_replay_comparison'][head]['failed_cells'] == 0
        assert report['saved_CUDA_comparison'][head]['failed_cells'] == 480
        assert artifact['saved_CUDA_comparison'][head] == report['saved_CUDA_comparison'][head]
    assert len(progress['second_pass_rows']) == 96 and progress['second_pass_rows'][-1]['index'] == 95
    assert all(comparison['all_within_original_tolerance']
               for row in progress['second_pass_rows'] for comparison in row['vs_CPU_baseline'].values())
    assert report['gradient_proof']['component_gradient_contract_verified'] is True
    assert report['gradient_proof']['actual_auxiliary_shared_gradient_zero'] is True
    assert report['gradient_proof']['local_verdict'] == 'NOT_LOCAL_CONFLICT'
    assert report['parameter_axis']['aggregation_dtype'] == 'float64'
    assert report['model_state_and_local_RNG_bookend_passed'] is report['retained_B96_packet_unmutated'] is True
    assert body._BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False and body._MODEL_FINITE_CHECK_SCOPE.get() is None
    assert torch.is_grad_enabled() is ambient
    assert len(models) == 1 and models[0].backbone.weight.item() == .25 and models[0].backbone.weight.grad is None
    assert not models[0]._forward_pre_hooks
    assert report['actual_training_cause'] == report['selected_causal_fix'] == report['economic_edge'] == 'UNKNOWN'
    assert progress['phase'] == 'COMPLETE_CPU_CONDITIONAL_GRADIENT_MEASUREMENT'
    json.dumps(report, allow_nan=False)


def test_cpu_conditional_first_row_anchor_failure_blocks_baseline_publication_and_derivatives(monkeypatch, tmp_path):
    plan, _, _, models, trace = cpu_conditional_calculation(monkeypatch, tmp_path, wrong_anchor=True)
    progress = {'execution_refs': trace['execution_refs']}
    with pytest.raises(ValueError, match='PRIOR_CPU_ANCHOR_MISMATCH'):
        m.calculate_cpu_conditional(plan, progress=progress)
    assert trace['calls'] == [0] and trace['publications'] == trace['grad_rows'] == trace['proof_calls'] == []
    assert progress['counts']['autograd_calls'] == 0 and progress['counts']['baseline_artifact_writes'] == 0
    assert progress['prior_CPU_anchor_comparison']['side_logits']['all_within_original_tolerance'] is False
    assert models[0].backbone.weight.grad is None and not models[0]._forward_pre_hooks


def test_cpu_conditional_late_row_mismatch_keeps_baseline_but_blocks_that_rows_derivatives_and_verdict(monkeypatch, tmp_path):
    from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as body
    plan, _, _, models, trace = cpu_conditional_calculation(monkeypatch, tmp_path, mismatch_row=3)
    progress = {'execution_refs': trace['execution_refs']}; ambient = torch.is_grad_enabled()
    with pytest.raises(ValueError, match='REPLAY_MISMATCH_BEFORE_DERIVATIVES'):
        m.calculate_cpu_conditional(plan, progress=progress)
    assert len(trace['calls']) == 100 and trace['grad_rows'] == [row for row in range(3) for _ in range(4)]
    assert len(trace['publications']) == 1 and trace['publications'][0]['counts']['autograd_calls'] == 0
    assert progress['cpu_baseline'] == trace['baseline_ref']
    assert progress['counts']['NN_forwards'] == 100 and progress['counts']['autograd_calls'] == 12
    assert len(progress['second_pass_rows']) == 4
    failed = progress['second_pass_rows'][-1]
    assert failed['index'] == 3 and set(failed['outputs']) == {'activity_logits', 'side_logits'}
    assert failed['vs_CPU_baseline']['activity_logits']['all_within_original_tolerance'] is True
    assert failed['vs_CPU_baseline']['side_logits']['all_within_original_tolerance'] is False
    assert trace['proof_calls'] == [] and len(trace['scales']) == 3
    assert 'gradient_proof' not in progress
    assert not models[0]._forward_pre_hooks and models[0].backbone.weight.grad is None
    assert body._BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False and body._MODEL_FINITE_CHECK_SCOPE.get() is None
    assert torch.is_grad_enabled() is ambient
    json.dumps(progress, allow_nan=False)


def test_cpu_conditional_inspection_rejects_old_SHA_and_keeps_every_previous_plan_consumed(monkeypatch, tmp_path):
    planfile, policyfile = tmp_path / 'CPU_PLAN.json', tmp_path / 'POLICY.json'
    monkeypatch.setattr(m, 'CPU_PLAN', planfile); monkeypatch.setattr(m, 'POLICY', policyfile)
    monkeypatch.setattr(m, 'CPU_ROOT', tmp_path / 'CPU_OUT'); monkeypatch.setattr(m, 'REPO', Path.cwd())
    inputs = {role: {'path': '/synthetic/' + role + '.json', 'sha256': digit*64}
              for role, digit in (('failed_gradient_plan', 'a'), ('first_parity_plan', 'b'), ('batch_geometry_plan', 'c'))}
    plan = {'source_base_commit': '1'*40, 'inputs': inputs}
    planfile.write_text(json.dumps(plan)); binding = m.ref(planfile)
    policy = dict.fromkeys(m.saved.FLAGS, False)
    for stage, role in ((m.STAGE, 'failed_gradient_plan'), (m.PARITY_STAGE, 'first_parity_plan'), (m.BATCH_STAGE, 'batch_geometry_plan')):
        policy[stage] = {'plan': inputs[role], 'execution_authorized': False, 'remaining_invocations': 0,
                        'relaunch_allowed': False, 'training_authorized': False, 'test_authorized': False}
    policy[m.CPU_STAGE] = m.gate_cpu_conditional_expected(binding, False)
    monkeypatch.setattr(m, 'build_cpu_conditional_plan', lambda **k: plan)
    monkeypatch.setattr(m, 'source_clean', lambda value: True)
    def inspect():
        policyfile.write_text(json.dumps(policy)); return m.inspect_cpu_conditional(binding['sha256'], admitted=False)
    assert inspect()[0] == plan and not m.CPU_ROOT.exists()
    with pytest.raises(ValueError): m.inspect_cpu_conditional(inputs['batch_geometry_plan']['sha256'], admitted=False)
    for key in m.saved.FLAGS:
        policy[key] = True
        with pytest.raises(ValueError, match='ONE_USE_FLAGS'): inspect()
        policy[key] = False
    for stage in (m.STAGE, m.PARITY_STAGE, m.BATCH_STAGE):
        policy[stage]['remaining_invocations'] = 1
        with pytest.raises(ValueError, match='OLD_PLANS_CLOSED'): inspect()
        policy[stage]['remaining_invocations'] = 0
        for key in ('execution_authorized', 'training_authorized', 'test_authorized', 'relaunch_allowed'):
            policy[stage][key] = True
            with pytest.raises(ValueError, match='OLD_PLANS_CLOSED'): inspect()
            policy[stage][key] = False
    policy[m.CPU_STAGE]['training_authorized'] = True
    with pytest.raises(ValueError, match='ONE_USE_FLAGS'): inspect()
    policy[m.CPU_STAGE]['training_authorized'] = False; m.CPU_ROOT.mkdir()
    with pytest.raises(ValueError, match='FRESH_ROOT'): inspect()


def test_cpu_conditional_missing_null_or_wrong_execution_refs_fail_before_payload_loading(monkeypatch, tmp_path):
    plan, _, _, models, trace = cpu_conditional_calculation(monkeypatch, tmp_path)
    def forbidden(*a, **k): raise AssertionError('NO_PAYLOAD_BEFORE_EXECUTION_REF_AUTHENTICATION')
    monkeypatch.setattr(m.adapter, 'load_context', forbidden)
    invalid = [{}, {'execution_refs': None}]
    wrong = copy.deepcopy(trace['execution_refs'])
    wrong['plan'] = wrong['policy_at_claim'].copy()
    invalid.append({'execution_refs': wrong})
    for progress in invalid:
        with pytest.raises(ValueError, match='EXECUTION_REF'):
            m.calculate_cpu_conditional(plan, progress=progress)
    assert models == trace['calls'] == trace['grad_rows'] == trace['publications'] == []


def test_cpu_conditional_builder_uses_only_CPU_limits_and_preserves_gradient_budget():
    tree = ast.parse(Path(m.__file__).read_text())
    builder = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == 'build_cpu_conditional_plan')
    assignment = lambda name: next(node for node in builder.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
    budget_statement = assignment('budget')
    budget_delete = next(node for node in builder.body if isinstance(node, ast.Delete)
        and any(isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                and target.value.id == 'budget' for target in node.targets))
    plan_expression = assignment('plan').value
    assert isinstance(plan_expression, ast.Dict)
    expressions = {key.value: value for key, value in zip(plan_expression.keys, plan_expression.values)
                   if isinstance(key, ast.Constant) and key.value in ('limits', 'budget')}
    assert set(expressions) == {'limits', 'budget'}
    scope = {'why_new_scope': 'CPU-conditional local risk; saved-CUDA gate stays FAILED.',
             'decision_before_run': {'both_outcomes': 'Historical AdamW cause and a selected fix remain UNKNOWN.'},
             'resource_budget': {'actual_full96_gradient_peak': 'CPU gradient peak UNMEASURED; no-grad memory is no gradient qualification.'}}
    stale_limits = ['No autograd', '169086976 B256-onlypacket']
    environment = {'scope': scope, 'resources': scope['resource_budget'],
        'base': {'limits': stale_limits, 'budget': {'NN_forwards_max': 1, 'autograd_calls_max': 0,
                                                   'packet_feature_bytes': 169086976}},
        'original_design': {'budget': {'NN_forwards_max': 192, 'autograd_calls_max': 384,
            'packet_feature_bytes': 63407616, 'original_packet_rows': 96, 'panel_rows': 96, 'parity_forwards': 96}}}
    statements = ast.Module(body=[copy.deepcopy(budget_statement), copy.deepcopy(budget_delete)], type_ignores=[])
    exec(compile(ast.fix_missing_locations(statements), '<actual_CPU_budget_expression>', 'exec'), environment)
    selected = ast.Expression(body=ast.Dict(keys=[ast.Constant('limits'), ast.Constant('budget')],
        values=[copy.deepcopy(expressions['limits']), copy.deepcopy(expressions['budget'])]))
    result = eval(compile(ast.fix_missing_locations(selected), '<actual_CPU_limits_and_budget>', 'eval'), environment)
    limits = result['limits']
    assert len(limits) == 4
    assert limits[:3] == [scope['why_new_scope'], scope['decision_before_run']['both_outcomes'],
                          scope['resource_budget']['actual_full96_gradient_peak']]
    assert all(stale not in '\n'.join(limits) for stale in stale_limits)
    own_limit = limits[3]
    assert 'Original B96 CPU-eval local interaction only' in own_limit
    assert '192 forwards/384 autograd calls' in own_limit
    for prohibition in ('no backward()', 'optimizer or parameter update', 'checkpoint RNG restoration',
                        'refit', 'new source HTTP', 'selected VAL/TEST', 'Entry/Exit', 'trading'):
        assert prohibition in own_limit
    assert {key: result['budget'][key] for key in ('NN_forwards_max', 'autograd_calls_max', 'packet_feature_bytes',
                                                   'original_packet_rows', 'panel_rows')} == {
        'NN_forwards_max': 192, 'autograd_calls_max': 384, 'packet_feature_bytes': 63407616,
        'original_packet_rows': 96, 'panel_rows': 96}
    assert 'parity_forwards' not in result['budget']


# All-TF contrast tests; earlier qualification cases remain unchanged.
def all_tf_strip_source(source):
    begin = '# All-TF frozen CPU contrast; original receipts remain unchanged.\n'
    end = '# End all-TF frozen CPU contrast.\n\n'
    dispatch = "    if sys.argv[4:5]==['--all-tf']:\n        return main_all_tf()\n\n"
    assert source.count(begin) == source.count(end) == 1
    first, tail = source.split(begin)
    _, last = tail.split(end)
    assert last.count(dispatch) == 1
    return first + last.replace(dispatch, '', 1)


def test_all_tf_source_addition_strips_to_exact_prior_CPU_owner_and_test_prefix():
    import subprocess
    anchor = '77287c59c03f16cc88240fd1db543a090e0096ca'
    before = subprocess.check_output(['git', 'show', anchor + ':' + m.SELF], cwd=m.REPO, text=True)
    assert all_tf_strip_source(Path(m.__file__).read_text()) == before
    old_tests = subprocess.check_output(['git', 'show', anchor + ':' + m.TEST], cwd=m.REPO)
    assert Path(__file__).read_bytes().startswith(old_tests)


class AllTFTransferState(torch.nn.Module):
    def __init__(self, contract):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros((1, 2), dtype=torch.float32))
        self.parameter_alias = self.parameter
        self.register_buffer('immutable', torch.tensor([3, 4], dtype=torch.int64))
        self.register_buffer('future_state_model_sha256', torch.tensor(
            list(bytes.fromhex(contract['contract_sha256'])), dtype=torch.uint8))
        self._contract = contract
        self._backbone_gradient_policy = None
    @property
    def research_objective(self):
        return self._contract


def all_tf_transfer_fixture():
    candidate_contract = {'schema_version': 'gx1_native_future_state_forward_model_v1', 'contract_sha256': 'b'*64}
    original_contract = {**candidate_contract, 'contract_sha256': 'a'*64,
                         'backbone_gradient_policy': {'policy': 'M5_primary'}}
    candidate = AllTFTransferState(candidate_contract)
    original_state = candidate.state_dict()
    original_weight = torch.tensor([[2., -3.]], dtype=torch.float32)
    original_state['parameter'] = original_weight
    original_state['parameter_alias'] = original_weight
    original_state['immutable'] = original_state['immutable'].clone()
    original_state['future_state_model_sha256'] = torch.tensor(list(bytes.fromhex('a'*64)), dtype=torch.uint8)
    return candidate, original_state, original_contract, candidate_contract


def test_all_tf_numeric_transfer_preserves_original_and_candidate_identity_with_exact_aliases(monkeypatch):
    candidate, original, old_contract, new_contract = all_tf_transfer_fixture()
    original_sha = m.core.state_sha(original)
    candidate_id = candidate.future_state_model_sha256.clone()
    strict_loads = []; load = candidate.load_state_dict
    def strict(state, *args, **kwargs):
        strict_loads.append((list(state), kwargs.copy()))
        assert kwargs['strict'] is True and kwargs.get('assign', False) is False
        assert torch.equal(state['future_state_model_sha256'], candidate_id)
        assert list(state) == list(original)
        return load(state, *args, **kwargs)
    monkeypatch.setattr(candidate, 'load_state_dict', strict)
    proof = m.all_tf_numeric_transfer(candidate, original, original_contract=old_contract, candidate_contract=new_contract)
    assert len(strict_loads) == 1
    assert torch.equal(candidate.parameter, original['parameter'])
    assert candidate.parameter_alias is candidate.parameter
    assert torch.equal(candidate.future_state_model_sha256, candidate_id)
    assert torch.equal(candidate.immutable, original['immutable'])
    assert m.core.state_sha(original) == original_sha
    assert proof['sole_state_difference'] == ['future_state_model_sha256']
    assert all(proof[key] is True for key in ('all_named_parameters_exact', 'all_other_persistent_buffers_exact',
                                            'strict_load', 'original_state_unmutated'))
    aliases = json.dumps(proof['parameter_alias_groups'])
    assert 'parameter_alias' in aliases and 'parameter' in aliases


@pytest.mark.parametrize('bad', ['missing', 'extra', 'shape', 'dtype', 'nan', 'alias', 'buffer', 'old_identity', 'new_identity', 'order'])
def test_all_tf_numeric_transfer_rejects_invalid_state_without_mutating_candidate_or_original(bad):
    from collections import OrderedDict
    candidate, original, old_contract, new_contract = all_tf_transfer_fixture()
    if bad == 'missing': del original['parameter_alias']
    elif bad == 'extra': original['unclaimed_buffer'] = torch.zeros(1)
    elif bad == 'shape': original['parameter'] = torch.zeros(3, dtype=torch.float32)
    elif bad == 'dtype': original['parameter'] = original['parameter'].double()
    elif bad == 'nan': original['parameter'] = torch.full((1, 2), float('nan'))
    elif bad == 'alias': original['parameter_alias'] = original['parameter'].clone() + 1.
    elif bad == 'buffer': original['immutable'][0] += 1
    elif bad == 'old_identity': original['future_state_model_sha256'][0] ^= 1
    elif bad == 'new_identity': candidate.future_state_model_sha256[0] ^= 1
    else: original = OrderedDict(reversed(list(original.items())))
    candidate_before, original_before = m.core.state_sha(candidate.state_dict()), m.core.state_sha(original)
    with pytest.raises(ValueError):
        m.all_tf_numeric_transfer(candidate, original, original_contract=old_contract, candidate_contract=new_contract)
    assert m.core.state_sha(candidate.state_dict()) == candidate_before
    assert m.core.state_sha(original) == original_before


def all_tf_contrast_vectors():
    # gRef=[1,0], gAll=[-1,0]; the candidate auxiliary direction is nonzero.
    values = [[-1., 0.], [2., 0.], [-1., 0.], [0., 1.], [-2., 0.], [0., -1.], [0., 0.], [0., 0.]]
    return {name: vector(value) for name, value in zip(m.ALL_TERMS, values)}


def test_all_tf_gradient_report_accepts_nonzero_auxiliary_and_contrasts_each_TF_direction():
    report = m.all_tf_gradient_report(all_tf_contrast_vectors(), tolerances=TOLERANCES)
    assert report['aux_gradient_status'] == 'NONZERO'
    assert set(report['per_tf']) == {'M5', 'M15', 'H1', 'H4', 'D1'}
    for tf, expected_ref, expected_all in (('M5', -1., 1.), ('M15', -2., 2.)):
        assert report['per_tf'][tf]['gRef']['dot'] == expected_ref
        assert report['per_tf'][tf]['gAll']['dot'] == expected_all
        assert report['per_tf'][tf]['gRef']['local_verdict'] == 'LOCAL_CONFLICT'
        assert report['per_tf'][tf]['gAll']['local_verdict'] == 'NOT_LOCAL_CONFLICT'
    for tf in ('H1', 'H4', 'D1'):
        assert report['per_tf'][tf]['gRef']['local_verdict'] == 'INDETERMINATE'
        assert report['per_tf'][tf]['gAll']['local_verdict'] == 'INDETERMINATE'
    assert report['per_tf']['H4']['gAll']['side_L2'] == 0.
    json.dumps(report, allow_nan=False)


def test_all_tf_gradient_report_retains_zero_auxiliary_as_measured_instead_of_forcing_nonzero():
    gradients = {name: [None] for name in m.ALL_TERMS}
    gradients[m.ALL_TERMS[0]], gradients[m.ALL_TERMS[1]], gradients[m.ALL_TERMS[2]] = vector([2.]), vector([1.]), vector([1.])
    report = m.all_tf_gradient_report(gradients, tolerances=TOLERANCES)
    assert report['aux_gradient_status'] == 'ZERO'
    assert report['per_tf']['M5']['gRef']['local_verdict'] == 'NOT_LOCAL_CONFLICT'
    assert report['per_tf']['D1']['gAll']['local_verdict'] == 'INDETERMINATE'


@pytest.mark.parametrize('bad', ['component_mismatch', 'missing', 'extra', 'nan', 'tolerance'])
def test_all_tf_gradient_report_rejects_broken_component_axis_or_unlocked_tolerance(bad):
    gradients = all_tf_contrast_vectors(); tolerances = copy.deepcopy(TOLERANCES)
    if bad == 'component_mismatch': gradients[m.ALL_TERMS[0]] = vector([1., 0.])
    elif bad == 'missing': del gradients[m.ALL_TERMS[-1]]
    elif bad == 'extra': gradients['unclaimed_component'] = vector([0., 0.])
    elif bad == 'nan': gradients[m.ALL_TERMS[1]] = vector([float('nan'), 0.])
    else: tolerances['component_gradient_global_relative_L2_max'] = .1
    with pytest.raises(ValueError): m.all_tf_gradient_report(gradients, tolerances=tolerances)


def all_tf_calculation(monkeypatch, tmp_path, *, baseline_mismatch=False, mismatch_row=None):
    from collections import OrderedDict
    from types import SimpleNamespace
    import weakref
    from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as body
    class Context(dict): pass
    class Body(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(0., dtype=torch.float32))
    candidate_contract = {'schema_version': 'gx1_native_future_state_forward_model_v1', 'contract_sha256': 'b'*64}
    original_contract = {**candidate_contract, 'contract_sha256': 'a'*64,
                         'backbone_gradient_policy': {'policy': 'M5_primary'}}
    class Candidate(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.backbone = Body(); self._backbone_gradient_policy = None
            self.register_buffer('immutable', torch.tensor([3, 4], dtype=torch.int64))
            self.register_buffer('future_state_model_sha256', torch.tensor(list(bytes.fromhex('b'*64)), dtype=torch.uint8))
        @property
        def research_objective(self): return candidate_contract
        def forward(self, **features):
            z = self.backbone.weight * features['seq_x'][:, 0, :]
            coefficients = torch.arange(1, 6, dtype=torch.float32).view(1, 5)
            return {'activity_logits': z*coefficients, 'side_logits': (-2*z)*coefficients}
    original_state = OrderedDict((('immutable', torch.tensor([3, 4], dtype=torch.int64)),
        ('future_state_model_sha256', torch.tensor(list(bytes.fromhex('a'*64)), dtype=torch.uint8)),
        ('backbone.weight', torch.tensor(.25, dtype=torch.float32))))
    cp = {'model_state': original_state}; cp_sha = m.core.state_sha(cp); stored_proof = {'synthetic_all_ten_authentication': True}
    n = 96; issues = np.arange(n, dtype=np.int64)*300_000_000_000 + int(m.train.START); rows = np.arange(n, dtype=np.int64)
    x = np.linspace(.1, .3, n, dtype=np.float32).reshape(n, 1)
    features = {'seq_x': torch.from_numpy(np.repeat(x[:, None, :], 96, axis=1)),
                'snap_x': torch.from_numpy(x.copy()), 'ctx_cont': torch.from_numpy(x.copy()),
                'ctx_cat': torch.zeros((n, 1), dtype=torch.int64)}
    windows = {'m5': 16, 'm15': 64, 'h1': 96, 'h4': 96, 'd1': 252}
    features.update({'seq_'+tf: torch.from_numpy(np.repeat(x[:, None, :], length, axis=1)) for tf, length in windows.items()})
    side_known = np.zeros((n, 5), bool); side_known[:, 0] = True
    side_known[::2, 1] = True; side_known[::6, 2] = True; side_known[::24, 3] = True; side_known[:4, 4] = True
    starts = np.column_stack([int(m.train.START)+((np.arange(n)//group)+1)*group*300_000_000_000
                              for group in (1, 3, 12, 48, 288)]).astype(np.int64)
    truth = {'issue_ns': issues, 'activity': np.zeros((n, 5), np.float32), 'side_up': np.zeros((n, 5), np.float32),
             'activity_known': np.ones((n, 5), bool), 'side_known': side_known, 'target_start_ns': starts}
    packet = {'features': features, 'truth': truth}
    coefficients = np.arange(1, 6, dtype=np.float32).reshape(1, 5)
    baseline = {'activity_logits': (x*.25)*coefficients, 'side_logits': (-2*(x*.25))*coefficients}
    capture = {'issue_ns': issues.copy(), **{head: value-np.float32(.01) for head, value in baseline.items()}}
    def record(value):
        return {'values': value.astype(np.float64).tolist(), 'shape': list(value.shape), 'dtype': value.dtype.str, 'sha256': m.core.digest(value)}
    predecessor = {'status': 'CPU_BASELINE_COMPLETE_BEFORE_DERIVATIVES',
                   'counts': {'NN_forwards': 96, 'autograd_calls': 0}, 'outputs': {head: record(value) for head, value in baseline.items()}}
    if baseline_mismatch:
        changed = baseline['side_logits'].copy(); changed[95, 4] += .02
        predecessor['outputs']['side_logits'] = record(changed)
    predecessor_path = tmp_path / 'PREDECESSOR_CPU_BASELINE.json'
    predecessor_path.write_text(json.dumps(predecessor, sort_keys=True, allow_nan=False))
    predecessor_ref = m.ref(predecessor_path)
    anchor = {'name': 'CPU_GRAD_ENABLED_UNMERGED', 'logits': {
        head: {**record(value[:1]), 'values': value[0].astype(np.float64).tolist()} for head, value in baseline.items()}}
    row0_proof = {key: {'shape': list(value[:1].shape), 'dtype': str(value.dtype),
                       'sha256': m.core.digest(value[:1].numpy())} for key, value in features.items()}
    plan = {'metadata': {'adapter_metadata': {}, 'original_epoch_plan': {'metadata': {'architecture': {}}},
                        'final_receipt': {'proof': stored_proof}}, 'original_model_contract': original_contract,
            'candidate_model_contract': candidate_contract,
            'prior_CPU_anchor': {'route': anchor, 'selected_issue_ns': int(issues[0]), 'selected_source_row': 0, 'feature_proof': row0_proof},
            'inputs': {'capture': {'path': '/synthetic/ALL_CAPTURE.npz', 'sha256': 'a'*64},
                       'predecessor_CPU_baseline': predecessor_ref},
            'panel': {'issue_ns_sha256': m.core.digest(issues), 'source_rows_sha256': m.core.digest(rows)},
            'TRAIN_issue_ns_sha256': m.core.digest(issues), 'capture_array_headers': {},
            'budget': {'packet_feature_bytes': sum(value.numel()*value.element_size() for value in features.values()),
                       'capture_ndarray_bytes_decoded': sum(value.nbytes for value in capture.values()),
                       'NN_forwards_max': 192, 'autograd_calls_max': 768}, 'route': copy.deepcopy(m.ALL_ROUTE),
            'tolerances': copy.deepcopy(TOLERANCES), 'fixed_original_tolerance': copy.deepcopy(TOLERANCES['saved_FINAL_logits']),
            'cache_scope': {'synthetic': True}, 'limits': []}
    monkeypatch.setattr(m, 'ALL_ROOT', tmp_path / 'ALL_CLAIMED'); m.ALL_ROOT.mkdir()
    monkeypatch.setattr(m, 'ALL_PLAN', tmp_path / 'ALL_PLAN.json'); monkeypatch.setattr(m, 'POLICY', tmp_path / 'POLICY.json')
    m.ALL_PLAN.write_text(json.dumps(plan, sort_keys=True, allow_nan=False)); plan_ref = m.ref(m.ALL_PLAN)
    policy = dict.fromkeys(m.saved.FLAGS, False); policy[m.ALL_STAGE] = m.gate_all_tf_expected(plan_ref, True)
    m.POLICY.write_text(json.dumps(policy, sort_keys=True)); policy_ref = m.ref(m.POLICY)
    claim = {'stage': m.ALL_STAGE, 'plan': plan_ref, 'policy_at_claim': policy_ref, 'remaining_invocations': 0, 'relaunch_allowed': False}
    claim_path = m.ALL_ROOT/'CLAIM.json'; claim_path.write_text(json.dumps(claim, sort_keys=True))
    refs = {'plan': plan_ref, 'claim': m.ref(claim_path), 'policy_at_claim': policy_ref}
    released, models, calls, events, grad_rows, gates, scales, checkpoint_checks, proof_calls = [], [], [], [], [], [], [], [], []
    original_grad, original_compare, original_scale, original_report = torch.autograd.grad, m.first_parity_compare, m.scaled_canonical_loss, m.all_tf_gradient_report
    def load_context(_):
        context = Context(provider=SimpleNamespace(bar_start_ns=np.array([int(m.train.START)], np.int64)),
                          source_rows=rows, normalization={}, prior={}, input_binding={}, target_contract={})
        released.append(weakref.ref(context)); return context
    def factory(**kwargs):
        assert kwargs['backbone_gradient_policy'] is None and kwargs['device'] == torch.device('cpu')
        assert released[0]() is None
        model = Candidate(); models.append(model); return model
    def checkpoint_proof(value):
        assert value is cp and m.core.state_sha(cp) == cp_sha
        checkpoint_checks.append(True); return stored_proof.copy()
    def helper(model, bound_features, *, grouped, equal_tf, capture):
        assert grouped is True and equal_tf is capture is False and torch.is_grad_enabled() is True and model.training is False
        assert events and events[0][0] == 'TRANSFER_PROOF.json'
        assert len(model._forward_pre_hooks) == 1
        assert all(not child._forward_hooks and not child._forward_pre_hooks for child in model.modules() if child is not model)
        index = len(calls); calls.append(index)
        with body.model_finite_check_scope('grouped'): output = model(**bound_features)
        if index >= 96 and index-96 == mismatch_row: output['side_logits'] = output['side_logits'] + .02
        activation = {'batch_eval_active': False, 'model_training': False, 'grad_enabled': True, 'scope_enabled': False,
                      'eligible_equal_length_TF_groups': {'16': ['M5'], '64': ['M15'], '96': ['H1', 'H4'], '252': ['D1']}}
        return output, None, activation
    def compare(reference, candidate, *, tolerance):
        result = original_compare(reference, candidate, tolerance=tolerance)
        if len(calls) > 96:
            gates.append((len(calls)-97, 'activity_logits' if candidate[0, 0] > 0 else 'side_logits', result['all_within_original_tolerance']))
        return result
    def gradient(*args, **kwargs):
        row = len(calls)-97
        assert [name for name, _ in events] == ['TRANSFER_PROOF.json', 'CPU_BASELINE.json']
        assert {head for index, head, passed in gates if index == row and passed} == {'activity_logits', 'side_logits'}
        grad_rows.append(row); return original_grad(*args, **kwargs)
    def publish(name, value):
        if name == 'TRANSFER_PROOF.json': assert calls == grad_rows == []
        elif name == 'CPU_BASELINE.json': assert len(calls) == 96 and grad_rows == []
        else: raise AssertionError('UNBUDGETED_PUBLICATION')
        events.append((name, copy.deepcopy(value)))
        return {'path': str(m.ALL_ROOT/name), 'sha256': ('e' if name.startswith('TRANSFER') else 'f')*64}
    def canonical(*, model_output, truth, **kwargs):
        losses = []
        for head, label, mask in (('activity_logits', 'activity', 'activity_known'), ('side_logits', 'side_up', 'side_known')):
            y, known = torch.from_numpy(truth[label]), torch.from_numpy(truth[mask])
            bce = torch.nn.functional.binary_cross_entropy_with_logits(model_output[head], y, reduction='none')
            losses.append(torch.where(known, bce, torch.zeros_like(bce)).sum())
        count = int(truth['activity_known'].sum()); return {'loss': sum(losses)/count, 'count': count}
    def scaling(loss, chunk_denominator, full_denominator):
        assert (chunk_denominator, full_denominator) == (5, 480)
        scales.append(True); return original_scale(loss, chunk_denominator, full_denominator)
    def report(gradients, **kwargs):
        assert all(value is None or value.dtype == torch.float64 for values in gradients.values() for value in values)
        proof_calls.append(True); return original_report(gradients, **kwargs)
    monkeypatch.setattr(m.adapter, 'load_context', load_context); monkeypatch.setattr(m.adapter, 'packet', lambda *a: packet)
    monkeypatch.setattr(m.train, 'slice_provider_hlc_truth', lambda value, take, **k: {key: array[take].copy() for key, array in value.items()})
    monkeypatch.setattr(m.saved, 'load_fields', lambda binding, fields, headers: {key: capture[key] for key in fields})
    monkeypatch.setattr(m.model_owner, 'build_native_future_state_model', factory); monkeypatch.setattr(m, 'safe_load_final', lambda plan: cp)
    monkeypatch.setattr(m.train, 'checkpoint_proof', checkpoint_proof); monkeypatch.setattr(m.core, 'future_cuda_forward', helper)
    monkeypatch.setattr(m, 'first_parity_compare', compare); monkeypatch.setattr(torch.autograd, 'grad', gradient)
    monkeypatch.setattr(m, 'all_tf_publish', publish); monkeypatch.setattr(m.objective, 'native_future_state_loss', canonical)
    monkeypatch.setattr(m, 'scaled_canonical_loss', scaling); monkeypatch.setattr(m, 'all_tf_gradient_report', report)
    monkeypatch.setattr(torch, 'set_num_threads', lambda count: None); monkeypatch.setattr(torch, 'get_num_threads', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    trace = {'refs': refs, 'calls': calls, 'events': events, 'grad_rows': grad_rows, 'scales': scales,
             'checkpoint_checks': checkpoint_checks, 'proof_calls': proof_calls, 'cp': cp, 'cp_sha': cp_sha}
    return plan, baseline, packet, models, trace


def all_tf_counts():
    counts = cpu_conditional_counts(); counts.update(autograd_calls=768, transfer_artifact_writes=1)
    return counts


def test_all_tf_complete_contrast_transfers_only_parameters_and_gates_all96_before_any_gradient(monkeypatch, tmp_path):
    plan, baseline, packet, models, trace = all_tf_calculation(monkeypatch, tmp_path)
    progress = {'execution_refs': trace['refs']}; report = m.calculate_all_tf(plan, progress=progress)
    assert report['counts'] == all_tf_counts() and len(models) == 1
    assert len(trace['calls']) == 192 and trace['grad_rows'] == [row for row in range(96) for _ in range(8)]
    assert len(trace['scales']) == 96 and trace['checkpoint_checks'] == trace['proof_calls'] == [True]
    assert [name for name, _ in trace['events']] == ['TRANSFER_PROOF.json', 'CPU_BASELINE.json']
    transfer, artifact = [value for _, value in trace['events']]
    assert transfer['status'] == 'TRANSFER_PROOF_COMPLETE_BEFORE_FORWARD'
    assert transfer['counts']['NN_forwards'] == transfer['counts']['autograd_calls'] == 0
    assert transfer['proof']['sole_state_difference'] == ['future_state_model_sha256']
    assert artifact['status'] == 'ALL_TF_CPU_BASELINE_COMPLETE_BEFORE_DERIVATIVES'
    assert artifact['counts']['NN_forwards'] == 96 and artifact['counts']['autograd_calls'] == 0
    assert all(value['all_within_original_tolerance'] for value in report['candidate_vs_predecessor_CPU_baseline'].values())
    assert report['known_activity_denominator_full_B96'] == 480
    assert report['gradient_proof']['aux_gradient_status'] == 'NONZERO'
    assert report['gradient_proof']['component_gradient_contract_verified'] is True
    assert report['strict_candidate_owned_state_installed'] is report['no_candidate_fit_warmstart'] is True
    assert report['original_model_contract'] != report['candidate_model_contract']
    assert m.core.state_sha(trace['cp']) == trace['cp_sha']
    for head in baseline:
        assert report['cpu_baseline_outputs'][head]['values'] == baseline[head].astype(np.float64).tolist()
        assert report['cpu_replay_comparison'][head]['failed_cells'] == 0
        assert report['saved_CUDA_comparison'][head]['failed_cells'] == 480
    expected_side_rows = [96, 48, 16, 4, 4]
    expected_distinct = [96, 32, 8, 2, 1]
    for j, tf in enumerate(('M5', 'M15', 'H1', 'H4', 'D1')):
        assert report['per_TF_support'][tf] == {'activity_rows': 96, 'side_rows': expected_side_rows[j],
            'activity_distinct_target_start_ns': expected_distinct[j], 'side_distinct_target_start_ns': expected_distinct[j]}
    assert report['per_TF_support']['D1']['side_rows'] == 4 and report['per_TF_support']['D1']['side_distinct_target_start_ns'] == 1
    assert report['original_saved_CUDA_gate'] == 'FAILED' and report['full96_gradient_qualification'] is False
    assert report['ALL_TF_counterfactual_local_interaction_only'] is report['TRANSFER_PROOF_published_before_forward'] is True
    assert report['retained_B96_packet_unmutated'] is report['model_state_and_local_RNG_bookend_passed'] is True
    assert models[0].backbone.weight.item() == .25 and models[0].backbone.weight.grad is None and not models[0]._forward_pre_hooks
    assert report['actual_training_cause'] == report['selected_causal_fix'] == report['economic_edge'] == 'UNKNOWN'
    json.dumps(report, allow_nan=False)


def test_all_tf_candidate_baseline_failure_after_row95_blocks_every_derivative(monkeypatch, tmp_path):
    plan, _, _, models, trace = all_tf_calculation(monkeypatch, tmp_path, baseline_mismatch=True)
    progress = {'execution_refs': trace['refs']}
    with pytest.raises(ValueError, match='CANDIDATE_BASELINE_MISMATCH_BEFORE_DERIVATIVES'):
        m.calculate_all_tf(plan, progress=progress)
    assert len(trace['calls']) == 96 and trace['grad_rows'] == trace['scales'] == trace['proof_calls'] == []
    assert [name for name, _ in trace['events']] == ['TRANSFER_PROOF.json']
    assert progress['counts']['autograd_calls'] == progress['counts']['baseline_artifact_writes'] == 0
    assert progress['candidate_vs_predecessor_CPU_baseline']['side_logits']['failed_cells'] == 1
    assert models[0].backbone.weight.grad is None and not models[0]._forward_pre_hooks


def test_all_tf_late_replay_mismatch_preserves_transfer_and_baseline_but_invalidates_entire_gradient(monkeypatch, tmp_path):
    plan, _, _, models, trace = all_tf_calculation(monkeypatch, tmp_path, mismatch_row=3)
    progress = {'execution_refs': trace['refs']}
    with pytest.raises(ValueError, match='REPLAY_MISMATCH_BEFORE_DERIVATIVES'):
        m.calculate_all_tf(plan, progress=progress)
    assert len(trace['calls']) == 100 and trace['grad_rows'] == [row for row in range(3) for _ in range(8)]
    assert [name for name, _ in trace['events']] == ['TRANSFER_PROOF.json', 'CPU_BASELINE.json']
    assert progress['counts']['autograd_calls'] == 24 and len(trace['scales']) == 3
    assert progress['second_pass_rows'][-1]['index'] == 3
    assert progress['second_pass_rows'][-1]['vs_CPU_baseline']['side_logits']['all_within_original_tolerance'] is False
    assert trace['proof_calls'] == [] and 'gradient_proof' not in progress
    assert m.core.state_sha(trace['cp']) == trace['cp_sha']
    assert models[0].backbone.weight.grad is None and not models[0]._forward_pre_hooks
    json.dumps(progress, allow_nan=False)


def test_all_tf_execution_refs_are_required_before_payload_or_factory(monkeypatch, tmp_path):
    plan, _, _, models, trace = all_tf_calculation(monkeypatch, tmp_path)
    def forbidden(*a, **k): raise AssertionError('NO_ALL_TF_PAYLOAD_BEFORE_EXECUTION_BINDING')
    monkeypatch.setattr(m.adapter, 'load_context', forbidden)
    wrong = copy.deepcopy(trace['refs']); wrong['plan'] = wrong['policy_at_claim'].copy()
    for progress in ({}, {'execution_refs': None}, {'execution_refs': wrong}):
        with pytest.raises(ValueError, match='EXECUTION_REF'):
            m.calculate_all_tf(plan, progress=progress)
    assert models == trace['calls'] == trace['events'] == trace['grad_rows'] == []


def test_all_tf_exact_counts_require_768_derivatives_and_one_transfer_artifact():
    assert m.all_tf_validate_counts(all_tf_counts(), {'NN_forwards_max': 192, 'autograd_calls_max': 768}) is True
    for key in ('autograd_calls', 'transfer_artifact_writes', 'baseline_artifact_writes', 'optimizer_updates'):
        counts = all_tf_counts(); counts[key] += 1
        with pytest.raises(ValueError): m.all_tf_validate_counts(counts, {'NN_forwards_max': 192, 'autograd_calls_max': 768})


def all_tf_lifecycle(monkeypatch, tmp_path):
    plan, _, pref, hooks, timers = cpu_conditional_lifecycle(monkeypatch, tmp_path)
    monkeypatch.setattr(m, 'ALL_ROOT', tmp_path / 'ALL_OUT'); monkeypatch.setattr(m, 'ALL_PLAN', tmp_path / 'ALL_PLAN.json')
    m.ALL_PLAN.write_text('{"synthetic_all_tf":true}'); binding = m.ref(m.ALL_PLAN)
    monkeypatch.setattr(m, 'inspect_all_tf', lambda *a, **k: (plan, binding, pref))
    monkeypatch.setattr(m, 'calculate_all_tf', lambda *a, **k: {'original_saved_CUDA_gate': 'FAILED'})
    return plan, binding, pref, hooks, timers


def test_all_tf_exact_nested_opt_in_preflight_cannot_load_payload_or_publish_transfer(monkeypatch, tmp_path, capsys):
    _, binding, _, hooks, _ = all_tf_lifecycle(monkeypatch, tmp_path)
    def forbidden(*a, **k): raise AssertionError('NO_ALL_TF_PAYLOAD_IN_PREFLIGHT')
    monkeypatch.setattr(m, 'calculate_all_tf', forbidden); monkeypatch.setattr(torch, 'load', forbidden); monkeypatch.setattr(np, 'load', forbidden)
    monkeypatch.setattr(sys, 'argv', ['audit', '--first-parity-plan-sha256', binding['sha256'], '--cpu-conditional-gradient', '--all-tf', '--preflight'])
    m.main(); result = json.loads(capsys.readouterr().out)
    assert result['plan'] == binding and result['claim_created'] is False
    assert all(result[key] == 0 for key in ('NN_forwards', 'autograd_calls', 'checkpoint_deserializations', 'NPZ_arrays_decoded', 'models'))
    assert result['CUDA_initialized'] is False and hooks == []
    assert not any(root.exists() for root in (m.ALL_ROOT, m.CPU_ROOT, m.BATCH_ROOT, m.PARITY_ROOT, m.ROOT))
    monkeypatch.setattr(sys, 'argv', sys.argv + ['--all-tf'])
    with pytest.raises(ValueError, match='EXACT_CLI'): m.main()
    monkeypatch.setattr(sys, 'argv', ['audit', '--first-parity-plan-sha256', binding['sha256'], '--cpu-conditional-gradient', '--preflight', '--all-tf'])
    with pytest.raises(SystemExit): m.main()


@pytest.mark.parametrize('failure', [False, True])
def test_all_tf_lifecycle_preserves_distinct_transfer_baseline_and_failed_original_identity(monkeypatch, tmp_path, failure):
    _, binding, pref, _, timers = all_tf_lifecycle(monkeypatch, tmp_path)
    def calculate(plan, *, progress):
        transfer = m.all_tf_publish('TRANSFER_PROOF.json', {'status': 'TRANSFER_PROOF_COMPLETE_BEFORE_FORWARD', 'NN_forwards': 0})
        baseline = m.all_tf_publish('CPU_BASELINE.json', {'status': 'ALL_TF_CPU_BASELINE_COMPLETE_BEFORE_DERIVATIVES', 'NN_forwards': 96, 'autograd_calls': 0})
        progress.update(transfer_proof=transfer, cpu_baseline=baseline, counts={'NN_forwards': 100, 'autograd_calls': 24})
        if failure: raise ValueError('SYNTHETIC_ALL_TF_LATE_FAILURE')
        return {'transfer_proof': transfer, 'cpu_baseline': baseline, 'original_saved_CUDA_gate': 'FAILED',
                'full96_gradient_qualification': False, 'no_candidate_fit_warmstart': True}
    monkeypatch.setattr(m, 'calculate_all_tf', calculate)
    if failure:
        with pytest.raises(ValueError, match='SYNTHETIC_ALL_TF_LATE_FAILURE'): m.run_all_tf(binding['sha256'])
    else:
        outputs = m.run_all_tf(binding['sha256']); assert outputs['result'] == m.ref(m.ALL_ROOT/'RESULT.json')
        result = json.loads((m.ALL_ROOT/'RESULT.json').read_text())
        assert result['report']['original_saved_CUDA_gate'] == 'FAILED'
        assert result['report']['full96_gradient_qualification'] is False and result['report']['no_candidate_fit_warmstart'] is True
    terminal = json.loads((m.ALL_ROOT/'TERMINAL.json').read_text()); claim = json.loads((m.ALL_ROOT/'CLAIM.json').read_text())
    assert terminal['status'] == ('FAILED_CONSUMED' if failure else 'COMPLETE') and terminal['exit_code'] == int(failure)
    assert terminal['stage'] == claim['stage'] == m.ALL_STAGE and terminal['plan'] == binding
    assert terminal['remaining_invocations'] == 0 and terminal['relaunch_allowed'] is claim['training_authorized'] is False
    assert claim['policy_at_claim'] == pref and (m.ALL_ROOT/'RESULT.json').exists() is (not failure)
    assert (m.ALL_ROOT/'TRANSFER_PROOF.json').is_file() and (m.ALL_ROOT/'CPU_BASELINE.json').is_file()
    if failure:
        assert terminal['result'] is None and terminal['progress']['counts']['autograd_calls'] == 24
        assert terminal['progress']['transfer_proof'] == m.ref(m.ALL_ROOT/'TRANSFER_PROOF.json')
    assert not any(root.exists() for root in (m.CPU_ROOT, m.BATCH_ROOT, m.PARITY_ROOT, m.ROOT))
    assert timers[-1] == (m.signal.ITIMER_REAL, 0)
    with pytest.raises(FileExistsError): m.run_all_tf(binding['sha256'])


@pytest.mark.parametrize('admitted', [False, True])
def test_all_tf_gate_is_read_only_one_use_and_never_fit_warmstart_authority(admitted):
    binding = {'path': '/synthetic/ALL_PLAN.json', 'sha256': 'e'*64}
    gate = m.gate_all_tf_expected(binding, admitted)
    assert gate['plan'] == binding and gate['execution_authorized'] is admitted and gate['remaining_invocations'] == int(admitted)
    assert gate['training_authorized'] is gate['test_authorized'] is gate['relaunch_allowed'] is False


def test_all_tf_inspection_keeps_all_four_predecessors_and_global_training_flags_closed(monkeypatch, tmp_path):
    planfile, policyfile = tmp_path/'ALL_PLAN.json', tmp_path/'POLICY.json'
    monkeypatch.setattr(m, 'ALL_PLAN', planfile); monkeypatch.setattr(m, 'POLICY', policyfile)
    monkeypatch.setattr(m, 'ALL_ROOT', tmp_path/'ALL_OUT'); monkeypatch.setattr(m, 'REPO', Path.cwd())
    predecessors = ((m.STAGE, 'failed_gradient_plan'), (m.PARITY_STAGE, 'first_parity_plan'),
                    (m.BATCH_STAGE, 'batch_geometry_plan'), (m.CPU_STAGE, 'predecessor_plan'))
    inputs = {role: {'path': '/synthetic/'+role+'.json', 'sha256': 'a'*64} for _, role in predecessors}
    plan = {'source_base_commit': '1'*40, 'inputs': inputs}; planfile.write_text(json.dumps(plan)); binding = m.ref(planfile)
    policy = dict.fromkeys(m.saved.FLAGS, False)
    for stage, role in predecessors:
        policy[stage] = {'plan': inputs[role], 'execution_authorized': False, 'remaining_invocations': 0,
                        'relaunch_allowed': False, 'training_authorized': False, 'test_authorized': False}
    policy[m.ALL_STAGE] = m.gate_all_tf_expected(binding, False)
    monkeypatch.setattr(m, 'build_all_tf_plan', lambda **k: plan); monkeypatch.setattr(m, 'source_clean', lambda value: True)
    def inspect():
        policyfile.write_text(json.dumps(policy)); return m.inspect_all_tf(binding['sha256'], admitted=False)
    assert inspect()[0] == plan
    for key in m.saved.FLAGS:
        policy[key] = True
        with pytest.raises(ValueError, match='ONE_USE_FLAGS'): inspect()
        policy[key] = False
    for stage, _ in predecessors:
        policy[stage]['remaining_invocations'] = 1
        with pytest.raises(ValueError, match='OLD_PLANS_CLOSED'): inspect()
        policy[stage]['remaining_invocations'] = 0
    policy[m.ALL_STAGE]['training_authorized'] = True
    with pytest.raises(ValueError, match='ONE_USE_FLAGS'): inspect()
    policy[m.ALL_STAGE]['training_authorized'] = False; m.ALL_ROOT.mkdir()
    with pytest.raises(ValueError, match='FRESH_ROOT'): inspect()


@pytest.mark.parametrize('case', ['buffer', 'parameter_alias'])
def test_all_tf_numeric_transfer_rejects_signed_zero_byte_difference_despite_numeric_equality(case):
    candidate, original, old_contract, new_contract = all_tf_transfer_fixture()
    if case == 'buffer':
        candidate.register_buffer('zero_buffer', torch.tensor([0.], dtype=torch.float32))
        original['zero_buffer'] = torch.tensor([-0.], dtype=torch.float32)
        left, right = candidate.zero_buffer, original['zero_buffer']
    else:
        original['parameter'] = torch.tensor([[0., 1.]], dtype=torch.float32)
        original['parameter_alias'] = torch.tensor([[-0., 1.]], dtype=torch.float32)
        left, right = original['parameter'], original['parameter_alias']
    assert torch.equal(left, right) is True
    assert m.core.state_sha(left) != m.core.state_sha(right)
    candidate_before, original_before = m.core.state_sha(candidate.state_dict()), m.core.state_sha(original)
    with pytest.raises(ValueError, match='TRANSFER_IMMUTABLE_BUFFER|TRANSFER_ALIAS_VALUES'):
        m.all_tf_numeric_transfer(candidate, original, original_contract=old_contract, candidate_contract=new_contract)
    assert m.core.state_sha(candidate.state_dict()) == candidate_before
    assert m.core.state_sha(original) == original_before


def test_all_tf_builder_keeps_external_baseline_reference_without_embedding_forecast_rows():
    tree = ast.parse(Path(m.__file__).read_text())
    builder = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'build_all_tf_plan')
    assignments = {target.id: node for node in builder.body if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Name) and target.id in ('inputs', 'plan')}
    assert set(assignments) == {'inputs', 'plan'}
    baseline_ref = {'path': '/synthetic/PRIVATE_PREDECESSOR_CPU_BASELINE.json', 'sha256': 'a'*64}
    binding = lambda name: {'path': '/synthetic/'+name+'.json', 'sha256': 'b'*64}
    baseline = {'status': 'CPU_BASELINE_COMPLETE_BEFORE_DERIVATIVES',
                'outputs': {'side_logits': {'values': [['DO_NOT_EMBED_PRIVATE_ROW_FORECASTS']]}}}
    scope = {'predecessor_CPU_baseline': baseline_ref, 'current_model_owner': binding('owner'),
             'question': 'Frozen same-point all-TF gradient contrast.', 'decision_before_run': 'No selected policy or causal history.',
             'numeric_transfer_contract': {'scope_limit': 'Diagnostic parameter transfer only.'}}
    base = {'inputs': {'scope': binding('CPU_scope'), 'capture': binding('capture')},
            'model_contract': {'contract_sha256': 'c'*64}, 'input_roles_not_JSON': ['capture']}
    environment = {'scope': scope, 'baseline': baseline, 'base': base,
        'accepted': {role: binding(role) for role in ('plan', 'result', 'terminal')},
        'candidate': {'contract_sha256': 'd'*64}, 'budget': {'NN_forwards_max': 192, 'autograd_calls_max': 768},
        'r': {'measured_predecessor': 'Predecessor capacity is not a new-gradient peak guarantee.'},
        **{name: getattr(m, name) for name in ('ALL_SCOPE', 'ALL_EVIDENCE', 'ALL_STAGE', 'ALL_PLAN', 'ALL_ROOT',
                                              'ALL_EXECUTION', 'ALL_PARENT', 'ALL_ROUTE')}}
    statements = ast.Module(body=[copy.deepcopy(assignments['inputs']), copy.deepcopy(assignments['plan'])], type_ignores=[])
    exec(compile(ast.fix_missing_locations(statements), '<actual_ALL_plan_reference_only>', 'exec'), environment)
    plan = environment['plan']
    assert plan['inputs']['predecessor_CPU_baseline'] == baseline_ref
    assert set(plan['inputs']['predecessor_CPU_baseline']) == {'path', 'sha256'}
    assert 'predecessor_CPU_baseline' not in plan and 'outputs' not in plan
    assert 'DO_NOT_EMBED_PRIVATE_ROW_FORECASTS' not in json.dumps(plan, sort_keys=True)
    assert plan['budget'] == {'NN_forwards_max': 192, 'autograd_calls_max': 768}
