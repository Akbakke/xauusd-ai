"""Synthetic readout-state and fixed-output algebra; no genuine payload or native model."""
import ast
import copy
import inspect as pyinspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from gx1.audit import diagnose_native_provider_hlc_readout_v1 as m

TF = ('M5', 'M15', 'H1', 'H4', 'D1')


def binding(name):
    return {'path': '/synthetic/' + name + '.json', 'sha256': 'a' * 64}


def head_state(biases, *, weight=0.):
    return {f'readout.side_heads.{tf}.{kind}':
            torch.full(shape, float(biases[j]) if kind == 'bias' else weight, dtype=torch.float32)
            for j, tf in enumerate(TF) for kind, shape in (('weight', (1, 128)), ('bias', (1,)))}


def heads(biases):
    return {tf: {'bias': float(biases[j]), 'weight_sha256': 'b' * 64, 'bias_sha256': 'c' * 64}
            for j, tf in enumerate(TF)}


@pytest.mark.parametrize('delta', [0., .25, -.25])
def test_logit_displacement_is_bias_delta_plus_combined_residual(delta):
    logits = np.array([[-4., -1., 0., 1., 3.], [2., .5, -2., 4., 0.]], np.float32)
    known = np.ones_like(logits, dtype=np.bool_)
    initial = np.array([-.2, -.1, 0., .1, .2], np.float32)
    final = initial.astype(np.float64) + delta
    report = m.decompose(logits, known, heads(initial), heads(final))
    for j, tf in enumerate(TF):
        cell = report[tf]
        assert cell['bias_initial'] == float(initial[j])
        assert cell['bias_final'] == float(final[j])
        assert cell['bias_delta'] == pytest.approx(delta, abs=1e-15)
        assert cell['uniform_bias_probability_effect_abs_bound'] == pytest.approx(abs(delta) / 4, abs=1e-15)
        for cohort in ('ALL', 'KNOWN_SIDE'):
            panel = cell[cohort]
            expected_mean = float(logits[:, j].astype(np.float64).mean())
            assert panel['count'] == 2
            assert panel['final_mean_logit'] == expected_mean
            assert panel['mean_logit_displacement'] == pytest.approx(expected_mean - float(initial[j]), abs=1e-15)
            assert panel['direct_bias_delta'] == pytest.approx(delta, abs=1e-15)
            assert panel['non_bias_residual_mean'] == pytest.approx(expected_mean - float(final[j]), abs=1e-15)
            assert panel['mean_logit_displacement'] == pytest.approx(panel['direct_bias_delta'] + panel['non_bias_residual_mean'], abs=1e-12)
            assert panel['identity_abs_error'] <= 1e-12
        # Only a signed-change magnitude bound is claimed, never a causal probability share.
        q = np.exp(-np.logaddexp(0., -logits[:, j].astype(np.float64)))
        fixed_residual_q = np.exp(-np.logaddexp(0., -(logits[:, j].astype(np.float64) - delta)))
        effect = q - fixed_residual_q
        assert np.max(np.abs(effect)) <= cell['uniform_bias_probability_effect_abs_bound'] + 1e-15
        if delta > 0: assert np.all(effect > 0)
        if delta < 0: assert np.all(effect < 0)


def test_mean_logit_is_not_logit_of_mean_probability():
    logits = np.repeat(np.array([[-4.], [2.]], np.float32), 5, axis=1)
    report = m.decompose(logits, np.ones_like(logits, dtype=np.bool_), heads(np.zeros(5)), heads(np.zeros(5)))
    q = np.exp(-np.logaddexp(0., -logits[:, 0].astype(np.float64)))
    for tf in TF:
        panel = report[tf]['ALL']
        assert panel['final_mean_logit'] == -1.
        assert panel['mean_probability_final'] == pytest.approx(float(q.mean()), abs=1e-15)
        assert not np.isclose(panel['final_mean_logit'], np.log(q.mean() / (1. - q.mean())))
        assert not np.isclose(panel['mean_probability_final'], 1. / (1. + np.exp(-panel['final_mean_logit'])))


def test_known_side_cohorts_remain_distinct_and_empty_support_is_unknown():
    logits = np.repeat(np.array([[-2.], [1.], [4.]], np.float32), 5, axis=1)
    known = np.zeros_like(logits, dtype=np.bool_); known[0, :4] = True
    report = m.decompose(logits, known, heads(np.zeros(5)), heads(np.ones(5)))
    assert report['M5']['ALL']['count'] == 3
    assert report['M5']['KNOWN_SIDE']['count'] == 1
    assert report['M5']['KNOWN_SIDE']['final_mean_logit'] == -2.
    assert report['D1']['KNOWN_SIDE']['count'] == 0
    assert report['D1']['KNOWN_SIDE']['final_mean_logit'] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('bad', ['logit_dtype', 'mask_dtype', 'mask_shape', 'logit_width', 'nonfinite', 'empty'])
def test_decomposition_rejects_invalid_fixed_output_or_mask(bad):
    logits = np.zeros((3, 5), np.float32); known = np.ones((3, 5), np.bool_)
    if bad == 'logit_dtype': logits = logits.astype(np.float64)
    elif bad == 'mask_dtype': known = known.astype(np.int64)
    elif bad == 'mask_shape': known = known[:2]
    elif bad == 'logit_width': logits = logits[:, :4]
    elif bad == 'nonfinite': logits[0, 0] = np.nan
    else: logits, known = logits[:0], known[:0]
    with pytest.raises(ValueError): m.decompose(logits, known, heads(np.zeros(5)), heads(np.zeros(5)))


def test_extract_heads_validates_actual_parameter_names_shapes_and_exact_initial_state():
    biases = np.array([-1., -.2, 0., .2, 1.], np.float32)
    state = head_state(biases)
    report = m.extract_heads(state, prior_bias=biases)
    assert set(report) == set(TF)
    for j, tf in enumerate(TF):
        assert report[tf]['bias'] == float(biases[j])
        assert len(report[tf]['weight_sha256']) == len(report[tf]['bias_sha256']) == 64
    assert all(torch.equal(state[f'readout.side_heads.{tf}.weight'], torch.zeros((1, 128))) for tf in TF)


@pytest.mark.parametrize('bad', ['missing', 'weight_shape', 'bias_shape', 'dtype', 'nan', 'requires_grad', 'initial_weight', 'initial_bias'])
def test_extract_heads_rejects_wrong_state_role_or_header(bad):
    biases = np.zeros(5, np.float32); state = head_state(biases)
    key = 'readout.side_heads.M5.bias'
    if bad == 'missing': del state[key]
    elif bad == 'weight_shape': state['readout.side_heads.M5.weight'] = torch.zeros((128,), dtype=torch.float32)
    elif bad == 'bias_shape': state[key] = torch.zeros((1, 1), dtype=torch.float32)
    elif bad == 'dtype': state[key] = state[key].double()
    elif bad == 'nan': state[key][0] = float('nan')
    elif bad == 'requires_grad': state[key].requires_grad_(True)
    elif bad == 'initial_weight': state['readout.side_heads.M5.weight'][0, 0] = 1.
    else: state[key][0] = .1
    with pytest.raises(ValueError): m.extract_heads(state, prior_bias=biases)


def synthetic_model_contract():
    return {'contract_sha256': 'f' * 64, 'target_contract_sha256': 'd' * 64,
            'objective_contract_sha256': '7' * 64, 'train_prior_sha256': '8' * 64,
            'readout': {'contract_sha256': '9' * 64, 'd_model': 128}}


def identity_buffers(contract):
    hashes = {'future_state_target_sha256': contract['target_contract_sha256'],
              'future_state_objective_sha256': contract['objective_contract_sha256'],
              'future_state_prior_sha256': contract['train_prior_sha256'],
              'future_state_readout_sha256': contract['readout']['contract_sha256']}
    return {'readout.' + key: torch.tensor(list(bytes.fromhex(value)), dtype=torch.uint8)
            for key, value in hashes.items()}


def initial_fixture(monkeypatch):
    prior = np.array([-.5, -.25, 0., .25, .5], np.float32)
    contract = synthetic_model_contract()
    state = {**head_state(prior), **identity_buffers(contract)}; runtime = {'synthetic': 'runtime'}
    meta = {'input_binding': {'synthetic': True}, 'normalization': binding('normalization'),
            'prior': binding('prior'), 'dataset': binding('dataset'),
            'target_contract': {'contract_sha256': 'd' * 64}, 'truth_sha256': 'e' * 64,
            'runtime': runtime}
    plan = {'model_contract': contract, 'metadata': meta, 'recipe': {'seed': 20260911}}
    planref = binding('initial_plan')
    cp = {'schema_version': 'gx1_native_provider_hlc_fresh_initial_checkpoint_v1', 'plan': planref,
          'model_contract': plan['model_contract'], 'input_binding': meta['input_binding'],
          'normalization': meta['normalization'], 'prior': meta['prior'], 'dataset': meta['dataset'],
          'target_contract': meta['target_contract'], 'truth_sha256': meta['truth_sha256'],
          'seed': plan['recipe']['seed'], 'runtime': runtime, 'model_state': state,
          'model_state_sha256': m.core.state_sha(state), 'rng_state': {'synthetic': 'RNG'},
          'training': True, 'optimizer': None, 'optimizer_updates': 0,
          'authority': 'INITIAL_ONLY_NO_FIT_OR_RESUME_AUTHORITY'}
    receipt = {'schema_version': cp['schema_version'], 'plan': planref,
               'checkpoint': {'path': '/synthetic/INITIAL.pt', 'sha256': '1' * 64},
               'checkpoint_tree_sha256': m.core.state_sha(cp),
               'model_state_sha256': cp['model_state_sha256'],
               'rng_sha256': m.core.state_sha(cp['rng_state']),
               'model_contract_sha256': plan['model_contract']['contract_sha256'],
               'target_contract_sha256': meta['target_contract']['contract_sha256'],
               'optimizer_updates': 0, 'initial_only': True}
    monkeypatch.setattr(m.core, 'future_cuda_runtime_matches', lambda a, b: m.need(a == b, 'SYNTHETIC_RUNTIME'))
    return cp, receipt, plan, planref, prior


def test_initial_validator_authenticates_distinct_full_tree_model_rng_and_prior(monkeypatch):
    cp, receipt, plan, planref, prior = initial_fixture(monkeypatch)
    before = m.core.state_sha(cp)
    result = m.validate_initial(cp, receipt, plan, planref, prior)
    assert set(result) == set(TF) and m.core.state_sha(cp) == before


@pytest.mark.parametrize('bad', ['extra', 'plan', 'model_hash', 'rng', 'optimizer', 'updates', 'authority', 'tensor'])
def test_initial_validator_rejects_corrupted_or_training_checkpoint_roles(monkeypatch, bad):
    cp, receipt, plan, planref, prior = initial_fixture(monkeypatch)
    if bad == 'extra': cp['unexpected'] = True
    elif bad == 'plan': cp['plan'] = binding('wrong')
    elif bad == 'model_hash': cp['model_state_sha256'] = '0' * 64
    elif bad == 'rng': cp['rng_state'] = {'foreign': True}
    elif bad == 'optimizer': cp['optimizer'] = {'state': {}}
    elif bad == 'updates': cp['optimizer_updates'] = 1
    elif bad == 'authority': cp['authority'] = 'TRAINING'
    else: cp['model_state']['readout.side_heads.M5.bias'][0] = 1.
    with pytest.raises(ValueError): m.validate_initial(cp, receipt, plan, planref, prior)


def final_fixture():
    schedule = {'logical_batches': 3694, 'batches_per_pass': 3694, 'update_flags': [True] * 3694}
    contract = synthetic_model_contract()
    original = {'schedule': schedule, 'model_contract': contract}
    originalref, executionref = binding('original_epoch'), binding('resume004')
    trajectory = {'schedule': schedule, 'contract_sha256': '2' * 64}
    runtime = {'synthetic': 'final_runtime'}
    cp = {'schema_version': 'gx1_native_future_state_training_checkpoint_v1', 'trajectory': trajectory,
          'producer': {'original_plan': originalref, 'lane': 'main', 'runtime': runtime},
          'position': m.train.position(schedule, 3694), 'model_state': {**head_state(np.ones(5, np.float32), weight=.01), **identity_buffers(contract)},
          'optimizer_state': {'opaque': 'not analysed'}, 'scheduler_state': None,
          'rng_state': {'opaque': 'not restored'}, 'training': True,
          'optimizer_parameter_names': [f'readout.side_heads.{tf}.{kind}' for tf in TF for kind in ('weight', 'bias')],
          'steps': [{'logical_step': j + 1} for j in range(3694)], 'scores': {'PASS1': {'synthetic': True}}}
    receipt = {'schema_version': 'gx1_native_future_state_training_checkpoint_receipt_v1',
               'checkpoint': {'path': '/synthetic/FINAL.pt', 'sha256': '3' * 64},
               'original_plan': originalref, 'execution_plan': executionref, 'producer_lane': 'main',
               'producer_runtime': runtime, 'logical_cursor': 3694, 'optimizer_updates': 3694,
               'trajectory_sha256': trajectory['contract_sha256'], 'proof': m.train.checkpoint_proof(cp)}
    return cp, receipt, original, originalref, trajectory, executionref


def test_final_validator_authenticates_all_ten_components_without_restoration():
    args = final_fixture(); before = m.train.checkpoint_proof(args[0])
    result = m.validate_final(*args)
    assert set(result) == set(TF) and m.train.checkpoint_proof(args[0]) == before


@pytest.mark.parametrize('bad', ['extra', 'trajectory', 'original', 'execution', 'cursor', 'producer', 'steps', 'proof', 'tensor'])
def test_final_validator_rejects_wrong_provenance_cursor_or_proof(bad):
    cp, receipt, original, originalref, trajectory, executionref = final_fixture()
    if bad == 'extra': cp['unexpected'] = True
    elif bad == 'trajectory': cp['trajectory'] = {}
    elif bad == 'original': receipt['original_plan'] = executionref
    elif bad == 'execution': receipt['execution_plan'] = originalref
    elif bad == 'cursor': receipt['logical_cursor'] = 3328
    elif bad == 'producer': cp['producer']['lane'] = 'wrong'
    elif bad == 'steps': cp['steps'].pop()
    elif bad == 'proof': receipt['proof']['optimizer_state'] = '0' * 64
    else: cp['model_state']['readout.side_heads.D1.bias'][0] = 2.
    with pytest.raises(ValueError): m.validate_final(cp, receipt, original, originalref, trajectory, executionref)


def test_owner_uses_only_explicit_CPU_weights_only_loads_without_restore_or_models():
    tree = ast.parse(Path(m.__file__).read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    loads = [node for node in calls if isinstance(node.func, ast.Attribute)
             and isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch'
             and node.func.attr == 'load']
    assert loads
    for call in loads:
        options = {kw.arg: kw.value for kw in call.keywords}
        assert ast.literal_eval(options['map_location']) == 'cpu'
        assert ast.literal_eval(options['weights_only']) is True
    forbidden = {'strict_restore', 'load_state_dict', 'build_native_future_state_model',
                 'restore_training_rng', 'AdamW', 'backward', 'add_safe_globals'}
    assert not any(isinstance(call.func, ast.Attribute) and call.func.attr in forbidden for call in calls)
    assert not any(isinstance(call.func, ast.Name) and call.func.id in forbidden for call in calls)


def lifecycle(monkeypatch, tmp_path):
    monkeypatch.setattr(m, 'ROOT', tmp_path / 'OUT')
    monkeypatch.setattr(m, 'PLAN', tmp_path / 'PLAN.json')
    monkeypatch.setattr(m, 'POLICY', tmp_path / 'POLICY.json')
    m.PLAN.write_text('{}'); m.POLICY.write_text('{}')
    pref, planref = m.ref(m.POLICY), m.ref(m.PLAN)
    roles = {'initial_checkpoint': 'INITIAL.pt', 'final_checkpoint': 'FINAL.pt',
             'capture': 'CAPTURE.npz', 'target_arrays': 'TARGETS.npz'}
    plan = {'inputs': {key: {'path': str(tmp_path / name), 'sha256': 'a' * 64}
                       for key, name in roles.items()}}
    monkeypatch.setattr(m, 'inspect', lambda *a, **k: (plan, planref, pref))
    monkeypatch.setattr(m, 'checked_inputs', lambda plan: None)
    monkeypatch.setattr(m, 'source_clean', lambda plan: True)
    monkeypatch.setattr(m, 'require_capped_cpu_audit_execution', lambda: {
        'memory_max': 4*1024**3, 'memory_high': 4*1024**3, 'swap': 512*1024**2,
        'numerical_threads': 1, 'cgroup_path': '/sys/fs/cgroup/synthetic'})
    monkeypatch.setattr(m, 'require_capped_lock_ancestry', lambda: {'scope': '/synthetic'})
    hooks, timers = [], []
    monkeypatch.setattr(sys, 'addaudithook', hooks.append)
    monkeypatch.setattr(m.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(m.signal, 'setitimer', lambda *a: timers.append(a))
    monkeypatch.setattr(m, 'calculate', lambda *a, **k: {'synthetic': True})
    return plan, planref, pref, hooks, timers


def test_preflight_exact_CLI_has_zero_decodes_and_no_claim(monkeypatch, tmp_path, capsys):
    _, planref, _, hooks, _ = lifecycle(monkeypatch, tmp_path)
    monkeypatch.setattr(m, 'calculate', lambda *a, **k: (_ for _ in ()).throw(AssertionError('NO_CALCULATE')))
    monkeypatch.setattr(torch, 'load', lambda *a, **k: (_ for _ in ()).throw(AssertionError('NO_CP_LOAD')))
    monkeypatch.setattr(np, 'load', lambda *a, **k: (_ for _ in ()).throw(AssertionError('NO_NPZ_LOAD')))
    monkeypatch.setattr(sys, 'argv', ['audit', '--plan-sha256', planref['sha256'], '--preflight'])
    m.main(); result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'PREFLIGHT_PROVIDER_HLC_READOUT_PASS'
    assert result['plan'] == planref and result['claim_created'] is False
    assert result['checkpoint_deserializations'] == result['NPZ_arrays_decoded'] == result['models'] == 0
    assert result['CUDA_initialized'] is False and hooks == [] and not m.ROOT.exists()
    monkeypatch.setattr(sys, 'argv', sys.argv + ['--preflight'])
    with pytest.raises(ValueError, match='EXACT_CLI'): m.main()


@pytest.mark.parametrize('failure', [None, 'calculate', 'bookend'])
def test_success_and_consumed_failure_reports_preserve_outputs(monkeypatch, tmp_path, failure):
    _, planref, pref, _, timers = lifecycle(monkeypatch, tmp_path)
    def calculate(*args, **kwargs):
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
        assert result['status'] == 'COMPLETE_CONDITIONAL_PROVIDER_HLC_READOUT_ATTRIBUTION'
        assert result['report'] == {'synthetic': True}
        assert result['test_data_used'] is result['economic_edge_proven'] is False
    terminal = json.loads((m.ROOT / 'TERMINAL.json').read_text())
    claim = json.loads((m.ROOT / 'CLAIM.json').read_text())
    assert terminal['status'] == ('FAILED_CONSUMED' if failure else 'COMPLETE')
    assert terminal['exit_code'] == int(bool(failure)) and terminal['plan'] == planref
    assert terminal['remaining_invocations'] == 0 and terminal['relaunch_allowed'] is False
    assert claim['policy_at_claim'] == pref and claim['training_authorized'] is False
    assert (m.ROOT / 'RESULT.json').exists() is (failure is None)
    assert (m.ROOT / 'PRESERVED.json').is_file()
    assert timers[-1] == (m.signal.ITIMER_REAL, 0)


def test_runtime_audit_allows_only_exact_two_CP_two_NPZ_and_no_network(monkeypatch, tmp_path):
    plan, planref, _, hooks, _ = lifecycle(monkeypatch, tmp_path)
    m.run(planref['sha256']); assert len(hooks) == 1
    for binding in plan['inputs'].values(): hooks[0]('open', (binding['path'],))
    for name in ('unbound.pt', 'FEATURE_INPUTS.npz', 'foreign.parquet', 'other.pkl'):
        with pytest.raises(RuntimeError, match='PAYLOAD_FORBIDDEN'):
            hooks[0]('open', (str(tmp_path / name),))
    for event in ('socket.connect', 'socket.getaddrinfo', 'urllib.Request'):
        with pytest.raises(RuntimeError, match='NETWORK_FORBIDDEN'): hooks[0](event, ())


@pytest.mark.parametrize('admitted', [False, True])
def test_exact_one_use_gate_preserves_no_training_test_or_relaunch(admitted):
    ref = binding('readout_plan'); gate = m.gate_expected(ref, admitted)
    assert gate['plan'] == ref and gate['remaining_invocations'] == int(admitted)
    assert gate['execution_authorized'] is admitted
    assert gate['training_authorized'] is gate['test_authorized'] is gate['relaunch_allowed'] is False


def test_inspect_rejects_every_global_training_flag_and_scope_training_flag(monkeypatch, tmp_path):
    planfile, policyfile, output = tmp_path / 'PLAN.json', tmp_path / 'POLICY.json', tmp_path / 'OUT'
    monkeypatch.setattr(m, 'PLAN', planfile); monkeypatch.setattr(m, 'POLICY', policyfile)
    monkeypatch.setattr(m, 'ROOT', output); monkeypatch.setattr(m, 'REPO', Path.cwd())
    plan = {'source_base_commit': '1' * 40}; planfile.write_text(json.dumps(plan)); planref = m.ref(planfile)
    flags = ('training_enabled', 'native_training_enabled', 'native_launch_authorized',
             'full_epoch_training_allowed', 'full_val_allowed')
    policy = dict.fromkeys(flags, False); policy[m.STAGE] = m.gate_expected(planref, False)
    monkeypatch.setattr(m, 'build_plan', lambda **kwargs: plan)
    monkeypatch.setattr(m, 'source_clean', lambda value: True)
    def inspect():
        policyfile.write_text(json.dumps(policy)); return m.inspect(planref['sha256'], admitted=False)
    assert inspect()[0] == plan and not output.exists()
    for key in flags:
        policy[key] = True
        with pytest.raises(ValueError, match='ONE_USE_ALL_TRAINING_CLOSED'): inspect()
        policy[key] = False
    policy[m.STAGE]['training_authorized'] = True
    with pytest.raises(ValueError, match='ONE_USE_ALL_TRAINING_CLOSED'): inspect()


def test_calculate_loads_two_whole_CP_and_four_declared_arrays_without_models(monkeypatch):
    n = 3; monkeypatch.setattr(m, 'ROWS', n)
    issues = np.arange(1, n + 1, dtype=np.int64)
    logits = np.array([[-1., 0., 1., 2., 3.], [.5, -.5, 0., 1., -1.], [2., 1., -.5, 0., .5]], np.float32)
    known = np.ones((n, 5), np.bool_)
    capture = {'issue_ns': issues, 'side_logits': logits}
    mask = {'issue_ns': issues.copy(), 'side_known': known}
    priors = np.zeros(5, np.float32)
    prior = {'per_task': {'side': {tf: {'count': n, 'mask_sha256': m.saved.array_sha(known[:, j])}
                                  for j, tf in enumerate(TF)}}}
    q = np.exp(-np.logaddexp(0., -logits.astype(np.float64))).mean(0)
    quality = {'all_TF': {tf: {'side': {'count': n, 'model': {'mean_positive_probability': float(q[j])}}}
                          for j, tf in enumerate(TF)}}
    batch = {'report': {'per_tf': {tf: {'FINAL_mean_q': float(q[j]), 'prior_q': .5}
                                  for j, tf in enumerate(TF)}}}
    inputs = {key: binding(key) for key in ('prior', 'initial_plan', 'original_epoch_plan',
               'completed_model_plan', 'full_quality', 'batch_risk_result')}
    inputs.update(initial_checkpoint={'path': '/synthetic/INITIAL.pt', 'sha256': 'a' * 64},
                  final_checkpoint={'path': '/synthetic/FINAL.pt', 'sha256': 'a' * 64},
                  capture={'path': '/synthetic/capture.npz', 'sha256': 'a' * 64},
                  target_arrays={'path': '/synthetic/targets.npz', 'sha256': 'a' * 64})
    plan = {'inputs': inputs, 'target_contract': {}, 'metadata': {'initial_receipt': {}, 'initial_plan': {},
             'final_receipt': {}, 'original_epoch_plan': {}, 'trajectory': {}},
            'budget': {'NPZ_ndarray_bytes_decoded': sum(a.nbytes for a in [*capture.values(), *mask.values()])},
            'TRAIN_issue_ns_sha256': m.saved.array_sha(issues), 'capture_array_headers': {},
            'target_array_headers': {}, 'limits': [], 'conditional_scope': {}}
    values = {inputs['prior']['path']: prior, inputs['full_quality']['path']: quality,
              inputs['batch_risk_result']['path']: batch}
    monkeypatch.setattr(m, 'read', lambda ref: values[ref['path']])
    monkeypatch.setattr(m, 'reference', lambda ref: Path(ref['path']))
    monkeypatch.setattr(m.objective, 'require_native_future_state_train_prior', lambda value, **k: value)
    monkeypatch.setattr(m.objective, 'native_future_state_train_prior_logits',
                        lambda *a, **k: {'side_logits': torch.from_numpy(priors[None, :])})
    loads, validations, archive_loads = [], [], []
    def load(path, **kwargs):
        loads.append((path, kwargs)); return {'synthetic_file': path}
    monkeypatch.setattr(torch, 'load', load)
    monkeypatch.setattr(torch, 'set_num_threads', lambda value: None)
    monkeypatch.setattr(torch, 'get_num_threads', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    def initial(cp, *args):
        validations.append(('initial', cp['synthetic_file'])); return heads(np.zeros(5))
    def final(cp, *args):
        validations.append(('final', cp['synthetic_file'])); return heads(np.full(5, .2))
    monkeypatch.setattr(m, 'validate_initial', initial); monkeypatch.setattr(m, 'validate_final', final)
    def fields(ref, selected, headers):
        archive_loads.append((ref['path'], selected))
        return capture if ref == inputs['capture'] else mask
    monkeypatch.setattr(m.saved, 'load_fields', fields)
    progress = {}; report = m.calculate(plan, progress=progress)
    assert loads == [(inputs[role]['path'], {'map_location': 'cpu', 'weights_only': True})
                     for role in ('initial_checkpoint', 'final_checkpoint')]
    assert validations == [('initial', inputs['initial_checkpoint']['path']), ('final', inputs['final_checkpoint']['path'])]
    assert archive_loads == [(inputs['capture']['path'], ('issue_ns', 'side_logits')),
                             (inputs['target_arrays']['path'], ('issue_ns', 'side_known'))]
    assert report['decoder_counts'] == {'checkpoint_deserializations': 2, 'capture_archive_loads': 1, 'target_archive_loads': 1}
    assert report['NN_forwards'] == report['model_objects_constructed'] == report['optimizer_updates'] == 0
    assert report['parameter_gradient_calculations'] == report['RNG_restore_calls'] == report['model_state_install_calls'] == 0
    assert report['actual_training_cause'] == report['selected_causal_fix'] == 'UNKNOWN'
    assert progress['phase'] == 'COMPLETE_STORED_LOGIT_ATTRIBUTION'

# Additive None stored-step support and terminal AdamW audit; old cases above preserved.
def stored_support_fixture():
    recipe = {'passes': 1, 'train_batch_rows': 96, 'eval_batch_rows': 256}
    schedule = {'rows': 354550, 'batch_rows': 96, 'logical_batches': 3694,
                'batches_per_pass': 3694, 'optimizer_updates': 3694, 'passes': 1,
                'update_flags': [True] * 3694, 'skipped_batches': 0,
                'shuffle': False, 'drop_last': False}
    steps = []
    for index in range(3694):
        rows = 22 if index == 3693 else 96
        activity = [rows] * 5
        side = [1, 0, int(index == 0), rows, int(index == 3693)]
        if index in (0, 2, 3, 3693): side[0] = 0
        if index == 4: side[0] = 2
        if index in (1, 2): activity[4] = 0
        steps.append({'logical_step': index + 1, 'skip_update': False,
                      'support': {'activity': activity, 'side': side}})
    quality = {'rows': 354550, 'all_TF': {tf: {task: {'count': sum(step['support'][task][j] for step in steps)}
               for task in ('activity', 'side')} for j, tf in enumerate(TF)}}
    return steps, recipe, schedule, quality


def test_stored_support_reports_exact_leading_internal_trailing_and_all_zero_spans():
    steps, recipe, schedule, quality = stored_support_fixture()
    before = m.core.state_sha((steps, recipe, schedule, quality))
    report = m.summarize_step_support(steps, recipe, schedule, quality)
    assert m.core.state_sha((steps, recipe, schedule, quality)) == before
    panels = report['per_TF']
    cell = panels['M5']['side']
    assert cell['known_cells'] == 3691 and cell['zero_support_steps'] == 4
    assert cell['zero_support_spans'] == [
        {'first_step': 1, 'last_step': 1, 'start_issue_row': 0, 'stop_issue_row': 96, 'steps': 1, 'issue_rows': 96},
        {'first_step': 3, 'last_step': 4, 'start_issue_row': 192, 'stop_issue_row': 384, 'steps': 2, 'issue_rows': 192},
        {'first_step': 3694, 'last_step': 3694, 'start_issue_row': 354528, 'stop_issue_row': 354550, 'steps': 1, 'issue_rows': 22}]
    assert cell['last_nonzero_support_step'] == 3693
    assert cell['trailing_zero_support_steps'] == 1 and cell['trailing_zero_support_issue_rows'] == 22
    assert cell['max_zero_support_steps'] == 2 and cell['max_zero_support_issue_rows'] == 192
    empty = panels['M15']['side']
    assert empty['known_cells'] == 0 and empty['last_nonzero_support_step'] is None
    assert empty['last_nonzero_support_step_issue_rows'] is None
    assert empty['zero_support_steps'] == empty['trailing_zero_support_steps'] == empty['max_zero_support_steps'] == 3694
    assert empty['trailing_zero_support_issue_rows'] == empty['max_zero_support_issue_rows'] == 354550
    assert empty['zero_support_spans'] == [
        {'first_step': 1, 'last_step': 3694, 'start_issue_row': 0, 'stop_issue_row': 354550, 'steps': 3694, 'issue_rows': 354550}]
    first = panels['H1']['side']
    assert first['known_cells'] == 1 and first['last_nonzero_support_step'] == 1
    assert first['trailing_zero_support_steps'] == 3693 and first['trailing_zero_support_issue_rows'] == 354454
    full = panels['H4']['side']
    assert full['known_cells'] == 354550 and full['last_nonzero_support_step'] == 3694
    assert full['zero_support_spans'] == [] and full['trailing_zero_support_issue_rows'] == 0
    final_only = panels['D1']['side']
    assert final_only['known_cells'] == 1 and final_only['last_nonzero_support_step'] == 3694
    assert final_only['max_zero_support_steps'] == 3693 and final_only['max_zero_support_issue_rows'] == 354528
    assert panels['D1']['activity']['known_cells'] == 354358
    for tf in TF:
        for task in ('activity', 'side'):
            assert panels[tf][task]['known_cells'] == quality['all_TF'][tf][task]['count']
    json.dumps(report, allow_nan=False)


def stored_moment_checkpoint_fixture():
    from tests.test_diagnose_native_future_state_alignment_v1 import moment_toy_options
    cp, receipt, original, originalref, trajectory, executionref = final_fixture()
    steps, recipe, schedule, quality = stored_support_fixture()
    original['schedule'].update(schedule); original['recipe'] = {**recipe, 'optimizer': moment_toy_options()}
    cp['steps'] = steps
    names = list(cp['optimizer_parameter_names']); ids = []
    slots = {}
    for index, name in enumerate(names):
        tf_index = index // 2
        parameter_id = 101 + 2 * tf_index if name.endswith('.weight') else 5 + 3 * tf_index
        ids.append(parameter_id)
        parameter = cp['model_state'][name]
        first_moment = .25 if name.endswith('.weight') else -.5
        slots[parameter_id] = {'step': torch.tensor(3694., dtype=torch.float32),
                               'exp_avg': torch.full_like(parameter, first_moment),
                               'exp_avg_sq': torch.ones_like(parameter)}
    cp['optimizer_state'] = {'state': slots, 'param_groups': [{**copy.deepcopy(original['recipe']['optimizer']), 'params': ids}]}
    receipt['proof'] = m.train.checkpoint_proof(cp)
    return cp, receipt, original, originalref, trajectory, executionref, quality


def test_stored_moments_keep_actual_opaque_ids_original_formula_and_all_ten_proofs(monkeypatch):
    cp, receipt, original, *_ = stored_moment_checkpoint_fixture()
    before = m.train.checkpoint_proof(cp)
    before_rng = m.core.state_sha(m.support_rng_proof())
    calls = []
    actual = m.alignment.moment_zero_gradient_delta
    def original_only(parameter, first, second, step, options, **kwargs):
        assert kwargs.get('reset_first_moment', False) is False
        calls.append((parameter.copy(), first.copy(), second.copy(), step, copy.deepcopy(options)))
        return actual(parameter, first, second, step, options, **kwargs)
    monkeypatch.setattr(m.alignment, 'moment_zero_gradient_delta', original_only)
    heads = m.alignment.moment_extract_heads(cp, original['recipe']['optimizer'])
    report = m.support_moment_summary(heads)
    assert report['formula_arms'] == 1 and report['parameter_formula_evaluations'] == len(calls) == 10
    assert report['reset_first_moment'] is False and report['actual_updates'] == 0
    assert report['per_TF']['M5']['parameter_binding']['parameter_ids'] == {'weight': 101, 'bias': 5}
    assert report['per_TF']['D1']['parameter_binding']['parameter_ids'] == {'weight': 109, 'bias': 17}
    for tf in TF:
        panel = report['per_TF'][tf]
        assert panel['parameter_binding']['step'] == 3694
        effect = panel['one_hypothetical_original_momentum_zero_gradient']['bias']['bias_decomposition']
        assert effect['observed_bias'] == 1.
        assert effect['moment_delta'] > 0 and effect['decay_delta'] < 0 and effect['total_delta'] > 0
        assert effect['identity_abs_error'] <= 1e-15
    assert all(call[3] == 3694 and call[4] == original['recipe']['optimizer'] for call in calls)
    assert m.train.checkpoint_proof(cp) == before == receipt['proof']
    assert m.core.state_sha(m.support_rng_proof()) == before_rng
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('bad,gate', [
    ('length', 'SUPPORT_STEP_CLOCK'), ('batch', 'SUPPORT_STEP_CLOCK'), ('rows', 'SUPPORT_STEP_CLOCK'),
    ('sequence', 'SUPPORT_STEP_RECORD'), ('bool_step', 'SUPPORT_STEP_RECORD'), ('skip', 'SUPPORT_STEP_RECORD'),
    ('support_keys', 'SUPPORT_STEP_RECORD'), ('width', 'SUPPORT_STEP_COUNTS'),
    ('float_count', 'SUPPORT_STEP_COUNTS'), ('bool_count', 'SUPPORT_STEP_COUNTS'),
    ('negative_count', 'SUPPORT_STEP_COUNTS'), ('tail23', 'SUPPORT_STEP_COUNTS'),
    ('side_above_activity', 'SUPPORT_STEP_COUNTS'), ('canonical_total', 'SUPPORT_CANONICAL_COUNTS'),
    ('canonical_float', 'SUPPORT_CANONICAL_COUNTS')])
def test_stored_support_rejects_the_intended_record_or_canonical_gate(bad, gate):
    steps, recipe, schedule, quality = stored_support_fixture()
    assert m.summarize_step_support(steps, recipe, schedule, quality)['per_TF']['M5']['side']['known_cells'] == 3691
    if bad == 'length': steps.pop()
    elif bad == 'batch': recipe['train_batch_rows'] = 95
    elif bad == 'rows': schedule['rows'] = 354551
    elif bad == 'sequence': steps[2]['logical_step'] = 2
    elif bad == 'bool_step': steps[0]['logical_step'] = True
    elif bad == 'skip': steps[0]['skip_update'] = True
    elif bad == 'support_keys': steps[0]['support']['foreign'] = [0] * 5
    elif bad == 'width': steps[0]['support']['side'].pop()
    elif bad == 'float_count': steps[1]['support']['side'][0] = 1.
    elif bad == 'bool_count': steps[1]['support']['side'][0] = True
    elif bad == 'negative_count': steps[1]['support']['side'][0] = -1
    elif bad == 'tail23': steps[-1]['support']['activity'][0] = 23
    elif bad == 'side_above_activity': steps[1]['support']['side'][4] = 1
    elif bad == 'canonical_total': quality['all_TF']['M5']['side']['count'] += 1
    else: quality['all_TF']['M5']['side']['count'] = 3691.
    with pytest.raises(ValueError, match=gate + '$'):
        m.summarize_step_support(steps, recipe, schedule, quality)


@pytest.mark.parametrize('bad,gate', [
    ('missing_TF', 'SUPPORT_MOMENT_HEADS'), ('bool_step', 'SUPPORT_MOMENT_TERMINAL_STEP'),
    ('wrong_step', 'SUPPORT_MOMENT_TERMINAL_STEP'), ('dtype', 'SUPPORT_MOMENT_SHAPES'),
    ('shape', 'SUPPORT_MOMENT_SHAPES'), ('nonfinite', 'SUPPORT_MOMENT_SHAPES'),
    ('negative_second', 'SUPPORT_MOMENT_SHAPES'), ('invalid_options', 'MOMENT_OPTIONS')])
def test_stored_moment_summary_rejects_exact_terminal_or_formula_gate(bad, gate):
    cp, _, original, *_ = stored_moment_checkpoint_fixture()
    heads = m.alignment.moment_extract_heads(cp, original['recipe']['optimizer'])
    assert m.support_moment_summary(heads)['parameter_formula_evaluations'] == 10
    if bad == 'missing_TF': del heads['D1']
    elif bad == 'bool_step': heads['M5']['step'] = True
    elif bad == 'wrong_step': heads['M5']['step'] = 3693
    elif bad == 'dtype': heads['M5']['weight'] = heads['M5']['weight'].astype(np.float32)
    elif bad == 'shape': heads['M5']['bias'] = np.zeros(2, np.float64)
    elif bad == 'nonfinite': heads['M5']['weight_exp_avg'][0] = np.nan
    elif bad == 'negative_second': heads['M5']['bias_exp_avg_sq'][0] = -1.
    else: heads['M5']['options']['eps'] = 0.
    with pytest.raises(ValueError, match=gate + '$'):
        m.support_moment_summary(heads)


@pytest.mark.parametrize('bad,gate', [('duplicate_ID', 'MOMENT_PARAMETER_ID_ORDER'),
                                     ('bool_ID', 'MOMENT_PARAMETER_ID_ORDER'),
                                     ('options', 'MOMENT_ACTUAL_RECIPE_amsgrad'),
                                     ('missing_slot_field', 'MOMENT_CPU_TENSORS')])
def test_stored_actual_parameter_binding_uses_authentic_ids_and_options(bad, gate):
    cp, _, original, *_ = stored_moment_checkpoint_fixture()
    recipe = original['recipe']['optimizer']
    assert m.alignment.moment_extract_heads(cp, recipe)['M5']['parameter_ids']['weight'] == 101
    group = cp['optimizer_state']['param_groups'][0]
    if bad == 'duplicate_ID': group['params'][1] = group['params'][0]
    elif bad == 'bool_ID': group['params'][0] = True
    elif bad == 'options': group['amsgrad'] = True
    else: del cp['optimizer_state']['state'][101]['exp_avg']
    with pytest.raises(ValueError, match=gate + '$'):
        m.alignment.moment_extract_heads(cp, recipe)


def test_stored_missing_moment_is_rejected_at_the_exact_actual_id_lookup():
    cp, _, original, *_ = stored_moment_checkpoint_fixture()
    assert m.alignment.moment_extract_heads(cp, original['recipe']['optimizer'])['M5']['parameter_ids']['weight'] == 101
    del cp['optimizer_state']['state'][101]
    with pytest.raises(KeyError, match='^101$'):
        m.alignment.moment_extract_heads(cp, original['recipe']['optimizer'])


def stored_calculation_fixture(monkeypatch):
    cp, receipt, original, originalref, trajectory, executionref, quality = stored_moment_checkpoint_fixture()
    plan = {'inputs': {'final_checkpoint': receipt['checkpoint'], 'original_epoch_plan': originalref,
                       'completed_model_plan': executionref},
            'metadata': {'original_epoch_plan': original, 'final_receipt': receipt,
                         'trajectory': trajectory, 'full_quality': quality},
            'budget': copy.deepcopy(m.SUPPORT_BUDGET), 'conditional_scope': {'synthetic_stored_facts_only': True}}
    loads = []
    def load(path, **kwargs):
        loads.append((path, kwargs)); return cp
    monkeypatch.setattr(torch, 'load', load)
    monkeypatch.setattr(m, 'reference', lambda value: Path(value['path']))
    def no_payload(*args, **kwargs): pytest.fail('NPZ/native payload forbidden')
    monkeypatch.setattr(np, 'load', no_payload)
    monkeypatch.setattr(m.saved, 'load_fields', no_payload)
    return plan, cp, receipt, loads


def test_stored_calculation_decodes_only_one_cpu_CP_and_preserves_proofs_rng_and_unknowns(monkeypatch):
    plan, cp, receipt, loads = stored_calculation_fixture(monkeypatch)
    proofs = m.train.checkpoint_proof(cp); rng = m.support_rng_proof(); progress = {}
    report = m.support_calculate(plan, progress=progress)
    assert loads == [(plan['inputs']['final_checkpoint']['path'], {'map_location': 'cpu', 'weights_only': True})]
    assert report['checkpoint_deserializations'] == report['payload_budget']['checkpoint_deserializations'] == 1
    assert report['payload_budget'] == m.SUPPORT_BUDGET
    for key in ('model_objects_constructed', 'NN_forwards', 'NPZ_arrays_decoded', 'autograd_calls', 'backward_calls',
                'optimizer_updates', 'model_state_install_calls', 'RNG_restore_calls', 'runtime_blocked_attempts'):
        assert report[key] == 0
    assert report['actual_training_cause'] == report['selected_causal_fix'] == 'UNKNOWN'
    assert report['later_generalization'] == report['economic_edge'] == 'UNKNOWN' and report['all20_approved'] is False
    assert report['all_ten_checkpoint_proofs_unchanged'] is report['process_Python_NumPy_Torch_CPU_RNG_unchanged'] is True
    assert m.train.checkpoint_proof(cp) == proofs == receipt['proof'] and m.support_rng_proof() == rng
    support = report['step_support']
    assert support['steps'] == 3694 and support['rows'] == 354550 and support['last_batch_rows'] == 22
    assert support['per_TF']['M5']['side']['last_nonzero_support_step_issue_rows'] == [354432, 354528]
    assert support['per_TF']['H4']['side']['last_nonzero_support_step_issue_rows'] == [354528, 354550]
    assert support['zero_support_reason'] == support['historical_per_TF_parameter_gradients'] == 'NOT_RECORDED'
    assert support['historical_grad_None_vs_zero_tensor'] == 'NOT_RECORDED' and support['unknown_is_not_FLAT'] is True
    assert 'UNKNOWN' in support['last_known_limit'] and 'not missing market data' in support['zero_support_semantics']
    assert progress['checkpoint_deserializations'] == 1 and progress['phase'] == 'COMPLETE_STORED_SUPPORT_AND_TERMINAL_MOMENTS'
    json.dumps(report, allow_nan=False)


def test_stored_calculation_rejects_post_extraction_checkpoint_mutation_at_proof_bookend(monkeypatch):
    plan, cp, _, loads = stored_calculation_fixture(monkeypatch)
    actual = m.alignment.moment_extract_heads; witnessed = []
    def mutating_extract(checkpoint, recipe):
        heads = actual(checkpoint, recipe); witnessed.append(True)
        checkpoint['optimizer_state']['state'][101]['exp_avg'][0, 0] += 1.
        return heads
    monkeypatch.setattr(m.alignment, 'moment_extract_heads', mutating_extract)
    with pytest.raises(ValueError, match='SUPPORT_ALL_TEN_PROOFS_UNCHANGED$'):
        m.support_calculate(plan)
    assert witnessed == [True] and len(loads) == 1


def test_stored_calculation_rejects_cpu_rng_mutation_without_selecting_a_fix(monkeypatch):
    plan, _, _, loads = stored_calculation_fixture(monkeypatch)
    actual = m.support_moment_summary; before = torch.get_rng_state().clone(); witnessed = []
    def random_summary(heads):
        result = actual(heads); torch.rand(1); witnessed.append(True); return result
    monkeypatch.setattr(m, 'support_moment_summary', random_summary)
    try:
        with pytest.raises(ValueError, match='SUPPORT_PROCESS_CPU_RNG_UNCHANGED$'):
            m.support_calculate(plan)
        assert witnessed == [True] and len(loads) == 1
    finally:
        # Isolate this deliberate test fault; the production runner forbids restores.
        torch.set_rng_state(before)


@pytest.mark.parametrize('action,name', [('NN', '__init__'), ('optimizer', '__init__'), ('CUDA', 'init'),
                                        ('autograd', 'grad'), ('restore', 'strict_restore'), ('seed', 'manual_seed')])
def test_stored_runtime_blocks_construction_cuda_derivatives_restores_and_seeding(action, name):
    actions = {'NN': lambda: torch.nn.Module.__init__(None),
               'optimizer': lambda: torch.optim.Optimizer.__init__(None, None, None),
               'CUDA': lambda: torch.cuda.init(), 'autograd': lambda: torch.autograd.grad(None, None),
               'restore': lambda: m.train.strict_restore(), 'seed': lambda: torch.manual_seed(1)}
    before = m.support_rng_proof()
    with m.support_runtime_blockers() as attempts:
        assert attempts == []
        with pytest.raises(RuntimeError, match='^SUPPORT_RUNTIME_' + name + '_FORBIDDEN$'):
            actions[action]()
        assert attempts == [name]
    assert m.support_rng_proof() == before and not torch.cuda.is_initialized()


def stored_lifecycle(monkeypatch, tmp_path):
    _, planref, pref, hooks, timers = lifecycle(monkeypatch, tmp_path)
    monkeypatch.setattr(m, 'SUPPORT_ROOT', m.ROOT); monkeypatch.setattr(m, 'SUPPORT_PLAN', m.PLAN)
    plan = {'inputs': {'final_checkpoint': {'path': str(tmp_path / 'FINAL.pt'), 'sha256': 'a' * 64}}}
    monkeypatch.setattr(m, 'support_inspect', lambda *args, **kwargs: (plan, planref, pref))
    monkeypatch.setattr(m, 'support_checked_inputs', lambda plan: None)
    monkeypatch.setattr(m, 'support_calculate', lambda *args, **kwargs: {'synthetic': True})
    return plan, planref, pref, hooks, timers


def test_stored_preflight_uses_new_exact_CLI_caps_and_zero_decodes(monkeypatch, tmp_path, capsys):
    _, planref, _, hooks, _ = stored_lifecycle(monkeypatch, tmp_path)
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: pytest.fail('preflight CP decode'))
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: pytest.fail('preflight NPZ decode'))
    calls = []; actual = m.support_caps
    def caps(): calls.append(True); return actual()
    monkeypatch.setattr(m, 'support_caps', caps)
    monkeypatch.setattr(sys, 'argv', ['audit', '--stored-support-plan-sha256', planref['sha256'], '--preflight'])
    m.main(); result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'PREFLIGHT_PROVIDER_HLC_NONE_SUPPORT_MOMENTS_PASS'
    assert result['checkpoint_deserializations'] == result['NPZ_arrays_decoded'] == result['models'] == result['NN_forwards'] == 0
    assert result['claim_created'] is result['CUDA_initialized'] is False
    assert calls == [True] and hooks == [] and not m.SUPPORT_ROOT.exists()
    monkeypatch.setattr(sys, 'argv', sys.argv + ['--preflight'])
    with pytest.raises(ValueError, match='SUPPORT_EXACT_CLI$'): m.main()


@pytest.mark.parametrize('failure', [None, 'calculate', 'bookend'])
def test_stored_run_is_one_use_with_preserved_failure_or_success(monkeypatch, tmp_path, failure):
    _, planref, pref, _, timers = stored_lifecycle(monkeypatch, tmp_path)
    def calculate(*args, **kwargs):
        (m.SUPPORT_ROOT / 'PRESERVED.json').write_text('{"synthetic":true}')
        if failure == 'calculate': raise ValueError('stored_test_fault')
        return {'synthetic': True}
    monkeypatch.setattr(m, 'support_calculate', calculate)
    if failure == 'bookend': monkeypatch.setattr(m, 'source_clean', lambda plan: False)
    if failure:
        with pytest.raises(ValueError, match='stored_test_fault$' if failure == 'calculate' else 'SUPPORT_SOURCE_INPUT_PLAN_POLICY_BOOT_BOOKEND$'):
            m.support_run(planref['sha256'])
    else:
        result = m.support_run(planref['sha256'])
        assert result['result'] == m.ref(m.SUPPORT_ROOT / 'RESULT.json')
        assert json.loads((m.SUPPORT_ROOT / 'RESULT.json').read_text())['status'] == 'COMPLETE_CONDITIONAL_NONE_STORED_SUPPORT_MOMENTS'
    claim = json.loads((m.SUPPORT_ROOT / 'CLAIM.json').read_text())
    terminal = json.loads((m.SUPPORT_ROOT / 'TERMINAL.json').read_text())
    assert claim['policy_at_claim'] == pref and claim['training_authorized'] is False
    assert terminal['status'] == ('FAILED_CONSUMED' if failure else 'COMPLETE')
    assert terminal['remaining_invocations'] == 0 and terminal['relaunch_allowed'] is False
    assert (m.SUPPORT_ROOT / 'PRESERVED.json').exists() and (m.SUPPORT_ROOT / 'RESULT.json').exists() is (failure is None)
    assert timers[-1] == (m.signal.ITIMER_REAL, 0)


def test_stored_runtime_payload_allowlist_excludes_old_initial_and_every_NPZ(monkeypatch, tmp_path):
    plan, planref, _, hooks, _ = stored_lifecycle(monkeypatch, tmp_path)
    m.support_run(planref['sha256']); assert len(hooks) == 1
    hooks[0]('open', (plan['inputs']['final_checkpoint']['path'],))
    for filename in ('INITIAL.pt', 'TARGETS.npz', 'CAPTURE.npz', 'FEATURES.npz', 'foreign.pt'):
        with pytest.raises(RuntimeError, match='^SUPPORT_PAYLOAD_FORBIDDEN$'):
            hooks[0]('open', (str(tmp_path / filename),))
    for event in ('socket.connect', 'socket.getaddrinfo', 'urllib.Request'):
        with pytest.raises(RuntimeError, match='^SUPPORT_NETWORK_FORBIDDEN$'): hooks[0](event, ())


@pytest.mark.parametrize('admitted', [False, True])
def test_stored_gate_is_separate_cpu_authority_with_exact_single_checkpoint_budget(admitted):
    gate = m.support_gate(binding('new_stored_plan'), admitted)
    assert gate['remaining_invocations'] == int(admitted) and gate['execution_authorized'] is admitted
    assert gate['training_authorized'] is gate['test_authorized'] is gate['relaunch_allowed'] is False
    assert m.SUPPORT_PLAN != m.PLAN and m.SUPPORT_ROOT != m.ROOT and m.SUPPORT_STAGE != m.STAGE
    budget = m.SUPPORT_BUDGET
    assert budget['checkpoint_deserializations'] == 1 and budget['final_checkpoint_bytes'] == 97916887
    assert budget['steps'] == 3694 and budget['rows'] == 354550 and budget['hidden_width'] == 128
    assert budget['checkpoint_load_policy'] == {'fallback_or_add_safe_globals': False, 'map_location': 'cpu', 'weights_only': True}
    for key in ('NPZ_archive_loads', 'NPZ_arrays_decoded', 'NN_forwards', 'model_objects_constructed', 'optimizer_updates',
                'optimizer_constructions', 'optimizer_restores', 'model_state_install_calls', 'RNG_restore_calls', 'autograd_calls',
                'backward_calls', 'HTTP_calls', 'VAL_TEST_rows', 'prior_refits'):
        assert budget[key] == 0


def test_stored_inspect_never_accepts_old_plan_gate_or_any_global_training_flag(monkeypatch, tmp_path):
    planfile, policyfile = tmp_path / 'SUPPORT_PLAN.json', tmp_path / 'POLICY.json'
    monkeypatch.setattr(m, 'SUPPORT_PLAN', planfile); monkeypatch.setattr(m, 'POLICY', policyfile)
    monkeypatch.setattr(m, 'SUPPORT_ROOT', tmp_path / 'OUT'); monkeypatch.setattr(m, 'REPO', Path.cwd())
    plan = {'source_base_commit': '1' * 40}; planfile.write_text(json.dumps(plan)); planref = m.ref(planfile)
    flags = ('training_enabled', 'native_training_enabled', 'native_launch_authorized', 'full_epoch_training_allowed', 'full_val_allowed')
    policy = dict.fromkeys(flags, False); policy[m.SUPPORT_STAGE] = m.support_gate(planref, False)
    monkeypatch.setattr(m, 'support_build_plan', lambda **kwargs: plan)
    monkeypatch.setattr(m, 'source_clean', lambda value: True)
    def inspect():
        policyfile.write_text(json.dumps(policy)); return m.support_inspect(planref['sha256'], admitted=False)
    assert inspect()[0] == plan
    for key in flags:
        policy[key] = True
        with pytest.raises(ValueError, match='SUPPORT_ONE_USE_ALL_TRAINING_CLOSED$'): inspect()
        policy[key] = False
    old_binding = {'path': str(m.PLAN), 'sha256': 'a' * 64}
    policy[m.SUPPORT_STAGE] = m.gate_expected(old_binding, False)
    with pytest.raises(ValueError, match='SUPPORT_ONE_USE_ALL_TRAINING_CLOSED$'): inspect()
