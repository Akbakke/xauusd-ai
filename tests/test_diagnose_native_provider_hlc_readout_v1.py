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
