"""Synthetic fresh-None INITIAL gates and lifecycle; no genuine payload or CUDA."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from gx1.audit import native_future_state_provider_hlc_initial_v1 as m

ANCHOR = 'f33da18dd6a0e3d59e4876fa26f8e62ec5ab97b8'


def test_all_tf_initial_preserves_every_legacy_function_constant_and_main_tail():
    before = subprocess.check_output(['git', 'show', ANCHOR+':'+m.SELF], cwd=m.REPO, text=True)
    after = Path(m.__file__).read_text(); old_tree, tree = ast.parse(before), ast.parse(after)
    old = {n.name: n for n in old_tree.body if isinstance(n, ast.FunctionDef)}
    new = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name, node in old.items():
        if name != 'main': assert ast.get_source_segment(before, node) == ast.get_source_segment(after, new[name])
    assignments = lambda value: [ast.dump(n, include_attributes=False) for n in value.body if isinstance(n, (ast.Assign, ast.AnnAssign))]
    assert all(value in assignments(tree) for value in assignments(old_tree))
    dispatch = new['main'].body[0]
    assert isinstance(dispatch, ast.If)
    expected = ast.parse("sys.argv[1:2] == ['--all-tf-plan-sha256']", mode='eval').body
    assert ast.dump(dispatch.test, include_attributes=False) == ast.dump(expected, include_attributes=False)
    assert ast.dump(ast.Module(body=new['main'].body[1:], type_ignores=[]), include_attributes=False) == \
           ast.dump(ast.Module(body=old['main'].body, type_ignores=[]), include_attributes=False)


@pytest.mark.parametrize('admitted', [False, True])
def test_all_tf_initial_gate_authorizes_only_one_zero_update_initial(admitted):
    binding = {'path': '/synthetic/ALL_INITIAL_PLAN.json', 'sha256': 'a'*64}
    gate = m.all_tf_gate(binding, admitted)
    assert gate['plan'] == binding and gate['execution_authorized'] is admitted
    assert gate['remaining_invocations'] == int(admitted)
    assert gate['training_authorized'] is gate['optimizer_authorized'] is gate['test_authorized'] is gate['relaunch_allowed'] is False


def test_all_tf_initial_still_requires_every_global_training_flag_false():
    policy = dict.fromkeys(m.FLAGS, False); m.validate_flags(policy)
    for key in m.FLAGS:
        policy[key] = True
        with pytest.raises(ValueError): m.validate_flags(policy)
        policy[key] = False


def initial_lifecycle(monkeypatch, tmp_path):
    from gx1.contracts import gx1_capped_execution_v1 as guard
    monkeypatch.setattr(m, 'ALL_ROOT', tmp_path/'ALL_INITIAL')
    monkeypatch.setattr(m, 'ALL_PLAN', tmp_path/'ALL_PLAN.json'); monkeypatch.setattr(m, 'POLICY', tmp_path/'POLICY.json')
    m.ALL_PLAN.write_text('{"synthetic":true}'); m.POLICY.write_text('{"training_enabled":false}')
    binding, pref = m.ref(m.ALL_PLAN), m.ref(m.POLICY)
    plan = {'inputs': {key: {'path': str(tmp_path/name), 'sha256': 'a'*64}
                      for key, name in (('feature_archive', 'FEATURES.npz'), ('target_arrays', 'TARGETS.npz'))}}
    live = {'physical_windows_boot_id': 1, 'wsl_boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    monkeypatch.setattr(guard, 'require_native_provider_hlc_all_tf_initial_execution', lambda **k: {'plan': binding, 'output_root': str(m.ALL_ROOT)})
    monkeypatch.setattr(m, 'all_tf_inspect', lambda *a, **k: (plan, binding, pref, live))
    monkeypatch.setattr(m, 'all_tf_check_sources_inputs', lambda plan: None)
    monkeypatch.setattr(m.core, 'physical_windows_boot_id', lambda: 1)
    monkeypatch.setattr(m.core.common, 'firewall_state', lambda: {'denied_attempts': 0})
    monkeypatch.setattr(m.core, 'future_cpu_process_identity', lambda: {'pid': 0})
    hooks, timers = [], []
    monkeypatch.setattr(sys, 'addaudithook', hooks.append)
    monkeypatch.setattr(m.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(m.signal, 'setitimer', lambda *a: timers.append(a))
    # Restore producer-local optimizer/backward bans after each unit test.
    monkeypatch.setattr(torch.optim.Optimizer, '__init__', torch.optim.Optimizer.__init__)
    monkeypatch.setattr(torch.Tensor, 'backward', torch.Tensor.backward)
    monkeypatch.setattr(torch.autograd, 'grad', torch.autograd.grad)
    monkeypatch.setattr(m, 'all_tf_physical', lambda *a, **k: {'fixture_model_calls': 0})
    return plan, binding, pref, live, hooks, timers


def test_all_tf_initial_exact_preflight_and_legacy_default_remain_separate(monkeypatch, tmp_path, capsys):
    _, binding, pref, live, hooks, _ = initial_lifecycle(monkeypatch, tmp_path)
    def forbidden(*a, **k): raise AssertionError('NO_INITIAL_PAYLOAD_IN_PREFLIGHT')
    monkeypatch.setattr(m, 'all_tf_physical', forbidden); monkeypatch.setattr(torch, 'load', forbidden); monkeypatch.setattr(np, 'load', forbidden)
    monkeypatch.setattr(sys, 'argv', ['initial', '--all-tf-plan-sha256', binding['sha256'], '--preflight'])
    m.main(); report = json.loads(capsys.readouterr().out)
    assert report['plan'] == binding and report['payloads_decoded'] == report['models'] == report['optimizer_updates'] == 0
    assert report['cuda_initialized'] is False and hooks == [] and not m.ALL_ROOT.exists()
    monkeypatch.setattr(sys, 'argv', sys.argv+['--preflight'])
    with pytest.raises(ValueError, match='EXACT_CLI'): m.main()
    seen = []
    monkeypatch.setattr(m, 'inspect', lambda *a, **k: (seen.append(a) or ({}, binding, pref, live)))
    monkeypatch.setattr(sys, 'argv', ['initial', '--plan-sha256', binding['sha256'], '--preflight'])
    m.main(); assert seen == [(binding['sha256'], False)]


@pytest.mark.parametrize('failure', [False, True])
def test_all_tf_initial_reporting_preserves_new_owned_initial_and_consumes_failure(monkeypatch, tmp_path, failure):
    _, binding, _, _, _, timers = initial_lifecycle(monkeypatch, tmp_path)
    def physical(*a, **k):
        (m.ALL_ROOT/'PRESERVED_INITIAL_METADATA.json').write_text('{"preserved":true}')
        if failure: raise ValueError('SYNTHETIC_BEFORE_ANY_MODEL')
        return {'fixture_model_calls': 0, 'optimizer_updates': 0}
    monkeypatch.setattr(m, 'all_tf_physical', physical)
    if failure:
        with pytest.raises(ValueError, match='SYNTHETIC_BEFORE_ANY_MODEL'): m.all_tf_run(binding['sha256'])
    else:
        outputs = m.all_tf_run(binding['sha256']); assert outputs['result'] == m.ref(m.ALL_ROOT/'RESULT.json')
    terminal = json.loads((m.ALL_ROOT/'TERMINAL.json').read_text())
    assert terminal['status'] == ('FAILED_CONSUMED' if failure else 'COMPLETE')
    assert terminal['remaining_invocations'] == 0 and terminal['relaunch_allowed'] is False and terminal['plan'] == binding
    assert (m.ALL_ROOT/'RESULT.json').exists() is (not failure)
    assert (m.ALL_ROOT/'PRESERVED_INITIAL_METADATA.json').is_file()
    assert timers[-1] == (m.signal.ITIMER_REAL, 0)
    with pytest.raises(FileExistsError): m.all_tf_run(binding['sha256'])


def test_all_tf_initial_runtime_rejects_every_existing_checkpoint_load(monkeypatch, tmp_path):
    plan, binding, _, _, hooks, _ = initial_lifecycle(monkeypatch, tmp_path)
    m.all_tf_run(binding['sha256']); assert len(hooks) == 1
    for value in plan['inputs'].values(): hooks[0]('open', (value['path'],))
    hooks[0]('open', (str(m.ALL_ROOT/'INITIAL.pt'),))
    for path in ('/synthetic/FINAL.pt', '/synthetic/OLD_INITIAL.pt', '/synthetic/CURSOR002.pt', '/synthetic/foreign.npz'):
        with pytest.raises(RuntimeError, match='PAYLOAD_PATH_FORBIDDEN'): hooks[0]('open', (path,))
    for event in ('socket.connect', 'socket.getaddrinfo', 'urllib.Request'):
        with pytest.raises(RuntimeError, match='NETWORK_FORBIDDEN'): hooks[0](event, ())


class FreshLeaf(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    def forward(self, value): return value*self.weight


class FreshNoneModel(torch.nn.Module):
    def __init__(self, contract):
        super().__init__(); self.backbone = FreshLeaf(); self.readout = FreshLeaf(); self._contract = contract
        self.register_buffer('future_state_model_sha256', torch.tensor(list(bytes.fromhex(contract['contract_sha256'])), dtype=torch.uint8))
    @property
    def research_objective(self): return self._contract
    def forward(self, value): return self.readout(self.backbone(value))


def fresh_fixture(monkeypatch):
    contract = {'contract_sha256': m.ALL_CONTRACT_SHA}
    reference = {'contract_sha256': 'a'*64, 'backbone_gradient_policy': {'policy': 'M5_primary'}}
    rng = {'fixture_cpu_rng': torch.tensor([1, 2, 3], dtype=torch.uint8)}
    state = {'backbone.weight': torch.zeros(1), 'readout.weight': torch.zeros(1),
             'future_state_model_sha256': torch.tensor(list(bytes.fromhex('a'*64)), dtype=torch.uint8)}
    plan = {'model_contract': contract, 'metadata': {'input_binding': {'fixture': 'input'},
        'normalization': {'path': '/synthetic/NORMALIZATION.json', 'sha256': 'a'*64},
        'target_contract': {'contract_sha256': 'c'*64}, 'truth_sha256': 'd'*64, 'architecture': {}},
        'reference_model_contract': reference, 'reference_initial': {'model_state_sha256': m.core.state_sha(state),
                                                                  'rng_sha256': m.core.state_sha(rng)}}
    monkeypatch.setattr(m.core, 'training_rng', lambda: copy.deepcopy(rng))
    return plan, rng


def test_all_tf_fresh_reference_comparison_normalizes_only_copy_and_never_loads_old_state(monkeypatch):
    plan, _ = fresh_fixture(monkeypatch); model = FreshNoneModel(plan['model_contract'])
    before = m.core.state_sha(model.state_dict())
    def forbidden(*a, **k): raise AssertionError('FRESH_COMPARISON_MUST_NOT_LOAD_STATE')
    monkeypatch.setattr(model, 'load_state_dict', forbidden)
    comparison = m.all_tf_compare_fresh_reference(model, plan)
    assert comparison['comparison_only_no_state_load'] is True
    assert comparison['policy_normalized_state_sha256'] == plan['reference_initial']['model_state_sha256']
    assert comparison['fresh_rng_sha256'] == plan['reference_initial']['rng_sha256']
    assert m.core.state_sha(model.state_dict()) == before
    assert model.future_state_model_sha256.tolist() == list(bytes.fromhex(m.ALL_CONTRACT_SHA))


@pytest.mark.parametrize('bad', ['parameter', 'RNG', 'old_policy'])
def test_all_tf_fresh_reference_rejects_wrong_parameters_RNG_or_M5_identity(monkeypatch, bad):
    plan, rng = fresh_fixture(monkeypatch); model = FreshNoneModel(plan['model_contract'])
    if bad == 'parameter': model.backbone.weight.data.add_(1.)
    elif bad == 'RNG':
        rng['fixture_cpu_rng'][0] ^= 1
        monkeypatch.setattr(m.core, 'training_rng', lambda: copy.deepcopy(rng))
    else: model._contract = plan['reference_model_contract']
    before = m.core.state_sha(model.state_dict())
    with pytest.raises(ValueError): m.all_tf_compare_fresh_reference(model, plan)
    assert m.core.state_sha(model.state_dict()) == before


def test_all_tf_own_INITIAL_roundtrip_is_CPU_weights_only_and_binds_new_contract(monkeypatch, tmp_path):
    plan, rng = fresh_fixture(monkeypatch); model = FreshNoneModel(plan['model_contract'])
    monkeypatch.setattr(m, 'ALL_ROOT', tmp_path/'NEW_INITIAL'); m.ALL_ROOT.mkdir()
    binding = {'path': '/synthetic/ALL_INITIAL_PLAN.json', 'sha256': 'a'*64}
    counts = dict.fromkeys(m.ALL_LIMITS, 0); loads, saved = [], []
    save, load = torch.save, torch.load
    def save_checkpoint(cp, stream):
        saved.append(copy.deepcopy(cp)); return save(cp, stream)
    def load_checkpoint(path, **kwargs):
        loads.append((path, kwargs)); return load(path, **kwargs)
    monkeypatch.setattr(torch, 'save', save_checkpoint); monkeypatch.setattr(torch, 'load', load_checkpoint)
    receipt_ref, state_sha = m.all_tf_write_initial_checkpoint(model, plan, binding, {'fixture': 'runtime'}, counts)
    receipt = m.read(receipt_ref); cp = saved[0]
    assert loads == [(str(m.ALL_ROOT/'INITIAL.pt'), {'map_location': 'cpu', 'weights_only': True})]
    assert counts['new_checkpoint_writes'] == counts['new_checkpoint_readbacks'] == 1
    assert counts['existing_checkpoint_loads'] == counts['model_state_restores'] == 0
    assert cp['model_contract'] == plan['model_contract'] and 'backbone_gradient_policy' not in cp['model_contract']
    assert cp['plan'] == receipt['plan'] == binding and cp['optimizer'] is None and cp['optimizer_updates'] == 0
    assert receipt['initial_only'] is True and receipt['model_contract_sha256'] == m.ALL_CONTRACT_SHA
    assert receipt['checkpoint_tree_sha256'] == m.core.state_sha(cp)
    assert receipt['model_state_sha256'] == state_sha == m.core.state_sha(model.state_dict())
    assert receipt['rng_sha256'] == m.core.state_sha(rng)
    assert cp['authority'] == 'INITIAL_ONLY_NO_FIT_OR_RESUME_AUTHORITY'


def fresh_physical_fixture(monkeypatch):
    from gx1.audit import native_future_state_provider_hlc_baseline_v1 as adapter
    from gx1.contracts import market_state_ta_integrated_model_v1 as model_owner
    plan, _ = fresh_fixture(monkeypatch)
    issue_ns = np.arange(m.ROWS, dtype=np.int64)
    unknown_side = np.broadcast_to(np.zeros((1, 5), dtype=np.bool_), (m.ROWS, 5))
    context = {'source_rows': np.arange(m.ROWS, dtype=np.int64), 'truth': {'issue_ns': issue_ns, 'side_known': unknown_side},
               'normalization': {'fixture': 'normalization'}, 'prior': {'fixture': 'v2-prior'}}
    order, models, publications = [], [], []
    def load_context(meta):
        assert meta is plan['metadata']; order.append('context'); return context
    def seed(request):
        assert request['recipe']['seed'] == m.SEED; order.append('fresh_seed'); return {'seed': m.SEED}
    def factory(**kwargs):
        assert order == ['context', 'fresh_seed']
        assert kwargs['backbone_gradient_policy'] is None and kwargs['device'] == torch.device('cuda')
        assert kwargs['input_normalization'] is context['normalization'] and kwargs['train_prior'] is context['prior']
        model = FreshNoneModel(plan['model_contract']); models.append(model); order.append('None_factory'); return model
    def write(model, bound_plan, binding, live, counts):
        assert order[-1] == 'None_factory'; counts['new_checkpoint_writes'] += 1; counts['new_checkpoint_readbacks'] += 1
        return {'path': '/synthetic/OWN_INITIAL.json', 'sha256': 'a'*64}, m.core.state_sha(model.state_dict())
    def publish(name, value): publications.append((name, copy.deepcopy(value))); return {'path': '/synthetic/'+name, 'sha256': 'b'*64}
    monkeypatch.setattr(adapter, 'load_context', load_context)
    monkeypatch.setattr(m.core, 'future_cuda_seed_and_runtime', seed)
    monkeypatch.setattr(model_owner, 'build_native_future_state_model', factory)
    monkeypatch.setattr(m, 'all_tf_write_initial_checkpoint', write); monkeypatch.setattr(m, 'all_tf_publish', publish)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda: 0)
    return plan, context, models, publications, adapter


@pytest.mark.parametrize('bad', [None, 'missing_last_batch', 'state_mutation'])
def test_all_tf_fresh_physical_delegates_full_qualified_v2_baseline_and_exact_zero_update_counts(monkeypatch, bad):
    plan, context, models, publications, adapter = fresh_physical_fixture(monkeypatch)
    def score(model, qualified_context, *, batch_rows, device, progress_callback):
        assert qualified_context is context and batch_rows == 256 and device == 'cuda'
        assert qualified_context['truth']['side_known'].shape == (354550, 5)
        assert not qualified_context['truth']['side_known'].any()
        stop = m.BATCHES - int(bad == 'missing_last_batch')
        for index in range(1, stop+1):
            model(torch.ones(1)); progress_callback({'batch_cursor': index})
        if bad == 'state_mutation': model.backbone.weight.data.add_(1.)
        return {'rows': 354550, 'batch_count': 1385, 'initial_prior_exact': True}
    monkeypatch.setattr(adapter, 'score_initial', score)
    progress = {}
    if bad:
        with pytest.raises(ValueError, match='WHOLE_TRAIN_INITIAL_STATE_AND_SCORE'):
            m.all_tf_physical(plan, {'path': '/synthetic/plan', 'sha256': 'a'*64}, {}, progress)
    else:
        report = m.all_tf_physical(plan, {'path': '/synthetic/plan', 'sha256': 'a'*64}, {}, progress)
        assert report['counts'] == m.ALL_LIMITS
        assert report['counts']['truth_fields_decoded'] == 14
        assert report['counts']['existing_checkpoint_loads'] == report['counts']['model_state_restores'] == report['counts']['optimizer_constructions'] == 0
        assert report['counts']['optimizer_steps'] == report['counts']['backward_calls'] == report['counts']['autograd_calls'] == 0
        assert report['whole_TRAIN_initialbaseline_qualified'] is True and report['backbone_gradient_policy'] is None
        assert report['optimizer_replay_qualified'] is report['learning_qualified'] is False
        assert report['later_generalization'] == report['economic_edge'] == 'UNKNOWN'
        assert report['fresh_reference_comparison']['comparison_only_no_state_load'] is True
        assert publications[-1][0] == 'PROGRESS_1385.json'
    assert len(models) == 1
    assert all(not module._forward_pre_hooks for module in models[0].modules())


def test_all_tf_builder_retains_actual_v2_target_gate_and_forbids_target_metadata_as_model_input():
    tree = ast.parse(Path(m.__file__).read_text())
    builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_all_tf_plan')
    statements = [n for n in ast.walk(builder) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name) and n.value.func.id == 'need' and len(n.value.args) == 2
        and isinstance(n.value.args[1], ast.Constant) and n.value.args[1].value == 'TARGET_DATASET']
    assert len(statements) == 1
    code = compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(statements[0])], type_ignores=[])), '<actual_initial_v2_target_gate>', 'exec')
    data = {'contract': m.hlc.provider_hlc_state_contract(), 'base_dataset': m.hlc.SOURCE_BINDINGS['base_dataset'],
            'cutoff_ns': m.hlc.TRAIN_UPPER_NS, 'model_input': False}
    environment = {'data': data, 'hlc': m.hlc, 'need': m.need}
    exec(code, environment)
    data['model_input'] = True
    with pytest.raises(ValueError, match='TARGET_DATASET'): exec(code, environment)
    data['model_input'] = False; data['contract'] = {'schema_version': 'OLD_V1_IS_NOT_V2_AUTHORITY'}
    with pytest.raises(ValueError, match='TARGET_DATASET'): exec(code, environment)
