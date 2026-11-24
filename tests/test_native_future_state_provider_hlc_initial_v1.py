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
    main_body = new['main'].body
    segment_test = ast.parse("sys.argv[1:2] == ['--all-tf-score-segment-plan-sha256']", mode='eval').body
    if ast.dump(main_body[0].test, include_attributes=False) == ast.dump(segment_test, include_attributes=False):
        main_body = main_body[1:]
    dispatch = main_body[0]
    assert isinstance(dispatch, ast.If)
    expected = ast.parse("sys.argv[1:2] == ['--all-tf-plan-sha256']", mode='eval').body
    assert ast.dump(dispatch.test, include_attributes=False) == ast.dump(expected, include_attributes=False)
    assert ast.dump(ast.Module(body=main_body[1:], type_ignores=[]), include_attributes=False) == \
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


# Fresh INITIAL003 segments: tiny output archives and metadata only.
def test_segment_addition_preserves_the_entire_9d7_producer_and_old_defaults():
    before = subprocess.check_output(['git', 'show', m.SEG_PARENT+':'+m.SELF], cwd=m.REPO, text=True)
    after = Path(m.__file__).read_text()
    begin = '# Durable all-TF INITIAL segments; previous producers remain consumed.\n'
    end = '# End durable all-TF INITIAL segments.\n\n'
    assert after.count(begin) == after.count(end) == 1
    left, rest = after.split(begin); _, right = rest.split(end)
    prefix = " if sys.argv[1:2]==['--all-tf-score-segment-plan-sha256']:return main_segment()\n"
    stripped = left+right
    assert stripped.count(prefix) == 1 and stripped.replace(prefix, '', 1) == before
    assert m.SEG_NAME.endswith('_003') and m.ALL_NAME.endswith('_002')
    assert m.SEG_RECIPE['rows'] == 354550 and m.SEG_RECIPE['batch_rows'] == 256
    assert m.SEG_RECIPE['batches'] == 1385 and m.SEG_RECIPE['last_rows'] == 246
    assert m.SEG_RECIPE['backbone_gradient_policy'] is None


@pytest.mark.parametrize('cursor,first', [(0, True), (129, False), (1385, False)])
def test_segment_limits_separate_new_physical_calls_from_committed_prefix(cursor, first):
    counts = m.segment_limits(cursor, first)
    assert set(counts) == set(m.SEG_COUNT_KEYS)
    assert counts['model_forwards'] == counts['backbone_forwards'] == counts['readout_forwards'] == 1385-cursor
    assert counts['committed_batches'] == counts['output_archive_write_calls'] == counts['output_archive_readbacks'] == counts['completed_batch_receipt_write_calls'] == 1385-cursor
    assert counts['prefix_output_archive_loads'] == cursor
    assert counts['new_checkpoint_writes'] == counts['new_checkpoint_readbacks'] == int(first)
    assert all(counts[key] == 0 for key in ('existing_checkpoint_loads', 'model_state_restores',
        'optimizer_constructions', 'optimizer_steps', 'backward_calls', 'autograd_calls', 'CONTROL_VAL_TEST_rows'))
    assert m.segment_gate({'path': '/synthetic/segment.json', 'sha256': 'a'*64}, False)['remaining_invocations'] == 0
    gate = m.segment_gate({'path': '/synthetic/segment.json', 'sha256': 'a'*64}, True)
    assert gate['remaining_invocations'] == 1 and gate['relaunch_allowed'] is gate['training_authorized'] is gate['optimizer_authorized'] is gate['test_authorized'] is False


def segment_fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(m, 'SEG_ROOT', tmp_path/m.SEG_NAME)
    monkeypatch.setattr(m, 'SEG_BASE_PLAN', tmp_path/'BASE003.json')
    monkeypatch.setattr(m, 'POLICY', tmp_path/'POLICY.json')
    monkeypatch.setattr(m, 'BATCH', 3); monkeypatch.setattr(m, 'BATCHES', 3); monkeypatch.setattr(m, 'ROWS', 8)
    m.SEG_ROOT.mkdir()
    fresh, rng = fresh_fixture(monkeypatch)
    model = FreshNoneModel(fresh['model_contract'])
    runtime = {'gpu': {'name': 'SYNTHETIC'}, 'library_versions': {'torch': 'SYNTHETIC'}}
    base = {'schema_version': m.SEG_BASE_SCHEMA, 'stage': m.SEG_STAGE,
        'output_root': str(m.SEG_ROOT), 'model_contract': fresh['model_contract'],
        'recipe': m.SEG_RECIPE, 'execution': m.SEG_EXECUTION, 'source_owners': {},
        'source_bridge': {'synthetic': 'frozen-source-bridge'},
        'inputs': {key: {'path': str(tmp_path/(key+'.npz')), 'sha256': 'a'*64}
                   for key in ('feature_archive', 'target_arrays')},
        'metadata': {'runtime': runtime, 'target_contract': fresh['metadata']['target_contract']}}
    base_ref = m.segment_publish(tmp_path, m.SEG_BASE_PLAN.name, base)
    policy_ref = m.segment_publish(tmp_path, m.POLICY.name, dict.fromkeys(m.FLAGS, False))
    empty = {'cursor': 0, 'rows_done': 0, 'head': None}
    plan = {'schema_version': m.SEG_SCHEMA, 'stage': m.SEG_STAGE, 'base_plan': base_ref,
        'base_root': str(m.SEG_ROOT), 'recipe': m.SEG_RECIPE, 'execution': m.SEG_EXECUTION,
        'model_contract': base['model_contract'], 'source_owners': {},
        'source_bridge': base['source_bridge'], 'metadata': dict(base['metadata']),
        'initial_mode': 'first_fresh', 'limits': m.segment_limits(0, True), 'prefix': empty,
        'parent_terminal': None, 'initial_receipt': None,
        'inputs': {**base['inputs'], 'base_plan': base_ref}}
    encoded = (json.dumps(plan, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    sha = m.hashlib.sha256(encoded).hexdigest()
    pp, root = m.segment_paths(sha); pp.parent.mkdir(); root.mkdir()
    binding = m.segment_publish(pp.parent, pp.name, plan)
    assert binding == {'path': str(pp), 'sha256': sha}
    claim = m.segment_publish(root, 'CLAIM.json', {'plan': binding, 'base_plan': base_ref,
        'policy': policy_ref, 'runtime': runtime, 'process_identity': {'pid': 0},
        'remaining_invocations': 0, 'relaunch_allowed': False})
    checkpoint_ref = m.segment_publish(m.SEG_ROOT, 'INITIAL.pt', b'SYNTHETIC_INITIAL_BYTES_NOT_A_TENSOR_CHECKPOINT')
    initial = {'schema_version': 'gx1_native_provider_hlc_fresh_initial_checkpoint_v1',
        'plan': base_ref, 'checkpoint': checkpoint_ref,
        'model_contract_sha256': m.ALL_CONTRACT_SHA,
        'target_contract_sha256': base['metadata']['target_contract']['contract_sha256'],
        'model_state_sha256': m.core.state_sha(model.state_dict()), 'rng_sha256': m.core.state_sha(rng),
        'initial_only': True, 'optimizer_updates': 0}
    initial_ref = m.segment_publish(m.SEG_ROOT, 'INITIAL.json', initial)
    outputs = {key: np.arange(15, dtype=np.float32).reshape(3, 5) for key in ('activity_logits', 'side_logits')}
    issues = np.arange(8, dtype=np.int64)+100; source = np.arange(8, dtype=np.int64)+1000
    event = {'batch_cursor': 1, 'batch_count': 3, 'rows_done': 3, 'rows_total': 8,
        'left': 0, 'right': 3, 'issue_ns_sha256': m.core.digest(issues[:3]),
        'source_rows_sha256': m.core.digest(source[:3]), 'model_state_sha256': initial['model_state_sha256'],
        'rng_sha256': initial['rng_sha256'], 'gradients_none': True,
        'batch': {'cursor': 1, 'rows': 3, 'rows_done': 3, 'known_activity_cells': 15,
                  'partial_NLL': 0.5, 'output_sha256': {key: m.core.digest(value) for key, value in outputs.items()},
                  'activation': {'batch_eval_active': True}}}
    monkeypatch.setattr(m.core, 'future_cpu_parent_stopped', lambda identity: None)
    monkeypatch.setattr(m.core, 'future_cuda_runtime_matches', lambda expected, actual: m.need(expected == actual, 'SYNTHETIC_RUNTIME'))
    return {'base': base, 'base_ref': base_ref, 'plan': plan, 'binding': binding, 'root': root,
        'claim': claim, 'policy_ref': policy_ref, 'initial': initial, 'initial_ref': initial_ref,
        'model': model, 'event': event, 'outputs': outputs,
        'context': {'truth': {'issue_ns': issues}, 'source_rows': source}}


def close_segment_fixture(case, *, bad=None):
    prefix = m.segment_durable_prefix(case['plan'], case['binding'], case['claim'], case['initial_ref'])
    counts = dict.fromkeys(m.SEG_COUNT_KEYS, 0)
    for key in ('model_forwards', 'backbone_forwards', 'readout_forwards', 'output_archive_write_calls',
                'output_archive_readbacks', 'completed_batch_receipt_write_calls', 'committed_batches'):
        counts[key] = prefix['cursor']
    proof = {'model_state_sha256': case['initial']['model_state_sha256'], 'rng_sha256': case['initial']['rng_sha256'],
             'gradients_none': True, 'fresh_constructor_no_weight_load': True}
    closure = {'status': 'CLOSED_SEGMENT_REUSE_PROOF_PASS', 'plan': case['binding'], 'base_plan': case['base_ref'],
        'claim': case['claim'], 'policy': case['policy_ref'], 'prefix_before': case['plan']['prefix'],
        'prefix_after': prefix, 'initial_receipt': case['initial_ref'], 'counts': counts, 'model_RNG_proof': proof,
        **dict.fromkeys(('source_inputs_unchanged', 'policy_unchanged', 'physical_boot_unchanged',
                        'wsl_boot_unchanged', 'model_RNG_grad_unchanged', 'M1_firewall_clear'), True)}
    if bad == 'closure': closure['model_RNG_grad_unchanged'] = False
    elif bad == 'state': proof['model_state_sha256'] = '0'*64
    elif bad == 'count': counts['committed_batches'] += 1
    closure_ref = m.segment_publish(case['root'], 'CLOSURE_PROOF.json', closure)
    terminal = {'status': 'FAILED_CONSUMED' if bad != 'running' else 'RUNNING', 'plan': case['binding'],
        'base_plan': case['base_ref'], 'exit_code': 1,
        'claim': case['claim'], 'closure_proof': closure_ref, 'prefix': prefix, 'initial_receipt': case['initial_ref'],
        'remaining_invocations': 0, 'relaunch_allowed': False, 'elapsed_seconds': 1.0}
    return m.segment_publish(case['root'], 'TERMINAL.json', terminal)


def test_segment_batch_archive_is_read_back_before_receipt_last_commit(monkeypatch, tmp_path):
    case = segment_fixture(monkeypatch, tmp_path)
    published = []; publish = m.segment_publish
    def observe(root, name, value):
        if name.endswith('.json'):
            assert (root/'BATCH_0001.npz').is_file()
            assert not (root/'BATCH_0001.json').exists()
        published.append(name); return publish(root, name, value)
    monkeypatch.setattr(m, 'segment_publish', observe)
    counts = dict.fromkeys(m.SEG_COUNT_KEYS, 0)
    receipt_ref = m.segment_publish_batch(case['plan'], case['binding'], case['claim'], case['event'], case['outputs'], None, counts=counts)
    receipt = m.read(receipt_ref)
    assert published == ['BATCH_0001.npz', 'BATCH_0001.json']
    assert receipt['base_plan'] == case['base_ref'] and receipt['execution_plan'] == case['binding'] and receipt['claim'] == case['claim']
    assert receipt['previous_receipt'] is None and receipt['event'] == case['event']
    assert counts['committed_batches'] == counts['output_archive_write_calls'] == counts['output_archive_readbacks'] == counts['completed_batch_receipt_write_calls'] == 1
    with pytest.raises(RuntimeError, match='already exists'):
        m.segment_publish_batch(case['plan'], case['binding'], case['claim'], case['event'], case['outputs'], None)
    assert m.read(receipt_ref) == receipt


@pytest.mark.parametrize('boundary', ['before_archive', 'before_receipt', 'after_receipt'])
def test_segment_crash_boundary_uses_only_durable_receipt_not_progress(monkeypatch, tmp_path, boundary):
    case = segment_fixture(monkeypatch, tmp_path)
    publish = m.segment_publish
    def interrupted(root, name, value):
        if boundary == 'before_archive' and name.endswith('.npz'): raise OSError('CRASH_BEFORE_ARCHIVE')
        if boundary == 'before_receipt' and name.endswith('.json'): raise OSError('CRASH_BEFORE_RECEIPT')
        result = publish(root, name, value)
        if boundary == 'after_receipt' and name.endswith('.json'): raise OSError('CRASH_AFTER_RECEIPT')
        return result
    monkeypatch.setattr(m, 'segment_publish', interrupted)
    counts = dict.fromkeys(m.SEG_COUNT_KEYS, 0)
    with pytest.raises(OSError):
        m.segment_publish_batch(case['plan'], case['binding'], case['claim'], case['event'], case['outputs'], None, counts=counts)
    assert counts['committed_batches'] == 0
    prefix = m.segment_durable_prefix(case['plan'], case['binding'], case['claim'], case['initial_ref'])
    assert prefix['cursor'] == (1 if boundary == 'after_receipt' else 0)
    assert prefix['rows_done'] == (3 if boundary == 'after_receipt' else 0)
    assert (case['root']/'BATCH_0001.npz').exists() is (boundary != 'before_archive')
    assert (case['root']/'BATCH_0001.json').exists() is (boundary == 'after_receipt')


def test_segment_corrupt_output_readback_never_publishes_commit(monkeypatch, tmp_path):
    case = segment_fixture(monkeypatch, tmp_path)
    load = np.load
    class CorruptReadback:
        files = list(case['outputs'])
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def __getitem__(self, key): return np.zeros((3, 5), dtype=np.float32)
    monkeypatch.setattr(np, 'load', lambda *a, **k: CorruptReadback())
    with pytest.raises(ValueError, match='OUTPUT_READBACK_PROOF'):
        m.segment_publish_batch(case['plan'], case['binding'], case['claim'], case['event'], case['outputs'], None)
    assert (case['root']/'BATCH_0001.npz').exists() and not (case['root']/'BATCH_0001.json').exists()
    assert m.segment_durable_prefix(case['plan'], case['binding'], case['claim'], case['initial_ref']) == case['plan']['prefix']


@pytest.mark.parametrize('bad', [None, 'closure', 'state', 'count', 'running', 'checkpoint_bytes'])
def test_segment_prefix_requires_closed_parent_and_exact_fresh_state_proof(monkeypatch, tmp_path, bad):
    case = segment_fixture(monkeypatch, tmp_path)
    receipt = m.segment_publish_batch(case['plan'], case['binding'], case['claim'], case['event'], case['outputs'], None)
    terminal = close_segment_fixture(case, bad=bad)
    if bad == 'checkpoint_bytes':
        Path(case['initial']['checkpoint']['path']).write_bytes(b'CORRUPTED_SYNTHETIC_INITIAL_BYTES')
    if bad:
        with pytest.raises(ValueError): m.segment_closed_parent(case['base_ref'], terminal)
    else:
        parent = m.segment_closed_parent(case['base_ref'], terminal)
        assert parent['prefix'] == {'cursor': 1, 'rows_done': 3, 'head': receipt}
        assert parent['initial_receipt'] == case['initial_ref'] and len(parent['receipts']) == len(parent['segments']) == 1
        plan = {**case['plan'], 'parent_terminal': terminal}
        counts = dict.fromkeys(m.SEG_COUNT_KEYS, 0)
        prefix = m.load_segment_prefix(plan, case['context'], counts=counts)
        assert counts['prefix_output_archive_loads'] == 1 and prefix['rows_done'] == 3
        for key in case['outputs']: np.testing.assert_array_equal(prefix['outputs'][key], case['outputs'][key])
        assert prefix['batches'] == [case['event']['batch']]


@pytest.mark.parametrize('bad', ['issue_ns', 'source_rows'])
def test_segment_prefix_rejects_changed_physical_clock_before_archive_decode(monkeypatch, tmp_path, bad):
    case = segment_fixture(monkeypatch, tmp_path)
    m.segment_publish_batch(case['plan'], case['binding'], case['claim'], case['event'], case['outputs'], None)
    terminal = close_segment_fixture(case)
    case['context']['truth']['issue_ns'][0] += int(bad == 'issue_ns')
    case['context']['source_rows'][0] += int(bad == 'source_rows')
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('archive decoded before clock proof'))
    with pytest.raises(ValueError, match='PREFIX_PHYSICAL_CLOCK'):
        m.load_segment_prefix({**case['plan'], 'parent_terminal': terminal}, case['context'])


@pytest.mark.parametrize('old_name', ['001', '002'])
def test_segment_rejects_consumed_old_initial_roots_before_reading_payload(monkeypatch, tmp_path, old_name):
    case = segment_fixture(monkeypatch, tmp_path)
    old = {'path': str(tmp_path/('NATIVE_PROVIDER_HLC_ALL_TF_WHOLE_TRAIN_INITIAL_20261124_'+old_name)/'TERMINAL.json'), 'sha256': 'a'*64}
    with pytest.raises(ValueError): m.segment_closed_parent(case['base_ref'], old)


@pytest.mark.parametrize('bad', [None, 'parameter', 'RNG', 'gradient', 'eval_mode', 'policy'])
def test_segment_continuation_fresh_constructor_comparison_never_loads_weights(monkeypatch, tmp_path, bad):
    case = segment_fixture(monkeypatch, tmp_path)
    model = case['model']
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('old checkpoint loaded'))
    monkeypatch.setattr(model, 'load_state_dict', lambda *a, **k: pytest.fail('model state installed'))
    if bad == 'parameter': model.backbone.weight.data.add_(1)
    elif bad == 'RNG': monkeypatch.setattr(m.core, 'training_rng', lambda: {'changed': True})
    elif bad == 'gradient': model.backbone.weight.grad = torch.zeros_like(model.backbone.weight)
    elif bad == 'eval_mode': model.eval()
    elif bad == 'policy': model._contract = {**model._contract, 'backbone_gradient_policy': {'policy': 'M5_primary'}}
    if bad:
        with pytest.raises(ValueError, match='SEGMENT_FRESH_SAME_STATE_RNG'): m.segment_compare_fresh(model, case['initial'])
    else:
        proof = m.segment_compare_fresh(model, case['initial'])
        assert proof['fresh_constructor_no_weight_load'] is proof['gradients_none'] is True
        assert proof['model_state_sha256'] == case['initial']['model_state_sha256']


def test_segment_exact_preflight_is_payload_free_and_old_default_dispatch_survives(monkeypatch, tmp_path, capsys):
    binding = {'path': '/synthetic/SEGMENT_PLAN.json', 'sha256': 'a'*64}
    plan = {'base_plan': {'path': str(m.SEG_BASE_PLAN), 'sha256': 'b'*64}, 'prefix': {'cursor': 0, 'rows_done': 0, 'head': None}}
    seen = []
    monkeypatch.setattr(m, 'inspect_segment', lambda sha, admitted: (seen.append((sha, admitted)) or (plan, binding, {'path': '/synthetic/POLICY.json', 'sha256': 'c'*64}, {})))
    monkeypatch.setattr(m, 'run_segment', lambda *a: pytest.fail('preflight launched segment'))
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('preflight checkpoint decode'))
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('preflight archive decode'))
    monkeypatch.setattr(sys, 'argv', ['initial', '--all-tf-score-segment-plan-sha256', 'a'*64, '--preflight'])
    m.main(); result = json.loads(capsys.readouterr().out)
    assert seen == [('a'*64, False)] and result['base_plan'] == plan['base_plan'] and result['prefix'] == plan['prefix']
    assert result['claim_created'] is result['cuda_initialized'] is False
    assert result['payloads_decoded'] == result['models'] == result['checkpoint_loads'] == result['optimizer_updates'] == 0
    monkeypatch.setattr(sys, 'argv', sys.argv+['--preflight'])
    with pytest.raises(ValueError, match='EXACT_SEGMENT_CLI'): m.main()
    old_calls = []; monkeypatch.setattr(m, 'main_all_tf', lambda: old_calls.append(True))
    monkeypatch.setattr(sys, 'argv', ['initial', '--all-tf-plan-sha256', 'a'*64, '--preflight'])
    m.main(); assert old_calls == [True]


@pytest.mark.parametrize('failure', [False, True, 'global_after_terminal'])
def test_segment_lifecycle_preserves_durable_files_and_closes_every_invocation(monkeypatch, tmp_path, failure):
    from gx1.contracts import gx1_capped_execution_v1 as guard
    case = segment_fixture(monkeypatch, tmp_path)
    global_failure = failure == 'global_after_terminal'
    binding = {'path': str(m.segment_paths('a'*64)[0]), 'sha256': 'a'*64}
    root = m.segment_paths(binding['sha256'])[1]
    plan = case['plan']; live = {'fixture': 'runtime'}
    hooks, timers = [], []
    monkeypatch.setattr(guard, 'require_native_provider_hlc_all_tf_initial_segment_execution',
        lambda **kwargs: {'plan': binding, 'output_root': str(root)}, raising=False)
    monkeypatch.setattr(m, 'inspect_segment', lambda *a: (plan, binding, case['policy_ref'], live))
    monkeypatch.setattr(m, 'segment_closed_parent', lambda *a: {'receipts': [], 'segments': []})
    monkeypatch.setattr(m.core, 'future_cpu_process_identity', lambda: {'pid': 0})
    monkeypatch.setattr(sys, 'addaudithook', hooks.append)
    monkeypatch.setattr(m.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(m.signal, 'setitimer', lambda *a: timers.append(a))
    monkeypatch.setattr(torch.optim.Optimizer, '__init__', torch.optim.Optimizer.__init__)
    monkeypatch.setattr(torch.Tensor, 'backward', torch.Tensor.backward)
    monkeypatch.setattr(torch.autograd, 'grad', torch.autograd.grad)
    def physical(p, b, claim, runtime, progress):
        assert p is plan and b == binding and runtime is live
        m.segment_publish(root, 'PRESERVED_COMMIT_MARKER.json', {'synthetic': True})
        progress.update(initial_receipt=case['initial_ref'], counts=dict.fromkeys(m.SEG_COUNT_KEYS, 0))
        if failure is True: raise ValueError('SYNTHETIC_STOP_AFTER_DURABLE_MARKER')
        progress['counts'] = dict(plan['limits'])
        progress['prefix'] = {'cursor': 3, 'rows_done': 8, 'head': {'path': '/synthetic/BATCH_0003.json', 'sha256': 'b'*64}}
        return {'fixture_only': True}
    def bookends(p, b, pref, runtime, progress, claim):
        return {'status': 'CLOSED_SEGMENT_REUSE_PROOF_PASS', 'counts': progress['counts']}
    monkeypatch.setattr(m, 'segment_physical', physical)
    monkeypatch.setattr(m, 'segment_bookends', bookends)
    if global_failure:
        publish = m.segment_publish
        def fail_global(root_path, name, value):
            if root_path == m.SEG_ROOT and name == 'RESULT.json':
                assert (root/'RESULT.json').is_file() and (root/'TERMINAL.json').is_file()
                assert m.read(m.ref(root/'TERMINAL.json'))['status'] == 'COMPLETE'
                raise OSError('SYNTHETIC_GLOBAL_RESULT_FAILURE_AFTER_COMPLETE_TERMINAL')
            return publish(root_path, name, value)
        monkeypatch.setattr(m, 'segment_publish', fail_global)
        with pytest.raises(OSError, match='SYNTHETIC_GLOBAL_RESULT_FAILURE_AFTER_COMPLETE_TERMINAL'):
            m.run_segment(binding['sha256'])
    elif failure is True:
        with pytest.raises(ValueError, match='SYNTHETIC_STOP_AFTER_DURABLE_MARKER'): m.run_segment(binding['sha256'])
    else:
        result = m.run_segment(binding['sha256'])
        assert result['result'] == m.ref(m.SEG_ROOT/'RESULT.json')
        global_result = m.read(result['result']); report = global_result['report']
        assert global_result['completed_execution_result'] == m.ref(root/'RESULT.json')
        assert global_result['completed_execution_terminal'] == m.ref(root/'TERMINAL.json')
        assert report['segments'][-1]['terminal'] == global_result['completed_execution_terminal']
        assert report['counts'] == report['physical_execution_counts'] == plan['limits']
    terminal = m.read(m.ref(root/'TERMINAL.json'))
    assert terminal['status'] == ('FAILED_CONSUMED' if failure is True else 'COMPLETE')
    assert terminal['remaining_invocations'] == 0 and terminal['relaunch_allowed'] is False
    if failure is not True: assert terminal['result'] == m.ref(root/'RESULT.json')
    if global_failure: assert (root/'PUBLICATION_FAILURE.json').is_file()
    assert (root/'PRESERVED_COMMIT_MARKER.json').is_file()
    assert (m.SEG_ROOT/'RESULT.json').exists() is (failure is False)
    assert (root/'RESULT.json').exists() is (failure is not True)
    assert (root/'CLOSURE_PROOF.json').is_file() and timers[-1] == (m.signal.ITIMER_REAL, 0)
    assert len(hooks) == 1
    for value in plan['inputs'].values(): hooks[0]('open', (value['path'],))
    hooks[0]('open', (str(root/'BATCH_0001.npz'),))
    hooks[0]('open', (str(m.SEG_ROOT/'INITIAL.pt'),))
    for path in ('/synthetic/OLD_INITIAL001.pt', '/synthetic/OLD_INITIAL002.pt', '/synthetic/FINAL.pt', '/synthetic/FOREIGN.npz'):
        with pytest.raises(RuntimeError, match='PAYLOAD_PATH_FORBIDDEN'): hooks[0]('open', (path,))
    with pytest.raises(RuntimeError, match='NETWORK_FORBIDDEN'): hooks[0]('socket.connect', ())
    with pytest.raises(FileExistsError): m.run_segment(binding['sha256'])


def test_segment_header_authenticates_inputs_metadata_and_frozen_source_bridge(monkeypatch, tmp_path):
    case = segment_fixture(monkeypatch, tmp_path)
    m.segment_validate_header(case['plan'], case['base'], case['base_ref'])
    for field in ('inputs', 'metadata', 'source_bridge'):
        changed = copy.deepcopy(case['plan'])
        if field == 'inputs': changed[field]['feature_archive']['sha256'] = '0'*64
        elif field == 'metadata': changed[field]['target_contract']['contract_sha256'] = '0'*64
        else: changed[field] = {'unqualified': 'bridge'}
        with pytest.raises(ValueError): m.segment_validate_header(changed, case['base'], case['base_ref'])
