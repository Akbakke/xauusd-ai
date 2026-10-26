"""Synthetic extraction mechanics; no native model, checkpoint or market input."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gx1.contracts import entry_native_direction_frozen_extraction_v1 as contract
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.scripts import extract_entry_native_direction_frozen_readout_v1 as worker


def ref(name):
    return {'path': contract.PROGRAM + '/' + name, 'sha256': '0' * 64}


def source_inputs():
    previous = {name: {'path': contract.REPO + '/' + name, 'sha256': digest}
        for name, digest in contract.previous.TRANSPORT_ORIGINAL_SHA256.items()}
    shared = tuple('gx1/synthetic_shared_' + str(i) + '.py' for i in range(6))
    previous.update({n: {'path': contract.REPO + '/' + n, 'sha256': '0' * 64} for n in shared})
    previous.update({f'gx1/synthetic_owner_{i}.py': {'path': contract.REPO + f'/gx1/synthetic_owner_{i}.py',
        'sha256': '0' * 64} for i in range(40)})
    reused = {n: deepcopy(previous[n]) for n in shared}
    name = 'gx1/contracts/entry_direction_probability_reference_v1.py'
    reused[name] = {'path': contract.REPO + '/' + name, 'sha256': contract.TRANSPORT_ORIGINAL_SHA256[name]}
    return {'source_owners': previous}, {'existing_sources_reused_without_change': reused}


def plan(unit='own0U_prefix'):
    original, proposal = source_inputs()
    owners = {**original['source_owners'], **proposal['existing_sources_reused_without_change'],
        **{n: {'path': contract.REPO + '/' + n, 'sha256': '2' * 64}
            for n in (*contract.previous.NEW_FILES, *contract.NEW_FILES)}}
    transitions = {}
    for name in contract.TRANSPORT_ORIGINAL_SHA256:
        current = {'path': contract.REPO + '/' + name, 'sha256': contract.PINNED_PRE_F.get(name, '1' * 64)}
        transitions[name] = {'old276': deepcopy(owners[name]), 'current': current}
        owners[name] = current
    root = contract.PROGRAM + '/F_FROZEN_EXTRACTION_' + unit + '_SYNTHETIC'
    return {'schema_version': contract.SCHEMA, 'operation': contract.OPERATION, 'unit': unit,
        'admission_stage': contract.STAGE, 'proposal': deepcopy(contract.PROPOSAL),
        'design': deepcopy(contract.DESIGN), 'units_design': deepcopy(contract.UNITS_DESIGN),
        'initial_report': deepcopy(contract.INITIAL_REPORT), 'initial_predictions': deepcopy(contract.INITIAL_PREDICTIONS),
        'terminal_report': deepcopy(contract.TERMINAL_REPORT), 'terminal_predictions': deepcopy(contract.TERMINAL_PREDICTIONS),
        'source_owners': owners, 'transport_source_transitions': transitions,
        'run_root': root, 'runtime_root': root + '_RUNTIME',
        'execution_binding_path': root + '_RUNTIME/EXECUTION_BINDING.json',
        'budget': deepcopy(contract.RESOURCE_LIMITS), 'operation_limits': contract.call_limits(unit),
        'max_invocations': 1, 'execution_authorized': True, 'data_reads_authorized': True,
        'checkpoint_reads_authorized': True, 'genuine_forwards_authorized': True,
        'gradient_reverse_authorized': False, 'optimizer_steps_authorized': False, 'test_data_used': False}


@pytest.mark.parametrize('unit', list(contract.UNITS))
def test_exact_units_and_original_numerical_source_closure(unit):
    recipe = plan(unit)
    assert contract.require_plan(recipe) == recipe
    old, proposal = source_inputs()
    assert contract.require_source_closure(recipe, old, proposal) == recipe['source_owners']
    spec = contract.unit_spec(unit)
    assert spec['forwards'] == (spec['rows'] + 15) // 16
    assert contract.call_limits(unit)['reverse'] == contract.call_limits(unit)['autograd_grad'] == 0
    assert contract.call_limits(unit)['torch_load'] == 1
    assert sum(contract.unit_spec(u)['forwards'] for u in contract.UNITS) == 8114


@pytest.mark.parametrize('fault', ['unit', 'reverse', 'optimizer', 'test', 'native_source', 'source_drop',
    'baseline', 'counter', 'root', 'relaunch', 'pre_f_cap'])
def test_scope_or_source_corruption_rejected(fault):
    p = plan()
    if fault == 'unit': p['unit'] = 'u256_prefix'
    elif fault == 'reverse': p['gradient_reverse_authorized'] = True
    elif fault == 'optimizer': p['optimizer_steps_authorized'] = True
    elif fault == 'test': p['test_data_used'] = True
    elif fault == 'native_source': p['source_owners']['gx1/synthetic_owner_0.py']['sha256'] = '3' * 64
    elif fault == 'source_drop': del p['source_owners'][contract.previous.NEW_FILES[0]]
    elif fault == 'baseline': p['terminal_predictions']['sha256'] = '3' * 64
    elif fault == 'counter': p['operation_limits']['native_forward'] += 1
    elif fault == 'root': p['runtime_root'] = p['run_root']
    elif fault == 'relaunch': p['max_invocations'] = True
    elif fault == 'pre_f_cap': p['source_owners']['scripts/gx1_capped_run.sh']['sha256'] = '3' * 64
    with pytest.raises(RuntimeError):
        contract.require_plan(p)
        contract.require_source_closure(p, *source_inputs())


@pytest.mark.parametrize('key', ['gradient_forward', 'reverse', 'autograd_grad', 'engine_reverse'])
def test_reverse_and_extra_operation_fail_before_call(key):
    budget = contract.CallBudget('terminal_later')
    with pytest.raises(RuntimeError, match='CALL_CEILING'):
        budget.start(key)
    assert budget.calls[key] == 0
    with pytest.raises(RuntimeError, match='UNDECLARED|CALL_CEILING'):
        budget.start('optimizer_step')
    with pytest.raises(RuntimeError, match='EXACT_COMPLETED'):
        budget.complete()


def window(unit):
    root = plan(unit)['run_root']
    return contract._seal({'schema_version': contract.WINDOW_SCHEMA, 'recipe': ref('SYNTHETIC_RECIPE.json'),
        'unit': unit, 'invocation_number': 1, 'max_invocation_seconds': 12000,
        'output_directory': root, 'progress_path': root + '/PROGRESS.json',
        'campaign_cursor_path': root + '/DIAGNOSTIC_POINTER.json', 'test_data_used': False}, 'policy_sha256')


@pytest.mark.parametrize('unit', list(contract.UNITS))
def test_complete_pointer_progress_and_partial_rejection(unit):
    w = window(unit)
    assert contract.require_frozen_extraction_window_policy(w, verify_files=False) == w
    n = contract.call_limits(unit)
    pointer = contract.build_frozen_extraction_pointer(recipe_ref=w['recipe'], result_ref=ref('SYNTHETIC_RESULT.json'),
        outcome='COMPLETE', calls=n, returns=n, unit=unit)
    assert contract.require_frozen_extraction_pointer(pointer, expected_recipe=w['recipe'],
        expected_unit=unit, verify_files=False) == pointer
    invocation = {'kind': contract.KIND, 'invocation_sha256': '1' * 64,
        'frozen_extraction_recipe': w['recipe'], 'frozen_extraction_window': w,
        'checkpoint': {'pointer_path': w['campaign_cursor_path']}}
    progress = contract.build_frozen_extraction_progress(window=w, plan={'plan_sha256': '2' * 64},
        invocation=invocation, pointer=pointer, checkpoint_pointer={'path': w['campaign_cursor_path'], 'sha256': '3' * 64})
    assert campaign.require_progress(progress, plan_sha256='2' * 64, invocation=invocation,
        expected_selection_receipt_sha256=None, verify_file=False) == progress
    partial = dict(n); partial['native_forward'] -= 1
    with pytest.raises(RuntimeError, match='POINTER_COMPLETE'):
        contract.build_frozen_extraction_pointer(recipe_ref=w['recipe'], result_ref=ref('SYNTHETIC_RESULT.json'),
            outcome='COMPLETE', calls=n, returns=partial, unit=unit)
    running = contract.build_frozen_extraction_running_progress(window=w, plan={'plan_sha256': '2' * 64},
        invocation=invocation, calls=n, returns=partial)
    with pytest.raises(RuntimeError, match='PROGRESS_IDENTITY'):
        campaign.require_progress(running, plan_sha256='2' * 64, invocation=invocation,
            expected_selection_receipt_sha256=None, verify_file=False)


@pytest.mark.parametrize('fault', ['unit', 'steps', 'time', 'pointer', 'extra'])
def test_window_mutations_rejected(fault):
    value = window('own0U_prefix'); value.pop('policy_sha256')
    if fault == 'unit': value['unit'] = 'terminal_selected_later'
    elif fault == 'steps': value['invocation_number'] = True
    elif fault == 'time': value['max_invocation_seconds'] += 1
    elif fault == 'pointer': value['campaign_cursor_path'] = value['progress_path']
    elif fault == 'extra': value['fit_authorized'] = True
    with pytest.raises(RuntimeError):
        contract.require_frozen_extraction_window_policy(contract._seal(value, 'policy_sha256'), verify_files=False)


def test_guard_is_required_before_payload_or_metadata_import(monkeypatch):
    from gx1.contracts import gx1_capped_execution_v1 as cap
    def stop():
        raise RuntimeError('SYNTHETIC_GUARD_REJECTED')
    monkeypatch.setattr(cap, 'require_guarded_cuda_trainer_execution', stop)
    with pytest.raises(RuntimeError, match='SYNTHETIC_GUARD_REJECTED'):
        worker.run_native_direction_frozen_extraction_window(policy_path=Path('/synthetic/window.json'),
            policy_file_sha256='0' * 64, progress_path=Path('/synthetic/PROGRESS.json'))


def test_capture_is_exact_observation_and_does_not_change_output_or_state():
    torch.manual_seed(17)
    head = torch.nn.Linear(128, 3)
    model = SimpleNamespace(direction_head=head)
    value = torch.arange(16 * 128, dtype=torch.float32).reshape(16, 128) / 1024
    before = {k: v.clone() for k, v in head.state_dict().items()}
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        expected = head(value)
        budget = contract.CallBudget('own0U_prefix')
        capture = worker.HiddenCapture(model, torch=torch, np=np, budget=budget, progress=lambda: None)
        try:
            actual = head(value)
        finally:
            capture.close()
        after = head(value)
    assert torch.equal(actual, expected) and torch.equal(after, expected)
    assert len(capture.parts) == 1 and np.array_equal(capture.parts[0], value.numpy())
    assert budget.calls['panel_forward'] == budget.returns['panel_forward'] == 1
    assert not capture.pending and torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(before[k], v) for k, v in head.state_dict().items())
    assert all(p.grad is None for p in head.parameters())
    assert not head._forward_hooks and not head._forward_pre_hooks


@pytest.mark.parametrize('fault', ['dtype', 'width', 'grad', 'nonfinite', 'too_large'])
def test_capture_rejects_invalid_head_input(fault):
    head = torch.nn.Linear(128, 3); value = torch.zeros((16, 128), dtype=torch.float32)
    if fault == 'dtype': value = value.double()
    elif fault == 'width': value = value[:, :127]
    elif fault == 'grad': value.requires_grad_(True)
    elif fault == 'nonfinite': value[0, 0] = float('nan')
    elif fault == 'too_large': value = torch.zeros((17, 128), dtype=torch.float32)
    capture = worker.HiddenCapture(SimpleNamespace(direction_head=head), torch=torch, np=np,
        budget=contract.CallBudget('own0U_prefix'), progress=lambda: None)
    try:
        with pytest.raises(RuntimeError, match='HEAD_CAPTURE'):
            head(value)
    finally:
        capture.close()


@pytest.mark.parametrize('unit', ['own0U_prefix', 'terminal_later'])
def test_full_declared_synthetic_row_mapping_preserves_batch_presentation(unit):
    spec = contract.unit_spec(unit); n = spec['rows']
    start = 0 if spec['cohort'] == 'prefix' else contract.FOLD['oof_start_row']
    selected = np.arange(start, start + n, dtype=np.int64)[::-1].copy()
    hidden = np.zeros((n, 128), dtype=np.float32)
    hidden[:, 0] = selected
    measured = {'arrays': {'parent_rows': selected.copy(), 'logit_up': np.zeros((n, 3), np.float32)}}
    arrays = worker.canonical_arrays(measured, [hidden], selected=selected, spec=spec,
        weight=np.zeros((3, 128), np.float32), bias=np.zeros(3, np.float32), np=np)
    assert np.array_equal(arrays['parent_rows'], np.arange(start, start + n))
    assert np.array_equal(arrays['hidden'][:, 0], arrays['parent_rows'])
    restored = arrays['parent_rows'][np.argsort(arrays['presentation_index'])]
    assert np.array_equal(restored, selected)
    selected[0] = selected[1]
    with pytest.raises(RuntimeError, match='PRESENTATION_JOIN|EXACT_HIDDEN_COHORT'):
        worker.canonical_arrays(measured, [hidden], selected=selected, spec=spec,
            weight=np.zeros((3, 128), np.float32), bias=np.zeros(3, np.float32), np=np)


def test_bit_parity_rejects_one_ulp_and_dtype_change():
    x = {'parent_rows': np.arange(256, dtype=np.int64), 'logit_up': np.zeros((256, 3), np.float32)}
    y = {k: v.copy() for k, v in x.items()}
    worker.require_parity(x, y, first_rows=256, np=np)
    y['logit_up'][0, 0] = np.nextafter(np.float32(0), np.float32(1))
    with pytest.raises(RuntimeError, match='BIT_PARITY'):
        worker.require_parity(x, y, first_rows=256, np=np)
    y['logit_up'] = x['logit_up'].astype(np.float64)
    with pytest.raises(RuntimeError, match='BIT_PARITY'):
        worker.require_parity(x, y, first_rows=256, np=np)


def test_hardware_policy_is_identical_to_existing_native_guard_profile():
    assert campaign.campaign_hardware_policy(contract.PHASE) == campaign.campaign_hardware_policy(contract.previous.PHASE)
    with pytest.raises(RuntimeError, match='DIRECT_MODULE_FORBIDDEN'):
        worker.main()

def invocation(unit):
    p = plan(unit); w = window(unit)
    argv = [contract.REPO + '/scripts/gx1_capped_run.sh', '--class', 'trainer', '--mem', '20G', '--swap', '512M',
        '--', contract.REPO + '/.venv/bin/python', '-m', contract.MODULE,
        '--window-policy', ref('SYNTHETIC_WINDOW.json')['path'], '--window-policy-file-sha256', '0' * 64,
        '--progress-path', w['progress_path']]
    value = {'schema_version': campaign.INVOCATION_SCHEMA, 'decision': 'PASS', 'invocation_number': 1,
        'invocation_id': 'invocation-0001', 'kind': contract.KIND, 'batch_size': 16,
        'epoch_index': None, 'window_index': 0, 'window_count': 1, 'optimizer_step_budget': 0,
        'expected_success_outcome': 'COMPLETE', 'source_commit': 'a' * 40,
        'execution_manifest': ref('SYNTHETIC_EXECUTION.json'), 'launcher_argv': argv,
        'launcher_argv_sha256': campaign.canonical_sha256(argv), 'progress_path': w['progress_path'],
        'guard_log_path': p['runtime_root'] + '/logs/invocation-0001.signed-guard.log',
        'checkpoint': {'pointer_path': w['campaign_cursor_path'], 'before_mode': 'GENESIS',
            'predecessor_invocation_number': None, 'write_mode': 'ATOMIC_UPDATE'},
        'maximum_wall_seconds': 13800, 'requires_fresh_windows_boot': True,
        'signed_guard_only': True, 'test_data_used': False}
    return contract._seal(value, 'invocation_sha256')


@pytest.mark.parametrize('unit', list(contract.UNITS))
def test_canonical_invocation_binds_same_unit_recipe_and_window(monkeypatch, unit):
    p = plan(unit); w = window(unit); inv = invocation(unit)
    execution = {'prelaunch_manifest': w['recipe'], 'train_session_manifest': ref('SYNTHETIC_WINDOW.json'),
        'prelaunch_manifest_sha256': campaign.canonical_sha256(p), 'train_session_manifest_sha256': w['policy_sha256']}
    campaign.require_invocation(inv, source_repo=Path(contract.REPO), source_commit='a' * 40, verify_files=False)
    monkeypatch.setattr(campaign, 'require_binding', lambda r, **kw: contract.plain_ref(r))
    monkeypatch.setattr(campaign, 'read_bound_json', lambda *args: deepcopy(w))
    monkeypatch.setattr(contract, 'require_frozen_extraction_recipe', lambda *args, **kw: deepcopy(p))
    contract.require_frozen_extraction_invocation_execution(inv, execution,
        source_repo=Path(contract.REPO), source_commit='a' * 40)
    assert inv['frozen_extraction_window'] == w
    assert inv['frozen_extraction_recipe'] == w['recipe']
    p['unit'] = 'terminal_prefix' if unit != 'terminal_prefix' else 'own0U_prefix'
    with pytest.raises(RuntimeError, match='INVOCATION_BINDING'):
        contract.require_frozen_extraction_invocation_execution(inv, execution,
            source_repo=Path(contract.REPO), source_commit='a' * 40)


@pytest.mark.parametrize('fault', ['updates', 'batch', 'resume', 'extra_window', 'smoke', 'unguarded'])
def test_campaign_rejects_extraction_scope_expansion(fault):
    inv = invocation('own0U_prefix'); inv.pop('invocation_sha256')
    if fault == 'updates': inv['optimizer_step_budget'] = 1
    elif fault == 'batch': inv['batch_size'] = 8
    elif fault == 'resume': inv['expected_success_outcome'] = 'RESUMABLE_OR_COMPLETE'
    elif fault == 'extra_window': inv['window_count'] = 2
    elif fault == 'smoke':
        inv['launcher_argv'].insert(1, '--attended-smoke')
        inv['launcher_argv_sha256'] = campaign.canonical_sha256(inv['launcher_argv'])
    elif fault == 'unguarded': inv['signed_guard_only'] = False
    with pytest.raises(campaign.RandomAccessCampaignError):
        campaign.require_invocation(contract._seal(inv, 'invocation_sha256'),
            source_repo=Path(contract.REPO), source_commit='a' * 40, verify_files=False)


@pytest.mark.parametrize('decision', ['COMPLETE_FROZEN_READOUT_EXTRACTION', 'FAILED'])
def test_actual_dispatch_function_keeps_guard_first_and_rejects_failed_worker(monkeypatch, decision):
    # Compile the actual source function alone, avoiding unrelated native model
    # module import side effects. The eventual full import remains a runtime gate.
    import ast
    source_path = Path(contract.REPO) / 'gx1/scripts/run_unified_exit_native_candidate_window_v1.py'
    tree = ast.parse(source_path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_window')
    calls = []
    def guard(**kw):
        assert kw == {'execution_tier': 'canonical'}
        calls.append('guard')
    def context(*args):
        calls.append('context')
        return {'schema_version': contract.WINDOW_SCHEMA}, {}, {}, {}
    def run(**kw):
        calls.append('worker')
        return {'decision': decision}
    monkeypatch.setattr(worker, 'run_native_direction_frozen_extraction_window', run)
    namespace = {'Path': Path, 'Any': object, 'file_verification_scope': campaign.file_verification_scope,
        'native': SimpleNamespace(trainer=SimpleNamespace(_require_cuda_trainer_guard_execution=guard)), '_context': context}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), 'exec'), namespace)
    args = {'policy_path': Path('/synthetic/window.json'), 'policy_file_sha256': '0' * 64,
        'progress_path': Path('/synthetic/PROGRESS.json')}
    if decision == 'FAILED':
        with pytest.raises(RuntimeError, match='FROZEN_EXTRACTION_FAILED'):
            namespace['run_window'](**args)
    else:
        assert namespace['run_window'](**args) == {'decision': decision}
    assert calls == ['guard', 'context', 'worker']
