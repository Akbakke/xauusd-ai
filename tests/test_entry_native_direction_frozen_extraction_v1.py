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
    for index, (name, (old, _current)) in enumerate(contract.GC_METADATA_SOURCE_TRANSITIONS.items(), start=38):
        del previous[f'gx1/synthetic_owner_{index}.py']
        previous[name] = {'path': contract.REPO + '/' + name, 'sha256': old}
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
    value = {'schema_version': contract.SCHEMA, 'operation': contract.OPERATION, 'unit': unit,
        'admission_stage': contract.STAGE, 'proposal': deepcopy(contract.PROPOSAL),
        'design': deepcopy(contract.DESIGN), 'units_design': contract.unit_design(unit),
        'initial_report': deepcopy(contract.INITIAL_REPORT), 'initial_predictions': deepcopy(contract.INITIAL_PREDICTIONS),
        'terminal_report': deepcopy(contract.TERMINAL_REPORT), 'terminal_predictions': deepcopy(contract.TERMINAL_PREDICTIONS),
        'source_owners': owners, 'transport_source_transitions': transitions,
        'run_root': root, 'runtime_root': root + '_RUNTIME',
        'execution_binding_path': root + '_RUNTIME/EXECUTION_BINDING.json',
        'budget': contract.resource_limits(unit), 'operation_limits': contract.call_limits(unit),
        'max_invocations': 1, 'execution_authorized': True, 'data_reads_authorized': True,
        'checkpoint_reads_authorized': True, 'genuine_forwards_authorized': True,
        'gradient_reverse_authorized': False, 'optimizer_steps_authorized': False, 'test_data_used': False}
    if unit in contract.PARTITION_UNITS:
        value['gc_layout_metadata_parity'] = deepcopy(contract.GC_LAYOUT_METADATA_PARITY)
        for name, (_old, current) in contract.GC_METADATA_SOURCE_TRANSITIONS.items():
            value['source_owners'][name] = {'path': contract.REPO + '/' + name, 'sha256': current}
    return value


@pytest.mark.parametrize('unit', [*contract.UNITS, contract.TIMING_UNIT, *contract.PARTITION_UNITS])
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


@pytest.mark.parametrize('unit', [*contract.UNITS, contract.TIMING_UNIT, *contract.PARTITION_UNITS])
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


@pytest.mark.parametrize('unit', [*contract.UNITS, contract.TIMING_UNIT, *contract.PARTITION_UNITS])
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



@pytest.mark.parametrize('schema,seconds,accepted', [
    (contract.WINDOW_SCHEMA, 12000, True),
    ('gx1_native_direction_checkpoint_diagnostic_window_policy_v1', 12000, True),
    ('gx1_native_direction_prefix_fit_window_policy_v1', 12000, True),
    ('gx1_native_direction_research_window_policy_v1', 12000, True),
    ('gx1_native_candidate_window_policy_v1', 12000, True),
    ('unknown_window', 12000, False),
    (contract.WINDOW_SCHEMA, 11999, False),
    (contract.WINDOW_SCHEMA, 12000.0, False),
    (contract.WINDOW_SCHEMA, True, False),
])
def test_actual_cap_timeout_expression_accepts_exact_windows_only(tmp_path, monkeypatch, capsys,
        schema, seconds, accepted):
    import json
    import re
    import sys

    source = (Path(contract.REPO) / 'scripts/gx1_capped_run.sh').read_text()
    lines = [line for line in source.splitlines() if 'native_seconds=$(' in line]
    assert len(lines) == 1
    expression = re.search(r" -c '([^']+)'", lines[0]).group(1)
    window = tmp_path / 'SYNTHETIC_WINDOW.json'
    window.write_text(json.dumps({'schema_version': schema, 'max_invocation_seconds': seconds}))
    monkeypatch.setattr(sys, 'argv', ['-c', str(window)])
    if accepted:
        exec(compile(expression, 'actual_cap_timeout_expression', 'exec'), {})
        assert capsys.readouterr().out == '12000\n'
    else:
        with pytest.raises(AssertionError):
            exec(compile(expression, 'actual_cap_timeout_expression', 'exec'), {})
        assert capsys.readouterr().out == ''


@pytest.mark.parametrize('logs_exist', [True, False])
def test_actual_cap_F_log_parent_preserves_fresh_output_claim(tmp_path, logs_exist):
    import json
    import subprocess
    import sys

    source = (Path(contract.REPO) / 'scripts/gx1_capped_run.sh').read_text()
    begin = source.index('    TRAINER_OUT_BUNDLE_DIR="${target_args[8]}"',
        source.index('if [[ "$module" == "$NATIVE_CANDIDATE_WINDOW_MODULE" ]]'))
    fragment = source[begin:source.index('\n    return', begin)]
    output = tmp_path / 'fresh_output'
    logs = tmp_path / 'runtime' / 'logs'
    if logs_exist:
        logs.mkdir(parents=True)
    progress = output / 'PROGRESS.json'
    guard_log = logs / 'invocation-0001.signed-guard.log'
    window = tmp_path / 'SYNTHETIC_WINDOW.json'
    window.write_text(json.dumps({'schema_version': contract.WINDOW_SCHEMA}))
    # The exact production selection fragment; the final directory assertion
    # exercises its required existing-parent precondition without starting cap.
    script = 'set -euo pipefail\nCANONICAL_TRAINER_PYTHON="$1"\n' + \
        'target_args=(unused unused unused unused "$2" unused unused unused "$3")\n' + \
        'GX1_CAMPAIGN_GUARD_LOG_PATH="$4"\n' + fragment + \
        '\n[[ "$TRAINER_OUT_BUNDLE_DIR" == "$5" ]]\n[[ -d "${TRAINER_OUT_BUNDLE_DIR%/*}" ]]\n'
    result = subprocess.run(['bash', '--noprofile', '--norc', '-c', script, '--', sys.executable,
        str(window), str(progress), str(guard_log), str(guard_log)],
        capture_output=True, text=True, timeout=10)
    assert result.returncode == (0 if logs_exist else 1), result.stderr
    assert not output.exists() and not progress.exists() and not guard_log.exists()
    assert logs.exists() is logs_exist

def test_timing_unit_does_not_replace_full_readout_cohorts():
    spec = contract.unit_spec(contract.TIMING_UNIT)
    assert spec['diagnostic_only'] and spec['rows'] == 256 and spec['forwards'] == 16
    assert spec['state'] == contract.unit_spec('own0U_prefix')['state']
    assert spec['parity'] == contract.INITIAL_PREDICTIONS
    assert contract.TIMING_UNIT not in contract.UNITS and len(contract.UNITS) == 4
    assert contract.resource_limits(contract.TIMING_UNIT)['max_wall_seconds'] == 600
    assert all(contract.resource_limits(u) == contract.RESOURCE_LIMITS for u in contract.UNITS)


@pytest.mark.parametrize('fault', ['full_design', 'forward17', 'wall601', 'full_wall', 'another_unit'])
def test_timing_plan_cannot_expand_scope(fault):
    value = plan(contract.TIMING_UNIT)
    if fault == 'full_design': value['units_design'] = deepcopy(contract.UNITS_DESIGN)
    elif fault == 'forward17': value['operation_limits']['panel_forward'] = 17
    elif fault == 'wall601': value['budget']['max_wall_seconds'] = 601
    elif fault == 'full_wall': value['budget'] = deepcopy(contract.RESOURCE_LIMITS)
    elif fault == 'another_unit': value['unit'] = 'terminal_prefix_timing256'
    with pytest.raises(RuntimeError):
        contract.require_plan(value)


def test_timing_canonical_rows_require_original_parity_parent_order():
    spec = contract.unit_spec(contract.TIMING_UNIT)
    parents = np.arange(1000, 1256, dtype=np.int64)[::-1].copy()
    hidden = np.zeros((256, 128), np.float32); hidden[:, 0] = parents
    measured = {'arrays': {'parent_rows': parents.copy(), 'logit_up': np.zeros((256, 3), np.float32)}}
    kwargs = dict(selected=parents, spec=spec, weight=np.zeros((3, 128), np.float32),
        bias=np.zeros(3, np.float32), np=np)
    result = worker.canonical_arrays(measured, [hidden], reference_parents=parents.copy(), **kwargs)
    assert np.array_equal(result['parent_rows'], np.sort(parents))
    assert np.array_equal(result['hidden'][:, 0], result['parent_rows'])
    for reference in (None, parents[::-1].copy(), parents.astype(np.int32)):
        with pytest.raises(RuntimeError, match='TIMING_ORIGINAL_PARENTS'):
            worker.canonical_arrays(measured, [hidden], reference_parents=reference, **kwargs)


def test_phase_timing_inclusive_nested_wall_and_cpu():
    outer, inner = object(), object()
    clocks = iter((0, 10, 40, 100)); cpus = iter((0, 5, 15, 50))
    timer = worker.PhaseTimings({outer: 'outer', inner: 'inner'},
        wall=lambda: next(clocks), cpu=lambda: next(cpus))
    a, b = SimpleNamespace(f_code=outer), SimpleNamespace(f_code=inner)
    for frame, event in ((a, 'call'), (b, 'call'), (b, 'return'), (a, 'return')):
        timer.observe(frame, event)
    report = timer.report()
    assert report['pending_frames'] == 0
    assert report['phases']['outer']['wall_seconds'] == 100 / 1e9
    assert report['phases']['inner']['wall_seconds'] == 30 / 1e9
    assert report['phases']['outer']['cpu_seconds'] == 50 / 1e9
    assert report['phases']['inner']['cpu_seconds'] == 10 / 1e9
    with pytest.raises(RuntimeError, match='TIMING_COMPLETE'):
        timer.require_complete(16)


def test_phase_timing_delegates_real_profile_events_and_preserves_result():
    import sys
    def sample(value):
        return sum(range(value))
    budget = contract.CallBudget(contract.TIMING_UNIT)
    prior_events = []
    old = sys.getprofile()
    def prior(frame, event, arg):
        if old is not None: old(frame, event, arg)
        if frame.f_code is sample.__code__: prior_events.append(event)
    guard = worker.saved._profile_budget(budget, {sample.__code__: 'batch'}, {'name': 'EXTRACT'}, prior)
    timer = worker.PhaseTimings({sample.__code__: 'sample'})
    old = sys.getprofile()
    try:
        sys.setprofile(timer.profile(guard))
        result = sample(11)
    finally:
        sys.setprofile(old)
    assert result == 55 and budget.calls['batch'] == budget.returns['batch'] == 1
    assert prior_events == ['call', 'c_call', 'c_return', 'return']
    assert timer.report()['phases']['sample']['calls'] == timer.report()['phases']['sample']['returns'] == 1


def test_timing_observer_cannot_bypass_original_forbidden_operation_guard():
    import sys
    namespace = {}
    exec(compile('def fit_forbidden():\n    return 123\n', contract.REPO + '/tests/synthetic_timing_guard.py', 'exec'), namespace)
    forbidden = namespace['fit_forbidden']
    budget = contract.CallBudget(contract.TIMING_UNIT)
    old = sys.getprofile()
    guard = worker.saved._profile_budget(budget, {}, {'name': 'EXTRACT'}, old)
    timer = worker.PhaseTimings({forbidden.__code__: 'forbidden'})
    try:
        sys.setprofile(timer.profile(guard))
        with pytest.raises(RuntimeError, match='FORBIDDEN_OPERATION'):
            forbidden()
    finally:
        sys.setprofile(old)
    assert timer.report()['phases'] == {} and not any(budget.calls.values())

def test_timing_result_cannot_be_promoted_to_full_cohort_or_partial_success(monkeypatch):
    unit = contract.TIMING_UNIT
    limits = contract.call_limits(unit)
    recipe_ref, result_ref = ref('TIMING_RECIPE.json'), ref('TIMING_RESULT.json')
    pointer = contract.build_frozen_extraction_pointer(recipe_ref=recipe_ref, result_ref=result_ref,
        outcome='COMPLETE', calls=limits, returns=limits, unit=unit)
    phases = {name: {'calls': count, 'returns': count} for name, count in
        {'prepare': 1, 'evaluate': 1, 'batch': 16, 'context_tags': 16,
            'device_batch': 16, 'forward': 16, 'hidden_capture': 16}.items()}
    result = {'recipe': recipe_ref, 'unit': unit, 'optimizer_steps': 0, 'native_optimizer_steps': 0,
        'test_data_used': False, 'observed_calls': limits, 'observed_returns': limits,
        'decision': 'COMPLETE_FROZEN_READOUT_EXTRACTION', 'representation_rows': 256, 'hidden_width': 128,
        'originals_stat_preserved': True, 'source_unchanged': True, 'representations': ref('SYNTHETIC_HIDDEN.npz'),
        'diagnostic_only': True, 'phase_timings': {'pending_frames': 0,
            'scope': 'inclusive_wall_and_process_cpu', 'phases': phases}}
    monkeypatch.setattr(campaign, 'require_binding', lambda value, **kwargs: value)
    monkeypatch.setattr(campaign, 'read_bound_json', lambda *args: result)
    def validate():
        return contract.require_frozen_extraction_pointer(pointer, expected_recipe=recipe_ref,
            expected_unit=unit, verify_files=True)
    assert validate() == pointer
    result['diagnostic_only'] = False
    with pytest.raises(RuntimeError, match='RESULT_DIAGNOSTIC_ROLE'): validate()
    result['diagnostic_only'] = True; result['representation_rows'] = contract.FOLD['prefix_rows']
    with pytest.raises(RuntimeError, match='RESULT_COMPLETE_DOMAIN'): validate()
    result['representation_rows'] = 256; phases['forward']['returns'] = 15
    with pytest.raises(RuntimeError, match='RESULT_TIMING_COMPLETE'): validate()


def test_phase_timing_cached_body_identity_and_cache_hits():
    from functools import lru_cache
    import sys
    import ast
    source = Path(contract.REPO) / 'gx1/contracts/entry_native_direction_feature_view_v1.py'
    node = next(n for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == '_canonical')
    # Compile the exact cached owner declaration, without unrelated import side effects.
    module = ast.Module(body=ast.parse('from __future__ import annotations').body + [node],
        type_ignores=[])
    namespace = {'lru_cache': lru_cache}
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
    actual_cached = namespace['_canonical']

    @lru_cache(maxsize=2)
    def cached(value):
        return value + 1

    # Include the actual decorated owner that failed at timing001 startup.
    codes = worker.phase_codes({cached: 'cached', actual_cached: 'canonical_layout'})
    assert codes[actual_cached.__wrapped__.__code__] == 'canonical_layout'
    timer = worker.PhaseTimings(codes)
    previous = sys.getprofile()
    def guard(frame, event, arg):
        if previous is not None:
            previous(frame, event, arg)
    sys.setprofile(timer.profile(guard))
    try:
        assert cached(3) == cached(3) == 4
        assert cached(7) == 8
    finally:
        sys.setprofile(previous)
    phase = timer.report()['phases']['cached']
    assert phase['calls'] == phase['returns'] == 2
    assert cached.cache_info().hits == 1 and cached.cache_info().misses == 2
    assert timer.report()['pending_frames'] == 0


def _profile_fixture_code(filename, name):
    from types import CodeType
    module = compile('def ' + name + '():\n    return None\n', filename, 'exec')
    return next(value for value in module.co_consts if isinstance(value, CodeType))


@pytest.mark.parametrize('filename,name,blocked', [('/home/andre2/src/GX1_CURRENT/gx1/synthetic_guard.py', 'fit', True), ('/home/andre2/src/GX1_CURRENT/gx1/synthetic_guard.py', 'fit_more', True), ('/home/andre2/src/GX1_CURRENT/gx1/synthetic_guard.py', 'partial_fit', True), ('/home/andre2/src/GX1_CURRENT/gx1/synthetic_guard.py', '_optimizer_step_with_finite_gradients', True), ('/home/andre2/src/GX1_CURRENT/gx1/synthetic_guard.py', '_save_state', True), ('/home/andre2/src/GX1_CURRENT/gx1/synthetic_guard.py', '_save_snapshot', True), ('/home/andre2/src/GX1_CURRENT/gx1/synthetic_guard.py', 'load_m1_feature_surface', True), ('/home/andre2/src/GX1_CURRENT/gx1/synthetic_guard.py', 'load_current', True), ('/home/andre2/src/GX1_CURRENT/gx1/synthetic_guard.py', 'query_dynamic_m1_prefix', True), ('/home/andre2/src/GX1_CURRENT/gx1/synthetic_guard.py', 'read_dynamic_m5_target_source', True), ('/home/andre2/src/GX1_CURRENT/gx1/synthetic_guard.py', 'build_dynamic_m5_batch_targets', True), ('/home/andre2/src/GX1_CURRENT/gx1/synthetic_guard.py', '_dynamic_m5_optimizer_task_losses', True), ('/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2/synthetic_guard.py', 'fit', True), ('/site-packages/torch/optim/adam.py', 'step', True), ('/site-packages/torch/autograd/x.py', 'backward', True), ('/site-packages/torch/serialization.py', 'save', True), ('/site-packages/sklearn/linear.py', 'fit', True), ('/site-packages/sklearn/linear.py', 'fit_more', True), ('/external/analysis.py', 'fit', False), ('/site-packages/torch/optim/adam.py', 'forward', False), ('/site-packages/torch/serialization.py', 'load', False), ('/home/andre2/src/GX1_CURRENT/gx1/synthetic_guard.py', 'evaluate', False)])
def test_profile_cache_preserves_forbidden_misses_and_hits(filename, name, blocked):
    seen = []
    budget = SimpleNamespace(start=lambda key: None, returned=lambda key: None)
    profile = worker.saved._profile_budget(budget, {}, {'name': 'EVALUATE'},
        lambda frame, event, arg: seen.append(event))
    frame = SimpleNamespace(f_code=_profile_fixture_code(filename, name))
    for _ in range(2):
        if blocked:
            with pytest.raises(RuntimeError, match='FORBIDDEN_OPERATION:' + name):
                profile(frame, 'call', None)
        else:
            for event in ('call', 'return', 'c_call', 'c_return'):
                profile(frame, event, None)
    assert seen == ([] if blocked else ['call', 'return', 'c_call', 'c_return'] * 2)


def test_profile_cached_constructor_still_checks_live_phase():
    seen = []
    budget = SimpleNamespace(start=lambda key: None, returned=lambda key: None)
    phase = {'name': 'CONSTRUCT'}
    profile = worker.saved._profile_budget(budget, {}, phase,
        lambda frame, event, arg: seen.append(event))
    frame = SimpleNamespace(f_code=_profile_fixture_code('/site-packages/torch/nn/modules/module.py', '__init__'))
    profile(frame, 'call', None)
    phase['name'] = 'EVALUATE'
    with pytest.raises(RuntimeError, match='EXTRA_MODULE_CONSTRUCTION'):
        profile(frame, 'call', None)
    phase['name'] = 'CONSTRUCT'
    profile(frame, 'call', None)
    assert seen == ['call', 'call']


def test_profile_cached_calls_keep_every_budget_check_and_return():
    budget = contract.CallBudget(contract.TIMING_UNIT)
    code = _profile_fixture_code('/synthetic/batch.py', 'batch')
    frame = SimpleNamespace(f_code=code)
    seen = []
    profile = worker.saved._profile_budget(budget, {code: 'batch'}, {'name': 'EVALUATE'},
        lambda frame, event, arg: seen.append(event))
    for _ in range(16):
        profile(frame, 'call', None)
        profile(frame, 'return', None)
    with pytest.raises(RuntimeError, match='CALL_CEILING_batch'):
        profile(frame, 'call', None)
    assert budget.calls['batch'] == budget.returns['batch'] == 16
    assert seen == ['call', 'return'] * 16


@pytest.mark.parametrize('reverse', [False, True])
def test_profile_cache_keeps_distinct_filename_authority(reverse):
    # Code objects may compare equal despite different filenames. The guard's
    # original authority is the exact filename/name pair, not code equality.
    allowed = _profile_fixture_code('/external/synthetic.py', 'fit')
    forbidden = allowed.replace(co_filename=contract.REPO + '/gx1/synthetic.py')
    order = [(allowed, False), (forbidden, True)]
    if reverse:
        order.reverse()
    profile = worker.saved._profile_budget(
        SimpleNamespace(start=lambda key: None, returned=lambda key: None), {}, {'name': 'EVALUATE'})
    for code, blocked in order * 2:
        frame = SimpleNamespace(f_code=code)
        if blocked:
            with pytest.raises(RuntimeError, match='FORBIDDEN_OPERATION:fit'):
                profile(frame, 'call', None)
        else:
            profile(frame, 'call', None)


def metadata_transition_plan(unit=None):
    value = plan(contract.TIMING_UNIT if unit is None else unit)
    value['gc_layout_metadata_parity'] = deepcopy(contract.GC_LAYOUT_METADATA_PARITY)
    for name, (_old, current) in contract.GC_METADATA_SOURCE_TRANSITIONS.items():
        value['source_owners'][name] = {'path': contract.REPO + '/' + name, 'sha256': current}
    return value


def test_exact_gc_metadata_transition_preserves_all_other_owners_and_caps():
    value = metadata_transition_plan()
    original, proposal = source_inputs()
    assert contract.require_plan(value) == value
    assert contract.require_source_closure(value, original, proposal) == value['source_owners']
    assert value['budget'] == plan(contract.TIMING_UNIT)['budget']
    assert value['operation_limits'] == plan(contract.TIMING_UNIT)['operation_limits']


@pytest.mark.parametrize('fault', ['proof', 'gc_source', 'prefix_source', 'other_source', 'original_gc', 'missing_bridge'])
def test_gc_metadata_transition_rejects_unqualified_changes(fault):
    value = metadata_transition_plan()
    original, proposal = source_inputs()
    paths = tuple(contract.GC_METADATA_SOURCE_TRANSITIONS)
    if fault == 'proof': value['gc_layout_metadata_parity']['sha256'] = '9' * 64
    elif fault == 'gc_source': value['source_owners'][paths[0]]['sha256'] = '9' * 64
    elif fault == 'prefix_source': value['source_owners'][paths[1]]['sha256'] = '9' * 64
    elif fault == 'other_source': value['source_owners']['gx1/synthetic_owner_0.py']['sha256'] = '9' * 64
    elif fault == 'original_gc': original['source_owners'][paths[0]]['sha256'] = '9' * 64
    else: del value['gc_layout_metadata_parity']
    with pytest.raises(RuntimeError):
        contract.require_plan(value)
        contract.require_source_closure(value, original, proposal)


@pytest.mark.parametrize('unit', list(contract.UNITS))
def test_gc_metadata_transition_does_not_admit_full_extraction(unit):
    with pytest.raises(RuntimeError, match='GC_METADATA_TIMING_SCOPE'):
        contract.require_plan(metadata_transition_plan(unit))


def _prefix_metadata_case():
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as prefix
    fold = {'fold_id': 1, 'prefix_rows': 4, 'train_cutoff_ns': 10}
    fitted_ref, constructor_ref = ref('SYNTHETIC_FITTED.json'), ref('SYNTHETIC_CONSTRUCTOR.json')
    item = {'fold_id': 1, 'prefix_parent_rows': 4, 'cutoff_ns': 10, 'output': fitted_ref,
        'combined_contract_sha256': 'a' * 64, 'base_normalization_sha256': 'b' * 64,
        'gc_layout_sha256': 'c' * 64, 'gc_normalization_sha256': 'd' * 64}
    gc_path = contract.REPO + '/gx1/contracts/entry_gc_mtf_history_v1.py'
    old_gc, current_gc = contract.GC_METADATA_SOURCE_TRANSITIONS['gx1/contracts/entry_gc_mtf_history_v1.py']
    other = {'path': contract.REPO + '/gx1/synthetic_view.py', 'sha256': '5' * 64}
    manifest = {'schema_version': 'gx1_native_direction_prefix_fit_input_manifest_v1',
        'source_design': prefix.SOURCE_DESIGN, 'physical_provider_plan': ref('SYNTHETIC_PROVIDER.json'),
        'physical_provider_worker': ref('SYNTHETIC_PROVIDER.py'), 'prefix_normalization': item,
        'constructor_metadata': constructor_ref, 'view_source_bindings': {
            'gc_gather_source': {'path': gc_path, 'sha256': old_gc}, 'other': other},
        'targets': ref('SYNTHETIC_TARGETS.npz'), 'archive_members': list(prefix.PRICE_MEMBERS),
        'state_qualification': ref('SYNTHETIC_STATE.json'),
        'native_private_qualification': prefix.PRIVATE_QUALIFICATION,
        'input_qualification': prefix.INPUT_QUALIFICATION}
    fitted = {'contract_sha256': item['combined_contract_sha256'],
        'base_normalization': {'contract_sha256': item['base_normalization_sha256']},
        'gc_mtf_history': {'layout': {'contract_sha256': item['gc_layout_sha256']},
            'normalization': {'contract_sha256': item['gc_normalization_sha256']}}}
    launch = {'input_manifest': ref('SYNTHETIC_INPUT_MANIFEST.json'), 'fold': fold}
    old_plan = {'worker': manifest['physical_provider_worker']}
    docs = {launch['input_manifest']['path']: manifest, prefix.SOURCE_DESIGN['path']: {'folds': [fold]},
        prefix.INPUT_QUALIFICATION['path']: {'normalizations': [item]}, prefix.PRIVATE_QUALIFICATION['path']: {},
        manifest['state_qualification']['path']: {}, manifest['physical_provider_plan']['path']: old_plan,
        fitted_ref['path']: fitted, constructor_ref['path']: {}}
    physical = SimpleNamespace(provider_module=lambda: object(), validate_inputs=lambda *args: (
        {}, {'axes': {'source': manifest['targets']}}, {'metadata': {}}))
    hashes = {gc_path: {'path': gc_path, 'sha256': current_gc}, other['path']: deepcopy(other)}
    common = SimpleNamespace(STATE_QUALIFICATION=manifest['state_qualification'],
        _read=lambda binding, metadata: deepcopy(docs[binding['path']]), _module=lambda binding: physical,
        _hash_file=lambda path: deepcopy(hashes[path]))
    return prefix, launch, common, manifest, hashes


@pytest.mark.parametrize('mode', ['default', 'qualified', 'wrong_proof', 'other_source', 'wrong_gc'])
def test_actual_prefix_metadata_checks_exact_transition_without_relabeling_manifest(mode):
    prefix, launch, common, manifest, hashes = _prefix_metadata_case()
    proof = None if mode == 'default' else deepcopy(contract.GC_LAYOUT_METADATA_PARITY)
    if mode == 'wrong_proof': proof['sha256'] = '8' * 64
    if mode in ('other_source', 'wrong_gc'):
        key = 'other' if mode == 'other_source' else 'gc_gather_source'
        hashes[manifest['view_source_bindings'][key]['path']]['sha256'] = '8' * 64
    if mode != 'qualified':
        with pytest.raises(RuntimeError, match='VIEW_SOURCE_SHA|GC_METADATA_PARITY_REF'):
            prefix._metadata(launch, common, gc_layout_metadata_parity=proof)
    else:
        actual = prefix._metadata(launch, common, gc_layout_metadata_parity=proof)
        assert actual['manifest'] == manifest
        assert actual['manifest']['view_source_bindings']['gc_gather_source']['sha256'] != hashes[
            manifest['view_source_bindings']['gc_gather_source']['path']]['sha256']


@pytest.mark.parametrize('parent', list(contract.UNITS))
def test_partition_reassembly_preserves_all_rows_bits_and_original_B16_calls(parent):
    base = contract.unit_spec(parent)
    start = 0 if base['cohort'] == 'prefix' else contract.FOLD['oof_start_row']
    order = np.arange(start, start + base['rows'], dtype=np.int64)
    if base['cohort'] == 'prefix':
        order = np.roll(order[::-1], 113).copy()
    hidden = np.zeros((len(order), 128), np.float32)
    hidden[:, 0] = order
    hidden[:, 127] = (order % 17).astype(np.float32)
    measured = {'arrays': {'parent_rows': order.copy(),
        'logit_up': np.repeat((order % 7).astype(np.float32)[:, None], 3, axis=1),
        'sign': np.repeat((order % 3 - 1).astype(np.int8)[:, None], 3, axis=1),
        'observed': np.repeat((order % 5 != 0)[:, None], 3, axis=1),
        'decision_ns': order * np.int64(300_000_000_000)}}
    weight, bias = np.zeros((3, 128), np.float32), np.zeros(3, np.float32)
    expected = worker.canonical_arrays(measured, [hidden], selected=order, spec=base,
        weight=weight, bias=bias, np=np)
    pieces, batches, cursor = [], [], 0
    units = [u for u, v in contract.PARTITION_UNITS.items() if v[0] == parent]
    for unit in units:
        spec = contract.unit_spec(unit)
        part = spec['partition']; lo, hi = part['presentation_start'], part['presentation_end']
        assert lo == cursor and (lo % 16 == 0) and (hi % 16 == 0 or hi == len(order))
        cursor = hi
        selected = worker.select_partition_parents(order, spec, np)
        batches.extend([selected[i:i+16].tolist() for i in range(0, len(selected), 16)])
        arrays = {k: v[lo:hi].copy() for k, v in measured['arrays'].items()}
        pieces.append(worker.canonical_arrays({'arrays': arrays}, [hidden[lo:hi]],
            selected=selected, spec=spec, weight=weight, bias=bias, np=np, reference_parents=order))
        assert contract.resource_limits(unit) == contract.resource_limits(parent)
        assert contract.call_limits(unit)['reverse'] == contract.call_limits(unit)['autograd_grad'] == 0
    assert cursor == len(order)
    assert batches == [order[i:i+16].tolist() for i in range(0, len(order), 16)]
    ids = np.concatenate([x['parent_rows'] for x in pieces])
    sort = np.argsort(ids, kind='stable')
    for key, value in expected.items():
        actual = pieces[0][key] if key in ('head_weight', 'head_bias') else np.ascontiguousarray(
            np.concatenate([x[key] for x in pieces])[sort])
        assert actual.dtype == value.dtype and actual.shape == value.shape
        assert actual.tobytes() == value.tobytes(), key
    assert sum(contract.unit_spec(u)['forwards'] for u in contract.PARTITION_UNITS) == 8114


@pytest.mark.parametrize('fault', ['missing_bridge', 'wrong_design', 'wrong_source', 'extra_range',
    'forward_count', 'time_budget', 'wrong_unit', 'parity_proof'])
def test_partition_plan_rejects_scope_or_source_expansion(fault):
    value = plan('terminal_prefix_part2')
    if fault == 'missing_bridge': del value['gc_layout_metadata_parity']
    elif fault == 'wrong_design': value['units_design'] = deepcopy(contract.UNITS_DESIGN)
    elif fault == 'wrong_source': value['source_owners']['gx1/synthetic_owner_0.py']['sha256'] = '9' * 64
    elif fault == 'extra_range': value['presentation_start'] = 16
    elif fault == 'forward_count': value['operation_limits']['native_forward'] += 1
    elif fault == 'time_budget': value['budget']['max_wall_seconds'] += 1
    elif fault == 'wrong_unit': value['unit'] = 'terminal_prefix_part4'
    else: value['gc_layout_metadata_parity']['sha256'] = '9' * 64
    with pytest.raises(RuntimeError):
        contract.require_plan(value)
        contract.require_source_closure(value, *source_inputs())


@pytest.mark.parametrize('fault', ['shift', 'boolean_start', 'missing', 'duplicate', 'dtype', 'later_reorder'])
def test_partition_selection_rejects_changed_full_order_or_interval(fault):
    spec = contract.unit_spec('own0U_later_full' if fault == 'later_reorder' else 'own0U_prefix_part1')
    start = 0 if spec['cohort'] == 'prefix' else contract.FOLD['oof_start_row']
    full = np.arange(start, start + spec['partition']['full_rows'], dtype=np.int64)
    if fault == 'shift': spec['partition']['presentation_start'] += 1
    elif fault == 'boolean_start': spec['partition']['presentation_start'] = False
    elif fault == 'missing': full = full[:-1]
    elif fault == 'duplicate': full[1] = full[0]
    elif fault == 'dtype': full = full.astype(np.int32)
    else: full = full[::-1].copy()
    with pytest.raises(RuntimeError):
        worker.select_partition_parents(full, spec, np)


@pytest.mark.parametrize('fault', ['missing', 'shift', 'full_rows', 'digest', 'boolean_start'])
def test_partition_result_requires_exact_interval_identity(monkeypatch, fault):
    unit = 'terminal_prefix_part1'; spec = contract.unit_spec(unit)
    recipe = ref('SYNTHETIC_RECIPE.json'); counts = contract.call_limits(unit)
    pointer = contract.build_frozen_extraction_pointer(recipe_ref=recipe, result_ref=ref('SYNTHETIC_RESULT.json'),
        outcome='COMPLETE', calls=counts, returns=counts, unit=unit)
    result = {'recipe': recipe, 'unit': unit, 'optimizer_steps': 0, 'test_data_used': False,
        'observed_calls': counts, 'observed_returns': counts, 'decision': 'COMPLETE_FROZEN_READOUT_EXTRACTION',
        'representation_rows': spec['rows'], 'hidden_width': 128, 'originals_stat_preserved': True,
        'source_unchanged': True, 'native_optimizer_steps': 0, 'representations': ref('SYNTHETIC_ARRAYS.npz'),
        'partition': {**spec['partition'], 'full_presentation_parent_rows_sha256': 'a' * 64}}
    monkeypatch.setattr(campaign, 'require_binding', lambda r, **kw: contract.plain_ref(r))
    monkeypatch.setattr(campaign, 'read_bound_json', lambda *a: deepcopy(result))
    contract.require_frozen_extraction_pointer(pointer, expected_recipe=recipe, expected_unit=unit)
    if fault == 'missing': del result['partition']
    elif fault == 'shift': result['partition']['presentation_start'] += 16
    elif fault == 'full_rows': result['partition']['full_rows'] -= 16
    elif fault == 'digest': result['partition']['full_presentation_parent_rows_sha256'] = 'z' * 64
    else: result['partition']['presentation_start'] = False
    with pytest.raises(RuntimeError, match='RESULT_PARTITION_IDENTITY'):
        contract.require_frozen_extraction_pointer(pointer, expected_recipe=recipe, expected_unit=unit)


# Only metadata objects: all bound reads below are synthetic and in-memory.
def _independent_case(monkeypatch, unit=contract.TIMING_UNIT):
    recipe = metadata_transition_plan(unit)
    stage, name = contract.INDEPENDENT_UNITS[unit]
    recipe.update(admission_stage=stage, run_root=contract.INDEPENDENT_ROOT + '/' + name,
        runtime_root=contract.INDEPENDENT_ROOT + '/' + name + '_RUNTIME')
    recipe['execution_binding_path'] = recipe['runtime_root'] + '/EXECUTION_BINDING.json'
    recipe['independent_completion'] = {'design': deepcopy(contract.INDEPENDENT_DESIGN), 'initial_parity': None}
    design = {'schema_version': 'gx1_independent_default_sign_0u_completion_design_v1',
        'canonical_repo': contract.REPO, 'canonical_branch': 'work/gx1-current',
        'authority': {'execution_authorized': False}, 'source_closure': {
            'baseline_current57_references': deepcopy(recipe['source_owners']),
            'only_runtime_code_edit_exceptions': {n: {} for n in contract.INDEPENDENT_CODE_FILES},
            'qualification_test_edit_allowlist': {contract.NEW_FILES[-1]: {}}},
        'original_bindings': {'own0U_state': {'ref': ref('SYNTHETIC_0U.pt')},
            'fresh_identity_sha256': 'a' * 64, 'feature_view_identity': {'synthetic_view': True}}}
    docs = {contract.INDEPENDENT_DESIGN['path']: (deepcopy(contract.INDEPENDENT_DESIGN), design)}
    seen = []
    def read(binding):
        expected, value = docs[binding['path']]
        contract.require(binding == expected, 'SYNTHETIC_BOUND_REF')
        seen.append(deepcopy(binding))
        return deepcopy(value)
    monkeypatch.setattr(campaign, 'read_bound_json', lambda path, digest: read({'path': str(path), 'sha256': digest}))
    return recipe, design, docs, read, seen


def _independent_part3_case(monkeypatch):
    recipe, design, docs, read, seen = _independent_case(monkeypatch, 'own0U_prefix_part3')
    timing = deepcopy(recipe)
    stage, name = contract.INDEPENDENT_UNITS[contract.TIMING_UNIT]
    timing.update(unit=contract.TIMING_UNIT, admission_stage=stage, units_design=contract.unit_design(contract.TIMING_UNIT),
        run_root=contract.INDEPENDENT_ROOT + '/' + name, runtime_root=contract.INDEPENDENT_ROOT + '/' + name + '_RUNTIME',
        budget=contract.resource_limits(contract.TIMING_UNIT), operation_limits=contract.call_limits(contract.TIMING_UNIT))
    timing['execution_binding_path'] = timing['runtime_root'] + '/EXECUTION_BINDING.json'
    timing_ref = {'path': contract.INDEPENDENT_ROOT + '/TIMING_RECIPE.json', 'sha256': '1' * 64}
    def bind(path, value):
        binding = {'path': path, 'sha256': '1' * 64};docs[path] = (binding, value);return binding
    parity = {'reference': deepcopy(contract.INITIAL_PREDICTIONS), 'rows': 256, 'bit_identical': True}
    report = {'unit': contract.TIMING_UNIT, 'cutoff_ns': contract.FOLD['train_cutoff_ns'],
        'original_prediction_parity': parity, 'own_prefix_prior': {'synthetic': True},
        'state_preservation': {'model': 'a' * 64, 'all_buffers': 'b' * 64, 'rng': 'c' * 64, 'modes': {}}}
    report_ref = bind(timing['run_root'] + '/NATIVE_REPORT.json', report)
    limits = contract.call_limits(contract.TIMING_UNIT)
    execution_ref = bind(contract.INDEPENDENT_ROOT + '/TIMING_EXECUTION.json', {
        'prelaunch_manifest': timing_ref, 'invocation_kind': contract.KIND, 'source_commit': 'a' * 40})
    invocation = contract._seal({'kind': contract.KIND, 'invocation_number': 1, 'optimizer_step_budget': 0,
        'test_data_used': False, 'source_commit': 'a' * 40, 'execution_manifest': execution_ref}, 'invocation_sha256')
    invocation_ref = bind(contract.INDEPENDENT_ROOT + '/TIMING_INVOCATION.json', invocation)
    bound_campaign = contract._seal({'recipe': timing_ref, 'runtime_root': timing['runtime_root'],
        'phase': contract.PHASE, 'test_data_used': False, 'source_commit': 'a' * 40,
        'invocations': [invocation_ref]}, 'plan_sha256')
    campaign_ref = bind(contract.INDEPENDENT_ROOT + '/TIMING_CAMPAIGN.json', bound_campaign)
    result = {'decision': 'COMPLETE_FROZEN_READOUT_EXTRACTION', 'error': None, 'recipe': timing_ref,
        'unit': contract.TIMING_UNIT, 'diagnostic_only': True,
        'state': design['original_bindings']['own0U_state']['ref'], 'state_role': 'own0U', 'cohort': 'prefix',
        'representation_rows': 256, 'hidden_width': 128, 'native_report': report_ref,
        'fresh_identity_sha256': design['original_bindings']['fresh_identity_sha256'],
        'feature_view_identity': design['original_bindings']['feature_view_identity'],
        'original_prediction_parity': parity, 'observed_calls': deepcopy(limits), 'observed_returns': deepcopy(limits),
        'calls': {'calls': deepcopy(limits), 'returns': deepcopy(limits), 'optimizer_steps': 0, 'reverse_calls': 0},
        'source_unchanged': True, 'originals_stat_preserved': True, 'optimizer_steps': 0,
        'native_optimizer_steps': 0, 'test_data_used': False, 'denied_phase_opens': [], 'delegated_denied_opens': [],
        'own_prefix_prior': {'synthetic': True}, 'campaign_plan': campaign_ref}
    result_ref = bind(timing['run_root'] + '/RESULT.json', result)
    pointer = contract.build_frozen_extraction_pointer(recipe_ref=timing_ref, result_ref=result_ref,
        outcome='COMPLETE', calls=limits, returns=limits, unit=contract.TIMING_UNIT)
    pointer_ref = bind(timing['run_root'] + '/DIAGNOSTIC_POINTER.json', pointer)
    receipt = contract._seal({'outcome': 'COMPLETE', 'kind': contract.KIND, 'guard_decision': 'PASS',
        'trainer_guard_exit_code': 0, 'progress_observer_exit_code': 0, 'invocation_number': 1,
        'test_data_used': False, 'checkpoint_pointer_after': pointer_ref,
        'plan_sha256': bound_campaign['plan_sha256'], 'invocation_sha256': invocation['invocation_sha256']}, 'receipt_sha256')
    receipt_ref = bind(timing['runtime_root'] + '/receipts/invocation-0001.json', receipt)
    review = {'schema_version': contract.INDEPENDENT_REVIEW_SCHEMA,
        'status': 'COMPLETE_AND_QUALIFIED_CURRENT_SIGN_ONLY_OWN0U256', 'design': deepcopy(contract.INDEPENDENT_DESIGN),
        'test_data_used': False, 'state_preserved': True, 'all_original_fields_bit_identical': True,
        'recipe': timing_ref, 'result': result_ref, 'native_report': report_ref, 'pointer': pointer_ref,
        'canonical_terminal_receipt': receipt_ref}
    review_ref = bind(timing['run_root'] + '/ACTUAL_ROOT_REVIEW.json', review)
    docs[timing_ref['path']] = (timing_ref, timing)
    recipe['independent_completion']['initial_parity'] = review_ref
    return recipe, design, docs, read, seen, timing, review, result, report, receipt, pointer


@pytest.mark.parametrize('unit', [contract.TIMING_UNIT, 'own0U_prefix_part3'])
def test_independent_exact_fixed_units_keep_original_caps_and_closure(monkeypatch, unit):
    case = _independent_case(monkeypatch) if unit == contract.TIMING_UNIT else _independent_part3_case(monkeypatch)
    recipe, design, docs, read, seen = case[:5]
    assert contract.require_plan(recipe) == recipe
    assert contract.require_source_closure(recipe, *source_inputs()) == recipe['source_owners']
    assert contract.require_independent_completion(recipe, read) == design
    assert contract.call_limits(unit)['reverse'] == contract.call_limits(unit)['autograd_grad'] == 0
    assert contract.unit_spec(unit)['rows'] == (256 if unit == contract.TIMING_UNIT else 15858)
    assert {r['path'] for r in seen} == set(docs)


@pytest.mark.parametrize('fault', ['unit', 'stage', 'run_root', 'runtime_root', 'binding', 'design',
    'extra_field', 'presumed_parity', 'gc_proof', 'budget', 'source_drop', 'source_extra', 'source_drift'])
def test_independent_scope_source_and_authority_expansion_rejected(monkeypatch, fault):
    recipe, design, docs, read, seen = _independent_case(monkeypatch)
    if fault == 'unit': recipe['unit'] = 'own0U_later_full'
    elif fault == 'stage': recipe['admission_stage'] = contract.STAGE
    elif fault == 'run_root': recipe['run_root'] += '_OTHER'
    elif fault == 'runtime_root': recipe['runtime_root'] += '_OTHER'
    elif fault == 'binding': recipe['execution_binding_path'] += '_OTHER'
    elif fault == 'design': recipe['independent_completion']['design']['sha256'] = '9' * 64
    elif fault == 'extra_field': recipe['independent_completion']['extra'] = True
    elif fault == 'presumed_parity': recipe['independent_completion']['initial_parity'] = ref('OLD_REVIEW.json')
    elif fault == 'gc_proof': recipe['gc_layout_metadata_parity']['sha256'] = '9' * 64
    elif fault == 'budget': recipe['budget']['max_wall_seconds'] += 1
    elif fault == 'source_drop': del recipe['source_owners']['gx1/synthetic_owner_0.py']
    elif fault == 'source_extra': recipe['source_owners']['gx1/extra.py'] = {'path': contract.REPO + '/gx1/extra.py', 'sha256': '9' * 64}
    else: recipe['source_owners']['gx1/synthetic_owner_0.py']['sha256'] = '9' * 64
    with pytest.raises(RuntimeError): contract.require_plan(recipe)


@pytest.mark.parametrize('fault', ['foreign_review', 'status', 'state_preserved', 'fields', 'same_sources',
    'result_failed', 'original_parity', 'state', 'fresh_identity', 'feature_view', 'forward_count',
    'reverse', 'optimizer', 'forbidden_open', 'report', 'pointer', 'receipt', 'campaign', 'invocation', 'execution'])
def test_independent_part3_requires_fresh_own_complete_timing_proof(monkeypatch, fault):
    recipe, design, docs, read, seen, timing, review, result, report, receipt, pointer = _independent_part3_case(monkeypatch)
    if fault == 'foreign_review': recipe['independent_completion']['initial_parity'] = deepcopy(contract.PARTITION_PARITY)
    elif fault == 'status': review['status'] = 'FAILED'
    elif fault == 'state_preserved': review['state_preserved'] = False
    elif fault == 'fields': review['all_original_fields_bit_identical'] = False
    elif fault == 'same_sources': timing['source_owners'][contract.NEW_FILES[0]]['sha256'] = '9' * 64
    elif fault == 'result_failed': result['decision'] = 'FAILED'
    elif fault == 'original_parity': result['original_prediction_parity'] = {**result['original_prediction_parity'], 'rows': 255}
    elif fault == 'state': result['state'] = ref('OTHER_0U.pt')
    elif fault == 'fresh_identity': result['fresh_identity_sha256'] = '9' * 64
    elif fault == 'feature_view': result['feature_view_identity'] = {}
    elif fault == 'forward_count': result['observed_returns']['panel_forward'] -= 1
    elif fault == 'reverse': result['observed_calls']['reverse'] = 1
    elif fault == 'optimizer': result['optimizer_steps'] = 1
    elif fault == 'forbidden_open': result['denied_phase_opens'] = ['synthetic-forbidden']
    elif fault == 'report': report['cutoff_ns'] += 1
    elif fault == 'pointer': pointer['outcome'] = 'FAILED'
    elif fault == 'receipt': receipt['trainer_guard_exit_code'] = 1
    elif fault == 'campaign': docs[result['campaign_plan']['path']][1]['recipe'] = ref('OLD_RECIPE.json')
    elif fault == 'invocation': docs[contract.INDEPENDENT_ROOT + '/TIMING_INVOCATION.json'][1]['optimizer_step_budget'] = 1
    else: docs[contract.INDEPENDENT_ROOT + '/TIMING_EXECUTION.json'][1]['prelaunch_manifest'] = ref('OTHER.json')
    with pytest.raises(RuntimeError): contract.require_plan(recipe)


def test_independent_optin_absent_preserves_legacy_namespace_and_reads_nothing(monkeypatch):
    def denied(*args): raise AssertionError('legacy validation must not read independent metadata')
    monkeypatch.setattr(campaign, 'read_bound_json', denied)
    recipe = metadata_transition_plan()
    assert contract.require_plan(recipe) == recipe and contract.require_independent_completion(recipe) is None
    recipe['run_root'] = contract.INDEPENDENT_ROOT + '/NATIVE_0U_TIMING256_001'
    with pytest.raises(RuntimeError, match='SEPARATE_UNIT_ROOTS'): contract.require_plan(recipe)


@pytest.mark.parametrize('fault', ['none', 'no_optin', 'wrong_component_old', 'wrong_component_current',
    'other_source', 'manifest', 'numerical_recipe', 'fold', 'design'])
def test_independent_prefix_metadata_preserves_original_manifest_and_exact_component_transition(monkeypatch, fault):
    prefix, launch, common, manifest, hashes = _prefix_metadata_case()
    component = {'path': contract.REPO + '/gx1/contracts/entry_native_direction_research_v1.py', 'sha256': '6' * 64}
    current = {**component, 'sha256': '7' * 64}
    manifest['view_source_bindings']['direction_component_source'] = deepcopy(component)
    hashes[component['path']] = deepcopy(current)
    launch['numerical_recipe'] = {'synthetic_sign_recipe': True}
    original = {k: deepcopy(manifest[k]) for k in ('prefix_normalization', 'constructor_metadata', 'targets',
        'archive_members', 'view_source_bindings', 'physical_provider_plan', 'physical_provider_worker')}
    original.update(input_manifest=deepcopy(launch['input_manifest']), fold=deepcopy(launch['fold']),
        original_numerical_recipe=deepcopy(launch['numerical_recipe']))
    design = {'schema_version': 'gx1_independent_default_sign_0u_completion_design_v1', 'canonical_repo': contract.REPO,
        'canonical_branch': 'work/gx1-current', 'authority': {'execution_authorized': False},
        'original_bindings': original, 'historical_source_verification': {
            'only_new_manifest_source_transition': {'original': component, 'current': current}}}
    read = common._read; seen = []
    def bound_read(binding, metadata):
        if binding['path'] == contract.INDEPENDENT_DESIGN['path']:
            assert binding == contract.INDEPENDENT_DESIGN
            metadata[binding['path']] = {'ref': deepcopy(binding)};seen.append(binding);return deepcopy(design)
        return read(binding, metadata)
    common._read = bound_read
    completion = {'design': deepcopy(contract.INDEPENDENT_DESIGN), 'initial_parity': None}
    if fault == 'wrong_component_old': design['historical_source_verification']['only_new_manifest_source_transition']['original'] = {**component, 'sha256': '9' * 64}
    elif fault == 'wrong_component_current': hashes[component['path']]['sha256'] = '9' * 64
    elif fault == 'other_source': hashes[manifest['view_source_bindings']['other']['path']]['sha256'] = '9' * 64
    elif fault == 'manifest': original['input_manifest'] = ref('OTHER_MANIFEST.json')
    elif fault == 'numerical_recipe': original['original_numerical_recipe'] = {}
    elif fault == 'fold': original['fold']['prefix_rows'] += 1
    elif fault == 'design': completion['design']['sha256'] = '9' * 64
    before = deepcopy(manifest)
    arguments = {'gc_layout_metadata_parity': deepcopy(contract.GC_LAYOUT_METADATA_PARITY)}
    if fault != 'no_optin': arguments['independent_completion'] = completion
    if fault == 'none':
        context = prefix._metadata(launch, common, **arguments)
        assert context['manifest'] == before == manifest and seen == [contract.INDEPENDENT_DESIGN]
        assert contract.INDEPENDENT_DESIGN['path'] in context['info']['metadata']
        assert context['manifest']['view_source_bindings']['direction_component_source'] != current
    else:
        with pytest.raises(RuntimeError): prefix._metadata(launch, common, **arguments)
        assert manifest == before
