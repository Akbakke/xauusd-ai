"""Synthetic fail-closed containment tests; no CUDA or model construction."""
import ast
import hashlib
import os
from pathlib import Path
import shlex
import subprocess

import pytest
from gx1.contracts import gx1_capped_execution_v1 as g

SHA = 'a' * 64
MODULE = 'gx1.audit.diagnose_native_future_state_bias_suffix_v1'
BASENAME = 'NATIVE_FUTURE_STATE_BIAS_SUFFIX_ORDER_20261122_003'
ENV = {
    'GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX': 'true',
    'GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256': SHA,
    'GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN': 'false',
    'GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256': '',
    'GX1_NATIVE_FUTURE_STATE_THROUGHPUT': 'false',
    'GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256': '',
    'GX1_NATIVE_FUTURE_STATE_TRAINING': 'false',
    'GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256': '',
    'GX1_NATIVE_FUTURE_STATE_TRAINING_LANE': '',
    'GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION': 'false',
    'GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS': 'false',
    'GX1_TRAINER_MAX_WALL_SECONDS': '3600',
    'GX1_TRAINER_MODEL_MAX_WALL_SECONDS': '3570',
    'GX1_TRAINER_ATTENDED_STAGE_REQUIRED': 'false',
    'GX1_TRAINER_GPU_INDEX': '0',
    'GX1_TRAINER_GPU_MAX_POWER_LIMIT_W': '250',
    'GX1_TRAINER_GPU_MAX_POWER_DRAW_W': '250',
    'GX1_TRAINER_GPU_MAX_CORE_TEMP_C': '75',
    'GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C': '80',
    'GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB': '12288',
    'GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS': '1',
    'GX1_TRAINER_CPU_MAX_TEMP_C': '80',
    'GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W': '125',
    'GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES': '8589934592',
    'GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES': '8589934592',
    'GX1_POWER_BENCHMARK_SCOPE_JSON': '',
    'GX1_POWER_BENCHMARK_SCOPE_SHA256': '',
}


def setup_env(monkeypatch):
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)


def test_only_exact_suffix_containment_and_no_training_or_optimizer_authority(monkeypatch):
    setup_env(monkeypatch)
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: {'canonical_scope': True})
    value = g.require_native_future_state_bias_suffix_execution(plan_sha256=SHA)
    assert value['canonical_scope'] and value['class'] == 'trainer'
    assert not value['training_authority_granted'] and not value['optimizer_authority_granted']
    assert value['outer_seconds'] == 3600 and value['internal_seconds'] == 3570
    assert value['cpu_affinity'] == list(range(19)) and value['numerical_threads'] == 8
    assert value['output_root'] == '/home/andre2/GX1_RUNS/' + BASENAME
    assert value['plan']['path'].endswith('/configs/research/' + BASENAME + '.json')


@pytest.mark.parametrize('name', sorted(ENV))
def test_every_mode_hash_deadline_hardware_or_scope_field_fails_closed_before_guard(monkeypatch, name):
    setup_env(monkeypatch)
    monkeypatch.setenv(name, ENV[name] + 'wrong')
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: pytest.fail('invalid env reached guard'))
    with pytest.raises(RuntimeError, match='SUFFIX_BIAS_ENV_INVALID'):
        g.require_native_future_state_bias_suffix_execution(plan_sha256=SHA)


@pytest.mark.parametrize('bad', ['', 'A' * 64, 'x' * 64, 'a' * 63, None, 1])
def test_bad_plan_digest_fails_before_any_environment_or_runtime(monkeypatch, bad):
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: pytest.fail('bad SHA reached runtime'))
    with pytest.raises(RuntimeError, match='SUFFIX_BIAS_PLAN_SHA256_INVALID'):
        g.require_native_future_state_bias_suffix_execution(plan_sha256=bad)


def test_actual_scope_failure_is_propagated(monkeypatch):
    setup_env(monkeypatch)
    def failed_scope():
        raise RuntimeError('actual cgroup or lock failed')
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', failed_scope)
    with pytest.raises(RuntimeError, match='actual cgroup or lock failed'):
        g.require_native_future_state_bias_suffix_execution(plan_sha256=SHA)


def test_all_old_guard_python_and_shell_authorities_preserved_by_existing_bridge():
    owners = {f'gx1/synthetic_owner_{i}.py': 'b' * 64 for i in range(56)}
    owners.update(g._NATIVE_FUTURE_STATE_GUARD_PARENT_HASHES)
    proof = g.native_future_state_guard_source_bridge(owners, producer_commit=g._NATIVE_FUTURE_STATE_GUARD_PARENT_COMMIT)
    assert proof['checked_guard_owners'] == 2
    assert proof['other_parent_owners_to_be_checked_by_caller'] == 56
    assert proof['owners']['scripts/gx1_capped_run.sh']['normalized_sha256'] == g._NATIVE_FUTURE_STATE_GUARD_PARENT_HASHES['scripts/gx1_capped_run.sh']


def shell_harness(tmp_path, *, mutation=None, preflight=False):
    repo = tmp_path / 'canonical'; repo.mkdir()
    config = repo / 'configs/research'; config.mkdir(parents=True)
    plan = config / (BASENAME + '.json'); plan.write_text('{"synthetic":true}\n')
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    argv = [str(repo / '.venv/bin/python'), '-m', MODULE, '--plan-sha256', digest]
    if preflight:
        argv.append('--preflight')
    if mutation is not None:
        argv = mutation(argv, plan)
    state = g._NATIVE_FUTURE_STATE_BIAS_SUFFIX_GUARD_SHELL_ADDITIONS[0][2]
    functions = g._NATIVE_FUTURE_STATE_BIAS_SUFFIX_GUARD_SHELL_ADDITIONS[1][2]
    script = state + functions + '\nREPO_ROOT=' + shlex.quote(str(repo))
    script += '\nCANONICAL_TRAINER_PYTHON=' + shlex.quote(str(repo / '.venv/bin/python'))
    script += '\nmatches_bound_native_future_state_bias_suffix ' + ' '.join(map(shlex.quote, argv)) + '\n'
    return subprocess.run(['bash', '-c', script], cwd=repo, capture_output=True, text=True)


@pytest.mark.parametrize('preflight', [False, True])
def test_shell_exact_command_and_optional_preflight_match(tmp_path, preflight):
    assert shell_harness(tmp_path, preflight=preflight).returncode == 0


@pytest.mark.parametrize('case', ['extra', 'wrong_interpreter', 'wrong_module', 'wrong_sha', 'wrong_flag', 'missing_plan', 'symlink_plan'])
def test_shell_rejects_cli_and_plan_substitutions(tmp_path, case):
    def mutation(argv, plan):
        if case == 'extra':
            argv += ['--unsafe']
        elif case == 'wrong_interpreter':
            argv[0] = '/usr/bin/python3'
        elif case == 'wrong_module':
            argv[2] = MODULE + '.other'
        elif case == 'wrong_sha':
            argv[4] = 'f' * 64
        elif case == 'wrong_flag':
            argv[3] = '--plan'
        elif case == 'missing_plan':
            plan.unlink()
        else:
            other = plan.with_suffix('.original'); plan.rename(other); plan.symlink_to(other)
        return argv
    result = shell_harness(tmp_path, mutation=mutation)
    assert result.returncode != 0 and 'command not found' not in result.stderr


def test_suffix_shell_insertions_reverse_to_qualified_frozen_shell():
    repo = Path(g.__file__).resolve().parents[2]
    text = (repo / 'scripts/gx1_capped_run.sh').read_text()
    for anchor, position, fragment in reversed(g._NATIVE_FUTURE_STATE_BIAS_SUFFIX_GUARD_SHELL_ADDITIONS):
        composite = fragment + anchor if position == 'before' else anchor + fragment
        assert text.count(fragment) == text.count(composite) == 1
        text = text.replace(composite, anchor, 1)
    original = subprocess.check_output(['git', 'show', '66870a838:scripts/gx1_capped_run.sh'], cwd=repo, text=True)
    assert text == original


def main_guard_owners():
    repo = Path(g.__file__).resolve().parents[2]
    owners = {f'gx1/synthetic_main_owner_{i}.py': 'b' * 64 for i in range(60)}
    for name in ('gx1/contracts/gx1_capped_execution_v1.py', 'scripts/gx1_capped_run.sh'):
        data = subprocess.check_output(['git', 'show', 'f96d89f88:' + name], cwd=repo)
        owners[name] = hashlib.sha256(data).hexdigest()
    return owners


def test_direct_MAIN_bridge_preserves_all_old_guard_logic_and_records_current_hashes():
    owners = main_guard_owners()
    proof = g.native_future_state_bias_guard_source_bridge(owners,
        producer_commit='f96d89f88ea742a9a90460d81cc386f8abea6e2a')
    assert proof['checked_guard_owners'] == 2 and proof['other_main_owners_to_be_checked_by_caller'] == 60
    delta = proof['qualified_resource_delta']
    assert delta['to_commit'] == '6749062726c8562db35194857f56ee8aa4859d23'
    assert delta['qualified_shell_sha256'] == '8a88a7da0daaff455e1bf25670c121163ccd123b6ac82c743e910d3fc864b3a1'
    assert delta['diff_sha256'] == '45627527b5019aa87d92acb18c2fff1177fdd5ec457daf835b5715c30a879804'
    repo = Path(g.__file__).resolve().parents[2]
    for name, cell in proof['owners'].items():
        assert cell['original_sha256'] == owners[name]
        assert cell['current_sha256'] == hashlib.sha256((repo / name).read_bytes()).hexdigest()


def test_direct_MAIN_bridge_rejects_wrong_producer_and_guard_hash():
    owners = main_guard_owners()
    with pytest.raises(RuntimeError, match='PRODUCER_COMMIT'):
        g.native_future_state_bias_guard_source_bridge(owners, producer_commit='a' * 40)
    owners['scripts/gx1_capped_run.sh'] = 'f' * 64
    with pytest.raises(RuntimeError, match='AUTHENTIC_MAIN_GUARD_HASH'):
        g.native_future_state_bias_guard_source_bridge(owners,
            producer_commit='f96d89f88ea742a9a90460d81cc386f8abea6e2a')


def test_frozen_and_older_resource_functions_are_AST_unchanged():
    repo = Path(g.__file__).resolve().parents[2]
    original = ast.parse(subprocess.check_output(['git', 'show',
        '66870a838:gx1/contracts/gx1_capped_execution_v1.py'], cwd=repo, text=True))
    current = ast.parse(Path(g.__file__).read_text())
    old_functions = {n.name: n for n in original.body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name: n for n in current.body if isinstance(n, ast.FunctionDef)}
    for name in ('require_native_future_state_bias_frozen_execution',
            '_native_future_state_leaf_execution', 'require_native_future_state_throughput_execution',
            'require_native_future_state_training_execution', 'require_guarded_cuda_trainer_execution'):
        assert ast.dump(old_functions[name], include_attributes=False) == ast.dump(new_functions[name], include_attributes=False)


@pytest.mark.parametrize('mutation', ['historical001', 'qualifiedresource', 'resourcediff'])
def test_bridge_rejects_substituted_historical_or_qualified_resource_source(monkeypatch, mutation):
    original = subprocess.check_output
    def changed(command, *args, **kwargs):
        value = original(command, *args, **kwargs)
        if (mutation == 'historical001' and command[:2] == ['git', 'show']
                and command[2].startswith('63a1ff0ac888') and command[2].endswith(':scripts/gx1_capped_run.sh')):
            return value + b'\n# unreviewed historical guard\n'
        if (mutation == 'qualifiedresource' and command[:2] == ['git', 'show']
                and command[2].startswith('6749062726c8') and command[2].endswith(':scripts/gx1_capped_run.sh')):
            return value.replace(b'TRAINER_CPU_MAX_TEMP_C=80', b'TRAINER_CPU_MAX_TEMP_C=99')
        if mutation == 'resourcediff' and command[:2] == ['git', 'diff']:
            return value + b'\n# unreviewed resource delta\n'
        return value
    monkeypatch.setattr(subprocess, 'check_output', changed)
    reasons = {'historical001': 'AUTHENTIC_001_GUARDS', 'qualifiedresource': 'QUALIFIED_RESOURCE_SHELL', 'resourcediff': 'EXACT_RESOURCE_DIFF'}
    with pytest.raises(RuntimeError, match=reasons[mutation]):
        g.native_future_state_bias_guard_source_bridge(main_guard_owners(),
            producer_commit='f96d89f88ea742a9a90460d81cc386f8abea6e2a')


def test_bridge_rejects_current_shell_change_outside_named002_route(monkeypatch):
    actual = Path.read_text
    target = Path(g.__file__).resolve().parents[2] / 'scripts/gx1_capped_run.sh'
    def changed(path, *args, **kwargs):
        text = actual(path, *args, **kwargs)
        return text.replace('TRAINER_CPU_MAX_TEMP_C=80', 'TRAINER_CPU_MAX_TEMP_C=99') if path == target else text
    monkeypatch.setattr(Path, 'read_text', changed)
    with pytest.raises(RuntimeError, match='ONLY_002_ROUTE_CHANGED'):
        g.native_future_state_bias_guard_source_bridge(main_guard_owners(),
            producer_commit='f96d89f88ea742a9a90460d81cc386f8abea6e2a')


def test_bridge_rejects_old_python_owner_change(monkeypatch):
    actual = Path.read_bytes
    target = Path(g.__file__).resolve()
    tree = ast.parse(actual(target))
    owner = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'require_guarded_cuda_trainer_execution')
    owner.body.append(ast.Expr(value=ast.Constant(value='unreviewed old guard owner')))
    changed_bytes = ast.unparse(tree).encode()
    def changed(path, *args, **kwargs):
        return changed_bytes if path == target else actual(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_bytes', changed)
    with pytest.raises(RuntimeError, match='OLD_001_PYTHON_MODULE_CHANGED'):
        g.native_future_state_bias_guard_source_bridge(main_guard_owners(),
            producer_commit='f96d89f88ea742a9a90460d81cc386f8abea6e2a')


def test_shell_rejects_consumed001_digest(tmp_path):
    consumed = 'd2c2f9e5b2b8b8519bbe27d8e2280c6dfa97f0c3fb3f7b6f21f2deee329f8cf5'
    assert shell_harness(tmp_path, mutation=lambda argv, plan: argv[:4] + [consumed]).returncode != 0


@pytest.mark.parametrize('mutation', ['duplicate_constant', 'changed_literal', 'missing_suffix_guard'])
def test_bridge_requires_exact_current_guard_owners_and_loaded_route_literal(monkeypatch, mutation):
    actual = Path.read_bytes
    target = Path(g.__file__).resolve()
    tree = ast.parse(actual(target))
    constant = next(n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name) and n.targets[0].id == '_NATIVE_FUTURE_STATE_BIAS_SUFFIX_GUARD_SHELL_ADDITIONS')
    if mutation == 'duplicate_constant':
        tree.body.append(constant)
    elif mutation == 'changed_literal':
        constant.value.elts[0].elts[0] = ast.Constant(value='unreviewed route anchor')
    else:
        tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef)
            and n.name == 'require_native_future_state_bias_suffix_execution')]
    changed_bytes = ast.unparse(tree).encode()
    def changed(path, *args, **kwargs):
        return changed_bytes if path == target else actual(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_bytes', changed)
    reasons = {'duplicate_constant': 'EXACT_002_ROUTE_CONSTANT', 'changed_literal': 'CURRENT_002_ROUTE_LITERAL', 'missing_suffix_guard': 'EXACT_002_GUARD_FUNCTIONS'}
    with pytest.raises(RuntimeError, match=reasons[mutation]):
        g.native_future_state_bias_guard_source_bridge(main_guard_owners(),
            producer_commit='f96d89f88ea742a9a90460d81cc386f8abea6e2a')



def _driver_runtimes():
    from copy import deepcopy
    current = {'schema_version': 'gx1_native_future_state_cuda_runtime_proof_v1',
        'status': 'VERIFIED_IDLE_RUNTIME_CURRENT_BOOT', 'observed_utc': '2026-11-22T20:00:00+00:00',
        'source_commit': 'a' * 40, 'wsl_boot_id': 'synthetic-boot', 'physical_windows_boot_id': 4,
        'library_versions': {'python': '3.10.12', 'numpy': '2.2.6', 'torch': '2.6.0+cu124', 'torch_cuda': '12.4', 'cudnn': 90100},
        'gpu': {'index': 0, 'uuid': 'GPU-8c6ac5f1-4254-6cec-9780-44b019cafd29', 'name': 'NVIDIA GeForce RTX 3090',
                'driver_version': '610.88', 'total_memory_bytes': 25769803776},
        'idle': {'gpu_compute_processes': [], 'current_project_model_processes': []},
        'cuda_initialized': False, 'no_model_or_checkpoint_opened': True}
    original = deepcopy({k: current[k] for k in ('library_versions', 'gpu')})
    original['gpu']['driver_version'] = '591.86'
    return original, current


def test_driver_metadata_declares_only_unproven_fixed_transition():
    original, current = _driver_runtimes()
    value = g.native_future_state_driver_requalification_metadata(original, current)
    assert value == {'schema_version': 'gx1_native_driver_requalification_transition_v1',
        'original_driver': '591.86', 'current_driver': '610.88',
        'other_runtime_fields_identical': True, 'equivalence_proven': False}


@pytest.mark.parametrize('mutation', ['same_driver', 'another_driver', 'old_driver', 'gpu_name', 'gpu_memory', 'gpu_index', 'gpu_uuid', 'torch', 'cudnn', 'original_index_bool', 'original_cudnn_float'])
def test_driver_metadata_rejects_every_other_runtime_delta(mutation):
    original, current = _driver_runtimes()
    if mutation == 'same_driver': current['gpu']['driver_version'] = '591.86'
    elif mutation == 'another_driver': current['gpu']['driver_version'] = '611.00'
    elif mutation == 'old_driver': original['gpu']['driver_version'] = '590.00'
    elif mutation == 'gpu_name': current['gpu']['name'] += ' different'
    elif mutation == 'gpu_memory': current['gpu']['total_memory_bytes'] += 1
    elif mutation == 'gpu_index': current['gpu']['index'] = 1
    elif mutation == 'gpu_uuid': current['gpu']['uuid'] = 'GPU-' + '0' * 36
    elif mutation == 'torch': current['library_versions']['torch'] += '.other'
    elif mutation == 'original_index_bool': original['gpu']['index'] = False
    elif mutation == 'original_cudnn_float': original['library_versions']['cudnn'] = float(original['library_versions']['cudnn'])
    else: current['library_versions']['cudnn'] += 1
    with pytest.raises((RuntimeError, ValueError)):
        g.native_future_state_driver_requalification_metadata(original, current)


def test_original_runtime_compatibility_remains_strict_without_explicit_authority(monkeypatch):
    from copy import deepcopy
    from gx1.audit import native_future_state_training_v1 as train
    original, current = _driver_runtimes()
    monkeypatch.setattr(g, 'require_native_future_state_bias_suffix_execution', lambda **k: pytest.fail('default reached special guard'))
    matching = deepcopy(current); matching['gpu'] = deepcopy(original['gpu'])
    assert train.runtime_compatible(original, matching) is True
    assert train.runtime_compatible(original, matching, driver_requalification=None) is True
    with pytest.raises(ValueError, match='RECOVERY_RUNTIME_COMPATIBILITY'):
        train.runtime_compatible(original, current)


@pytest.mark.parametrize('authority', [True, False, {}, {'plan_sha256': 'A' * 64}, {'plan_sha256': 'a' * 63}, {'plan_sha256': SHA, 'ignore_driver': True}])
def test_driver_loader_rejects_unbound_authority_before_special_guard(monkeypatch, authority):
    from gx1.audit import native_future_state_training_v1 as train
    original, current = _driver_runtimes()
    monkeypatch.setattr(g, 'require_native_future_state_bias_suffix_execution', lambda **k: pytest.fail('invalid authority reached guard'))
    with pytest.raises(ValueError, match='EXACT_DRIVER_REQUALIFICATION_AUTHORITY'):
        train.runtime_compatible(original, current, driver_requalification=authority)


def _driver_authority_fixture(monkeypatch):
    from copy import deepcopy
    from gx1.audit import native_future_state_training_v1 as train
    from gx1.audit import diagnose_native_future_state_bias_suffix_v1 as suffix
    original, current = _driver_runtimes()
    stage = 'native_future_state_bias_suffix_order_20261122_003'
    binding = {'path': str(train.REPO / 'configs/research' / (BASENAME + '.json')), 'sha256': SHA}
    main = {'path': str(train.MAIN_PLAN), 'sha256': 'b' * 64}
    cpref = {'path': str(train.MAIN_ROOT / 'UPDATE003328.json'), 'sha256': 'c' * 64}
    pref = {'path': str(train.POLICY), 'sha256': 'd' * 64}
    plan = {'stage': stage, 'runtime': deepcopy(current),
        'driver_requalification': g.native_future_state_driver_requalification_metadata(original, current),
        'inputs': {'main_plan': deepcopy(main), 'checkpoint_receipt': deepcopy(cpref)}}
    producer = deepcopy(current); producer['gpu'] = deepcopy(original['gpu'])
    receipt = {'logical_cursor': 3328, 'optimizer_updates': 3328, 'producer_lane': 'main',
        'original_plan': deepcopy(main), 'execution_plan': deepcopy(main), 'producer_runtime': producer,
        'proof': {'model_state': 'e' * 64}}
    policy = {k: False for k in ('training_enabled', 'native_training_enabled', 'native_launch_authorized', 'full_epoch_training_allowed', 'full_val_allowed')}
    policy['training_enabled'] = policy['native_training_enabled'] = True
    policy[stage] = suffix.gate_expected(binding, True)
    refs = {str(train.MAIN_PLAN): main, str(train.MAIN_ROOT / 'UPDATE003328.json'): cpref, str(train.POLICY): pref}
    objects = {binding['path']: plan, cpref['path']: receipt, pref['path']: policy}
    expected = {binding['path']: binding, cpref['path']: cpref, pref['path']: pref}
    def read(ref):
        assert ref == expected[ref['path']]
        return objects[ref['path']]
    guard = {'kind': 'bias_suffix_driver_requalification_order', 'plan': binding,
        'plan_path': binding['path'], 'output_root': str(Path('/home/andre2/GX1_RUNS') / BASENAME)}
    monkeypatch.setattr(g, 'require_native_future_state_bias_suffix_execution', lambda **k: guard)
    monkeypatch.setattr(train, 'ref', lambda path: deepcopy(refs[str(path)]))
    monkeypatch.setattr(train, 'read', read)
    return train, original, current, plan, policy, receipt, guard, main


def test_driver_loader_accepts_only_admitted003_metadata_and_authentic3328_binding(monkeypatch):
    train, original, current, plan, policy, receipt, guard, main = _driver_authority_fixture(monkeypatch)
    proof = train.runtime_compatible(original, current, driver_requalification={'plan_sha256': SHA})
    assert proof == {'original_plan': main, 'expected_cursor': 3328, 'expected_state_proof': receipt['proof']}
    assert plan['driver_requalification']['equivalence_proven'] is False


@pytest.mark.parametrize('field', ['training_enabled', 'native_training_enabled', 'native_launch_authorized', 'full_epoch_training_allowed', 'full_val_allowed'])
def test_driver_loader_rejects_any_wrong_global_flag(monkeypatch, field):
    train, original, current, plan, policy, receipt, guard, main = _driver_authority_fixture(monkeypatch)
    policy[field] = not policy[field]
    with pytest.raises(ValueError, match='EXACT_TECHNICAL_TRAINING_FLAGS'):
        train.runtime_compatible(original, current, driver_requalification={'plan_sha256': SHA})


@pytest.mark.parametrize('mutation', ['old_guard', 'old_plan_path', 'old_output', 'unadmitted_gate', 'wrong_stage', 'changed_runtime', 'claimed_equivalence', 'wrong_main', 'wrong_checkpoint', 'wrong_cursor', 'wrong_lane', 'wrong_producer_runtime'])
def test_driver_loader_rejects_scope_runtime_or_source_checkpoint_substitution(monkeypatch, mutation):
    train, original, current, plan, policy, receipt, guard, main = _driver_authority_fixture(monkeypatch)
    if mutation == 'old_guard': guard['kind'] = 'bias_suffix_order'
    elif mutation == 'old_plan_path': guard['plan_path'] = guard['plan_path'].replace('_003.json', '_002.json')
    elif mutation == 'old_output': guard['output_root'] = guard['output_root'].replace('_003', '_002')
    elif mutation == 'unadmitted_gate': policy[plan['stage']]['execution_authorized'] = False
    elif mutation == 'wrong_stage': plan['stage'] += '_other'
    elif mutation == 'changed_runtime': plan['runtime']['physical_windows_boot_id'] += 1
    elif mutation == 'claimed_equivalence': plan['driver_requalification']['equivalence_proven'] = True
    elif mutation == 'wrong_main': plan['inputs']['main_plan']['sha256'] = 'f' * 64
    elif mutation == 'wrong_checkpoint': plan['inputs']['checkpoint_receipt']['path'] = str(train.MAIN_ROOT / 'FINAL.json')
    elif mutation == 'wrong_cursor': receipt['logical_cursor'] = 3329
    elif mutation == 'wrong_lane': receipt['producer_lane'] = 'other'
    else: receipt['producer_runtime']['gpu']['driver_version'] = '610.88'
    with pytest.raises(ValueError):
        train.runtime_compatible(original, current, driver_requalification={'plan_sha256': SHA})


@pytest.mark.parametrize('mutation', ['cursor', 'main', 'proof'])
def test_strict_restore_driver_binding_rejects_substitution_before_parameter_operations(monkeypatch, mutation):
    import sys, types
    from copy import deepcopy
    train, original, current, plan, policy, receipt, guard, main = _driver_authority_fixture(monkeypatch)
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace())
    cursor = 3329 if mutation == 'cursor' else 3328
    original_plan = deepcopy(main)
    if mutation == 'main': original_plan['sha256'] = 'f' * 64
    proof = deepcopy(receipt['proof'])
    if mutation == 'proof': proof['model_state'] = 'f' * 64
    trajectory = {'schedule': {'logical_batches': 3694, 'batches_per_pass': 3694, 'update_flags': [True] * 3694},
                  'runtime_compatibility': original}
    checkpoint = {'schema_version': 'gx1_native_future_state_training_checkpoint_v1', 'trajectory': trajectory,
        'producer': {'original_plan': original_plan}, 'position': train.position(trajectory['schedule'], cursor),
        'model_state': {}, 'optimizer_state': {}, 'scheduler_state': None, 'rng_state': {}, 'training': True,
        'optimizer_parameter_names': [], 'steps': [{}] * cursor, 'scores': {}}
    monkeypatch.setattr(train, 'checkpoint_proof', lambda cp: proof)
    with pytest.raises(ValueError, match='EXACT_DRIVER_REQUALIFICATION_RESTORE_SCOPE'):
        train.strict_restore(None, None, checkpoint, trajectory=trajectory, original_plan_ref=original_plan,
            expected_cursor=cursor, current_runtime=current, expected_state_proof=proof,
            driver_requalification={'plan_sha256': SHA})


def main_loader_owners():
    owners = main_guard_owners()
    owners.pop('gx1/synthetic_main_owner_0.py')
    repo = Path(g.__file__).resolve().parents[2]
    name = 'gx1/audit/native_future_state_training_v1.py'
    owners[name] = hashlib.sha256(subprocess.check_output(['git', 'show', 'f96d89f88:' + name], cwd=repo)).hexdigest()
    return owners


def test_driver_source_bridge_preserves_all_loader_kernel_operations_and59_owners():
    proof = g.native_future_state_bias_guard_source_bridge(main_loader_owners(),
        producer_commit='f96d89f88ea742a9a90460d81cc386f8abea6e2a', driver_requalification=True)
    assert proof['checked_guard_owners'] == 3 and proof['other_main_owners_to_be_checked_by_caller'] == 59
    assert 'gx1/audit/native_future_state_training_v1.py' in proof['owners']
    assert 'all_tensor_optimizer_rng_operations_AST_equal' in proof['owners']['gx1/audit/native_future_state_training_v1.py']['proof']


@pytest.mark.parametrize('mutation', ['kernel', 'option_default', 'metadata_binding'])
def test_driver_source_bridge_rejects_any_loader_kernel_or_metadata_bypass(monkeypatch, mutation):
    actual = Path.read_bytes
    target = Path(g.__file__).resolve().parents[2] / 'gx1/audit/native_future_state_training_v1.py'
    tree = ast.parse(actual(target));strict = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'strict_restore')
    if mutation == 'kernel':
        call = next(n for n in ast.walk(strict) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'load_state_dict' and any(k.arg == 'strict' for k in n.keywords))
        next(k for k in call.keywords if k.arg == 'strict').value = ast.Constant(value=False)
    elif mutation == 'option_default': strict.args.kw_defaults[-1] = ast.Constant(value=True)
    else:
        conditional = next(n for n in strict.body if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name) and n.test.left.id == 'driver_requalification')
        conditional.body = [ast.Pass()]
    changed_bytes = ast.unparse(tree).encode()
    monkeypatch.setattr(Path, 'read_bytes', lambda path, *a, **kw: changed_bytes if path == target else actual(path, *a, **kw))
    reason = {'kernel': 'ALL_LOADER_KERNEL_AST_UNCHANGED', 'option_default': 'EXACT_DRIVER_LOADER_OPTION', 'metadata_binding': 'EXACT_DRIVER_RESTORE_METADATA_BLOCK'}[mutation]
    with pytest.raises(RuntimeError, match=reason):
        g.native_future_state_bias_guard_source_bridge(main_loader_owners(),
            producer_commit='f96d89f88ea742a9a90460d81cc386f8abea6e2a', driver_requalification=True)
