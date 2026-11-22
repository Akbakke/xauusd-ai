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
BASENAME = 'NATIVE_FUTURE_STATE_BIAS_SUFFIX_ORDER_20261122_002'
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
