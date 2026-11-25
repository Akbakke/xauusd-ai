"""Exact row-membership bindings reuse the qualified all-TF containment, without NN."""
import ast
import copy
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest
from gx1.contracts import gx1_capped_execution_v1 as g

REPO = Path(__file__).resolve().parents[1]
ANCHOR = '8fd74a76cc45ac7306da4b905c82e9cd9b12976d'
NAME = 'NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_ONE_EPOCH_20261125_001'
ORDER_NAME = 'NATIVE_PROVIDER_HLC_ALL_TF_BLOCK_ORDER_20261124_001'
OLD_NAME = 'NATIVE_PROVIDER_HLC_ALL_TF_ONE_EPOCH_20261124_002'
MODULE = 'gx1.audit.native_provider_hlc_row_membership_fit_v1'
ORDER_MODULE = 'gx1.audit.native_provider_hlc_block_order_v1'
OLD_MODULE = 'gx1.audit.native_provider_hlc_learning_v1'
PY_OWNER = 'gx1/contracts/gx1_capped_execution_v1.py'
SHELL_OWNER = 'scripts/gx1_capped_run.sh'
SHELL_CONSTANT = '_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_GUARD_SHELL_ADDITIONS'


def literal_from_test(name):
    text = (REPO / 'tests/test_native_provider_hlc_all_tf_learning_guard_v1.py').read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return ast.literal_eval(node.value)


ENV = literal_from_test('ENV')
EXECUTION = literal_from_test('EXECUTION')


def identity(route, mode):
    name, module, prefix, schema_prefix = {
        'row': (NAME, MODULE, '--row-membership-', 'gx1_native_provider_hlc_row_membership_'),
        'order': (ORDER_NAME, ORDER_MODULE, '--block-order-', 'gx1_native_provider_hlc_block_order_'),
        'legacy': (OLD_NAME, OLD_MODULE, '--all-tf-learning-', 'gx1_native_provider_hlc_all_tf_'),
    }[route]
    flag = prefix + ('resume-plan-sha256' if mode == 'resume' else 'plan-sha256')
    schema = schema_prefix + ('remaining_epoch_plan_v1' if mode == 'resume' else 'one_epoch_plan_v1')
    return name, module, flag, schema


@pytest.fixture(params=[(route, mode) for route in ('row', 'order', 'legacy') for mode in ('main', 'resume')])
def scope(monkeypatch, tmp_path, request):
    route, mode = request.param
    name, module, flag, schema = identity(route, mode)
    repo = Path(g.__file__).resolve().parents[2]
    base = Path('/home/andre2/GX1_RUNS') / name
    file = tmp_path / 'fixture.json'; file.write_text('{}')
    file_stat, dir_stat = file.stat(), tmp_path.stat()
    read_bytes, lstat, path_stat, resolve = Path.read_bytes, Path.lstat, Path.stat, Path.resolve
    exists, is_symlink = Path.exists, Path.is_symlink
    state = {'route': route, 'mode': mode, 'repo': repo, 'base': base}
    monkeypatch.setattr(Path, 'read_bytes', lambda p: state['raw'] if p == state.get('plan') else read_bytes(p))
    monkeypatch.setattr(Path, 'lstat', lambda p: file_stat if p == state.get('plan') else lstat(p))
    monkeypatch.setattr(Path, 'stat', lambda p, *a, **kw: dir_stat if p == state.get('output', base).parent else path_stat(p, *a, **kw))
    monkeypatch.setattr(Path, 'resolve', lambda p, strict=False: p if p in (state.get('plan'), state.get('output', base).parent) else resolve(p, strict=strict))
    monkeypatch.setattr(Path, 'exists', lambda p: False if p == state.get('output') else exists(p))
    monkeypatch.setattr(Path, 'is_symlink', lambda p: False if p == state.get('output') else is_symlink(p))
    monkeypatch.setattr(Path, 'cwd', lambda: repo)
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: {'kernel_scope': True, 'training_authority_granted': True})
    def bind(plan):
        raw = json.dumps(plan, sort_keys=True).encode(); sha = hashlib.sha256(raw).hexdigest()
        state.update(data=plan, raw=raw, sha=sha,
                     plan=repo / 'configs/research' / (name + '.json') if mode == 'main' else base / 'RECOVERY_PLANS' / (sha + '.json'),
                     output=base if mode == 'main' else base / ('RECOVERY_' + sha))
        state['command'] = [str(repo / '.venv/bin/python'), '-m', module, flag, sha]
        monkeypatch.setattr(g.sys, 'orig_argv', state['command'].copy())
        monkeypatch.setattr(g.sys, 'executable', state['command'][0])
        for key, value in ENV.items(): monkeypatch.setenv(key, sha if value == 'SHA' else mode if value == 'MODE' else value)
    state['bind'] = bind
    bind({'schema_version': schema, 'stage': name.lower() + ('_recovery' if mode == 'resume' else ''),
          'output_root': str(base), **({'output_directory_prefix': 'RECOVERY_'} if mode == 'resume' else {}),
          'recipe': {'passes': 1}, 'backbone_gradient_policy': None,
          'model_contract': {'contract_sha256': '6d6ac22819e54c7b12aebcf4e1afe27b990d8a4d90d16500e352c7353d99cca0'},
          'execution': copy.deepcopy(EXECUTION)})
    return state


def proof(scope, **kwargs):
    return g.require_native_provider_hlc_all_tf_learning_execution(plan_sha256=scope['sha'], mode=kwargs.get('mode', scope['mode']))


@pytest.mark.parametrize('preflight', [False, True])
def test_new_and_old_exact_scope_reuse_resource_proof_without_authority(monkeypatch, scope, preflight):
    if preflight: monkeypatch.setattr(g.sys, 'orig_argv', scope['command'] + ['--preflight'])
    result = proof(scope)
    assert result['kernel_scope'] and result['resource_proof_only']
    assert result['training_authority_granted'] is result['optimizer_authority_granted'] is False
    assert result['plan'] == {'path': str(scope['plan']), 'sha256': scope['sha']}
    assert result['output_root'] == str(scope['output'])
    assert (result['outer_seconds'], result['internal_seconds'], result['numerical_threads']) == (12000, 11970, 8)
    assert result['cpu_affinity'] == list(range(19))


def test_legacy_resume_v2_remains_accepted(scope):
    if scope['route'] != 'legacy' or scope['mode'] != 'resume': return
    plan = copy.deepcopy(scope['data']); plan['schema_version'] = 'gx1_native_provider_hlc_all_tf_remaining_epoch_plan_v2'
    scope['bind'](plan); assert proof(scope)['kernel_scope']


@pytest.mark.parametrize('case', ['module_flag_cross', 'foreign_module', 'foreign_python', 'script', 'equals',
                                  'reordered', 'extra', 'lane', 'wrong_mode', 'cwd'])
def test_foreign_or_crossed_cli_never_selects_another_route(monkeypatch, scope, case):
    argv = scope['command'].copy()
    if case == 'module_flag_cross': argv[2] = {'row': ORDER_MODULE, 'order': OLD_MODULE, 'legacy': MODULE}[scope['route']]
    elif case == 'foreign_module': argv[2] += '_foreign'
    elif case == 'foreign_python': argv[0] = '/usr/bin/python3'
    elif case == 'script': argv[1:3] = [str(scope['repo'] / 'gx1/audit/native_provider_hlc_row_membership_fit_v1.py')]
    elif case == 'equals': argv[3:5] = [argv[3] + '=' + argv[4]]
    elif case == 'reordered': argv[3:5] = [argv[4], argv[3]]
    elif case == 'extra': argv += ['--preflight', '--preflight']
    elif case == 'lane': argv += ['--lane', 'main']
    elif case == 'wrong_mode':
        with pytest.raises(RuntimeError): proof(scope, mode='resume' if scope['mode'] == 'main' else 'main')
        return
    elif case == 'cwd': monkeypatch.setattr(Path, 'cwd', lambda: scope['repo'].parent)
    monkeypatch.setattr(g.sys, 'orig_argv', argv)
    with pytest.raises(RuntimeError, match='(CLI|CWD)_INVALID'): proof(scope)


@pytest.mark.parametrize('case', ['other_schema', 'foreign_schema', 'stage', 'root', 'prefix', 'two_passes',
                                  'policy', 'recipe_policy', 'contract', 'execution'])
def test_valid_sha_does_not_authorize_another_plan_contract(scope, case):
    plan = copy.deepcopy(scope['data'])
    if case == 'other_schema': plan['schema_version'] = identity('legacy' if scope['route'] == 'order' else 'order', scope['mode'])[3]
    elif case == 'foreign_schema': plan['schema_version'] += '_foreign'
    elif case == 'stage': plan['stage'] += '_foreign'
    elif case == 'root': plan['output_root'] += '_foreign'
    elif case == 'prefix':
        if scope['mode'] == 'main': return
        plan['output_directory_prefix'] = 'OTHER_'
    elif case == 'two_passes': plan['recipe']['passes'] = 2
    elif case == 'policy': plan['backbone_gradient_policy'] = 'M5_primary'
    elif case == 'recipe_policy': plan['recipe']['backbone_gradient_policy'] = None
    elif case == 'contract': plan['model_contract']['contract_sha256'] = 'f' * 64
    elif case == 'execution': plan['execution']['GPU_core_stop_C'] = 80
    scope['bind'](plan)
    with pytest.raises(RuntimeError): proof(scope)


@pytest.mark.parametrize('case', ['existing_output', 'output_symlink', 'plan_hash', 'new_mode', 'new_lane', 'mixed_old_lane'])
def test_nonfresh_paths_hash_or_environment_fail(monkeypatch, scope, case):
    if case == 'existing_output':
        prior = Path.exists; monkeypatch.setattr(Path, 'exists', lambda p: True if p == scope['output'] else prior(p))
    elif case == 'output_symlink':
        prior = Path.is_symlink; monkeypatch.setattr(Path, 'is_symlink', lambda p: True if p == scope['output'] else prior(p))
    elif case == 'plan_hash':
        prior = Path.read_bytes; monkeypatch.setattr(Path, 'read_bytes', lambda p: scope['raw'] + b' ' if p == scope['plan'] else prior(p))
    elif case == 'new_mode': monkeypatch.setenv('GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE', 'order')
    elif case == 'new_lane': monkeypatch.setenv('GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_LANE', '')
    elif case == 'mixed_old_lane': monkeypatch.setenv('GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE', 'cold')
    with pytest.raises(RuntimeError): proof(scope)


def shell_scope(tmp_path, route, mode, *, case=None, dispatch=False):
    repo = tmp_path / 'repo'; repo.mkdir(); parent = repo / 'runs'; parent.mkdir()
    name, module, flag, _ = identity(route, mode)
    base = parent / name
    raw = b'{"synthetic_block_order_guard":true}\n'; sha = hashlib.sha256(raw).hexdigest()
    if mode == 'main': directory = repo / 'configs/research'
    else: directory = base / 'RECOVERY_PLANS'
    directory.mkdir(parents=True)
    plan = directory / (name + '.json' if mode == 'main' else sha + '.json'); plan.write_bytes(raw)
    output = base if mode == 'main' else base / ('RECOVERY_' + sha)
    argv = [str(repo / '.venv/bin/python'), '-m', module, flag, sha]
    if case == 'preflight': argv += ['--preflight']
    elif case == 'crossed_module': argv[2] = {'row': ORDER_MODULE, 'order': OLD_MODULE, 'legacy': MODULE}[route]
    elif case == 'wrong_sha': argv[4] = 'f' * 64
    elif case == 'wrong_flag': argv[3] = '--plan-sha256'
    elif case == 'wrong_python': argv[0] = '/usr/bin/python3'
    elif case == 'equals': argv[3:5] = [flag + '=' + sha]
    elif case == 'extra': argv += ['--lane', 'main']
    elif case == 'script': argv[1:3] = ['gx1/audit/native_provider_hlc_row_membership_fit_v1.py']
    elif case == 'existing_output': output.mkdir()
    elif case == 'output_symlink': output.symlink_to(repo / 'absent')
    elif case == 'plan_symlink':
        real = plan.with_suffix('.original'); plan.rename(real); plan.symlink_to(real)
    state, functions, dispatch_body = [g._NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_GUARD_SHELL_ADDITIONS[i][2] for i in (0, 1, 2)]
    for scoped_name in (NAME, ORDER_NAME, OLD_NAME): functions = functions.replace('/home/andre2/GX1_RUNS/' + scoped_name, str(parent / scoped_name))
    values = {'REPO_ROOT': str(repo), 'CANONICAL_TRAINER_PYTHON': str(repo / '.venv/bin/python'),
              'NATIVE_PROVIDER_HLC_LEARNING_MODULE': OLD_MODULE, 'JOB_CLASS': 'trainer', 'ATTENDED_SMOKE': 'false',
              'CUDA_PRODUCER_GUARD': 'false', 'MEM': '20G', 'SWAP': '512M', 'POWER_BENCHMARK_SCOPE_JSON': '', 'POWER_BENCHMARK_SCOPE_SHA256': ''}
    script = state + functions + ''.join('\n' + k + '=' + shlex.quote(v) for k, v in values.items())
    if dispatch: script += '\nvalidate_target_command() {\n' + dispatch_body + 'echo FALLTHROUGH; return 99; }\nvalidate_target_command '
    else: script += '\nconfigure_native_provider_hlc_all_tf_learning '
    script += ' '.join(map(shlex.quote, argv)) + '\n'
    script += 'printf "%s,%s,%s,%s,%s,%s,%s\\n" "$NATIVE_PROVIDER_HLC_ALL_TF_LEARNING" "$NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE" "$TRAINER_MAX_WALL_SECONDS" "$TRAINER_MODEL_MAX_WALL_SECONDS" "$TRAINER_GPU_MAX_CORE_TEMP_C" "$TRAINER_GPU_MAX_MEMORY_USED_MIB" "$TRAINER_OUT_BUNDLE_DIR"\n'
    env = {k:v for k,v in os.environ.items() if not k.startswith(('GX1_CAPPED_', 'GX1_NATIVE_', 'GX1_TRAINER_', 'GX1_POWER_'))}
    result = subprocess.run(['bash', '-c', script], cwd=repo, env=env, text=True, capture_output=True)
    assert not (output / 'CLAIM.json').exists()
    return result, output


@pytest.mark.parametrize('route', ['row', 'order', 'legacy'])
@pytest.mark.parametrize('mode', ['main', 'resume'])
@pytest.mark.parametrize('preflight', [False, True])
@pytest.mark.parametrize('dispatch', [False, True])
def test_same_shell_route_and_caps_preserve_old_and_add_only_exact_new_scope(tmp_path, route, mode, preflight, dispatch):
    result, output = shell_scope(tmp_path, route, mode, case='preflight' if preflight else None, dispatch=dispatch)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'true,' + mode + ',12000,11970,75,12288,' + str(output)
    assert not output.exists()


@pytest.mark.parametrize('route', ['row', 'order', 'legacy'])
@pytest.mark.parametrize('mode', ['main', 'resume'])
@pytest.mark.parametrize('case', ['crossed_module', 'wrong_sha', 'wrong_flag', 'wrong_python', 'equals', 'extra',
                                  'script', 'existing_output', 'output_symlink', 'plan_symlink'])
def test_shell_rejects_crossed_modules_flags_hashes_and_nonfresh_paths(tmp_path, route, mode, case):
    result, _ = shell_scope(tmp_path, route, mode, case=case)
    assert result.returncode == 75, (result.stdout, result.stderr)


@pytest.mark.parametrize('case', ['wrong_flag', 'script', 'equals', 'extra'])
def test_order_module_or_alias_cannot_fall_through_old_dispatch(tmp_path, case):
    result, _ = shell_scope(tmp_path, 'row', 'main', case=case, dispatch=True)
    assert result.returncode == 75 and 'FALLTHROUGH' not in result.stdout


def source_node(text, name, assignment=False):
    nodes = [n for n in ast.parse(text).body if (isinstance(n, ast.Assign) and assignment and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
             or (isinstance(n, ast.FunctionDef) and not assignment and n.name == name)]
    assert len(nodes) == 1
    return ast.get_source_segment(text, nodes[0]), nodes[0]


def shell_function(text, name):
    start = text.index(name + '() {\n'); stop = text.index('\n}\n', start) + 3
    return text[start:stop]


PYTHON_INSERTIONS = (('    block_flag = "--block-order-plan-sha256" if mode == "main" else "--block-order-resume-plan-sha256"\n', '    row_flag = "--row-membership-plan-sha256" if mode == "main" else "--row-membership-resume-plan-sha256"\n    row_membership = sys.orig_argv[1:4] == ["-m", "gx1.audit.native_provider_hlc_row_membership_fit_v1", row_flag]\n'), ('    base = Path("/home/andre2/GX1_RUNS") / basename\n', '    if row_membership:\n        basename = "NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_ONE_EPOCH_20261125_001"\n'), ('    command = [str(repo / ".venv/bin/python"), "-m", module, flag, plan_sha256]\n', '    if row_membership:\n        module, flag = "gx1.audit.native_provider_hlc_row_membership_fit_v1", row_flag\n'), ('    stage = basename.lower() + ("_recovery" if mode == "resume" else "")\n', '    if row_membership:\n        schemas = (("gx1_native_provider_hlc_row_membership_one_epoch_plan_v1",) if mode == "main" else\n                   ("gx1_native_provider_hlc_row_membership_remaining_epoch_plan_v1",))\n'))


def test_production_bridge_preserves_all_8fd_bytes_outside_exact_declarations():
    report = g.native_provider_hlc_row_membership_fit_guard_source_bridge()
    assert report['producer_commit'] == ANCHOR and report['checked_guard_owners'] == 2
    assert report['old_guard_routes_preserved'] and report['old_numeric_resource_environment_function_unchanged']
    for name, proof in report['owners'].items():
        before = subprocess.check_output(['git', 'show', ANCHOR + ':' + name], cwd=REPO)
        assert proof['original_sha256'] == proof['normalized_sha256'] == hashlib.sha256(before).hexdigest()
        assert proof['current_sha256'] == hashlib.sha256((REPO / name).read_bytes()).hexdigest()
    old = subprocess.check_output(['git', 'show', ANCHOR + ':' + PY_OWNER], cwd=REPO).decode()
    current = (REPO / PY_OWNER).read_text()
    assert source_node(old, 'require_native_provider_hlc_all_tf_learning_execution')[0] == source_node(current, 'require_native_provider_hlc_all_tf_learning_execution')[0]
    original_helper = source_node(old, '_native_provider_hlc_all_tf_learning_bound_plan')[0]
    helper = source_node(current, '_native_provider_hlc_all_tf_learning_bound_plan')[0]
    for anchor, fragment in reversed(PYTHON_INSERTIONS):
        assert helper.count(fragment + anchor) == 1
        helper = helper.replace(fragment + anchor, anchor, 1)
    assert helper == original_helper
    before = ast.literal_eval(source_node(old, SHELL_CONSTANT, True)[1].value)
    current = getattr(g, SHELL_CONSTANT)
    assert len(before) == len(current) == 5
    assert all(before[index] == current[index] for index in (0, 3, 4))


@pytest.mark.parametrize('case', ['other_python', 'resource_function', 'other_shell', 'duplicate_matcher', 'moved_dispatch', 'extra_bridge_symbol'])
def test_production_source_bridge_rejects_every_undeclared_delta(monkeypatch, case):
    original_read = Path.read_text
    def modified(path, *args, **kwargs):
        text = original_read(path, *args, **kwargs)
        if path == REPO / PY_OWNER:
            if case == 'other_python': text += '\n# unregistered old-owner delta\n'
            elif case == 'resource_function': text = text.replace('"resource_proof_only": True', '"resource_proof_only": False', 1)
            elif case == 'extra_bridge_symbol': text = text.replace('\n# END GX1 ROW MEMBERSHIP FIT GUARD BRIDGE 20261125 001\n', '\nFOREIGN_AUTHORITY = True\n# END GX1 ROW MEMBERSHIP FIT GUARD BRIDGE 20261125 001\n', 1)
        elif path == REPO / SHELL_OWNER:
            if case == 'other_shell': text += '\n# unregistered shell delta\n'
            elif case == 'duplicate_matcher': text += getattr(g, SHELL_CONSTANT)[1][2]
            elif case == 'moved_dispatch':
                fragment = getattr(g, SHELL_CONSTANT)[2][2]
                text = text.replace(fragment, '', 1) + fragment
        return text
    monkeypatch.setattr(Path, 'read_text', modified)
    with pytest.raises(RuntimeError, match='GX1_ROW_MEMBERSHIP_FIT_GUARD_BRIDGE'):
        g.native_provider_hlc_row_membership_fit_guard_source_bridge()


def test_bridge_accepts_only_the_measured_full_source_anchor():
    with pytest.raises(RuntimeError, match='EXACT_PRODUCER'):
        g.native_provider_hlc_row_membership_fit_guard_source_bridge(producer_commit='8fd74a76')


@pytest.mark.parametrize('field', ['GX1_CAPPED_MEMORY_BYTES', 'GX1_CAPPED_SWAP_BYTES', 'GX1_CAPPED_TASKS_MAX', 'OMP_NUM_THREADS', 'GX1_TRAINER_MAX_WALL_SECONDS', 'GX1_TRAINER_MODEL_MAX_WALL_SECONDS', 'GX1_TRAINER_GPU_MAX_CORE_TEMP_C', 'GX1_TRAINER_GPU_MAX_POWER_DRAW_W'])
def test_reused_hard_limits_cannot_be_weakened_for_any_namespace(monkeypatch, scope, field):
    monkeypatch.setenv(field, os.environ[field] + '0')
    with pytest.raises(RuntimeError, match='ENV_INVALID'): proof(scope)
