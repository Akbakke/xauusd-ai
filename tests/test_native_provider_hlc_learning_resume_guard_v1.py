"""Synthetic command/resource/source proofs for cursor512 resume containment; no worker or payload."""
import hashlib
import shlex
import stat
import subprocess
from pathlib import Path

import pytest
from gx1.contracts import gx1_capped_execution_v1 as g

RAW = b'{"synthetic_guard_fixture":true}\n'
SHA = hashlib.sha256(RAW).hexdigest()
MODULE = 'gx1.audit.native_provider_hlc_learning_v1'
BASENAME = 'NATIVE_PROVIDER_HLC_ONE_EPOCH_RESUME_20261123_001'
PARENT = 'aa6f23bb6ad0d867fc23f3604b8399e89b633ed6'
GUARDS = {
    'gx1/contracts/gx1_capped_execution_v1.py': '2a26866baba68182d460a38f556004b043d8aef1aaaa8d21e4eaecc77c534f8c',
    'scripts/gx1_capped_run.sh': '313d1848242840f99637b66df7aad46cb3847129cf8675009901ce79fc3a4b31',
}
ENV = {
    'GX1_CAPPED_MEMORY_BYTES': '21474836480', 'GX1_CAPPED_SWAP_BYTES': '536870912',
    'GX1_CAPPED_TASKS_MAX': '128', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
    'OPENBLAS_NUM_THREADS': '8', 'NUMEXPR_NUM_THREADS': '8', 'VECLIB_MAXIMUM_THREADS': '8',
    'GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME': 'true', 'GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256': SHA,
    'GX1_NATIVE_PROVIDER_HLC_LEARNING': 'false', 'GX1_NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256': '',
    'GX1_NATIVE_PROVIDER_HLC_REPLAY': 'false', 'GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256': '',
    'GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE': '', 'GX1_NATIVE_PROVIDER_HLC_INITIAL': 'false',
    'GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256': '', 'GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE': 'false',
    'GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256': '',
    'GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE': 'false', 'GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256': '',
    'GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX': 'false', 'GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256': '',
    'GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN': 'false', 'GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256': '',
    'GX1_NATIVE_FUTURE_STATE_THROUGHPUT': 'false', 'GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256': '',
    'GX1_NATIVE_FUTURE_STATE_TRAINING': 'false', 'GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256': '',
    'GX1_NATIVE_FUTURE_STATE_TRAINING_LANE': '', 'GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION': 'false',
    'GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256': '', 'GX1_NATIVE_FUTURE_STATE_CUDA_LANE': '',
    'GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS': 'false', 'GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256': '',
    'GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE': '', 'GX1_TRAINER_MAX_WALL_SECONDS': '12000',
    'GX1_TRAINER_MODEL_MAX_WALL_SECONDS': '11970', 'GX1_TRAINER_ATTENDED_STAGE_REQUIRED': 'false',
    'GX1_TRAINER_GPU_INDEX': '0', 'GX1_TRAINER_GPU_MAX_POWER_LIMIT_W': '250',
    'GX1_TRAINER_GPU_MAX_POWER_DRAW_W': '250', 'GX1_TRAINER_GPU_MAX_CORE_TEMP_C': '75',
    'GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C': '80', 'GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB': '12288',
    'GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS': '1', 'GX1_TRAINER_CPU_MAX_TEMP_C': '80',
    'GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W': '125', 'GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES': '8589934592',
    'GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES': '8589934592', 'GX1_POWER_BENCHMARK_SCOPE_JSON': '',
    'GX1_POWER_BENCHMARK_SCOPE_SHA256': '',
}


@pytest.fixture
def scope(monkeypatch, tmp_path):
    """Replace only declared plan/output metadata; read no payload or model."""
    repo = Path(g.__file__).resolve().parents[2]
    plan = repo / 'configs/research' / (BASENAME + '.json')
    output = Path('/home/andre2/GX1_RUNS') / BASENAME
    fixture = tmp_path / 'plan.json'; fixture.write_bytes(RAW)
    read_bytes, lstat, resolve = Path.read_bytes, Path.lstat, Path.resolve
    exists, is_symlink = Path.exists, Path.is_symlink
    monkeypatch.setattr(Path, 'read_bytes', lambda p: RAW if p == plan else read_bytes(p))
    monkeypatch.setattr(Path, 'lstat', lambda p: fixture.stat() if p == plan else tmp_path.stat() if p == output.parent else lstat(p))
    monkeypatch.setattr(Path, 'resolve', lambda p, strict=False: p if p in (plan, output.parent) else resolve(p, strict=strict))
    monkeypatch.setattr(Path, 'exists', lambda p: False if p == output else exists(p))
    monkeypatch.setattr(Path, 'is_symlink', lambda p: False if p == output else is_symlink(p))
    monkeypatch.setattr(Path, 'cwd', lambda: repo)
    command = [str(repo / '.venv/bin/python'), '-m', MODULE, '--resume-plan-sha256', SHA]
    monkeypatch.setattr(g.sys, 'orig_argv', command.copy())
    monkeypatch.setattr(g.sys, 'executable', command[0])
    for name, value in ENV.items(): monkeypatch.setenv(name, value)
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: {'kernel_scope': True})
    return {'repo': repo, 'plan': plan, 'output': output, 'command': command}


@pytest.mark.parametrize('preflight', [False, True])
def test_learning_exact_fresh_scope_is_resource_proof_only(monkeypatch, scope, preflight):
    if preflight: monkeypatch.setattr(g.sys, 'orig_argv', scope['command'] + ['--preflight'])
    proof = g.require_native_provider_hlc_learning_resume_execution(plan_sha256=SHA)
    assert proof['kernel_scope'] and proof['kind'] == 'provider_hlc_learning_resume'
    assert proof['resource_proof_only'] is True
    assert proof['training_authority_granted'] is proof['optimizer_authority_granted'] is False
    assert (proof['outer_seconds'], proof['internal_seconds'], proof['kill_after_seconds']) == (12000, 11970, 10)
    assert proof['cpu_affinity'] == list(range(19)) and proof['numerical_threads'] == 8
    assert proof['plan'] == {'path': str(scope['plan']), 'sha256': SHA}
    assert proof['output_root'] == str(scope['output'])
    assert not scope['output'].exists()


@pytest.mark.parametrize('field', list(ENV))
def test_learning_rejects_each_changed_mode_or_resource_before_kernel(monkeypatch, scope, field):
    monkeypatch.setenv(field, ENV[field] + 'wrong')
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: pytest.fail('invalid scope reached kernel'))
    with pytest.raises(RuntimeError, match='PROVIDER_HLC_LEARNING_RESUME_ENV_INVALID'):
        g.require_native_provider_hlc_learning_resume_execution(plan_sha256=SHA)


@pytest.mark.parametrize('bad', ['A' * 64, 'f' * 63, None, True])
def test_learning_invalid_sha_before_scope(monkeypatch, bad):
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: pytest.fail('invalid SHA reached kernel'))
    with pytest.raises(RuntimeError, match='PLAN_SHA256_INVALID'):
        g.require_native_provider_hlc_learning_resume_execution(plan_sha256=bad)


@pytest.mark.parametrize('case', ['wrong_python', 'script', 'lane', 'extra', 'reordered', 'bad_executable', 'wrong_cwd', 'old_flag', 'equals_flag', 'double_flag'])
def test_learning_rejects_noncanonical_python_cli_or_cwd(monkeypatch, scope, case):
    argv = scope['command'].copy()
    if case == 'wrong_python': argv[0] = '/usr/bin/python3'
    elif case == 'old_flag': argv[3] = '--plan-sha256'
    elif case == 'equals_flag': argv[3:5] = ['--resume-plan-sha256=' + SHA]
    elif case == 'double_flag': argv += ['--plan-sha256', SHA]
    elif case == 'script': argv[1:3] = [str(scope['repo'] / 'gx1/audit/native_provider_hlc_learning_v1.py')]
    elif case == 'lane': argv += ['--lane', 'main']
    elif case == 'extra': argv += ['--preflight', '--preflight']
    elif case == 'reordered': argv[3:5] = [SHA, '--resume-plan-sha256']
    elif case == 'bad_executable': monkeypatch.setattr(g.sys, 'executable', '/usr/bin/python3')
    elif case == 'wrong_cwd': monkeypatch.setattr(Path, 'cwd', lambda: scope['repo'].parent)
    monkeypatch.setattr(g.sys, 'orig_argv', argv)
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: pytest.fail('invalid CLI reached kernel'))
    with pytest.raises(RuntimeError, match='(CLI|CWD)_INVALID'):
        g.require_native_provider_hlc_learning_resume_execution(plan_sha256=SHA)


@pytest.mark.parametrize('case', ['plan_hash', 'plan_symlink', 'plan_parent_symlink', 'plan_missing',
                                  'existing_output', 'output_symlink', 'output_parent_symlink', 'missing_parent'])
def test_learning_rejects_changed_plan_or_nonfresh_output(monkeypatch, scope, case):
    lstat, resolve = Path.lstat, Path.resolve
    def changed_lstat(p):
        if p == scope['plan'] and case == 'plan_missing' or p == scope['output'].parent and case == 'missing_parent':
            raise FileNotFoundError(str(p))
        if p == scope['plan'] and case == 'plan_symlink':
            return type('Metadata', (), {'st_mode': stat.S_IFLNK})()
        return lstat(p)
    monkeypatch.setattr(Path, 'lstat', changed_lstat)
    if case == 'plan_hash':
        original = Path.read_bytes
        monkeypatch.setattr(Path, 'read_bytes', lambda p: RAW + b'changed' if p == scope['plan'] else original(p))
    elif case in ('plan_parent_symlink', 'output_parent_symlink'):
        changed = scope['plan'] if case == 'plan_parent_symlink' else scope['output'].parent
        monkeypatch.setattr(Path, 'resolve', lambda p, strict=False: Path('/different/canonical/path') if p == changed else resolve(p, strict=strict))
    elif case == 'existing_output':
        original = Path.exists
        monkeypatch.setattr(Path, 'exists', lambda p: True if p == scope['output'] else original(p))
    elif case == 'output_symlink':
        original = Path.is_symlink
        monkeypatch.setattr(Path, 'is_symlink', lambda p: True if p == scope['output'] else original(p))
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: pytest.fail('invalid paths reached kernel'))
    with pytest.raises(RuntimeError, match='PLAN_OUTPUT_INVALID'):
        g.require_native_provider_hlc_learning_resume_execution(plan_sha256=SHA)


def harness(tmp_path, *, case=None, configure=False, dispatch=False):
    repo = tmp_path / 'repo'; repo.mkdir()
    config = repo / 'configs/research'; config.mkdir(parents=True)
    plan = config / (BASENAME + '.json'); plan.write_bytes(RAW)
    argv = [str(repo / '.venv/bin/python'), '-m', MODULE, '--resume-plan-sha256', SHA]
    output = repo / 'learning-output'
    if case == 'preflight': argv += ['--preflight']
    elif case == 'extra': argv += ['--unsafe']
    elif case == 'wrong_module': argv[2] = 'gx1.audit.native_provider_hlc_replay_v1'
    elif case == 'wrong_python': argv[0] = '/usr/bin/python3'
    elif case == 'wrong_sha': argv[4] = 'f' * 64
    elif case == 'lane': argv += ['--lane', 'main']
    elif case == 'reordered': argv[3:5] = [SHA, '--resume-plan-sha256']
    elif case == 'old_flag': argv[3] = '--plan-sha256'
    elif case == 'equals_flag': argv[3:5] = ['--resume-plan-sha256=' + SHA]
    elif case == 'double_flag': argv += ['--plan-sha256', SHA]
    elif case == 'script': argv[1:3] = [str(repo / 'gx1/audit/native_provider_hlc_learning_v1.py')]
    elif case == 'symlink':
        target = plan.with_suffix('.original'); plan.rename(target); plan.symlink_to(target)
    elif case == 'missing_plan': plan.unlink()
    elif case == 'existing_output': output.mkdir()
    elif case == 'output_symlink': output.symlink_to(repo / 'absent')
    elif case == 'output_parent_symlink':
        parent = repo / 'real-parent'; parent.mkdir(); (repo / 'linked-parent').symlink_to(parent)
        output = repo / 'linked-parent/learning-output'
    state, functions = (g._NATIVE_PROVIDER_HLC_LEARNING_RESUME_GUARD_SHELL_ADDITIONS[i][2] for i in (0, 1))
    functions = functions.replace('/home/andre2/GX1_RUNS/' + BASENAME, str(output))
    functions += g._NATIVE_PROVIDER_HLC_LEARNING_GUARD_SHELL_ADDITIONS[1][2]
    script = g._NATIVE_PROVIDER_HLC_LEARNING_GUARD_SHELL_ADDITIONS[0][2] + state + functions + '\nREPO_ROOT=' + shlex.quote(str(repo))
    script += '\nCANONICAL_TRAINER_PYTHON=' + shlex.quote(str(repo / '.venv/bin/python'))
    values = {'JOB_CLASS': 'trainer', 'ATTENDED_SMOKE': 'false', 'CUDA_PRODUCER_GUARD': 'false',
              'MEM': '20G', 'SWAP': '512M', 'POWER_BENCHMARK_SCOPE_JSON': '', 'POWER_BENCHMARK_SCOPE_SHA256': ''}
    if case == 'audit_class': values['JOB_CLASS'] = 'audit'
    elif case == 'wrong_memory': values['MEM'] = '4G'
    elif case == 'wrong_swap': values['SWAP'] = '1G'
    elif case == 'attended': values['ATTENDED_SMOKE'] = 'true'
    elif case == 'cuda_producer': values['CUDA_PRODUCER_GUARD'] = 'true'
    elif case == 'power_scope': values['POWER_BENCHMARK_SCOPE_SHA256'] = 'f' * 64
    script += ''.join('\n' + key + '=' + shlex.quote(value) for key, value in values.items())
    if dispatch:
        script += '\nvalidate_target_command() {\n' + g._NATIVE_PROVIDER_HLC_LEARNING_RESUME_GUARD_SHELL_ADDITIONS[2][2]
        script += g._NATIVE_PROVIDER_HLC_LEARNING_GUARD_SHELL_ADDITIONS[2][2]
        script += 'echo UNEXPECTED_FALLTHROUGH; }\nvalidate_target_command '
    else:
        script += '\n' + ('configure_native_provider_hlc_learning_resume ' if configure else 'matches_bound_native_provider_hlc_learning_resume ')
    script += ' '.join(map(shlex.quote, argv)) + '\n'
    if configure or dispatch:
        script += 'printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\\n" "$NATIVE_PROVIDER_HLC_LEARNING_RESUME" "$TRAINER_MAX_WALL_SECONDS" "$TRAINER_MODEL_MAX_WALL_SECONDS" "$TRAINER_GPU_INDEX" "$TRAINER_GPU_MAX_POWER_LIMIT_W" "$TRAINER_GPU_MAX_POWER_DRAW_W" "$TRAINER_GPU_MAX_CORE_TEMP_C" "$TRAINER_GPU_MAX_MEMORY_TEMP_C" "$TRAINER_GPU_MAX_MEMORY_USED_MIB" "$TRAINER_CPU_MAX_TEMP_C" "$TRAINER_CPU_MAX_PACKAGE_POWER_W" "$TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES"\n'
    result = subprocess.run(['bash', '-c', script], cwd=tmp_path if case == 'wrong_cwd' else repo, text=True, capture_output=True)
    assert not output.exists() or case == 'existing_output'
    assert not (output / 'CLAIM.json').exists()
    return result


@pytest.mark.parametrize('case', [None, 'preflight'])
def test_learning_shell_exact_cli_and_readonly_preflight(tmp_path, case):
    assert harness(tmp_path, case=case).returncode == 0


@pytest.mark.parametrize('case', ['extra', 'wrong_module', 'wrong_python', 'wrong_sha', 'lane', 'reordered',
                                  'script', 'symlink', 'missing_plan', 'wrong_cwd', 'old_flag', 'equals_flag', 'double_flag'])
def test_learning_shell_rejects_other_commands_and_plans(tmp_path, case):
    assert harness(tmp_path, case=case).returncode != 0


@pytest.mark.parametrize('preflight', [False, True])
@pytest.mark.parametrize('dispatch', [False, True])
def test_learning_configure_keeps_new_tight_caps_and_never_old_main_caps(tmp_path, preflight, dispatch):
    result = harness(tmp_path, configure=True, dispatch=dispatch, case='preflight' if preflight else None)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'true,12000,11970,0,250,250,75,80,12288,80,125,8589934592'


@pytest.mark.parametrize('case', ['existing_output', 'output_symlink', 'output_parent_symlink', 'audit_class',
                                  'wrong_memory', 'wrong_swap', 'attended', 'cuda_producer', 'power_scope'])
def test_learning_configure_rejects_nonfresh_output_and_cross_class(tmp_path, case):
    assert harness(tmp_path, configure=True, case=case).returncode == 75


@pytest.mark.parametrize('field', [None, *[key for key in ENV if key.startswith(('GX1_NATIVE_', 'GX1_TRAINER_', 'GX1_POWER_'))], 'GX1_TRAINER_EXECUTION_MODE'])
def test_nested_learning_requires_exact_mode_plan_deadline_and_caps(field):
    values = {**ENV, 'GX1_TRAINER_EXECUTION_MODE': 'canonical'}
    if field: values[field] += 'wrong'
    script = 'NATIVE_PROVIDER_HLC_LEARNING=false\nNATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256=\nNATIVE_PROVIDER_HLC_LEARNING_RESUME=true\nNATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256=' + SHA + '\n'
    script += ''.join('export ' + key + '=' + shlex.quote(value) + '\n' for key, value in values.items())
    script += g._NATIVE_PROVIDER_HLC_LEARNING_RESUME_GUARD_SHELL_ADDITIONS[3][2]
    result = subprocess.run(['bash', '-c', script], text=True, capture_output=True)
    assert result.returncode == (75 if field else 0), result.stderr


def test_learning_source_bridge_preserves_all_prior_ast_and_shell():
    proof = g.native_provider_hlc_learning_resume_source_bridge(GUARDS, producer_commit=PARENT)
    assert proof['checked_guard_owners'] == 2 and proof['old_guard_routes_preserved'] is True


@pytest.mark.parametrize('name', list(GUARDS))
def test_learning_source_bridge_rejects_unrelated_source_delta(monkeypatch, name):
    path = Path(g.__file__).resolve().parents[2] / name
    original = Path.read_bytes
    def changed(candidate):
        raw = original(candidate)
        if candidate == path:
            raw += b'\nUNEXPECTED_OLD_OWNER_DELTA = 1\n' if name.endswith('.py') else b'\n# old shell delta\n'
        return raw
    monkeypatch.setattr(Path, 'read_bytes', changed)
    with pytest.raises(RuntimeError):
        g.native_provider_hlc_learning_resume_source_bridge(GUARDS, producer_commit=PARENT)


@pytest.mark.parametrize('case', ['producer', 'guard_hash', 'owner_set'])
def test_learning_source_bridge_rejects_other_generation_or_guard_set(case):
    guards = dict(GUARDS); producer = PARENT
    if case == 'producer': producer = '27fc84b9111d198ad0a6f6570d7e8efa7d0f1d49'
    elif case == 'guard_hash': guards['scripts/gx1_capped_run.sh'] = 'f' * 64
    else: guards['gx1/audit/unrelated.py'] = 'f' * 64
    with pytest.raises(RuntimeError):
        g.native_provider_hlc_learning_resume_source_bridge(guards, producer_commit=producer)


@pytest.mark.parametrize('case', ['extra', 'wrong_module', 'wrong_python', 'wrong_sha', 'lane', 'reordered',
                                  'script', 'equals_flag', 'double_flag'])
def test_resume_dispatch_rejects_alternate_cli_before_previous_main(tmp_path, case):
    result = harness(tmp_path, configure=True, dispatch=True, case=case)
    assert result.returncode == 75
    assert 'provider HLC learning resume CLI/plan is not exactly bound' in result.stderr
    assert 'UNEXPECTED_FALLTHROUGH' not in result.stdout


def test_resume_dispatch_is_before_previous_module_only_dispatch():
    source = (Path(g.__file__).resolve().parents[2] / 'scripts/gx1_capped_run.sh').read_text()
    entry = source.split('validate_target_command() {\n', 1)[1]
    resume = g._NATIVE_PROVIDER_HLC_LEARNING_RESUME_GUARD_SHELL_ADDITIONS[2][2]
    previous = g._NATIVE_PROVIDER_HLC_LEARNING_GUARD_SHELL_ADDITIONS[2][2]
    assert entry.startswith(resume + previous)
    assert '--resume-plan-sha256|--resume-plan-sha256=*' in resume
    assert '$NATIVE_PROVIDER_HLC_LEARNING_MODULE' in previous


@pytest.mark.parametrize('case', ['mode', 'sha'])
def test_nonresume_nested_parent_rejects_resume_mode_or_sha(case):
    script = 'NATIVE_PROVIDER_HLC_LEARNING_RESUME=false\nNATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256=\n'
    script += 'GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME=' + ('true' if case == 'mode' else 'false') + '\n'
    script += 'GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256=' + (SHA if case == 'sha' else '') + '\n'
    script += g._NATIVE_PROVIDER_HLC_LEARNING_RESUME_GUARD_SHELL_ADDITIONS[3][2]
    assert subprocess.run(['bash', '-c', script], capture_output=True).returncode == 75


def test_resume_mode_and_sha_are_exported_separately_from_previous_main():
    source = (Path(g.__file__).resolve().parents[2] / 'scripts/gx1_capped_run.sh').read_text()
    new = g._NATIVE_PROVIDER_HLC_LEARNING_RESUME_GUARD_SHELL_ADDITIONS[4][2]
    old = g._NATIVE_PROVIDER_HLC_LEARNING_GUARD_SHELL_ADDITIONS[4][2]
    assert source.count(new + old) == 1
    assert 'NATIVE_PROVIDER_HLC_LEARNING_RESUME=false\n' in g._NATIVE_PROVIDER_HLC_LEARNING_RESUME_GUARD_SHELL_ADDITIONS[0][2]



def test_resume_propagates_kernel_guard_rejection(monkeypatch, scope):
    def denied():
        raise RuntimeError('[SYNTHETIC_KERNEL_CONTAINMENT_DENIED]')
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', denied)
    with pytest.raises(RuntimeError, match='SYNTHETIC_KERNEL_CONTAINMENT_DENIED'):
        g.require_native_provider_hlc_learning_resume_execution(plan_sha256=SHA)
