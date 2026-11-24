"""Synthetic None-policy main/recovery containment and source closure; no payload/model."""
import copy
import hashlib
import json
import os
import shlex
import stat
import subprocess
from pathlib import Path

import pytest
from gx1.contracts import gx1_capped_execution_v1 as g

ACTUAL_CUDA_TRAINER_GUARD = g.require_guarded_cuda_trainer_execution

NAME = 'NATIVE_PROVIDER_HLC_ALL_TF_ONE_EPOCH_20261124_002'
MODULE = 'gx1.audit.native_provider_hlc_learning_v1'
PARENT = '1d83a588c21af611973afdfda6a7c960fd060436'
GUARDS = {
    'gx1/contracts/gx1_capped_execution_v1.py': 'e4fa7ef20eb8fe29ef84e93016e7823540c2c719556aa173e5fc9336b67ff963',
    'scripts/gx1_capped_run.sh': 'b23f7471ab0c62b883324319739652982394e7572df554c8ea1fc0c20f59bb45',
}
ENV = {'GX1_CAPPED_MEMORY_BYTES': '21474836480',
 'GX1_CAPPED_SWAP_BYTES': '536870912',
 'GX1_CAPPED_TASKS_MAX': '128',
 'OMP_NUM_THREADS': '8',
 'MKL_NUM_THREADS': '8',
 'OPENBLAS_NUM_THREADS': '8',
 'NUMEXPR_NUM_THREADS': '8',
 'VECLIB_MAXIMUM_THREADS': '8',
 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL': 'false',
 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256': '',
 'GX1_NATIVE_PROVIDER_HLC_INITIAL': 'false',
 'GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256': '',
 'GX1_NATIVE_PROVIDER_HLC_REPLAY': 'false',
 'GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256': '',
 'GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE': '',
 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY': 'false',
 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256': '',
 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE': '',
 'GX1_NATIVE_PROVIDER_HLC_LEARNING': 'false',
 'GX1_NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256': '',
 'GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME': 'false',
 'GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256': '',
 'GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE': 'false',
 'GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256': '',
 'GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE': 'false',
 'GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256': '',
 'GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX': 'false',
 'GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256': '',
 'GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN': 'false',
 'GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256': '',
 'GX1_NATIVE_FUTURE_STATE_THROUGHPUT': 'false',
 'GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256': '',
 'GX1_NATIVE_FUTURE_STATE_TRAINING': 'false',
 'GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256': '',
 'GX1_NATIVE_FUTURE_STATE_TRAINING_LANE': '',
 'GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION': 'false',
 'GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS': 'false',
 'GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256': '',
 'GX1_NATIVE_FUTURE_STATE_CUDA_LANE': '',
 'GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256': '',
 'GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE': '',
 'GX1_TRAINER_MAX_WALL_SECONDS': '12000',
 'GX1_TRAINER_MODEL_MAX_WALL_SECONDS': '11970',
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
 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING': 'true',
 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256': 'SHA',
 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE': 'MODE',
 'GX1_CAPPED_CLASS': 'trainer',
 'GX1_CUDA_PRODUCER_GUARD': 'false',
 'GX1_TRAINER_DEVICE': 'cuda',
 'GX1_TRAINER_EXECUTION_MODE': 'canonical',
 'BLIS_NUM_THREADS': '8',
 'ARROW_NUM_THREADS': '1',
 'POLARS_MAX_THREADS': '1'}
EXECUTION = {'class': 'trainer',
 'device': 'cuda:0',
 'memory': '20G',
 'swap': '512M',
 'numerical_threads': 8,
 'cpu_affinity': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
 'tasks_max': 128,
 'outer_seconds': 12000,
 'internal_seconds': 11970,
 'kill_after_seconds': 10,
 'cuda_peak_reserved_bytes_max': 12884901888,
 'GPU_core_stop_C': 75,
 'GPU_memory_stop_C': 80,
 'GPU_power_limit_W': 250,
 'GPU_draw_stop_W': 250,
 'CPU_stop_C': 80,
 'CPU_package_stop_W': 125,
 'host_min_available_bytes': 8589934592,
 'host_min_commit_headroom_bytes': 8589934592,
 'watchdog_interval_seconds': 1}


def _plan(mode):
    return {'schema_version': 'gx1_native_provider_hlc_all_tf_one_epoch_plan_v1' if mode == 'main'
            else 'gx1_native_provider_hlc_all_tf_remaining_epoch_plan_v1',
            'stage': NAME.lower() + ('_recovery' if mode == 'resume' else ''),
            'output_root': '/home/andre2/GX1_RUNS/' + NAME,
            **({'output_directory_prefix': 'RECOVERY_'} if mode == 'resume' else {}),
            'recipe': {'passes': 1}, 'backbone_gradient_policy': None,
            'model_contract': {'contract_sha256': '6d6ac22819e54c7b12aebcf4e1afe27b990d8a4d90d16500e352c7353d99cca0'},
            'execution': copy.deepcopy(EXECUTION)}


@pytest.fixture(params=['main', 'resume'])
def scope(monkeypatch, tmp_path, request):
    repo = Path(g.__file__).resolve().parents[2]
    base = Path('/home/andre2/GX1_RUNS') / NAME
    file = tmp_path / 'plan.json'; file.write_text('{}')
    file_stat, dir_stat = file.stat(), tmp_path.stat()
    read_bytes, lstat, path_stat, resolve = Path.read_bytes, Path.lstat, Path.stat, Path.resolve
    exists, is_symlink = Path.exists, Path.is_symlink
    state = {'mode': request.param, 'repo': repo, 'base': base}
    monkeypatch.setattr(Path, 'read_bytes', lambda p: state['raw'] if p == state.get('plan') else read_bytes(p))
    monkeypatch.setattr(Path, 'lstat', lambda p: file_stat if p == state.get('plan') else lstat(p))
    monkeypatch.setattr(Path, 'stat', lambda p, *a, **kw: dir_stat if p == state.get('output', base).parent else path_stat(p, *a, **kw))
    monkeypatch.setattr(Path, 'resolve', lambda p, strict=False: p if p in (state.get('plan'), state.get('output', base).parent) else resolve(p, strict=strict))
    monkeypatch.setattr(Path, 'exists', lambda p: False if p == state.get('output') else exists(p))
    monkeypatch.setattr(Path, 'is_symlink', lambda p: False if p == state.get('output') else is_symlink(p))
    monkeypatch.setattr(Path, 'cwd', lambda: repo)
    # Even a stronger lower-level proof cannot grant optimizer/policy authority.
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: {
        'kernel_scope': True, 'training_authority_granted': True, 'optimizer_authority_granted': True})
    def bind(plan, raw=None):
        raw = json.dumps(plan, sort_keys=True).encode() if raw is None else raw
        sha = hashlib.sha256(raw).hexdigest()
        mode = state['mode']
        state.update(data=plan, raw=raw, sha=sha,
                     plan=repo / 'configs/research' / (NAME + '.json') if mode == 'main'
                     else base / 'RECOVERY_PLANS' / (sha + '.json'),
                     output=base if mode == 'main' else base / ('RECOVERY_' + sha))
        command = [str(repo / '.venv/bin/python'), '-m', MODULE,
                   '--all-tf-learning-plan-sha256' if mode == 'main' else '--all-tf-learning-resume-plan-sha256', sha]
        state['command'] = command
        monkeypatch.setattr(g.sys, 'orig_argv', command.copy())
        monkeypatch.setattr(g.sys, 'executable', command[0])
        for name, value in ENV.items():
            monkeypatch.setenv(name, sha if value == 'SHA' else mode if value == 'MODE' else value)
    state['bind'] = bind
    bind(_plan(state['mode']))
    return state


def _proof(scope):
    return g.require_native_provider_hlc_all_tf_learning_execution(plan_sha256=scope['sha'], mode=scope['mode'])


@pytest.mark.parametrize('preflight', [False, True])
def test_exact_main_or_fresh_recovery_only_proves_resources(monkeypatch, scope, preflight):
    if preflight:
        monkeypatch.setattr(g.sys, 'orig_argv', scope['command'] + ['--preflight'])
    proof = _proof(scope)
    assert proof['kernel_scope'] is proof['resource_proof_only'] is True
    assert proof['training_authority_granted'] is proof['optimizer_authority_granted'] is False
    assert (proof['outer_seconds'], proof['internal_seconds'], proof['kill_after_seconds']) == (12000, 11970, 10)
    assert proof['numerical_threads'] == 8 and proof['cpu_affinity'] == list(range(19))
    assert proof['mode'] == scope['mode']
    assert proof['plan'] == {'path': str(scope['plan']), 'sha256': scope['sha']}
    assert proof['output_root'] == str(scope['output'])
    assert not scope['output'].exists()


@pytest.mark.parametrize('wrong_actual_affinity', [False, True])
def test_consumed_affinity_hint_uses_actual_process_affinity(monkeypatch, scope, wrong_actual_affinity):
    # SCOPE_GUARD consumes/unsets GX1_CPU_AFFINITY before taskset launches Python.
    if wrong_actual_affinity:
        monkeypatch.setenv('GX1_CPU_AFFINITY', '0-18')
    else:
        monkeypatch.delenv('GX1_CPU_AFFINITY', raising=False)
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', ACTUAL_CUDA_TRAINER_GUARD)
    monkeypatch.setattr(g.os, 'sched_getaffinity', lambda pid: set(range(18 if wrong_actual_affinity else 19)))
    proofs = []
    def actual_scope(**kwargs):
        proofs.append(kwargs)
        assert kwargs['max_memory_bytes'] == 20 * 1024**3 and kwargs['max_pids'] == 128
        return {'kernel_scope': True}
    monkeypatch.setattr(g, '_require_capped_cgroup_limits', actual_scope)
    monkeypatch.setattr(g, 'require_capped_lock_ancestry', lambda: {'canonical_lock': True})
    if wrong_actual_affinity:
        with pytest.raises(RuntimeError, match='GX1_CUDA_TRAINER_AFFINITY_INVALID'):
            _proof(scope)
        assert not proofs
    else:
        proof = _proof(scope)
        assert proof['kernel_scope'] is True and proof['lock'] == {'canonical_lock': True}
        assert len(proofs) == 1
        assert proof['training_authority_granted'] is proof['optimizer_authority_granted'] is False


@pytest.mark.parametrize('field', list(ENV))
def test_each_changed_flag_lane_or_resource_fails_before_kernel(monkeypatch, scope, field):
    monkeypatch.setenv(field, os.environ[field] + 'wrong')
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: pytest.fail('invalid scope reached kernel'))
    with pytest.raises(RuntimeError, match='ENV_INVALID'):
        _proof(scope)


@pytest.mark.parametrize('field', ['GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_LANE',
                                 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_EXTRA'])
def test_unknown_new_route_environment_is_rejected_even_empty(monkeypatch, scope, field):
    monkeypatch.setenv(field, '')
    with pytest.raises(RuntimeError, match='ENV_POLLUTION'):
        _proof(scope)


@pytest.mark.parametrize('bad', ['A' * 64, 'f' * 63, None, True])
def test_invalid_sha_is_rejected_before_plan_or_kernel(bad):
    with pytest.raises(RuntimeError, match='PLAN_SHA256_INVALID'):
        g.require_native_provider_hlc_all_tf_learning_execution(plan_sha256=bad)


@pytest.mark.parametrize('bad', ['cold', 'main ', None, True])
def test_only_exact_main_or_resume_mode_is_accepted(bad):
    with pytest.raises(RuntimeError, match='MODE_INVALID'):
        g.require_native_provider_hlc_all_tf_learning_execution(plan_sha256='f' * 64, mode=bad)


@pytest.mark.parametrize('case', ['wrong_python', 'script', 'lane', 'extra', 'reordered', 'bad_executable',
                                  'wrong_cwd', 'legacy_flag', 'equals_flag', 'mixed_flag', 'cross_mode'])
def test_noncanonical_or_mixed_cli_is_rejected(monkeypatch, scope, case):
    argv = scope['command'].copy()
    if case == 'wrong_python': argv[0] = '/usr/bin/python3'
    elif case == 'script': argv[1:3] = [str(scope['repo'] / 'gx1/audit/native_provider_hlc_learning_v1.py')]
    elif case == 'lane': argv += ['--lane', 'main']
    elif case == 'extra': argv += ['--preflight', '--preflight']
    elif case == 'reordered': argv[3:5] = [scope['sha'], argv[3]]
    elif case == 'bad_executable': monkeypatch.setattr(g.sys, 'executable', '/usr/bin/python3')
    elif case == 'wrong_cwd': monkeypatch.setattr(Path, 'cwd', lambda: scope['repo'].parent)
    elif case == 'legacy_flag': argv[3] = '--plan-sha256'
    elif case == 'equals_flag': argv[3:5] = [argv[3] + '=' + scope['sha']]
    elif case == 'mixed_flag': argv += ['--resume-plan-sha256', scope['sha']]
    elif case == 'cross_mode': argv[3] = '--all-tf-learning-resume-plan-sha256' if scope['mode'] == 'main' else '--all-tf-learning-plan-sha256'
    monkeypatch.setattr(g.sys, 'orig_argv', argv)
    with pytest.raises(RuntimeError, match='(CLI|CWD)_INVALID'):
        _proof(scope)


@pytest.mark.parametrize('case', ['schema', 'stage', 'output_root', 'prefix', 'top_policy', 'missing_policy',
                                  'recipe_policy', 'passes', 'bool_passes', 'contract_policy', 'contract_sha', 'execution'])
def test_plan_hash_cannot_authorize_another_contract_or_execution(scope, case):
    plan = copy.deepcopy(scope['data'])
    if case == 'schema': plan['schema_version'] = 'gx1_native_provider_hlc_one_epoch_plan_v1'
    elif case == 'stage': plan['stage'] += '_other'
    elif case == 'output_root': plan['output_root'] += '_other'
    elif case == 'prefix':
        if scope['mode'] == 'main': return
        plan['output_directory_prefix'] = 'OTHER_'
    elif case == 'top_policy': plan['backbone_gradient_policy'] = 'M5_primary'
    elif case == 'missing_policy': del plan['backbone_gradient_policy']
    elif case == 'recipe_policy': plan['recipe']['backbone_gradient_policy'] = 'M5_primary'
    elif case == 'passes': plan['recipe']['passes'] = 2
    elif case == 'bool_passes': plan['recipe']['passes'] = True
    elif case == 'contract_policy': plan['model_contract']['backbone_gradient_policy'] = None
    elif case == 'contract_sha': plan['model_contract']['contract_sha256'] = 'f' * 64
    elif case == 'execution': plan['execution']['GPU_core_stop_C'] = 99
    scope['bind'](plan)
    with pytest.raises(RuntimeError, match='(PLAN_IDENTITY|PLAN_ROOT|EXPLICIT_NONE_ONE_EPOCH_CONTRACT|EXACT_EXECUTION)'):
        _proof(scope)


@pytest.mark.parametrize('case', ['hash', 'missing', 'plan_symlink', 'plan_parent_symlink', 'existing_output',
                                  'output_symlink', 'output_parent_symlink', 'missing_parent', 'bad_json'])
def test_changed_metadata_or_nonfresh_root_fails_before_kernel(monkeypatch, scope, case):
    lstat, read_bytes, resolve, exists, is_symlink, path_stat = Path.lstat, Path.read_bytes, Path.resolve, Path.exists, Path.is_symlink, Path.stat
    if case == 'hash': monkeypatch.setattr(Path, 'read_bytes', lambda p: scope['raw'] + b' ' if p == scope['plan'] else read_bytes(p))
    elif case == 'missing':
        def missing(p):
            if p == scope['plan']: raise FileNotFoundError(str(p))
            return lstat(p)
        monkeypatch.setattr(Path, 'lstat', missing)
    elif case == 'plan_symlink': monkeypatch.setattr(Path, 'lstat', lambda p: type('Link', (), {'st_mode': stat.S_IFLNK})() if p == scope['plan'] else lstat(p))
    elif case in ('plan_parent_symlink', 'output_parent_symlink'):
        changed = scope['plan'] if case == 'plan_parent_symlink' else scope['output'].parent
        monkeypatch.setattr(Path, 'resolve', lambda p, strict=False: Path('/different/canonical/path') if p == changed else resolve(p, strict=strict))
    elif case == 'existing_output': monkeypatch.setattr(Path, 'exists', lambda p: True if p == scope['output'] else exists(p))
    elif case == 'output_symlink': monkeypatch.setattr(Path, 'is_symlink', lambda p: True if p == scope['output'] else is_symlink(p))
    elif case == 'missing_parent':
        def missing_parent(p):
            if p == scope['output'].parent: raise FileNotFoundError(str(p))
            return path_stat(p)
        monkeypatch.setattr(Path, 'stat', missing_parent)
    elif case == 'bad_json': scope['bind']({}, raw=b'not JSON')
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: pytest.fail('invalid paths reached kernel'))
    with pytest.raises(RuntimeError, match='(PLAN_SHA|CANONICAL_PLAN|FRESH_ROOT|PLAN_OUTPUT_INVALID)'):
        _proof(scope)


def _shell(tmp_path, mode, case=None, dispatch=True):
    repo = tmp_path / 'repo'; repo.mkdir()
    base = repo / 'runs' / NAME; base.parent.mkdir()
    raw = b'{"synthetic_guard_fixture":true}\n'; sha = hashlib.sha256(raw).hexdigest()
    if mode == 'main':
        config = repo / 'configs/research'; config.mkdir(parents=True)
        plan = config / (NAME + '.json'); output = base
    else:
        config = base / 'RECOVERY_PLANS'; config.mkdir(parents=True)
        plan = config / (sha + '.json'); output = base / ('RECOVERY_' + sha)
    plan.write_bytes(raw)
    argv = [str(repo / '.venv/bin/python'), '-m', MODULE,
            '--all-tf-learning-plan-sha256' if mode == 'main' else '--all-tf-learning-resume-plan-sha256', sha]
    if case == 'preflight': argv += ['--preflight']
    elif case == 'extra': argv += ['--unsafe']
    elif case == 'wrong_module': argv[2] = 'gx1.audit.native_provider_hlc_replay_v1'
    elif case == 'wrong_python': argv[0] = '/usr/bin/python3'
    elif case == 'wrong_sha': argv[4] = 'f' * 64
    elif case == 'lane': argv += ['--lane', 'main']
    elif case == 'reordered': argv[3:5] = [sha, argv[3]]
    elif case == 'equals': argv[3:5] = [argv[3] + '=' + sha]
    elif case == 'mixed': argv += ['--plan-sha256', sha]
    elif case == 'script': argv[1:3] = [str(repo / 'gx1/audit/native_provider_hlc_learning_v1.py')]
    elif case == 'symlink':
        original = plan.with_suffix('.original'); plan.rename(original); plan.symlink_to(original)
    elif case == 'missing_plan': plan.unlink()
    elif case == 'existing_output': output.mkdir()
    elif case == 'output_symlink': output.symlink_to(repo / 'absent')
    state, functions, route = [item[2] for item in g._NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_GUARD_SHELL_ADDITIONS[:3]]
    functions = functions.replace('/home/andre2/GX1_RUNS/' + NAME, str(base))
    values = {'REPO_ROOT': str(repo), 'CANONICAL_TRAINER_PYTHON': argv[0] if case != 'wrong_python' else str(repo / '.venv/bin/python'),
              'NATIVE_PROVIDER_HLC_LEARNING_MODULE': MODULE,
              'JOB_CLASS': 'trainer', 'ATTENDED_SMOKE': 'false', 'CUDA_PRODUCER_GUARD': 'false',
              'MEM': '20G', 'SWAP': '512M', 'POWER_BENCHMARK_SCOPE_JSON': '', 'POWER_BENCHMARK_SCOPE_SHA256': ''}
    if case == 'audit_class': values['JOB_CLASS'] = 'audit'
    elif case == 'wrong_memory': values['MEM'] = '4G'
    elif case == 'wrong_swap': values['SWAP'] = '1G'
    elif case == 'attended': values['ATTENDED_SMOKE'] = 'true'
    elif case == 'cuda_producer': values['CUDA_PRODUCER_GUARD'] = 'true'
    elif case == 'power_scope': values['POWER_BENCHMARK_SCOPE_SHA256'] = 'f' * 64
    script = state + functions + ''.join('\n' + k + '=' + shlex.quote(v) for k, v in values.items())
    if dispatch:
        script += '\nvalidate_target_command() {\n' + route + 'echo UNEXPECTED_FALLTHROUGH; return 99; }\nvalidate_target_command '
    else:
        script += '\nconfigure_native_provider_hlc_all_tf_learning '
    script += ' '.join(map(shlex.quote, argv)) + '\n'
    script += 'printf "%s,%s,%s,%s,%s,%s,%s,%s\\n" "$NATIVE_PROVIDER_HLC_ALL_TF_LEARNING" "$NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE" "$TRAINER_MAX_WALL_SECONDS" "$TRAINER_MODEL_MAX_WALL_SECONDS" "$TRAINER_GPU_MAX_CORE_TEMP_C" "$TRAINER_GPU_MAX_MEMORY_USED_MIB" "$TRAINER_CPU_MAX_PACKAGE_POWER_W" "$TRAINER_OUT_BUNDLE_DIR"\n'
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GX1_CAPPED_', 'GX1_NATIVE_', 'GX1_TRAINER_', 'GX1_POWER_'))}
    pollution = {'old_flag': ('GX1_NATIVE_PROVIDER_HLC_LEARNING', 'true'),
                 'old_lane': ('GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE', 'cold'),
                 'old_sha': ('GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256', sha),
                 'new_flag': ('GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING', 'true'),
                 'new_lane': ('GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_LANE', ''),
                 'partial_nested': ('GX1_CAPPED_MEMORY_BYTES', '21474836480')}
    if case in pollution: env[pollution[case][0]] = pollution[case][1]
    result = subprocess.run(['bash', '-c', script], cwd=tmp_path if case == 'wrong_cwd' else repo,
                            text=True, capture_output=True, env=env)
    assert not (output / 'CLAIM.json').exists()
    return result, output


@pytest.mark.parametrize('mode', ['main', 'resume'])
@pytest.mark.parametrize('preflight', [False, True])
@pytest.mark.parametrize('dispatch', [False, True])
def test_shell_routes_exact_main_or_recovery_without_creating_output(tmp_path, mode, preflight, dispatch):
    result, output = _shell(tmp_path, mode, 'preflight' if preflight else None, dispatch)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'true,' + mode + ',12000,11970,75,12288,125,' + str(output)
    assert not output.exists()


@pytest.mark.parametrize('mode', ['main', 'resume'])
@pytest.mark.parametrize('case', ['extra', 'wrong_module', 'wrong_python', 'wrong_sha', 'lane', 'reordered',
                                  'equals', 'mixed', 'script', 'symlink', 'missing_plan', 'wrong_cwd',
                                  'existing_output', 'output_symlink', 'audit_class', 'wrong_memory', 'wrong_swap',
                                  'attended', 'cuda_producer', 'power_scope', 'old_flag', 'old_lane', 'old_sha',
                                  'new_flag', 'new_lane', 'partial_nested'])
def test_shell_rejects_other_cli_paths_scopes_and_ambient_pollution(tmp_path, mode, case):
    result, _ = _shell(tmp_path, mode, case)
    assert result.returncode == 75, (result.stdout, result.stderr)
    assert 'UNEXPECTED_FALLTHROUGH' not in result.stdout


@pytest.mark.parametrize('mode', ['main', 'resume'])
@pytest.mark.parametrize('field', [None, *ENV])
def test_nested_route_cannot_change_any_mode_lane_or_fixed_resource(mode, field):
    sha = 'e' * 64
    values = {name: sha if value == 'SHA' else mode if value == 'MODE' else value for name, value in ENV.items()}
    if field: values[field] += 'wrong'
    script = 'NATIVE_PROVIDER_HLC_ALL_TF_LEARNING=true\nNATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256=' + sha
    script += '\nNATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE=' + mode + '\n'
    script += ''.join('export ' + key + '=' + shlex.quote(value) + '\n' for key, value in values.items())
    script += g._NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_GUARD_SHELL_ADDITIONS[3][2]
    result = subprocess.run(['bash', '-c', script], text=True, capture_output=True)
    assert result.returncode == (75 if field else 0), result.stderr


# BEGIN GX1 HISTORICAL ALL TF GUARD SOURCE FIXTURE 20261124 001
HISTORICAL_GUARD_SOURCE = 'f111b32ac032f1fe3b893e37ec37ea736b67f942'


@pytest.fixture
def historical_guard_source(monkeypatch):
    """Run the original bridge against its authenticated pre-endpoint generation."""
    import ast
    repo = Path(g.__file__).resolve().parents[2]
    snapshots = {repo / name: subprocess.check_output(
        ['git', 'show', HISTORICAL_GUARD_SOURCE + ':' + name], cwd=repo) for name in GUARDS}
    tree = ast.parse(snapshots[repo / 'gx1/contracts/gx1_capped_execution_v1.py'])
    constant = '_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_GUARD_SHELL_ADDITIONS'
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == constant)
    monkeypatch.setattr(g, constant, ast.literal_eval(node.value))
    original_read_bytes = Path.read_bytes
    monkeypatch.setattr(Path, 'read_bytes', lambda path: snapshots[path] if path in snapshots else original_read_bytes(path))
    return snapshots


def test_current_source_bridge_rejects_later_unscoped_endpoint_generation():
    with pytest.raises(RuntimeError, match='SOURCE_BRIDGE_EXACT_SHELL_INSERTION'):
        g.native_provider_hlc_all_tf_learning_source_bridge(GUARDS)
# END GX1 HISTORICAL ALL TF GUARD SOURCE FIXTURE 20261124 001

def test_historical_source_bridge_preserves_every_old_owner_byte_and_ast(historical_guard_source):
    proof = g.native_provider_hlc_all_tf_learning_source_bridge(GUARDS)
    assert proof['producer_commit'] == PARENT
    assert proof['checked_guard_owners'] == 2 and proof['old_guard_routes_preserved'] is True
    assert all(item['normalized_sha256'] == item['original_sha256'] for item in proof['owners'].values())


@pytest.mark.parametrize('case', ['old_python', 'old_shell', 'extra_function', 'extra_assignment',
                                  'changed_route', 'duplicate_fragment', 'missing_marker'])
def test_source_bridge_rejects_unscoped_source_or_route_authority(monkeypatch, historical_guard_source, case):
    repo = Path(g.__file__).resolve().parents[2]
    original = Path.read_bytes
    def changed(path):
        raw = original(path)
        if path == repo / 'gx1/contracts/gx1_capped_execution_v1.py':
            if case == 'old_python': raw += b'\nUNEXPECTED_OLD_OWNER_DELTA = 1\n'
            elif case in ('extra_function', 'extra_assignment'):
                addition = b'def unexpected_authority():\n    return True\n' if case == 'extra_function' else b'UNEXPECTED_AUTHORITY = True\n'
                raw = raw.replace(b'# END GX1 ALL TF LEARNING GUARD ROUTE 20261124 001\n', addition + b'# END GX1 ALL TF LEARNING GUARD ROUTE 20261124 001\n')
            elif case == 'missing_marker': raw = raw.replace(b'# BEGIN GX1 ALL TF LEARNING GUARD ROUTE 20261124 001\n', b'# removed marker\n')
        elif path == repo / 'scripts/gx1_capped_run.sh':
            if case == 'old_shell': raw += b'\n# old shell delta\n'
            elif case == 'changed_route': raw = raw.replace(b'NATIVE_PROVIDER_HLC_ALL_TF_LEARNING=false\n', b'NATIVE_PROVIDER_HLC_ALL_TF_LEARNING=true\n', 1)
            elif case == 'duplicate_fragment': raw += g._NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_GUARD_SHELL_ADDITIONS[0][2].encode()
        return raw
    monkeypatch.setattr(Path, 'read_bytes', changed)
    with pytest.raises(RuntimeError, match='SOURCE_BRIDGE'):
        g.native_provider_hlc_all_tf_learning_source_bridge(GUARDS)


@pytest.mark.parametrize('case', ['producer', 'hash', 'owner_set'])
def test_source_bridge_accepts_only_the_exact_two_parent_owners(case):
    guards = dict(GUARDS); producer = PARENT
    if case == 'producer': producer = 'f' * 40
    elif case == 'hash': guards['scripts/gx1_capped_run.sh'] = 'f' * 64
    else: guards['gx1/audit/unrelated.py'] = 'f' * 64
    with pytest.raises(RuntimeError, match='SOURCE_BRIDGE'):
        g.native_provider_hlc_all_tf_learning_source_bridge(guards, producer_commit=producer)
