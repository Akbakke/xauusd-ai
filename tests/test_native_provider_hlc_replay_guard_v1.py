"""Exact technical replay guard metadata; no numerical/model/data job."""
import hashlib
import shlex
import subprocess
from pathlib import Path
import pytest
from gx1.contracts import gx1_capped_execution_v1 as g

SHA = 'a' * 64
MODULE = 'gx1.audit.native_provider_hlc_replay_v1'
BASENAME = 'NATIVE_PROVIDER_HLC_COLD_REPLAY_20261123_001'
PARENT = '27fc84b9111d198ad0a6f6570d7e8efa7d0f1d49'
GUARDS = {'gx1/contracts/gx1_capped_execution_v1.py': 'dc34de2817016bbba33c27c4b8a5bcda8ea4f77fb978daa69a81aee3b4c5043b',
          'scripts/gx1_capped_run.sh': '791527f0c502867d2b5a78cc2cd976de70e3f5d02eee67407aa4d2a74d062e05'}
ENV = {'GX1_CAPPED_MEMORY_BYTES': '21474836480', 'GX1_CAPPED_SWAP_BYTES': '536870912', 'GX1_CAPPED_TASKS_MAX': '128', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8', 'NUMEXPR_NUM_THREADS': '8', 'VECLIB_MAXIMUM_THREADS': '8', 'GX1_NATIVE_PROVIDER_HLC_INITIAL': 'false', 'GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256': '', 'GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE': 'false', 'GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256': '', 'GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX': 'false', 'GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256': '', 'GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN': 'false', 'GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256': '', 'GX1_NATIVE_FUTURE_STATE_THROUGHPUT': 'false', 'GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256': '', 'GX1_NATIVE_FUTURE_STATE_TRAINING': 'false', 'GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256': '', 'GX1_NATIVE_FUTURE_STATE_TRAINING_LANE': '', 'GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION': 'false', 'GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS': 'false', 'GX1_TRAINER_EXECUTION_MODE': 'canonical', 'GX1_TRAINER_MAX_WALL_SECONDS': '1800', 'GX1_TRAINER_MODEL_MAX_WALL_SECONDS': '1770', 'GX1_TRAINER_ATTENDED_STAGE_REQUIRED': 'false', 'GX1_TRAINER_GPU_INDEX': '0', 'GX1_TRAINER_GPU_MAX_POWER_LIMIT_W': '250', 'GX1_TRAINER_GPU_MAX_POWER_DRAW_W': '250', 'GX1_TRAINER_GPU_MAX_CORE_TEMP_C': '75', 'GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C': '80', 'GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB': '12288', 'GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS': '1', 'GX1_TRAINER_CPU_MAX_TEMP_C': '80', 'GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W': '125', 'GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES': '8589934592', 'GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES': '8589934592', 'GX1_POWER_BENCHMARK_SCOPE_JSON': '', 'GX1_POWER_BENCHMARK_SCOPE_SHA256': '', 'GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE': 'false', 'GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256': '', 'GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256': '', 'GX1_NATIVE_FUTURE_STATE_CUDA_LANE': '', 'GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256': '', 'GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE': '', 'GX1_NATIVE_PROVIDER_HLC_REPLAY': 'true', 'GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE': 'warm'}


def environment(monkeypatch, lane='warm'):
    for name, value in ENV.items():
        monkeypatch.setenv(name, lane if name == 'GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE' else value)


@pytest.mark.parametrize('lane', ['warm', 'cold'])
def test_replay_scope_contains_resources_but_grants_no_optimizer_admission(monkeypatch, lane):
    environment(monkeypatch, lane)
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: {'kernel_scope': True})
    proof = g.require_native_provider_hlc_replay_execution(plan_sha256=SHA, lane=lane)
    assert proof['kernel_scope'] and proof['kind'] == 'provider_hlc_replay' and proof['lane'] == lane
    assert proof['resource_proof_only'] is True
    assert proof['training_authority_granted'] is proof['optimizer_authority_granted'] is False
    assert (proof['outer_seconds'], proof['internal_seconds'], proof['kill_after_seconds']) == (1800, 1770, 10)
    assert proof['cpu_affinity'] == list(range(19)) and proof['numerical_threads'] == 8
    assert proof['plan'] == {'path': str(Path(g.__file__).resolve().parents[2] / 'configs/research' / (BASENAME + '.json')), 'sha256': SHA}
    assert proof['output_root'] == '/home/andre2/GX1_RUNS/' + BASENAME + '_' + lane


@pytest.mark.parametrize('field', [
    'GX1_NATIVE_PROVIDER_HLC_REPLAY', 'GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256',
    'GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE', 'GX1_NATIVE_PROVIDER_HLC_INITIAL',
    'GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256', 'GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE',
    'GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256', 'GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE',
    'GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256', 'GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX',
    'GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256', 'GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN',
    'GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256', 'GX1_NATIVE_FUTURE_STATE_THROUGHPUT',
    'GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256', 'GX1_NATIVE_FUTURE_STATE_TRAINING',
    'GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256', 'GX1_NATIVE_FUTURE_STATE_TRAINING_LANE',
    'GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION', 'GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256',
    'GX1_NATIVE_FUTURE_STATE_CUDA_LANE', 'GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS',
    'GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256', 'GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE',
    'GX1_CAPPED_MEMORY_BYTES', 'GX1_CAPPED_SWAP_BYTES', 'GX1_CAPPED_TASKS_MAX', 'OMP_NUM_THREADS',
    'GX1_TRAINER_MAX_WALL_SECONDS', 'GX1_TRAINER_MODEL_MAX_WALL_SECONDS',
    'GX1_TRAINER_GPU_MAX_POWER_LIMIT_W', 'GX1_TRAINER_GPU_MAX_POWER_DRAW_W',
    'GX1_TRAINER_GPU_MAX_CORE_TEMP_C', 'GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C',
    'GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB', 'GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS',
    'GX1_TRAINER_CPU_MAX_TEMP_C', 'GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W',
    'GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES', 'GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES',
])
def test_replay_rejects_cross_mode_and_changed_containment_before_kernel(monkeypatch, field):
    environment(monkeypatch)
    monkeypatch.setenv(field, ENV[field] + 'wrong')
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: pytest.fail('invalid env reached kernel proof'))
    with pytest.raises(RuntimeError, match='PROVIDER_HLC_REPLAY_ENV_INVALID'):
        g.require_native_provider_hlc_replay_execution(plan_sha256=SHA, lane='warm')


@pytest.mark.parametrize('bad', ['A' * 64, 'f' * 63, None, True])
def test_replay_invalid_sha_stops_before_scope(monkeypatch, bad):
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: pytest.fail('invalid SHA reached scope'))
    with pytest.raises(RuntimeError, match='PLAN_SHA256_INVALID'):
        g.require_native_provider_hlc_replay_execution(plan_sha256=bad, lane='warm')


@pytest.mark.parametrize('bad', ['main', 'joint', 'WARM', '', None, True])
def test_replay_invalid_lane_stops_before_scope(monkeypatch, bad):
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: pytest.fail('invalid lane reached scope'))
    with pytest.raises(RuntimeError, match='LANE_INVALID'):
        g.require_native_provider_hlc_replay_execution(plan_sha256=SHA, lane=bad)


def harness(tmp_path, *, lane='warm', case=None, configure=False):
    repo = tmp_path / 'repo'; repo.mkdir()
    config = repo / 'configs/research'; config.mkdir(parents=True)
    plan = config / (BASENAME + '.json'); plan.write_text('{"fixture":true}\n')
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    argv = [str(repo / '.venv/bin/python'), '-m', MODULE, '--plan-sha256', digest, '--lane', lane]
    if case == 'preflight': argv += ['--preflight']
    elif case == 'extra': argv += ['--unsafe']
    elif case == 'wrong_module': argv[2] = 'gx1.audit.native_future_state_training_v1'
    elif case == 'wrong_python': argv[0] = '/usr/bin/python3'
    elif case == 'wrong_sha': argv[4] = 'f' * 64
    elif case == 'wrong_lane': argv[6] = 'main'
    elif case == 'missing_lane': argv = argv[:5]
    elif case == 'reordered': argv = argv[:3] + argv[5:7] + argv[3:5]
    elif case == 'symlink':
        target = plan.with_suffix('.original'); plan.rename(target); plan.symlink_to(target)
    elif case == 'missing_plan': plan.unlink()
    elif case == 'existing_output': (repo / ('replay-output_' + lane)).mkdir()
    elif case == 'output_symlink': (repo / ('replay-output_' + lane)).symlink_to(repo / 'absent')
    state, functions = (g._NATIVE_PROVIDER_HLC_REPLAY_GUARD_SHELL_ADDITIONS[i][2] for i in (0, 1))
    functions = functions.replace('/home/andre2/GX1_RUNS/' + BASENAME, str(repo / 'replay-output'))
    script = state + functions + '\nREPO_ROOT=' + shlex.quote(str(repo))
    script += '\nCANONICAL_TRAINER_PYTHON=' + shlex.quote(str(repo / '.venv/bin/python'))
    if configure:
        values = {'JOB_CLASS': 'trainer', 'ATTENDED_SMOKE': 'false', 'CUDA_PRODUCER_GUARD': 'false',
                  'MEM': '20G', 'SWAP': '512M', 'POWER_BENCHMARK_SCOPE_JSON': '', 'POWER_BENCHMARK_SCOPE_SHA256': ''}
        if case == 'audit_class': values['JOB_CLASS'] = 'audit'
        elif case == 'wrong_memory': values['MEM'] = '4G'
        elif case == 'attended': values['ATTENDED_SMOKE'] = 'true'
        elif case == 'power_scope': values['POWER_BENCHMARK_SCOPE_SHA256'] = 'f' * 64
        script += ''.join('\n' + key + '=' + shlex.quote(value) for key, value in values.items())
        script += '\nconfigure_native_provider_hlc_replay ' + ' '.join(map(shlex.quote, argv))
        script += '\nprintf "%s,%s,%s,%s,%s,%s,%s,%s\n" "$NATIVE_PROVIDER_HLC_REPLAY" "$NATIVE_PROVIDER_HLC_REPLAY_LANE" "$TRAINER_MAX_WALL_SECONDS" "$TRAINER_MODEL_MAX_WALL_SECONDS" "$TRAINER_GPU_MAX_POWER_DRAW_W" "$TRAINER_CPU_MAX_TEMP_C" "$TRAINER_CPU_MAX_PACKAGE_POWER_W" "$TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES"\n'
    else:
        script += '\nmatches_bound_native_provider_hlc_replay ' + ' '.join(map(shlex.quote, argv)) + '\n'
    return subprocess.run(['bash', '-c', script], cwd=repo, text=True, capture_output=True)


@pytest.mark.parametrize('lane', ['warm', 'cold'])
@pytest.mark.parametrize('case', [None, 'preflight'])
def test_replay_exact_shell_cli_and_optional_preflight(tmp_path, lane, case):
    assert harness(tmp_path, lane=lane, case=case).returncode == 0


@pytest.mark.parametrize('case', ['extra', 'wrong_module', 'wrong_python', 'wrong_sha', 'wrong_lane',
                                  'missing_lane', 'reordered', 'symlink', 'missing_plan'])
def test_replay_shell_rejects_wrong_command_plan_and_lane(tmp_path, case):
    assert harness(tmp_path, case=case).returncode != 0


@pytest.mark.parametrize('lane', ['warm', 'cold'])
def test_replay_configure_exact_safety_and_sibling_root(tmp_path, lane):
    result = harness(tmp_path, configure=True, lane=lane)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'true,' + lane + ',1800,1770,250,80,125,8589934592'


@pytest.mark.parametrize('case', ['existing_output', 'output_symlink', 'audit_class', 'wrong_memory', 'attended', 'power_scope'])
def test_replay_configure_rejects_nonfresh_output_and_other_resource_routes(tmp_path, case):
    assert harness(tmp_path, configure=True, case=case).returncode == 75


@pytest.mark.parametrize('lane', ['warm', 'cold'])
@pytest.mark.parametrize('field', [None, 'GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE', 'GX1_NATIVE_PROVIDER_HLC_INITIAL',
                                  'GX1_NATIVE_FUTURE_STATE_TRAINING', 'GX1_TRAINER_MODEL_MAX_WALL_SECONDS',
                                  'GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W'])
def test_nested_replay_requires_same_lane_mode_deadline_and_host_caps(lane, field):
    values = {**ENV, 'GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE': lane}
    if field: values[field] += 'wrong'
    script = 'NATIVE_PROVIDER_HLC_REPLAY=true\nNATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256=' + SHA
    script += '\nNATIVE_PROVIDER_HLC_REPLAY_LANE=' + lane + '\n'
    script += ''.join('export ' + key + '=' + shlex.quote(value) + '\n' for key, value in values.items())
    script += g._NATIVE_PROVIDER_HLC_REPLAY_GUARD_SHELL_ADDITIONS[3][2]
    result = subprocess.run(['bash', '-c', script], text=True, capture_output=True)
    assert result.returncode == (75 if field else 0), result.stderr


def test_replay_source_bridge_preserves_old_python_and_shell():
    proof = g.native_provider_hlc_replay_source_bridge(GUARDS, producer_commit=PARENT)
    assert proof['checked_guard_owners'] == 2 and proof['old_guard_routes_preserved'] is True


@pytest.mark.parametrize('bad', ['producer', 'hash', 'extra'])
def test_replay_source_bridge_rejects_unbound_anchor(bad):
    owners = dict(GUARDS); producer = PARENT
    if bad == 'producer': producer = 'a' * 40
    elif bad == 'hash': owners['scripts/gx1_capped_run.sh'] = 'f' * 64
    else: owners['gx1/extra.py'] = 'f' * 64
    with pytest.raises(RuntimeError):
        g.native_provider_hlc_replay_source_bridge(owners, producer_commit=producer)


@pytest.mark.parametrize('name', ['gx1/contracts/gx1_capped_execution_v1.py', 'scripts/gx1_capped_run.sh'])
def test_replay_source_bridge_rejects_unrelated_source_delta(monkeypatch, name):
    path = Path(g.__file__).resolve().parents[2] / name
    original = Path.read_bytes
    def changed(candidate):
        raw = original(candidate)
        if candidate == path:
            raw += b'\nUNEXPECTED_OLD_OWNER_DELTA = 1\n' if name.endswith('.py') else b'\n# old shell delta\n'
        return raw
    monkeypatch.setattr(Path, 'read_bytes', changed)
    with pytest.raises(RuntimeError):
        g.native_provider_hlc_replay_source_bridge(GUARDS, producer_commit=PARENT)
