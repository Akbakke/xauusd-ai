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
MODULE = 'gx1.audit.diagnose_native_future_state_bias_v1'
BASENAME = 'NATIVE_FUTURE_STATE_BIAS_FROZEN_20261122_001'
ENV = {
    'GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN': 'true',
    'GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256': SHA,
    'GX1_NATIVE_FUTURE_STATE_THROUGHPUT': 'false',
    'GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256': '',
    'GX1_NATIVE_FUTURE_STATE_TRAINING': 'false',
    'GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256': '',
    'GX1_NATIVE_FUTURE_STATE_TRAINING_LANE': '',
    'GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION': 'false',
    'GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS': 'false',
    'GX1_TRAINER_MAX_WALL_SECONDS': '3600',
    'GX1_TRAINER_MODEL_MAX_WALL_SECONDS': '3600',
    'GX1_TRAINER_ATTENDED_STAGE_REQUIRED': 'false',
    'GX1_TRAINER_GPU_INDEX': '0',
    'GX1_TRAINER_GPU_MAX_POWER_LIMIT_W': '300',
    'GX1_TRAINER_GPU_MAX_POWER_DRAW_W': '310',
    'GX1_TRAINER_GPU_MAX_CORE_TEMP_C': '85',
    'GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C': '80',
    'GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB': '12288',
    'GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS': '1',
    'GX1_POWER_BENCHMARK_SCOPE_JSON': '',
    'GX1_POWER_BENCHMARK_SCOPE_SHA256': '',
}


def setup_env(monkeypatch):
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)


def test_only_exact_frozen_containment_and_no_training_or_optimizer_authority(monkeypatch):
    setup_env(monkeypatch)
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: {'canonical_scope': True})
    value = g.require_native_future_state_bias_frozen_execution(plan_sha256=SHA)
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
    with pytest.raises(RuntimeError, match='FROZEN_BIAS_ENV_INVALID'):
        g.require_native_future_state_bias_frozen_execution(plan_sha256=SHA)


@pytest.mark.parametrize('bad', ['', 'A' * 64, 'x' * 64, 'a' * 63, None, 1])
def test_bad_plan_digest_fails_before_any_environment_or_runtime(monkeypatch, bad):
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: pytest.fail('bad SHA reached runtime'))
    with pytest.raises(RuntimeError, match='FROZEN_BIAS_PLAN_SHA256_INVALID'):
        g.require_native_future_state_bias_frozen_execution(plan_sha256=bad)


def test_actual_scope_failure_is_propagated(monkeypatch):
    setup_env(monkeypatch)
    def failed_scope():
        raise RuntimeError('actual cgroup or lock failed')
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', failed_scope)
    with pytest.raises(RuntimeError, match='actual cgroup or lock failed'):
        g.require_native_future_state_bias_frozen_execution(plan_sha256=SHA)


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
    state = g._NATIVE_FUTURE_STATE_BIAS_GUARD_SHELL_ADDITIONS[0][2]
    functions = g._NATIVE_FUTURE_STATE_BIAS_GUARD_SHELL_ADDITIONS[1][2]
    script = state + functions + '\nREPO_ROOT=' + shlex.quote(str(repo))
    script += '\nCANONICAL_TRAINER_PYTHON=' + shlex.quote(str(repo / '.venv/bin/python'))
    script += '\nmatches_bound_native_future_state_bias_frozen ' + ' '.join(map(shlex.quote, argv)) + '\n'
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
    assert shell_harness(tmp_path, mutation=mutation).returncode != 0


def test_shell_insertions_are_exactly_reversible_to_current_pre_diagnosis_shell():
    repo = Path(g.__file__).resolve().parents[2]
    text = (repo / 'scripts/gx1_capped_run.sh').read_text()
    for anchor, position, fragment in reversed(g._NATIVE_FUTURE_STATE_BIAS_GUARD_SHELL_ADDITIONS + g._NATIVE_FUTURE_STATE_BIAS_SUFFIX_GUARD_SHELL_ADDITIONS):
        composite = fragment + anchor if position == 'before' else anchor + fragment
        assert text.count(fragment) == text.count(composite) == 1
        text = text.replace(composite, anchor, 1)
    original = subprocess.check_output(['git', 'show', 'f5d6e633f:scripts/gx1_capped_run.sh'], cwd=repo, text=True)
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
