"""Synthetic endpoint-cross route containment; no model, payload, GPU or job."""
import hashlib
import os
import shlex
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / 'scripts/gx1_capped_run.sh'
ANCHOR = '804145939a03d6d4b000fb472ae08640d9d2e8f6'
OLD_SHELL_SHA256 = '1c06c95eaac4752395b71b4da65f7cb21f9613b389f4c59fb97fb552723c4e5d'
NAME = 'NATIVE_PROVIDER_HLC_NONE_ENDPOINT_CROSS_20261124_002'
MODULE = 'gx1.audit.diagnose_native_provider_hlc_endpoint_cross_v1'
RAW = b'{"synthetic_endpoint_cross_plan":true}\n'
SHA = hashlib.sha256(RAW).hexdigest()
MARKER = 'GX1 PROVIDER HLC ENDPOINT CROSS ROUTE 20261124 001'
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
 'GX1_TRAINER_MAX_WALL_SECONDS': '900',
 'GX1_TRAINER_MODEL_MAX_WALL_SECONDS': '870',
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
 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING': 'false',
 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256': '',
 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE': '',
 'GX1_CAPPED_CLASS': 'trainer',
 'GX1_CUDA_PRODUCER_GUARD': 'false',
 'GX1_TRAINER_DEVICE': 'cuda',
 'GX1_TRAINER_EXECUTION_MODE': 'canonical',
 'BLIS_NUM_THREADS': '8',
 'ARROW_NUM_THREADS': '1',
 'POLARS_MAX_THREADS': '1',
 'GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS': 'true',
 'GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_PLAN_SHA256': 'SHA'}
EXPORTS = '  --setenv=GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS="$NATIVE_PROVIDER_HLC_ENDPOINT_CROSS" \\\n  --setenv=GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_PLAN_SHA256="$NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_PLAN_SHA256" \\\n'


def fragments(text):
    result = []
    anchors = [('STATE', '# BEGIN GX1 ALL TF LEARNING ROUTE 20261124 001 STATE\n', 'before'),
               ('MATCHERS', 'validate_target_command() {\n', 'before'),
               ('DISPATCH', 'validate_target_command() {\n', 'after'),
               ('NESTED', '# BEGIN GX1 ALL TF LEARNING ROUTE 20261124 001 NESTED\n', 'before')]
    for section, anchor, position in anchors:
        begin = '# BEGIN ' + MARKER + ' ' + section + '\n'
        end = '# END ' + MARKER + ' ' + section + '\n'
        assert text.count(begin) == text.count(end) == 1
        start = text.index(begin); stop = text.index(end, start) + len(end)
        result.append((anchor, position, text[start:stop]))
    result.append(('  --setenv=GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING="$NATIVE_PROVIDER_HLC_ALL_TF_LEARNING" \\\n', 'before', EXPORTS))
    return result


def strip_route(text):
    for anchor, position, fragment in reversed(fragments(text)):
        composite = fragment + anchor if position == 'before' else anchor + fragment
        assert text.count(fragment) == text.count(composite) == 1
        text = text.replace(composite, anchor, 1)
    return text


FRAGMENTS = fragments(RUNNER.read_text())
RESOURCE_VALUES = {
    'TRAINER_DEVICE': 'cuda', 'TRAINER_EXECUTION_MODE': 'canonical',
    'TRAINER_MAX_WALL_SECONDS': '900', 'TRAINER_MODEL_MAX_WALL_SECONDS': '870',
    'TRAINER_GPU_INDEX': '0', 'TRAINER_GPU_MAX_POWER_LIMIT_W': '250',
    'TRAINER_GPU_MAX_POWER_DRAW_W': '250', 'TRAINER_GPU_MAX_CORE_TEMP_C': '75',
    'TRAINER_GPU_MAX_MEMORY_TEMP_C': '80', 'TRAINER_GPU_MAX_MEMORY_USED_MIB': '12288',
    'TRAINER_GPU_MONITOR_INTERVAL_SECONDS': '1', 'TRAINER_CPU_MAX_TEMP_C': '80',
    'TRAINER_CPU_MAX_PACKAGE_POWER_W': '125',
    'TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES': '8589934592',
    'TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES': '8589934592',
}


def harness(tmp_path, *, case=None, dispatch=False, extra_env=None):
    repo = tmp_path / 'repo'; repo.mkdir()
    config = repo / 'configs/research'; config.mkdir(parents=True)
    plan = config / (NAME + '.json'); plan.write_bytes(RAW)
    parent = repo / 'runs'; parent.mkdir(); output = parent / NAME
    argv = [str(repo / '.venv/bin/python'), '-m', MODULE, '--plan-sha256', SHA]
    if case == 'preflight': argv += ['--preflight']
    elif case == 'extra': argv += ['--unsafe']
    elif case == 'duplicate_preflight': argv += ['--preflight', '--preflight']
    elif case == 'wrong_module': argv[2] = 'gx1.audit.native_provider_hlc_learning_v1'
    elif case == 'wrong_python': argv[0] = '/usr/bin/python3'
    elif case == 'wrong_sha': argv[4] = 'f' * 64
    elif case == 'uppercase_sha': argv[4] = SHA.upper()
    elif case == 'short_sha': argv[4] = SHA[:-1]
    elif case == 'lane': argv += ['--lane', 'main']
    elif case == 'reordered': argv[3:5] = [SHA, '--plan-sha256']
    elif case == 'equals': argv[3:5] = ['--plan-sha256=' + SHA]
    elif case == 'legacy_flag': argv[3] = '--all-tf-learning-plan-sha256'
    elif case == 'mixed_flag': argv += ['--all-tf-learning-plan-sha256', SHA]
    elif case == 'script': argv[1:3] = [str(repo / 'gx1/audit/diagnose_native_provider_hlc_endpoint_cross_v1.py')]
    elif case == 'relative_script': argv[1:3] = ['gx1/audit/diagnose_native_provider_hlc_endpoint_cross_v1.py']
    elif case == 'env_launcher': argv = ['/usr/bin/env', *argv]
    elif case == 'plan_symlink':
        original = plan.with_suffix('.original'); plan.rename(original); plan.symlink_to(original)
    elif case == 'plan_parent_symlink':
        real = repo / 'real-config'; config.rename(real); config.symlink_to(real)
    elif case == 'missing_plan': plan.unlink()
    elif case == 'existing_output': output.mkdir()
    elif case == 'output_symlink': output.symlink_to(repo / 'absent')
    elif case == 'output_parent_symlink':
        real = repo / 'real-runs'; parent.rename(real); parent.symlink_to(real)
    elif case == 'missing_output_parent': parent.rmdir()
    state, functions, route = [item[2] for item in FRAGMENTS[:3]]
    functions = functions.replace('/home/andre2/GX1_RUNS/' + NAME, str(output))
    values = {'REPO_ROOT': str(repo), 'CANONICAL_TRAINER_PYTHON': str(repo / '.venv/bin/python'),
              'JOB_CLASS': 'trainer', 'ATTENDED_SMOKE': 'false', 'CUDA_PRODUCER_GUARD': 'false',
              'MEM': '20G', 'SWAP': '512M', 'POWER_BENCHMARK_SCOPE_JSON': '', 'POWER_BENCHMARK_SCOPE_SHA256': ''}
    if case == 'audit_class': values['JOB_CLASS'] = 'audit'
    elif case == 'producer_class': values['JOB_CLASS'] = 'producer'
    elif case == 'wrong_memory': values['MEM'] = '4G'
    elif case == 'wrong_swap': values['SWAP'] = '1G'
    elif case == 'attended': values['ATTENDED_SMOKE'] = 'true'
    elif case == 'cuda_producer': values['CUDA_PRODUCER_GUARD'] = 'true'
    elif case == 'power_scope': values['POWER_BENCHMARK_SCOPE_SHA256'] = 'f' * 64
    script = state + functions + ''.join('\n' + key + '=' + shlex.quote(value) for key, value in values.items())
    if dispatch:
        script += '\nvalidate_target_command() {\n' + route + 'echo UNEXPECTED_FALLTHROUGH; return 99; }\nvalidate_target_command '
    else:
        script += '\nconfigure_native_provider_hlc_endpoint_cross '
    script += ' '.join(map(shlex.quote, argv)) + '\n'
    print_fields = {'NATIVE_PROVIDER_HLC_ENDPOINT_CROSS': 'true',
                    'NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_PLAN_SHA256': SHA,
                    **RESOURCE_VALUES, 'TRAINER_OUT_BUNDLE_DIR': str(output)}
    script += ''.join('printf "%s=%s\\n" ' + shlex.quote(key) + ' "$' + key + '"\n' for key in print_fields)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('GX1_CAPPED_', 'GX1_NATIVE_', 'GX1_TRAINER_', 'GX1_POWER_'))}
    env.update(extra_env or {})
    result = subprocess.run(['bash', '-c', script], cwd=tmp_path if case == 'wrong_cwd' else repo,
                            text=True, capture_output=True, env=env)
    assert not (output / 'CLAIM.json').exists()
    return result, output, print_fields


@pytest.mark.parametrize('dispatch', [False, True])
@pytest.mark.parametrize('preflight', [False, True])
def test_exact_cli_and_preflight_fix_caps_without_creating_scope(tmp_path, dispatch, preflight):
    result, output, expected = harness(tmp_path, dispatch=dispatch, case='preflight' if preflight else None)
    assert result.returncode == 0, result.stderr
    assert dict(line.split('=', 1) for line in result.stdout.splitlines()) == expected
    assert not output.exists()


@pytest.mark.parametrize('case', ['extra', 'duplicate_preflight', 'wrong_module', 'wrong_python', 'wrong_sha',
                                  'uppercase_sha', 'short_sha', 'lane', 'reordered', 'equals', 'legacy_flag',
                                  'mixed_flag', 'script', 'relative_script', 'env_launcher', 'plan_symlink',
                                  'plan_parent_symlink', 'missing_plan', 'existing_output', 'output_symlink',
                                  'output_parent_symlink', 'missing_output_parent', 'wrong_cwd', 'audit_class',
                                  'producer_class', 'wrong_memory', 'wrong_swap', 'attended', 'cuda_producer', 'power_scope'])
def test_route_rejects_foreign_cli_paths_resources_or_consumed_output(tmp_path, case):
    result, _, _ = harness(tmp_path, case=case)
    assert result.returncode == 75, (result.stdout, result.stderr)


@pytest.mark.parametrize('case', ['script', 'relative_script', 'env_launcher', 'mixed_flag', 'equals', 'legacy_flag'])
def test_module_reference_cannot_fall_through_into_a_legacy_route(tmp_path, case):
    result, _, _ = harness(tmp_path, case=case, dispatch=True)
    assert result.returncode == 75 and 'UNEXPECTED_FALLTHROUGH' not in result.stdout


@pytest.mark.parametrize('field', [key for key in ENV if key.startswith('GX1_NATIVE_')
                                 and not key.startswith('GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS')])
def test_every_legacy_mode_plan_lane_or_mode_pollution_is_rejected(tmp_path, field):
    result, _, _ = harness(tmp_path, extra_env={field: 'forged'})
    assert result.returncode == 75, result.stderr


@pytest.mark.parametrize('extra_env', [
    {'GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS': 'true'},
    {'GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_PLAN_SHA256': SHA},
    {'GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_LANE': ''},
    {'GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_MODE': 'main'},
    {'GX1_CAPPED_MEMORY_BYTES': '21474836480'},
    {'GX1_CAPPED_CLASS': 'trainer', 'GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS': 'true',
     'GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_PLAN_SHA256': 'f' * 64},
])
def test_outer_or_partial_nested_endpoint_authority_is_rejected(tmp_path, extra_env):
    result, _, _ = harness(tmp_path, extra_env=extra_env)
    assert result.returncode == 75, result.stderr


@pytest.mark.parametrize('field', [None, *ENV])
def test_nested_parent_requires_every_exact_deadline_hardware_thread_and_mode_field(field):
    values = {key: SHA if value == 'SHA' else value for key, value in ENV.items()}
    if field: values[field] += 'wrong'
    script = 'NATIVE_PROVIDER_HLC_ENDPOINT_CROSS=true\nNATIVE_PROVIDER_HLC_ENDPOINT_CROSS_PLAN_SHA256=' + SHA + '\n'
    script += ''.join('export ' + key + '=' + shlex.quote(value) + '\n' for key, value in values.items())
    script += FRAGMENTS[3][2]
    result = subprocess.run(['bash', '-c', script], text=True, capture_output=True)
    assert result.returncode == (75 if field else 0), result.stderr


def test_consumed_affinity_hint_is_not_required_from_nested_python_environment():
    assert 'GX1_CPU_AFFINITY' not in ENV
    assert 'GX1_CPU_AFFINITY' not in FRAGMENTS[3][2]
    original = subprocess.check_output(['git', 'show', ANCHOR + ':scripts/gx1_capped_run.sh'], cwd=REPO)
    current = RUNNER.read_bytes()
    assert current.split(b"SCOPE_GUARD='\n", 1)[1].replace(EXPORTS.encode(), b'', 1) == original.split(b"SCOPE_GUARD='\n", 1)[1]


def test_every_old_shell_byte_and_shared_python_guard_remains_authentic():
    original = subprocess.check_output(['git', 'show', ANCHOR + ':scripts/gx1_capped_run.sh'], cwd=REPO)
    assert hashlib.sha256(original).hexdigest() == OLD_SHELL_SHA256
    assert strip_route(RUNNER.read_text()).encode() == original
    name = 'gx1/contracts/gx1_capped_execution_v1.py'
    assert (REPO / name).read_bytes() == subprocess.check_output(['git', 'show', ANCHOR + ':' + name], cwd=REPO)


@pytest.mark.parametrize('case', ['old_delta', 'duplicate', 'moved', 'export_delta'])
def test_legacy_strip_proof_rejects_extra_delta_duplicate_or_moved_route(case):
    text = RUNNER.read_text()
    if case == 'old_delta': text += '\n# unrelated old-owner delta\n'
    elif case == 'duplicate': text += FRAGMENTS[0][2]
    elif case == 'moved':
        text = text.replace(FRAGMENTS[0][2], '', 1) + FRAGMENTS[0][2]
    elif case == 'export_delta': text = text.replace(EXPORTS, EXPORTS.replace('="$NATIVE_PROVIDER_HLC_ENDPOINT_CROSS"', '=true'), 1)
    original = subprocess.check_output(['git', 'show', ANCHOR + ':scripts/gx1_capped_run.sh'], cwd=REPO)
    with pytest.raises(AssertionError):
        assert strip_route(text).encode() == original
