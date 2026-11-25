"""Synthetic row-membership qualification route containment; no model, payload, GPU or job."""
import ast
import hashlib
import re
import os
import shlex
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / 'scripts/gx1_capped_run.sh'
ANCHOR = 'c357b7ade09a50c3d3855b3c01f86f4fbe24fb44'
OLD_SHELL_SHA256 = 'e1bbd4d6f12e2fc475a32d54c017dbfa7103cc4ed84be46253d05591804d89d7'
NAME = 'NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_20261125_001'
MODULE = 'gx1.audit.native_provider_hlc_row_membership_qualification_v1'
RAW = b'{"synthetic_gradient_profile_plan":true}\n'
SHA = hashlib.sha256(RAW).hexdigest()
MARKER = 'GX1 PROVIDER HLC ROW MEMBERSHIP QUALIFICATION ROUTE 20261125 001'


def legacy_literal(name):
    path = REPO / 'tests/test_native_provider_hlc_endpoint_cross_guard_v1.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return ast.literal_eval(node.value)


ENV = dict(legacy_literal('ENV'))
ENV.update({'GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS': 'false',
            'GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_PLAN_SHA256': '',
            'GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION': 'true',
            'GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256': 'SHA',
            'GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE': 'warm',
            'GX1_NATIVE_PROVIDER_HLC_GRADIENT_PROFILE': 'false',
            'GX1_NATIVE_PROVIDER_HLC_GRADIENT_PROFILE_PLAN_SHA256': '',
            'GX1_TRAINER_MAX_WALL_SECONDS': '1800',
            'GX1_TRAINER_MODEL_MAX_WALL_SECONDS': '1770'})
EXPORTS = '  --setenv=GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION="$NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION" \\\n  --setenv=GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256="$NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256" \\\n  --setenv=GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE="$NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE" \\\n'


def fragments(text):
    result = []
    anchors = [('STATE', '# BEGIN GX1 PROVIDER HLC GRADIENT PROFILE ROUTE 20261125 001 STATE\n', 'before'),
               ('MATCHERS', 'validate_target_command() {\n', 'before'),
               ('DISPATCH', 'validate_target_command() {\n', 'after'),
               ('NESTED', '# BEGIN GX1 PROVIDER HLC GRADIENT PROFILE ROUTE 20261125 001 NESTED\n', 'before')]
    for section, anchor, position in anchors:
        begin = '# BEGIN ' + MARKER + ' ' + section + '\n'
        end = '# END ' + MARKER + ' ' + section + '\n'
        assert text.count(begin) == text.count(end) == 1
        start = text.index(begin); stop = text.index(end, start) + len(end)
        result.append((anchor, position, text[start:stop]))
    result.append(('  --setenv=GX1_NATIVE_PROVIDER_HLC_GRADIENT_PROFILE="$NATIVE_PROVIDER_HLC_GRADIENT_PROFILE" \\\n', 'before', EXPORTS))
    return result


def strip_route(text):
    for anchor, position, fragment in reversed(fragments(text)):
        composite = fragment + anchor if position == 'before' else anchor + fragment
        assert text.count(fragment) == text.count(composite) == 1
        text = text.replace(composite, anchor, 1)
    return text


FRAGMENTS = fragments(RUNNER.read_text())
NATIVE_STATES = '\n'.join(re.findall(r'^NATIVE_[A-Z0-9_]+=.*$', RUNNER.read_text(), re.M)) + '\n'
RESOURCE_VALUES = {**legacy_literal('RESOURCE_VALUES'), 'TRAINER_MAX_WALL_SECONDS': '1800', 'TRAINER_MODEL_MAX_WALL_SECONDS': '1770'}
NATIVE_EXPORT_BINDINGS = re.findall(r'--setenv=(GX1_NATIVE_[A-Z0-9_]+)="\$(NATIVE_[A-Z0-9_]+)"', RUNNER.read_text())


def harness(tmp_path, *, case=None, dispatch=False, extra_env=None, capture_exports=False, lane='warm'):
    repo = tmp_path / 'repo'; repo.mkdir()
    config = repo / 'configs/research'; config.mkdir(parents=True)
    plan = config / (NAME + '.json'); plan.write_bytes(RAW)
    parent = repo / 'runs'; parent.mkdir(); output = parent / (NAME + '_' + lane)
    argv = [str(repo / '.venv/bin/python'), '-m', MODULE, '--membership-qualification-plan-sha256', SHA, '--lane', lane]
    if case == 'preflight': argv += ['--preflight']
    elif case == 'extra': argv += ['--unsafe']
    elif case == 'duplicate_preflight': argv += ['--preflight', '--preflight']
    elif case == 'wrong_module': argv[2] = 'gx1.audit.native_provider_hlc_learning_v1'
    elif case == 'wrong_python': argv[0] = '/usr/bin/python3'
    elif case == 'wrong_sha': argv[4] = 'f' * 64
    elif case == 'uppercase_sha': argv[4] = SHA.upper()
    elif case == 'short_sha': argv[4] = SHA[:-1]
    elif case == 'lane': argv += ['--lane', 'main']
    elif case == 'reordered': argv[3:5] = [SHA, '--membership-qualification-plan-sha256']
    elif case == 'equals': argv[3:5] = ['--membership-qualification-plan-sha256=' + SHA]
    elif case == 'legacy_flag': argv[3] = '--all-tf-learning-plan-sha256'
    elif case == 'mixed_flag': argv += ['--all-tf-learning-plan-sha256', SHA]
    elif case == 'script': argv[1:3] = [str(repo / 'gx1/audit/native_provider_hlc_row_membership_qualification_v1.py')]
    elif case == 'relative_script': argv[1:3] = ['gx1/audit/native_provider_hlc_row_membership_qualification_v1.py']
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
    functions = functions.replace('/home/andre2/GX1_RUNS/' + NAME, str(parent / NAME))
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
    script = NATIVE_STATES + functions + ''.join('\n' + key + '=' + shlex.quote(value) for key, value in values.items())
    if dispatch:
        script += '\nvalidate_target_command() {\n' + route + 'echo UNEXPECTED_FALLTHROUGH; return 99; }\nvalidate_target_command '
    else:
        script += '\nconfigure_native_provider_hlc_row_membership_qualification '
    script += ' '.join(map(shlex.quote, argv)) + '\n'
    print_fields = {'NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION': 'true',
                    'NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256': SHA,
                    'NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE': lane,
                    **RESOURCE_VALUES, 'TRAINER_OUT_BUNDLE_DIR': str(output)}
    if capture_exports:
        print_fields = {key: SHA if value == 'SHA' else value for key, value in ENV.items()
                        if key.startswith('GX1_NATIVE_')}
        print_fields['GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE'] = lane
        print_variables = NATIVE_EXPORT_BINDINGS
    else:
        print_variables = [(key, key) for key in print_fields]
    script += ''.join('printf "%s=%s\\n" ' + shlex.quote(key) + ' "$' + variable + '"\n'
                      for key, variable in print_variables)
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
                                 and not key.startswith('GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION')])
def test_every_legacy_mode_plan_lane_or_mode_pollution_is_rejected(tmp_path, field):
    result, _, _ = harness(tmp_path, extra_env={field: 'forged'})
    assert result.returncode == 75, result.stderr


@pytest.mark.parametrize('extra_env', [
    {'GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION': 'true'},
    {'GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256': SHA},
    {'GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE': 'foreign'},
    {'GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_MODE': 'main'},
    {'GX1_CAPPED_MEMORY_BYTES': '21474836480'},
    {'GX1_CAPPED_CLASS': 'trainer', 'GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION': 'true',
     'GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256': 'f' * 64},
])
def test_outer_or_partial_nested_endpoint_authority_is_rejected(tmp_path, extra_env):
    result, _, _ = harness(tmp_path, extra_env=extra_env)
    assert result.returncode == 75, result.stderr


@pytest.mark.parametrize('field', [None, *ENV])
def test_nested_parent_requires_every_exact_deadline_hardware_thread_and_mode_field(field):
    values = {key: SHA if value == 'SHA' else value for key, value in ENV.items()}
    if field: values[field] += 'wrong'
    script = NATIVE_STATES + FRAGMENTS[1][2]
    script += ''.join(key + '=' + shlex.quote(value) + '\n' for key, value in {**RESOURCE_VALUES, 'TRAINER_ATTENDED_STAGE_REQUIRED': 'false'}.items())
    script += 'NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION=true\nNATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256=' + SHA + '\nNATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE=warm\n'
    script += ''.join('export ' + key + '=' + shlex.quote(value) + '\n' for key, value in values.items())
    script += FRAGMENTS[3][2]
    result = subprocess.run(['bash', '-c', script], text=True, capture_output=True)
    assert result.returncode == (75 if field else 0), result.stderr


def test_consumed_affinity_hint_is_not_required_from_nested_python_environment():
    assert 'GX1_CPU_AFFINITY' not in ENV
    assert 'GX1_CPU_AFFINITY' not in FRAGMENTS[3][2]
    original = subprocess.check_output(['git', 'show', ANCHOR + ':scripts/gx1_capped_run.sh'], cwd=REPO)
    current = RUNNER.read_bytes()
    assert strip_route(current.decode()).split("SCOPE_GUARD='\n", 1)[1] == original.decode().split("SCOPE_GUARD='\n", 1)[1]


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
    elif case == 'export_delta': text = text.replace(EXPORTS, EXPORTS.replace('="$NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION"', '=true'), 1)
    original = subprocess.check_output(['git', 'show', ANCHOR + ':scripts/gx1_capped_run.sh'], cwd=REPO)
    with pytest.raises(AssertionError):
        assert strip_route(text).encode() == original


@pytest.mark.parametrize('extra_env', [
    {'GX1_NATIVE_UNDECLARED': ''},
    {'GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_MODULE': MODULE},
    {'GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_UNKNOWN': ''},
    {'GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256': SHA},
])
def test_unknown_native_fields_and_legacy_alias_authority_fail_closed(tmp_path, extra_env):
    result, _, _ = harness(tmp_path, extra_env=extra_env)
    assert result.returncode == 75, result.stderr


def test_exact_protected_parent_can_reenter_only_its_same_command(tmp_path):
    values = {key: SHA if value == 'SHA' else value for key, value in ENV.items()}
    result, output, expected = harness(tmp_path, extra_env=values)
    assert result.returncode == 0, result.stderr
    assert dict(line.split('=', 1) for line in result.stdout.splitlines()) == expected
    assert not output.exists()


@pytest.mark.parametrize('field', [key for key in ENV if key.startswith('GX1_NATIVE_')])
def test_nested_parent_cannot_omit_any_native_mode_plan_or_lane(field):
    values = {key: SHA if value == 'SHA' else value for key, value in ENV.items() if key != field}
    script = NATIVE_STATES + FRAGMENTS[1][2]
    script += ''.join(key + '=' + shlex.quote(value) + '\n' for key, value in {**RESOURCE_VALUES, 'TRAINER_ATTENDED_STAGE_REQUIRED': 'false'}.items())
    script += 'NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION=true\nNATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256=' + SHA + '\nNATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE=warm\n'
    script += ''.join('export ' + key + '=' + shlex.quote(value) + '\n' for key, value in values.items())
    script += FRAGMENTS[3][2]
    env = {key: value for key, value in os.environ.items() if not key.startswith('GX1_NATIVE_')}
    result = subprocess.run(['bash', '-c', script], text=True, capture_output=True, env=env)
    assert result.returncode == 75, result.stderr


def test_new_route_keeps_taskset_and_physical_keeper_outside_additions():
    source = RUNNER.read_text()
    old = strip_route(source)
    assert 'CPU_AFFINITY=0-18' in old
    assert 'NUMERICAL_THREAD_COUNT=8' in old
    assert 'TRAINER_TASKS_MAX=128' in old
    assert 'GX1_CPU_AFFINITY' not in ''.join(fragment for _, _, fragment in FRAGMENTS)
    assert 'nvidia-smi' not in ''.join(fragment for _, _, fragment in FRAGMENTS)



def test_actual_native_export_keyset_is_exactly_the_profile_contract():
    actual = re.findall(r'--setenv=(GX1_NATIVE_[A-Z0-9_]+)=', RUNNER.read_text())
    assert len(actual) == len(set(actual)) == len(NATIVE_EXPORT_BINDINGS)
    assert set(actual) == {key for key in ENV if key.startswith('GX1_NATIVE_')}


@pytest.mark.parametrize('preflight', [False, True])
def test_actual_shell_configure_exports_enter_only_the_new_owner_scope(tmp_path, monkeypatch, preflight):
    # Runs only during the integrator's capped synthetic suite; no model is built.
    from gx1.audit import native_provider_hlc_row_membership_qualification_v1 as owner
    result, output, expected = harness(tmp_path, case='preflight' if preflight else None,
                                       capture_exports=True)
    assert result.returncode == 0, result.stderr
    actual = dict(line.split('=', 1) for line in result.stdout.splitlines())
    assert actual == expected
    repo = output.parent.parent
    monkeypatch.setattr(owner, 'REPO', repo)
    monkeypatch.setattr(owner, 'PLAN', repo / 'configs/research' / (NAME + '.json'))
    monkeypatch.setattr(owner, 'ROOTS', {'warm': output, 'cold': output.parent / (NAME + '_cold')})
    monkeypatch.chdir(repo)
    calls = []
    monkeypatch.setattr(owner, 'require_guarded_cuda_trainer_execution',
                        lambda: calls.append(True) or {'synthetic_common_guard': True})
    env = {key: SHA if value == 'SHA' else value for key, value in ENV.items()
           if not key.startswith('GX1_NATIVE_')}
    env.update(actual)
    argv = [str(repo / '.venv/bin/python'), '-m', MODULE, '--membership-qualification-plan-sha256', SHA, '--lane', 'warm']
    if preflight: argv.append('--preflight')
    assert owner.require_scope(SHA, 'warm', preflight=preflight, environ=env, argv=argv) == {'synthetic_common_guard': True}
    assert calls == [True]
    env['GX1_NATIVE_UNDECLARED'] = ''
    with pytest.raises(ValueError, match='EXACT_SCOPE_ENV'):
        owner.require_scope(SHA, 'warm', preflight=preflight, environ=env, argv=argv)
    assert calls == [True]


@pytest.mark.parametrize('lane', ['cold', 'main', 'resume', ''])
def test_foreign_parent_lane_cannot_authorize_the_warm_command(tmp_path, lane):
    values = {key: SHA if value == 'SHA' else value for key, value in ENV.items()}
    values['GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE'] = lane
    result, _, _ = harness(tmp_path, extra_env=values)
    assert result.returncode == 75


@pytest.mark.parametrize('lane', ['warm', 'cold'])
def test_each_declared_technical_lane_has_its_own_fresh_root(tmp_path, lane):
    result, output, expected = harness(tmp_path, dispatch=True, lane=lane)
    assert result.returncode == 0, result.stderr
    assert output.name == NAME + '_' + lane and not output.exists()
    assert dict(line.split('=', 1) for line in result.stdout.splitlines()) == expected
