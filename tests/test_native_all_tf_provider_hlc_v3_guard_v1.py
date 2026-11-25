"""V3 guard metadata, exact CLI and authenticated source reversal; no hardware IO."""
import ast
import copy
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from gx1.contracts import gx1_capped_execution_v1 as g
from gx1.contracts import native_future_state_provider_hlc_v1 as hlc
from tests import test_native_future_state_provider_hlc_baseline_guard_v1 as legacy

ANCHOR = '8fd74a76cc45ac7306da4b905c82e9cd9b12976d'
INITIAL = 'NATIVE_ALL_TF_PROVIDER_HLC_V3_WHOLE_TRAIN_INITIAL_20261125_001'
REPLAY = 'NATIVE_ALL_TF_PROVIDER_HLC_V3_COLD_REPLAY_20261125_001'
OWNER_NAMES = ('gx1/contracts/gx1_capped_execution_v1.py', 'scripts/gx1_capped_run.sh')


def _owners():
    repo = Path(g.__file__).resolve().parents[2]
    return {name: hashlib.sha256(subprocess.check_output(
        ['git', 'show', ANCHOR + ':' + name], cwd=repo)).hexdigest() for name in OWNER_NAMES}


def _fixture(monkeypatch, tmp_path, kind='initial', bad=None):
    repo, runs = tmp_path / 'repo', tmp_path / 'runs'
    (repo / 'configs/research').mkdir(parents=True)
    (repo / 'gx1/contracts').mkdir(parents=True)
    (repo / 'gx1/audit').mkdir(parents=True)
    runs.mkdir()
    initial = kind == 'initial'
    lane = None if initial else kind.removeprefix('replay_')
    name = INITIAL if initial else REPLAY
    owner = repo / 'gx1/audit' / ('native_future_state_provider_hlc_initial_v1.py'
        if initial else 'native_provider_hlc_replay_v1.py')
    owner.write_text('# metadata-only fixture\n')
    contract = {'target_contract_sha256': hlc.all_tf_provider_hlc_state_contract()['contract_sha256'],
        'target_metadata_as_model_input': False, 'contract_sha256': 'a' * 64}
    if bad == 'wrong_target': contract['target_contract_sha256'] = hlc.provider_hlc_state_contract()['contract_sha256']
    root = runs / name
    if initial:
        (root / 'PLANS').mkdir(parents=True)
        base = {'schema_version': 'gx1_native_all_tf_provider_hlc_v3_resumable_initial_base_plan_v1',
            'stage': name.lower(), 'output_root': str(root),
            'recipe': copy.deepcopy(legacy.SEGMENT_RECIPE),
            'execution': copy.deepcopy(legacy.SEGMENT_EXECUTION), 'model_contract': contract}
        base_path = repo / 'configs/research' / (name + '.json')
        base_path.write_text(json.dumps(base, sort_keys=True))
        plan = {'schema_version': 'gx1_native_all_tf_provider_hlc_v3_initial_segment_plan_v1',
            'stage': name.lower(), 'base_root': str(root),
            'base_plan': {'path': str(base_path), 'sha256': hashlib.sha256(base_path.read_bytes()).hexdigest()},
            'recipe': copy.deepcopy(base['recipe']), 'execution': copy.deepcopy(base['execution']),
            'model_contract': copy.deepcopy(contract)}
    else:
        plan = {'schema_version': 'gx1_native_all_tf_provider_hlc_v3_cold_replay_plan_v1',
            'stage': name.lower(), 'model_contract': contract,
            'output_roots': {value: str(runs / (name + '_' + value)) for value in ('warm', 'cold')}}
    if bad == 'wrong_schema': plan['schema_version'] = 'gx1_native_provider_hlc_all_tf_initial_segment_plan_v1'
    raw = json.dumps(plan, sort_keys=True).encode()
    digest = hashlib.sha256(raw).hexdigest()
    path = root / 'PLANS' / (digest + '.json') if initial else repo / 'configs/research' / (name + '.json')
    path.write_bytes(raw)
    output = root / ('SEGMENT_' + digest) if initial else runs / (name + '_' + lane)
    if bad == 'existing_output': output.mkdir()
    if bad == 'plan_symlink':
        target = path.with_suffix('.original'); path.rename(target); path.symlink_to(target)
    real_path = Path
    adapted = lambda value: runs if str(value) == '/home/andre2/GX1_RUNS' else real_path(value)
    adapted.cwd = real_path.cwd
    monkeypatch.setattr(g, 'Path', adapted)
    monkeypatch.setattr(g, '__file__', str(repo / OWNER_NAMES[0]))
    monkeypatch.chdir(repo)
    flag = '--all-tf-v3-score-segment-plan-sha256' if initial else '--all-tf-v3-plan-sha256'
    argv = [str(owner), flag, digest] + ([] if initial else ['--lane', lane])
    if bad == 'extra_arg': argv += ['--unsafe']
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(sys, 'executable', str(repo / '.venv/bin/python'))
    env = legacy.all_tf_environment_values(digest, lane or 'warm')
    if not initial:
        env.update(GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL='false',
            GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256='',
            GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY='true',
            GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256=digest,
            GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE=lane,
            GX1_TRAINER_MAX_WALL_SECONDS='1800', GX1_TRAINER_MODEL_MAX_WALL_SECONDS='1770')
    for key, value in env.items(): monkeypatch.setenv(key, value)
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: {'kernel_scope': True})
    def call():
        return (g.require_native_all_tf_provider_hlc_v3_initial_segment_execution(plan_sha256=digest)
            if initial else g.require_native_all_tf_provider_hlc_v3_replay_execution(plan_sha256=digest, lane=lane))
    return {'repo': repo, 'runs': runs, 'path': path, 'digest': digest, 'output': output,
        'owner': owner, 'flag': flag, 'lane': lane, 'call': call}


@pytest.mark.parametrize('kind', ['initial', 'replay_warm', 'replay_cold'])
def test_exact_v3_guard_and_optional_preflight_reuse_containment_without_fit_authority(monkeypatch, tmp_path, kind):
    fixture = _fixture(monkeypatch, tmp_path, kind)
    for preflight in (False, True):
        if preflight: sys.argv.append('--preflight')
        proof = fixture['call']()
        assert proof['kernel_scope'] and proof['plan']['sha256'] == fixture['digest']
        assert proof['output_root'] == str(fixture['output'])
        assert proof['training_authority_granted'] is proof['optimizer_authority_granted'] is False
        assert proof['numerical_threads'] == 8 and proof['cpu_affinity'] == list(range(19))
        assert (proof['outer_seconds'], proof['internal_seconds']) == ((3600, 3570) if kind == 'initial' else (1800, 1770))


@pytest.mark.parametrize('bad', ['extra_arg', 'wrong_schema', 'wrong_target', 'existing_output', 'plan_symlink', 'legacy_guard'])
def test_v3_parent_cli_schema_target_and_freshness_are_exact(monkeypatch, tmp_path, bad):
    fixture = _fixture(monkeypatch, tmp_path, bad=bad)
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: pytest.fail('invalid plan reached kernel proof'))
    with pytest.raises(RuntimeError):
        if bad == 'legacy_guard': g.require_native_provider_hlc_all_tf_initial_segment_execution(plan_sha256=fixture['digest'])
        else: fixture['call']()


@pytest.mark.parametrize('kind,field', [('initial', 'GX1_TRAINER_GPU_MAX_POWER_LIMIT_W'), ('replay_warm', 'GX1_CAPPED_MEMORY_BYTES')])
def test_v3_changed_shared_resource_environment_fails_before_kernel(monkeypatch, tmp_path, kind, field):
    fixture = _fixture(monkeypatch, tmp_path, kind)
    monkeypatch.setenv(field, 'wrong')
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: pytest.fail('invalid env reached kernel proof'))
    with pytest.raises(RuntimeError, match='ENV_INVALID'): fixture['call']()


@pytest.mark.parametrize('kind', ['initial', 'replay_cold'])
def test_v3_shell_dispatch_binds_exact_module_flag_plan_and_resource_profile(monkeypatch, tmp_path, kind):
    fixture = _fixture(monkeypatch, tmp_path, kind)
    functions, dispatch = (part[2] for part in g._ALL_TF_V3_GUARD_SHELL_ADDITIONS)
    functions = functions.replace('/home/andre2/GX1_RUNS', str(fixture['runs']))
    state = {'REPO_ROOT': str(fixture['repo']), 'CANONICAL_TRAINER_PYTHON': str(fixture['repo'] / '.venv/bin/python'),
        'NATIVE_PROVIDER_HLC_INITIAL_MODULE': 'gx1.audit.native_future_state_provider_hlc_initial_v1',
        'NATIVE_PROVIDER_HLC_REPLAY_MODULE': 'gx1.audit.native_provider_hlc_replay_v1',
        'JOB_CLASS': 'trainer', 'ATTENDED_SMOKE': 'false', 'CUDA_PRODUCER_GUARD': 'false',
        'MEM': '20G', 'SWAP': '512M', 'POWER_BENCHMARK_SCOPE_JSON': '', 'POWER_BENCHMARK_SCOPE_SHA256': ''}
    script = 'set -e\n' + ''.join(key + '=' + shlex.quote(value) + '\n' for key, value in state.items())
    script += functions + '\nvalidate_target_command() {\n' + dispatch + '\nreturn 88\n}\n'
    module = state['NATIVE_PROVIDER_HLC_INITIAL_MODULE' if kind == 'initial' else 'NATIVE_PROVIDER_HLC_REPLAY_MODULE']
    argv = [state['CANONICAL_TRAINER_PYTHON'], '-m', module, fixture['flag'], fixture['digest']]
    if fixture['lane']: argv += ['--lane', fixture['lane']]
    for wrong_flag in (False, True):
        command = list(argv)
        if wrong_flag: command[3] = '--all-tf-plan-sha256'
        tail = '\nvalidate_target_command ' + ' '.join(map(shlex.quote, command)) + '\nprintf "%s,%s,%s\\n" "$TRAINER_GPU_MAX_POWER_DRAW_W" "$TRAINER_CPU_MAX_TEMP_C" "$TRAINER_OUT_BUNDLE_DIR"\n'
        result = subprocess.run(['bash', '-c', script + tail], cwd=fixture['repo'], capture_output=True, text=True)
        assert result.returncode == (88 if wrong_flag else 0), result.stderr
        if not wrong_flag: assert result.stdout.strip() == '250,80,' + str(fixture['output'])


def test_v3_source_bridge_authenticates_pre_v3_row_membership_parent_and_resources():
    proof = g.native_all_tf_provider_hlc_v3_initial_source_bridge(_owners(), producer_commit=ANCHOR)
    assert proof['checked_guard_owners'] == 2
    assert proof['old_guard_routes_preserved'] is proof['old_numeric_resource_environment_function_unchanged'] is True
    assert set(proof['owners']) == set(OWNER_NAMES)


@pytest.mark.parametrize('bad', ['shared_limit', 'legacy_bound_call', 'duplicate_function', 'extra_statement', 'misplaced_shell', 'parent_sha'])
def test_v3_source_bridge_rejects_unproven_resource_legacy_source_or_parent_delta(monkeypatch, bad):
    owners = _owners()
    if bad == 'parent_sha':
        monkeypatch.setattr(g, '_ALL_TF_V3_GUARD_PARENT', {**g._ALL_TF_V3_GUARD_PARENT, 'sha256': '0' * 64})
    else:
        repo = Path(g.__file__).resolve().parents[2]
        original = Path.read_text
        def changed(path, *args, **kwargs):
            text = original(path, *args, **kwargs)
            if path == repo / OWNER_NAMES[0] and bad != 'misplaced_shell':
                if bad in ('shared_limit', 'legacy_bound_call', 'duplicate_function'):
                    name = ('require_native_provider_hlc_all_tf_initial_segment_execution' if bad == 'legacy_bound_call'
                        else '_require_provider_hlc_all_tf_initial_environment')
                    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
                    source = ast.get_source_segment(text, node)
                    if bad == 'shared_limit': altered = source.replace('"250"', '"251"', 1)
                    elif bad == 'legacy_bound_call': altered = source.replace('_native_provider_hlc_all_tf_initial_segment_bound_plan(', '_native_all_tf_provider_hlc_v3_bound_plan(', 1)
                    else: altered = source + '\n\n' + source
                    assert altered != source; text = text.replace(source, altered, 1)
                else: text = text.replace('\n# END GX1 ALL TF V3 GUARDS 20261125 001', '\nUNEXPECTED_V3_GLOBAL = True\n# END GX1 ALL TF V3 GUARDS 20261125 001', 1)
            if path == repo / OWNER_NAMES[1] and bad == 'misplaced_shell':
                anchor, _, fragment = g._ALL_TF_V3_GUARD_SHELL_ADDITIONS[0]
                assert fragment + anchor in text; text = text.replace(fragment + anchor, anchor + fragment, 1)
            return text
        monkeypatch.setattr(Path, 'read_text', changed)
    with pytest.raises(RuntimeError): g.native_all_tf_provider_hlc_v3_initial_source_bridge(owners, producer_commit=ANCHOR)
