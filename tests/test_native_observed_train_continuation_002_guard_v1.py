"""002 own-FINAL648 admission/source boundaries; synthetic JSON and code only."""
import copy
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from gx1.contracts import gx1_capped_execution_v1 as g

_spec = importlib.util.spec_from_file_location('gx1_passing_observed_001_guard_fixtures',
    Path(__file__).with_name('test_native_observed_train_continuation_guard_v1.py'))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
OWNERS = _base.OWNERS
NEW = ('gx1/audit/native_future_state_observed_train_continuation_002_v1.py',
    'tests/test_native_future_state_observed_train_continuation_002_v1.py',
    'tests/test_native_observed_train_continuation_002_guard_v1.py')


def fixture(monkeypatch, tmp_path, preflight=False, bad=None):
    f = _base.fixture(monkeypatch, tmp_path, preflight=True)
    repo, runs = f['repo'], f['runs']; parent_plan = copy.deepcopy(f['plan'])
    # The passing001 fixture has all648 vectors and a sealed synthetic schedule.
    # Supply only the additional synthetic trajectory identity consumed by002.
    parent_plan['metadata']['trajectory']['contract_sha256'] = 'b' * 64
    original = parent_plan['metadata']['producer']; original_ref = parent_plan['producer_plan']
    parent_ref = dict(path=str(repo / 'configs/research' / (g._V3_OBSERVED_TRAIN_CONTINUATION_NAME + '.json')),
        sha256='c4e9b8ff8f760b062d773b08a585caeff88cec551ced3f6e57704da61c50f41f')
    oldroot = runs / g._V3_OBSERVED_TRAIN_CONTINUATION_NAME
    receipt_ref = dict(path=str(oldroot / 'FINAL648.json'), sha256='f3534bdeeb460a56f04b5c8fa7b7142a56e43341ef6075b653fd30558fc8d0b3')
    checkpoint_ref = dict(path=str(oldroot / 'FINAL648.pt'), sha256='c21159dcade8f0fe36eddd4d0d53c62d1fa07be2985a19b18b7b0d4232b8d150')
    receipt = dict(logical_cursor=648, optimizer_updates=648, producer_lane='continuation',
        original_plan=parent_ref, execution_plan=parent_ref, checkpoint=checkpoint_ref,
        trajectory_sha256=parent_plan['metadata']['trajectory']['contract_sha256'])
    owners = dict(parent_plan['source_owners'])
    anchored = {n: (repo / n).read_bytes() for n in owners}
    for name in NEW:
        p = repo / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# synthetic002 source fixture\n')
        owners[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    def command(argv, **kwargs):
        return anchored[argv[2].split(':', 1)[1]] if argv[:2] == ['git', 'show'] else 'a' * 40 + '\n'
    monkeypatch.setattr(subprocess, 'check_output', command)
    source = parent_plan['schedule']; old = original['schedule']; schedule = copy.deepcopy(source); schedule.pop('contract_sha256')
    schedule.update(schema_version='gx1_native_v3_observed_train_continuation_002_schedule_v1', logical_batches=864,
        optimizer_updates=864, passes=4, skipped_batches=0, original_prefix_schedule_sha256=source['contract_sha256'],
        original_take_indices_appended=list(range(216, 432)), order_mode='repeat_locked_original_second_pass_once')
    for key in ('cohort_takes', 'logical_step_rows', 'cohort_take_state_sha256_by_logical_batch', 'update_flags'):
        schedule[key] = copy.deepcopy(source[key] + old[key][216:432])
    for key in ('raw_permutation_state_sha256_by_pass', 'actual_sorted_partition_stream_state_sha256_by_pass'):
        schedule[key] = source[key] + [old[key][1]]
    schedule['contract_sha256'] = '8' * 64
    trajectory = dict(schedule=schedule, original_prefix_trajectory_sha256=parent_plan['metadata']['trajectory']['contract_sha256'])
    evidence = runs / 'GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001'
    design_ref = dict(path=str(evidence / 'V3_OBSERVED_TRAIN_CONTINUATION_DESIGN_20261128_002.json'),
        sha256='f52d056a1715f13180b6aee375ae17fd45eb992644bf447f38a0239ebf34bf30')
    alias = dict(actual_before='BASELINE648', actual_candidate='FINAL864', numeric_owner_phase='PASS2')
    design = dict(phase_alias=alias, limits=copy.deepcopy(parent_plan['limits']))
    bridge = dict(producer_commit=g._V3_OBSERVED_TRAIN_CONTINUATION_002_PARENT,
        owners={n: dict(original_sha256='f' * 64, current_sha256=owners[n]) for n in OWNERS})
    output = runs / g._V3_OBSERVED_TRAIN_CONTINUATION_002_NAME
    plan = dict(schema_version='gx1_native_v3_observed_train_continuation_plan_v2',
        stage=g._V3_OBSERVED_TRAIN_CONTINUATION_002_STAGE, source_commit='a' * 40, output_root=str(output),
        execution=g._native_v3_development_execution('warm'), execution_profile=g._V3_OBSERVED_TRAIN_CONTINUATION_002_PROFILE,
        design=design_ref, producer_plan=parent_ref, original_producer_plan=original_ref,
        authentic_final_receipt=receipt_ref, authentic_final_checkpoint=checkpoint_ref,
        metadata=dict(parent_plan=parent_ref, original_plan=original_ref, receipt=copy.deepcopy(receipt), trajectory=trajectory),
        recipe=copy.deepcopy(parent_plan['recipe']), model_contract=copy.deepcopy(parent_plan['model_contract']),
        physical_input_identity=copy.deepcopy(parent_plan['physical_input_identity']), backbone_gradient_policy=None,
        schedule=schedule, starting_cursor=648, ending_cursor=864, checkpoint_cursors=[649, 864], observer_enabled=True,
        phase_alias=alias, limits=copy.deepcopy(parent_plan['limits']), inputs=copy.deepcopy(parent_plan['inputs']),
        source_owners=owners, guard_source_bridge=bridge)
    if bad == 'schema': plan['schema_version'] = 'v1'
    if bad == 'profile': plan['execution_profile'] = 'generic'
    if bad == 'execution': plan['execution']['GPU_draw_stop_W'] = 251
    if bad == 'old432_initializer': plan['authentic_final_checkpoint'] = parent_plan['authentic_final_checkpoint']
    if bad == 'wrong_producer': plan['producer_plan'] = original_ref
    if bad == 'receipt_432': receipt['logical_cursor'] = 432
    if bad == 'receipt_original': receipt['original_plan'] = original_ref
    if bad == 'thin_metadata': plan['metadata']['parent_plan'] = original_ref
    if bad == 'old_input_collision': plan['inputs']['recovery_design'] = design_ref
    if bad == 'prefix': schedule['cohort_takes'][647][0] += 1
    if bad == 'append': schedule['cohort_takes'][648][0] += 1
    if bad == 'bool_take': schedule['cohort_takes'][648][0] = False
    if bad == 'bool_update': schedule['update_flags'][648] = 1
    if bad == 'cursor': plan['starting_cursor'] = 432
    if bad == 'endpoint': plan['ending_cursor'] = 865
    if bad == 'alias': plan['phase_alias'] = dict(alias, actual_candidate='PASS2')
    if bad == 'extra_update': plan['limits']['optimizer_steps'] += 1
    if bad == 'baseline_forward': plan['limits']['model_forwards'] += 81
    if bad == 'autograd': plan['limits']['autograd_calls'] = 1
    if bad == 'bool_budget': plan['limits']['model_constructions'] = True
    if bad == 'bridge': plan['guard_source_bridge'] = {**bridge, 'extra': True}
    if bad == 'source': owners[NEW[0]] = '0' * 64
    if bad == 'old001_source':
        p = repo / NEW[0].replace('_002_v1.py', '_v1.py'); p.write_bytes(p.read_bytes() + b'# mutation\n')
        owners[str(p.relative_to(repo))] = hashlib.sha256(p.read_bytes()).hexdigest()
    path = repo / 'configs/research' / (g._V3_OBSERVED_TRAIN_CONTINUATION_002_NAME + '.json')
    path.write_text(json.dumps(plan)); digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = evidence / 'SYNTHETIC_FINAL648_MANIFEST.json'
    files = [receipt_ref, checkpoint_ref] if bad != 'backup_coverage' else [receipt_ref]
    manifest_path.write_text(json.dumps({'files': files})); manifest_ref = dict(path=str(manifest_path), sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    backup_path = evidence / 'SYNTHETIC_FINAL648_BACKUP.json'
    backup_path.write_text(json.dumps(dict(F_FULLFSYNC=bad != 'backup_unverified', separate_mac_disk=True,
        independent_member_readback=True, manifest=manifest_ref, manifest_sha256=manifest_ref['sha256'])))
    backup_ref = dict(path=str(backup_path), sha256=hashlib.sha256(backup_path.read_bytes()).hexdigest())
    pp = repo / 'NEXT_RUN_POLICY.json'; policy = json.loads(pp.read_text())
    for flag in ('training_enabled', 'native_training_enabled'): policy[flag] = not preflight
    previous = policy[g._V3_OBSERVED_TRAIN_CONTINUATION_STAGE]
    previous.update(status='CLOSED_CONSUMED_COMPLETE', plan=parent_ref, execution_authorized=False,
        training_authorized=False, remaining_invocations=0, relaunch_allowed=False, final_backup=backup_ref)
    if bad == 'old001_open': previous['execution_authorized'] = True
    if bad == 'old001_boolean_count': previous['remaining_invocations'] = False
    if bad == 'old001_failed': previous['status'] = 'FAILED_CONSUMED'
    if bad == 'backup_missing': previous.pop('final_backup')
    parent = dict(status='PREPARED_CLOSED_NOT_ADMITTED' if preflight else 'ADMITTED_EXACT_V3_OBSERVED_TRAIN_CONTINUATION_002',
        source_preparation_authorized=True, design=design_ref, plan=dict(path=str(path), sha256=digest),
        execution_authorized=not preflight, training_authorized=not preflight, remaining_invocations=int(not preflight),
        relaunch_allowed=False, test_authorized=False, spending_authorized=False)
    policy[g._V3_OBSERVED_TRAIN_CONTINUATION_002_STAGE] = parent
    if bad == 'flags': policy['native_launch_authorized'] = True
    if bad == 'bool_gate': parent['remaining_invocations'] = not preflight
    if bad == 'spent': parent['remaining_invocations'] = 0
    pp.write_text(json.dumps(policy))
    if not preflight:
        (evidence / 'V3_OBSERVED_TRAIN_CONTINUATION_ADMITTED_POLICY_20261128_002.json').write_bytes(b'{}' if bad == 'snapshot' else pp.read_bytes())
    if bad == 'existing': output.mkdir()
    if bad == 'symlink':
        other = path.with_suffix('.original'); path.rename(other); path.symlink_to(other)
    read = g._native_v3_development_read
    def adapted(candidate, candidate_digest=None):
        for binding, value in ((parent_ref, parent_plan), (original_ref, original), (receipt_ref, receipt), (design_ref, design)):
            if str(candidate) == binding['path']:
                assert candidate_digest == binding['sha256']; return value
        return read(candidate, candidate_digest)
    monkeypatch.setattr(g, '_native_v3_development_read', adapted)
    monkeypatch.setattr(g, 'native_v3_observed_train_continuation_002_guard_source_bridge', lambda *a, **k: bridge)
    argv = [str(repo / NEW[0]), '--v3-observed-train-continuation-002-plan-sha256', digest] + (['--preflight'] if preflight else [])
    if bad == 'argv': argv += ['--lane', 'fit']
    monkeypatch.setattr(sys, 'argv', argv)
    if bad == 'python': monkeypatch.setattr(sys, 'executable', '/usr/bin/python3')
    for key in list(os.environ):
        if key.startswith('GX1_NATIVE_'): monkeypatch.delenv(key)
    env = dict(GX1_NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_002='true', GX1_NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_002_PLAN_SHA256=digest)
    if bad == 'hardware_env': env['GX1_TRAINER_GPU_MAX_POWER_DRAW_W'] = '251'
    if bad == 'legacy_env': env['GX1_NATIVE_V3_OBSERVED_TRAIN_CONTINUATION'] = 'true'
    if bad == 'unknown_env': env['GX1_NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_002_EXTRA'] = 'false'
    for key, value in env.items(): monkeypatch.setenv(key, value)
    return dict(f, path=path, digest=digest, plan=plan, output=output,
        call=lambda: g.require_native_v3_observed_train_continuation_002_execution(plan_sha256=digest))


@pytest.mark.parametrize('preflight', (False, True))
def test_002_own648_resource_gate_accepts_only_fixed_scope(monkeypatch, tmp_path, preflight):
    f = fixture(monkeypatch, tmp_path, preflight); proof = f['call']()
    assert proof['preflight'] is preflight
    assert proof['training_authority_granted'] is proof['optimizer_authority_granted'] is False
    assert f['called'] == ['cuda'] and not f['output'].exists()


@pytest.mark.parametrize('bad', ('schema', 'profile', 'execution', 'old432_initializer', 'wrong_producer',
    'receipt_432', 'receipt_original', 'thin_metadata', 'old_input_collision', 'prefix', 'append', 'bool_take',
    'bool_update', 'cursor', 'endpoint', 'alias', 'extra_update', 'baseline_forward', 'autograd', 'bool_budget',
    'bridge', 'source', 'old001_source', 'old001_open', 'old001_boolean_count', 'old001_failed', 'backup_missing',
    'backup_unverified', 'backup_coverage', 'flags', 'bool_gate', 'spent', 'snapshot', 'existing', 'symlink',
    'argv', 'python', 'hardware_env', 'legacy_env', 'unknown_env'))
def test_002_rejects_wrong_lineage_scope_backup_or_authority(monkeypatch, tmp_path, bad):
    f = fixture(monkeypatch, tmp_path, bad=bad)
    with pytest.raises(RuntimeError): f['call']()
    assert not f['called'] and not (f['output'] / 'CLAIM.json').exists()


def test_002_preflight_rejects_boolean_zero_gate(monkeypatch, tmp_path):
    f = fixture(monkeypatch, tmp_path, preflight=True, bad='bool_gate')
    with pytest.raises(RuntimeError, match='EXACT_ONE_USE_GATE'): f['call']()
    assert not f['called']


def test_002_shell_targets_actual_new_module_and_unchanged_caps(monkeypatch, tmp_path):
    f = fixture(monkeypatch, tmp_path)
    variables, functions, dispatch, _ = [v[2] for v in g._V3_OBSERVED_TRAIN_CONTINUATION_002_SHELL_ADDITIONS]
    functions = functions.replace('/home/andre2/GX1_RUNS', str(f['runs']))
    state = dict(REPO_ROOT=str(f['repo']), CANONICAL_TRAINER_PYTHON=str(f['repo'] / '.venv/bin/python'),
        JOB_CLASS='trainer', MEM='20G', SWAP='512M', ATTENDED_SMOKE='false', CUDA_PRODUCER_GUARD='false',
        POWER_BENCHMARK_SCOPE_JSON='', POWER_BENCHMARK_SCOPE_SHA256='')
    script = 'set -eu\n' + ''.join(k + '=' + shlex.quote(v) + '\n' for k, v in state.items()) + variables + functions
    script += 'validate_target_command() {\n' + dispatch + 'return 88\n}\n'
    for module, extra, expected in (('native_future_state_observed_train_continuation_002_v1', [], 0),
            ('native_future_state_observed_train_continuation_002_v1', ['--preflight'], 0),
            ('native_future_state_observed_train_continuation_v1', [], 75),
            ('native_future_state_observed_train_continuation_002_v1', ['--lane', 'fit'], 75)):
        argv = [state['CANONICAL_TRAINER_PYTHON'], '-m', 'gx1.audit.' + module,
            '--v3-observed-train-continuation-002-plan-sha256', f['digest']] + extra
        tail = 'validate_target_command ' + ' '.join(map(shlex.quote, argv))
        tail += '\nprintf "%s,%s,%s,%s\\n" "$NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_002" "$TRAINER_MAX_WALL_SECONDS" "$TRAINER_MODEL_MAX_WALL_SECONDS" "$TRAINER_OUT_BUNDLE_DIR"\n'
        result = subprocess.run(['bash', '-c', script + tail], cwd=f['repo'], capture_output=True, text=True)
        assert result.returncode == expected, result.stderr
        if expected == 0: assert result.stdout.strip() == 'true,1800,1770,' + str(f['output'])


def _guard_hashes():
    repo = Path(g.__file__).resolve().parents[2]
    return {n: hashlib.sha256(subprocess.check_output(['git', 'show', g._V3_OBSERVED_TRAIN_CONTINUATION_002_PARENT + ':' + n], cwd=repo)).hexdigest() for n in OWNERS}


def test_002_bridge_keeps_all_old_guards_001_adapter_and_seven_NN_owners():
    proof = g.native_v3_observed_train_continuation_002_guard_source_bridge(_guard_hashes())
    assert proof['old_guard_routes_preserved'] and proof['old_numeric_resource_environment_function_unchanged']
    assert len(proof['old_NN_and_001_source_byte_unchanged']) == 7


@pytest.mark.parametrize('bad', ('producer_hash', 'legacy_limit', 'route_limit', '001_adapter'))
def test_002_bridge_rejects_unreviewed_changes(monkeypatch, bad):
    hashes = _guard_hashes(); repo = Path(g.__file__).resolve().parents[2]
    if bad == 'producer_hash': hashes[OWNERS[0]] = '0' * 64
    elif bad == '001_adapter':
        read = Path.read_bytes; target = repo / 'gx1/audit/native_future_state_observed_train_continuation_v1.py'
        monkeypatch.setattr(Path, 'read_bytes', lambda p, *a, **k: read(p, *a, **k) + (b'\n# mutation\n' if p == target else b''))
    else:
        read = Path.read_text
        def changed(path, *a, **kw):
            text = read(path, *a, **kw)
            if path == repo / OWNERS[0] and bad == 'legacy_limit': text = text.replace('_MAX_MEMORY_BYTES = 20 * 1024**3', '_MAX_MEMORY_BYTES = 21 * 1024**3', 1)
            if path == repo / OWNERS[1] and bad == 'route_limit':
                fragment = g._V3_OBSERVED_TRAIN_CONTINUATION_002_SHELL_ADDITIONS[1][2]
                text = text.replace(fragment, fragment.replace('TRAINER_GPU_MAX_POWER_DRAW_W=250', 'TRAINER_GPU_MAX_POWER_DRAW_W=251'), 1)
            return text
        monkeypatch.setattr(Path, 'read_text', changed)
    with pytest.raises(RuntimeError): g.native_v3_observed_train_continuation_002_guard_source_bridge(hashes)
