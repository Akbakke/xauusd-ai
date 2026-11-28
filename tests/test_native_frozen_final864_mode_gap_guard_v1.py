"""Frozen864 read-only admission tests; synthetic metadata, no NN/payload decode."""
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

_spec = importlib.util.spec_from_file_location('gx1_passing_observed_002_guard_fixtures',
    Path(__file__).with_name('test_native_observed_train_continuation_002_guard_v1.py'))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
OWNERS = _base.OWNERS
NEW = ('gx1/audit/native_future_state_frozen_final864_mode_gap_v1.py',
    'tests/test_native_future_state_frozen_final864_mode_gap_v1.py',
    'tests/test_native_frozen_final864_mode_gap_guard_v1.py')
FLAGS = ('training_enabled', 'native_training_enabled', 'native_launch_authorized',
    'full_epoch_training_allowed', 'full_val_allowed', 'exit_training_and_evaluation_this_wave')


def fixture(monkeypatch, tmp_path, preflight=False, bad=None):
    f = _base.fixture(monkeypatch, tmp_path, preflight=True)
    repo, runs = f['repo'], f['runs']; parent_plan = copy.deepcopy(f['plan'])
    parent_plan['metadata']['trajectory']['contract_sha256'] = 'c' * 64
    original_ref = parent_plan['original_producer_plan']
    original = copy.deepcopy(g._native_v3_development_read(original_ref['path'], original_ref['sha256']))
    parent_ref = dict(path=str(repo / 'configs/research' / (g._V3_OBSERVED_TRAIN_CONTINUATION_002_NAME + '.json')),
        sha256='be53cb01fce7c80f8ba120c88de2f36d299112f20a573e85b85cda82d8db6875')
    oldroot = runs / g._V3_OBSERVED_TRAIN_CONTINUATION_002_NAME
    receipt_ref = dict(path=str(oldroot / 'FINAL864.json'), sha256='06b48e4fa08306f1979d201c334573adaf80f9f6d9f4b05dfd220c15f69a605d')
    checkpoint_ref = dict(path=str(oldroot / 'FINAL864.pt'), sha256='26bbcc969348592b5cdea4ee85fe5629d9ed6250bd1aa9415e570fdc46250368')
    receipt = dict(logical_cursor=864, optimizer_updates=864, producer_lane='continuation',
        original_plan=parent_ref, execution_plan=parent_ref, checkpoint=checkpoint_ref,
        trajectory_sha256=parent_plan['metadata']['trajectory']['contract_sha256'])
    owners = dict(parent_plan['source_owners']); anchored = {n: (repo / n).read_bytes() for n in owners}
    for name in NEW:
        p = repo / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# synthetic frozen864 fixture\n')
        owners[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    def command(argv, **kwargs):
        return anchored[argv[2].split(':', 1)[1]] if argv[:2] == ['git', 'show'] else 'a' * 40 + '\n'
    monkeypatch.setattr(subprocess, 'check_output', command)
    evidence = runs / 'GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001'
    design_ref = dict(path=str(evidence / 'V3_FROZEN_FINAL864_MODE_GAP_DESIGN_20261128_001.json'),
        sha256='3b49a1de3bb2bc257e157d9fd8de137b57121d94fe30b76c75edac0b7519c097')
    limits = dict.fromkeys(parent_plan['limits'], 0)
    limits.update(model_constructions=1, optimizer_constructions=1, feature_cache_payload_loads=1,
        target_archive_loads=1, checkpoint_loads=1, model_restores=1, model_forwards=405, backbone_forwards=405,
        readout_forwards=405, logit_archive_writes=5, logit_archive_readbacks=5, saved_eval_logit_loads=1)
    passes = [dict(label='EVAL_PARITY', training=False, equal_tf=True)] + [
        dict(label='TRAIN_DRAW_' + str(i).zfill(2), training=True, equal_tf=False) for i in range(1, 5)]
    saved = {k: dict(path=str(oldroot / 'FINAL864' / v), sha256=str(i) * 64)
        for i, (k, v) in enumerate((('saved_eval_logits', 'PASS2_LOGITS.npz'),
        ('saved_eval_numeric_owner_score', 'PASS2_SCORE.json'), ('saved_eval_score_wrapper', 'WRAPPER.json')), 1)}
    design = dict(limits=copy.deepcopy(limits), passes=copy.deepcopy(passes), **saved)
    inputs = copy.deepcopy(parent_plan['inputs'])
    inputs.update({'frozen_final864_mode_gap_' + k: v for k, v in saved.items()})
    bridge = dict(producer_commit=g._V3_FROZEN_FINAL864_MODE_GAP_PARENT,
        owners={n: dict(original_sha256='f' * 64, current_sha256=owners[n]) for n in OWNERS})
    output = runs / g._V3_FROZEN_FINAL864_MODE_GAP_NAME
    plan = dict(schema_version='gx1_native_v3_frozen_final864_mode_gap_plan_v1', stage=g._V3_FROZEN_FINAL864_MODE_GAP_STAGE,
        source_commit='a' * 40, output_root=str(output), execution=g._native_v3_development_execution('warm'),
        execution_profile=g._V3_FROZEN_FINAL864_MODE_GAP_PROFILE, design=design_ref, producer_plan=parent_ref,
        original_producer_plan=original_ref, authentic_final_receipt=receipt_ref, authentic_final_checkpoint=checkpoint_ref,
        metadata=dict(parent_plan=parent_ref, original_plan=original_ref, receipt=copy.deepcopy(receipt),
            trajectory=copy.deepcopy(parent_plan['metadata']['trajectory'])),
        recipe=copy.deepcopy(parent_plan['recipe']), model_contract=copy.deepcopy(parent_plan['model_contract']),
        physical_input_identity=copy.deepcopy(parent_plan['physical_input_identity']), backbone_gradient_policy=None,
        actual_global_cursor=864, numeric_owner_phase='PASS2', passes=passes, limits=limits,
        inputs=inputs, source_owners=owners, guard_source_bridge=bridge)
    if bad == 'schema': plan['schema_version'] = 'old'
    if bad == 'profile': plan['execution_profile'] = 'generic'
    if bad == 'execution': plan['execution']['GPU_draw_stop_W'] = 251
    if bad == 'old648_initializer': plan['authentic_final_checkpoint'] = parent_plan['authentic_final_checkpoint']
    if bad == 'receipt_cursor': receipt['logical_cursor'] = 648
    if bad == 'receipt_producer': receipt['original_plan'] = parent_plan['producer_plan']
    if bad == 'trajectory': plan['metadata']['trajectory'] = {}
    if bad == 'model': plan['model_contract'] = {}
    if bad == 'recipe': plan['recipe'] = {}
    if bad == 'cursor': plan['actual_global_cursor'] = 648
    if bad == 'phase': plan['numeric_owner_phase'] = 'FINAL864'
    if bad == 'extra_draw': passes.append(copy.deepcopy(passes[-1]))
    if bad == 'integer_mode': passes[1]['training'] = 1
    if bad in ('optimizer_steps', 'backward_calls', 'autograd_calls', 'checkpoint_writes', 'normalization_fits', 'prior_fits'):
        limits[bad] = 1
    if bad == 'extra_forward': limits['model_forwards'] += 1
    if bad == 'extra_checkpoint_load': limits['checkpoint_loads'] += 1
    if bad == 'bool_budget': limits['model_constructions'] = True
    if bad == 'old_input_collision': inputs['recovery_design'] = design_ref
    if bad == 'saved_logits': inputs['frozen_final864_mode_gap_saved_eval_logits'] = {}
    if bad == 'bridge': plan['guard_source_bridge'] = {**bridge, 'extra': True}
    if bad == 'source': owners[NEW[0]] = '0' * 64
    if bad == 'old_loop_source':
        p = repo / 'gx1/audit/native_provider_hlc_learning_v1.py'; p.write_bytes(p.read_bytes() + b'# mutation\n')
        owners[str(p.relative_to(repo))] = hashlib.sha256(p.read_bytes()).hexdigest()
    path = repo / 'configs/research' / (g._V3_FROZEN_FINAL864_MODE_GAP_NAME + '.json')
    path.write_text(json.dumps(plan)); digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = evidence / 'SYNTHETIC_FINAL864_MANIFEST.json'
    manifest_path.write_text(json.dumps({'files': [receipt_ref] if bad == 'backup_coverage' else [receipt_ref, checkpoint_ref]}))
    manifest_ref = dict(path=str(manifest_path), sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    backup_path = evidence / 'SYNTHETIC_FINAL864_BACKUP.json'
    backup_path.write_text(json.dumps(dict(F_FULLFSYNC=bad != 'backup_unverified', separate_mac_disk=True,
        independent_member_readback=True, manifest=manifest_ref, manifest_sha256=manifest_ref['sha256'])))
    backup_ref = dict(path=str(backup_path), sha256=hashlib.sha256(backup_path.read_bytes()).hexdigest())
    pp = repo / 'NEXT_RUN_POLICY.json'; policy = json.loads(pp.read_text())
    for flag in FLAGS: policy[flag] = False
    previous = policy[g._V3_OBSERVED_TRAIN_CONTINUATION_002_STAGE]
    previous.update(status='CLOSED_CONSUMED_COMPLETE', plan=parent_ref, execution_authorized=False,
        training_authorized=False, remaining_invocations=0, relaunch_allowed=False, final_backup=backup_ref)
    if bad == 'old002_open': previous['execution_authorized'] = True
    if bad == 'old002_failed': previous['status'] = 'FAILED_CONSUMED'
    if bad == 'backup_missing': previous.pop('final_backup')
    parent = dict(status='PREPARED_CLOSED_NOT_ADMITTED' if preflight else 'ADMITTED_EXACT_V3_FROZEN_FINAL864_MODE_GAP',
        source_preparation_authorized=True, design=design_ref, plan=dict(path=str(path), sha256=digest),
        execution_authorized=not preflight, training_authorized=False, remaining_invocations=int(not preflight),
        relaunch_allowed=False, test_authorized=False, spending_authorized=False)
    policy[g._V3_FROZEN_FINAL864_MODE_GAP_STAGE] = parent
    if bad == 'training_authority': parent['training_authorized'] = True
    if bad == 'bool_gate': parent['remaining_invocations'] = not preflight
    if bad == 'spent': parent['remaining_invocations'] = 0
    pp.write_text(json.dumps(policy))
    if not preflight:
        (evidence / 'V3_FROZEN_FINAL864_MODE_GAP_ADMITTED_POLICY_20261128_001.json').write_bytes(b'{}' if bad == 'snapshot' else pp.read_bytes())
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
    monkeypatch.setattr(g, 'native_v3_frozen_final864_mode_gap_guard_source_bridge', lambda *a, **k: bridge)
    argv = [str(repo / NEW[0]), '--v3-frozen-final864-mode-gap-plan-sha256', digest] + (['--preflight'] if preflight else [])
    if bad == 'argv': argv += ['--lane', 'fit']
    monkeypatch.setattr(sys, 'argv', argv)
    if bad == 'python': monkeypatch.setattr(sys, 'executable', '/usr/bin/python3')
    for key in list(os.environ):
        if key.startswith('GX1_NATIVE_'): monkeypatch.delenv(key)
    env = dict(GX1_NATIVE_V3_FROZEN_FINAL864_MODE_GAP='true', GX1_NATIVE_V3_FROZEN_FINAL864_MODE_GAP_PLAN_SHA256=digest)
    if bad == 'hardware_env': env['GX1_TRAINER_GPU_MAX_POWER_DRAW_W'] = '251'
    if bad == 'legacy_env': env['GX1_NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_002'] = 'true'
    if bad == 'unknown_env': env['GX1_NATIVE_V3_FROZEN_FINAL864_MODE_GAP_EXTRA'] = 'false'
    for key, value in env.items(): monkeypatch.setenv(key, value)
    return dict(f, path=path, digest=digest, plan=plan, output=output,
        call=lambda: g.require_native_v3_frozen_final864_mode_gap_execution(plan_sha256=digest))


@pytest.mark.parametrize('preflight', (False, True))
def test_frozen864_gate_is_read_only_even_when_execution_admitted(monkeypatch, tmp_path, preflight):
    f = fixture(monkeypatch, tmp_path, preflight); proof = f['call']()
    assert proof['preflight'] is preflight
    assert proof['training_authority_granted'] is proof['optimizer_authority_granted'] is False
    assert f['called'] == ['cuda'] and not f['output'].exists()


@pytest.mark.parametrize('preflight', (False, True))
@pytest.mark.parametrize('flag', FLAGS)
def test_frozen864_every_global_flag_must_stay_false(monkeypatch, tmp_path, preflight, flag):
    f = fixture(monkeypatch, tmp_path, preflight)
    pp = f['repo'] / 'NEXT_RUN_POLICY.json'; policy = json.loads(pp.read_text()); policy[flag] = True
    pp.write_text(json.dumps(policy))
    with pytest.raises(RuntimeError, match='ALL_SIX_GLOBAL_FLAGS_FALSE'): f['call']()
    assert not f['called']


@pytest.mark.parametrize('bad', ('schema', 'profile', 'execution', 'old648_initializer', 'receipt_cursor', 'receipt_producer',
    'trajectory', 'model', 'recipe', 'cursor', 'phase', 'extra_draw', 'integer_mode', 'optimizer_steps', 'backward_calls',
    'autograd_calls', 'checkpoint_writes', 'normalization_fits', 'prior_fits', 'extra_forward', 'extra_checkpoint_load',
    'bool_budget', 'old_input_collision', 'saved_logits', 'bridge', 'source', 'old_loop_source', 'old002_open',
    'old002_failed', 'backup_missing', 'backup_coverage', 'backup_unverified', 'training_authority', 'bool_gate', 'spent',
    'snapshot', 'existing', 'symlink', 'argv', 'python', 'hardware_env', 'legacy_env', 'unknown_env'))
def test_frozen864_rejects_updates_wrong_lineage_or_unqualified_execution(monkeypatch, tmp_path, bad):
    f = fixture(monkeypatch, tmp_path, bad=bad)
    with pytest.raises(RuntimeError): f['call']()
    assert not f['called'] and not (f['output'] / 'CLAIM.json').exists()


def test_frozen864_shell_routes_only_actual_new_module(monkeypatch, tmp_path):
    f = fixture(monkeypatch, tmp_path)
    variables, functions, dispatch, _ = [v[2] for v in g._V3_FROZEN_FINAL864_MODE_GAP_SHELL_ADDITIONS]
    functions = functions.replace('/home/andre2/GX1_RUNS', str(f['runs']))
    state = dict(REPO_ROOT=str(f['repo']), CANONICAL_TRAINER_PYTHON=str(f['repo'] / '.venv/bin/python'),
        JOB_CLASS='trainer', MEM='20G', SWAP='512M', ATTENDED_SMOKE='false', CUDA_PRODUCER_GUARD='false',
        POWER_BENCHMARK_SCOPE_JSON='', POWER_BENCHMARK_SCOPE_SHA256='')
    script = 'set -eu\n' + ''.join(k + '=' + shlex.quote(v) + '\n' for k, v in state.items()) + variables + functions
    script += 'validate_target_command() {\n' + dispatch + 'return 88\n}\n'
    for module, extra, expected in (('native_future_state_frozen_final864_mode_gap_v1', [], 0),
            ('native_future_state_frozen_final864_mode_gap_v1', ['--preflight'], 0),
            ('native_future_state_observed_train_continuation_002_v1', [], 75),
            ('native_future_state_frozen_final864_mode_gap_v1', ['--lane', 'fit'], 75)):
        argv = [state['CANONICAL_TRAINER_PYTHON'], '-m', 'gx1.audit.' + module,
            '--v3-frozen-final864-mode-gap-plan-sha256', f['digest']] + extra
        tail = 'validate_target_command ' + ' '.join(map(shlex.quote, argv))
        tail += '\nprintf "%s,%s,%s,%s\\n" "$NATIVE_V3_FROZEN_FINAL864_MODE_GAP" "$TRAINER_MAX_WALL_SECONDS" "$TRAINER_MODEL_MAX_WALL_SECONDS" "$TRAINER_OUT_BUNDLE_DIR"\n'
        result = subprocess.run(['bash', '-c', script + tail], cwd=f['repo'], capture_output=True, text=True)
        assert result.returncode == expected, result.stderr
        if expected == 0: assert result.stdout.strip() == 'true,1800,1770,' + str(f['output'])


def _guard_hashes():
    repo = Path(g.__file__).resolve().parents[2]
    return {n: hashlib.sha256(subprocess.check_output(['git', 'show', g._V3_FROZEN_FINAL864_MODE_GAP_PARENT + ':' + n], cwd=repo)).hexdigest() for n in OWNERS}


def test_frozen864_bridge_keeps_all_old_guards_001_002_and_mode_gap_loop():
    proof = g.native_v3_frozen_final864_mode_gap_guard_source_bridge(_guard_hashes())
    assert proof['old_guard_routes_preserved'] and proof['old_numeric_resource_environment_function_unchanged']
    assert len(proof['old_NN_and_mode_gap_loop_byte_unchanged']) == 9


@pytest.mark.parametrize('bad', ('producer_hash', 'legacy_limit', 'route_limit', 'mode_gap_loop'))
def test_frozen864_bridge_rejects_unreviewed_changes(monkeypatch, bad):
    hashes = _guard_hashes(); repo = Path(g.__file__).resolve().parents[2]
    if bad == 'producer_hash': hashes[OWNERS[0]] = '0' * 64
    elif bad == 'mode_gap_loop':
        read = Path.read_bytes; target = repo / 'gx1/audit/native_provider_hlc_learning_v1.py'
        monkeypatch.setattr(Path, 'read_bytes', lambda p, *a, **k: read(p, *a, **k) + (b'\n# mutation\n' if p == target else b''))
    else:
        read = Path.read_text
        def changed(path, *a, **kw):
            text = read(path, *a, **kw)
            if path == repo / OWNERS[0] and bad == 'legacy_limit': text = text.replace('_MAX_MEMORY_BYTES = 20 * 1024**3', '_MAX_MEMORY_BYTES = 21 * 1024**3', 1)
            if path == repo / OWNERS[1] and bad == 'route_limit':
                fragment = g._V3_FROZEN_FINAL864_MODE_GAP_SHELL_ADDITIONS[1][2]
                text = text.replace(fragment, fragment.replace('TRAINER_GPU_MAX_POWER_DRAW_W=250', 'TRAINER_GPU_MAX_POWER_DRAW_W=251'), 1)
            return text
        monkeypatch.setattr(Path, 'read_text', changed)
    with pytest.raises(RuntimeError): g.native_v3_frozen_final864_mode_gap_guard_source_bridge(hashes)
