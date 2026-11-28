"""Source-only exact admission/containment tests; no NN or private payload decoding."""
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
_fixture_spec = importlib.util.spec_from_file_location('gx1_existing_v3_guard_fixtures',
    Path(__file__).with_name('test_native_all_tf_provider_hlc_v3_development_guard_v1.py'))
_fixture_module = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture_module)
actual_step_observer_cold_recovery_fixture = _fixture_module.actual_step_observer_cold_recovery_fixture
OWNERS = _fixture_module.OWNERS

NEW = ('gx1/audit/native_future_state_observed_train_continuation_v1.py',
    'tests/test_native_future_state_observed_train_continuation_v1.py',
    'tests/test_native_observed_train_continuation_guard_v1.py')


def fixture(monkeypatch, tmp_path, preflight=False, bad=None):
    f = actual_step_observer_cold_recovery_fixture(monkeypatch, tmp_path, preflight=True)
    repo, runs = f['repo'], f['runs']; producer = copy.deepcopy(f['plan']['metadata']['producer'])
    producer['inputs'] = {'recovery_design': {'path': '/original/design.json', 'sha256': '1' * 64}}
    # The reused fixture has complete structural vectors without a sealed identity.
    producer['schedule']['contract_sha256'] = hashlib.sha256(json.dumps(producer['schedule'], sort_keys=True).encode()).hexdigest()
    producer['metadata']['trajectory']['schedule'] = producer['schedule']
    producer_ref = f['plan']['producer_plan']; receipt_ref = f['plan']['authentic_final_receipt']
    receipt = copy.deepcopy(g._native_v3_development_read(receipt_ref['path'], receipt_ref['sha256']))
    checkpoint_ref = f['plan']['authentic_final_checkpoint']
    owners = dict(f['plan']['source_owners'])
    for name in NEW:
        p = repo / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# synthetic source fixture\n')
        owners[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    anchored = {n: (repo / n).read_bytes() for n in owners if n not in NEW}
    def command(argv, **kwargs):
        return anchored[argv[2].split(':', 1)[1]] if argv[:2] == ['git', 'show'] else 'a' * 40 + '\n'
    monkeypatch.setattr(subprocess, 'check_output', command)
    old = producer['schedule']; schedule = copy.deepcopy(old); schedule.pop('contract_sha256')
    schedule.update(schema_version='gx1_native_v3_observed_train_continuation_schedule_v1', logical_batches=648,
        optimizer_updates=648, passes=3, skipped_batches=0, original_prefix_schedule_sha256=old['contract_sha256'],
        original_take_indices_appended=list(range(216, 432)), order_mode='repeat_locked_original_second_pass_once')
    for key in ('cohort_takes', 'logical_step_rows', 'cohort_take_state_sha256_by_logical_batch', 'update_flags'):
        schedule[key] = copy.deepcopy(old[key] + old[key][216:432])
    for key in ('raw_permutation_state_sha256_by_pass', 'actual_sorted_partition_stream_state_sha256_by_pass'):
        schedule[key] = old[key] + [old[key][1]]
    schedule['contract_sha256'] = '7' * 64
    evidence = runs / 'GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001'
    design_ref = dict(path=str(evidence / 'V3_OBSERVED_TRAIN_CONTINUATION_DESIGN_20261128_001.json'),
        sha256='5b04f35e2765c45c2cf331cada40f0d96497b1561e69183add7a38be6a5eb269')
    review = dict(path=str(evidence / 'OBSERVER_REVIEW.json'), sha256='5' * 64)
    alias = dict(actual_before='BASELINE432', actual_candidate='FINAL648', numeric_owner_phase='PASS2')
    design = dict(phase_alias=alias, basis=review)
    limits = dict.fromkeys(producer['limits'], 0)
    limits.update(model_constructions=1, optimizer_constructions=1, feature_cache_payload_loads=1,
        target_archive_loads=1, checkpoint_loads=1, model_restores=1, model_forwards=297, backbone_forwards=297,
        readout_forwards=297, backward_calls=216, optimizer_steps=216, observer_records=216,
        checkpoint_writes=2, checkpoint_readbacks=2, checkpoint_prepublication_loads=2,
        checkpoint_receipt_writes=2, logit_archive_writes=1, logit_archive_readbacks=1)
    bridge = dict(producer_commit=g._V3_OBSERVED_TRAIN_CONTINUATION_PARENT,
        owners={n: dict(original_sha256='f' * 64, current_sha256=owners[n]) for n in OWNERS})
    output = runs / g._V3_OBSERVED_TRAIN_CONTINUATION_NAME
    plan = dict(schema_version='gx1_native_v3_observed_train_continuation_plan_v1',
        stage=g._V3_OBSERVED_TRAIN_CONTINUATION_STAGE, source_commit='a' * 40, output_root=str(output),
        execution=g._native_v3_development_execution('warm'), execution_profile=g._V3_OBSERVED_TRAIN_CONTINUATION_PROFILE,
        design=design_ref, producer_plan=producer_ref, authentic_final_receipt=receipt_ref,
        authentic_final_checkpoint=checkpoint_ref, recipe=producer['recipe'], model_contract=producer['model_contract'],
        physical_input_identity=producer.get('physical_input_identity'), backbone_gradient_policy=None,
        metadata=dict(original=producer['metadata']['original'], producer=producer, receipt=receipt,
            trajectory={'schedule': schedule}), schedule=schedule, starting_cursor=432, ending_cursor=648,
        checkpoint_cursors=[433, 648], observer_enabled=True, phase_alias=alias,
        limits=limits, inputs=copy.deepcopy(producer['inputs']), source_owners=owners, guard_source_bridge=bridge)
    if bad == 'schema': plan['schema_version'] = 'old'
    if bad == 'profile': plan['execution_profile'] = 'generic'
    if bad == 'execution': plan['execution']['GPU_draw_stop_W'] = 251
    if bad == 'technical_initializer': plan['authentic_final_checkpoint'] = f['plan']['inputs']['reference_checkpoint']
    if bad == 'old_input_collision': plan['inputs']['recovery_design'] = design_ref
    if bad == 'model': plan['model_contract'] = {}
    if bad == 'recipe': plan['recipe'] = {}
    if bad == 'cursor': plan['starting_cursor'] = 434
    if bad == 'endpoint': plan['ending_cursor'] = 649
    if bad == 'observer': plan['observer_enabled'] = False
    if bad == 'phase_alias': plan['phase_alias'] = dict(alias, actual_candidate='PASS2')
    if bad == 'order': schedule['cohort_takes'][432][0] += 1
    if bad == 'prefix': schedule['cohort_takes'][0][0] += 1
    if bad == 'bool_take': schedule['cohort_takes'][0][0] = False
    if bad == 'bool_update': schedule['update_flags'][432] = 1
    if bad == 'wrong_hash_tail': schedule['cohort_take_state_sha256_by_logical_batch'][432] = '0' * 64
    if bad == 'new_shuffle': schedule['raw_permutation_state_sha256_by_pass'][2] = '0' * 64
    if bad == 'updates': limits['optimizer_steps'] += 1
    if bad == 'forwards': limits['model_forwards'] += 1
    if bad == 'baseline_forward': limits['model_forwards'] += 81
    if bad == 'autograd': limits['autograd_calls'] = 1
    if bad == 'prior_fit': limits['prior_fits'] = 1
    if bad == 'records': limits['observer_records'] = 215
    if bad == 'bool_budget': limits['model_constructions'] = True
    if bad == 'bridge': plan['guard_source_bridge'] = {**bridge, 'extra': True}
    if bad == 'source': plan['source_owners'][NEW[0]] = '0' * 64
    if bad == 'old_source':
        p = repo / g._V3_ACTUAL_STEP_OBSERVER_CORE; p.write_bytes(p.read_bytes() + b'# changed\n')
        owners[g._V3_ACTUAL_STEP_OBSERVER_CORE] = hashlib.sha256(p.read_bytes()).hexdigest()
    path = repo / 'configs/research' / (g._V3_OBSERVED_TRAIN_CONTINUATION_NAME + '.json')
    path.write_text(json.dumps(plan)); digest = hashlib.sha256(path.read_bytes()).hexdigest()
    pp = repo / 'NEXT_RUN_POLICY.json'; policy = json.loads(pp.read_text())
    for flag in ('training_enabled', 'native_training_enabled'): policy[flag] = not preflight
    recovery = policy[g._V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_STAGE]
    recovery.update(status='CLOSED_CONSUMED_COMPLETE', actual_review=review)
    parent = dict(status='PREPARED_CLOSED_NOT_ADMITTED' if preflight else 'ADMITTED_EXACT_V3_OBSERVED_TRAIN_CONTINUATION',
        source_preparation_authorized=True, design=design_ref, plan=dict(path=str(path), sha256=digest),
        execution_authorized=not preflight, training_authorized=not preflight, remaining_invocations=int(not preflight),
        relaunch_allowed=False, test_authorized=False, spending_authorized=False)
    policy[g._V3_OBSERVED_TRAIN_CONTINUATION_STAGE] = parent
    if bad == 'flags': policy['native_launch_authorized'] = True
    if bad == 'old_open': recovery['execution_authorized'] = True
    if bad == 'old_not_qualified': recovery['status'] = 'FAILED_CONSUMED'
    if bad == 'old_bool_gate': recovery['remaining_invocations'] = False
    if bad == 'bool_gate': parent['remaining_invocations'] = not preflight
    if bad == 'spent': parent['remaining_invocations'] = 0
    pp.write_text(json.dumps(policy))
    if not preflight:
        (evidence / 'V3_OBSERVED_TRAIN_CONTINUATION_ADMITTED_POLICY_20261128_001.json').write_bytes(b'{}' if bad == 'snapshot' else pp.read_bytes())
    if bad == 'existing': output.mkdir()
    if bad == 'symlink':
        other = path.with_suffix('.original'); path.rename(other); path.symlink_to(other)
    read = g._native_v3_development_read
    def adapted(candidate, candidate_digest=None):
        for binding, value in ((producer_ref, producer), (receipt_ref, receipt), (design_ref, design)):
            if str(candidate) == binding['path']:
                assert candidate_digest == binding['sha256']; return value
        return read(candidate, candidate_digest)
    monkeypatch.setattr(g, '_native_v3_development_read', adapted)
    monkeypatch.setattr(g, 'native_v3_observed_train_continuation_guard_source_bridge', lambda *a, **k: bridge)
    argv = [str(repo / NEW[0]), '--v3-observed-train-continuation-plan-sha256', digest] + (['--preflight'] if preflight else [])
    if bad == 'argv': argv += ['--lane', 'fit']
    monkeypatch.setattr(sys, 'argv', argv)
    if bad == 'python': monkeypatch.setattr(sys, 'executable', '/usr/bin/python3')
    for key in list(os.environ):
        if key.startswith('GX1_NATIVE_'): monkeypatch.delenv(key)
    env = dict(GX1_NATIVE_V3_OBSERVED_TRAIN_CONTINUATION='true', GX1_NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_PLAN_SHA256=digest)
    if bad == 'hardware_env': env['GX1_TRAINER_GPU_MAX_POWER_DRAW_W'] = '251'
    if bad == 'legacy_env': env['GX1_NATIVE_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY'] = 'true'
    if bad == 'unknown_env': env['GX1_NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_EXTRA'] = 'false'
    for key, value in env.items(): monkeypatch.setenv(key, value)
    return dict(f, path=path, digest=digest, plan=plan, output=output,
        call=lambda: g.require_native_v3_observed_train_continuation_execution(plan_sha256=digest))


@pytest.mark.parametrize('preflight', (False, True))
def test_continuation_exact_bound_role_has_resource_authority_only(monkeypatch, tmp_path, preflight):
    f = fixture(monkeypatch, tmp_path, preflight); proof = f['call']()
    assert proof['preflight'] is preflight
    assert proof['training_authority_granted'] is proof['optimizer_authority_granted'] is False
    assert f['called'] == ['cuda'] and not f['output'].exists()


@pytest.mark.parametrize('bad', ('schema', 'profile', 'execution', 'technical_initializer', 'old_input_collision',
    'model', 'recipe', 'cursor', 'endpoint', 'observer', 'phase_alias', 'order', 'prefix', 'bool_take', 'bool_update',
    'wrong_hash_tail', 'new_shuffle', 'updates', 'forwards', 'baseline_forward', 'autograd', 'prior_fit', 'records',
    'bool_budget', 'bridge', 'source', 'old_source', 'flags', 'old_open', 'old_not_qualified', 'old_bool_gate',
    'bool_gate', 'spent', 'snapshot', 'existing', 'symlink', 'argv', 'python', 'hardware_env', 'legacy_env', 'unknown_env'))
def test_continuation_rejects_unqualified_or_modified_scope(monkeypatch, tmp_path, bad):
    f = fixture(monkeypatch, tmp_path, bad=bad)
    with pytest.raises(RuntimeError): f['call']()
    assert not f['called'] and not (f['output'] / 'CLAIM.json').exists()


def test_continuation_preflight_rejects_boolean_zero_gate(monkeypatch, tmp_path):
    f = fixture(monkeypatch, tmp_path, preflight=True, bad='bool_gate')
    with pytest.raises(RuntimeError, match='EXACT_ONE_USE_GATE'): f['call']()
    assert not f['called']


def test_continuation_shell_routes_only_the_actual_new_module_with_exact_caps(monkeypatch, tmp_path):
    f = fixture(monkeypatch, tmp_path)
    variables, functions, dispatch, _ = [v[2] for v in g._V3_OBSERVED_TRAIN_CONTINUATION_SHELL_ADDITIONS]
    functions = functions.replace('/home/andre2/GX1_RUNS', str(f['runs']))
    state = dict(REPO_ROOT=str(f['repo']), CANONICAL_TRAINER_PYTHON=str(f['repo'] / '.venv/bin/python'),
        JOB_CLASS='trainer', MEM='20G', SWAP='512M', ATTENDED_SMOKE='false', CUDA_PRODUCER_GUARD='false',
        POWER_BENCHMARK_SCOPE_JSON='', POWER_BENCHMARK_SCOPE_SHA256='')
    script = 'set -eu\n' + ''.join(k + '=' + shlex.quote(v) + '\n' for k, v in state.items()) + variables + functions
    script += 'validate_target_command() {\n' + dispatch + 'return 88\n}\n'
    for module, extra, expected in (('native_future_state_observed_train_continuation_v1', [], 0),
            ('native_future_state_observed_train_continuation_v1', ['--preflight'], 0),
            ('native_future_state_step_observer_cold_recovery_v1', [], 75),
            ('native_future_state_observed_train_continuation_v1', ['--lane', 'fit'], 75)):
        argv = [state['CANONICAL_TRAINER_PYTHON'], '-m', 'gx1.audit.' + module,
            '--v3-observed-train-continuation-plan-sha256', f['digest']] + extra
        tail = 'validate_target_command ' + ' '.join(map(shlex.quote, argv))
        tail += '\nprintf "%s,%s,%s,%s\\n" "$NATIVE_V3_OBSERVED_TRAIN_CONTINUATION" "$TRAINER_MAX_WALL_SECONDS" "$TRAINER_MODEL_MAX_WALL_SECONDS" "$TRAINER_OUT_BUNDLE_DIR"\n'
        result = subprocess.run(['bash', '-c', script + tail], cwd=f['repo'], capture_output=True, text=True)
        assert result.returncode == expected, result.stderr
        if expected == 0: assert result.stdout.strip() == 'true,1800,1770,' + str(f['output'])


def _guard_hashes():
    repo = Path(g.__file__).resolve().parents[2]
    return {n: hashlib.sha256(subprocess.check_output(['git', 'show', g._V3_OBSERVED_TRAIN_CONTINUATION_PARENT + ':' + n], cwd=repo)).hexdigest() for n in OWNERS}


def test_continuation_bridge_preserves_every_old_guard_and_six_NN_owners():
    proof = g.native_v3_observed_train_continuation_guard_source_bridge(_guard_hashes())
    assert proof['old_guard_routes_preserved'] and proof['old_numeric_resource_environment_function_unchanged']
    assert len(proof['old_NN_source_byte_unchanged']) == 6


@pytest.mark.parametrize('bad', ('producer_hash', 'legacy_limit', 'route_limit', 'NN_source'))
def test_continuation_bridge_rejects_unreviewed_changes(monkeypatch, bad):
    hashes = _guard_hashes(); repo = Path(g.__file__).resolve().parents[2]
    if bad == 'producer_hash': hashes[OWNERS[0]] = '0' * 64
    elif bad == 'NN_source':
        read = Path.read_bytes
        monkeypatch.setattr(Path, 'read_bytes', lambda p, *a, **k: read(p, *a, **k) + (b'\n# mutation\n' if p == repo / g._V3_ACTUAL_STEP_OBSERVER_CORE else b''))
    else:
        read = Path.read_text
        def changed(path, *a, **kw):
            text = read(path, *a, **kw)
            if path == repo / OWNERS[0] and bad == 'legacy_limit': text = text.replace('_MAX_MEMORY_BYTES = 20 * 1024**3', '_MAX_MEMORY_BYTES = 21 * 1024**3', 1)
            if path == repo / OWNERS[1] and bad == 'route_limit':
                fragment = g._V3_OBSERVED_TRAIN_CONTINUATION_SHELL_ADDITIONS[1][2]
                text = text.replace(fragment, fragment.replace('TRAINER_GPU_MAX_POWER_DRAW_W=250', 'TRAINER_GPU_MAX_POWER_DRAW_W=251'), 1)
            return text
        monkeypatch.setattr(Path, 'read_text', changed)
    with pytest.raises(RuntimeError): g.native_v3_observed_train_continuation_guard_source_bridge(hashes)
