"""Synthetic exact-suffix metadata, reporting and orchestration; no native model/PT IO."""
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from gx1.audit import native_provider_hlc_learning_v1 as m

_LATEST_INPUTS = copy.deepcopy(m.RESUME_INPUTS)
_ACTIVE_INPUTS = {key: copy.deepcopy(_LATEST_INPUTS[key]) for key in
                  ('original_plan', 'original_claim', 'original_terminal', 'original_admitted_policy',
                   'original_stop_review', 'original_backup', 'ancestral_checkpoint_receipt',
                   'ancestral_checkpoint', 'ancestral_restore_proof')}
_ACTIVE_INPUTS.update(checkpoint_receipt=_LATEST_INPUTS['previous_initial_receipt'],
                      original_checkpoint=_LATEST_INPUTS['previous_initial_checkpoint'],
                      previous_execution_plan=_LATEST_INPUTS['earlier_execution_plan'],
                      previous_initial_receipt=_LATEST_INPUTS['intervening_checkpoint_receipt'],
                      previous_initial_checkpoint=_LATEST_INPUTS['intervening_checkpoint'],
                      earlier_execution_plan=_LATEST_INPUTS['original_resume_execution_plan'],
                      previous_restore_proof=_LATEST_INPUTS['intervening_restore_proof'])
_ACTIVE_INPUTS.update({key: {'path': '/synthetic/' + key + '.json', 'sha256': 'a' * 64}
                      for key in ('previous_claim', 'previous_terminal', 'previous_admitted_policy',
                                  'previous_stop_review', 'previous_backup', 'resume_scope')})
_ACTIVE_LIMITS = copy.deepcopy(m.RESUME_LIMITS)


def _suffix_limits(cursor):
    limits = copy.deepcopy(_ACTIVE_LIMITS)
    updates = 3694 - cursor
    checkpoints = len(range((cursor // 256 + 1) * 256, 3694, 256)) + 1
    limits.update(model_forwards=updates + 1385, backbone_forwards=updates + 1385,
                  readout_forwards=updates + 1385, optimizer_steps=updates,
                  backward_calls=updates, checkpoint_writes=checkpoints,
                  checkpoint_readbacks=checkpoints, checkpoint_receipt_writes=checkpoints)
    return limits



def _binding(name):
    return {'path': '/synthetic/' + name + '.json', 'sha256': 'a' * 64}


def _metadata_fixture(monkeypatch):
    from gx1.contracts import gx1_capped_execution_v1 as guard
    from gx1.contracts import market_state_ta_integrated_model_v1 as model_owner
    inputs = {key: copy.deepcopy(_ACTIVE_INPUTS[key]) for key in
              ('original_plan', 'original_claim', 'original_terminal',
               'original_admitted_policy', 'original_stop_review', 'original_backup')}
    inputs.update(checkpoint_receipt=copy.deepcopy(_ACTIVE_INPUTS['ancestral_checkpoint_receipt']),
                  original_checkpoint=copy.deepcopy(_ACTIVE_INPUTS['ancestral_checkpoint']))
    monkeypatch.setattr(m, 'RESUME_INPUTS', inputs)
    monkeypatch.setattr(m, 'RESUME_CURSOR', 512)
    monkeypatch.setattr(m, 'RESUME_LIMITS', _suffix_limits(512))
    source_bytes = b'synthetic-source'
    digest = hashlib.sha256(source_bytes).hexdigest()
    runtime = {'source_commit': m.RESUME_PRODUCER, 'gpu': {'synthetic': True},
               'library_versions': {'synthetic': True}, 'wsl_boot_id': 'synthetic-boot',
               'physical_windows_boot_id': 1, 'observed_utc': '2026-11-23T19:00:00+00:00'}
    binding = {'contract_sha256': 'b' * 64, 'input_normalization_sha256': 'c' * 64,
               'feature_provider_contract_sha256': 'd' * 64}
    normalization = {'contract_sha256': 'c' * 64,
                     'feature_provider_contract_sha256': 'd' * 64}
    metadata = {'runtime': runtime, 'input_binding': binding, 'architecture': {},
                'target_contract': {'contract_sha256': 'e' * 64},
                'truth_sha256': 'f' * 64, 'prior': _binding('prior'),
                'normalization': _binding('normalization'),
                'input_binding_ref': _binding('input_binding')}
    recipe = {'passes': 1, 'train_batch_rows': 96, 'eval_batch_rows': 256,
              'checkpoint_every_optimizer_updates': 256}
    schedule = {'rows': 354550, 'logical_batches': 3694, 'batches_per_pass': 3694,
                'passes': 1, 'optimizer_updates': 3694, 'skipped_batches': 0,
                'update_flags': [True] * 3694}
    contract = {'contract_sha256': '1' * 64}
    initial = {'model_contract_sha256': '1' * 64,
               'target_contract_sha256': 'e' * 64,
               'model_state_sha256': '2' * 64, 'rng_sha256': '3' * 64}
    original = {'schema_version': 'gx1_native_provider_hlc_one_epoch_plan_v1',
                'stage': m.STAGE, 'source_commit': m.RESUME_PRODUCER,
                'output_root': str(m.ROOT), 'execution': copy.deepcopy(m.EXECUTION),
                'backbone_gradient_policy': 'M5_primary', 'recipe': recipe,
                'schedule': schedule, 'metadata': metadata, 'model_contract': contract,
                'initial_reference': initial,
                'limits': {**_ACTIVE_LIMITS, 'model_forwards': 5079,
                           'backbone_forwards': 5079, 'readout_forwards': 5079,
                           'optimizer_steps': 3694, 'backward_calls': 3694,
                           'checkpoint_writes': 16, 'checkpoint_readbacks': 16,
                           'checkpoint_receipt_writes': 16, 'checkpoint_loads': 0,
                           'model_restores': 0},
                'inputs': {'baseline_result': _binding('baseline_result'),
                           'baseline_initial_receipt': _binding('baseline_initial_receipt')},
                'source_owners': {key: digest for key in ('numeric_owner.py', m.SELF, *m.GUARDS)},
                'scientific_scope': copy.deepcopy(m.SCIENTIFIC_SCOPE)}
    original_ref = m.RESUME_INPUTS['original_plan']
    traj = m.trajectory(original)
    proof = dict.fromkeys(('trajectory', 'position', 'model_state', 'optimizer_state',
                           'scheduler_state', 'rng_state', 'training',
                           'optimizer_parameter_names', 'steps', 'scores'), '4' * 64)
    scores = {'INITIAL': {'reused_qualified_result': original['inputs']['baseline_result'],
                          'no_repeated_initial_forwards': True}}
    proof.update(trajectory=m.core.state_sha(traj),
                 position=m.core.state_sha(m.train.position(schedule, 512)),
                 scheduler_state=m.core.state_sha(None), training=m.core.state_sha(True),
                 scores=m.core.state_sha(scores))
    receipt = {'schema_version': 'gx1_native_future_state_training_checkpoint_receipt_v1',
               'checkpoint': m.RESUME_INPUTS['original_checkpoint'],
               'original_plan': original_ref, 'execution_plan': original_ref,
               'producer_lane': 'main', 'producer_runtime': runtime,
               'logical_cursor': 512, 'optimizer_updates': 512,
               'trajectory_sha256': traj['contract_sha256'], 'proof': proof}
    claim = {'plan': original_ref, 'policy_at_claim': {
                 'path': str(m.POLICY), 'sha256': m.RESUME_INPUTS['original_admitted_policy']['sha256']},
             'runtime': runtime, 'remaining_invocations': 0, 'TRAIN_diagnostic_only': True,
             'process_identity': {'synthetic': True}}
    policy = dict.fromkeys(m.FLAGS, True); policy['full_val_allowed'] = False
    policy[m.STAGE] = m.gate(original_ref, True)
    terminal = {'status': 'FAILED_CONSUMED', 'exit_code': 1, 'plan': original_ref,
                'claim': m.RESUME_INPUTS['original_claim'], 'remaining_invocations': 0,
                'relaunch_allowed': False}
    review = {'plan': original_ref, 'original_process_stopped': True,
              'completed_epoch': False, 'last_durable_cursor': 512, 'last_durable_rows': 49152,
              'last_durable_checkpoint': m.RESUME_INPUTS['original_checkpoint'],
              'last_durable_receipt': m.RESUME_INPUTS['checkpoint_receipt'],
              'guard_stop_reason': 'host_maintenance_active'}
    backup = {'schema_version': 'gx1_bound_separate_disk_backup_ack_v1',
              'status': 'PASS_INITIAL_AND_DURABLE256_512_CHECKPOINTS_AND_ORIGINAL_RECEIPTS',
              'Mac_F_FULLFSYNC_and_disk_SHA_readback': True, 'checkpoint_files': 3,
              'files': 32, 'total_bytes': 248176021}
    values = {m.RESUME_INPUTS[key]['path']: value for key, value in
              (('original_plan', original), ('checkpoint_receipt', receipt),
               ('original_claim', claim), ('original_terminal', terminal),
               ('original_admitted_policy', policy), ('original_stop_review', review),
               ('original_backup', backup))}
    values.update({metadata['input_binding_ref']['path']: binding,
                   metadata['normalization']['path']: normalization,
                   metadata['prior']['path']: {},
                   original['inputs']['baseline_initial_receipt']['path']: initial})
    references = []
    def reference(binding, expected_path=None):
        references.append(binding)
        return m.train.require_recovery_ref(binding, expected_path=expected_path,
                                             suffix=Path(binding['path']).suffix)
    monkeypatch.setattr(m, 'read', lambda binding: values[binding['path']])
    monkeypatch.setattr(m, 'resume_reference', reference)
    monkeypatch.setattr(m, 'ref', lambda path: {'path': str(path), 'sha256': digest})
    monkeypatch.setattr(m.subprocess, 'check_output', lambda *a, **k: source_bytes)
    monkeypatch.setattr(m, 'resume_source_bridge', lambda: {'synthetic': 'append_only'})
    monkeypatch.setattr(m.core, 'future_cuda_require_runtime', lambda value: True)
    stopped = []
    monkeypatch.setattr(m.core, 'future_cpu_parent_stopped', stopped.append)
    monkeypatch.setattr(guard, 'native_provider_hlc_learning_resume_source_bridge',
                        lambda *a, **k: {'synthetic': 'guard_bridge'})
    monkeypatch.setattr(model_owner, 'native_future_state_model_contract',
                        lambda *a, **k: copy.deepcopy(contract))
    live = copy.deepcopy(runtime); live['source_commit'] = '9' * 40
    return original, receipt, claim, policy, terminal, review, backup, values, live, references, stopped


def test_resume_plan_preserves_original_trajectory_and_has_exact_remaining_budgets(monkeypatch):
    original, receipt, _, _, _, _, _, _, live, refs, stopped = _metadata_fixture(monkeypatch)
    before = copy.deepcopy(original); plan = m.build_resume_plan(live)
    assert original == before
    assert plan['metadata']['original'] == original
    assert plan['metadata']['trajectory'] == m.trajectory(original)
    assert plan['metadata']['trajectory']['numeric_source_owners'] == original['source_owners']
    assert plan['metadata']['receipt'] == receipt
    assert plan['metadata']['runtime'] == live and plan['source_commit'] == live['source_commit']
    assert plan['recipe'] == original['recipe'] and plan['schedule'] == original['schedule']
    assert plan['initial_reference'] == original['initial_reference']
    assert plan['limits'] == m.RESUME_LIMITS
    assert plan['limits']['model_forwards'] == (3694 - 512) + 1385 == 4567
    assert plan['limits']['optimizer_steps'] == 3182
    assert plan['limits']['checkpoint_writes'] == len(range(768, 3694, 256)) + 1 == 13
    assert plan['limits']['checkpoint_loads'] == plan['limits']['model_restores'] == 1
    assert plan['starting_cursor'] == 512 and plan['remaining_TRAIN_rows'] == 354550 - 512 * 96
    assert plan['remaining_FINAL_rows'] == 354550 and plan['new_INITIAL_writes'] == 0
    assert set(plan['source_owners']) == set(original['source_owners']) | {m.RESUME_TEST, m.RESUME_GUARD_TEST}
    assert all(Path(binding['path']).is_absolute() for binding in refs)
    assert stopped == [{'synthetic': True}]
    assert plan['scientific_scope']['chronological_generalization'] == 'UNKNOWN'
    assert plan['scientific_scope']['economic_edge'] == 'UNKNOWN'


@pytest.mark.parametrize('bad', ['schema', 'proof_missing', 'proof_extra', 'proof_digest',
    'trajectory_contract_as_proof', 'cursor', 'cursor_bool', 'updates', 'position',
    'old_original_plan', 'old_execution_plan', 'checkpoint', 'runtime', 'lane',
    'scheduler', 'training', 'scores', 'terminal_success', 'terminal_claim',
    'claim_policy_path', 'claim_policy_sha', 'claim_plan', 'old_flags', 'old_gate',
    'stop_cursor', 'backup', 'recipe', 'clock', 'model_identity', 'normalization',
    'initial_receipt', 'numeric_owner'])
def test_resume_plan_rejects_unbound_receipt_closure_or_function_identity(monkeypatch, bad):
    original, receipt, claim, policy, terminal, review, backup, values, live, _, _ = _metadata_fixture(monkeypatch)
    if bad == 'schema': receipt['unexpected'] = True
    elif bad == 'proof_missing': receipt['proof'].pop('rng_state')
    elif bad == 'proof_extra': receipt['proof']['unexpected'] = 'a' * 64
    elif bad == 'proof_digest': receipt['proof']['steps'] = 'not-a-digest'
    elif bad == 'trajectory_contract_as_proof': receipt['proof']['trajectory'] = receipt['trajectory_sha256']
    elif bad == 'cursor': receipt['logical_cursor'] = 256
    elif bad == 'cursor_bool': receipt['logical_cursor'] = True
    elif bad == 'updates': receipt['optimizer_updates'] = 511
    elif bad == 'position': receipt['proof']['position'] = '0' * 64
    elif bad == 'old_original_plan': receipt['original_plan'] = _binding('foreign')
    elif bad == 'old_execution_plan': receipt['execution_plan'] = _binding('new_resume')
    elif bad == 'checkpoint': receipt['checkpoint'] = _binding('foreign')
    elif bad == 'runtime':
        receipt['producer_runtime'] = copy.deepcopy(receipt['producer_runtime'])
        receipt['producer_runtime']['gpu'] = {'foreign': True}
    elif bad == 'lane': receipt['producer_lane'] = 'cold'
    elif bad in ('scheduler', 'training', 'scores'):
        receipt['proof'][{'scheduler': 'scheduler_state', 'training': 'training', 'scores': 'scores'}[bad]] = '0' * 64
    elif bad == 'terminal_success': terminal['status'] = 'COMPLETE'
    elif bad == 'terminal_claim': terminal['claim'] = _binding('foreign')
    elif bad == 'claim_policy_path': claim['policy_at_claim'] = _binding('policy')
    elif bad == 'claim_policy_sha': claim['policy_at_claim']['sha256'] = '0' * 64
    elif bad == 'claim_plan': claim['plan'] = _binding('foreign')
    elif bad == 'old_flags': policy['full_val_allowed'] = True
    elif bad == 'old_gate': policy[m.STAGE]['remaining_invocations'] = 0
    elif bad == 'stop_cursor': review['last_durable_cursor'] = 513
    elif bad == 'backup': backup['Mac_F_FULLFSYNC_and_disk_SHA_readback'] = False
    elif bad == 'recipe': original['recipe']['passes'] = 2
    elif bad == 'clock': original['schedule']['update_flags'][0] = False
    elif bad == 'model_identity': original['initial_reference']['model_contract_sha256'] = '0' * 64
    elif bad == 'normalization': values[original['metadata']['normalization']['path']]['contract_sha256'] = '0' * 64
    elif bad == 'initial_receipt': values[original['inputs']['baseline_initial_receipt']['path']] = {}
    else: original['source_owners']['numeric_owner.py'] = '0' * 64
    with pytest.raises(ValueError):
        m.build_resume_plan(live)


@pytest.mark.parametrize('mismatch', [None, 'source_commit', 'wsl_boot_id',
                                      'physical_windows_boot_id', 'gpu', 'library_versions',
                                      'claim_observed_utc'])
def test_original_producer_runtime_reuses_canonical_matcher_and_exact_claim(monkeypatch, mismatch):
    original, receipt, claim, _, _, _, _, _, _, _, _ = _metadata_fixture(monkeypatch)
    producer = copy.deepcopy(original['metadata']['runtime'])
    producer['observed_utc'] = '2026-11-23T20:00:00+00:00'
    if mismatch in ('source_commit', 'wsl_boot_id'):
        producer[mismatch] = '8' * 40
    elif mismatch == 'physical_windows_boot_id': producer[mismatch] += 1
    elif mismatch in ('gpu', 'library_versions'): producer[mismatch] = {'foreign': True}
    receipt['producer_runtime'] = producer
    claim['runtime'] = copy.deepcopy(producer)
    if mismatch == 'claim_observed_utc':
        claim['runtime']['observed_utc'] = '2026-11-23T20:00:01+00:00'
    inputs = {**original['inputs'], **m.RESUME_INPUTS}
    if mismatch:
        with pytest.raises(ValueError): m.resume_original_metadata(original, inputs)
    else:
        actual_receipt, traj = m.resume_original_metadata(original, inputs)
        assert actual_receipt is receipt and traj == m.trajectory(original)
        assert receipt['producer_runtime'] != original['metadata']['runtime']
        assert receipt['producer_runtime'] == claim['runtime']


def test_resume_metadata_rejects_still_running_original_before_claim(monkeypatch):
    original, _, _, _, _, _, _, _, _, _, _ = _metadata_fixture(monkeypatch)
    monkeypatch.setattr(m.core, 'future_cpu_parent_stopped',
                        lambda identity: (_ for _ in ()).throw(ValueError('PARENT_STILL_RUNNING')))
    with pytest.raises(ValueError, match='PARENT_STILL_RUNNING'):
        m.resume_original_metadata(original, {**original['inputs'], **m.RESUME_INPUTS})


@pytest.mark.parametrize('admitted', [False, True])
def test_exact_resume_gate_never_reopens_old_plan_or_extra_epoch(admitted):
    binding = _binding('new_resume')
    gate = m.resume_gate(binding, admitted)
    assert gate['plan'] == binding and gate['remaining_invocations'] == int(admitted)
    assert gate['execution_authorized'] is admitted
    assert gate['full_epoch_training_allowed'] is admitted
    assert all(gate[key] is False for key in ('additional_epochs_allowed', 'test_authorized', 'relaunch_allowed'))


@pytest.mark.parametrize('bad', ['relative', 'dotdot', 'symlink', 'digest'])
def test_resume_reference_rejects_noncanonical_or_changed_files(tmp_path, bad):
    path = tmp_path / 'artifact.json'; path.write_text('{}')
    binding = m.ref(path)
    if bad == 'relative': binding['path'] = 'relative.json'
    elif bad == 'dotdot': binding['path'] = str(tmp_path / '..' / tmp_path.name / 'artifact.json')
    elif bad == 'symlink':
        alias = tmp_path / 'alias.json'; alias.symlink_to(path); binding['path'] = str(alias)
    else: binding['sha256'] = '0' * 64
    with pytest.raises(ValueError): m.resume_reference(binding)


def _run_fixture(monkeypatch, tmp_path):
    from gx1.contracts import gx1_capped_execution_v1 as guard
    planfile = tmp_path / 'PLAN.json'; policyfile = tmp_path / 'POLICY.json'
    planfile.write_text('{}'); policyfile.write_text('{}')
    monkeypatch.setattr(m, 'RESUME_PLAN', planfile); monkeypatch.setattr(m, 'POLICY', policyfile)
    monkeypatch.setattr(m, 'RESUME_ROOT', tmp_path / 'resume_output')
    binding, pref = m.ref(planfile), m.ref(policyfile)
    plan = {'starting_cursor': 512, 'schedule': {'logical_batches': 3694},
            'checkpoint_global_cursors': list(range(768, 3694, 256)) + [3694],
            'metadata': {'original_plan': _binding('original')},
            'inputs': {key: {'path': str(tmp_path / name)} for key, name in
                       (('feature_archive', 'features.npz'), ('target_arrays', 'targets.npz'),
                        ('original_checkpoint', 'UPDATE000512.pt'), ('checkpoint_receipt', 'UPDATE000512.json'))}}
    live = {'physical_windows_boot_id': 1,
            'wsl_boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    monkeypatch.setattr(guard, 'require_native_provider_hlc_learning_resume_execution',
                        lambda **k: {'plan': binding, 'output_root': str(m.RESUME_ROOT)})
    monkeypatch.setattr(m, 'inspect_resume', lambda *a: (plan, binding, pref, live))
    monkeypatch.setattr(m, 'check_resume_sources_inputs', lambda plan: None)
    monkeypatch.setattr(m.core, 'physical_windows_boot_id', lambda: 1)
    monkeypatch.setattr(m.core, 'future_cpu_process_identity', lambda: {'synthetic': True})
    monkeypatch.setattr(m.core.common, 'firewall_state', lambda: {'denied_attempts': 0})
    hooks, timers = [], []
    monkeypatch.setattr(m.sys, 'addaudithook', hooks.append)
    monkeypatch.setattr(m.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(m.signal, 'setitimer', lambda *a: timers.append(a))
    monkeypatch.setattr(m, 'physical_resume', lambda *a: {'models': 0})
    return plan, binding, pref, hooks, timers


@pytest.mark.parametrize('failure', [None, 'physical', 'bookend'])
def test_resume_success_and_consumed_failure_reports_preserve_outputs(monkeypatch, tmp_path, failure):
    _, binding, pref, _, timers = _run_fixture(monkeypatch, tmp_path)
    def physical(*args):
        (m.RESUME_ROOT / 'preserved.json').write_text('{}')
        if failure == 'physical': raise ValueError('synthetic_physical')
        return {'models': 0}
    monkeypatch.setattr(m, 'physical_resume', physical)
    if failure == 'bookend':
        monkeypatch.setattr(m, 'check_resume_sources_inputs',
                            lambda plan: (_ for _ in ()).throw(ValueError('synthetic_bookend')))
    if failure:
        with pytest.raises(ValueError, match='synthetic_' + failure): m.run_resume(binding['sha256'])
    else:
        result = m.run_resume(binding['sha256'])
        assert result['result'] == m.ref(m.RESUME_ROOT / 'RESULT.json')
        payload = json.loads((m.RESUME_ROOT / 'RESULT.json').read_text())
        assert payload['status'] == 'COMPLETE_ORIGINAL_V2_EPOCH_FROM_CURSOR512'
        assert payload['report'] == {'models': 0} and payload['test_data_used'] is False
    claim = json.loads((m.RESUME_ROOT / 'CLAIM.json').read_text())
    terminal = json.loads((m.RESUME_ROOT / 'TERMINAL.json').read_text())
    assert claim['plan'] == binding and claim['policy_at_claim'] == pref
    assert claim['starting_cursor'] == 512 and claim['remaining_invocations'] == 0
    assert terminal['plan'] == binding and terminal['exit_code'] == int(bool(failure))
    assert terminal['status'] == ('FAILED_CONSUMED' if failure else 'COMPLETE')
    assert terminal['remaining_invocations'] == 0 and terminal['relaunch_allowed'] is False
    assert (m.RESUME_ROOT / 'RESULT.json').exists() is (failure is None)
    assert (m.RESUME_ROOT / 'preserved.json').is_file()
    assert timers[-1] == (m.signal.ITIMER_REAL, 0)


def test_resume_audit_allows_only_old512_and_new_suffix_payloads(monkeypatch, tmp_path):
    plan, binding, _, hooks, _ = _run_fixture(monkeypatch, tmp_path)
    m.run_resume(binding['sha256']); assert len(hooks) == 1
    audit = hooks[0]
    for key in ('feature_archive', 'target_arrays', 'original_checkpoint'):
        audit('open', (plan['inputs'][key]['path'],))
    for name in ('UPDATE000768.pt', 'UPDATE003584.pt', 'FINAL.pt', 'FINAL_LOGITS.npz'):
        audit('open', (str(m.RESUME_ROOT / name),))
    for name in ('INITIAL.pt', 'UPDATE000256.pt', 'UPDATE000512.pt', 'foreign.npz'):
        with pytest.raises(RuntimeError, match='RESUME_PAYLOAD_PATH_FORBIDDEN'):
            audit('open', (str(m.RESUME_ROOT / name),))
    for event in ('socket.connect', 'socket.getaddrinfo', 'urllib.Request'):
        with pytest.raises(RuntimeError, match='RESUME_NETWORK_FORBIDDEN'): audit(event, ())


def test_physical_resume_has_one_load_restore_and_no_initial_checkpoint_authority():
    tree = ast.parse(inspect.getsource(m.physical_resume))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    def named(call, owner, name):
        return isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == owner and call.func.attr == name
    loads = [call for call in calls if named(call, 'torch', 'load')]
    restores = [call for call in calls if named(call, 'train', 'strict_restore')]
    assert len(loads) == len(restores) == 1
    assert {kw.arg: ast.literal_eval(kw.value) for kw in loads[0].keywords} == {'map_location': 'cpu', 'weights_only': True}
    kwargs = {kw.arg: kw.value for kw in restores[0].keywords}
    assert ast.unparse(kwargs['expected_cursor']) == 'start'
    assert ast.unparse(kwargs['original_plan_ref']) == "meta['original_plan']"
    assert ast.unparse(kwargs['expected_state_proof']) == "receipt['proof']"
    assert ast.unparse(kwargs['trajectory']) == 'traj'
    assert 'INITIAL.pt' not in inspect.getsource(m.physical_resume)
    assert ast.unparse(next(call for call in calls if named(call, 'train', 'training_step')).args[-1]) == 'j'


@pytest.mark.parametrize('start', [512, 1792, 2048, 3328])
@pytest.mark.parametrize('restore_failure', [False, True])
def test_physical_resume_synthetic_full_suffix_preserves_history_and_exact_counts(monkeypatch, restore_failure, start):
    """Exercise orchestration with fake construction/load/step/score, never native data/model."""
    import numpy as np
    import torch
    from gx1.contracts import market_state_ta_integrated_model_v1 as model_owner
    from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
    original, receipt, _, _, _, _, _, _, live, _, _ = _metadata_fixture(monkeypatch)
    if start in (1792, 2048, 3328):
        receipt = copy.deepcopy(receipt)
        checkpoint = (_LATEST_INPUTS['original_checkpoint'] if start == 3328 else
                      _ACTIVE_INPUTS['previous_initial_checkpoint' if start == 1792 else 'original_checkpoint'])
        receipt.update(logical_cursor=start, optimizer_updates=start, checkpoint=checkpoint)
    monkeypatch.setattr(m, 'RESUME_CURSOR', start)
    monkeypatch.setattr(m, 'RESUME_LIMITS', _suffix_limits(start))
    original['schedule'].update(batch_source_rows_sha256=['batch'] * 3694,
                                batch_issue_ns_sha256=['batch'] * 3694)
    original['recipe']['optimizer'] = {'synthetic': True}
    plan = {'starting_cursor': start, 'remaining_TRAIN_rows': 354550 - start * 96,
            'metadata': {'original': original, 'original_plan': m.RESUME_INPUTS['original_plan'],
                         'receipt': receipt, 'trajectory': m.trajectory(original)},
            'inputs': {'checkpoint_receipt': m.RESUME_INPUTS['checkpoint_receipt'],
                       'original_checkpoint': receipt['checkpoint']},
            'limits': copy.deepcopy(m.RESUME_LIMITS)}
    context = {'source_rows': np.arange(354550, dtype=np.int64),
               'truth': {'issue_ns': np.arange(354550, dtype=np.int64)},
               'normalization': {}, 'input_binding': {}, 'target_contract': {}, 'prior': {}}
    before_steps = [{'logical_step': j + 1} for j in range(start)]
    before_scores = {'INITIAL': {'synthetic_reused_initial': True}}
    cp = {'producer': {'original_plan': plan['metadata']['original_plan'],
                       'lane': 'main', 'runtime': receipt['producer_runtime']},
          'steps': copy.deepcopy(before_steps), 'scores': copy.deepcopy(before_scores)}
    class Hook:
        def remove(self): pass
    class Component:
        def __init__(self): self.hooks = []
        def register_forward_pre_hook(self, hook): self.hooks.append(hook); return Hook()
        def fire(self):
            for hook in self.hooks: hook()
    class Model(Component):
        def __init__(self):
            super().__init__(); self.backbone = Component(); self.readout = Component()
            self.research_objective = original['model_contract']
        def state_dict(self): return {'synthetic_model': True}
        def parameters(self): return []
        def fire_all(self): self.fire(); self.backbone.fire(); self.readout.fire()
    class Optimizer:
        def __init__(self): self.state = {}
        def zero_grad(self, **kwargs): pass
    model, optimizer = Model(), Optimizer()
    constructions, loads, restores, indices, checkpoints = [], [], [], [], []
    monkeypatch.setattr(m.adapter, 'load_context', lambda metadata: context)
    monkeypatch.setattr(m.adapter, 'packet', lambda context, take: {'truth': {}})
    monkeypatch.setattr(m.core, 'digest', lambda value: 'batch')
    monkeypatch.setattr(m.core, 'future_cuda_seed_and_runtime', lambda value: None)
    monkeypatch.setattr(m.core, 'training_rng', lambda: {'synthetic_rng': True})
    monkeypatch.setattr(m.core, 'state_sha', lambda value:
                        original['initial_reference']['rng_sha256'] if 'synthetic_rng' in value
                        else original['initial_reference']['model_state_sha256'])
    def construct(**kwargs): constructions.append(kwargs); return model
    monkeypatch.setattr(model_owner, 'build_native_future_state_model', construct)
    monkeypatch.setattr(torch.optim, 'AdamW', lambda *a, **k: optimizer)
    def load(*args, **kwargs): loads.append((args, kwargs)); return cp
    monkeypatch.setattr(torch, 'load', load)
    def restore(actual_model, actual_optimizer, actual_checkpoint, **kwargs):
        assert actual_model is model and actual_optimizer is optimizer and actual_checkpoint is cp
        assert kwargs['original_plan_ref'] == m.RESUME_INPUTS['original_plan']
        assert kwargs['expected_cursor'] == start and kwargs['expected_state_proof'] == receipt['proof']
        assert kwargs['trajectory'] is plan['metadata']['trajectory']
        restores.append(kwargs)
        if restore_failure: raise ValueError('synthetic_restore_failure')
        return m.train.position(original['schedule'], start)
    monkeypatch.setattr(m.train, 'strict_restore', restore)
    monkeypatch.setattr(objective, '_native_future_state_truth',
                        lambda *a, **k: {'activity': np.ones(1, dtype=np.bool_)})
    def step(actual_model, actual_optimizer, packet, used_plan, counts, j):
        assert used_plan is original; indices.append(j); model.fire_all()
        counts['optimizer_steps'] += 1; counts['backward_calls'] += 1
        return {'logical_step': j + 1}
    monkeypatch.setattr(m.train, 'training_step', step)
    monkeypatch.setattr(m.train, 'release_phase_allocator_cache', lambda *a, **k: {'synthetic': True})
    canonical = {'scores': {'per_tf': {tf: {task: {'count': 2, 'log_loss_sum': .2, 'brier_sum': .4}
                  for task in ('activity', 'side')} for tf in m.train.TF}}}
    quality = {'all5_pooled_side_TRAIN_fit': 'UNKNOWN', 'all_TF': {
               tf: {task: {'count': 2, 'model': {'NLL': .1, 'brier': .2}}
                    for task in ('activity', 'side')} for tf in m.train.TF}}
    def score(*args, **kwargs):
        assert kwargs == {'batch_rows': 256, 'device': 'cuda'}
        for _ in range(1385): model.fire_all()
        return canonical, {'synthetic_logits': np.zeros(1, dtype=np.float32)}
    monkeypatch.setattr(m.adapter, 'score_final', score)
    monkeypatch.setattr(m, 'quality_panels', lambda *a: quality)
    monkeypatch.setattr(m.np, 'savez_compressed', lambda memory, **k: memory.write(b'synthetic-capture'))
    def make_checkpoint(actual_model, actual_optimizer, **kwargs):
        assert kwargs['trajectory'] is plan['metadata']['trajectory']
        assert kwargs['original_plan_ref'] == m.RESUME_INPUTS['original_plan']
        assert kwargs['lane'] == 'main'
        assert kwargs['steps'][:start] == before_steps
        assert len(kwargs['steps']) == kwargs['cursor']
        assert kwargs['scores']['INITIAL'] == before_scores['INITIAL']
        return kwargs
    monkeypatch.setattr(m.train, 'make_checkpoint', make_checkpoint)
    new_binding = _binding('new_execution')
    def publish_checkpoint(path, checkpoint, counts, *, execution_ref):
        assert execution_ref == new_binding
        checkpoints.append((path, checkpoint['cursor']))
        for key in ('checkpoint_writes', 'checkpoint_readbacks', 'checkpoint_receipt_writes'):
            counts[key] += 1
        return {'path': str(path.with_suffix('.json')), 'sha256': 'a' * 64}
    monkeypatch.setattr(m.train, 'publish_checkpoint_receipt', publish_checkpoint)
    events = []
    def publish(root, name, value):
        events.append((name, copy.deepcopy(value)))
        return {'path': str(root / name), 'sha256': 'a' * 64}
    monkeypatch.setattr(m, 'pub', publish)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda: 0)
    if restore_failure:
        with pytest.raises(ValueError, match='synthetic_restore_failure'):
            m.physical_resume(plan, new_binding, _binding('policy'), live)
        assert len(constructions) == len(loads) == len(restores) == 1
        assert events == [] and indices == [] and checkpoints == []
        return
    report = m.physical_resume(plan, new_binding, _binding('policy'), live)
    assert events[0][0] == 'RESTORE%d_PROOF.json' % start
    proof = events[0][1]
    assert proof['canonical_strict_restore_completed'] is True and proof['new_model_forwards'] == 0
    assert proof['checkpoint'] == receipt['checkpoint'] and proof['expected_state_proof'] == receipt['proof']
    assert proof['position'] == m.train.position(original['schedule'], start)
    assert proof['original_plan'] == m.RESUME_INPUTS['original_plan'] and proof['execution_plan'] == new_binding
    assert proof['uninterrupted_next_step_parity'] == 'NOT_MEASURED'
    assert len(constructions) == len(loads) == len(restores) == 1
    assert loads[0] == ((receipt['checkpoint']['path'],), {'map_location': 'cpu', 'weights_only': True})
    assert indices == list(range(start, 3694))
    assert [cursor for _, cursor in checkpoints] == list(range((start // 256 + 1) * 256, 3694, 256)) + [3694]
    assert all(path.parent == m.RESUME_ROOT and path.name != 'INITIAL.pt' for path, _ in checkpoints)
    assert cp['steps'] == before_steps and cp['scores'] == before_scores
    assert report['counts'] == m.RESUME_LIMITS
    assert report['new_optimizer_updates'] == 3694 - start and report['actual_optimizer_updates'] == 3694
    assert report['trajectory'] is plan['metadata']['trajectory']
    assert report['original_plan'] == m.RESUME_INPUTS['original_plan']
    assert report['execution_plan'] == new_binding and report['later_generalization'] == 'UNKNOWN'


def _resume1792_fixture(monkeypatch):
    original, ancestor, _, _, _, _, _, values, live, refs, stopped = _metadata_fixture(monkeypatch)
    base_inputs = copy.deepcopy(m.RESUME_INPUTS)
    inputs1792 = copy.deepcopy(_ACTIVE_INPUTS)
    for key in ('previous_initial_receipt', 'previous_initial_checkpoint', 'earlier_execution_plan', 'ancestral_restore_proof'):
        inputs1792.pop(key)
    inputs1792.update(checkpoint_receipt=_ACTIVE_INPUTS['previous_initial_receipt'],
                      original_checkpoint=_ACTIVE_INPUTS['previous_initial_checkpoint'],
                      previous_execution_plan=_ACTIVE_INPUTS['earlier_execution_plan'],
                      previous_restore_proof=_ACTIVE_INPUTS['ancestral_restore_proof'],
                      resume_scope=_binding('scope1792'))
    monkeypatch.setattr(m, 'RESUME_INPUTS', inputs1792)
    monkeypatch.setattr(m, 'RESUME_CURSOR', 1792)
    monkeypatch.setattr(m, 'RESUME_LIMITS', _suffix_limits(1792))
    inputs = m.RESUME_INPUTS
    previous_ref = inputs['previous_execution_plan']
    original_ref = inputs['original_plan']
    prepared = copy.deepcopy(original['metadata']['runtime'])
    prepared.update(source_commit='233929b8be7fbb39c2955768a3efa14787abb98b',
                    observed_utc='2026-11-23T20:40:00+00:00')
    producer = copy.deepcopy(prepared); producer['observed_utc'] = '2026-11-23T20:41:00+00:00'
    traj = m.trajectory(original)
    previous = {'schema_version': 'gx1_native_provider_hlc_remaining_epoch_plan_v1',
                'source_commit': prepared['source_commit'],
                'stage': Path(previous_ref['path']).stem.lower(),
                'output_root': str(Path(inputs['original_checkpoint']['path']).parent),
                'starting_cursor': 512, 'recipe': copy.deepcopy(original['recipe']),
                'schedule': copy.deepcopy(original['schedule']), 'execution': copy.deepcopy(m.EXECUTION),
                'model_contract': copy.deepcopy(original['model_contract']),
                'initial_reference': copy.deepcopy(original['initial_reference']),
                'inputs': {**original['inputs'], **base_inputs},
                'metadata': {'runtime': prepared, 'original_plan': original_ref,
                             'original': copy.deepcopy(original), 'trajectory': traj,
                             'receipt': copy.deepcopy(ancestor)}}
    receipt = copy.deepcopy(ancestor)
    receipt.update(checkpoint=inputs['original_checkpoint'], execution_plan=previous_ref,
                   producer_runtime=producer, logical_cursor=1792, optimizer_updates=1792)
    receipt['proof']['position'] = m.core.state_sha(m.train.position(original['schedule'], 1792))
    claim = {'plan': previous_ref, 'original_plan': original_ref, 'runtime': copy.deepcopy(producer),
             'policy_at_claim': {'path': str(m.POLICY),
                                 'sha256': inputs['previous_admitted_policy']['sha256']},
             'remaining_invocations': 0, 'TRAIN_diagnostic_only': True,
             'starting_cursor': 512, 'restored_checkpoint_receipt': inputs['ancestral_checkpoint_receipt'],
             'process_identity': {'synthetic_previous': True}}
    policy = dict.fromkeys(m.FLAGS, True); policy['full_val_allowed'] = False
    policy[previous['stage']] = m.resume_gate(previous_ref, True, cursor=512)
    terminal = {'status': 'FAILED_CONSUMED', 'exit_code': 1, 'plan': previous_ref,
                'claim': inputs['previous_claim'], 'remaining_invocations': 0, 'relaunch_allowed': False}
    restore = {'status': 'PASS_CANONICAL_CURSOR512_MODEL_ADAMW_RNG_RESTORE',
               'canonical_strict_restore_completed': True, 'new_model_forwards': 0,
               'original_plan': original_ref, 'execution_plan': previous_ref,
               'checkpoint': inputs['ancestral_checkpoint'],
               'checkpoint_receipt': inputs['ancestral_checkpoint_receipt'],
               'position': m.train.position(original['schedule'], 512),
               'expected_state_proof': copy.deepcopy(ancestor['proof']),
               'uninterrupted_next_step_parity': 'NOT_MEASURED'}
    review = {'execution_plan': previous_ref, 'original_plan': original_ref, 'process_stopped': True,
              'completed_epoch': False, 'last_durable_cursor': 1792, 'last_durable_rows': 172032,
              'last_durable_checkpoint': inputs['original_checkpoint'],
              'last_durable_receipt': inputs['checkpoint_receipt'],
              'restore512_proof': inputs['previous_restore_proof'],
              'actual_cursor512_tensor_restore_qualified': True, 'guard_stop_reason': 'host_maintenance_active'}
    backup = {'schema_version': 'gx1_bound_separate_disk_backup_ack_v1',
              'status': 'PASS_FIVE_SUFFIX_CHECKPOINTS_THROUGH1792_AND_ORIGINAL_RECEIPTS',
              'Mac_F_FULLFSYNC_and_disk_SHA_readback': True, 'checkpoint_files': 5,
              'files': 43, 'total_bytes': 499529425}
    scope = {'original_plan': original_ref, 'previous_checkpoint_receipt': inputs['checkpoint_receipt'],
             'previous_checkpoint': inputs['original_checkpoint'], 'previous_execution_plan': previous_ref,
             'previous_claim': inputs['previous_claim'], 'previous_terminal': inputs['previous_terminal'],
             'previous_admitted_policy': inputs['previous_admitted_policy'],
             'previous_stop_review': inputs['previous_stop_review'], 'previous_stop_backup': inputs['previous_backup'],
             'previous_source_commit': prepared['source_commit'], 'starting_cursor': 1792,
             'new_plan_path': str(m.RESUME_PLAN), 'new_output_root': str(m.RESUME_ROOT)}
    values.update({inputs[key]['path']: value for key, value in
                   (('previous_execution_plan', previous), ('checkpoint_receipt', receipt),
                    ('previous_claim', claim), ('previous_admitted_policy', policy),
                    ('previous_terminal', terminal), ('previous_restore_proof', restore),
                    ('previous_stop_review', review), ('previous_backup', backup), ('resume_scope', scope))})
    return original, receipt, previous, claim, policy, terminal, restore, review, backup, scope, values, live, stopped


def test_1792_plan_authenticates_previous_producer_and_derives_remaining_epoch(monkeypatch):
    original, receipt, previous, claim, _, _, _, _, _, _, _, live, stopped = _resume1792_fixture(monkeypatch)
    plan = m.build_resume_plan(live)
    assert receipt['original_plan'] == m.RESUME_INPUTS['original_plan']
    assert receipt['execution_plan'] == m.RESUME_INPUTS['previous_execution_plan']
    assert receipt['producer_runtime'] == claim['runtime']
    assert receipt['producer_runtime']['source_commit'] == previous['source_commit']
    assert previous['source_commit'] != original['source_commit']
    assert plan['metadata']['trajectory'] == m.trajectory(original)
    assert plan['starting_cursor'] == 1792 and plan['remaining_TRAIN_rows'] == 182518
    assert plan['remaining_FINAL_rows'] == 354550
    assert plan['limits'] == _suffix_limits(1792)
    assert plan['limits']['optimizer_steps'] == 1902
    assert plan['limits']['model_forwards'] == 1902 + 1385 == 3287
    assert plan['checkpoint_global_cursors'] == [2048, 2304, 2560, 2816, 3072, 3328, 3584, 3694]
    assert plan['limits']['checkpoint_writes'] == 8
    assert plan['new_INITIAL_writes'] == 0
    assert stopped == [{'synthetic': True}, {'synthetic_previous': True}]


@pytest.mark.parametrize('bad', ['active_original', 'active_execution', 'active_position', 'active_cursor',
    'producer_original_source', 'previous_source', 'previous_original', 'previous_trajectory',
    'previous_receipt', 'claim_plan', 'claim_runtime', 'claim_original', 'claim_cursor',
    'claim_policy', 'admission', 'terminal', 'restore_original', 'restore_execution', 'restore_proof',
    'restore_position', 'review_cursor', 'backup', 'scope_ref', 'scope_cursor'])
def test_1792_rejects_wrong_previous_execution_ancestry_or_checkpoint_roles(monkeypatch, bad):
    original, receipt, previous, claim, policy, terminal, restore, review, backup, scope, _, live, _ = _resume1792_fixture(monkeypatch)
    if bad == 'active_original': receipt['original_plan'] = m.RESUME_INPUTS['previous_execution_plan']
    elif bad == 'active_execution': receipt['execution_plan'] = m.RESUME_INPUTS['original_plan']
    elif bad == 'active_position': receipt['proof']['position'] = '0' * 64
    elif bad == 'active_cursor': receipt['logical_cursor'] = receipt['optimizer_updates'] = 512
    elif bad == 'producer_original_source': receipt['producer_runtime']['source_commit'] = original['source_commit']
    elif bad == 'previous_source': previous['source_commit'] = original['source_commit']
    elif bad == 'previous_original': previous['metadata']['original']['source_commit'] = '0' * 40
    elif bad == 'previous_trajectory': previous['metadata']['trajectory'] = {}
    elif bad == 'previous_receipt': previous['metadata']['receipt']['proof']['steps'] = '0' * 64
    elif bad == 'claim_plan': claim['plan'] = m.RESUME_INPUTS['original_plan']
    elif bad == 'claim_runtime': claim['runtime']['observed_utc'] = '2026-11-23T20:41:01+00:00'
    elif bad == 'claim_original': claim['original_plan'] = m.RESUME_INPUTS['previous_execution_plan']
    elif bad == 'claim_cursor': claim['starting_cursor'] = 1792
    elif bad == 'claim_policy': claim['policy_at_claim']['sha256'] = '0' * 64
    elif bad == 'admission': policy[previous['stage']]['remaining_invocations'] = 0
    elif bad == 'terminal': terminal['plan'] = m.RESUME_INPUTS['original_plan']
    elif bad == 'restore_original': restore['original_plan'] = m.RESUME_INPUTS['previous_execution_plan']
    elif bad == 'restore_execution': restore['execution_plan'] = m.RESUME_INPUTS['original_plan']
    elif bad == 'restore_proof': restore['expected_state_proof']['steps'] = '0' * 64
    elif bad == 'restore_position': restore['position']['logical_cursor'] = 1792
    elif bad == 'review_cursor': review['last_durable_cursor'] = 1793
    elif bad == 'backup': backup['Mac_F_FULLFSYNC_and_disk_SHA_readback'] = False
    elif bad == 'scope_ref': scope['previous_execution_plan'] = _binding('foreign')
    else: scope['starting_cursor'] = 512
    with pytest.raises(ValueError): m.build_resume_plan(live)


def test_resume_bookend_hashing_allows_exact_ancestral_checkpoint_without_load(monkeypatch, tmp_path):
    plan, binding, _, hooks, _ = _run_fixture(monkeypatch, tmp_path)
    plan['inputs']['ancestral_checkpoint'] = {'path': str(tmp_path / 'ancestral512.pt')}
    plan['inputs']['previous_initial_checkpoint'] = {'path': str(tmp_path / 'ancestral2048.pt')}
    plan['inputs']['intervening_checkpoint'] = {'path': str(tmp_path / 'ancestral1792.pt')}
    hashes = []
    def bookend(checked):
        for key in ('original_checkpoint', 'ancestral_checkpoint', 'previous_initial_checkpoint', 'intervening_checkpoint'):
            hooks[0]('open', (checked['inputs'][key]['path'],))
            hashes.append(checked['inputs'][key]['path'])
    monkeypatch.setattr(m, 'check_resume_sources_inputs', bookend)
    m.run_resume(binding['sha256'])
    assert hashes == [plan['inputs'][key]['path'] for key in ('original_checkpoint', 'ancestral_checkpoint', 'previous_initial_checkpoint', 'intervening_checkpoint')]


def _resume2048_fixture(monkeypatch):
    original, initial, earlier, earlier_claim, _, _, preserved, _, _, _, values, live, stopped = _resume1792_fixture(monkeypatch)
    prior_inputs = copy.deepcopy(m.RESUME_INPUTS)
    monkeypatch.setattr(m, 'RESUME_INPUTS', copy.deepcopy(_ACTIVE_INPUTS))
    monkeypatch.setattr(m, 'RESUME_CURSOR', 2048)
    monkeypatch.setattr(m, 'RESUME_LIMITS', _suffix_limits(2048))
    inputs = m.RESUME_INPUTS; original_ref = inputs['original_plan']; previous_ref = inputs['previous_execution_plan']
    prepared = copy.deepcopy(earlier['metadata']['runtime'])
    prepared.update(source_commit='dc615829bcb91d6eb0c4d7d4aa9e17563f805e7e', observed_utc='2026-11-23T22:00:00+00:00')
    producer = copy.deepcopy(prepared); producer['observed_utc'] = '2026-11-23T22:01:00+00:00'
    previous = copy.deepcopy(earlier)
    previous.update(source_commit=prepared['source_commit'], stage=Path(previous_ref['path']).stem.lower(),
                    output_root=str(Path(inputs['original_checkpoint']['path']).parent), starting_cursor=1792,
                    inputs={**original['inputs'], **prior_inputs})
    previous['metadata'].update(runtime=prepared, receipt=copy.deepcopy(initial))
    receipt = copy.deepcopy(initial)
    receipt.update(checkpoint=inputs['original_checkpoint'], execution_plan=previous_ref,
                   producer_runtime=producer, logical_cursor=2048, optimizer_updates=2048)
    receipt['proof']['position'] = m.core.state_sha(m.train.position(original['schedule'], 2048))
    claim = copy.deepcopy(earlier_claim)
    claim.update(plan=previous_ref, runtime=copy.deepcopy(producer), starting_cursor=1792,
                 restored_checkpoint_receipt=inputs['previous_initial_receipt'],
                 process_identity={'synthetic_previous002': True},
                 policy_at_claim={'path': str(m.POLICY), 'sha256': inputs['previous_admitted_policy']['sha256']})
    policy = dict.fromkeys(m.FLAGS, True); policy['full_val_allowed'] = False
    policy[previous['stage']] = m.resume_gate(previous_ref, True, cursor=1792)
    terminal = {'status': 'FAILED_CONSUMED', 'exit_code': 1, 'plan': previous_ref,
                'claim': inputs['previous_claim'], 'remaining_invocations': 0, 'relaunch_allowed': False}
    restore = copy.deepcopy(preserved)
    restore.update(status='PASS_CANONICAL_CURSOR1792_MODEL_ADAMW_RNG_RESTORE', execution_plan=previous_ref,
                   checkpoint=inputs['previous_initial_checkpoint'], checkpoint_receipt=inputs['previous_initial_receipt'],
                   position=m.train.position(original['schedule'], 1792), expected_state_proof=copy.deepcopy(initial['proof']))
    review = {'execution_plan': previous_ref, 'original_plan': original_ref, 'process_stopped': True,
              'completed_epoch': False, 'last_durable_cursor': 2048, 'last_durable_rows': 196608,
              'last_durable_checkpoint': inputs['original_checkpoint'], 'last_durable_receipt': inputs['checkpoint_receipt'],
              'restore1792_proof': inputs['previous_restore_proof'], 'actual_cursor1792_tensor_restore_qualified': True,
              'guard_stop_reason': 'host_maintenance_active'}
    backup = {'schema_version': 'gx1_bound_separate_disk_backup_ack_v1',
              'status': 'PASS_ONE_SUFFIX_CHECKPOINT_THROUGH2048_AND_ORIGINAL_RECEIPTS',
              'Mac_F_FULLFSYNC_and_disk_SHA_readback': True, 'checkpoint_files': 1, 'files': 41, 'total_bytes': 114414747}
    scope = {role: inputs[key] for key, role in {'original_plan': 'original_plan', 'checkpoint_receipt': 'previous_checkpoint_receipt',
             'original_checkpoint': 'previous_checkpoint', 'previous_execution_plan': 'previous_execution_plan',
             'previous_claim': 'previous_claim', 'previous_terminal': 'previous_terminal', 'previous_admitted_policy': 'previous_admitted_policy',
             'previous_stop_review': 'previous_stop_review', 'previous_backup': 'previous_stop_backup'}.items()}
    scope.update(starting_cursor=2048, previous_source_commit=prepared['source_commit'],
                 new_plan_path=str(m.RESUME_PLAN), new_output_root=str(m.RESUME_ROOT))
    values.update({inputs[key]['path']: value for key, value in
                  (('previous_execution_plan', previous), ('checkpoint_receipt', receipt), ('previous_claim', claim),
                   ('previous_admitted_policy', policy), ('previous_terminal', terminal), ('previous_restore_proof', restore),
                   ('previous_stop_review', review), ('previous_backup', backup), ('resume_scope', scope),
                   ('previous_initial_receipt', initial), ('ancestral_restore_proof', preserved))})
    return original, receipt, previous, claim, restore, values, live


def test_2048_plan_preserves_immediate1792_and_original512_proofs(monkeypatch):
    original, receipt, previous, claim, restore, _, live = _resume2048_fixture(monkeypatch)
    plan = m.build_resume_plan(live)
    assert plan['starting_cursor'] == 2048 and plan['remaining_TRAIN_rows'] == 157942
    assert plan['limits'] == _suffix_limits(2048)
    assert plan['limits']['optimizer_steps'] == 1646 and plan['limits']['model_forwards'] == 3031
    assert plan['checkpoint_global_cursors'] == [2304, 2560, 2816, 3072, 3328, 3584, 3694]
    assert plan['limits']['checkpoint_writes'] == 7
    assert receipt['producer_runtime'] == claim['runtime']
    assert previous['starting_cursor'] == restore['position']['logical_cursor'] == 1792
    assert restore['expected_state_proof'] == previous['metadata']['receipt']['proof']
    assert plan['metadata']['trajectory'] == m.trajectory(original)


@pytest.mark.parametrize('bad', ['restore_points512', 'restore_proof512', 'initial_receipt',
                                'previous_initial_role', 'preserved_proof', 'producer_original'])
def test_2048_rejects_immediate_vs_deep_ancestor_confusion(monkeypatch, bad):
    original, receipt, previous, _, restore, values, live = _resume2048_fixture(monkeypatch)
    if bad == 'restore_points512': restore['checkpoint'] = m.RESUME_INPUTS['ancestral_checkpoint']
    elif bad == 'restore_proof512': restore['expected_state_proof'] = values[m.RESUME_INPUTS['ancestral_checkpoint_receipt']['path']]['proof']
    elif bad == 'initial_receipt': previous['metadata']['receipt']['logical_cursor'] = 512
    elif bad == 'previous_initial_role': previous['inputs']['checkpoint_receipt'] = m.RESUME_INPUTS['ancestral_checkpoint_receipt']
    elif bad == 'preserved_proof': values[m.RESUME_INPUTS['ancestral_restore_proof']['path']]['expected_state_proof']['steps'] = '0' * 64
    else: receipt['producer_runtime']['source_commit'] = original['source_commit']
    with pytest.raises(ValueError): m.build_resume_plan(live)


def _resume3328_fixture(monkeypatch):
    original, initial, earlier, earlier_claim, intervening_restore, values, live = _resume2048_fixture(monkeypatch)
    prior_inputs = copy.deepcopy(m.RESUME_INPUTS)
    monkeypatch.setattr(m, 'RESUME_INPUTS', copy.deepcopy(_LATEST_INPUTS))
    monkeypatch.setattr(m, 'RESUME_CURSOR', 3328)
    monkeypatch.setattr(m, 'RESUME_LIMITS', _suffix_limits(3328))
    inputs = m.RESUME_INPUTS; original_ref = inputs['original_plan']; previous_ref = inputs['previous_execution_plan']
    prepared = copy.deepcopy(earlier['metadata']['runtime'])
    prepared.update(source_commit='222669c12bfff403997c7c80299dcbf38593537b', observed_utc='2026-11-23T22:30:00+00:00')
    producer = copy.deepcopy(prepared); producer['observed_utc'] = '2026-11-23T22:31:00+00:00'
    previous = copy.deepcopy(earlier)
    previous.update(source_commit=prepared['source_commit'], stage=Path(previous_ref['path']).stem.lower(),
                    output_root=str(Path(inputs['original_checkpoint']['path']).parent), starting_cursor=2048,
                    inputs={**original['inputs'], **prior_inputs})
    previous['metadata'].update(runtime=prepared, receipt=copy.deepcopy(initial))
    receipt = copy.deepcopy(initial)
    receipt.update(checkpoint=inputs['original_checkpoint'], execution_plan=previous_ref,
                   producer_runtime=producer, logical_cursor=3328, optimizer_updates=3328)
    receipt['proof']['position'] = m.core.state_sha(m.train.position(original['schedule'], 3328))
    claim = copy.deepcopy(earlier_claim)
    claim.update(plan=previous_ref, runtime=copy.deepcopy(producer), starting_cursor=2048,
                 restored_checkpoint_receipt=inputs['previous_initial_receipt'], process_identity={'synthetic_previous003': True},
                 policy_at_claim={'path': str(m.POLICY), 'sha256': inputs['previous_admitted_policy']['sha256']})
    policy = dict.fromkeys(m.FLAGS, True); policy['full_val_allowed'] = False
    policy[previous['stage']] = m.resume_gate(previous_ref, True, cursor=2048)
    terminal = {'status': 'FAILED_CONSUMED', 'exit_code': 1, 'plan': previous_ref,
                'claim': inputs['previous_claim'], 'remaining_invocations': 0, 'relaunch_allowed': False}
    restore = copy.deepcopy(intervening_restore)
    restore.update(status='PASS_CANONICAL_CURSOR2048_MODEL_ADAMW_RNG_RESTORE', execution_plan=previous_ref,
                   checkpoint=inputs['previous_initial_checkpoint'], checkpoint_receipt=inputs['previous_initial_receipt'],
                   position=m.train.position(original['schedule'], 2048), expected_state_proof=copy.deepcopy(initial['proof']))
    review = {'execution_plan': previous_ref, 'original_plan': original_ref, 'process_stopped': True,
              'completed_epoch': False, 'last_durable_cursor': 3328, 'last_durable_rows': 319488,
              'last_durable_checkpoint': inputs['original_checkpoint'], 'last_durable_receipt': inputs['checkpoint_receipt'],
              'restore2048_proof': inputs['previous_restore_proof'], 'actual_cursor2048_tensor_restore_qualified': True,
              'guard_stop_reason': 'host_maintenance_active'}
    backup = {'schema_version': 'gx1_bound_separate_disk_backup_ack_v1',
              'status': 'PASS_FIVE_SUFFIX_CHECKPOINTS_THROUGH3328_AND_ORIGINAL_RECEIPTS',
              'Mac_F_FULLFSYNC_and_disk_SHA_readback': True, 'checkpoint_files': 5, 'files': 40, 'total_bytes': 502681793}
    scope = {role: inputs[key] for key, role in {'original_plan': 'original_plan', 'checkpoint_receipt': 'previous_checkpoint_receipt',
             'original_checkpoint': 'previous_checkpoint', 'previous_execution_plan': 'previous_execution_plan',
             'previous_claim': 'previous_claim', 'previous_terminal': 'previous_terminal', 'previous_admitted_policy': 'previous_admitted_policy',
             'previous_stop_review': 'previous_stop_review', 'previous_backup': 'previous_stop_backup'}.items()}
    scope.update(starting_cursor=3328, previous_source_commit=prepared['source_commit'],
                 new_plan_path=str(m.RESUME_PLAN), new_output_root=str(m.RESUME_ROOT))
    preserved = values[prior_inputs['ancestral_restore_proof']['path']]
    intermediate = values[prior_inputs['previous_initial_receipt']['path']]
    values.update({inputs[key]['path']: value for key, value in
                  (('previous_execution_plan', previous), ('checkpoint_receipt', receipt), ('previous_claim', claim),
                   ('previous_admitted_policy', policy), ('previous_terminal', terminal), ('previous_restore_proof', restore),
                   ('previous_stop_review', review), ('previous_backup', backup), ('resume_scope', scope),
                   ('previous_initial_receipt', initial), ('ancestral_restore_proof', preserved),
                   ('intervening_checkpoint_receipt', intermediate), ('intervening_restore_proof', intervening_restore))})
    return original, receipt, previous, claim, restore, values, live


def test_3328_plan_preserves_distinct2048_1792_512_producer_proofs(monkeypatch):
    original, receipt, previous, claim, restore, _, live = _resume3328_fixture(monkeypatch)
    plan = m.build_resume_plan(live)
    assert plan['starting_cursor'] == 3328 and plan['remaining_TRAIN_rows'] == 35062
    assert plan['limits'] == _ACTIVE_LIMITS == _suffix_limits(3328)
    assert plan['limits']['optimizer_steps'] == 366 and plan['limits']['model_forwards'] == 1751
    assert plan['checkpoint_global_cursors'] == [3584, 3694] and plan['limits']['checkpoint_writes'] == 2
    assert previous['starting_cursor'] == restore['position']['logical_cursor'] == 2048
    assert restore['expected_state_proof'] == previous['metadata']['receipt']['proof']
    assert receipt['producer_runtime'] == claim['runtime']
    assert plan['metadata']['trajectory'] == m.trajectory(original)
    assert plan['inputs']['earlier_execution_plan'] != plan['inputs']['original_resume_execution_plan']


@pytest.mark.parametrize('bad', ['restore_points1792', 'restore_proof512', 'initial_receipt',
                                'original_resume_ref', 'intervening_proof', 'intervening_receipt',
                                'preserved_proof', 'producer_original'])
def test_3328_rejects_immediate_vs_intervening_vs_original_confusion(monkeypatch, bad):
    original, receipt, previous, _, restore, values, live = _resume3328_fixture(monkeypatch)
    if bad == 'restore_points1792': restore['checkpoint'] = m.RESUME_INPUTS['intervening_checkpoint']
    elif bad == 'restore_proof512': restore['expected_state_proof'] = values[m.RESUME_INPUTS['ancestral_checkpoint_receipt']['path']]['proof']
    elif bad == 'initial_receipt': previous['metadata']['receipt']['logical_cursor'] = 1792
    elif bad == 'original_resume_ref': previous['inputs']['earlier_execution_plan'] = m.RESUME_INPUTS['earlier_execution_plan']
    elif bad == 'intervening_proof': values[m.RESUME_INPUTS['intervening_restore_proof']['path']]['expected_state_proof']['steps'] = '0' * 64
    elif bad == 'intervening_receipt': values[m.RESUME_INPUTS['intervening_checkpoint_receipt']['path']]['execution_plan'] = m.RESUME_INPUTS['earlier_execution_plan']
    elif bad == 'preserved_proof': values[m.RESUME_INPUTS['ancestral_restore_proof']['path']]['expected_state_proof']['steps'] = '0' * 64
    else: receipt['producer_runtime']['source_commit'] = original['source_commit']
    with pytest.raises(ValueError): m.build_resume_plan(live)


@pytest.mark.parametrize('bad', [None, 'size', 'bool_size', 'extra', 'sha'])
def test_resume_reference_optional_bytes_requires_exact_real_file_binding(tmp_path, bad):
    path = tmp_path / 'producer_plan.json'; path.write_text('{"producer":"resume003"}\n')
    binding = {**m.ref(path), 'bytes': path.stat().st_size}
    if bad == 'size': binding['bytes'] += 1
    elif bad == 'bool_size': binding['bytes'] = True
    elif bad == 'extra': binding['unexpected'] = 'must-not-be-ignored'
    elif bad == 'sha': binding['sha256'] = '0' * 64
    if bad:
        with pytest.raises(ValueError): m.resume_reference(binding, path)
    else: assert m.resume_reference(binding, path) == path


@pytest.mark.parametrize('wrong_bytes', [False, True])
def test_3328_stop_review_execution_plan_accepts_only_verified_optional_size(monkeypatch, wrong_bytes):
    _, _, _, _, _, values, live = _resume3328_fixture(monkeypatch)
    review = values[m.RESUME_INPUTS['previous_stop_review']['path']]
    review['execution_plan'] = {**review['execution_plan'], 'bytes': 1955378 + int(wrong_bytes)}
    previous_reference = m.resume_reference
    seen = []
    def reference(binding, expected_path=None):
        if 'bytes' in binding:
            seen.append(binding)
            assert expected_path == Path(m.RESUME_INPUTS['previous_execution_plan']['path'])
            m.need(type(binding['bytes']) is int and binding['bytes'] == 1955378, 'RESUME_REFERENCE_BYTES')
            return previous_reference({key: binding[key] for key in ('path', 'sha256')}, expected_path)
        return previous_reference(binding, expected_path)
    monkeypatch.setattr(m, 'resume_reference', reference)
    if wrong_bytes:
        with pytest.raises(ValueError, match='RESUME_REFERENCE_BYTES'): m.build_resume_plan(live)
    else:
        plan = m.build_resume_plan(live)
        assert plan['starting_cursor'] == 3328 and seen == [review['execution_plan']]
