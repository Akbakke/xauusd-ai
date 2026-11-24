"""Synthetic phase-boundary recovery; no native NN, checkpoint decode or CUDA work."""
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gx1.audit import native_provider_hlc_learning_v1 as m
from tests.test_native_provider_hlc_all_tf_learning_v1 import _fixture as legacy_fixture
from tests.test_native_provider_hlc_all_tf_learning_guard_v1 import scope, _proof as guard_proof

SCOPE_V2 = 'gx1_native_provider_hlc_all_tf_recovery_scope_v2'
PLAN_V1 = 'gx1_native_provider_hlc_all_tf_remaining_epoch_plan_v1'
PLAN_V2 = 'gx1_native_provider_hlc_all_tf_remaining_epoch_plan_v2'


def _authenticated(monkeypatch, cursor=3694, producer_v2=True):
    runtime, control, base, replay, _, _, provenance = legacy_fixture(monkeypatch)
    binding_ref = {'path': '/synthetic/input_binding.json', 'sha256': 'a' * 64}
    norm_ref = {'path': '/synthetic/normalization.json', 'sha256': 'a' * 64}
    binding = {'input_normalization_sha256': 'normalization', 'feature_provider_contract_sha256': 'provider'}
    normalization = {'contract_sha256': 'normalization', 'feature_provider_contract_sha256': 'provider'}
    for candidate in (control, base, replay):
        candidate['metadata'].update(input_binding=copy.deepcopy(binding), input_binding_ref=binding_ref, normalization=norm_ref)
    provenance['metadata_equal_fields'] = sorted(set(base['metadata']) - {'runtime'})
    original = m.build_all_tf_plan(runtime)
    owner_bytes = b'synthetic unchanged numeric owner'
    original['source_owners'] = {'synthetic_numeric.py': hashlib.sha256(owner_bytes).hexdigest()}
    original_ref = {'path': str(m.ALL_TF_PLAN), 'sha256': 'a' * 64}
    producer_ref = {'path': str(m.ALL_TF_ROOT / 'RECOVERY_PLANS' / ('a' * 64 + '.json')), 'sha256': 'a' * 64}
    root = m.all_tf_recovery_root(producer_ref['sha256'])
    traj = m.trajectory(original)
    producer = {'schema_version': PLAN_V2 if producer_v2 else PLAN_V1, 'stage': m.ALL_TF_RECOVERY_STAGE,
                'output_root': str(m.ALL_TF_ROOT), 'output_directory_prefix': 'RECOVERY_',
                'metadata': {'original': original, 'original_plan': original_ref, 'trajectory': traj, 'runtime': runtime},
                'starting_cursor': 3584 if cursor == 3694 else 2560, 'model_contract': original['model_contract'],
                'initial_reference': original['initial_reference'], 'backbone_gradient_policy': None,
                'recipe': original['recipe'], 'schedule': original['schedule'], 'execution': m.EXECUTION,
                'source_commit': runtime['source_commit'], 'source_owners': copy.deepcopy(original['source_owners'])}
    if producer_v2:
        producer.update(train_end_checkpoint_before_score=True, score_only=False)
    def ref(path):
        return {'path': str(path), 'sha256': 'a' * 64}
    name = 'TRAIN_END' if cursor == 3694 else 'UPDATE%06d' % cursor
    scope_value = {'schema_version': SCOPE_V2, 'original_plan': original_ref, 'producer_plan': producer_ref,
                   'producer_claim': ref(root / 'CLAIM.json'), 'producer_terminal': ref(root / 'TERMINAL.json'),
                   'producer_admitted_policy': ref('/synthetic/admitted_policy.json'),
                   'checkpoint_receipt': ref(root / (name + '.json')), 'checkpoint': ref(root / (name + '.pt')),
                   'stop_review': ref('/synthetic/stop_review.json'), 'backup': ref('/synthetic/backup.json')}
    scores = {'INITIAL': {'reused_qualified_result': original['inputs']['baseline_result'], 'no_repeated_initial_forwards': True}}
    proof = {key: 'c' * 64 for key in ('trajectory', 'position', 'model_state', 'optimizer_state', 'scheduler_state',
                                     'rng_state', 'training', 'optimizer_parameter_names', 'steps', 'scores')}
    proof.update(trajectory=m.core.state_sha(traj), position=m.core.state_sha(m.train.position(original['schedule'], cursor)),
                 scheduler_state=m.core.state_sha(None), training=m.core.state_sha(True), scores=m.core.state_sha(scores))
    receipt = {'schema_version': 'gx1_native_future_state_training_checkpoint_receipt_v1',
               'checkpoint': scope_value['checkpoint'], 'original_plan': original_ref, 'execution_plan': producer_ref,
               'producer_lane': 'main', 'producer_runtime': runtime, 'logical_cursor': cursor, 'optimizer_updates': cursor,
               'trajectory_sha256': traj['contract_sha256'], 'proof': proof}
    claim = {'plan': producer_ref, 'policy_at_claim': ref(m.POLICY), 'runtime': runtime, 'remaining_invocations': 0,
             'TRAIN_diagnostic_only': True, 'process_identity': {'synthetic_stopped': True}}
    terminal = {'status': 'FAILED_CONSUMED', 'exit_code': 1, 'plan': producer_ref,
                'claim': scope_value['producer_claim'], 'remaining_invocations': 0, 'relaunch_allowed': False}
    review = {'execution_plan': producer_ref, 'original_plan': original_ref, 'process_stopped': True, 'completed_epoch': False,
              'last_durable_cursor': cursor, 'last_durable_checkpoint': scope_value['checkpoint'],
              'last_durable_receipt': scope_value['checkpoint_receipt']}
    backup = {'schema_version': 'gx1_bound_separate_disk_backup_ack_v1', 'Mac_F_FULLFSYNC_and_disk_SHA_readback': True,
              'checkpoint': scope_value['checkpoint'], 'checkpoint_receipt': scope_value['checkpoint_receipt']}
    values = {original_ref['path']: original, producer_ref['path']: producer,
              scope_value['checkpoint_receipt']['path']: receipt, scope_value['producer_claim']['path']: claim,
              scope_value['producer_terminal']['path']: terminal, scope_value['stop_review']['path']: review,
              scope_value['backup']['path']: backup,
              scope_value['producer_admitted_policy']['path']: {m.ALL_TF_RECOVERY_STAGE: m.all_tf_resume_gate(producer_ref, True, producer['starting_cursor'])},
              binding_ref['path']: binding, norm_ref['path']: normalization}
    scope_ref = ref('/synthetic/scope_v2.json'); values[scope_ref['path']] = scope_value
    previous_read = m.read
    monkeypatch.setattr(m, 'read', lambda value: values[value['path']] if value['path'] in values else previous_read(value))
    def synthetic_resume_reference(value, expected_path=None):
        m.need(type(value) is dict and set(value) == {'path', 'sha256'}, 'RESUME_REFERENCE_FIELDS')
        path = m.train.require_recovery_ref(value, expected_path=expected_path, suffix=Path(value['path']).suffix)
        m.need(m.ref(path) == value, 'RESUME_CANONICAL_REFERENCE')
        return path
    monkeypatch.setattr(m, 'resume_reference', synthetic_resume_reference)
    monkeypatch.setattr(m, 'validate_flags', lambda policy, admitted: None)
    monkeypatch.setattr(m.core, 'future_cuda_runtime_matches', lambda expected, actual: None)
    monkeypatch.setattr(m.core, 'future_cpu_parent_stopped', lambda identity: None)
    monkeypatch.setattr(m.subprocess, 'check_output', lambda *args, **kwargs: owner_bytes)
    monkeypatch.setattr(m, 'all_tf_sources', lambda *plans: copy.deepcopy(original['source_owners']))
    return SimpleNamespace(runtime=runtime, original=original, producer=producer, scope=scope_value, scope_ref=scope_ref,
                           receipt=receipt, terminal=terminal, review=review, backup=backup, values=values, scores=scores)


@pytest.mark.parametrize('cursor,forwards,updates,checkpoints,rows', [(3584, 1495, 110, 2, 10486), (3694, 1385, 0, 1, 0)])
def test_v2_builder_binds_exact_training_or_score_only_suffix(monkeypatch, cursor, forwards, updates, checkpoints, rows):
    fixture = _authenticated(monkeypatch, cursor)
    before = copy.deepcopy((fixture.original, fixture.producer, fixture.scope))
    plan = m.build_all_tf_resume_plan_v2(fixture.runtime, fixture.scope_ref)
    assert (fixture.original, fixture.producer, fixture.scope) == before
    assert plan['schema_version'] == PLAN_V2 and plan['score_only'] is (cursor == 3694)
    assert plan['train_end_checkpoint_before_score'] is True
    assert plan['remaining_TRAIN_rows'] == rows and plan['remaining_FINAL_rows'] == 354550
    assert plan['recipe'] == fixture.original['recipe'] and plan['schedule'] == fixture.original['schedule']
    assert plan['initial_reference'] == fixture.original['initial_reference'] and plan['new_INITIAL_writes'] == 0
    assert plan['source_owners'][m.ALL_TF_V2_TEST] == 'a' * 64
    limits = plan['limits']
    assert limits['model_forwards'] == limits['backbone_forwards'] == limits['readout_forwards'] == forwards
    assert limits['optimizer_steps'] == limits['backward_calls'] == updates
    assert limits['checkpoint_writes'] == limits['checkpoint_readbacks'] == limits['checkpoint_receipt_writes'] == checkpoints
    assert limits['checkpoint_loads'] == limits['model_restores'] == 1
    assert limits['autograd_calls'] == limits['skipped_batches'] == 0


def test_v1_scope_and_suffix_budget_remain_distinct(monkeypatch):
    fixture = _authenticated(monkeypatch, 3584, producer_v2=False)
    with pytest.raises(ValueError, match='ALL_TF_RECOVERY_SCOPE_SCHEMA$'):
        m.all_tf_recovery_metadata(fixture.original, fixture.scope)
    fixture.scope['schema_version'] = 'gx1_native_provider_hlc_all_tf_recovery_scope_v1'
    receipt, _, _ = m.all_tf_recovery_metadata(fixture.original, fixture.scope)
    assert receipt['logical_cursor'] == 3584
    plan = m.build_all_tf_resume_plan(fixture.runtime, fixture.scope_ref)
    assert plan['schema_version'] == PLAN_V1 and plan['limits']['checkpoint_writes'] == 1
    assert plan['limits']['optimizer_steps'] == 110 and plan['limits']['model_forwards'] == 1495
    assert 'score_only' not in plan and 'train_end_checkpoint_before_score' not in plan
    assert m.ALL_TF_V2_TEST not in plan['source_owners']


def test_v1_never_accepts_train_end_as_legacy_update(monkeypatch):
    fixture = _authenticated(monkeypatch, 3694, producer_v2=False)
    fixture.scope['schema_version'] = 'gx1_native_provider_hlc_all_tf_recovery_scope_v1'
    with pytest.raises(ValueError, match='ALL_TF_RECOVERY_EXACT_DURABLE_CURSOR$'):
        m.all_tf_recovery_metadata(fixture.original, fixture.scope)


@pytest.mark.parametrize('bad', ['scope_v1', 'UPDATE3694', 'FINAL', 'cursor3693', 'bool_cursor', 'optimizer_cursor',
                               'receipt_checkpoint', 'proof_training', 'proof_position', 'proof_scores', 'missing_proof',
                               'producer_v1', 'producer_no_boundary', 'producer_score_only', 'producer_equal_cursor',
                               'producer_completed', 'producer_result', 'foreign_original', 'backup_missing'])
def test_v2_rejects_unqualified_or_completed_endpoint(monkeypatch, bad):
    f = _authenticated(monkeypatch)
    # Every attack starts from an accepted synthetic chain; an unrelated first
    # reference failure cannot make a negative assertion pass.
    accepted, _, _ = m.all_tf_recovery_metadata(f.original, f.scope, phase_recovery=True)
    assert accepted is f.receipt
    expected = {
        'scope_v1': 'ALL_TF_RECOVERY_SCOPE_SCHEMA',
        'UPDATE3694': 'RECOVERY_REFERENCE_PATH', 'FINAL': 'RECOVERY_REFERENCE_PATH',
        'cursor3693': 'ALL_TF_RECOVERY_EXACT_DURABLE_CURSOR', 'bool_cursor': 'ALL_TF_RECOVERY_EXACT_DURABLE_CURSOR',
        'optimizer_cursor': 'ALL_TF_RECOVERY_EXACT_DURABLE_CURSOR', 'receipt_checkpoint': 'ALL_TF_RECOVERY_EXACT_DURABLE_CURSOR',
        'proof_training': 'ALL_TF_RECOVERY_AUTHENTIC_STATE_PROOFS', 'proof_position': 'ALL_TF_RECOVERY_AUTHENTIC_STATE_PROOFS',
        'proof_scores': 'ALL_TF_RECOVERY_AUTHENTIC_STATE_PROOFS', 'missing_proof': 'RECOVERY_CHECKPOINT_PROOF_SCHEMA',
        'producer_v1': 'ALL_TF_RECOVERY_AUTHENTIC_TRAIN_END_PRODUCER',
        'producer_no_boundary': 'ALL_TF_RECOVERY_AUTHENTIC_TRAIN_END_PRODUCER',
        'producer_score_only': 'ALL_TF_RECOVERY_AUTHENTIC_TRAIN_END_PRODUCER',
        'producer_equal_cursor': 'ALL_TF_RECOVERY_PREVIOUS_SUFFIX_LINEAGE',
        'producer_completed': 'ALL_TF_RECOVERY_FAILED_CONSUMED_PRODUCER',
        'producer_result': 'ALL_TF_RECOVERY_FAILED_CONSUMED_PRODUCER',
        'foreign_original': 'ALL_TF_RECOVERY_EXACT_DURABLE_CURSOR', 'backup_missing': 'ALL_TF_RECOVERY_DURABLE_BACKUP',
    }[bad]
    if bad == 'scope_v1': f.scope['schema_version'] = 'gx1_native_provider_hlc_all_tf_recovery_scope_v1'
    elif bad in ('UPDATE3694', 'FINAL'):
        stem = 'UPDATE003694' if bad == 'UPDATE3694' else 'FINAL'
        f.scope['checkpoint']['path'] = str(Path(f.scope['checkpoint']['path']).with_name(stem + '.pt'))
        old = f.scope['checkpoint_receipt']['path']
        f.scope['checkpoint_receipt']['path'] = str(Path(old).with_name(stem + '.json'))
        f.values[f.scope['checkpoint_receipt']['path']] = f.receipt
    elif bad == 'cursor3693': f.receipt.update(logical_cursor=3693, optimizer_updates=3693)
    elif bad == 'bool_cursor': f.receipt.update(logical_cursor=True, optimizer_updates=True)
    elif bad == 'optimizer_cursor': f.receipt['optimizer_updates'] = 3693
    elif bad == 'receipt_checkpoint': f.receipt['checkpoint'] = {'path': '/foreign/TRAIN_END.pt', 'sha256': 'a' * 64}
    elif bad == 'proof_training': f.receipt['proof']['training'] = m.core.state_sha(False)
    elif bad == 'proof_position': f.receipt['proof']['position'] = m.core.state_sha(m.train.position(f.original['schedule'], 3584))
    elif bad == 'proof_scores': f.receipt['proof']['scores'] = m.core.state_sha({**f.scores, 'PASS1': {'partial': True}})
    elif bad == 'missing_proof': f.receipt['proof'].pop('model_state')
    elif bad == 'producer_v1': f.producer['schema_version'] = PLAN_V1
    elif bad == 'producer_no_boundary': f.producer['train_end_checkpoint_before_score'] = False
    elif bad == 'producer_score_only': f.producer['score_only'] = True
    elif bad == 'producer_equal_cursor': f.producer['starting_cursor'] = 3694
    elif bad == 'producer_completed': f.terminal['status'] = 'COMPLETE'
    elif bad == 'producer_result': f.terminal['result'] = {'path': '/foreign/result.json', 'sha256': 'a' * 64}
    elif bad == 'foreign_original': f.receipt['original_plan'] = {'path': '/foreign/original.json', 'sha256': 'a' * 64}
    else: f.backup['Mac_F_FULLFSYNC_and_disk_SHA_readback'] = False
    with pytest.raises(ValueError, match=expected + '$'):
        m.all_tf_recovery_metadata(f.original, f.scope, phase_recovery=True)


@pytest.mark.parametrize('bad', [0, 3585, 3693, 3695, True, 3694.0])
def test_v2_budget_rejects_non_checkpoint_cursor(monkeypatch, bad):
    _, control, *_ = legacy_fixture(monkeypatch)
    with pytest.raises(ValueError, match='ALL_TF_RECOVERY_PHASE_CURSOR$'):
        m.all_tf_resume_limits(control, bad, phase_recovery=True)


def _physical(monkeypatch, start, fail_score=False):
    f = _authenticated(monkeypatch, start)
    plan = m.build_all_tf_resume_plan_v2(f.runtime, f.scope_ref)
    original = plan['metadata']['original']; events = []; saved = {}; disk = {}
    rng = {'state': {'position': 0}}
    class Endpoint:
        def __init__(self): self.hooks = []
        def register_forward_pre_hook(self, callback):
            self.hooks.append(callback)
            return SimpleNamespace(remove=lambda: self.hooks.remove(callback))
        def fire(self):
            for callback in self.hooks: callback(self, ())
    class Model(Endpoint):
        def __init__(self):
            super().__init__(); self.state = {'position': 0}; self.training = True
            self.research_objective = original['model_contract']; self.backbone = Endpoint(); self.readout = Endpoint()
        def state_dict(self): return self.state
        def parameters(self): return []
        def named_parameters(self): return [('synthetic', None)]
        def forward_event(self):
            for endpoint in (self, self.backbone, self.readout): endpoint.fire()
    class Optimizer:
        def __init__(self, *args, **kwargs): self.state = {}
        def state_dict(self): return self.state
        def zero_grad(self, **kwargs): events.append('zero_grad')
    model = Model(); optimizer = Optimizer()
    original['initial_reference']['model_state_sha256'] = m.core.state_sha(model.state_dict())
    original['initial_reference']['rng_sha256'] = m.core.state_sha(rng['state'])
    monkeypatch.setattr(m.core, 'training_rng', lambda: copy.deepcopy(rng['state']))
    monkeypatch.setattr(m.core, 'future_cuda_seed_and_runtime', lambda value: None)
    from gx1.contracts import market_state_ta_integrated_model_v1 as model_owner
    from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
    monkeypatch.setattr(model_owner, 'build_native_future_state_model', lambda **kwargs: model)
    monkeypatch.setattr(torch.optim, 'AdamW', lambda *args, **kwargs: optimizer)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda: 0)
    for target, name in ((torch.autograd, 'grad'), (torch.autograd, 'backward'), (torch.Tensor, 'backward')):
        monkeypatch.setattr(target, name, lambda *args, **kwargs: pytest.fail('actual autograd forbidden'))
    source_rows = np.arange(354550, dtype=np.int64)
    context = {'source_rows': source_rows, 'truth': {'issue_ns': source_rows.copy()},
               'target_contract': original['metadata']['target_contract'],
               'normalization': f.values[original['metadata']['normalization']['path']],
               'input_binding': original['metadata']['input_binding'], 'prior': {}}
    batches = m.core.all_score_batches(source_rows, 96)
    original['schedule']['batch_source_rows_sha256'] = [m.core.digest(source_rows[take]) for take in batches]
    original['schedule']['batch_issue_ns_sha256'] = original['schedule']['batch_source_rows_sha256'].copy()
    traj = m.trajectory(original); plan['metadata']['trajectory'] = traj
    checkpoint = {'schema_version': 'gx1_native_future_state_training_checkpoint_v1', 'trajectory': traj,
                  'producer': {'original_plan': plan['metadata']['original_plan'], 'lane': 'main', 'runtime': f.runtime},
                  'position': m.train.position(original['schedule'], start), 'model_state': {'position': start},
                  'optimizer_state': {'position': start}, 'scheduler_state': None, 'rng_state': {'position': start},
                  'training': True, 'optimizer_parameter_names': ['synthetic'],
                  'steps': [{'cursor': j + 1} for j in range(start)], 'scores': copy.deepcopy(f.scores)}
    plan['metadata']['receipt']['proof'] = m.train.checkpoint_proof(checkpoint)
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: copy.deepcopy(checkpoint))
    def restore(actual_model, actual_optimizer, cp, **kwargs):
        assert kwargs['expected_cursor'] == start and kwargs['expected_state_proof'] == m.train.checkpoint_proof(cp)
        model.state = copy.deepcopy(cp['model_state']); optimizer.state = copy.deepcopy(cp['optimizer_state'])
        rng['state'] = copy.deepcopy(cp['rng_state']); return cp['position']
    monkeypatch.setattr(m.train, 'strict_restore', restore)
    monkeypatch.setattr(m.adapter, 'load_context', lambda metadata: context)
    monkeypatch.setattr(m.adapter, 'packet', lambda context, take: {'truth': {'rows': len(take)}})
    monkeypatch.setattr(objective, '_native_future_state_truth', lambda *args, **kwargs: {'activity': np.array([True])})
    def training_step(actual_model, actual_optimizer, packet, original_plan, counts, cursor):
        assert start < 3694, 'score-only recovery called training_step'
        model.forward_event(); model.state = {'position': cursor + 1}; optimizer.state = {'position': cursor + 1}
        rng['state'] = {'position': cursor + 1}; counts['optimizer_steps'] += 1; counts['backward_calls'] += 1
        events.append('update'); return {'cursor': cursor + 1}
    monkeypatch.setattr(m.train, 'training_step', training_step)
    def fake_cpu_publish(path, cp, counts):
        before = m.core.state_sha((model.state_dict(), rng['state']))
        saved[path.name] = copy.deepcopy(cp); events.append(path.name)
        counts['checkpoint_writes'] += 1; counts['checkpoint_readbacks'] += 1
        assert m.core.state_sha((model.state_dict(), rng['state'])) == before
        return {'path': str(path), 'sha256': 'a' * 64}
    # Keep real future_cuda_publish_checkpoint and publish_checkpoint_receipt:
    # their RNG bookend, ten proofs, counters and receipt readback run unchanged.
    monkeypatch.setattr(m.core, 'future_cpu_publish_checkpoint', fake_cpu_publish)
    def publish(path, value):
        disk[str(path)] = copy.deepcopy(value)
        return {'path': str(path), 'sha256': 'a' * 64}
    monkeypatch.setattr(m.train, 'publish', publish)
    monkeypatch.setattr(m.train, 'read', lambda binding: disk[binding['path']])
    monkeypatch.setattr(m, 'pub', lambda root, name, value: publish(root / name, value))
    monkeypatch.setattr(m.train, 'release_phase_allocator_cache', lambda *args, **kwargs: {'synthetic': True})
    def final_score(actual_model, actual_context, *, batch_rows, device):
        events.append('first_score')
        assert events.index('zero_grad') < events.index('first_score')
        if start < 3694:
            assert events.index('TRAIN_END.pt') < events.index('first_score')
            assert saved['TRAIN_END.pt']['position']['logical_cursor'] == 3694
            assert len(saved['TRAIN_END.pt']['steps']) == 3694 and saved['TRAIN_END.pt']['scores'] == f.scores
        assert actual_context is context and batch_rows == 256 and device == 'cuda'
        if fail_score:
            model.forward_event()
            raise RuntimeError('synthetic interrupted full FINAL')
        for _ in range(1385): model.forward_event()
        return {'rows': 354550, 'batch_count': 1385, 'scores': {'per_tf': {tf: {task: {'count': 0} for task in ('activity', 'side')} for tf in m.train.TF}}}, {'synthetic': np.array([1.], dtype=np.float32)}
    monkeypatch.setattr(m.adapter, 'score_final', final_score)
    monkeypatch.setattr(m, 'quality_panels', lambda *args: {'all5_pooled_side_TRAIN_fit': False, 'all_TF': {tf: {task: {'count': 0} for task in ('activity', 'side')} for tf in m.train.TF}})
    monkeypatch.setattr(m.np, 'savez_compressed', lambda stream, **kwargs: stream.write(b'synthetic logits'))
    binding = {'path': '/synthetic/execution.json', 'sha256': 'b' * 64}
    return plan, binding, f, events, saved, disk


@pytest.mark.parametrize('start', [3584, 3694])
def test_real_resume_control_flow_preserves_train_end_before_complete_final(monkeypatch, start):
    plan, binding, f, events, saved, disk = _physical(monkeypatch, start)
    report = m.physical_all_tf_resume(plan, binding, {'synthetic': True}, f.runtime)
    assert report['counts'] == plan['limits'] and report['new_optimizer_updates'] == 3694 - start
    assert report['whole_FINAL_score']['rows'] == 354550 and report['whole_FINAL_score']['batch_count'] == 1385
    assert events.count('update') == 3694 - start and events.count('first_score') == 1
    assert report['score_only'] is (start == 3694)
    assert saved['FINAL.pt']['scores']['PASS1'] == report['whole_FINAL_score']
    assert saved['FINAL.pt']['position']['logical_cursor'] == 3694
    if start < 3694:
        train_end = saved['TRAIN_END.pt']; final = saved['FINAL.pt']
        for key in ('model_state', 'optimizer_state', 'rng_state', 'position', 'steps', 'training'):
            assert m.core.state_sha(train_end[key]) == m.core.state_sha(final[key])
        assert 'PASS1' not in train_end['scores']
    else:
        assert 'TRAIN_END.pt' not in saved
        assert report['train_end_checkpoint_receipt'] == plan['inputs']['checkpoint_receipt']
        assert events.count('update') == report['counts']['backward_calls'] == report['counts']['autograd_calls'] == 0


def test_interrupted_final_keeps_train_end_and_never_publishes_final_checkpoint(monkeypatch):
    plan, binding, f, events, saved, disk = _physical(monkeypatch, 3584, fail_score=True)
    with pytest.raises(RuntimeError, match='interrupted full FINAL'):
        m.physical_all_tf_resume(plan, binding, {'synthetic': True}, f.runtime)
    assert saved['TRAIN_END.pt']['position']['logical_cursor'] == 3694
    assert saved['TRAIN_END.pt']['scores'] == f.scores and 'FINAL.pt' not in saved
    assert not any(Path(path).name in ('FINAL_QUALITY.json', 'FINAL_LOGITS.npz', 'FINAL.json') for path in disk)


def test_existing_checkpoint_wrapper_rejects_rng_mutation(monkeypatch):
    state = {'rng': 1}
    monkeypatch.setattr(m.core, 'training_rng', lambda: dict(state))
    def mutating_publish(*args):
        state['rng'] = 2
        return {'path': '/synthetic/TRAIN_END.pt', 'sha256': 'a' * 64}
    monkeypatch.setattr(m.core, 'future_cpu_publish_checkpoint', mutating_publish)
    with pytest.raises(ValueError, match='FUTURE_CUDA_DISK_CHECKPOINT_ALL_RNG'):
        m.core.future_cuda_publish_checkpoint(Path('/synthetic/TRAIN_END.pt'), {}, {})


@pytest.mark.parametrize('scope', ['resume'], indirect=True)
@pytest.mark.parametrize('schema,accepted', [(PLAN_V1, True), (PLAN_V2, True), ('gx1_native_provider_hlc_all_tf_remaining_epoch_plan_v3', False)])
def test_guard_accepts_only_explicit_v1_or_v2_resume(scope, schema, accepted):
    plan = copy.deepcopy(scope['data']); plan['schema_version'] = schema; scope['bind'](plan)
    if accepted:
        assert guard_proof(scope)['plan']['sha256'] == scope['sha']
    else:
        with pytest.raises((ValueError, RuntimeError)):
            guard_proof(scope)


@pytest.mark.parametrize('scope', ['main'], indirect=True)
def test_guard_never_admits_v2_as_new_main(scope):
    plan = copy.deepcopy(scope['data']); plan['schema_version'] = PLAN_V2; scope['bind'](plan)
    with pytest.raises((ValueError, RuntimeError)):
        guard_proof(scope)
