"""Synthetic metadata and exact-route checks; no NN, payload or CUDA work."""
import ast
import copy
from pathlib import Path

import pytest

from gx1.audit import native_provider_hlc_learning_v1 as m


def _fixture(monkeypatch):
    from gx1.contracts import market_state_ta_integrated_model_v1 as model_owner
    def ref(path):
        return {'path': str(path), 'sha256': 'a' * 64}
    runtime = {'source_commit': '1' * 40, 'gpu': {'uuid': 'synthetic'},
               'library_versions': {'torch': 'synthetic'}}
    contract = {'contract_sha256': '6d6ac22819e54c7b12aebcf4e1afe27b990d8a4d90d16500e352c7353d99cca0'}
    initial = {'model_state_sha256': '55ce06ac8df8fdf3892003d571a3902ffa9616909edb26d8a60372bd09432fb9',
               'rng_sha256': '343bc9e441680c73e99ed7fe89595ed2d423ac91b7f2071cb19d74043494305d',
               'model_contract_sha256': contract['contract_sha256'],
               'target_contract_sha256': 'target-v2', 'initial_only': True,
               'optimizer_updates': 0}
    recipe = {'passes': 1, 'train_batch_rows': 96, 'eval_batch_rows': 256,
              'checkpoint_every_optimizer_updates': 256, 'seed': 20260911,
              'scheduler': None, 'gradient_clip_norm': 1., 'optimizer': {'lr': .0001}}
    schedule = {'contract_sha256': 'ffe58ecfb95bb3e5958b31fa017db8277a56ab05c880d37c8ac383b962809631',
                'rows': 354550, 'batch_rows': 96, 'logical_batches': 3694,
                'batches_per_pass': 3694, 'optimizer_updates': 3694, 'passes': 1,
                'skipped_batches': 0, 'shuffle': False, 'drop_last': False,
                'update_flags': [True] * 3694,
                'source_rows_sha256': '9fa3aabd86cc726c0dca3e1ac9b6629bc549f3f2524626d69075fa255f382cae',
                'issue_ns_sha256': 'c0805f39affb9bd80e679e0b41f69a558ea975d0e479ddeee815c7dd370997ea'}
    meta = {'runtime': copy.deepcopy(runtime), 'input_binding': {'synthetic': True},
            'architecture': {'synthetic': True}, 'target_contract': {'contract_sha256': 'target-v2'},
            'prior': ref('/synthetic/prior.json'), 'truth_sha256': 'truth-v2'}
    inputs = {key: ref('/synthetic/' + key + '.json') for key in
              ('baseline_acceptance', 'baseline_initial_receipt', 'baseline_result')}
    inputs['baseline_plan'] = m.ALL_TF_BASE
    limits = {'model_constructions': 1, 'optimizer_constructions': 1,
              'feature_cache_payload_loads': 1, 'target_archive_loads': 1,
              'model_forwards': 5079, 'backbone_forwards': 5079, 'readout_forwards': 5079,
              'optimizer_steps': 3694, 'backward_calls': 3694, 'autograd_calls': 0,
              'skipped_batches': 0, 'checkpoint_writes': 16, 'checkpoint_readbacks': 16,
              'checkpoint_receipt_writes': 16, 'checkpoint_loads': 0, 'model_restores': 0}
    common = {'metadata': meta, 'model_contract': contract,
              'source_owners': {'synthetic_numeric.py': 'a' * 64}, 'backbone_gradient_policy': None}
    control = copy.deepcopy(common);control.update(recipe=copy.deepcopy(recipe), schedule=schedule, limits=limits)
    base = copy.deepcopy(common)
    replay = copy.deepcopy(common);replay.update(inputs=inputs, initial_reference=initial, recipe=copy.deepcopy(recipe))
    acceptance = {'plan': m.ALL_TF_REPLAY, 'optimizer_replay_qualified': True,
                  'all10_proofs_bitexact': True, 'all10_proofs': {str(j): 'b' * 64 for j in range(10)},
                  'both_lanes_complete_consumed': True, 'restored_nonzero_moments': {'backbone': 426, 'readout': 20},
                  'equal_TF_eval_parity_bitexact': True, 'finite_gradient_parameters_all_bitexact': True}
    initial_acceptance = {'full_initialbaseline_qualified': True, 'actual_NN_forward_rows': 354550,
                          'actual_reference_comparison': {'output_sha256': True, 'scores': True},
                          'source_and_each_segment_bookends_pass': True, 'initializer': initial,
                          'initial_receipt': inputs['baseline_initial_receipt'], 'result': inputs['baseline_result']}
    design = {'schema_version': 'gx1_full_population_None_comparison_design_v1',
              'TRAIN_issue_rows': 354550, 'timeframes': list(m.train.TF),
              'question': 'synthetic comparison', 'selection': 'entire original TRAIN clock',
              'primary_review_rule': 'fixed proper score comparison',
              'single_intended_numerical_change': 'M5_primary to None'}
    provenance = {'status': 'MATCHED_RETAINED_M5_PRIMARY_CONTROL_ELIGIBLE_FOR_NONE_TRAIN_DIAGNOSTIC',
                  'control_plan': m.ALL_TF_CONTROL, 'None_initial_base_plan': m.ALL_TF_BASE,
                  'None_initial_receipt': inputs['baseline_initial_receipt'],
                  'current_owner_byte_identical_to_historical': True,
                  'new_model_forwards': 0, 'new_checkpoint_loads': 0,
                  'None_model_state_sha256': initial['model_state_sha256'], 'RNG_sha256': initial['rng_sha256'],
                  'metadata_equal_fields': sorted(set(meta) - {'runtime'}),
                  'recipe': copy.deepcopy(recipe), 'schedule_contract_sha256': schedule['contract_sha256']}
    values = {m.ALL_TF_CONTROL['path']: control, m.ALL_TF_BASE['path']: base,
              m.ALL_TF_REPLAY['path']: replay, m.ALL_TF_REPLAY_ACCEPTANCE['path']: acceptance,
              m.ALL_TF_DESIGN['path']: design, m.ALL_TF_PROVENANCE['path']: provenance,
              inputs['baseline_acceptance']['path']: initial_acceptance,
              inputs['baseline_initial_receipt']['path']: initial, meta['prior']['path']: {}}
    monkeypatch.setattr(m, 'read', lambda binding: values[binding['path']])
    monkeypatch.setattr(m, 'ref', ref)
    monkeypatch.setattr(m.core, 'future_cuda_require_runtime', lambda runtime: None)
    monkeypatch.setattr(m, 'all_tf_source_bridge', lambda: {'old_numeric_bytes_and_AST_preserved': True})
    monkeypatch.setattr(m, 'all_tf_guard_bridge', lambda: {'synthetic': True})
    def model_contract(*args, **kwargs):
        assert kwargs['backbone_gradient_policy'] is None
        return copy.deepcopy(contract)
    monkeypatch.setattr(model_owner, 'native_future_state_model_contract', model_contract)
    return runtime, control, base, replay, acceptance, initial_acceptance, provenance


def test_none_builder_retains_full_control_and_own_initial_without_old_builder(monkeypatch):
    runtime, control, base, replay, _, _, _ = _fixture(monkeypatch)
    monkeypatch.setattr(m, 'build_plan', lambda *a, **k: pytest.fail('old M5 builder called'))
    before = copy.deepcopy((control, base, replay));plan = m.build_all_tf_plan(runtime)
    assert (control, base, replay) == before
    assert plan['backbone_gradient_policy'] is None
    assert plan['recipe'] == control['recipe'] and 'backbone_gradient_policy' not in plan['recipe']
    assert plan['schedule'] == control['schedule']
    assert plan['initial_reference'] == replay['initial_reference']
    assert plan['inputs']['baseline_result'] == replay['inputs']['baseline_result']
    assert plan['inputs']['baseline_initial_receipt'] == replay['inputs']['baseline_initial_receipt']
    assert plan['limits']['model_forwards'] == 3694 + 1385
    assert plan['limits']['optimizer_steps'] == 3694 and plan['limits']['checkpoint_writes'] == 16
    assert plan['limits']['checkpoint_loads'] == plan['limits']['model_restores'] == 0
    assert plan['source_owners'][m.ALL_TF_TEST] == plan['source_owners'][m.ALL_TF_GUARD_TEST] == 'a' * 64
    assert plan['scientific_scope']['new_epochs_or_selection'] is False
    assert plan['scientific_scope']['chronological_generalization'] == plan['scientific_scope']['economic_edge'] == 'UNKNOWN'


@pytest.mark.parametrize('bad', ['rows', 'tail', 'policy', 'initial', 'prior', 'replay', 'proofs',
                               'recipe', 'metadata', 'provenance', 'numeric', 'gpu'])
def test_none_builder_rejects_scientific_or_qualification_mismatch(monkeypatch, bad):
    runtime, control, base, replay, acceptance, initial_acceptance, provenance = _fixture(monkeypatch)
    if bad == 'rows': control['schedule']['rows'] = 20727
    elif bad == 'tail': control['schedule']['drop_last'] = True
    elif bad == 'policy': replay['backbone_gradient_policy'] = 'M5_primary'
    elif bad == 'initial': replay['initial_reference']['model_state_sha256'] = 'f' * 64
    elif bad == 'prior': initial_acceptance['actual_reference_comparison']['scores'] = False
    elif bad == 'replay': acceptance['optimizer_replay_qualified'] = False
    elif bad == 'proofs': acceptance['all10_proofs'].pop('9')
    elif bad == 'recipe': control['recipe']['optimizer']['lr'] = .1
    elif bad == 'metadata': base['metadata']['truth_sha256'] = 'foreign'
    elif bad == 'provenance': provenance['metadata_equal_fields'].pop()
    elif bad == 'numeric': replay['source_owners']['synthetic_numeric.py'] = 'f' * 64
    else: runtime['gpu']['uuid'] = 'foreign'
    with pytest.raises(ValueError, match='PROVIDER_HLC_LEARNING_'):
        m.build_all_tf_plan(runtime)


@pytest.mark.parametrize('cursor', [256, 512, 1792, 2048, 3328, 3584])
def test_recovery_budget_is_only_original_suffix_and_full_final_score(monkeypatch, cursor):
    runtime, control, *_ = _fixture(monkeypatch)
    monkeypatch.setattr(m.train, 'position', lambda schedule, at: {'optimizer_updates': at})
    limits, cursors = m.all_tf_resume_limits(control, cursor)
    assert limits['optimizer_steps'] == limits['backward_calls'] == 3694 - cursor
    assert limits['model_forwards'] == limits['backbone_forwards'] == limits['readout_forwards'] == 3694 - cursor + 1385
    assert cursors == list(range(cursor + 256, 3694, 256)) + [3694]
    assert limits['checkpoint_writes'] == limits['checkpoint_readbacks'] == limits['checkpoint_receipt_writes'] == len(cursors)
    assert limits['checkpoint_loads'] == limits['model_restores'] == 1
    assert limits['autograd_calls'] == limits['skipped_batches'] == 0


def test_additive_source_bridge_and_physical_none_routes_preserve_original_functions():
    assert m.all_tf_source_bridge()['old_numeric_bytes_and_AST_preserved'] is True
    tree = ast.parse((m.REPO / m.SELF).read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ('physical_all_tf', 'physical_all_tf_resume'):
        calls = [node for node in ast.walk(functions[name]) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'build_native_future_state_model']
        assert len(calls) == 1
        assert any(k.arg == 'backbone_gradient_policy' and isinstance(k.value, ast.Constant)
                   and k.value.value is None for k in calls[0].keywords)
    dispatches = [node for node in functions['main'].body[:2] if isinstance(node, ast.If)]
    assert len(dispatches) == 2


@pytest.mark.parametrize('bad', ['', 'a' * 63, 'A' * 64, '../' + 'a' * 64, True, None])
def test_recovery_paths_reject_noncanonical_sha(bad):
    with pytest.raises(ValueError, match='ALL_TF_RECOVERY_SHA256'):
        m.all_tf_recovery_root(bad)


@pytest.mark.parametrize('admitted', [False, True])
def test_distinct_one_use_main_and_recovery_gates(admitted):
    binding = {'path': str(m.ALL_TF_PLAN), 'sha256': 'a' * 64}
    main = m.all_tf_gate(binding, admitted);recovery = m.all_tf_resume_gate(binding, admitted, 512)
    for gate in (main, recovery):
        assert gate['remaining_invocations'] == int(admitted)
        assert gate['execution_authorized'] is gate['full_epoch_training_allowed'] is admitted
        assert all(gate[key] is False for key in ('additional_epochs_allowed', 'test_authorized', 'relaunch_allowed'))
    assert recovery['starting_cursor'] == 512 and recovery['new_epoch_allowed'] is False
