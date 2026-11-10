"""Metadata-only limits and source bindings for the original CUDA prefix replay."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

from gx1.contracts import entry_native_direction_research_campaign_v1 as research
from gx1.contracts import market_state_native_F_private_campaign_v1 as scope

SHA = '1' * 64


def sealed(value, key):
    value = deepcopy(value)
    value.pop(key, None)
    return {**value, key: scope.campaign.canonical_sha256(value)}


@pytest.fixture
def metadata(monkeypatch):
    names = sorted(scope.TA_M5_ORIGINAL_CUDA_PREFIX_ROUTING_PATHS) + [
        scope._TA_PREFIX_BATCH_PATH, '.venv/lib/python3.10/site-packages/torch/optim/adamw.py',
        *['gx1/contracts/prefix_fixture_owner_' + str(i) + '.py' for i in range(79)]]
    old = {name: {'path': str(scope.CANONICAL_REPO / name), 'sha256': 'a' * 64}
           for name in names}
    original = {
        'source_owners': deepcopy(old), 'input_manifest': {'path': '/fixture/manifest.json', 'sha256': SHA},
        'input_refs': {key: {'path': '/fixture/' + key + '.json', 'sha256': SHA}
                       for key in (*scope.TA_M5_ORIGINAL_CUDA_PREFIX_CONSUMED_REF_KEYS,
                                   'unused_original_checkpoint')},
        'expected_view_sha256': SHA, 'expected_model_contract_sha256': SHA,
        'expected_loss_contract_sha256': SHA,
        'expected_initial_model_state_sha256': scope.TA_M5_EQUAL_LOSS_MODEL_SHA256,
        'numerical_recipe': {'contract_sha256': '5085579ff8ad90257412ad1d42406c6c481a1d272f0cf2849ae293da87974361',
            'max_updates': 1849, 'batch_size': 16, 'grad_accum_steps': 1, 'm5_equal_observation': True},
    }
    qualified = deepcopy(original)
    qualified['source_owners'][scope._TA_PREFIX_BATCH_PATH]['sha256'] = scope._TA_PREFIX_BATCH_OLD
    owners = deepcopy(qualified['source_owners'])
    owners[scope._TA_PREFIX_BATCH_PATH]['sha256'] = scope._TA_PREFIX_BATCH_CURRENT
    for name in scope.TA_M5_ORIGINAL_CUDA_PREFIX_ROUTING_PATHS:
        owners[name]['sha256'] = 'b' * 64
    worker = {'path': str(scope.CANONICAL_REPO / scope.TA_M5_ORIGINAL_CUDA_PREFIX_WORKER),
              'sha256': 'c' * 64}
    refs = scope.TA_M5_ORIGINAL_CUDA_PREFIX_REFS
    override = {'original_ref': qualified['source_owners'][scope._TA_PREFIX_BATCH_PATH],
                'current_ref': owners[scope._TA_PREFIX_BATCH_PATH]}
    report = {
        'source_commit': scope.TA_M5_ORIGINAL_CUDA_PREFIX_SOURCE, 'outcome': 'COMPLETE',
        'phase_state': 'COMPLETE', 'source_unchanged': True, 'test_data_used': False,
        'optimizer_updates': 1849, 'parent_cursor': 29574, 'skipped_batches': 0,
        'recipe': refs['original_recipe'], 'own_initial_state': refs['original_initial_checkpoint'],
        'initial_measurement': refs['initial_measurement'],
        'parent_order_sha256': scope.TA_M5_EQUAL_LOSS_ORDER_SHA256,
    }
    documents = {
        'original_recipe': original, 'source_reference_recipe': qualified, 'original_result': report,
        'initial_measurement': {'evaluated_checkpoint': refs['original_initial_checkpoint'],
            'evaluated_model_state_sha256': scope.TA_M5_EQUAL_LOSS_MODEL_SHA256,
            'optimizer_updates': 0, 'test_data_used': False},
        'contiguous_batch_review': {
            'plan': refs['mixed_plan'], 'result': refs['mixed_result'], 'review': refs['mixed_review'],
            'focused_tests': refs['contiguous_batch_tests'],
            'source_change': {'focused_tests_passed': 50, 'legacy_contiguous_route_preserved': True,
                'query_forward_targets_loss_suffix_byte_identical': True, 'runtime_source_override': override}},
        'contiguous_batch_tests': {'exit_code': 0, 'source': override['current_ref'],
                                  'query_forward_targets_loss_suffix_byte_identical': True},
        'mixed_plan': {'runtime_source_override': override}, 'mixed_result': {}, 'mixed_review': {},
    }
    by_path = {refs[key]['path']: value for key, value in documents.items()}
    monkeypatch.setattr(scope, '_read', lambda ref, label: deepcopy(by_path[ref['path']]))
    monkeypatch.setattr(scope.campaign, 'require_binding', lambda ref, **kwargs: dict(ref))
    hashes = {ref['path']: ref['sha256'] for ref in owners.values()}
    hashes[worker['path']] = worker['sha256']
    monkeypatch.setattr(scope.campaign, 'file_sha256', lambda path: hashes[str(path)])
    monkeypatch.setattr(scope, '_ta_m5_git_source_sha',
        lambda commit, name: (original if commit == scope.TA_M5_ORIGINAL_CUDA_PREFIX_SOURCE
                              else qualified)['source_owners'][name]['sha256'])
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0))
    return SimpleNamespace(original=original, qualified=qualified, owners=owners,
                           worker=worker, documents=by_path)


def recipe(metadata):
    return scope.build_ta_m5_original_cuda_prefix_recipe(source_commit='d' * 40)


def test_original_numerical_contract_and_256_execution_are_separate(metadata):
    value = recipe(metadata)
    assert value['numerical_recipe'] == metadata.original['numerical_recipe']
    assert value['numerical_recipe']['max_updates'] == 1849
    assert value['max_private_updates'] == value['max_forwards'] == value['max_backwards'] == 256
    assert value['selected_fit_rows'] == 4096
    assert value['max_checkpoint_loads'] == 2 and value['max_full_state_restores'] == 1
    assert value['max_checkpoint_writes'] == value['initial_evaluation_forwards'] == 0
    assert value['canonical_resume_authority'] is False
    assert research.guarded_research_power_limit(value) == 250


def test_consumed_original_inputs_are_explicitly_physically_bound(metadata, monkeypatch):
    seen = []
    def checked(ref, **kwargs):
        seen.append((dict(ref), kwargs))
        return dict(ref)
    monkeypatch.setattr(scope.campaign, 'require_binding', checked)
    value = recipe(metadata)
    expected = {'original_input_' + key: metadata.original['input_refs'][key]
                for key in scope.TA_M5_ORIGINAL_CUDA_PREFIX_CONSUMED_REF_KEYS}
    expected['original_input_manifest'] = metadata.original['input_manifest']
    assert {key: value['input_refs'][key] for key in expected} == expected
    assert all(any(ref == item and options['verify_file'] is True
                   for item, options in seen) for ref in expected.values())
    assert 'original_input_unused_original_checkpoint' not in value['input_refs']
    assert len(value['input_refs']) == len(scope.TA_M5_ORIGINAL_CUDA_PREFIX_REFS) + len(expected)


@pytest.mark.parametrize('key', [
    'original_input_candidate_tape', 'original_input_F_tape',
    'original_input_fold1_normalization', 'original_input_manifest',
])
def test_consumed_original_binding_cannot_be_replaced(metadata, key):
    launch = recipe(metadata)
    launch['input_refs'][key]['sha256'] = 'e' * 64
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='ORIGINAL_PREFIX_CONSUMED_INPUT_REFS'):
        scope.read_ta_m5_original_cuda_prefix_origin(launch)


@pytest.mark.parametrize('key,value', [
    ('max_private_updates', 257), ('max_forwards', 257), ('max_backwards', 257),
    ('selected_fit_rows', 4112), ('max_model_seconds', 721), ('max_trace_bytes', 67108865),
    ('max_checkpoint_writes', 1), ('initial_evaluation_forwards', 1),
    ('canonical_resume_authority', True), ('test_data_used', True),
])
def test_changed_execution_or_authority_is_rejected(metadata, key, value):
    launch = recipe(metadata)
    launch[key] = value
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='ORIGINAL_PREFIX_RECIPE_SCOPE'):
        scope.require_ta_m5_original_cuda_prefix_recipe(sealed(launch, 'recipe_sha256'),
            source_repo=scope.CANONICAL_REPO, source_commit='d' * 40)


def test_replacing_original_numerical_budget_is_rejected(metadata):
    launch = recipe(metadata)
    launch['numerical_recipe']['max_updates'] = 256
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='ORIGINAL_PREFIX_RECIPE_SCOPE'):
        scope.require_ta_m5_original_cuda_prefix_recipe(sealed(launch, 'recipe_sha256'),
            source_repo=scope.CANONICAL_REPO, source_commit='d' * 40)


def test_metadata_validation_cannot_disable_physical_source_checks(metadata):
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='ORIGINAL_PREFIX_PHYSICAL_SOURCE'):
        scope.require_ta_m5_original_cuda_prefix_recipe({}, source_repo=scope.CANONICAL_REPO,
                                                      source_commit='d' * 40, verify_files=False)


@pytest.mark.parametrize('which', ['core', 'runtime', 'old_batch', 'new_batch', 'worker'])
def test_source_bridge_rejects_unreviewed_changes(metadata, which):
    owners, qualified, worker = deepcopy(metadata.owners), deepcopy(metadata.qualified), deepcopy(metadata.worker)
    if which == 'core':
        owners['gx1/contracts/prefix_fixture_owner_0.py']['sha256'] = 'e' * 64
    elif which == 'runtime':
        owners['.venv/lib/python3.10/site-packages/torch/optim/adamw.py']['sha256'] = 'e' * 64
    elif which == 'old_batch':
        qualified['source_owners'][scope._TA_PREFIX_BATCH_PATH]['sha256'] = 'e' * 64
    elif which == 'new_batch':
        owners[scope._TA_PREFIX_BATCH_PATH]['sha256'] = 'e' * 64
    else:
        worker['path'] = '/tmp/unbound_worker.py'
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='ORIGINAL_PREFIX_'):
        scope._ta_m5_original_cuda_prefix_sources(owners, metadata.original, qualified, worker)


def cursor_args(outcome='COMPLETE', updates=256):
    return {
        'recipe_ref': {'path': str(scope.TA_M5_ORIGINAL_CUDA_PREFIX_RECIPE_PATH), 'sha256': SHA},
        'outcome': outcome, 'optimizer_updates': updates, 'parent_cursor': updates * 16,
        'parent_order_sha256': scope.TA_M5_EQUAL_LOSS_ORDER_SHA256,
        'full_state': scope.TA_M5_ORIGINAL_CUDA_PREFIX_REFS['original_p4096_checkpoint'] if outcome == 'COMPLETE' else None,
        'result': {'path': str(scope.TA_M5_ORIGINAL_CUDA_PREFIX_ROOT / 'RESULT.json'), 'sha256': SHA},
        'batch_size': 16,
    }


def test_complete_and_failed_cursor_cannot_grant_resume(metadata):
    complete = research.build_research_cursor(**cursor_args())
    failed = research.build_research_cursor(**cursor_args('FAILED', 7))
    assert complete['schema_version'] == scope.TA_M5_ORIGINAL_CUDA_PREFIX_CURSOR_SCHEMA
    assert complete['canonical_resume_authority'] is failed['canonical_resume_authority'] is False
    assert complete['full_state'] == scope.TA_M5_ORIGINAL_CUDA_PREFIX_REFS['original_p4096_checkpoint']
    assert failed['full_state'] is None and failed['parent_cursor'] == 112


@pytest.mark.parametrize('updates', [255, 257])
def test_only_exact_prefix_can_complete(metadata, updates):
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='ORIGINAL_PREFIX_CURSOR'):
        research.build_research_cursor(**cursor_args(updates=updates))


@pytest.mark.parametrize('broken', ['first_step_exact', 'final_full_state_exact', 'count', 'checkpoint_write'])
def test_cursor_verifies_result_parity_and_physical_budget(metadata, broken):
    value = research.build_research_cursor(**cursor_args())
    report = {
        'schema_version': scope.TA_M5_ORIGINAL_CUDA_PREFIX_RESULT_SCHEMA,
        'recipe': value['recipe'], 'parent_cursor': 4096, 'physical_optimizer_updates': 256,
        'parent_order_sha256': value['parent_order_sha256'], 'final_state': value['full_state'],
        'canonical_resume_authority': False, 'checkpoint_writes': 0, 'test_data_used': False,
        'decision': 'PASS_TA_M5_ORIGINAL_CUDA_PREFIX_TRACE',
        'first_step_exact': True, 'final_full_state_exact': True,
        'counts': {'model_constructions': 1, 'optimizer_constructions': 1, 'forwards': 256,
            'backwards': 256, 'optimizer_step_calls': 256, 'optimizer_updates': 256,
            'checkpoint_loads': 2, 'strict_full_state_restore_calls': 1, 'checkpoint_writes': 0},
    }
    metadata.documents[value['result']['path']] = report
    assert research.require_research_cursor(value, expected_recipe=value['recipe'], batch_size=16) == value
    if broken == 'count':
        report['counts']['forwards'] = 257
    elif broken == 'checkpoint_write':
        report['checkpoint_writes'] = 1
    else:
        report[broken] = False
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='ORIGINAL_PREFIX_CURSOR'):
        research.require_research_cursor(value, expected_recipe=value['recipe'], batch_size=16)


def test_research_progress_uses_exact_new_256_budget(metadata):
    value = research.build_research_cursor(**cursor_args())
    pointer = {'path': str(scope.TA_M5_ORIGINAL_CUDA_PREFIX_RUNTIME_ROOT / 'research-cursor/RESUME_CURSOR.json'), 'sha256': SHA}
    invocation = {'kind': research.KIND, 'batch_size': 16, 'optimizer_step_budget': 256,
        'invocation_sha256': SHA, 'direction_research_recipe': value['recipe'],
        'checkpoint': {'pointer_path': pointer['path']}}
    progress = research.build_research_progress(window={'recipe': value['recipe']},
        plan={'plan_sha256': SHA}, invocation=invocation, checkpoint_pointer=pointer,
        optimizer_updates=256, parent_cursor=4096,
        parent_order_sha256=scope.TA_M5_EQUAL_LOSS_ORDER_SHA256, outcome='COMPLETE')
    assert progress['completed_units'] == progress['total_units'] == 256
    progress['parent_cursor'] = 4080
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='ORIGINAL_CUDA_PREFIX_PROGRESS'):
        research.require_research_progress(sealed(progress, 'progress_sha256'),
            plan_sha256=SHA, invocation=invocation, expected_selection_receipt_sha256=None, verify_file=False)


def test_dispatch_only_calls_bound_new_worker(monkeypatch):
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.scripts import run_entry_native_direction_research_window_v1 as runner
    calls = []
    monkeypatch.setattr(cap, 'require_guarded_cuda_trainer_execution', lambda: {})
    monkeypatch.setattr(research, 'require_native_direction_research_context',
        lambda *args: ({}, {}, {}, {}, {'run_scope': research.TA_M5_ORIGINAL_CUDA_PREFIX_SCOPE}))
    name = 'gx1.audit.trace_market_state_ta_m5_original_cuda_prefix_v1'
    monkeypatch.setitem(sys.modules, name,
        SimpleNamespace(run_original_cuda_prefix=lambda **kwargs: calls.append(kwargs) or {'decision': 'sentinel'}))
    args = {'policy_path': Path('/bound/window.json'), 'policy_file_sha256': SHA,
            'progress_path': Path('/bound/progress.json')}
    assert runner.run_native_direction_research_window(**args) == {'decision': 'sentinel'}
    assert calls == [args]


@pytest.mark.parametrize('key,value', [
    ('status', 'PREPARING'), ('training_enabled', False),
    ('optimizer_steps_authorized', 257), ('selected_fit_rows', 4112),
    ('canonical_resume_authority', True), ('candidate_selection', True),
])
def test_admission_cannot_expand_or_relabel_replay(metadata, key, value):
    launch = recipe(metadata)
    gate = {
        'decision': 'ADMITTED', 'status': 'ADMITTED', 'ta_m5_original_cuda_prefix_authorized': True,
        'fold_id': 1, 'selected_fit_rows': 4096, 'model_constructions_authorized': 1,
        'optimizer_constructions_authorized': 1, 'model_forwards_authorized': 256,
        'backward_calls_authorized': 256, 'optimizer_steps_authorized': 256,
        'training_enabled': True, 'native_training_enabled': True,
        'genuine_updates_authorized': True, 'genuine_backward_authorized': True,
        'genuine_predictor_fit_authorized': False, 'canonical_resume_authority': False,
        'candidate_selection': False, 'test_data_used': False,
    }
    scope.require_ta_m5_original_cuda_prefix_gate(gate, launch)
    gate[key] = value
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='ORIGINAL_PREFIX_POLICY_GATE'):
        scope.require_ta_m5_original_cuda_prefix_gate(gate, launch)


def test_legacy_four_step_cursor_keeps_its_original_budget(metadata):
    args = cursor_args()
    args.update(recipe_ref={'path': '/fixture/old_recipe.json', 'sha256': SHA},
                optimizer_updates=4, parent_cursor=32,
                full_state={'path': '/fixture/old_state.pt', 'sha256': SHA})
    value = research.build_research_cursor(**args)
    assert value['schema_version'] == research.BPS_CURSOR_SCHEMA
    assert 'canonical_resume_authority' not in value
    args['optimizer_updates'] = 256
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='CURSOR_COUNTER'):
        research.build_research_cursor(**args)
