"""Bounded metadata routing; no feature, checkpoint, model or optimizer loads."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

from gx1.contracts import entry_native_direction_research_campaign_v1 as research
from gx1.contracts import market_state_native_F_private_campaign_v1 as scope

SHA = '1' * 64


def seal(value, key):
    value = deepcopy(value)
    value.pop(key, None)
    return {**value, key: scope.campaign.canonical_sha256(value)}


@pytest.fixture
def metadata(monkeypatch):
    refs = deepcopy(scope.TA_M5_CENTERED_HEAD_COORDINATES_REFS)
    original_refs = deepcopy(scope.TA_M5_ORIGINAL_CUDA_PREFIX_REFS)
    owner = 'gx1/contracts/fixture.py'
    original = {'source_owners': {owner: {'path': str(scope.CANONICAL_REPO / owner), 'sha256': SHA}}}
    base = {
        'schema_version': scope.TA_M5_ORIGINAL_CUDA_PREFIX_RECIPE_SCHEMA,
        'source_repo': str(scope.CANONICAL_REPO), 'input_refs': original_refs,
        'input_manifest': {'path': '/fixture/manifest.json', 'sha256': SHA},
        'numerical_recipe': {'max_updates': 1849, 'batch_size': 16, 'grad_accum_steps': 1},
        'max_private_updates': 256, 'max_backwards': 256, 'max_checkpoint_loads': 2,
        'max_checkpoint_writes': 0, 'max_model_constructions': 1, 'max_optimizer_constructions': 1,
        'selected_fit_rows': 4096, 'fold_id': 1, 'training_enabled': True,
        'first_step_exact_required': True, 'final_full_state_exact_required': True,
        'canonical_resume_authority': False, 'candidate_selection': False, 'test_data_used': False,
    }
    documents = {
        'original_prefix_result': {
            'decision': 'PASS_TA_M5_ORIGINAL_CUDA_PREFIX_TRACE',
            'recipe': refs['original_prefix_recipe'], 'first_step_exact': True,
            'final_full_state_exact': True, 'physical_optimizer_updates': 256, 'parent_cursor': 4096,
            'skipped_batches': 0, 'source_unchanged': True, 'test_data_used': False,
            'trace': refs['original_trace']},
        'original_prefix_terminal': {'status': 'COMPLETE', 'exit_code': 0,
            'result': refs['original_prefix_result'], 'no_relaunch_under_this_plan': True,
            'test_data_used': False},
        'geometry_result': {'status': 'COMPLETE', 'source_unchanged': True,
            'means': refs['frozen_means'], 'mean_dtype': 'float64', 'mean_estimates': 2,
            'mean_population': 'all29574initialTRAINrows;neverlabelconditioned',
            'later_arrays_consumed': False, 'test_data_used': False},
        'geometry_terminal': {'status': 'COMPLETE', 'exit_code': 0, 'result': refs['geometry_result']},
        'short_panel_plan': {'fit_eval_ranges': scope.TA_M5_CENTERED_HEAD_COORDINATES_TRAIN_RANGES,
            'score_eval_ranges': scope.TA_M5_CENTERED_HEAD_COORDINATES_SCORE_RANGES,
            'worker': refs['short_panel_helper']},
        'trajectory_result': {'status': 'COMPLETE', 'source_unchanged': True, 'test_data_used': False,
            'trajectory': [
                {'measurement': refs['panel_' + label + '_measurement'],
                 'capture': refs['panel_' + label + '_capture'],
                 'checkpoint_identity': {
                     'checkpoint': original_refs['original_initial_checkpoint' if cursor == 0 else 'original_p4096_checkpoint'],
                     'cursor': cursor, 'optimizer_updates': cursor // 16,
                     'source_commit': scope.TA_M5_ORIGINAL_CUDA_PREFIX_SOURCE}}
                for label, cursor in (('initial', 0), ('p4096', 4096))]},
    }
    seen = []
    def checked(ref, **kwargs):
        seen.append((dict(ref), kwargs))
        return dict(ref)
    monkeypatch.setattr(scope, '_ta_m5_original_cuda_prefix_expected',
        lambda: (deepcopy(base), deepcopy(original), deepcopy(original)))
    monkeypatch.setattr(scope, 'read_ta_m5_original_cuda_prefix_origin',
        lambda launch: {'original_recipe': original, 'original_result': {}, 'initial_measurement': {},
                        'source_reference_recipe': original})
    monkeypatch.setattr(scope, '_read', lambda ref, label:
        deepcopy(next(value for key, value in documents.items() if refs[key] == ref)))
    monkeypatch.setattr(scope, '_ta_m5_original_cuda_prefix_sources', lambda *args: {'qualified': True})
    monkeypatch.setattr(scope.campaign, 'require_binding', checked)
    monkeypatch.setattr(scope.campaign, 'file_sha256', lambda path: SHA)
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0))
    return SimpleNamespace(documents=documents, seen=seen, base=base)


def recipe(metadata):
    return scope.build_ta_m5_centered_head_coordinates_recipe(source_commit='d' * 40)


def test_exact_counts_panels_and_numeric_identity(metadata):
    launch = recipe(metadata)
    assert launch['numerical_recipe'] == metadata.base['numerical_recipe']
    assert launch['numerical_recipe']['max_updates'] == 1849
    assert launch['max_private_updates'] == launch['max_training_forwards'] == 256
    assert launch['max_forwards'] == 328 and launch['max_evaluation_forwards'] == 72
    assert launch['max_checkpoint_loads'] == 2 and launch['max_full_state_restores'] == 3
    assert launch['max_candidate_evidence_writes'] == 1 and launch['max_checkpoint_writes'] == 0
    assert launch['max_model_seconds'] == 900
    assert launch['evaluate_both_panels_unconditionally'] is True
    assert launch['first_forward_raw_gradient_parity_required'] is True
    assert 'first_step_exact_required' not in launch and 'final_full_state_exact_required' not in launch
    assert launch['canonical_resume_authority'] is False and launch['candidate_selection'] is False
    assert launch['trace_worker']['path'] == str(scope.CANONICAL_REPO / scope.TA_M5_CENTERED_HEAD_COORDINATES_WORKER)
    for ref in scope.TA_M5_CENTERED_HEAD_COORDINATES_REFS.values():
        assert any(value == ref and options['verify_file'] is True for value, options in metadata.seen)


@pytest.mark.parametrize('key,value', [
    ('max_forwards', 329), ('max_training_forwards', 257), ('max_evaluation_forwards', 71),
    ('max_private_updates', 257), ('max_full_state_restores', 4), ('max_checkpoint_loads', 3),
    ('max_candidate_evidence_writes', 2), ('max_model_seconds', 901),
    ('evaluate_both_panels_unconditionally', False), ('canonical_resume_authority', True),
    ('fit_eval_ranges', [[0, 192]]), ('score_eval_ranges', [[0, 192]]),
])
def test_budget_panel_or_authority_change_rejected(metadata, key, value):
    launch = recipe(metadata)
    launch[key] = value
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='CENTERED_RECIPE_SCOPE'):
        scope.require_ta_m5_centered_head_coordinates_recipe(seal(launch, 'recipe_sha256'),
            source_repo=scope.CANONICAL_REPO, source_commit='d' * 40)


@pytest.mark.parametrize('document,key,value,reason', [
    ('geometry_result', 'mean_population', 'side_label_conditioned', 'CENTERED_MEAN_POPULATION'),
    ('geometry_result', 'later_arrays_consumed', True, 'CENTERED_MEAN_POPULATION'),
    ('original_prefix_result', 'final_full_state_exact', False, 'CENTERED_ORIGINAL_CONTROL'),
    ('short_panel_plan', 'score_eval_ranges', [[0, 192]], 'CENTERED_PANEL_DEFINITION'),
])
def test_unqualified_origin_rejected(metadata, document, key, value, reason):
    metadata.documents[document][key] = value
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match=reason):
        recipe(metadata)


def cursor_args(outcome='COMPLETE', updates=256):
    return dict(recipe_ref={'path': str(scope.TA_M5_CENTERED_HEAD_COORDINATES_RECIPE_PATH), 'sha256': SHA},
        outcome=outcome, optimizer_updates=updates, parent_cursor=updates * 16,
        parent_order_sha256=scope.TA_M5_EQUAL_LOSS_ORDER_SHA256, full_state=None,
        result={'path': str(scope.TA_M5_CENTERED_HEAD_COORDINATES_ROOT / 'RESULT.json'), 'sha256': SHA},
        batch_size=16)


def test_terminal_cursor_has_no_resume_state(metadata, monkeypatch):
    cursor = research.build_research_cursor(**cursor_args())
    assert cursor['schema_version'] == scope.TA_M5_CENTERED_HEAD_COORDINATES_CURSOR_SCHEMA
    assert cursor['full_state'] is None and cursor['canonical_resume_authority'] is False
    report = dict(schema_version=scope.TA_M5_CENTERED_HEAD_COORDINATES_RESULT_SCHEMA,
        recipe=cursor['recipe'], parent_cursor=4096, physical_optimizer_updates=256,
        parent_order_sha256=cursor['parent_order_sha256'], final_state=None,
        canonical_resume_authority=False, checkpoint_writes=0, test_data_used=False,
        decision='PASS_TA_M5_CENTERED_HEAD_COORDINATES_TECHNICAL',
        first_forward_raw_gradient_exact=True, control_p4096_restore_exact=True,
        function_equivalent_initial=True, evaluation_state_unchanged=True, all_panels_complete=True,
        skipped_batches=0, score_gate_pass=False,
        candidate_evidence={'path': str(scope.TA_M5_CENTERED_HEAD_COORDINATES_ROOT / 'CANDIDATE_EVIDENCE.pt'), 'sha256': SHA},
        counts=dict(model_constructions=1, optimizer_constructions=1, forwards=328,
            training_forwards=256, evaluation_forwards=72, backwards=256, optimizer_step_calls=256,
            optimizer_updates=256, checkpoint_loads=2, strict_full_state_restore_calls=3,
            checkpoint_writes=0, candidate_evidence_writes=1))
    monkeypatch.setattr(scope, '_read', lambda *args: report)
    assert research.require_research_cursor(cursor, expected_recipe=cursor['recipe'], batch_size=16) == cursor
    report['all_panels_complete'] = False
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='CENTERED_TECHNICAL_COMPLETION'):
        research.require_research_cursor(cursor, expected_recipe=cursor['recipe'], batch_size=16)


@pytest.mark.parametrize('field,value', [('full_state', {'path': '/fake.pt', 'sha256': SHA}),
                                         ('optimizer_updates', 257)])
def test_cursor_rejects_resume_or_extra_updates(metadata, field, value):
    args = cursor_args()
    args[field] = value
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='CENTERED_CURSOR'):
        research.build_research_cursor(**args)


def test_failed_cursor_preserves_actual_partial_count(metadata):
    cursor = research.build_research_cursor(**cursor_args('FAILED', 7))
    assert cursor['optimizer_updates'] == 7 and cursor['parent_cursor'] == 112
    assert cursor['full_state'] is None


def test_dispatch_only_calls_centered_worker(monkeypatch):
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.scripts import run_entry_native_direction_research_window_v1 as runner
    calls = []
    monkeypatch.setattr(cap, 'require_guarded_cuda_trainer_execution', lambda: {})
    monkeypatch.setattr(research, 'require_native_direction_research_context',
        lambda *args: ({}, {}, {}, {}, {'run_scope': research.TA_M5_CENTERED_HEAD_COORDINATES_SCOPE}))
    monkeypatch.setitem(sys.modules, 'gx1.audit.run_market_state_ta_m5_centered_head_coordinates_v1',
        SimpleNamespace(run_centered_head_coordinates=lambda **kwargs: calls.append(kwargs) or {'decision': 'sentinel'}))
    args = {'policy_path': Path('/bound/window.json'), 'policy_file_sha256': SHA,
            'progress_path': Path('/bound/progress.json')}
    assert runner.run_native_direction_research_window(**args) == {'decision': 'sentinel'}
    assert calls == [args]


@pytest.mark.parametrize('run_scope,ceiling', [
    (scope.TA_M5_CENTERED_HEAD_COORDINATES_SCOPE, 256),
    (scope.TA_M5_ORIGINAL_CUDA_PREFIX_SCOPE, 256),
    (scope.TA_M5_IMPORT_SCOPE, 0), (scope.SCOPE, 4),
])
def test_readiness_reports_actual_execution_ceiling(monkeypatch, run_scope, ceiling):
    repo = scope.CANONICAL_REPO
    recipe_ref = {'path': '/fixture/recipe.json', 'sha256': SHA}
    launch = {'run_scope': run_scope, 'admission_stage': 'fixture', 'runtime_root': '/fixture/runtime'}
    training = run_scope != scope.TA_M5_IMPORT_SCOPE
    gate = dict(recipe=recipe_ref, run_root=launch['runtime_root'], max_invocations=1,
                claimed_invocations=0, completed_invocations=0, relaunch_allowed=False)
    gate.update({key: True for key in ('execution_authorized', 'data_reads_authorized',
        'genuine_constructor_authorized', 'genuine_forward_authorized')})
    gate.update({key: training for key in ('genuine_backward_authorized', 'genuine_updates_authorized',
        'training_enabled', 'native_training_enabled')})
    gate.update({key: False for key in ('genuine_predictor_fit_authorized', 'native_launch_authorized',
        'normalization_fit_authorized', 'synthetic_test_execution_authorized', 'test_data_used',
        'exit_evaluation', 'broker_access_allowed')})
    policy = dict(schema_version='gx1_next_native_run_policy_v1', canonical_source_repo=str(repo),
        canonical_branch='work/gx1-current', training_module=research.MODULE, training_enabled=training,
        native_training_enabled=training, full_epoch_training_allowed=False, fixture=gate,
        gpu_clock_launcher={'path': '/fixture/clock.ps1', 'sha256': SHA})
    monkeypatch.setattr(scope.campaign, 'read_bound_json',
        lambda path, sha: policy if path.name == 'NEXT_RUN_POLICY.json' else {'recipe': recipe_ref})
    monkeypatch.setattr(scope.campaign, 'file_sha256', lambda path: SHA)
    monkeypatch.setattr(scope.campaign, 'require_binding', lambda ref, **kwargs: dict(ref))
    monkeypatch.setattr(research, 'require_research_window_policy', lambda value: value)
    monkeypatch.setattr(research, 'require_research_launch_recipe', lambda *args, **kwargs: launch)
    monkeypatch.setattr(research, '_is_ta_m5_import_recipe_ref', lambda ref: not training)
    for name in ('require_ta_m5_centered_head_coordinates_gate', 'require_ta_m5_original_cuda_prefix_gate',
                 'require_ta_m5_import_gate', 'require_gate'):
        monkeypatch.setattr(scope, name, lambda *args: None)
    monkeypatch.setattr(subprocess, 'check_output', lambda argv, **kwargs:
        'd' * 40 if 'rev-parse' in argv else 'work/gx1-current' if 'branch' in argv else '')
    ready = research.require_native_direction_research_readiness(repo,
        window_path=Path('/fixture/window.json'), window_file_sha256=SHA)
    assert ready['native_window_scope']['optimizer_step_ceiling'] == ceiling


@pytest.mark.parametrize('key,value', [
    ('status', 'PREPARING'), ('training_enabled', False), ('optimizer_steps_authorized', 257),
    ('model_forwards_authorized', 329), ('evaluate_both_panels_unconditionally', False),
])
def test_stage_gate_cannot_expand_or_skip_later_panel(metadata, key, value):
    launch = recipe(metadata)
    gate = dict(decision='ADMITTED', status='ADMITTED', ta_m5_centered_head_coordinates_authorized=True,
        fold_id=1, selected_fit_rows=4096, model_constructions_authorized=1,
        optimizer_constructions_authorized=1, model_forwards_authorized=328,
        backward_calls_authorized=256, optimizer_steps_authorized=256,
        training_enabled=True, native_training_enabled=True, genuine_updates_authorized=True,
        genuine_backward_authorized=True, genuine_predictor_fit_authorized=False,
        canonical_resume_authority=False, candidate_selection=False, test_data_used=False,
        evaluate_both_panels_unconditionally=True)
    scope.require_ta_m5_centered_head_coordinates_gate(gate, launch)
    gate[key] = value
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='CENTERED_POLICY_GATE'):
        scope.require_ta_m5_centered_head_coordinates_gate(gate, launch)


def test_progress_uses_256_actual_updates_not_numeric1849(metadata):
    cursor = research.build_research_cursor(**cursor_args())
    pointer = {'path': '/fixture/cursor.json', 'sha256': SHA}
    invocation = dict(kind=research.KIND, invocation_sha256=SHA, direction_research_recipe=cursor['recipe'],
        batch_size=16, optimizer_step_budget=256, checkpoint={'pointer_path': pointer['path']})
    progress = research.build_research_progress(window={'recipe': cursor['recipe']},
        plan={'plan_sha256': SHA}, invocation=invocation, checkpoint_pointer=pointer,
        optimizer_updates=256, parent_cursor=4096,
        parent_order_sha256=scope.TA_M5_EQUAL_LOSS_ORDER_SHA256, outcome='COMPLETE')
    assert progress['completed_units'] == progress['total_units'] == 256
    assert progress['parent_cursor'] == 4096
