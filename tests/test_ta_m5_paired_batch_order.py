"""Bounded paired-order metadata; no model, payload or checkpoint loads."""
from copy import deepcopy
from types import SimpleNamespace
import subprocess

import pytest

from gx1.contracts import market_state_native_F_private_campaign_v1 as scope

SHA = '1' * 64


def seal(value, key):
    value = deepcopy(value)
    value.pop(key, None)
    return {**value, key: scope.campaign.canonical_sha256(value)}


@pytest.fixture
def metadata(monkeypatch):
    refs = deepcopy(scope.TA_M5_PAIRED_BATCH_ORDER_REFS)
    original_refs = deepcopy(scope.TA_M5_ORIGINAL_CUDA_PREFIX_REFS)
    original = {'source_owners': {'gx1/contracts/fixture.py': {
        'path': str(scope.CANONICAL_REPO / 'gx1/contracts/fixture.py'), 'sha256': SHA}}}
    base = dict(schema_version=scope.TA_M5_ORIGINAL_CUDA_PREFIX_RECIPE_SCHEMA,
        source_repo=str(scope.CANONICAL_REPO), input_refs=original_refs,
        numerical_recipe={'max_updates': 1849, 'batch_size': 16, 'grad_accum_steps': 1},
        max_model_constructions=1, max_optimizer_constructions=1, max_checkpoint_writes=0,
        parent_order_rows=29574, parent_order_sha256=scope.TA_M5_EQUAL_LOSS_ORDER_SHA256,
        first_step_exact_required=True, final_full_state_exact_required=True,
        full_epoch_training_allowed=False, test_data_used=False)
    baseline = dict(initial_model_sha256=scope.TA_M5_EQUAL_LOSS_MODEL_SHA256,
                    short_INITIAL0_used=False)
    positions = [i for a, b in scope.TA_M5_PAIRED_BATCH_ORDER_SCORE_RANGES for i in range(a, b)]
    documents = {
        'original_prefix_result': dict(decision='PASS_TA_M5_ORIGINAL_CUDA_PREFIX_TRACE',
            recipe=refs['original_prefix_recipe'], first_step_exact=True, final_full_state_exact=True,
            physical_optimizer_updates=256, parent_cursor=4096, skipped_batches=0,
            source_unchanged=True, test_data_used=False, trace=refs['original_trace']),
        'original_prefix_terminal': dict(status='COMPLETE', exit_code=0,
            result=refs['original_prefix_result'], no_relaunch_under_this_plan=True, test_data_used=False),
        'objective_audit_result': dict(status='COMPLETE', source_unchanged=True,
            test_data_used=False, later_arrays_consumed=False, plan=refs['objective_audit_plan'],
            baseline=deepcopy(baseline), inputs=dict(old_fit_cache=refs['initial_fit_cache'],
                readout_outputs=refs['imported_predictions'], fullfit_recipe=original_refs['original_recipe'],
                initial_measurement=original_refs['initial_measurement']),
            profile=dict(fit_rows=29574, fit_batches=1849, batch_size=16, pass_index=0,
                fit_cache_sha256=refs['initial_fit_cache']['sha256'], later_data_used_for_normalization=False)),
        'objective_audit_review': dict(status='PASS_INDEPENDENT_CACHED_OBJECTIVE_REVIEW',
            result=refs['objective_audit_result'], plan=refs['objective_audit_plan'],
            test_data_used=False, later_arrays_consumed=False, rows=29574, batches=1849,
            last_batch_rows=6, baseline=deepcopy(baseline)),
        'centered_result': dict(decision='PASS_TA_M5_CENTERED_HEAD_COORDINATES_TECHNICAL',
            recipe=refs['centered_recipe'], source_unchanged=True, test_data_used=False,
            evaluation_state_unchanged=True, measurement_refs={'INITIAL_LATER': refs['initial_later_measurement']},
            captures={'INITIAL_LATER': refs['initial_later_capture']},
            restore_proofs={'INITIAL_BASELINE': {'all_numerical_state_exact': True}}),
        'centered_terminal': dict(status='COMPLETE', exit_code=0,
            result=refs['centered_result'], no_relaunch_under_this_plan=True),
        'centered_score_review': dict(status='PASS_INDEPENDENT_CACHED_SCORES_NEGATIVE_COMPARISON',
            result=refs['centered_result'], test_data_used=False),
        'initial_later_measurement': dict(evaluated_checkpoint=original_refs['original_initial_checkpoint'],
            evaluated_model_state_sha256=scope.TA_M5_EQUAL_LOSS_MODEL_SHA256,
            capture=refs['initial_later_capture'], fixed_subset_positions=positions,
            optimizer_updates=0, fold_id=1, test_data_used=False),
    }
    seen = []
    def checked(ref, **kwargs):
        seen.append((dict(ref), kwargs))
        return dict(ref)
    monkeypatch.setattr(scope, '_ta_m5_original_cuda_prefix_expected',
        lambda: (deepcopy(base), deepcopy(original), deepcopy(original)))
    monkeypatch.setattr(scope, 'read_ta_m5_original_cuda_prefix_origin', lambda launch: {
        'original_recipe': original, 'original_result': {}, 'initial_measurement': {},
        'source_reference_recipe': original})
    monkeypatch.setattr(scope, '_read', lambda ref, label:
        deepcopy(next(value for key, value in documents.items() if refs[key] == ref)))
    monkeypatch.setattr(scope, '_ta_m5_original_cuda_prefix_sources', lambda *args: {'qualified': True})
    monkeypatch.setattr(scope.campaign, 'require_binding', checked)
    monkeypatch.setattr(scope.campaign, 'file_sha256', lambda path: SHA)
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0))
    return SimpleNamespace(base=base, documents=documents, seen=seen)


def recipe(metadata):
    return scope.build_ta_m5_paired_batch_order_recipe(source_commit='d' * 40)


def test_fixed_intact_batches_cover_same4096_without_tail(metadata):
    value = recipe(metadata)
    control, interleaved = (value['arm_batch_indices'][a] for a in scope.TA_M5_PAIRED_BATCH_ORDER_ARMS)
    assert control == [(i * 1847) // 255 for i in range(256)]
    assert interleaved == [control[r * 16 + c] for c in range(16) for r in range(16)]
    assert set(control) == set(interleaved) and len(set(control)) == 256
    assert control != interleaved and control[0] == 0 and control[-1] == 1847
    rows = [16 * batch + offset for batch in control for offset in range(16)]
    assert len(set(rows)) == 4096 and max(rows) == 29567
    assert all(type(index) is int for index in control + interleaved)
    assert value['numerical_recipe'] == metadata.base['numerical_recipe']
    assert value['max_private_updates'] == value['max_backwards'] == value['max_training_forwards'] == 512
    assert value['max_forwards'] == 1048 and value['max_evaluation_forwards'] == 536
    assert value['max_checkpoint_loads'] == 1 and value['max_full_state_restores'] == 2
    assert value['initial_evaluation_forwards'] == 0 and value['max_candidate_evidence_writes'] == 2
    assert value['max_model_seconds'] == 2400 and value['max_invocation_seconds'] == 3000
    assert value['canonical_resume_authority'] is value['per_update_pareto_requirement'] is False
    for ref in scope.TA_M5_PAIRED_BATCH_ORDER_REFS.values():
        assert any(ref == actual and options['verify_file'] is True for actual, options in metadata.seen)


@pytest.mark.parametrize('key,value', [
    ('max_forwards', 1049), ('max_private_updates', 256), ('max_full_state_restores', 3),
    ('initial_evaluation_forwards', 1), ('max_model_seconds', 2401), ('max_invocation_seconds', 3001),
    ('fit_evaluation_query_domain', 'native_S_endpoints'), ('per_update_pareto_requirement', True),
    ('canonical_resume_authority', True), ('evaluate_both_panels_unconditionally', False),
])
def test_resealed_scope_expansion_rejected(metadata, key, value):
    launch = recipe(metadata)
    launch[key] = value
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='PAIRED_ORDER_RECIPE_SCOPE'):
        scope.require_ta_m5_paired_batch_order_recipe(seal(launch, 'recipe_sha256'),
            source_repo=scope.CANONICAL_REPO, source_commit='d' * 40)


def test_resealed_batch_mutation_and_numeric_type_rejected(metadata):
    for bad in (1848, 0.0):
        launch = recipe(metadata)
        launch['arm_batch_indices']['CONTROL'][0] = bad
        with pytest.raises(scope.campaign.RandomAccessCampaignError, match='PAIRED_ORDER_RECIPE_SCOPE'):
            scope.require_ta_m5_paired_batch_order_recipe(seal(launch, 'recipe_sha256'),
                source_repo=scope.CANONICAL_REPO, source_commit='d' * 40)


@pytest.mark.parametrize('document,key,value,reason', [
    ('objective_audit_result', 'later_arrays_consumed', True, 'TRAIN_BASELINE'),
    ('objective_audit_review', 'rows', 4096, 'TRAIN_BASELINE'),
    ('initial_later_measurement', 'evaluated_model_state_sha256', '2' * 64, 'LATER_BASELINE_IDENTITY'),
    ('initial_later_measurement', 'fixed_subset_positions', list(range(192)), 'LATER_BASELINE_IDENTITY'),
    ('centered_result', 'evaluation_state_unchanged', False, 'LATER_BASELINE_PROOF'),
])
def test_wrong_baseline_or_population_fails_closed(metadata, document, key, value, reason):
    metadata.documents[document][key] = value
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='PAIRED_ORDER_' + reason):
        recipe(metadata)


def test_no_metadata_only_recipe_bypass_or_forged_bridge(metadata):
    launch = recipe(metadata)
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='PAIRED_ORDER_PHYSICAL_SOURCE'):
        scope.require_ta_m5_paired_batch_order_recipe(launch, source_repo=scope.CANONICAL_REPO,
                                                    source_commit='d' * 40, verify_files=False)
    launch['source_bridge'] = {'qualified': False}
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='PAIRED_ORDER_SOURCE_BRIDGE'):
        scope.require_ta_m5_paired_batch_order_recipe(seal(launch, 'recipe_sha256'),
            source_repo=scope.CANONICAL_REPO, source_commit='d' * 40)


def arm_progress(control=256, interleaved=256):
    return {arm: dict(optimizer_updates=updates, parent_cursor=16 * updates,
                     parent_order_sha256=digest * 64,
                     batch_order_sha256=scope.campaign.canonical_sha256(
                         scope.TA_M5_PAIRED_BATCH_ORDER_BATCH_INDICES[arm]))
            for arm, updates, digest in [('CONTROL', control, '2'), ('INTERLEAVED', interleaved, '3')]}


def make_cursor(outcome='COMPLETE', progress=None, updates=512):
    return scope.build_ta_m5_paired_batch_order_cursor(
        recipe_ref={'path': str(scope.TA_M5_PAIRED_BATCH_ORDER_RECIPE_PATH), 'sha256': SHA},
        outcome=outcome, optimizer_updates=updates,
        arm_progress=arm_progress() if progress is None else progress,
        result={'path': str(scope.TA_M5_PAIRED_BATCH_ORDER_ROOT / 'RESULT.json'), 'sha256': SHA})


def test_paired_cursor_preserves_two_orders_and_actual_partial_work(metadata):
    complete = make_cursor()
    assert complete['optimizer_updates'] == 512
    assert complete['parent_cursor'] is complete['parent_order_sha256'] is complete['full_state'] is None
    failed = make_cursor('FAILED', arm_progress(256, 7), 263)
    assert failed['arm_progress']['INTERLEAVED']['parent_cursor'] == 112
    assert failed['optimizer_updates'] == 263
    pending = arm_progress(0, 0)
    for arm in pending.values():
        arm['parent_order_sha256'] = None
    assert make_cursor('FAILED', pending, 0)['optimizer_updates'] == 0


@pytest.mark.parametrize('key,value', [('optimizer_updates', 256), ('parent_cursor', 8192),
    ('parent_order_sha256', SHA), ('full_state', {'path': '/fake.pt', 'sha256': SHA})])
def test_paired_cursor_cannot_masquerade_as_one_canonical_pass(metadata, key, value):
    cursor = make_cursor()
    cursor[key] = value
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='PAIRED_ORDER_CURSOR'):
        scope.require_ta_m5_paired_batch_order_cursor(seal(cursor, 'cursor_sha256'),
                                                     expected_recipe=cursor['recipe'], verify_files=False)


def test_arm_progress_and_fixed_order_cannot_be_relabelled(metadata):
    for modify in ('order', 'rows', 'interleaved_before_control'):
        progress = arm_progress()
        if modify == 'order':
            progress['INTERLEAVED']['batch_order_sha256'] = progress['CONTROL']['batch_order_sha256']
        elif modify == 'rows':
            progress['CONTROL']['parent_cursor'] = 8192
        else:
            progress['CONTROL'].update(optimizer_updates=255, parent_cursor=4080)
        with pytest.raises(scope.campaign.RandomAccessCampaignError, match='PAIRED_ORDER_CURSOR'):
            make_cursor(progress=progress)


def completion(cursor):
    result = {key: deepcopy(cursor[key]) for key in ('recipe', 'outcome', 'optimizer_updates',
        'parent_cursor', 'parent_order_sha256', 'arm_progress', 'full_state', 'test_data_used',
        'canonical_resume_authority')}
    result.update(schema_version=scope.TA_M5_PAIRED_BATCH_ORDER_RESULT_SCHEMA,
        physical_optimizer_updates=512, final_state=None, checkpoint_writes=0,
        decision='PASS_TA_M5_PAIRED_BATCH_ORDER_TECHNICAL', source_unchanged=True, skipped_batches=0,
        counts=deepcopy(scope.TA_M5_PAIRED_BATCH_ORDER_COMPLETE_COUNTS), score_gate_pass=False,
        candidate_evidence={arm: {'path': str(scope.TA_M5_PAIRED_BATCH_ORDER_ROOT / (arm + '_EVIDENCE.pt')),
                                 'sha256': SHA} for arm in scope.TA_M5_PAIRED_BATCH_ORDER_ARMS})
    result.update({key: True for key in ('same_selected_rows', 'same_initial_state',
        'same_training_queries', 'rng_stream_parity', 'evaluation_state_unchanged', 'all_panels_complete')})
    return result


def test_negative_scores_are_complete_but_missing_physical_work_is_not(metadata, monkeypatch):
    cursor = make_cursor()
    result = completion(cursor)
    monkeypatch.setattr(scope, '_read', lambda *args: result)
    assert scope.require_ta_m5_paired_batch_order_cursor(cursor, expected_recipe=cursor['recipe']) == cursor
    result['counts']['evaluation_forwards'] = 535
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='PAIRED_ORDER_TECHNICAL_COMPLETION'):
        scope.require_ta_m5_paired_batch_order_cursor(cursor, expected_recipe=cursor['recipe'])


def test_gate_requires_exact_new_authority(metadata):
    launch = recipe(metadata)
    gate = dict(decision='ADMITTED', status='ADMITTED', ta_m5_paired_batch_order_authorized=True,
        fold_id=1, selected_fit_rows=4096, arms=list(scope.TA_M5_PAIRED_BATCH_ORDER_ARMS),
        model_constructions_authorized=1, optimizer_constructions_authorized=1,
        model_forwards_authorized=1048, backward_calls_authorized=512, optimizer_steps_authorized=512,
        training_enabled=True, native_training_enabled=True, genuine_updates_authorized=True,
        genuine_backward_authorized=True, genuine_predictor_fit_authorized=False,
        canonical_resume_authority=False, candidate_selection=False, test_data_used=False,
        evaluate_both_panels_unconditionally=True, per_update_pareto_requirement=False)
    scope.require_ta_m5_paired_batch_order_gate(gate, launch)
    for key, bad in [('training_enabled', False), ('optimizer_steps_authorized', 513),
                     ('ta_m5_paired_batch_order_authorized', False)]:
        changed = {**gate, key: bad}
        with pytest.raises(scope.campaign.RandomAccessCampaignError, match='PAIRED_ORDER_POLICY_GATE'):
            scope.require_ta_m5_paired_batch_order_gate(changed, launch)
    launch['run_scope'] = scope.TA_M5_CENTERED_HEAD_COORDINATES_SCOPE
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='PAIRED_ORDER_POLICY_GATE'):
        scope.require_ta_m5_paired_batch_order_gate(gate, launch)



def test_research_routing_paired_cursor_progress_and_legacy(metadata, monkeypatch):
    from gx1.contracts import entry_native_direction_research_campaign_v1 as research
    args = dict(recipe_ref={'path': str(scope.TA_M5_PAIRED_BATCH_ORDER_RECIPE_PATH), 'sha256': SHA},
        outcome='COMPLETE', optimizer_updates=512, parent_cursor=None, parent_order_sha256=None,
        full_state=None, result={'path': str(scope.TA_M5_PAIRED_BATCH_ORDER_ROOT / 'RESULT.json'), 'sha256': SHA},
        batch_size=16, arm_progress=arm_progress())
    cursor = research.build_research_cursor(**args)
    report = completion(cursor)
    monkeypatch.setattr(scope, '_read', lambda *unused: report)
    monkeypatch.setattr(scope.campaign, 'read_bound_json', lambda *unused: cursor)
    pointer = {'path': '/fixture/cursors/RESUME_CURSOR.json', 'sha256': SHA}
    invocation = dict(kind=research.KIND, invocation_sha256=SHA,
        direction_research_recipe=cursor['recipe'], batch_size=16,
        optimizer_step_budget=512, checkpoint={'pointer_path': pointer['path']})
    progress = research.build_research_progress(window={'recipe': cursor['recipe']},
        plan={'plan_sha256': SHA}, invocation=invocation, checkpoint_pointer=pointer,
        optimizer_updates=512, parent_cursor=None, parent_order_sha256=None, outcome='COMPLETE')
    assert progress['completed_units'] == progress['total_units'] == 512
    assert research.require_research_progress(progress, plan_sha256=SHA, invocation=invocation,
        expected_selection_receipt_sha256=None, verify_file=True) == progress
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='PAIRED_NO_CANONICAL_CURSOR'):
        research.build_research_cursor(**{**args, 'parent_cursor': 8192})
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='PAIRED_NO_CANONICAL_CURSOR'):
        research.require_research_progress(seal({**progress, 'parent_cursor': 8192}, 'progress_sha256'),
            plan_sha256=SHA, invocation=invocation, expected_selection_receipt_sha256=None, verify_file=False)
    old_args = {**args, 'recipe_ref': {'path': str(scope.TA_M5_CENTERED_HEAD_COORDINATES_RECIPE_PATH), 'sha256': SHA},
        'optimizer_updates': 256, 'parent_cursor': 4096, 'parent_order_sha256': scope.TA_M5_EQUAL_LOSS_ORDER_SHA256,
        'result': {'path': str(scope.TA_M5_CENTERED_HEAD_COORDINATES_ROOT / 'RESULT.json'), 'sha256': SHA},
        'arm_progress': None}
    old_cursor = research.build_research_cursor(**old_args)
    assert old_cursor['schema_version'] == scope.TA_M5_CENTERED_HEAD_COORDINATES_CURSOR_SCHEMA
    assert old_cursor['optimizer_updates'] == 256 and old_cursor['parent_cursor'] == 4096
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='UNEXPECTED_ARM_PROGRESS'):
        research.build_research_cursor(**{**old_args, 'arm_progress': arm_progress()})


def test_research_routing_recipe_requires_exact_path(metadata, monkeypatch):
    from gx1.contracts import entry_native_direction_research_campaign_v1 as research
    value = {'run_scope': scope.TA_M5_PAIRED_BATCH_ORDER_SCOPE}
    calls = []
    monkeypatch.setattr(scope.campaign, 'read_bound_json', lambda *unused: value)
    monkeypatch.setattr(scope, 'require_ta_m5_paired_batch_order_recipe',
        lambda received, **kwargs: calls.append((received, kwargs)) or received)
    ref = {'path': str(scope.TA_M5_PAIRED_BATCH_ORDER_RECIPE_PATH), 'sha256': SHA}
    kwargs = dict(source_repo=scope.CANONICAL_REPO, source_commit='d' * 40)
    assert research.require_research_launch_recipe(ref, **kwargs) == value
    assert len(calls) == 1
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='PAIRED_ORDER_RECIPE_PATH'):
        research.require_research_launch_recipe({**ref, 'path': str(scope.TA_M5_CENTERED_HEAD_COORDINATES_RECIPE_PATH)}, **kwargs)
    value['run_scope'] = scope.TA_M5_CENTERED_HEAD_COORDINATES_SCOPE
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='PAIRED_ORDER_RECIPE_PATH'):
        research.require_research_launch_recipe(ref, **kwargs)
    assert len(calls) == 1


def test_research_routing_exact_paired_time_and_step_bounds_preserve_legacy(metadata, monkeypatch):
    from gx1.contracts import entry_native_direction_research_campaign_v1 as research
    for recipe_path, run_scope, seconds, wall, steps in (
        (scope.TA_M5_PAIRED_BATCH_ORDER_RECIPE_PATH, scope.TA_M5_PAIRED_BATCH_ORDER_SCOPE, 3000, 3300, 512),
        (scope.TA_M5_CENTERED_HEAD_COORDINATES_RECIPE_PATH, scope.TA_M5_CENTERED_HEAD_COORDINATES_SCOPE, 12000, 13800, 256)):
        ref = {'path': str(recipe_path), 'sha256': SHA}
        launch = dict(run_scope=run_scope, numerical_recipe={'batch_size': 16}, recipe_sha256=SHA)
        window = seal(dict(schema_version=research.WINDOW_SCHEMA, recipe=ref, invocation_number=1,
            max_invocation_seconds=seconds, progress_path='/fixture/progress/window.json',
            campaign_cursor_path='/fixture/cursors/RESUME_CURSOR.json', output_directory='/fixture/output',
            test_data_used=False), 'policy_sha256')
        assert research.require_research_window_policy(window) == window
        with pytest.raises(scope.campaign.RandomAccessCampaignError, match='WINDOW_SCOPE'):
            research.require_research_window_policy(seal({**window, 'max_invocation_seconds':
                12000 if seconds == 3000 else 3000}, 'policy_sha256'))
        monkeypatch.setattr(research, 'require_research_launch_recipe', lambda *args, **kwargs: launch)
        monkeypatch.setattr(scope.campaign, 'read_bound_json', lambda *unused: window)
        window_ref = {'path': '/fixture/window.json', 'sha256': SHA}
        execution = dict(prelaunch_manifest=ref, prelaunch_manifest_sha256=SHA,
            train_session_manifest=window_ref, train_session_manifest_sha256=window['policy_sha256'])
        invocation = dict(requires_fresh_windows_boot=True, kind=research.KIND, batch_size=16,
            epoch_index=None, invocation_number=1, window_index=0, window_count=1,
            optimizer_step_budget=steps, expected_success_outcome='COMPLETE', maximum_wall_seconds=wall,
            checkpoint=dict(pointer_path=window['campaign_cursor_path'], before_mode='GENESIS',
                predecessor_invocation_number=None, write_mode='ATOMIC_UPDATE'),
            progress_path=window['progress_path'], launcher_argv=['guard', '--',
                str(scope.CANONICAL_REPO / '.venv/bin/python'), '-m', research.MODULE,
                '--window-policy', window_ref['path'], '--window-policy-file-sha256', SHA,
                '--progress-path', window['progress_path']])
        kwargs = dict(source_repo=scope.CANONICAL_REPO, source_commit='d' * 40)
        research.require_research_invocation_execution(invocation, execution, **kwargs)
        for key, wrong in [('maximum_wall_seconds', 13800 if wall == 3300 else 3300),
                           ('optimizer_step_budget', 256 if steps == 512 else 512)]:
            with pytest.raises(scope.campaign.RandomAccessCampaignError, match='INVOCATION_BINDING'):
                research.require_research_invocation_execution({**invocation, key: wrong}, execution, **kwargs)


def test_research_routing_dispatch_selects_paired_and_preserves_centered(monkeypatch):
    import sys
    from pathlib import Path
    from gx1.contracts import entry_native_direction_research_campaign_v1 as research
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.scripts import run_entry_native_direction_research_window_v1 as runner
    calls = []
    monkeypatch.setattr(cap, 'require_guarded_cuda_trainer_execution', lambda: {})
    monkeypatch.setitem(sys.modules, 'gx1.audit.run_market_state_ta_m5_paired_batch_order_v1',
        SimpleNamespace(run_paired_batch_order=lambda **kwargs: calls.append(('paired', kwargs)) or 'paired'))
    monkeypatch.setitem(sys.modules, 'gx1.audit.run_market_state_ta_m5_centered_head_coordinates_v1',
        SimpleNamespace(run_centered_head_coordinates=lambda **kwargs: calls.append(('centered', kwargs)) or 'centered'))
    args = dict(policy_path=Path('/fixture/window.json'), policy_file_sha256=SHA,
                progress_path=Path('/fixture/progress/window.json'))
    for run_scope, expected in [(scope.TA_M5_PAIRED_BATCH_ORDER_SCOPE, 'paired'),
                                (scope.TA_M5_CENTERED_HEAD_COORDINATES_SCOPE, 'centered')]:
        monkeypatch.setattr(research, 'require_native_direction_research_context',
            lambda *unused: ({}, {}, {}, {}, {'run_scope': run_scope}))
        assert runner.run_native_direction_research_window(**args) == expected
    assert calls == [('paired', args), ('centered', args)]
