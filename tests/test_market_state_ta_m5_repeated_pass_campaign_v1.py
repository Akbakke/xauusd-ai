"""Repeated CONTROL passes: metadata admission and routing, without model loads."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

from gx1.contracts import market_state_native_F_private_campaign_v1 as scope

SHA = '1' * 64
CONTROL_MODEL = '2' * 64
CONTROL_ORDER = '3' * 64


def seal(value, key):
    value = deepcopy(value)
    value.pop(key, None)
    return {**value, key: scope.campaign.canonical_sha256(value)}


@pytest.fixture
def metadata(monkeypatch):
    base_refs = {**deepcopy(scope.TA_M5_ORIGINAL_CUDA_PREFIX_REFS),
                 **deepcopy(scope.TA_M5_PAIRED_BATCH_ORDER_REFS)}
    refs = {**base_refs, **deepcopy(scope.TA_M5_REPEATED_PASS_REFS)}
    numerical = {'max_updates': 1849, 'batch_size': 16, 'grad_accum_steps': 1}
    original = dict(numerical_recipe=numerical, source_owners={'gx1/contracts/fixture.py': {
        'path': str(scope.CANONICAL_REPO / 'gx1/contracts/fixture.py'), 'sha256': SHA}})
    base = dict(schema_version=scope.TA_M5_PAIRED_BATCH_ORDER_RECIPE_SCHEMA,
        run_scope=scope.TA_M5_PAIRED_BATCH_ORDER_SCOPE, admission_stage=scope.TA_M5_PAIRED_BATCH_ORDER_STAGE,
        source_repo=str(scope.CANONICAL_REPO), input_refs=deepcopy(base_refs),
        numerical_recipe=deepcopy(numerical), expected_view_sha256=SHA,
        expected_model_contract_sha256=SHA, expected_loss_contract_sha256=SHA,
        expected_initial_model_state_sha256=scope.TA_M5_EQUAL_LOSS_MODEL_SHA256,
        parent_order_rows=29574, parent_order_sha256=scope.TA_M5_EQUAL_LOSS_ORDER_SHA256,
        pass_index=0, max_model_constructions=1, max_optimizer_constructions=1,
        max_checkpoint_writes=0, max_candidate_evidence_bytes=scope.MAX_CHECKPOINT_BYTES,
        full_epoch_training_allowed=False, test_data_used=False, canonical_resume_authority=False,
        arm_batch_indices=deepcopy(scope.TA_M5_PAIRED_BATCH_ORDER_BATCH_INDICES),
        arm_batch_order_sha256={}, arms=list(scope.TA_M5_PAIRED_BATCH_ORDER_ARMS),
        updates_per_arm=256, both_arms_same_original_INITIAL_required=True,
        rng_stream_parity_required=True, same_selected_rows_required=True,
        same_training_queries_required=True, evaluation_state_unchanged_required=True,
        evaluate_both_panels_unconditionally=True, per_update_pareto_requirement=False,
        selected_fit_rows=4096, score_eval_ranges=deepcopy(scope.TA_M5_PAIRED_BATCH_ORDER_SCORE_RANGES),
        baseline_parity={'raw_atol': 2e-5, 'probability_atol': 1e-6, 'score_atol': 1e-6})
    paired = {**deepcopy(base), 'trace_worker': refs['paired_worker']}
    report = dict(schema_version=scope.TA_M5_PAIRED_BATCH_ORDER_RESULT_SCHEMA,
        recipe=refs['paired_recipe'], outcome='COMPLETE',
        decision='PASS_TA_M5_PAIRED_BATCH_ORDER_TECHNICAL',
        source_commit='d' * 40, physical_optimizer_updates=512, optimizer_updates=512,
        checkpoint_writes=0, skipped_batches=0, source_unchanged=True, test_data_used=False,
        canonical_resume_authority=False, candidate_evidence={'CONTROL': refs['control_evidence']},
        counts=deepcopy(scope.TA_M5_PAIRED_BATCH_ORDER_COMPLETE_COUNTS),
        arm_progress={'CONTROL': dict(optimizer_updates=256, parent_cursor=4096,
            parent_order_sha256=CONTROL_ORDER, batch_order_sha256=scope.campaign.canonical_sha256(
                scope.TA_M5_PAIRED_BATCH_ORDER_BATCH_INDICES['CONTROL']))},
        measurement_refs={}, captures={})
    report.update({key: True for key in ('same_selected_rows', 'same_initial_state',
        'same_training_queries', 'rng_stream_parity', 'evaluation_state_unchanged', 'all_panels_complete')})
    documents = {
        'paired_recipe': paired, 'paired_result': report,
        'paired_terminal': dict(status='COMPLETE', exit_code=0, recipe=refs['paired_recipe'],
            result=refs['paired_result'], source_unchanged=True,
            no_relaunch_under_this_plan=True, test_data_used=False),
        'paired_review': dict(status='PASS_INDEPENDENT_CACHED_SCORES_CONTROL_LOCAL_GAIN_ORDER_CHANGE_FAIL',
            result=refs['paired_result'], matched_immutable_axes_queries_truth=True, test_data_used=False),
    }
    for label in ('INITIAL_TRAIN', 'INITIAL_LATER', 'CONTROL_TRAIN', 'CONTROL_LATER'):
        key = label.lower()
        train, control = label.endswith('TRAIN'), label.startswith('CONTROL')
        measure = dict(rows=4096 if train else 192, batches=256 if train else 12,
            feature_view_sha256=SHA, model_contract_sha256=SHA,
            loss_contract_sha256=SHA if train else '4' * 64)
        if train:
            measure.update(pass_index=0, seed=20260911,
                query_domain={'state': 'sampled_pass0', 'future': 'native_endpoints'})
        documents[key + '_measurement'] = dict(capture=refs[key + '_capture'],
            optimizer_updates=256 if control else 0, test_data_used=False,
            evaluated_model_state_sha256=CONTROL_MODEL if control else scope.TA_M5_EQUAL_LOSS_MODEL_SHA256,
            measurement=measure)
        report['measurement_refs'][label] = refs[key + '_measurement']
        report['captures'][label] = refs[key + '_capture']
    seen, bridge_calls = [], []
    def binding(ref, **kwargs):
        seen.append((dict(ref), kwargs))
        return dict(ref)
    def bridge(*args, **kwargs):
        bridge_calls.append((args, kwargs))
        return {'qualified_paired_bridge': True}
    monkeypatch.setattr(scope, '_ta_m5_paired_batch_order_expected',
        lambda: (deepcopy(base), deepcopy(original), deepcopy(original)))
    def prior_origin(launch):
        assert launch['input_refs'] == base_refs
        return dict(original_recipe=deepcopy(original), original_result={},
                    source_reference_recipe=deepcopy(original))
    monkeypatch.setattr(scope, 'read_ta_m5_paired_batch_order_origin', prior_origin)
    monkeypatch.setattr(scope, '_read', lambda ref, label:
        deepcopy(next(value for key, value in documents.items() if refs[key] == ref)))
    monkeypatch.setattr(scope, '_ta_m5_original_cuda_prefix_sources', bridge)
    monkeypatch.setattr(scope.campaign, 'require_binding', binding)
    monkeypatch.setattr(scope.campaign, 'file_sha256', lambda path: SHA)
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0))
    return SimpleNamespace(base=base, refs=refs, documents=documents, seen=seen, bridge_calls=bridge_calls)


def recipe(metadata):
    return scope.build_ta_m5_repeated_pass_recipe(source_commit='d' * 40)


def test_exact_continuation_reuses_control_not_original_initial(metadata):
    launch = recipe(metadata)
    assert launch['passes'] == [2, 3] and launch['updates_per_pass'] == 256
    assert launch['batch_indices'] == scope.TA_M5_PAIRED_BATCH_ORDER_BATCH_INDICES['CONTROL']
    assert launch['starting_optimizer_updates'] == 256 and launch['final_optimizer_updates'] == 768
    assert launch['expected_initial_model_state_sha256'] == CONTROL_MODEL
    assert launch['parent_order_sha256'] == CONTROL_ORDER and launch['parent_order_rows'] == 29574
    assert launch['numerical_recipe'] == metadata.base['numerical_recipe']
    assert launch['pass_index'] == launch['unchanged_query_pass_index'] == 0
    assert launch['max_forwards'] == 1050 and launch['max_evaluation_forwards'] == 538
    assert launch['max_training_forwards'] == launch['max_private_updates'] == launch['max_backwards'] == 512
    assert launch['max_checkpoint_loads'] == launch['max_full_state_restores'] == 1
    assert launch['max_candidate_evidence_writes'] == 1 and launch['max_checkpoint_writes'] == 0
    assert launch['baseline_parity_forwards'] == 2 and launch['initial_evaluation_forwards'] == 0
    assert launch['max_model_seconds'] == 2400 and launch['max_invocation_seconds'] == 12000
    assert launch['canonical_resume_authority'] is launch['automatic_continuation'] is False
    assert launch['unselected_suffix_consumed'] is False
    assert not {'arms', 'arm_batch_indices', 'arm_batch_order_sha256', 'updates_per_arm',
                'both_arms_same_original_INITIAL_required', 'rng_stream_parity_required'} & set(launch)
    assert all(kwargs == {'paired_order_guard': True} for _, kwargs in metadata.bridge_calls)
    assert all(args[3] == metadata.refs['original_trace_helper'] for args, _ in metadata.bridge_calls)
    for ref in scope.TA_M5_REPEATED_PASS_REFS.values():
        assert any(ref == actual and options['verify_file'] is True for actual, options in metadata.seen)


@pytest.mark.parametrize('key,value', [
    ('max_forwards', 1048), ('max_private_updates', 768), ('max_full_state_restores', 2),
    ('starting_optimizer_updates', 0), ('final_optimizer_updates', 512), ('passes', [1, 2]),
    ('pass_index', 1), ('batch_indices', list(range(256))), ('initial_evaluation_forwards', 12),
    ('baseline_parity_forwards', 0), ('canonical_resume_authority', True),
    ('automatic_continuation', True), ('max_invocation_seconds', 3000),
    ('evaluate_both_panels_unconditionally', False),
])
def test_resealed_recipe_change_is_rejected(metadata, key, value):
    launch = {**recipe(metadata), key: value}
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='REPEATED_PASS_RECIPE_SCOPE'):
        scope.require_ta_m5_repeated_pass_recipe(seal(launch, 'recipe_sha256'),
            source_repo=scope.CANONICAL_REPO, source_commit='d' * 40)


@pytest.mark.parametrize('document,key,value,reason', [
    ('paired_recipe', 'numerical_recipe', {'max_updates': 512}, 'PAIRED_RECIPE'),
    ('paired_recipe', 'pass_index', 1, 'PAIRED_RECIPE'),
    ('paired_result', 'source_unchanged', False, 'PAIRED_COMPLETION'),
    ('paired_terminal', 'status', 'FAILED', 'PAIRED_REVIEW'),
    ('paired_review', 'matched_immutable_axes_queries_truth', False, 'PAIRED_REVIEW'),
    ('control_later_measurement', 'evaluated_model_state_sha256', '5' * 64, 'CONTROL_MODEL'),
    ('initial_train_measurement', 'evaluated_model_state_sha256', CONTROL_MODEL, 'INITIAL_MODEL'),
])
def test_prior_capsule_and_baseline_proof_cannot_be_relabelled(metadata, document, key, value, reason):
    metadata.documents[document][key] = value
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='REPEATED_PASS_' + reason):
        recipe(metadata)


def test_fixed_ref_and_train_query_boundary(metadata):
    launch = recipe(metadata)
    launch['input_refs']['control_evidence']['sha256'] = SHA
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='REPEATED_PASS_FIXED_REFS'):
        scope.read_ta_m5_repeated_pass_origin(launch)
    metadata.documents['control_train_measurement']['measurement']['query_domain']['state'] = 'native_endpoints'
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='REPEATED_PASS_TRAIN_QUERY'):
        recipe(metadata)


def test_live_source_bridge_and_worker_cannot_be_bypassed(metadata):
    launch = recipe(metadata)
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='REPEATED_PASS_PHYSICAL_SOURCE'):
        scope.require_ta_m5_repeated_pass_recipe(launch, source_repo=scope.CANONICAL_REPO,
            source_commit='d' * 40, verify_files=False)
    for key, bad, reason in (
        ('trace_worker', {'path': str(scope.CANONICAL_REPO / scope.TA_M5_PAIRED_BATCH_ORDER_WORKER),
                          'sha256': SHA}, 'WORKER_PATH'),
        ('source_bridge', {}, 'SOURCE_BRIDGE')):
        with pytest.raises(scope.campaign.RandomAccessCampaignError, match='REPEATED_PASS_' + reason):
            scope.require_ta_m5_repeated_pass_recipe(seal({**launch, key: bad}, 'recipe_sha256'),
                source_repo=scope.CANONICAL_REPO, source_commit='d' * 40)


def make_cursor(**changes):
    args = dict(recipe_ref={'path': str(scope.TA_M5_REPEATED_PASS_RECIPE_PATH), 'sha256': SHA},
        outcome='COMPLETE', optimizer_updates=512, current_pass=3, batch_index=256,
        cumulative_optimizer_updates=768,
        result={'path': str(scope.TA_M5_REPEATED_PASS_ROOT / 'RESULT.json'), 'sha256': SHA})
    return scope.build_ta_m5_repeated_pass_cursor(**{**args, **changes})


def test_cursor_tracks_new_and_cumulative_steps_without_canonical_resume():
    complete = make_cursor()
    assert complete['optimizer_updates'] == 512 and complete['cumulative_optimizer_updates'] == 768
    assert complete['parent_cursor'] is complete['parent_order_sha256'] is complete['full_state'] is None
    for current_pass, batch_index, updates in ((2, 0, 0), (2, 7, 7), (2, 256, 256), (3, 0, 256), (3, 9, 265)):
        failed = make_cursor(outcome='FAILED', current_pass=current_pass, batch_index=batch_index,
            optimizer_updates=updates, cumulative_optimizer_updates=256 + updates)
        assert failed['optimizer_updates'] == updates


@pytest.mark.parametrize('change,reason', [
    ({'optimizer_updates': 768}, 'SCOPE'), ({'current_pass': 4}, 'PROGRESS'),
    ({'batch_index': 257}, 'PROGRESS'), ({'cumulative_optimizer_updates': 512}, 'PROGRESS'),
    ({'current_pass': 2, 'batch_index': 256, 'optimizer_updates': 256,
      'cumulative_optimizer_updates': 512}, 'COMPLETE'),
    ({'current_pass': True}, 'PROGRESS'), ({'batch_size': 2}, 'SCOPE'),
    ({'batch_size': 16.0}, 'SCOPE'),
    ({'recipe_ref': {'path': str(scope.TA_M5_PAIRED_BATCH_ORDER_RECIPE_PATH), 'sha256': SHA}}, 'SCOPE'),
])
def test_invalid_cursor_progress_is_rejected(change, reason):
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='REPEATED_PASS_CURSOR_' + reason):
        make_cursor(**change)


def completion(cursor):
    value = {key: deepcopy(cursor[key]) for key in ('recipe', 'outcome', 'optimizer_updates',
        'current_pass', 'batch_index', 'cumulative_optimizer_updates', 'parent_cursor',
        'parent_order_sha256', 'full_state', 'test_data_used', 'canonical_resume_authority')}
    value.update(schema_version=scope.TA_M5_REPEATED_PASS_RESULT_SCHEMA,
        physical_optimizer_updates=512, final_state=None, checkpoint_writes=0,
        decision='PASS_TA_M5_REPEATED_PASS_TECHNICAL', source_unchanged=True, skipped_batches=0,
        counts=deepcopy(scope.TA_M5_REPEATED_PASS_COMPLETE_COUNTS), score_gate_pass=False,
        candidate_evidence={'path': str(scope.TA_M5_REPEATED_PASS_ROOT / 'E3_EVIDENCE.pt'), 'sha256': SHA})
    value.update({key: True for key in ('control_restore_exact', 'baseline_output_parity',
        'evaluation_state_unchanged', 'all_measurements_complete', 'optimizer_state_continued')})
    return value


def test_complete_requires_restore_and_all_measurements_not_score_improvement(metadata, monkeypatch):
    cursor = make_cursor()
    report = completion(cursor)
    monkeypatch.setattr(scope, '_read', lambda *unused: report)
    assert scope.require_ta_m5_repeated_pass_cursor(cursor, expected_recipe=cursor['recipe']) == cursor
    for key in ('control_restore_exact', 'baseline_output_parity', 'evaluation_state_unchanged',
                'all_measurements_complete', 'optimizer_state_continued'):
        report[key] = False
        with pytest.raises(scope.campaign.RandomAccessCampaignError, match='REPEATED_PASS_TECHNICAL_COMPLETION'):
            scope.require_ta_m5_repeated_pass_cursor(cursor, expected_recipe=cursor['recipe'])
        report[key] = True
    report['counts']['evaluation_forwards'] = 536
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='REPEATED_PASS_TECHNICAL_COMPLETION'):
        scope.require_ta_m5_repeated_pass_cursor(cursor, expected_recipe=cursor['recipe'])


def test_scope_requires_fresh_exact_training_authority(metadata):
    launch = recipe(metadata)
    gate = dict(decision='ADMITTED', status='ADMITTED', ta_m5_repeated_pass_authorized=True,
        fold_id=1, selected_fit_rows=4096, passes=[2, 3], starting_optimizer_updates=256,
        final_optimizer_updates=768, model_constructions_authorized=1,
        optimizer_constructions_authorized=1, model_forwards_authorized=1050,
        backward_calls_authorized=512, optimizer_steps_authorized=512, training_enabled=True,
        native_training_enabled=True, genuine_updates_authorized=True, genuine_backward_authorized=True,
        genuine_predictor_fit_authorized=False, canonical_resume_authority=False, candidate_selection=False,
        test_data_used=False, evaluate_both_panels_unconditionally=True, per_update_pareto_requirement=False)
    scope.require_ta_m5_repeated_pass_gate(gate, launch)
    for key, bad in [('training_enabled', False), ('optimizer_steps_authorized', 768),
                     ('ta_m5_repeated_pass_authorized', False), ('passes', [1, 2])]:
        with pytest.raises(scope.campaign.RandomAccessCampaignError, match='REPEATED_PASS_POLICY_GATE'):
            scope.require_ta_m5_repeated_pass_gate({**gate, key: bad}, launch)
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='REPEATED_PASS_POLICY_GATE'):
        scope.require_ta_m5_repeated_pass_gate(gate, {**launch, 'run_scope': scope.TA_M5_PAIRED_BATCH_ORDER_SCOPE})


def test_actual_research_cursor_and_progress_dispatch(metadata, monkeypatch):
    from gx1.contracts import entry_native_direction_research_campaign_v1 as research
    args = dict(recipe_ref={'path': str(scope.TA_M5_REPEATED_PASS_RECIPE_PATH), 'sha256': SHA},
        outcome='COMPLETE', optimizer_updates=512, parent_cursor=None, parent_order_sha256=None,
        full_state=None, result={'path': str(scope.TA_M5_REPEATED_PASS_ROOT / 'RESULT.json'), 'sha256': SHA},
        batch_size=16, repeated_progress=dict(current_pass=3, batch_index=256, cumulative_optimizer_updates=768))
    cursor = research.build_research_cursor(**args)
    report = completion(cursor)
    monkeypatch.setattr(scope, '_read', lambda *unused: report)
    monkeypatch.setattr(scope.campaign, 'read_bound_json', lambda *unused: cursor)
    pointer = {'path': '/fixture/cursors/RESUME_CURSOR.json', 'sha256': SHA}
    invocation = dict(kind=research.KIND, invocation_sha256=SHA, direction_research_recipe=cursor['recipe'],
        batch_size=16, optimizer_step_budget=512, checkpoint={'pointer_path': pointer['path']})
    progress = research.build_research_progress(window={'recipe': cursor['recipe']},
        plan={'plan_sha256': SHA}, invocation=invocation, checkpoint_pointer=pointer,
        optimizer_updates=512, parent_cursor=None, parent_order_sha256=None, outcome='COMPLETE')
    assert progress['completed_units'] == progress['total_units'] == 512
    assert research.require_research_progress(progress, plan_sha256=SHA, invocation=invocation,
        expected_selection_receipt_sha256=None, verify_file=True) == progress
    for key, bad in [('parent_cursor', 8192), ('parent_order_sha256', SHA), ('arm_progress', {})]:
        with pytest.raises(scope.campaign.RandomAccessCampaignError, match='REPEATED_NO_CANONICAL_CURSOR'):
            research.build_research_cursor(**{**args, key: bad})
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='UNEXPECTED_REPEATED_PROGRESS'):
        research.build_research_cursor(**{**args, 'recipe_ref': {
            'path': str(scope.TA_M5_PAIRED_BATCH_ORDER_RECIPE_PATH), 'sha256': SHA}})


def test_actual_recipe_route_requires_exact_new_path(metadata, monkeypatch):
    from gx1.contracts import entry_native_direction_research_campaign_v1 as research
    value, calls = {'run_scope': scope.TA_M5_REPEATED_PASS_SCOPE}, []
    monkeypatch.setattr(scope.campaign, 'read_bound_json', lambda *unused: value)
    monkeypatch.setattr(scope, 'require_ta_m5_repeated_pass_recipe',
        lambda received, **kwargs: calls.append(received) or received)
    ref = {'path': str(scope.TA_M5_REPEATED_PASS_RECIPE_PATH), 'sha256': SHA}
    kwargs = dict(source_repo=scope.CANONICAL_REPO, source_commit='d' * 40)
    assert research.require_research_launch_recipe(ref, **kwargs) == value and calls == [value]
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='REPEATED_PASS_RECIPE_PATH'):
        research.require_research_launch_recipe({**ref, 'path': str(scope.TA_M5_PAIRED_BATCH_ORDER_RECIPE_PATH)}, **kwargs)
    value['run_scope'] = scope.TA_M5_PAIRED_BATCH_ORDER_SCOPE
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='REPEATED_PASS_RECIPE_PATH'):
        research.require_research_launch_recipe(ref, **kwargs)
    assert len(calls) == 1


def test_actual_dispatch_uses_new_worker_and_keeps_paired(monkeypatch):
    from gx1.contracts import entry_native_direction_research_campaign_v1 as research
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.scripts import run_entry_native_direction_research_window_v1 as runner
    calls = []
    monkeypatch.setattr(cap, 'require_guarded_cuda_trainer_execution', lambda: {})
    monkeypatch.setitem(sys.modules, 'gx1.audit.run_market_state_ta_m5_repeated_pass_v1',
        SimpleNamespace(run_repeated_pass=lambda **kwargs: calls.append(('repeated', kwargs)) or 'repeated'))
    monkeypatch.setitem(sys.modules, 'gx1.audit.run_market_state_ta_m5_paired_batch_order_v1',
        SimpleNamespace(run_paired_batch_order=lambda **kwargs: calls.append(('paired', kwargs)) or 'paired'))
    args = dict(policy_path=Path('/fixture/window.json'), policy_file_sha256=SHA,
                progress_path=Path('/fixture/progress/window.json'))
    for run_scope, expected in ((scope.TA_M5_REPEATED_PASS_SCOPE, 'repeated'),
                                (scope.TA_M5_PAIRED_BATCH_ORDER_SCOPE, 'paired')):
        monkeypatch.setattr(research, 'require_native_direction_research_context',
            lambda *unused: ({}, {}, {}, {}, {'run_scope': run_scope}))
        assert runner.run_native_direction_research_window(**args) == expected
    assert calls == [('repeated', args), ('paired', args)]


def test_repeated_window_keeps_existing_12000_guard_and_13800_outer(metadata, monkeypatch):
    from gx1.contracts import entry_native_direction_research_campaign_v1 as research
    ref = {'path': str(scope.TA_M5_REPEATED_PASS_RECIPE_PATH), 'sha256': SHA}
    window = seal(dict(schema_version=research.WINDOW_SCHEMA, recipe=ref, invocation_number=1,
        max_invocation_seconds=12000, progress_path='/fixture/progress/window.json',
        campaign_cursor_path='/fixture/cursors/RESUME_CURSOR.json', output_directory='/fixture/output',
        test_data_used=False), 'policy_sha256')
    assert research.require_research_window_policy(window) == window
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='WINDOW_SCOPE'):
        research.require_research_window_policy(seal({**window, 'max_invocation_seconds': 3000}, 'policy_sha256'))

    launch = dict(run_scope=scope.TA_M5_REPEATED_PASS_SCOPE,
                  numerical_recipe={'batch_size': 16}, recipe_sha256=SHA)
    monkeypatch.setattr(research, 'require_research_launch_recipe', lambda *args, **kwargs: launch)
    monkeypatch.setattr(scope.campaign, 'read_bound_json', lambda *unused: window)
    window_ref = {'path': '/fixture/window.json', 'sha256': SHA}
    execution = dict(prelaunch_manifest=ref, prelaunch_manifest_sha256=SHA,
        train_session_manifest=window_ref, train_session_manifest_sha256=window['policy_sha256'])
    invocation = dict(requires_fresh_windows_boot=True, kind=research.KIND, batch_size=16,
        epoch_index=None, invocation_number=1, window_index=0, window_count=1,
        optimizer_step_budget=512, expected_success_outcome='COMPLETE', maximum_wall_seconds=13800,
        checkpoint=dict(pointer_path=window['campaign_cursor_path'], before_mode='GENESIS',
            predecessor_invocation_number=None, write_mode='ATOMIC_UPDATE'),
        progress_path=window['progress_path'], launcher_argv=['guard', '--',
            str(scope.CANONICAL_REPO / '.venv/bin/python'), '-m', research.MODULE,
            '--window-policy', window_ref['path'], '--window-policy-file-sha256', SHA,
            '--progress-path', window['progress_path']])
    kwargs = dict(source_repo=scope.CANONICAL_REPO, source_commit='d' * 40)
    research.require_research_invocation_execution(invocation, execution, **kwargs)
    for key, bad in [('maximum_wall_seconds', 3300), ('optimizer_step_budget', 768)]:
        with pytest.raises(scope.campaign.RandomAccessCampaignError, match='INVOCATION_BINDING'):
            research.require_research_invocation_execution({**invocation, key: bad}, execution, **kwargs)
