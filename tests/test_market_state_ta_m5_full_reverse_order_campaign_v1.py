"""Exact full-population reverse-order admission, with no model or data loads."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

from gx1.contracts import market_state_native_F_private_campaign_v1 as scope
from gx1.contracts import entry_native_direction_research_campaign_v1 as research

SHA = '1' * 64
REVERSED_ORDER = '3' * 64


def seal(value, key):
    value = deepcopy(value)
    value.pop(key, None)
    return {**value, key: scope.campaign.canonical_sha256(value)}


@pytest.fixture
def metadata(monkeypatch):
    base_refs = {**deepcopy(scope.TA_M5_ORIGINAL_CUDA_PREFIX_REFS),
                 **deepcopy(scope.TA_M5_PAIRED_BATCH_ORDER_REFS)}
    refs = {**base_refs, **deepcopy(scope.TA_M5_FULL_REVERSE_ORDER_REFS)}
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
        fold_id=1, pass_index=0, training_enabled=True, max_model_constructions=1,
        max_optimizer_constructions=1, max_checkpoint_writes=0,
        max_candidate_evidence_bytes=scope.MAX_CHECKPOINT_BYTES,
        full_epoch_training_allowed=False, test_data_used=False, canonical_resume_authority=False,
        arm_batch_indices=deepcopy(scope.TA_M5_PAIRED_BATCH_ORDER_BATCH_INDICES),
        arm_batch_order_sha256={}, arms=list(scope.TA_M5_PAIRED_BATCH_ORDER_ARMS),
        updates_per_arm=256, both_arms_same_original_INITIAL_required=True,
        rng_stream_parity_required=True, same_selected_rows_required=True,
        same_training_queries_required=True, evaluation_state_unchanged_required=True,
        evaluate_both_panels_unconditionally=True, per_update_pareto_requirement=False,
        selected_fit_rows=4096, score_eval_ranges=deepcopy(scope.TA_M5_PAIRED_BATCH_ORDER_SCORE_RANGES),
        baseline_parity={'raw_atol': 2e-5, 'probability_atol': 1e-6, 'score_atol': 1e-6})
    documents = {
        'final_cache_result': dict(status='COMPLETE', source_unchanged=True, model_state_unchanged=True,
            test_data_used=False, sampled_S_pass_index=0, sampled_F_reuses_native_1=True,
            checkpoint=dict(source_commit=scope.TA_M5_ORIGINAL_CUDA_PREFIX_SOURCE,
                model_sha256='23ff344b183a2256f2c8a1a8e32b811ed14f10c1cb1423becca14b36d5340d5b'),
            cache_partitions={part: dict(artifact=refs['final_' + part + '_cache'], rows=rows, partial=False)
                              for part, rows in (('fit', 29574), ('score', 17753))}),
        'final_cache_review': dict(status='COMPLETE_CONSUMED_CACHE_QUALIFIED',
            result=refs['final_cache_result'], source_unchanged=True, test_data_used=False,
            all_twenty_partition_task_TF_pairs_identical=True),
        'readout_plan': dict(inputs={key: refs[name] for key, name in (
            ('old_fit_cache', 'initial_fit_cache'), ('old_score_cache', 'initial_score_cache'),
            ('new_fit_cache', 'final_fit_cache'), ('new_score_cache', 'final_score_cache'),
            ('new_cache_result', 'final_cache_result'), ('new_cache_review', 'final_cache_review'))}),
        'readout_result': dict(status='COMPLETE', source_unchanged=True, test_data_used=False,
            plan=refs['readout_plan'], private_outputs=refs['imported_predictions']),
        'readout_review': dict(status='COMPLETE_CONSUMED_READOUT_RECOVERS_TRAIN_NOT_LATER_EDGE',
            result=refs['readout_result'], source_unchanged=True, test_data_used=False),
    }
    seen, bridges = [], []
    def binding(ref, **kwargs):
        seen.append((dict(ref), kwargs))
        return dict(ref)
    def bridge(*args, **kwargs):
        bridges.append((args, kwargs))
        return {'qualified_existing_source_bridge': True}
    def prior_origin(launch):
        assert launch['input_refs'] == base_refs
        return dict(original_recipe=deepcopy(original), original_result={},
                    source_reference_recipe=deepcopy(original))
    monkeypatch.setattr(scope, '_ta_m5_paired_batch_order_expected',
        lambda: (deepcopy(base), deepcopy(original), deepcopy(original)))
    monkeypatch.setattr(scope, 'read_ta_m5_paired_batch_order_origin', prior_origin)
    monkeypatch.setattr(scope, '_read', lambda ref, label:
        deepcopy(next(value for key, value in documents.items() if refs[key] == ref)))
    monkeypatch.setattr(scope, '_ta_m5_original_cuda_prefix_sources', bridge)
    monkeypatch.setattr(scope.campaign, 'require_binding', binding)
    monkeypatch.setattr(scope.campaign, 'file_sha256', lambda path: SHA)
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0))
    return SimpleNamespace(base=base, refs=refs, documents=documents, seen=seen, bridges=bridges)


def recipe(metadata):
    return scope.build_ta_m5_full_reverse_order_recipe(source_commit='d' * 40)


def test_all_original_batches_once_with_tail_last_and_no_paired_inheritance(metadata):
    launch = recipe(metadata)
    assert launch['batch_indices'] == list(range(1847, -1, -1)) + [1848]
    rows = [row for batch in launch['batch_indices'] for row in range(16 * batch, min(16 * batch + 16, 29574))]
    assert len(rows) == len(set(rows)) == 29574 and sorted(rows) == list(range(29574))
    assert rows[-6:] == list(range(29568, 29574))
    assert launch['arm'] == 'REVERSED' and launch['selected_fit_rows'] == 29574
    assert launch['parent_order_sha256'] == scope.TA_M5_EQUAL_LOSS_ORDER_SHA256
    assert launch['expected_initial_model_state_sha256'] == scope.TA_M5_EQUAL_LOSS_MODEL_SHA256
    assert launch['numerical_recipe'] == metadata.base['numerical_recipe']
    assert launch['pass_index'] == launch['unchanged_query_pass_index'] == 0
    assert launch['max_private_updates'] == launch['max_training_forwards'] == launch['max_backwards'] == 1849
    assert launch['max_forwards'] == 3710 and launch['max_evaluation_forwards'] == 1861
    assert launch['initial_evaluation_forwards'] == 0 and launch['later_evaluation_forwards'] == 12
    assert launch['max_checkpoint_loads'] == launch['max_full_state_restores'] == 1
    assert launch['max_candidate_evidence_writes'] == 1 and launch['max_checkpoint_writes'] == 0
    assert launch['max_model_seconds'] == 7200 and launch['max_invocation_seconds'] == 12000
    assert launch['full_epoch_training_allowed'] is launch['genuine_predictor_fit_authorized'] is True
    assert launch['canonical_resume_authority'] is launch['automatic_continuation'] is False
    assert not {'arms', 'arm_batch_indices', 'arm_batch_order_sha256', 'updates_per_arm',
        'both_arms_same_original_INITIAL_required', 'rng_stream_parity_required',
        'same_selected_rows_required', 'first_step_exact_required', 'final_full_state_exact_required',
        'previous_selected_batch_indices', 'cohort_rows'} & set(launch)
    assert all(kwargs == {'paired_order_guard': True} for _, kwargs in metadata.bridges)
    assert research.guarded_research_power_limit(launch) == 250
    for ref in scope.TA_M5_FULL_REVERSE_ORDER_REFS.values():
        assert any(ref == actual and options['verify_file'] is True for actual, options in metadata.seen)


@pytest.mark.parametrize('key,value', [
    ('max_forwards', 3709), ('max_private_updates', 1848), ('max_full_state_restores', 2),
    ('initial_evaluation_forwards', 12), ('pass_index', 1), ('tail_rows', 0),
    ('batch_indices', [1848] + list(range(1847, -1, -1))),
    ('batch_indices', list(range(1849))), ('full_epoch_training_allowed', False),
    ('genuine_predictor_fit_authorized', False), ('canonical_resume_authority', True),
    ('automatic_continuation', True), ('max_invocation_seconds', 7800),
    ('run_scope', scope.TA_M5_PAIRED_BATCH_ORDER_SCOPE),
])
def test_resealed_recipe_cannot_change_population_authority_or_counts(metadata, key, value):
    launch = seal({**recipe(metadata), key: value}, 'recipe_sha256')
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='FULL_REVERSE_RECIPE_SCOPE'):
        scope.require_ta_m5_full_reverse_order_recipe(launch,
            source_repo=scope.CANONICAL_REPO, source_commit='d' * 40)


@pytest.mark.parametrize('document,key,value,reason', [
    ('final_cache_result', 'model_state_unchanged', False, 'FINAL_BASELINE'),
    ('final_cache_result', 'sampled_S_pass_index', 1, 'FINAL_BASELINE'),
    ('final_cache_result', 'checkpoint', {'model_sha256': SHA}, 'FINAL_BASELINE'),
    ('final_cache_review', 'all_twenty_partition_task_TF_pairs_identical', False, 'FINAL_BASELINE'),
    ('readout_result', 'private_outputs', {'path': '/wrong.npz', 'sha256': SHA}, 'READOUT_PROOF'),
    ('readout_plan', 'inputs', {}, 'BASELINE_AXES_BINDING'),
])
def test_unqualified_baseline_or_query_lineage_is_rejected(metadata, document, key, value, reason):
    metadata.documents[document][key] = value
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='FULL_REVERSE_' + reason):
        recipe(metadata)


def test_physical_source_worker_and_bridge_are_required(metadata):
    launch = recipe(metadata)
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='FULL_REVERSE_PHYSICAL_SOURCE'):
        scope.require_ta_m5_full_reverse_order_recipe(launch, source_repo=scope.CANONICAL_REPO,
            source_commit='d' * 40, verify_files=False)
    for key, bad, reason in (
        ('trace_worker', {'path': str(scope.CANONICAL_REPO / scope.TA_M5_PAIRED_BATCH_ORDER_WORKER),
                          'sha256': SHA}, 'WORKER_PATH'),
        ('source_bridge', {}, 'SOURCE_BRIDGE')):
        with pytest.raises(scope.campaign.RandomAccessCampaignError, match='FULL_REVERSE_' + reason):
            scope.require_ta_m5_full_reverse_order_recipe(seal({**launch, key: bad}, 'recipe_sha256'),
                source_repo=scope.CANONICAL_REPO, source_commit='d' * 40)


def cursor_args(**changes):
    return dict(recipe_ref={'path': str(scope.TA_M5_FULL_REVERSE_ORDER_RECIPE_PATH), 'sha256': SHA},
        outcome='COMPLETE', optimizer_updates=1849, parent_cursor=29574,
        parent_order_sha256=REVERSED_ORDER,
        result={'path': str(scope.TA_M5_FULL_REVERSE_ORDER_ROOT / 'RESULT.json'), 'sha256': SHA},
        batch_size=16) | changes


def report_for(cursor):
    report = {key: deepcopy(cursor[key]) for key in ('recipe', 'outcome', 'optimizer_updates',
        'parent_cursor', 'parent_order_sha256', 'full_state', 'test_data_used', 'canonical_resume_authority')}
    report.update(schema_version=scope.TA_M5_FULL_REVERSE_ORDER_RESULT_SCHEMA,
        physical_optimizer_updates=cursor['optimizer_updates'], final_state=None, checkpoint_writes=0,
        decision='PASS_TA_M5_FULL_REVERSE_ORDER_TECHNICAL', source_unchanged=True, skipped_batches=0,
        counts=deepcopy(scope.TA_M5_FULL_REVERSE_ORDER_COMPLETE_COUNTS), score_gate_pass=False,
        candidate_evidence={'path': str(scope.TA_M5_FULL_REVERSE_ORDER_ROOT / 'REVERSED_EVIDENCE.pt'), 'sha256': SHA})
    report.update({key: True for key in ('same_initial_state', 'same_training_queries',
        'evaluation_state_unchanged', 'all_panels_complete', 'all_original_rows_once', 'tail6_preserved')})
    return report


@pytest.mark.parametrize('outcome,updates,rows,order', [
    ('FAILED', 0, 0, None), ('FAILED', 1848, 29568, REVERSED_ORDER),
    ('COMPLETE', 1849, 29574, REVERSED_ORDER),
])
def test_actual_cursor_and_durable_progress_cover_early_failure_and_tail(metadata, monkeypatch, outcome, updates, rows, order):
    args = cursor_args(outcome=outcome, optimizer_updates=updates, parent_cursor=rows, parent_order_sha256=order)
    cursor = research.build_research_cursor(**args, full_state=None)
    assert cursor['canonical_resume_authority'] is False and cursor['full_state'] is None
    monkeypatch.setattr(scope, '_read', lambda *unused: report_for(cursor))
    monkeypatch.setattr(scope.campaign, 'read_bound_json', lambda *unused: cursor)
    pointer = {'path': '/fixture/cursors/RESUME_CURSOR.json', 'sha256': SHA}
    invocation = dict(kind=research.KIND, invocation_sha256=SHA, direction_research_recipe=cursor['recipe'],
        batch_size=16, optimizer_step_budget=1849, checkpoint={'pointer_path': pointer['path']})
    progress = research.build_research_progress(window={'recipe': cursor['recipe']},
        plan={'plan_sha256': SHA}, invocation=invocation, checkpoint_pointer=pointer,
        optimizer_updates=updates, parent_cursor=rows, parent_order_sha256=order, outcome=outcome)
    assert progress['completed_units'] == updates and progress['total_units'] == 1849
    assert research.require_research_progress(progress, plan_sha256=SHA, invocation=invocation,
        expected_selection_receipt_sha256=None, verify_file=True) == progress
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='FULL_REVERSE_NO_CANONICAL_STATE'):
        research.build_research_cursor(**args, full_state={'path': '/fixture/state.pt', 'sha256': SHA})


@pytest.mark.parametrize('changes,reason', [
    ({'optimizer_updates': 1848, 'parent_cursor': 29568}, 'CURSOR_COMPLETE'),
    ({'parent_cursor': 29584}, 'CURSOR_SCOPE'),
    ({'optimizer_updates': True}, 'CURSOR_SCOPE'),
    ({'parent_order_sha256': None}, 'MISSING_ORDER'),
    ({'parent_order_sha256': scope.TA_M5_EQUAL_LOSS_ORDER_SHA256}, 'UNCHANGED_ORDER'),
    ({'recipe_ref': {'path': str(scope.TA_M5_PAIRED_BATCH_ORDER_RECIPE_PATH), 'sha256': SHA}}, 'CURSOR_SCOPE'),
])
def test_invalid_cursor_cannot_claim_full_population_or_canonical_order(changes, reason):
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='FULL_REVERSE_' + reason):
        scope.build_ta_m5_full_reverse_order_cursor(**cursor_args(**changes))


def test_technical_complete_requires_all_panels_and_counts_even_when_score_fails(metadata, monkeypatch):
    cursor = scope.build_ta_m5_full_reverse_order_cursor(**cursor_args())
    report = report_for(cursor)
    monkeypatch.setattr(scope, '_read', lambda *unused: report)
    assert scope.require_ta_m5_full_reverse_order_cursor(cursor, expected_recipe=cursor['recipe']) == cursor
    for key in ('same_initial_state', 'same_training_queries', 'evaluation_state_unchanged',
                'all_panels_complete', 'all_original_rows_once', 'tail6_preserved'):
        report[key] = False
        with pytest.raises(scope.campaign.RandomAccessCampaignError, match='FULL_REVERSE_TECHNICAL_COMPLETION'):
            scope.require_ta_m5_full_reverse_order_cursor(cursor, expected_recipe=cursor['recipe'])
        report[key] = True
    report['counts']['evaluation_forwards'] = 1860
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='FULL_REVERSE_TECHNICAL_COMPLETION'):
        scope.require_ta_m5_full_reverse_order_cursor(cursor, expected_recipe=cursor['recipe'])


def test_exact_fullfit_gate_is_required(metadata):
    launch = recipe(metadata)
    gate = dict(decision='ADMITTED', status='ADMITTED', ta_m5_full_reverse_order_authorized=True,
        fold_id=1, selected_fit_rows=29574, arm='REVERSED', tail_rows=6,
        model_constructions_authorized=1, optimizer_constructions_authorized=1,
        model_forwards_authorized=3710, backward_calls_authorized=1849, optimizer_steps_authorized=1849,
        training_enabled=True, native_training_enabled=True, genuine_updates_authorized=True,
        genuine_backward_authorized=True, genuine_predictor_fit_authorized=True,
        full_epoch_training_allowed=True, canonical_resume_authority=False, candidate_selection=False,
        native_launch_authorized=False, normalization_fit_authorized=False,
        synthetic_test_execution_authorized=False, test_data_used=False,
        exit_evaluation=False, broker_access_allowed=False,
        evaluate_both_panels_unconditionally=True, per_update_pareto_requirement=False)
    scope.require_ta_m5_full_reverse_order_gate(gate, launch)
    assert launch['admission_stage'] == 'market_state_ta_m5_full_reverse_order_20261110_002'
    for key in ('native_launch_authorized', 'normalization_fit_authorized',
                'synthetic_test_execution_authorized', 'test_data_used', 'exit_evaluation',
                'broker_access_allowed'):
        missing = {name: value for name, value in gate.items() if name != key}
        for invalid in ({**gate, key: True}, missing):
            with pytest.raises(scope.campaign.RandomAccessCampaignError, match='FULL_REVERSE_POLICY_GATE'):
                scope.require_ta_m5_full_reverse_order_gate(invalid, launch)
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='FULL_REVERSE_POLICY_GATE'):
        scope.require_ta_m5_full_reverse_order_gate(gate, {
            **launch, 'admission_stage': 'market_state_ta_m5_full_reverse_order_20261110_001'})
    for key, bad in [('full_epoch_training_allowed', False), ('genuine_predictor_fit_authorized', False),
                     ('optimizer_steps_authorized', 1848), ('tail_rows', 0)]:
        with pytest.raises(scope.campaign.RandomAccessCampaignError, match='FULL_REVERSE_POLICY_GATE'):
            scope.require_ta_m5_full_reverse_order_gate({**gate, key: bad}, launch)
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='FULL_REVERSE_POLICY_GATE'):
        scope.require_ta_m5_full_reverse_order_gate(gate, {**launch, 'run_scope': scope.TA_M5_PAIRED_BATCH_ORDER_SCOPE})


@pytest.mark.parametrize('full_fit', [False, True])
def test_readiness_fullfit_flags_cannot_cross_old_and_new_scopes(monkeypatch, full_fit):
    recipe_path = scope.TA_M5_FULL_REVERSE_ORDER_RECIPE_PATH if full_fit else scope.TA_M5_PAIRED_BATCH_ORDER_RECIPE_PATH
    recipe_ref = {'path': str(recipe_path), 'sha256': SHA}
    run_scope = scope.TA_M5_FULL_REVERSE_ORDER_SCOPE if full_fit else scope.TA_M5_PAIRED_BATCH_ORDER_SCOPE
    launch = {'run_scope': run_scope, 'admission_stage': 'fixture', 'runtime_root': '/fixture/runtime'}
    gate = dict(recipe=recipe_ref, run_root=launch['runtime_root'], max_invocations=1,
        claimed_invocations=0, completed_invocations=0, relaunch_allowed=False,
        genuine_predictor_fit_authorized=full_fit, full_epoch_training_allowed=full_fit)
    gate.update({key: True for key in ('execution_authorized', 'data_reads_authorized',
        'genuine_constructor_authorized', 'genuine_forward_authorized', 'genuine_backward_authorized',
        'genuine_updates_authorized', 'training_enabled', 'native_training_enabled')})
    gate.update({key: False for key in ('native_launch_authorized', 'normalization_fit_authorized',
        'synthetic_test_execution_authorized', 'test_data_used', 'exit_evaluation', 'broker_access_allowed')})
    policy = dict(schema_version='gx1_next_native_run_policy_v1', canonical_source_repo=str(scope.CANONICAL_REPO),
        canonical_branch='work/gx1-current', training_module=research.MODULE, training_enabled=True,
        native_training_enabled=True, full_epoch_training_allowed=full_fit, fixture=gate,
        gpu_clock_launcher={'path': '/fixture/clock.ps1', 'sha256': SHA})
    monkeypatch.setattr(scope.campaign, 'read_bound_json',
        lambda path, sha: policy if path.name == 'NEXT_RUN_POLICY.json' else {'recipe': recipe_ref})
    monkeypatch.setattr(scope.campaign, 'file_sha256', lambda path: SHA)
    monkeypatch.setattr(scope.campaign, 'require_binding', lambda ref, **kwargs: dict(ref))
    monkeypatch.setattr(research, 'require_research_window_policy', lambda value: value)
    monkeypatch.setattr(research, 'require_research_launch_recipe', lambda *args, **kwargs: launch)
    monkeypatch.setattr(scope, 'require_ta_m5_full_reverse_order_gate', lambda *args: None)
    monkeypatch.setattr(scope, 'require_ta_m5_paired_batch_order_gate', lambda *args: None)
    monkeypatch.setattr(subprocess, 'check_output', lambda argv, **kwargs:
        'd' * 40 if 'rev-parse' in argv else 'work/gx1-current' if 'branch' in argv else '')
    def ready():
        return research.require_native_direction_research_readiness(scope.CANONICAL_REPO,
            window_path=Path('/fixture/window.json'), window_file_sha256=SHA)
    value = ready()['native_window_scope']
    assert value['optimizer_step_ceiling'] == (1849 if full_fit else 512)
    assert value['full_epoch_training_allowed'] is full_fit
    policy['full_epoch_training_allowed'] = not full_fit
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='READINESS_GLOBAL'):
        ready()
    policy['full_epoch_training_allowed'] = full_fit
    gate['full_epoch_training_allowed'] = not full_fit
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='READINESS_EXACT_FULL_EPOCH_GATE'):
        ready()
    gate['full_epoch_training_allowed'] = full_fit
    gate['genuine_predictor_fit_authorized'] = not full_fit
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='READINESS_.*GATE'):
        ready()
    if not full_fit:
        gate['genuine_predictor_fit_authorized'] = False
        del gate['full_epoch_training_allowed']
        assert ready()['native_window_scope']['full_epoch_training_allowed'] is False


def test_actual_recipe_and_worker_dispatch_are_exact(metadata, monkeypatch):
    value, calls = {'run_scope': scope.TA_M5_FULL_REVERSE_ORDER_SCOPE}, []
    monkeypatch.setattr(scope.campaign, 'read_bound_json', lambda *unused: value)
    monkeypatch.setattr(scope, 'require_ta_m5_full_reverse_order_recipe',
        lambda received, **kwargs: calls.append(received) or received)
    ref = {'path': str(scope.TA_M5_FULL_REVERSE_ORDER_RECIPE_PATH), 'sha256': SHA}
    kwargs = dict(source_repo=scope.CANONICAL_REPO, source_commit='d' * 40)
    assert research.require_research_launch_recipe(ref, **kwargs) == value
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='FULL_REVERSE_RECIPE_PATH'):
        research.require_research_launch_recipe({**ref, 'path': str(scope.TA_M5_PAIRED_BATCH_ORDER_RECIPE_PATH)}, **kwargs)
    value['run_scope'] = scope.TA_M5_PAIRED_BATCH_ORDER_SCOPE
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='FULL_REVERSE_RECIPE_PATH'):
        research.require_research_launch_recipe(ref, **kwargs)
    assert len(calls) == 1
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.scripts import run_entry_native_direction_research_window_v1 as runner
    dispatched = []
    monkeypatch.setattr(cap, 'require_guarded_cuda_trainer_execution', lambda: {})
    modules = [('full_reverse_order', scope.TA_M5_FULL_REVERSE_ORDER_SCOPE),
               ('paired_batch_order', scope.TA_M5_PAIRED_BATCH_ORDER_SCOPE),
               ('repeated_pass', scope.TA_M5_REPEATED_PASS_SCOPE)]
    args = dict(policy_path=Path('/fixture/window.json'), policy_file_sha256=SHA,
                progress_path=Path('/fixture/progress.json'))
    for name, run_scope in modules:
        monkeypatch.setitem(sys.modules, 'gx1.audit.run_market_state_ta_m5_' + name + '_v1',
            SimpleNamespace(**{'run_' + name: lambda _name=name, **kwargs: dispatched.append((_name, kwargs)) or _name}))
        monkeypatch.setattr(research, 'require_native_direction_research_context',
            lambda *unused: ({}, {}, {}, {}, {'run_scope': run_scope}))
        assert runner.run_native_direction_research_window(**args) == name
    assert dispatched == [(name, args) for name, _ in modules]


def test_existing_12000_window_13800_outer_and_1849_update_budget(metadata, monkeypatch):
    ref = {'path': str(scope.TA_M5_FULL_REVERSE_ORDER_RECIPE_PATH), 'sha256': SHA}
    window = seal(dict(schema_version=research.WINDOW_SCHEMA, recipe=ref, invocation_number=1,
        max_invocation_seconds=12000, progress_path='/fixture/progress/window.json',
        campaign_cursor_path='/fixture/cursors/RESUME_CURSOR.json', output_directory='/fixture/output',
        test_data_used=False), 'policy_sha256')
    assert research.require_research_window_policy(window) == window
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='WINDOW_SCOPE'):
        research.require_research_window_policy(seal({**window, 'max_invocation_seconds': 7800}, 'policy_sha256'))
    launch = dict(run_scope=scope.TA_M5_FULL_REVERSE_ORDER_SCOPE,
                  numerical_recipe={'batch_size': 16}, recipe_sha256=SHA)
    monkeypatch.setattr(research, 'require_research_launch_recipe', lambda *args, **kwargs: launch)
    monkeypatch.setattr(scope.campaign, 'read_bound_json', lambda *unused: window)
    window_ref = {'path': '/fixture/window.json', 'sha256': SHA}
    execution = dict(prelaunch_manifest=ref, prelaunch_manifest_sha256=SHA,
        train_session_manifest=window_ref, train_session_manifest_sha256=window['policy_sha256'])
    invocation = dict(requires_fresh_windows_boot=True, kind=research.KIND, batch_size=16,
        epoch_index=None, invocation_number=1, window_index=0, window_count=1,
        optimizer_step_budget=1849, expected_success_outcome='COMPLETE', maximum_wall_seconds=13800,
        checkpoint=dict(pointer_path=window['campaign_cursor_path'], before_mode='GENESIS',
            predecessor_invocation_number=None, write_mode='ATOMIC_UPDATE'),
        progress_path=window['progress_path'], launcher_argv=['guard', '--',
            str(scope.CANONICAL_REPO / '.venv/bin/python'), '-m', research.MODULE,
            '--window-policy', window_ref['path'], '--window-policy-file-sha256', SHA,
            '--progress-path', window['progress_path']])
    kwargs = dict(source_repo=scope.CANONICAL_REPO, source_commit='d' * 40)
    research.require_research_invocation_execution(invocation, execution, **kwargs)
    for key, bad in [('maximum_wall_seconds', 8400), ('optimizer_step_budget', 512)]:
        with pytest.raises(scope.campaign.RandomAccessCampaignError, match='INVOCATION_BINDING'):
            research.require_research_invocation_execution({**invocation, key: bad}, execution, **kwargs)
