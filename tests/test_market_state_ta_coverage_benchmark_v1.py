"""Fixed coverage contract checks using synthetic metadata only; no model/data."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from gx1.contracts import market_state_ta_coverage_benchmark_v1 as scope


def ref(path):
    return dict(path=str(path), sha256='a' * 64)


def seal(value, key):
    value[key] = scope.campaign.canonical_sha256({k: v for k, v in value.items() if k != key})
    return value


def preparation():
    batches, excluded = scope._selection()
    return dict(status='METADATA_QUALIFIED_SOURCE_IMPLEMENTATION_AND_MODEL_ADMISSION_PENDING',
        benchmark_result=scope.CONTROL_RESULT_REF, benchmark_recipe=scope.CONTROL_RECIPE_REF,
        training_authorized=False, execution_authorized=False, test_data_used=False,
        batch_indices_by_pass=batches, excluded_complement_batch_indices=excluded,
        counts=dict(control_unique_rows=4096, candidate_unique_rows=12288, presentations_each=12288,
            batch_visits_each=768, required_actual_optimizer_updates_each=768, required_skips_each=0,
            excluded_complement_rows=736, partial_last_batch_rows_not_selected=6),
        shared_numerics=dict(initial_model_state_sha256=scope.MODEL_SHA256,
            initial_rng_state_sha256=scope.INITIAL_RNG_SHA256, optimizer_initialization='fresh_adamw_empty_state',
            fixed_pass0_query_policy=True, objective_recipe=dict(batch_size=16, future_weight=1.)),
        common_pass1=dict(model_state_sha256=scope.COMMON_PASS1_MODEL_SHA256,
            training_rng_sha256=scope.COMMON_PASS1_RNG_SHA256, checkpoint_or_optimizer_state_retained=False),
        unchanged_measurement_panels=dict(TRAIN=scope.TRAIN_BATCH_INDICES, LATER=scope.LATER_BATCH_INDICES))


def test_fixed_schedule_preserves_first_pass_and_excludes_original_complement_panel():
    first, second, third = [scope.training_batch_indices(p) for p in (1, 2, 3)]
    _, excluded = scope._selection()
    assert first == scope.base.TRAIN_BATCH_INDICES
    assert second[:3] == [2, 9, 16] and second[-3:] == [1827, 1834, 1841]
    assert third[:3] == [5, 12, 19] and third[-3:] == [1830, 1837, 1845]
    assert all(len(x) == len(set(x)) == 256 and x == sorted(x) for x in (first, second, third))
    assert len(set(first + second + third)) == 768
    assert len(excluded) == 46 and not set(excluded) & set(first + second + third)
    assert all(0 <= x < 1848 for x in first + second + third)
    first[0] = -1
    assert scope.training_batch_indices(1)[0] == 0
    assert scope.ARMS == ('UNIQUE_COVERAGE',)
    assert scope.MEASUREMENT_LABELS == ('INITIAL_TRAIN', 'INITIAL_LATER',
        'UNIQUE_COVERAGE_PASS1_TRAIN', 'UNIQUE_COVERAGE_PASS1_LATER',
        'UNIQUE_COVERAGE_PASS3_TRAIN', 'UNIQUE_COVERAGE_PASS3_LATER')


@pytest.mark.parametrize('value', [True, False, 0, 4, 1., '1', None])
def test_training_pass_requires_exact_integer_in_fixed_schedule(value):
    with pytest.raises(RuntimeError, match='PASS_NUMBER'):
        scope.training_batch_indices(value)


@pytest.mark.parametrize('fault', ['none', 'duplicate', 'reordered', 'keepout', 'partial_tail', 'count', 'reset', 'fake_optimizer_parity'])
def test_bound_preparation_cannot_change_schedule_or_claim_extra_evidence(fault):
    value = preparation()
    if fault == 'duplicate': value['batch_indices_by_pass']['2'][1] = value['batch_indices_by_pass']['2'][0]
    elif fault == 'reordered': value['batch_indices_by_pass']['3'].reverse()
    elif fault == 'keepout': value['excluded_complement_batch_indices'].pop()
    elif fault == 'partial_tail': value['batch_indices_by_pass']['3'][-1] = 1848
    elif fault == 'count': value['counts']['required_skips_each'] = 1
    elif fault == 'reset': value['shared_numerics']['initial_rng_state_sha256'] = '0' * 64
    elif fault == 'fake_optimizer_parity': value['common_pass1']['checkpoint_or_optimizer_state_retained'] = True
    if fault == 'none': assert scope._require_preparation(value) is value
    else:
        with pytest.raises(RuntimeError): scope._require_preparation(value)


def test_expected_recipe_keeps_full_loss_and_measurement_panels(monkeypatch):
    metadata = preparation(); full = metadata['shared_numerics']['objective_recipe']
    parent = dict(source_owners={})
    original = dict(numerical_recipe=deepcopy(full), input_refs={'unchanged': ref('/synthetic/input.json')},
        panel_batch_indices=list(scope.TRAIN_BATCH_INDICES), train_batch_indices=list(scope.TRAIN_BATCH_INDICES),
        later_batch_indices=list(scope.LATER_BATCH_INDICES), passes=[1, 2, 3], evaluation_passes=[0, 1, 3])
    monkeypatch.setattr(scope.base, '_expected', lambda: (deepcopy(original), parent))
    monkeypatch.setattr(scope, '_read', lambda r: metadata)
    expected, old = scope._expected()
    assert old is parent and expected['numerical_recipe'] == full
    assert expected['objective_recipes'] == {'UNIQUE_COVERAGE': full}
    assert expected['panel_batch_indices'] == expected['train_batch_indices'] == original['panel_batch_indices']
    assert expected['selected_fit_rows'] == expected['unique_training_rows'] == 12288
    assert expected['max_forwards'] == 2304 and expected['max_training_forwards'] == 768
    assert expected['max_evaluation_forwards'] == 1536 and expected['max_model_seconds'] == 6500
    assert expected['max_full_state_restores'] == expected['max_checkpoint_loads'] == 1
    assert expected['input_refs']['unchanged'] == original['input_refs']['unchanged']
    assert set(scope.CONTROL_REFS) <= set(expected['input_refs'])
    assert scope.base.ARMS == ('FULL_OBJECTIVE', 'M5_DIRECTION_ONLY')
    assert scope.base.BUDGETS['max_forwards'] == 4096


def test_recipe_rejects_resealed_budget_loss_selection_and_path_changes(monkeypatch):
    expected = dict(run_scope=scope.SCOPE, admission_stage=scope.STAGE, numerical_recipe={'batch_size':16},
        input_refs={}, panel_batch_indices=scope.TRAIN_BATCH_INDICES,
        batch_indices_by_pass=scope._selection()[0], **scope.BUDGETS)
    monkeypatch.setattr(scope, '_expected', lambda: (deepcopy(expected), dict(source_owners={})))
    monkeypatch.setattr(scope, 'read_origin', lambda v: {})
    monkeypatch.setattr(scope.campaign, 'require_binding', lambda r, **kw: r)
    monkeypatch.setattr(scope.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0))
    owners = {name: ref(scope.REPO/name) for name in scope.EXTRA_OWNERS}
    value = dict(expected, source_owners=owners, trace_worker=owners[scope.WORKER], mechanical_source_commit=scope.BASE_COMMIT)
    seal(value, 'recipe_sha256')
    assert scope.require_recipe(value, source_repo=scope.REPO, source_commit=scope.BASE_COMMIT) == value
    for field, changed in [('max_forwards', 2305), ('max_private_updates', 769),
            ('numerical_recipe', {'batch_size':16, 'm5_direction_only':True}), ('run_scope', scope.base.SCOPE),
            ('batch_indices_by_pass', {'1':list(range(256)), '2':list(range(256)), '3':list(range(256))})]:
        bad = deepcopy(value); bad[field] = changed; seal(bad, 'recipe_sha256')
        with pytest.raises(RuntimeError): scope.require_recipe(bad, source_repo=scope.REPO, source_commit=scope.BASE_COMMIT)
    with pytest.raises(RuntimeError): scope.require_recipe(value, source_repo=scope.REPO, source_commit=scope.BASE_COMMIT, verify_files=False)


def test_new_gate_does_not_accept_old_authority_or_boolean_counts():
    recipe = dict(run_scope=scope.SCOPE, admission_stage=scope.STAGE)
    scope.require_gate(scope.gate_fields(), recipe)
    for key, value in [('ta_coverage_benchmark_authorized',False), ('optimizer_steps_authorized',769),
                       ('model_constructions_authorized',True), ('canonical_resume_authority',True)]:
        gate = scope.gate_fields(); gate[key] = value
        with pytest.raises(RuntimeError): scope.require_gate(gate, recipe)
    with pytest.raises(RuntimeError): scope.require_gate(scope.base.gate_fields(), recipe)


def cursor(*, visits=768, updates=768, outcome='COMPLETE'):
    return scope.build_cursor(recipe_ref=ref(scope.RECIPE_PATH), outcome=outcome, optimizer_updates=updates,
        parent_cursor=16*visits, parent_order_sha256=None, full_state=None, result=ref(scope.ROOT/'RESULT.json'))


@pytest.mark.parametrize('outcome,visits,updates,valid', [
    ('COMPLETE',768,768,True), ('FAILED',0,0,True), ('FAILED',100,99,True),
    ('COMPLETE',768,767,False), ('COMPLETE',767,767,False), ('FAILED',769,768,False), ('FAILED',1,2,False)])
def test_cursor_distinguishes_consumed_failure_from_fixed768_completion(outcome, visits, updates, valid):
    if valid:
        value = cursor(outcome=outcome, visits=visits, updates=updates)
        assert value['canonical_resume_authority'] is False and value['full_state'] is None
    else:
        with pytest.raises(RuntimeError): cursor(outcome=outcome, visits=visits, updates=updates)


def test_cursor_rejects_other_root_or_canonical_resume_even_if_resealed():
    value = cursor()
    for change in [dict(recipe=ref(scope.base.RECIPE_PATH)), dict(full_state=ref('/synthetic/state.pt')),
                   dict(parent_order_sha256='b'*64), dict(result=ref(scope.base.ROOT/'RESULT.json')),
                   dict(optimizer_updates=True)]:
        bad = {**value, **change}; seal(bad, 'cursor_sha256')
        with pytest.raises(RuntimeError): scope.require_cursor(bad, expected_recipe=value['recipe'], verify_files=False)


def completed_documents():
    value = cursor()
    rng = {'1':scope.COMMON_PASS1_RNG_SHA256, '2':'2'*64, '3':'3'*64}
    records = [dict(pass_number=p, original_batch_index=b, loss=.5, gradient_norm=1., skip_update=False)
               for p in scope.PASSES for b in scope.training_batch_indices(p)]
    control = dict(arms={'FULL_OBJECTIVE':dict(training_rng_by_pass=rng)},
                   training_records={'FULL_OBJECTIVE':deepcopy(records)})
    counters = dict(model_constructions=1, model_construction_returns=1, optimizer_constructions=1,
        checkpoint_loads=1, resume_state_load_calls=1, resume_state_load_returns=1,
        strict_full_state_restore_calls=1, strict_full_state_restore_returns=1, initial_rng_imports=1,
        training_forwards=768, evaluation_forwards=1536, forwards=2304, forward_returns=2304,
        candidate_evidence_writes=1, checkpoint_writes=0, control_capture_archive_decodes=4,
        input_sidecar_archive_decodes=12, skipped_batches=0,
        **{k:768 for k in ('backwards','backward_returns','optimizer_step_calls','optimizer_step_returns','optimizer_updates','batch_visits')})
    arm = dict(batch_visits=768, optimizer_updates=768, skipped_batches=0, passes_completed=3,
        initial_model_state_sha256=scope.MODEL_SHA256, initial_rng_state_sha256=scope.INITIAL_RNG_SHA256,
        fixed_pass0_query_policy=True, training_rng_by_pass=deepcopy(rng), final_model_state_sha256='f'*64,
        evidence_capsule=ref(scope.ROOT/'UNIQUE_COVERAGE_EVIDENCE.pt'))
    proof = dict(arrays_exact=True, model_state_exact=True, rows=4096)
    parity = dict(initial={p:deepcopy(proof) for p in ('TRAIN','LATER')},
        pass1=dict(model_state_exact=True, rng_exact=True, training_records_exact=True,
                   **{p:deepcopy(proof) for p in ('TRAIN','LATER')}), rng_after_each_pass_exact=True)
    refs = {k:ref(scope.ROOT/(k+'.json')) for k in scope.MEASUREMENT_LABELS}
    captures = {k:ref(scope.ROOT/(k+'_READOUT.npz')) for k in scope.MEASUREMENT_LABELS}
    docs = {scope.CONTROL_RESULT_REF['path']:control}
    for label in scope.MEASUREMENT_LABELS:
        panel = label.rsplit('_',1)[1]
        updates, digest = ((0,scope.MODEL_SHA256) if label.startswith('INITIAL_') else
            (256,scope.COMMON_PASS1_MODEL_SHA256) if '_PASS1_' in label else (768,arm['final_model_state_sha256']))
        docs[refs[label]['path']] = dict(capture=captures[label], evaluation_objective='FULL_OBJECTIVE',
            measurement=dict(rows=4096,batches=256,observation_chain_sha256=('1' if panel=='TRAIN' else '2')*64),
            optimizer_updates=updates,evaluated_model_state_sha256=digest,
            panel_batch_indices=scope.TRAIN_BATCH_INDICES if panel=='TRAIN' else scope.LATER_BATCH_INDICES,
            evaluation_state_unchanged=True,test_data_used=False)
    report = dict(schema_version=scope.RESULT_SCHEMA,outcome='COMPLETE',recipe=value['recipe'],
        parent_cursor=12288,optimizer_updates=768,batch_visits=768,skipped_batches=0,
        canonical_resume_authority=False,test_data_used=False,counts=counters,arms={'UNIQUE_COVERAGE':arm},
        measurement_refs=refs,panel_captures=captures,initial_restore_exact=True,evaluation_state_unchanged=True,
        source_unchanged=True,same_training_queries=True,rng_stream_parity=True,
        optimizer_initialization='fresh_adamw_empty_state',initial_model_state_sha256=scope.MODEL_SHA256,
        coverage_control_parity=parity,training_records={'UNIQUE_COVERAGE':records})
    docs[value['result']['path']] = report
    return value, report, docs


@pytest.mark.parametrize('fault', ['none','skip','control_decodes','initial_arrays','pass1_records','pass1_rows',
    'late_rng','repeated_batch','changed_prefix_record','pass1_model','wrong_capture','wrong_chain','wrong_evidence','source'])
def test_complete_requires_actual_control_parity_schedule_and_six_matched_panels(monkeypatch, fault):
    value, report, docs = completed_documents(); arm = report['arms']['UNIQUE_COVERAGE']
    if fault == 'skip': report['skipped_batches'] = 1
    elif fault == 'control_decodes': report['counts']['control_capture_archive_decodes'] = 3
    elif fault == 'initial_arrays': report['coverage_control_parity']['initial']['LATER']['arrays_exact'] = False
    elif fault == 'pass1_records': report['coverage_control_parity']['pass1']['training_records_exact'] = False
    elif fault == 'pass1_rows': report['coverage_control_parity']['pass1']['TRAIN']['rows'] = 4095
    elif fault == 'late_rng': arm['training_rng_by_pass']['3'] = '0'*64
    elif fault == 'repeated_batch': report['training_records']['UNIQUE_COVERAGE'][256]['original_batch_index'] = 0
    elif fault == 'changed_prefix_record': report['training_records']['UNIQUE_COVERAGE'][7]['gradient_norm'] = 2.
    elif fault == 'pass1_model': docs[report['measurement_refs']['UNIQUE_COVERAGE_PASS1_TRAIN']['path']]['evaluated_model_state_sha256'] = '0'*64
    elif fault == 'wrong_capture': docs[report['measurement_refs']['INITIAL_TRAIN']['path']]['capture'] = ref('/synthetic/wrong.npz')
    elif fault == 'wrong_chain': docs[report['measurement_refs']['UNIQUE_COVERAGE_PASS3_LATER']['path']]['measurement']['observation_chain_sha256'] = '0'*64
    elif fault == 'wrong_evidence': arm['evidence_capsule'] = ref(scope.base.ROOT/'FULL_OBJECTIVE_EVIDENCE.pt')
    elif fault == 'source': report['source_unchanged'] = False
    monkeypatch.setattr(scope.campaign, 'require_binding', lambda r, **kw:r)
    monkeypatch.setattr(scope, '_read', lambda r:docs[r['path']])
    if fault == 'none': assert scope.require_cursor(value,expected_recipe=value['recipe']) == value
    else:
        with pytest.raises(RuntimeError): scope.require_cursor(value,expected_recipe=value['recipe'])


def test_origin_binds_retained_control_and_checks_unchanged_numerical_sources(monkeypatch):
    metadata = preparation(); full = metadata['shared_numerics']['objective_recipe']
    records = [dict(pass_number=1, original_batch_index=b, loss=.5, gradient_norm=1., skip_update=False)
               for b in scope.TRAIN_BATCH_INDICES]
    refs = {p:ref('/synthetic/PASS1_'+p+'.json') for p in ('TRAIN','LATER')}
    metadata['common_pass1'].update(records=deepcopy(records),measurements=refs)
    control = dict(outcome='COMPLETE',source_unchanged=True,initial_restore_exact=True,evaluation_state_unchanged=True,
        same_training_queries=True,rng_stream_parity=True,test_data_used=False,recipe=scope.CONTROL_RECIPE_REF,
        arms={'FULL_OBJECTIVE':dict(optimizer_updates=768,batch_visits=768,skipped_batches=0,
            initial_model_state_sha256=scope.MODEL_SHA256,initial_rng_state_sha256=scope.INITIAL_RNG_SHA256,
            fixed_pass0_query_policy=True,training_rng_by_pass={'1':scope.COMMON_PASS1_RNG_SHA256})},
        training_records={'FULL_OBJECTIVE':records*3},
        measurement_refs={'FULL_OBJECTIVE_PASS1_'+p:r for p,r in refs.items()})
    numerical_owner = 'gx1/contracts/market_state_time_query_model_v2.py'
    original = dict(numerical_recipe=full,objective_recipes={'FULL_OBJECTIVE':full},expected_view_sha256='v',
        source_owners={n:ref(scope.REPO/n) for n in (numerical_owner,scope.WORKER)})
    recipe = dict(input_refs=deepcopy(scope.CONTROL_REFS),numerical_recipe=full,
        objective_recipes={'UNIQUE_COVERAGE':full},expected_view_sha256='v')
    docs = {scope.COVERAGE_METADATA_REF['path']:metadata,scope.CONTROL_RESULT_REF['path']:control,
            scope.CONTROL_RECIPE_REF['path']:original}
    docs.update({r['path']:dict(evaluated_model_state_sha256=scope.COMMON_PASS1_MODEL_SHA256,
        optimizer_updates=256,evaluation_state_unchanged=True) for r in refs.values()})
    checked = []
    monkeypatch.setattr(scope.base, 'read_origin', lambda value:dict(view={'contract_sha256':'v'}))
    monkeypatch.setattr(scope, '_read', lambda r:docs[r['path']])
    monkeypatch.setattr(scope.campaign, 'require_binding', lambda r, **kw:checked.append(r))
    got = scope.read_origin(recipe)
    assert got['coverage_control'] is control and got['coverage_metadata'] is metadata
    assert checked == [original['source_owners'][numerical_owner]]
    metadata['common_pass1']['records'][0]['loss'] = .6
    with pytest.raises(RuntimeError,match='COMMON_PASS1_RECORDS'): scope.read_origin(recipe)


def test_metadata_preflight_never_runs_a_model_or_loads_a_checkpoint(monkeypatch):
    calls = []
    monkeypatch.setattr(scope, 'require_recipe', lambda value, **kw:calls.append(kw))
    result = scope.preflight_read_only({},source_commit=scope.BASE_COMMIT)
    assert len(calls) == 1
    assert result['model_constructions'] == result['checkpoint_loads'] == result['data_payload_reads'] == 0
    assert result['optimizer_updates'] == 0 and result['reused_control'] is True
