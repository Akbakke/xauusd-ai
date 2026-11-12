"""Synthetic metadata/cursor proofs only; no model, array or checkpoint loads."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from gx1.contracts import market_state_ta_full_prefix_learning_v1 as scope


def ref(path):
    return dict(path=str(path), sha256='a' * 64)


def seal(value, key):
    value[key] = scope.campaign.canonical_sha256({k: v for k, v in value.items() if k != key})
    return value


def preparation():
    return dict(schema_version='gx1_ta_full_prefix_learning_preparation_v1',
        status='METADATA_QUALIFIED_SOURCE_IMPLEMENTATION_AND_MODEL_ADMISSION_PENDING',
        control_recipe=scope.CONTROL_RECIPE_REF, control_result=scope.CONTROL_RESULT_REF,
        feature_view=scope.VIEW_REF, training_authorized=False, execution_authorized=False, test_data_used=False,
        batch_indices_by_pass={str(p):list(range(1849)) for p in (1,2)},
        evaluation_passes=[0,1,2], primary_pass=2,
        counts=dict(unique_training_rows=29574, training_presentations=59148, passes=2,
            batches_per_pass=1849, batch_visits=3698, required_optimizer_updates=3698,
            required_skips=0, tail_rows_each_pass=6, previous_complement_rows_included=736),
        shared_numerics=dict(initial_model_state_sha256=scope.MODEL_SHA256,
            initial_rng_state_sha256=scope.INITIAL_RNG_SHA256, optimizer_initialization='fresh_adamw_empty_state',
            fixed_pass0_query_policy=True, objective_recipe=dict(batch_size=16, future_weight=1.)),
        unchanged_measurement_panels=dict(TRAIN=scope.TRAIN_BATCH_INDICES, LATER=scope.LATER_BATCH_INDICES))


def test_two_full_schedules_include_tail_and_every_old_complement_batch():
    one, two = scope.training_batch_indices(1), scope.training_batch_indices(2)
    assert one == two == list(range(1849))
    assert sum(scope.batch_rows(b) for b in one) == 29574
    assert [scope.batch_rows(b) for b in one[-2:]] == [16, 6]
    assert set(scope.coverage._selection()[1]) <= set(one)
    one[0] = -1
    assert scope.training_batch_indices(1)[0] == 0
    assert scope.ARMS == ('FULL_PREFIX',) and scope.EVALUATION_PASSES == (0,1,2)
    assert scope.MEASUREMENT_LABELS == ('INITIAL_TRAIN', 'INITIAL_LATER',
        'FULL_PREFIX_PASS1_TRAIN', 'FULL_PREFIX_PASS1_LATER', 'FULL_PREFIX_PASS2_TRAIN', 'FULL_PREFIX_PASS2_LATER')


@pytest.mark.parametrize('value', [True, False, 0, 3, 1., '1', None])
def test_pass_is_exact_integer_one_or_two(value):
    with pytest.raises(RuntimeError, match='PASS_NUMBER'):
        scope.training_batch_indices(value)


@pytest.mark.parametrize('value', [True, -1, 1849, 1848., None])
def test_batch_index_does_not_accept_alias_or_out_of_range(value):
    with pytest.raises(RuntimeError, match='BATCH_INDEX'):
        scope.batch_rows(value)


@pytest.mark.parametrize('visits,rows', [(0,0),(1,16),(1848,29568),(1849,29574),(1850,29590),(3697,59142),(3698,59148)])
def test_cursor_counts_actual_tail_rows_across_both_passes(visits, rows):
    assert scope.cursor_from_batch_visits(visits) == rows
    assert scope._visits_from_cursor(rows) == visits


@pytest.mark.parametrize('rows', [True, -1, 6, 29569, 29584, 59149, 59168])
def test_cursor_rejects_partial_or_naive_sixteen_times_visits(rows):
    with pytest.raises(RuntimeError, match='CURSOR_BOUNDARY'):
        scope._visits_from_cursor(rows)


@pytest.mark.parametrize('fault', ['none','drop_tail','repeat_subset','counts','primary','initial_rng','authority'])
def test_preparation_fixes_population_initial_and_no_admission(fault):
    value = preparation()
    if fault == 'drop_tail': value['batch_indices_by_pass']['2'].pop()
    elif fault == 'repeat_subset': value['batch_indices_by_pass']['1'][-1] = 0
    elif fault == 'counts': value['counts']['tail_rows_each_pass'] = 16
    elif fault == 'primary': value['primary_pass'] = 1
    elif fault == 'initial_rng': value['shared_numerics']['initial_rng_state_sha256'] = '0'*64
    elif fault == 'authority': value['execution_authorized'] = True
    if fault == 'none': assert scope._require_preparation(value) is value
    else:
        with pytest.raises(RuntimeError): scope._require_preparation(value)


def test_expected_recipe_preserves_control_numerics_and_evaluation_panels(monkeypatch):
    metadata = preparation(); full = metadata['shared_numerics']['objective_recipe']
    original = dict(numerical_recipe=deepcopy(full), objective_recipes={'UNIQUE_COVERAGE':deepcopy(full)},
        input_refs={'feature_view':scope.VIEW_REF}, source_owners={},
        panel_batch_indices=list(scope.TRAIN_BATCH_INDICES), later_batch_indices=list(scope.LATER_BATCH_INDICES),
        common_pass1_model_state_sha256='b'*64, common_pass1_rng_state_sha256='c'*64,
        full_epoch_training_allowed=False, canonical_resume_authority=False)
    before = deepcopy(original)
    monkeypatch.setattr(scope, '_read', lambda r:metadata if r == scope.PREPARATION_REF else original)
    value, parent = scope._expected()
    assert original == before and parent is original
    assert value['numerical_recipe'] == full and value['objective_recipes'] == {'FULL_PREFIX':full}
    assert value['dedicated_tf_activity'] is True and value['dedicated_non_m5_side'] is False
    assert value['panel_batch_indices'] == scope.TRAIN_BATCH_INDICES
    assert value['train_batch_indices'] == list(range(1849))
    assert value['excluded_complement_batch_indices'] == []
    assert value['selected_fit_rows'] == value['unique_training_rows'] == 29574
    assert value['training_presentations'] == 59148 and value['primary_pass'] == 2
    assert value['max_forwards'] == 5234 and value['max_training_forwards'] == 3698
    assert value['max_evaluation_forwards'] == 1536 and value['max_model_seconds'] == 10500
    assert value['max_control_capture_archive_decodes'] == 2
    assert 'common_pass1_model_state_sha256' not in value
    assert value['full_epoch_training_allowed'] is value['canonical_resume_authority'] is False
    assert scope.base.BUDGETS['max_forwards'] == 4096 and scope.coverage.PASSES == (1,2,3)


def test_recipe_rejects_resealed_budget_scope_variant_and_schedule(monkeypatch):
    expected = dict(run_scope=scope.SCOPE, admission_stage=scope.STAGE, input_refs={},
        numerical_recipe={'batch_size':16}, dedicated_non_m5_side=False,
        batch_indices_by_pass={str(p):scope.training_batch_indices(p) for p in scope.PASSES}, **scope.BUDGETS)
    control = dict(source_owners={scope.WORKER:ref(scope.REPO/scope.WORKER)})
    monkeypatch.setattr(scope, '_expected', lambda:(deepcopy(expected),control))
    monkeypatch.setattr(scope, 'read_origin', lambda value:{})
    monkeypatch.setattr(scope.campaign, 'require_binding', lambda value, **kw:value)
    monkeypatch.setattr(scope.subprocess, 'run', lambda *a, **kw:SimpleNamespace(returncode=0))
    owners = {n:ref(scope.REPO/n) for n in set(control['source_owners']) | scope.EXTRA_OWNERS}
    value = dict(expected,source_owners=owners,trace_worker=owners[scope.WORKER],mechanical_source_commit=scope.BASE_COMMIT)
    seal(value,'recipe_sha256')
    assert scope.require_recipe(value,source_repo=scope.REPO,source_commit=scope.BASE_COMMIT) == value
    for field, changed in [('max_private_updates',3697),('max_model_seconds',12000),
            ('dedicated_non_m5_side',True),('run_scope',scope.coverage.SCOPE),
            ('batch_indices_by_pass',{'1':list(range(1848)),'2':list(range(1848))})]:
        bad = deepcopy(value); bad[field] = changed; seal(bad,'recipe_sha256')
        with pytest.raises(RuntimeError): scope.require_recipe(bad,source_repo=scope.REPO,source_commit=scope.BASE_COMMIT)
    with pytest.raises(RuntimeError):
        scope.require_recipe(value,source_repo=scope.REPO,source_commit=scope.BASE_COMMIT,verify_files=False)


def test_gate_is_new_exact_scope_without_reusing_old_permission():
    recipe = dict(run_scope=scope.SCOPE,admission_stage=scope.STAGE)
    scope.require_gate(scope.gate_fields(),recipe)
    for change in [dict(ta_full_prefix_learning_authorized=False),dict(training_presentations=59168),
                   dict(model_constructions_authorized=True),dict(canonical_resume_authority=True)]:
        with pytest.raises(RuntimeError): scope.require_gate({**scope.gate_fields(),**change},recipe)
    with pytest.raises(RuntimeError): scope.require_gate(scope.coverage.gate_fields(),recipe)


def cursor(*, visits=3698, updates=3698, outcome='COMPLETE'):
    return scope.build_cursor(recipe_ref=ref(scope.RECIPE_PATH),outcome=outcome,optimizer_updates=updates,
        parent_cursor=scope.cursor_from_batch_visits(visits),parent_order_sha256=None,full_state=None,
        result=ref(scope.ROOT/'RESULT.json'))


@pytest.mark.parametrize('visits,updates,outcome,valid', [(3698,3698,'COMPLETE',True),
    (0,0,'FAILED',True),(1849,1849,'FAILED',True),(1850,1849,'FAILED',True),
    (3698,3697,'COMPLETE',False),(1849,1849,'COMPLETE',False),(1,2,'FAILED',False)])
def test_cursor_keeps_failure_truth_and_requires_full_completion(visits,updates,outcome,valid):
    if valid:
        got = cursor(visits=visits,updates=updates,outcome=outcome)
        assert got['canonical_resume_authority'] is False and got['full_state'] is None
    else:
        with pytest.raises(RuntimeError): cursor(visits=visits,updates=updates,outcome=outcome)


def completed_documents():
    value = cursor(); counters = dict(model_constructions=1,model_construction_returns=1,optimizer_constructions=1,
        checkpoint_loads=1,resume_state_load_calls=1,resume_state_load_returns=1,
        strict_full_state_restore_calls=1,strict_full_state_restore_returns=1,initial_rng_imports=1,
        training_forwards=3698,evaluation_forwards=1536,forwards=5234,forward_returns=5234,
        candidate_evidence_writes=1,checkpoint_writes=0,control_capture_archive_decodes=2,
        input_sidecar_archive_decodes=12,training_presentations=59148,skipped_batches=0,
        **{k:3698 for k in ('backwards','backward_returns','optimizer_step_calls','optimizer_step_returns','optimizer_updates','batch_visits')})
    arm = dict(batch_visits=3698,optimizer_updates=3698,skipped_batches=0,passes_completed=2,
        training_presentations=59148,initial_model_state_sha256=scope.MODEL_SHA256,
        initial_rng_state_sha256=scope.INITIAL_RNG_SHA256,fixed_pass0_query_policy=True,
        training_rng_by_pass={'1':'1'*64,'2':'2'*64},final_model_state_sha256='f'*64,
        evidence_capsule=ref(scope.ROOT/'FULL_PREFIX_EVIDENCE.pt'))
    records = [dict(pass_number=p,original_batch_index=b,rows=6 if b==1848 else 16,skip_update=False)
        for p in (1,2) for b in range(1849)]
    refs = {k:ref(scope.ROOT/(k+'.json')) for k in scope.MEASUREMENT_LABELS}
    captures = {k:ref(scope.ROOT/(k+'_READOUT.npz')) for k in scope.MEASUREMENT_LABELS}
    initial = {f'INITIAL_{p}':dict(rows=4096,batches=256,
        observation_chain_sha256=('1' if p=='TRAIN' else '2')*64) for p in ('TRAIN','LATER')}
    control = dict(measurements=deepcopy(initial))
    docs = {scope.CONTROL_RESULT_REF['path']:control}
    for label in scope.MEASUREMENT_LABELS:
        panel = label.rsplit('_',1)[1]
        phase = 0 if label.startswith('INITIAL_') else int(label.split('_PASS')[1].split('_')[0])
        docs[refs[label]['path']] = dict(capture=captures[label],evaluation_objective='FULL_OBJECTIVE',
            measurement=deepcopy(initial['INITIAL_'+panel]),optimizer_updates=phase*1849,
            evaluated_model_state_sha256=scope.MODEL_SHA256 if phase==0 else ('b'*64 if phase==1 else 'f'*64),
            panel_batch_indices=scope.TRAIN_BATCH_INDICES if panel=='TRAIN' else scope.LATER_BATCH_INDICES,
            evaluation_state_unchanged=True,test_data_used=False)
    report = dict(schema_version=scope.RESULT_SCHEMA,outcome='COMPLETE',recipe=value['recipe'],
        parent_cursor=59148,training_presentations=59148,optimizer_updates=3698,batch_visits=3698,skipped_batches=0,
        canonical_resume_authority=False,test_data_used=False,counts=counters,arms={'FULL_PREFIX':arm},
        measurement_refs=refs,panel_captures=captures,initial_restore_exact=True,evaluation_state_unchanged=True,
        source_unchanged=True,same_training_queries=True,rng_stream_parity=None,
        optimizer_initialization='fresh_adamw_empty_state',initial_model_state_sha256=scope.MODEL_SHA256,
        full_prefix_control_parity=dict(initial={p:dict(arrays_exact=True,model_state_exact=True,rows=4096)
            for p in ('TRAIN','LATER')}),training_records={'FULL_PREFIX':records})
    docs[value['result']['path']] = report
    return value,report,docs


@pytest.mark.parametrize('fault', ['none','tail_rows','row_total','repeat','skip','control_decodes',
    'initial_arrays','invented_rng_parity','missing_rng','initial_loss','phase_model','phase_updates','chain','evidence','source'])
def test_complete_requires_two_actual_tail_batches_initial_parity_and_both_endpoints(monkeypatch,fault):
    value,report,docs = completed_documents(); arm=report['arms']['FULL_PREFIX']
    if fault == 'tail_rows': report['training_records']['FULL_PREFIX'][1848]['rows'] = 16
    elif fault == 'row_total': report['counts']['training_presentations'] = 59168
    elif fault == 'repeat': report['training_records']['FULL_PREFIX'][1849]['original_batch_index'] = 1
    elif fault == 'skip': report['training_records']['FULL_PREFIX'][10]['skip_update'] = True
    elif fault == 'control_decodes': report['counts']['control_capture_archive_decodes'] = 4
    elif fault == 'initial_arrays': report['full_prefix_control_parity']['initial']['LATER']['arrays_exact'] = False
    elif fault == 'invented_rng_parity': report['rng_stream_parity'] = True
    elif fault == 'missing_rng': arm['training_rng_by_pass'].pop('2')
    elif fault == 'initial_loss': docs[report['measurement_refs']['INITIAL_TRAIN']['path']]['measurement']['loss'] = .2
    elif fault == 'phase_model': docs[report['measurement_refs']['FULL_PREFIX_PASS1_LATER']['path']]['evaluated_model_state_sha256'] = 'c'*64
    elif fault == 'phase_updates': docs[report['measurement_refs']['FULL_PREFIX_PASS1_TRAIN']['path']]['optimizer_updates'] = 256
    elif fault == 'chain': docs[report['measurement_refs']['FULL_PREFIX_PASS2_TRAIN']['path']]['measurement']['observation_chain_sha256'] = '0'*64
    elif fault == 'evidence': arm['evidence_capsule'] = ref(scope.coverage.ROOT/'UNIQUE_COVERAGE_EVIDENCE.pt')
    elif fault == 'source': report['source_unchanged'] = False
    monkeypatch.setattr(scope.campaign,'require_binding',lambda r,**kw:r)
    monkeypatch.setattr(scope,'_read',lambda r:docs[r['path']])
    if fault == 'none': assert scope.require_cursor(value,expected_recipe=value['recipe']) == value
    else:
        with pytest.raises(RuntimeError): scope.require_cursor(value,expected_recipe=value['recipe'])


def test_origin_reuses_current_source_bridge_but_only_control_model_and_initial(monkeypatch):
    metadata=preparation(); full=metadata['shared_numerics']['objective_recipe']
    wrappers={p:ref('/synthetic/INITIAL_'+p+'.json') for p in ('TRAIN','LATER')}
    captures={p:ref('/synthetic/INITIAL_'+p+'.npz') for p in ('TRAIN','LATER')}
    metadata['initial_measurements']=wrappers
    measured={p:dict(rows=4096) for p in ('TRAIN','LATER')}
    control=dict(outcome='COMPLETE',source_unchanged=True,initial_restore_exact=True,evaluation_state_unchanged=True,
        same_training_queries=True,test_data_used=False,recipe=scope.CONTROL_RECIPE_REF,
        optimizer_updates=768,skipped_batches=0,initial_model_state_sha256=scope.MODEL_SHA256,
        arms={'UNIQUE_COVERAGE':dict(initial_rng_state_sha256=scope.INITIAL_RNG_SHA256)},
        measurement_refs={'INITIAL_'+p:r for p,r in wrappers.items()},
        panel_captures={'INITIAL_'+p:r for p,r in captures.items()},measurements={'INITIAL_'+p:m for p,m in measured.items()})
    original=dict(numerical_recipe=full,expected_view_sha256='v')
    recipe=dict(input_refs={**scope.CONTROL_REFS,'feature_view':scope.VIEW_REF},numerical_recipe=full,
        objective_recipes={'FULL_PREFIX':full},expected_view_sha256='v',dedicated_tf_activity=True,dedicated_non_m5_side=False)
    docs={scope.PREPARATION_REF['path']:metadata,scope.CONTROL_RESULT_REF['path']:control,
        scope.CONTROL_TERMINAL_REF['path']:dict(status='COMPLETE',exit_code=0,result=scope.CONTROL_RESULT_REF)}
    docs.update({wrappers[p]['path']:dict(evaluated_model_state_sha256=scope.MODEL_SHA256,optimizer_updates=0,
        evaluation_state_unchanged=True,test_data_used=False,measurement=measured[p],capture=captures[p]) for p in wrappers})
    checks=[]
    monkeypatch.setattr(scope.qualified,'_metadata',lambda:(original,{'verified':True},None,None,None))
    monkeypatch.setattr(scope.qualified,'_require_source_bridge',lambda *args:checks.append(args))
    monkeypatch.setattr(scope.base,'read_origin',lambda r:dict(view={'contract_sha256':'v'}))
    monkeypatch.setattr(scope,'_read',lambda r:docs[r['path']])
    got=scope.read_origin(recipe)
    assert got['full_prefix_control'] is control and got['full_prefix_metadata'] is metadata
    assert checks[0][0] is recipe and checks[0][1] is original
    recipe['dedicated_non_m5_side']=True
    with pytest.raises(RuntimeError,match='ORIGIN_REFS_AND_NUMERICS'): scope.read_origin(recipe)


def test_preflight_is_metadata_only(monkeypatch):
    calls=[]
    monkeypatch.setattr(scope,'require_recipe',lambda value,**kw:calls.append(kw))
    result=scope.preflight_read_only({},source_commit=scope.BASE_COMMIT)
    assert len(calls)==1 and result['model_constructions']==result['checkpoint_loads']==result['data_payload_reads']==0
    assert result['optimizer_updates']==0 and result['unique_training_rows']==29574
    assert result['training_presentations']==59148 and result['tail_rows_each_pass']==6
