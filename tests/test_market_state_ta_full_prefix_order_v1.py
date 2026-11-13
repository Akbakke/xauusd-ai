"""Synthetic fixed-order contracts; no model, market payload or checkpoint reads."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from gx1.contracts import market_state_ta_full_prefix_order_v1 as scope
from tests.test_market_state_ta_full_prefix_learning_v1 import completed_documents, ref, seal

full = scope.full
MODEL_OWNER = 'gx1/contracts/market_state_time_query_model_v2.py'


def synthetic_recipe_origin():
    original = dict(schema_version=full.RECIPE_SCHEMA, admission_stage=full.STAGE, run_scope=full.SCOPE,
        runtime_root=str(full.RUNTIME_ROOT), execution_binding_path=str(full.RUNTIME_ROOT/'EXECUTION_BINDING.json'),
        input_refs={**full.CONTROL_REFS, 'feature_view':scope.VIEW_REF}, numerical_recipe={'batch_size':16,'future_weight':1.},
        objective_recipes={'FULL_PREFIX':{'batch_size':16,'future_weight':1.}}, expected_view_sha256='v',
        candidate_inputs={**full.CONTROL_REFS, 'feature_view':scope.VIEW_REF},
        batch_indices_by_pass={str(p):full.training_batch_indices(p) for p in full.PASSES},
        train_batch_indices=full.training_batch_indices(1), parent_selection='chronological',
        panel_batch_indices=list(scope.TRAIN_BATCH_INDICES), later_batch_indices=list(scope.LATER_BATCH_INDICES),
        selected_fit_rows=29574, unique_training_rows=29574, training_presentations=59148,
        arms=list(scope.ARMS), passes=list(scope.PASSES), evaluation_passes=list(scope.EVALUATION_PASSES),
        primary_pass=2, dedicated_tf_activity=True, dedicated_non_m5_side=False,
        full_epoch_training_allowed=False, canonical_resume_authority=False, test_data_used=False, **full.BUDGETS)
    names={scope.WORKER,full.OWNER,MODEL_OWNER,'scripts/gx1_capped_run.sh'}
    owners={name:ref(scope.REPO/name) for name in names}
    control=dict(deepcopy(original),source_owners=owners,trace_worker=owners[scope.WORKER],
        mechanical_source_commit=full.BASE_COMMIT)
    seal(control,'recipe_sha256')
    return original,control


def install_recipe_origin(monkeypatch):
    original,control=synthetic_recipe_origin()
    monkeypatch.setattr(full,'_expected',lambda:(deepcopy(original),{}))
    monkeypatch.setattr(scope,'_read',lambda r:control if r==scope.ORDER_CONTROL_RECIPE_REF else None)
    monkeypatch.setattr(scope,'_ref',lambda path:ref(path))
    monkeypatch.setattr(scope,'read_origin',lambda recipe:{})
    monkeypatch.setattr(scope.campaign,'require_binding',lambda value,**kw:value)
    monkeypatch.setattr(scope.subprocess,'run',lambda *args,**kw:SimpleNamespace(returncode=0))
    return original,control


def test_transpose_preserves_original_groups_tail_and_two_complete_populations():
    order=scope.training_batch_indices(1)
    assert len(order)==1849 and sorted(order)==list(range(1849))
    assert order[:3]==[0,43,86] and order[42:45]==[1806,1,44]
    assert order[-2:]==[1805,1848] and scope.batch_rows(order[-1])==6
    assert sum(scope.batch_rows(b) for b in order)==29574
    assert scope.training_batch_indices(2)==order
    assert full.training_batch_indices(1)==full.training_batch_indices(2)==list(range(1849))
    assert scope.BUDGETS==full.BUDGETS and scope.BUDGETS['max_forwards']==5234
    assert scope.ARMS==('FULL_PREFIX',) and scope.EVALUATION_PASSES==(0,1,2)
    order[0]=-1
    assert scope.training_batch_indices(1)[0]==0


@pytest.mark.parametrize('value',[True,0,3])
def test_pass_requires_one_or_two_exact_integers(value):
    with pytest.raises(RuntimeError,match='PASS_NUMBER'):scope.training_batch_indices(value)


def test_new_expected_recipe_changes_only_scope_refs_and_fixed_visit_order(monkeypatch):
    original,control=install_recipe_origin(monkeypatch)
    before=deepcopy((original,control))
    expected,parent=scope._expected()
    assert (original,control)==before and parent is control
    changed={k for k in original if expected[k]!=original[k]}
    assert changed=={'schema_version','admission_stage','run_scope','runtime_root','execution_binding_path',
        'input_refs','candidate_inputs','batch_indices_by_pass','train_batch_indices','parent_selection'}
    assert expected['numerical_recipe']==original['numerical_recipe']
    assert expected['panel_batch_indices']==original['panel_batch_indices']
    assert expected['candidate_inputs']==expected['input_refs']
    assert all(expected['input_refs'][k]==v for k,v in scope.ORDER_CONTROL_REFS.items())
    assert expected['primary_pass']==2 and expected['full_epoch_training_allowed'] is False


def test_wrong_resealed_full_control_recipe_is_rejected(monkeypatch):
    _,control=install_recipe_origin(monkeypatch)
    control['max_private_updates']=1849;seal(control,'recipe_sha256')
    with pytest.raises(RuntimeError,match='EXACT_FULL_CONTROL_RECIPE'):scope._expected()


@pytest.mark.parametrize('fault',['duplicate','incomplete','chronological','tail_moved','bool_index',
    'budget','scope','control_ref','numerics','test_authority'])
def test_fixed_recipe_rejects_resealed_schedule_budget_scope_and_control(monkeypatch,fault):
    install_recipe_origin(monkeypatch)
    value=scope.build_recipe(source_commit=scope.BASE_COMMIT)
    if fault=='duplicate':value['batch_indices_by_pass']['2'][1]=0
    elif fault=='incomplete':value['batch_indices_by_pass']['1'].pop()
    elif fault=='chronological':value['batch_indices_by_pass']['2']=list(range(1849))
    elif fault=='tail_moved':value['batch_indices_by_pass']['1'][0],value['batch_indices_by_pass']['1'][-1]=1848,0
    elif fault=='bool_index':value['batch_indices_by_pass']['1'][0]=False
    elif fault=='budget':value['max_private_updates']=3697
    elif fault=='scope':value['run_scope']=full.SCOPE
    elif fault=='control_ref':value['input_refs']['order_control_result']=ref('/wrong/RESULT.json')
    elif fault=='numerics':value['numerical_recipe']['future_weight']=.5
    elif fault=='test_authority':value['test_data_used']=True
    seal(value,'recipe_sha256')
    with pytest.raises(RuntimeError,match='FIXED_RECIPE'):
        scope.require_recipe(value,source_repo=scope.REPO,source_commit=scope.BASE_COMMIT)


def test_source_bridge_allows_only_existing_orchestration_and_explicit_contract(monkeypatch):
    _,control=install_recipe_origin(monkeypatch)
    recipe=scope.build_recipe(source_commit=scope.BASE_COMMIT)
    for name in (scope.WORKER,full.OWNER,'scripts/gx1_capped_run.sh'):
        recipe['source_owners'][name]['sha256']='b'*64
    scope._require_source_bridge(recipe,control)
    recipe['source_owners'][MODEL_OWNER]['sha256']='b'*64
    with pytest.raises(RuntimeError,match='UNCHANGED_NUMERICAL_SOURCE'):
        scope._require_source_bridge(recipe,control)
    recipe['source_owners'][MODEL_OWNER]=deepcopy(control['source_owners'][MODEL_OWNER])
    recipe['source_owners'][scope.WORKER]['path']='/wrong/worker.py'
    with pytest.raises(RuntimeError,match='SOURCE_PATH'):scope._require_source_bridge(recipe,control)
    recipe['source_owners'].pop(scope.OWNER)
    with pytest.raises(RuntimeError,match='SOURCE_INVENTORY'):scope._require_source_bridge(recipe,control)


def origin_documents(monkeypatch):
    _,control_recipe=synthetic_recipe_origin()
    _,control,docs=completed_documents()
    control['recipe']=scope.ORDER_CONTROL_RECIPE_REF
    control['measurements']=deepcopy(docs[full.CONTROL_RESULT_REF['path']]['measurements'])
    control['arms']['FULL_PREFIX']['training_rng_state_sha256']='2'*64
    terminal=dict(status='COMPLETE',exit_code=0,recipe=scope.ORDER_CONTROL_RECIPE_REF,
        result=scope.ORDER_CONTROL_RESULT_REF,source_unchanged=True,no_relaunch_under_this_plan=True,test_data_used=False)
    docs[scope.ORDER_CONTROL_RESULT_REF['path']]=control
    docs[scope.ORDER_CONTROL_TERMINAL_REF['path']]=terminal
    recipe=dict(input_refs=deepcopy(scope.ORDER_CONTROL_REFS),numerical_recipe=control_recipe['numerical_recipe'],
        objective_recipes=control_recipe['objective_recipes'],expected_view_sha256=control_recipe['expected_view_sha256'],
        source_owners={**deepcopy(control_recipe['source_owners']),scope.OWNER:ref(scope.REPO/scope.OWNER)})
    monkeypatch.setattr(scope,'_expected',lambda:({},control_recipe))
    monkeypatch.setattr(scope,'_read',lambda r:docs[r['path']])
    monkeypatch.setattr(full,'read_origin',lambda r:{'full_prefix_control':docs[full.CONTROL_RESULT_REF['path']]})
    monkeypatch.setattr(scope.campaign,'require_binding',lambda value,**kw:value)
    return recipe,control,terminal,docs


@pytest.mark.parametrize('fault',['none','terminal_ref','updates','initial_rng','initial_measurement','initial_capture','initial_rows'])
def test_origin_binds_exact_full002_receipts_and_both_initial_panels(monkeypatch,fault):
    recipe,control,terminal,docs=origin_documents(monkeypatch)
    initial=docs[control['measurement_refs']['INITIAL_LATER']['path']]
    if fault=='terminal_ref':terminal['result']=ref('/wrong/RESULT.json')
    elif fault=='updates':control['optimizer_updates']=1849
    elif fault=='initial_rng':control['arms']['FULL_PREFIX']['initial_rng_state_sha256']='0'*64
    elif fault=='initial_measurement':initial['measurement']['loss']=123.
    elif fault=='initial_capture':initial['capture']=ref('/wrong/capture.npz')
    elif fault=='initial_rows':initial['panel_batch_indices']=list(reversed(scope.LATER_BATCH_INDICES))
    if fault=='none':
        result=scope.read_origin(recipe)
        assert result['order_control'] is control and 'full_prefix_control' in result
    else:
        with pytest.raises(RuntimeError):scope.read_origin(recipe)


def order_completed_documents():
    _,control,old_docs=completed_documents()
    report=deepcopy(control)
    report.update(schema_version=scope.RESULT_SCHEMA,recipe=ref(scope.RECIPE_PATH),rng_stream_parity=True,
        exposure_schedule={str(p):scope.training_batch_indices(p) for p in scope.PASSES},primary_endpoint_pass=2)
    report['arms']['FULL_PREFIX']['evidence_capsule']=ref(scope.ROOT/'FULL_PREFIX_EVIDENCE.pt')
    report['arms']['FULL_PREFIX']['training_rng_state_sha256']='2'*64
    report['training_records']['FULL_PREFIX']=[dict(pass_number=p,original_batch_index=b,
        rows=scope.batch_rows(b),skip_update=False) for p in scope.PASSES for b in scope.training_batch_indices(p)]
    docs={full.CONTROL_RESULT_REF['path']:old_docs[full.CONTROL_RESULT_REF['path']],
        scope.ORDER_CONTROL_RESULT_REF['path']:control}
    for label in scope.MEASUREMENT_LABELS:
        measured=deepcopy(old_docs[control['measurement_refs'][label]['path']])
        report['measurement_refs'][label]=ref(scope.ROOT/(label+'.json'))
        report['panel_captures'][label]=ref(scope.ROOT/(label+'_READOUT.npz'))
        measured['capture']=report['panel_captures'][label]
        docs[report['measurement_refs'][label]['path']]=measured
    result_ref=ref(scope.ROOT/'RESULT.json');docs[result_ref['path']]=report
    cursor=scope.build_cursor(recipe_ref=report['recipe'],outcome='COMPLETE',optimizer_updates=3698,
        parent_cursor=59148,parent_order_sha256=None,full_state=None,result=result_ref)
    return cursor,report,docs


@pytest.mark.parametrize('fault',['none','rng_flag','rng_pass','rng_endpoint','chronological_records',
    'incomplete','tail','skip','exposure_schedule','initial_parity','forward_count'])
def test_complete_checks_actual_order_rng_tail_and_reuses_original_completion(monkeypatch,fault):
    value,report,docs=order_completed_documents()
    arm=report['arms']['FULL_PREFIX']
    if fault=='rng_flag':report['rng_stream_parity']=None
    elif fault=='rng_pass':arm['training_rng_by_pass']['1']='c'*64
    elif fault=='rng_endpoint':arm['training_rng_state_sha256']='c'*64
    elif fault=='chronological_records':report['training_records']['FULL_PREFIX'].sort(key=lambda r:(r['pass_number'],r['original_batch_index']))
    elif fault=='incomplete':report['training_records']['FULL_PREFIX'].pop()
    elif fault=='tail':report['training_records']['FULL_PREFIX'][1848]['rows']=16
    elif fault=='skip':report['training_records']['FULL_PREFIX'][10]['skip_update']=True
    elif fault=='exposure_schedule':report['exposure_schedule']['2']=list(range(1849))
    elif fault=='initial_parity':report['full_prefix_control_parity']['initial']['TRAIN']['arrays_exact']=False
    elif fault=='forward_count':report['counts']['forwards']=5233
    monkeypatch.setattr(scope.campaign,'require_binding',lambda r,**kw:r)
    monkeypatch.setattr(full,'_read',lambda r:docs[r['path']])
    monkeypatch.setattr(scope,'_read',lambda r:docs[r['path']])
    if fault=='none':assert scope.require_cursor(value,expected_recipe=value['recipe'])==value
    else:
        with pytest.raises(RuntimeError):scope.require_cursor(value,expected_recipe=value['recipe'])


@pytest.mark.parametrize('visits,rows',[(0,0),(1848,29568),(1849,29574),(3697,59142),(3698,59148)])
def test_failure_cursor_retains_real_tail_presentations_and_new_schema(visits,rows):
    assert scope.cursor_from_batch_visits(visits)==rows
    got=scope.build_cursor(recipe_ref=ref(scope.RECIPE_PATH),outcome='FAILED',optimizer_updates=max(0,visits-1),
        parent_cursor=rows,parent_order_sha256=None,full_state=None,result=ref(scope.ROOT/'RESULT.json'))
    assert got['schema_version']==scope.CURSOR_SCHEMA and got['parent_cursor']==rows
    assert got['canonical_resume_authority'] is False


def test_parameterization_preserves_old_completion_and_cannot_relabel_old_cursor(monkeypatch):
    old,report,docs=completed_documents()
    before=deepcopy(old)
    monkeypatch.setattr(scope.campaign,'require_binding',lambda r,**kw:r)
    monkeypatch.setattr(full,'_read',lambda r:docs[r['path']])
    assert full.require_cursor(old,expected_recipe=old['recipe'])==before
    assert report['rng_stream_parity'] is None
    with pytest.raises(RuntimeError):scope.require_cursor(old,expected_recipe=old['recipe'])
    report['rng_stream_parity']=True
    with pytest.raises(RuntimeError):full.require_cursor(old,expected_recipe=old['recipe'])


def test_new_gate_does_not_reuse_consumed_full_permission_or_open_readiness():
    recipe=dict(run_scope=scope.SCOPE,admission_stage=scope.STAGE)
    scope.require_gate(scope.gate_fields(),recipe)
    assert 'ta_full_prefix_learning_authorized' not in scope.gate_fields()
    with pytest.raises(RuntimeError,match='POLICY_GATE'):scope.require_gate(full.gate_fields(),recipe)
    with pytest.raises(RuntimeError,match='POLICY_GATE'):
        scope.require_gate(scope.gate_fields(),dict(run_scope=full.SCOPE,admission_stage=full.STAGE))
    for key in ('native_launch_authorized','normalization_fit_authorized','synthetic_test_execution_authorized',
                'full_epoch_training_allowed','exit_evaluation','broker_access_allowed'):
        gate=scope.gate_fields();assert gate[key] is False;gate[key]=True
        with pytest.raises(RuntimeError,match='POLICY_GATE'):scope.require_gate(gate,recipe)


def test_preflight_only_qualifies_metadata_without_claim_or_payload(monkeypatch):
    calls=[]
    monkeypatch.setattr(scope,'require_recipe',lambda recipe,**kw:calls.append((recipe,kw)))
    result=scope.preflight_read_only({},source_commit=scope.BASE_COMMIT)
    assert len(calls)==1 and result['order_control']==scope.ORDER_CONTROL_RESULT_REF
    assert result['model_constructions']==result['data_payload_reads']==result['checkpoint_loads']==result['optimizer_updates']==0
    assert result['unique_training_rows']==29574 and result['training_presentations']==59148
