"""Focused receipt, admission and cursor checks for isolated TF activity learning."""
from copy import deepcopy

import pytest

from gx1.contracts import market_state_ta_tf_activity_short_learning_v1 as short
from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
from gx1.contracts import local_random_access_campaign_v2 as campaign


def ref(path):
    return dict(path=str(path),sha256='a'*64)


def cursor(complete=False):
    return short.build_cursor(recipe_ref=ref(short.RECIPE_PATH),outcome='COMPLETE' if complete else 'FAILED',
        optimizer_updates=256 if complete else 0,parent_cursor=4096 if complete else 0,
        parent_order_sha256=short.legacy.TA_M5_EQUAL_LOSS_ORDER_SHA256,
        full_state=ref(short.RUNTIME_ROOT/'private-research/FINAL_C171_SHORT_STATE.pt') if complete else None,
        result=ref(short.ROOT/'RESULT.json'))


def reseal(value,key):
    value[key] = campaign.canonical_sha256({k:v for k,v in value.items() if k!=key})
    return value


def test_failed_and_complete_cursors_route_to_exact_new_scope():
    for completed in (False,True):
        value=cursor(completed)
        assert authority.require_research_cursor(value,expected_recipe=ref(short.RECIPE_PATH),
            verify_files=False,batch_size=16)==value
        assert value['canonical_resume_authority'] is False


@pytest.mark.parametrize('changes',[
    dict(optimizer_updates=255),dict(parent_cursor=4095),dict(full_state=None),
    dict(parent_order_sha256='0'*64),dict(canonical_resume_authority=True),dict(test_data_used=True),
    dict(optimizer_updates=256.0),dict(recipe=ref(short.ROOT/'OTHER.json')),
])
def test_completed_cursor_cannot_widen_or_undercount(changes):
    value=cursor(True);value.update(changes);reseal(value,'cursor_sha256')
    with pytest.raises((RuntimeError,AttributeError)):
        short.require_cursor(value,expected_recipe=ref(short.RECIPE_PATH),verify_files=False)


@pytest.mark.parametrize('key,value',[
    ('optimizer_steps_authorized',257),('model_forwards_authorized',289),
    ('training_enabled',False),('candidate_selection',True),('canonical_resume_authority',True),
    ('model_constructions_authorized',True),('genuine_predictor_fit_authorized',True),
])
def test_policy_gate_rejects_scope_changes(key,value):
    gate=short.gate_fields();gate[key]=value
    with pytest.raises(RuntimeError):
        short.require_gate(gate,dict(run_scope=short.SCOPE,admission_stage=short.STAGE))


def test_policy_gate_accepts_only_current_short_recipe():
    short.require_gate(short.gate_fields(),dict(run_scope=short.SCOPE,admission_stage=short.STAGE))
    with pytest.raises(RuntimeError):
        short.require_gate(short.gate_fields(),dict(run_scope=short.SCOPE,admission_stage='old'))


def test_recipe_dispatch_checks_path_before_specialized_validation(monkeypatch):
    called=[]
    monkeypatch.setattr(campaign,'require_binding',lambda v,**kw:v)
    monkeypatch.setattr(campaign,'read_bound_json',lambda *a,**kw:dict(run_scope=short.SCOPE))
    monkeypatch.setattr(short,'require_recipe',lambda value,**kw:called.append((value,kw)) or value)
    assert authority.require_research_launch_recipe(ref(short.RECIPE_PATH),source_repo=short.REPO,
        source_commit=short.BASE_COMMIT)['run_scope']==short.SCOPE
    assert len(called)==1
    with pytest.raises(Exception):
        authority.require_research_launch_recipe(ref(short.ROOT/'WRONG.json'),source_repo=short.REPO,
            source_commit=short.BASE_COMMIT)
    assert len(called)==1


def test_exact_short_window_is_one_hour_and_old_windows_unchanged(monkeypatch):
    monkeypatch.setattr(campaign,'require_binding',lambda v,**kw:v)
    value=dict(schema_version=authority.WINDOW_SCHEMA,recipe=ref(short.RECIPE_PATH),invocation_number=1,
        max_invocation_seconds=3600,progress_path=str(short.RUNTIME_ROOT/'progress/one.json'),
        campaign_cursor_path=str(short.RUNTIME_ROOT/'cursor/one.json'),
        output_directory=str(short.RUNTIME_ROOT/'output'),test_data_used=False)
    reseal(value,'policy_sha256')
    assert authority.require_research_window_policy(value,verify_files=False)==value
    for limit in (1800,12000):
        changed=deepcopy(value);changed['max_invocation_seconds']=limit;reseal(changed,'policy_sha256')
        with pytest.raises(Exception):authority.require_research_window_policy(changed,verify_files=False)


@pytest.mark.parametrize('outcome,updates',[('COMPLETE',256),('FAILED',0),('FAILED',128)])
def test_progress_has_exact_short_budget_on_success_and_failure(outcome,updates):
    path=short.RUNTIME_ROOT/'research-cursor/RESUME_CURSOR.json'
    recipe=ref(short.RECIPE_PATH)
    invocation=dict(kind=authority.KIND,invocation_sha256='b'*64,direction_research_recipe=recipe,
        batch_size=16,optimizer_step_budget=256,checkpoint=dict(pointer_path=str(path)))
    result=authority.build_research_progress(window=dict(recipe=recipe),plan=dict(plan_sha256='c'*64),
        invocation=invocation,checkpoint_pointer=ref(path),optimizer_updates=updates,
        parent_cursor=16*updates,parent_order_sha256=short.legacy.TA_M5_EQUAL_LOSS_ORDER_SHA256,outcome=outcome)
    assert result['total_units']==256 and result['completed_units']==updates


@pytest.mark.parametrize('mutate',['none','wrong_after_rows','different_observations','wrong_model','missing_steps',
    'rng_false','rng_missing','wrong_rng_count','reference_missing','wrong_reference','wrong_reference_rows','wrong_tolerance','activity_not_qualified','side_not_exact','wrong_import_count','nonfinite_activity_delta'])
def test_completed_cursor_validates_actual_measurement_receipts(monkeypatch,mutate):
    value=cursor(True)
    refs={label:ref(short.ROOT/(label+'.json')) for label in ('INITIAL_TRAIN','FINAL_TRAIN')}
    measured={label:dict(measurement=dict(rows=4096,batches=256,observation_chain_sha256='d'*64),
        optimizer_updates=updates,panel_batch_indices=short.PANEL_BATCH_INDICES,
        evaluation_state_unchanged=True,test_data_used=False,
        evaluated_model_state_sha256=short.MODEL_SHA256 if updates==0 else 'e'*64)
        for label,updates in (('INITIAL_TRAIN',0),('FINAL_TRAIN',256))}
    counts=dict(model_constructions=2,optimizer_constructions=1,forwards=768,backwards=256,
        optimizer_step_calls=256,optimizer_updates=256,checkpoint_loads=2,strict_full_state_restore_calls=1,
        checkpoint_writes=2,staged_state_load_calls=2,staged_state_load_returns=2,
        initial_rng_imports=1,head_tensor_imports=20,training_forwards=256,evaluation_forwards=512,
        loss_component_owner_calls=512,archive_decodes=1,extra_target_gathers=1536)
    report=dict(schema_version=short.RESULT_SCHEMA,outcome='COMPLETE',recipe=value['recipe'],
        parent_cursor=4096,physical_optimizer_updates=256,parent_order_sha256=value['parent_order_sha256'],
        final_state=value['full_state'],canonical_resume_authority=False,test_data_used=False,counts=counts,
        initial_restore_exact=True,evaluation_state_unchanged=True,source_unchanged=True,
        optimizer_initialization='fresh_adamw_empty_state',initial_model_state_sha256=short.MODEL_SHA256,
        final_model_state_sha256='e'*64,panel_batch_indices=short.PANEL_BATCH_INDICES,
        initial_state=ref(short.RUNTIME_ROOT/'private-research/INITIAL_C171_STATE.pt'),
        measurement_refs=refs,measurements={label:dict(v['measurement']) for label,v in measured.items()})
    report.update(original_training_rng_endpoints_matched=True,
        full_initial_reference=dict(reference=short.INPUTS['full_prefix_capture'],rows=4096,batches=256,
            axes_targets_all_sides_bitexact=True,max_abs_activity={'state':9.5367431640625e-7,'future':2.384185791015625e-7},
            max_abs_loss=2.384185791015625e-7,atol=2e-6,rtol=1e-6,loss_atol=5e-6,cloned_activity_within_declared_tolerance=True))
    if mutate=='rng_false':report['original_training_rng_endpoints_matched']=False
    if mutate=='rng_missing':report.pop('original_training_rng_endpoints_matched')
    if mutate=='wrong_rng_count':counts['initial_rng_imports']=0
    if mutate=='reference_missing':report.pop('full_initial_reference')
    if mutate=='wrong_reference':report['full_initial_reference']['reference']=ref(short.ROOT/'OTHER.npz')
    if mutate=='wrong_reference_rows':report['full_initial_reference']['rows']=256
    if mutate=='wrong_tolerance':report['full_initial_reference']['atol']=1e-2
    if mutate=='activity_not_qualified':report['full_initial_reference']['cloned_activity_within_declared_tolerance']=False
    if mutate=='side_not_exact':report['full_initial_reference']['axes_targets_all_sides_bitexact']=False
    if mutate=='wrong_import_count':counts['head_tensor_imports']=16
    if mutate=='nonfinite_activity_delta':report['full_initial_reference']['max_abs_activity']['state']=float('inf')
    if mutate=='wrong_after_rows':measured['FINAL_TRAIN']['measurement']['rows']=255
    if mutate=='different_observations':measured['FINAL_TRAIN']['measurement']['observation_chain_sha256']='f'*64
    if mutate=='wrong_model':measured['FINAL_TRAIN']['evaluated_model_state_sha256']=short.MODEL_SHA256
    if mutate=='missing_steps':counts['optimizer_updates']=255
    documents={value['result']['path']:report,**{refs[k]['path']:v for k,v in measured.items()}}
    monkeypatch.setattr(campaign,'require_binding',lambda v,**kw:v)
    monkeypatch.setattr(short,'_read',lambda r:documents[r['path']])
    if mutate=='none':
        assert short.require_cursor(value,expected_recipe=value['recipe'])==value
    else:
        with pytest.raises(RuntimeError):short.require_cursor(value,expected_recipe=value['recipe'])


def test_old_and_new_recipe_paths_are_distinct():
    from gx1.contracts import market_state_ta_m5_short_learning_v1 as old
    assert authority._is_ta_tf_activity_short_learning_recipe_ref(ref(short.RECIPE_PATH))
    assert not authority._is_ta_m5_c171_short_learning_recipe_ref(ref(short.RECIPE_PATH))
    assert not authority._is_ta_tf_activity_short_learning_recipe_ref(ref(old.RECIPE_PATH))
    assert authority._is_ta_short_learning_recipe_ref(ref(old.RECIPE_PATH))
    assert authority._is_ta_short_learning_recipe_ref(ref(short.RECIPE_PATH))
    from gx1.contracts import market_state_ta_non_m5_short_learning_v1 as sides
    assert not authority._is_ta_tf_activity_short_learning_recipe_ref(ref(sides.RECIPE_PATH))
    assert authority._is_ta_short_learning_recipe_ref(ref(sides.RECIPE_PATH))


def test_numerical_settings_must_match_except_three_derived_identities():
    from gx1.contracts.market_state_time_query_fold1_fit_v1 import ta_sf_numerical_recipe
    old=ta_sf_numerical_recipe(dedicated_m5_side=True,m5_equal_observation=True)
    new=ta_sf_numerical_recipe(dedicated_m5_side=True,m5_equal_observation=True,dedicated_tf_activity=True)
    short.require_unchanged_numerics(new,old)
    new['optimizer']['kwargs']['lr']*=2
    with pytest.raises(RuntimeError,match='UNCHANGED_NUMERICAL_SETTINGS'):short.require_unchanged_numerics(new,old)


def test_source_bridge_rejects_unreviewed_numeric_owner_and_missing_required_change(monkeypatch):
    import hashlib
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
    names=set(short.NUMERICAL_CHANGES)|{f'fixed{i}.py' for i in range(85-len(short.NUMERICAL_CHANGES))}
    original={n:dict(path=str(short.REPO/n),sha256=hashlib.sha256(n.encode()).hexdigest()) for n in names}
    current=deepcopy(original)
    for n in short.NUMERICAL_CHANGES:current[n]['sha256']='b'*64
    current.update({n:dict(path=str(short.REPO/n),sha256='c'*64) for n in short.EXTRA_OWNERS})
    monkeypatch.setattr(fit,'required_source_paths',lambda **kw:names)
    monkeypatch.setattr(short.campaign,'require_binding',lambda *a,**kw:None)
    monkeypatch.setattr(short.subprocess,'check_output',lambda args,**kw:args[2].partition(':')[2].encode())
    short.current_source_bridge(original,current)
    changed=deepcopy(current);changed['fixed0.py']['sha256']='d'*64
    with pytest.raises(RuntimeError,match='UNREVIEWED_SOURCE'):short.current_source_bridge(original,changed)
    changed=deepcopy(current);n=next(iter(short.NUMERICAL_CHANGES));changed[n]=original[n]
    with pytest.raises(RuntimeError,match='EXACT_NUMERICAL_CHANGE_SET'):short.current_source_bridge(original,changed)


@pytest.mark.parametrize('path,expected',[
    ('/home/andre2/GX1_RUNS/MARKET_STATE_TA_TF_ACTIVITY_SHORT_LEARNING_20261112_001/RECIPE.json',True),
    ('/home/andre2/GX1_RUNS/MARKET_STATE_TA_NON_M5_SHORT_LEARNING_20261112_001/RECIPE.json',True),
    ('/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_C171_SHORT_LEARNING_20261111_002/RECIPE.json',True),
    ('/home/andre2/GX1_RUNS/MARKET_STATE_TA_TF_ACTIVITY_SHORT_LEARNING_20261112_002/RECIPE.json',False),
    ('/arbitrary/RECIPE.json',False),
])
def test_actual_cap_window_expression_accepts_only_exact_short_paths(tmp_path,path,expected):
    import json,re,subprocess,sys
    from pathlib import Path
    cap=(Path(__file__).resolve().parents[1]/'scripts/gx1_capped_run.sh').read_text()
    line=next(v for v in cap.splitlines() if v.strip().startswith('native_seconds='))
    code=re.search(r" -c '([^']+)' ",line).group(1)
    fixture=tmp_path/'window.json'
    fixture.write_text(json.dumps(dict(schema_version=authority.WINDOW_SCHEMA,max_invocation_seconds=3600,recipe={'path':path})))
    result=subprocess.run([sys.executable,'-c',code,str(fixture)],text=True,capture_output=True,timeout=10)
    assert (result.returncode==0) is expected
    if expected:assert result.stdout.strip()=='3600'


@pytest.mark.parametrize('fault',[
    'none','peer_status','peer_result_ref','peer_loss_tolerance','peer_activity_tolerance',
    'peer_side_parity','peer_axis_count','wrong_candidate','wrong_imports',
])
def test_origin_uses_actual_activity_peer_schema_and_accepts_nonzero_loss(monkeypatch,fault):
    from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero
    source_view=dict(contract_sha256='1'*64,research_objective={})
    view=dict(contract_sha256='2'*64,research_objective={'dedicated_tf_activity':True},dedicated_tf_activity=True)
    original_numerical=dict(research_objective_sha256='3'*64,native_F_loss_contract_sha256='4'*64,
                           contract_sha256='5'*64,future_weight=1.0)
    numerical=dict(research_objective_sha256='6'*64,native_F_loss_contract_sha256='7'*64,
                   contract_sha256='8'*64,future_weight=1.0)
    recipe=dict(candidate_inputs=short.INPUTS,source_owners={},numerical_recipe=numerical,
                expected_view_sha256=view['contract_sha256'])
    q=dict(view=source_view,receipts={},origin=dict(original_recipe={},original_result={}))
    plan=dict(source_view_sha256=source_view['contract_sha256'],trial_view_sha256=view['contract_sha256'])
    counts=dict(backbone_encodings=32,forward_returns=32,head_tensor_imports=20,optimizer_updates=0)
    capture=ref(short.initial.ROOT/'TRAIN_FUSION.npz')
    proof=dict(status='COMPLETE',source_unchanged=True,passivity={'source':True,'trial':True},counts=counts,
        phase_states={'CURRENT_SHARED_INITIAL':{'model':short.original_short.MODEL_SHA256},
                      'TF_ACTIVITY_INITIAL':{'model':short.MODEL_SHA256}},
        source_view_sha256=source_view['contract_sha256'],trial_view_sha256=view['contract_sha256'],
        artifacts={'initial_parity_capture':capture})
    terminal=dict(status='COMPLETE',exit_code=0,result=short.INPUTS['initial_parity_result'],
                  plan=short.INPUTS['initial_parity_plan'])
    peer=dict(status='PASS',inputs={'RESULT.json':short.INPUTS['initial_parity_result'],
        'TERMINAL.json':short.INPUTS['initial_parity_terminal'],'PLAN.json':short.INPUTS['initial_parity_plan'],
        'TRAIN_FUSION.npz':capture},native_counts=dict(counts),receipt_source_and_passivity_checks=True,
        all_side_channels_bitexact=True,paired_axis_target_fields_bitexact=10,
        phases=['CURRENT_SHARED_INITIAL','TF_ACTIVITY_INITIAL'],rows_per_phase=256,unique_rows=256,
        max_abs_stored_loss_delta=2.384185791015625e-7,
        activity_by_task_timeframe={task:{tf:dict(max_abs_activity=1e-7,max_tolerance_fraction=.1)
            for tf in ('M5','M15','H1','H4','D1')} for task in ('state','future')})
    full=dict(status='COMPLETE',source_unchanged=True,artifacts={'paired_used_prefix':short.INPUTS['full_prefix_capture']},
        candidate_model_state_sha256=short.original_short.MODEL_SHA256,batch_indices=short.PANEL_BATCH_INDICES,
        evaluation_state_unchanged=True)
    if fault=='peer_status':peer['status']='PASS_INDEPENDENT_RETAINED_INITIAL_PARITY'
    if fault=='peer_result_ref':peer['inputs']['RESULT.json']=ref(short.ROOT/'OTHER.json')
    if fault=='peer_loss_tolerance':peer['max_abs_stored_loss_delta']=5.1e-6
    if fault=='peer_activity_tolerance':peer['activity_by_task_timeframe']['state']['D1']['max_tolerance_fraction']=1.01
    if fault=='peer_side_parity':peer['all_side_channels_bitexact']=False
    if fault=='peer_axis_count':peer['paired_axis_target_fields_bitexact']=9
    if fault=='wrong_candidate':proof['phase_states']['TF_ACTIVITY_INITIAL']['model']='9'*64
    if fault=='wrong_imports':counts['head_tensor_imports']=16
    documents={'short_recipe':{'numerical_recipe':original_numerical},'short_result':{},'short_initial_measurement':{},
        'initial_parity_plan':plan,'initial_parity_result':proof,'initial_parity_terminal':terminal,
        'initial_parity_independent':peer,'full_prefix_result':full}
    paths={short.INPUTS[key]['path']:value for key,value in documents.items()}
    monkeypatch.setattr(short.campaign,'require_binding',lambda *a,**kw:None)
    monkeypatch.setattr(short.cache,'_qualify_reversed_origin',lambda *a,**kw:q)
    monkeypatch.setattr(zero,'_bind_ta_view',lambda *a,**kw:view)
    monkeypatch.setattr(short.initial.previous,'_qualify_short',lambda *a:None)
    monkeypatch.setattr(short,'_read',lambda value:paths[value['path']])
    if fault=='none':
        result=short.read_origin(recipe)
        assert result['view']==view and result['initial_proof']==proof
    else:
        with pytest.raises(RuntimeError,match='MEASURED_INITIAL'):
            short.read_origin(recipe)
