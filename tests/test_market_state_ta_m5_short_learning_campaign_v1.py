"""Focused admission/cursor checks for the one c171 short native run."""
from copy import deepcopy

import pytest

from gx1.contracts import market_state_ta_m5_short_learning_v1 as short
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


@pytest.mark.parametrize('mutate',['none','wrong_after_rows','different_observations','wrong_model','missing_steps'])
def test_completed_cursor_validates_actual_measurement_receipts(monkeypatch,mutate):
    value=cursor(True)
    refs={label:ref(short.ROOT/(label+'.json')) for label in ('INITIAL_TRAIN','FINAL_TRAIN')}
    measured={label:dict(measurement=dict(rows=256,batches=16,observation_chain_sha256='d'*64),
        optimizer_updates=updates,panel_batch_indices=short.PANEL_BATCH_INDICES,
        evaluation_state_unchanged=True,test_data_used=False,
        evaluated_model_state_sha256=short.MODEL_SHA256 if updates==0 else 'e'*64)
        for label,updates in (('INITIAL_TRAIN',0),('FINAL_TRAIN',256))}
    counts=dict(model_constructions=1,optimizer_constructions=1,forwards=288,backwards=256,
        optimizer_step_calls=256,optimizer_updates=256,checkpoint_loads=2,strict_full_state_restore_calls=1,
        checkpoint_writes=2,staged_state_load_calls=2,staged_state_load_returns=2)
    report=dict(schema_version=short.RESULT_SCHEMA,outcome='COMPLETE',recipe=value['recipe'],
        parent_cursor=4096,physical_optimizer_updates=256,parent_order_sha256=value['parent_order_sha256'],
        final_state=value['full_state'],canonical_resume_authority=False,test_data_used=False,counts=counts,
        initial_restore_exact=True,evaluation_state_unchanged=True,source_unchanged=True,
        optimizer_initialization='fresh_adamw_empty_state',initial_model_state_sha256=short.MODEL_SHA256,
        final_model_state_sha256='e'*64,panel_batch_indices=short.PANEL_BATCH_INDICES,
        initial_state=ref(short.RUNTIME_ROOT/'private-research/INITIAL_C171_STATE.pt'),
        measurement_refs=refs,measurements={label:dict(v['measurement']) for label,v in measured.items()})
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
