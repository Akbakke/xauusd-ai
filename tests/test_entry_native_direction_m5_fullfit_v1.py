"""M5-only full-fit proof, transition and input guard; synthetic fixtures only."""
import copy
import hashlib
import json
import subprocess
from pathlib import Path
import pytest
from gx1.contracts import entry_native_direction_m5_fullfit_v1 as m5
from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as owner
from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as runner
from gx1.scripts import run_entry_native_direction_research_window_v1 as common

def test_m5_fit_maps_to_qualified_price_only_adapter():
    original={'run_scope':owner.M5_PRICE_FIT_SCOPE,'candidate_mode':'price_only'}
    assert runner._fit_input_launch(original,owner)=={**original,'run_scope':owner.M1_PRICE_PRIVATE_ADAPTER_SCOPE}
    assert original['run_scope']==owner.M5_PRICE_FIT_SCOPE

def test_protocol_rejects_m1_wrong_targets_baselines_and_evaluation(monkeypatch):
    path=Path(__file__).resolve().parents[1]/'docs/DIRECTION_M5_PRICE_ONLY_PROTOCOL_20261029.json'
    protocol=json.loads(path.read_text());recipe={'fold':protocol['fold'],'dense_target':protocol['target'],'source_repo':str(path.parents[1])}
    monkeypatch.setattr(m5.campaign,'require_binding',lambda ref,**kwargs:ref)
    m5.require_protocol(protocol,recipe)
    for section,key,value in ((None,'candidate_input_mode','price_and_m1'),(None,'m1_values_or_masks_as_entry_inputs',True),
            (None,'test_data_used',True),('fit','seed',1),('comparison','primary_minutes',1),
            ('comparison','threshold_or_checkpoint_selection',True),('hardware','gpu_power_limit_w_during_and_after',160),
            ('baseline','predictions',{'path':'/unbound','sha256':'0'*64})):
        wrong=copy.deepcopy(protocol);(wrong if section is None else wrong[section])[key]=value
        with pytest.raises(RuntimeError):m5.require_protocol(wrong,recipe)
    wrong=copy.deepcopy(protocol);wrong['comparison_sources'].pop(next(iter(wrong['comparison_sources'])))
    with pytest.raises(RuntimeError,match='COMPARISON_SOURCES'):m5.require_protocol(wrong,recipe)

@pytest.fixture
def evidence(monkeypatch):
    ref=lambda n:{'path':'/'+n,'sha256':n[0]*64}
    qualification=ref('qualification');initial=ref('initial');plan_ref=ref('plan');prior_ref=ref('prior');result_ref=ref('result');receipt_ref=ref('receipt');power_ref=ref('power')
    monkeypatch.setattr(m5,'QUALIFICATION',qualification);monkeypatch.setattr(m5,'INITIAL',initial)
    monkeypatch.setattr(m5.campaign,'require_binding',lambda ref,**kwargs:ref)
    transitions=[];monkeypatch.setattr(m5,'require_qualified_source_transition',lambda *a:transitions.append(a))
    common_fields={'input_manifest':ref('input'),'fold':{'fold_id':1},'dense_target':ref('target'),
        'dense_target_result':ref('targetresult'),'dense_source_transition_proof':ref('sourceproof'),
        'decimal_input_audit':ref('audit'),'decimal_target_refs':{'corrected':ref('target')},
        'candidate_mode':'price_only','parent_order_algorithm':'fixed'}
    old_owners={'immutable':ref('immutable')};owners=copy.deepcopy(old_owners)
    prior={**common_fields,'run_scope':'private_m5','qualification_weights_for_main_fit':False,
        'numerical_recipe':{'max_updates':2,'contract_sha256':'y','seed':7},'source_owners':old_owners}
    recipe={**common_fields,'native_dense_qualification':qualification,'decimal_initial_predictions':initial,
        'numerical_recipe':{'max_updates':2976,'contract_sha256':'x','seed':7}}
    review={'schema_version':'gx1_m5_price_private_root_review_v1','status':'PASS_CONSUMED_PRICE_ONLY_PRIVATE_NATIVE_MECHANICS',
        'candidate_mode':'price_only','m1_values_and_masks_as_entry_inputs':False,'initial_predictions':initial,'initial_parent_count':256,
        'all_sixty_train_priors_checked':True,'all_sixty_heads_have_gradient_and_eligible_labels_on_fixed_batches':True,
        'four_price_features_shared_projection_gradients_nonzero_in_both_trajectories':True,'bit_exact_resume_reported_and_bound':True,
        'private_optimizer_updates':4,'qualification_weights_for_main_fit':False,'test_data_used':False,
        'plan':plan_ref,'source_commit':'a'*40,'result':result_ref,'receipt':receipt_ref,'power_after':power_ref}
    plan={'source_commit':'a'*40,'recipe':prior_ref,'plan_sha256':'z'}
    receipt={'plan_sha256':'z','outcome':'COMPLETE','guard_decision':'PASS','trainer_guard_exit_code':0,'progress_observer_exit_code':0,'test_data_used':False}
    result={'schema_version':'gx1_native_m1_price_private_qualification_result_v1','decision':'PASS_M1_PRICE_PRIVATE_MECHANICS',
        'recipe':prior_ref,'candidate_mode':'price_only','candidate_m1_refs':None,'initial_predictions':initial,
        'decimal_prior_against_sign_audit':True,'bit_exact_resume':True,'physical_optimizer_updates':4,
        'source_unchanged':True,'test_data_used':False,'private_weights_for_main_fit':False}
    power={'campaign_task_state':'Disabled','configured_power_limit_w':250,'actual_power_limit_w':250,'keeper_state':'Running','bridge_state':'Running'}
    objects={qualification['path']:review,plan_ref['path']:plan,prior_ref['path']:prior,result_ref['path']:result,receipt_ref['path']:receipt,power_ref['path']:power}
    monkeypatch.setattr(m5,'_read',lambda ref,label:objects[ref['path']])
    return locals()

def test_qualification_rejects_mismatched_inputs_result_and_source(evidence):
    f=evidence;call=lambda:m5.require_qualification(f['recipe'],f['owners'],source_repo=Path('/repo'),private_scope='private_m5')
    call();assert len(f['transitions'])==1
    for obj,key,value in ((f['review'],'m1_values_and_masks_as_entry_inputs',True),
        (f['review'],'four_price_features_shared_projection_gradients_nonzero_in_both_trajectories',False),
        (f['review'],'bit_exact_resume_reported_and_bound',False),(f['result'],'candidate_m1_refs',{}),
        (f['result'],'private_weights_for_main_fit',True),(f['receipt'],'outcome','FAILED'),
        (f['receipt'],'trainer_guard_exit_code',1),(f['prior'],'candidate_mode','price_and_m1'),
        (f['power'],'configured_power_limit_w',160)):
        previous=obj[key];obj[key]=value
        with pytest.raises(RuntimeError):call()
        obj[key]=previous
    f['recipe']['candidate_m1_refs']={}
    with pytest.raises(RuntimeError,match='QUALIFICATION_BINDING'):call()
    del f['recipe']['candidate_m1_refs']
    f['recipe']['decimal_initial_predictions']={**f['initial'],'sha256':'0'*64}
    with pytest.raises(RuntimeError,match='QUALIFICATION_BINDING'):call()
    f['recipe']['decimal_initial_predictions']=f['initial']
    f['owners']['immutable']={**f['owners']['immutable'],'sha256':'0'*64}
    with pytest.raises(RuntimeError,match='QUALIFIED_UNCHANGED_SOURCES'):call()

def test_source_transition_rejects_changed_computation(tmp_path,monkeypatch):
    old={};new={};before={}
    for path,rules in m5.TRANSITIONS.items():
        file=tmp_path/path;file.parent.mkdir(parents=True,exist_ok=True)
        original=('VALUE = 1\n\ndef math_step(x):\n    return x + 1\n'+''.join('\ndef '+name+'():\n    pass\n' for name in rules['changed'])).encode()
        current=original+''.join('\n'+name+' = 3\n' for name in rules['new_assignments']).encode()
        file.write_bytes(current);before[path]=original
        old[path]={'path':str(file),'sha256':hashlib.sha256(original).hexdigest()};new[path]={'path':str(file),'sha256':hashlib.sha256(current).hexdigest()}
    monkeypatch.setattr(subprocess,'check_output',lambda args:before[args[-1].split(':',1)[1]])
    m5.require_qualified_source_transition(tmp_path,'a'*40,old,new)
    name=next(iter(m5.TRANSITIONS));file=tmp_path/name;changed=file.read_bytes().replace(b'return x + 1',b'return x + 2');file.write_bytes(changed);new[name]['sha256']=hashlib.sha256(changed).hexdigest()
    with pytest.raises(RuntimeError,match='TRANSITION_AST'):m5.require_qualified_source_transition(tmp_path,'a'*40,old,new)

def test_read_guard_allows_only_m5_own_baseline_without_m1(monkeypatch,tmp_path):
    initial=m5.INITIAL
    context={'ancestor_plan':{'payloads':{},'native_cache_dir':'/cache','gc_surfaces':{'path':'/gc/surfaces.json'}},
        'info':{'metadata':{},'audit':{'manifest_path':'/forbidden'}},'candidate_inputs':{'mode':'price_only'},'decimal_initial_predictions':initial}
    hooks=[];monkeypatch.setattr(common.sys,'addaudithook',hooks.append)
    common._install_read_guard(context,tmp_path/'output',tmp_path/'runtime',candidate_initial_predictions=initial)
    assert len(hooks)==1;hooks[0]('open',(initial['path'],'r',0))
    with pytest.raises(RuntimeError,match='ORIGINAL_WRITE'):hooks[0]('open',(initial['path'],'w',1))
    with pytest.raises(RuntimeError,match='CANDIDATE_INITIAL_BINDING'):
        common._install_read_guard(context,tmp_path/'output',tmp_path/'runtime',candidate_initial_predictions={**initial,'sha256':'0'*64})
    with pytest.raises(RuntimeError,match='CANDIDATE_INITIAL_BINDING'):
        common._install_read_guard(context,tmp_path/'output',tmp_path/'runtime',candidate_initial_predictions=initial,candidate_m1_refs={})
