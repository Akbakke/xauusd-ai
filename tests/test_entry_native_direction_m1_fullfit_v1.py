"""Bound M1 full-fit metadata and source-transition tests; synthetic only."""
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from gx1.contracts import entry_native_direction_m1_fullfit_v1 as m1
from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as owner
from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as runner
from gx1.scripts import run_entry_native_direction_research_window_v1 as common


def test_m1_fullfit_reuses_only_the_private_input_adapter():
    for mode in ('price_and_m1_coarsened','price_and_m1'):
        original={'run_scope':owner.M1_PRICE_FIT_SCOPE,'candidate_mode':mode}
        assert runner._fit_input_launch(original,owner)=={**original,'run_scope':owner.M1_PRICE_PRIVATE_ADAPTER_SCOPE}
        assert original['run_scope']==owner.M1_PRICE_FIT_SCOPE
    existing={'run_scope':owner.DENSE_SCOPE}
    assert runner._fit_input_launch(existing,owner) is existing


def test_predeclared_protocol_rejects_changes_in_targets_or_evaluation():
    path=Path(__file__).resolve().parents[1]/'docs/DIRECTION_MATCHED_M1_M5_FULLFIT_PROTOCOL_20261029.json'
    protocol=json.loads(path.read_text());recipe={'fold':protocol['fold'],'dense_target':protocol['corrected_target'],'source_repo':str(path.parents[1])}
    m1.require_protocol(protocol,recipe)
    for field,value in (('primary_horizon_minutes',1),('paired_utc_day_week_bootstrap_replicates',1),('threshold_or_checkpoint_selection',True)):
        wrong=copy.deepcopy(protocol);wrong['evaluation'][field]=value
        with pytest.raises(RuntimeError,match='PROTOCOL'):m1.require_protocol(wrong,recipe)
    wrong=copy.deepcopy(protocol);wrong['test_data_used']=True
    with pytest.raises(RuntimeError,match='PROTOCOL'):m1.require_protocol(wrong,recipe)


@pytest.fixture
def evidence(monkeypatch):
    mode='price_and_m1'; ref=lambda n:{'path':'/'+n,'sha256':n[0]*64}
    qualification=ref('qualification');initial=ref('initial');pair_ref=ref('pair');plan_ref=ref('plan');prior_ref=ref('prior');result_ref=ref('result');receipt_ref=ref('receipt');idle=ref('idle')
    monkeypatch.setattr(m1,'QUALIFICATIONS',{mode:qualification});monkeypatch.setattr(m1,'INITIALS',{mode:initial});monkeypatch.setattr(m1,'PAIR_PARITY',pair_ref)
    monkeypatch.setattr(m1.campaign,'require_binding',lambda ref,**kwargs:ref)
    transitions=[];monkeypatch.setattr(m1,'require_qualified_source_transition',lambda *a:transitions.append(a))
    fold={'fold_id':1};common_fields={'input_manifest':ref('input'),'fold':fold,'dense_target':ref('target'),
        'dense_target_result':ref('targetresult'),'dense_source_transition_proof':ref('sourceproof'),
        'decimal_input_audit':ref('audit'),'decimal_target_refs':{'corrected':ref('target')},
        'candidate_mode':mode,'candidate_m1_refs':{'source':ref('source')},'parent_order_algorithm':'fixed'}
    numerical={'max_updates':2976,'contract_sha256':'x','seed':7}
    old_owners={'immutable':ref('immutable')};owners=dict(old_owners)
    prior={**common_fields,'run_scope':'private_m1','qualification_weights_for_main_fit':False,
        'numerical_recipe':{'max_updates':2,'contract_sha256':'y','seed':7},'source_owners':old_owners}
    recipe={**common_fields,'native_dense_qualification':qualification,'decimal_initial_predictions':initial,
        'm1_pair_initial_parity':pair_ref,'numerical_recipe':numerical}
    review={'schema_version':'gx1_m1_price_private_root_review_v1','status':'PASS_CONSUMED_ORDERED_PRIVATE_NATIVE_MECHANICS',
        'all_sixty_train_priors_checked':True,'all_sixty_heads_have_gradient_and_eligible_labels_on_fixed_batches':True,
        'm1_value_weight_gradient_nonzero_both_second_steps':True,'four_price_features_shared_projection_gradients_nonzero_in_both_trajectories':True,
        'bit_exact_resume_reported_and_bound':True,'private_optimizer_updates':4,'qualification_weights_for_main_fit':False,'test_data_used':False,
        'plan':plan_ref,'source_commit':'a'*40,'result':result_ref,'receipt':receipt_ref,'idle_restore':idle}
    pair={'status':'PASS_MATCHED_INITIAL_PREDICTIONS','test_data_used':False,'fixed_32_train_parents_identical':True,
        'all_60_train_priors_identical':True,'comparison':{k:{'byte_identical':True} for k in ('parent_rows','bar_start_ns','decision_ns','logit_up')}}
    plan={'source_commit':'a'*40,'recipe':prior_ref,'plan_sha256':'z'}
    receipt={'plan_sha256':'z','outcome':'COMPLETE','guard_decision':'PASS','test_data_used':False}
    result={'schema_version':'gx1_native_m1_price_private_qualification_result_v1','decision':'PASS_M1_PRICE_PRIVATE_MECHANICS',
        'recipe':prior_ref,'candidate_mode':mode,'candidate_m1_refs':recipe['candidate_m1_refs'],'initial_predictions':initial,
        'decimal_prior_against_sign_audit':True,'bit_exact_resume':True,'physical_optimizer_updates':4,
        'source_unchanged':True,'test_data_used':False,'private_weights_for_main_fit':False}
    objects={qualification['path']:review,pair_ref['path']:pair,plan_ref['path']:plan,prior_ref['path']:prior,result_ref['path']:result,receipt_ref['path']:receipt}
    monkeypatch.setattr(m1,'_read',lambda ref,label:objects[ref['path']])
    return locals()


def test_qualified_arm_requires_bound_private_proof_and_current_sources(evidence):
    f=evidence; m1.require_qualification(f['recipe'],f['owners'],source_repo=Path('/repo'),private_scope='private_m1')
    assert len(f['transitions'])==1
    for obj,key,value in ((f['review'],'bit_exact_resume_reported_and_bound',False),
        (f['review'],'m1_value_weight_gradient_nonzero_both_second_steps',False),
        (f['receipt'],'outcome','FAILED'),(f['result'],'private_weights_for_main_fit',True),
        (f['prior'],'candidate_mode','price_and_m1_coarsened')):
        old=obj[key];obj[key]=value
        with pytest.raises(RuntimeError):m1.require_qualification(f['recipe'],f['owners'],source_repo=Path('/repo'),private_scope='private_m1')
        obj[key]=old
    f['pair']['comparison']['logit_up']['byte_identical']=False
    with pytest.raises(RuntimeError,match='PAIR_INITIAL_PARITY'):m1.require_qualification(f['recipe'],f['owners'],source_repo=Path('/repo'),private_scope='private_m1')
    f['pair']['comparison']['logit_up']['byte_identical']=True
    f['owners']['immutable']={**f['owners']['immutable'],'sha256':'0'*64}
    with pytest.raises(RuntimeError,match='QUALIFIED_UNCHANGED_SOURCES'):m1.require_qualification(f['recipe'],f['owners'],source_repo=Path('/repo'),private_scope='private_m1')


def test_source_transition_rejects_changes_to_unmodified_computation(tmp_path,monkeypatch):
    old={};new={};before={}
    for path,rules in m1.TRANSITIONS.items():
        file=tmp_path/path;file.parent.mkdir(parents=True,exist_ok=True)
        original=('VALUE = 1\n\ndef math_step(x):\n    return x + 1\n'+''.join('\ndef '+name+'():\n    pass\n' for name in rules['changed'])).encode()
        current=original+''.join('\n'+name+' = 3\n' for name in rules['new_assignments']).encode()
        file.write_bytes(current);before[path]=original
        old[path]={'path':str(file),'sha256':hashlib.sha256(original).hexdigest()};new[path]={'path':str(file),'sha256':hashlib.sha256(current).hexdigest()}
    monkeypatch.setattr(subprocess,'check_output',lambda args:before[args[-1].split(':',1)[1]])
    m1.require_qualified_source_transition(tmp_path,'a'*40,old,new)
    name=next(iter(m1.TRANSITIONS));file=tmp_path/name;changed=file.read_bytes().replace(b'return x + 1',b'return x + 2');file.write_bytes(changed);new[name]['sha256']=hashlib.sha256(changed).hexdigest()
    with pytest.raises(RuntimeError,match='TRANSITION_AST'):m1.require_qualified_source_transition(tmp_path,'a'*40,old,new)


def test_candidate_initial_read_guard_needs_the_exact_bound_prediction(monkeypatch,tmp_path):
    initial=m1.INITIALS['price_and_m1'];refs={k:{'path':f'/outside/{k}.'+('npz' if k=='source' else 'json'),'sha256':'a'*64} for k in ('source','roll','normalization')}
    context={'ancestor_plan':{'payloads':{},'native_cache_dir':'/cache','gc_surfaces':{'path':'/gc/surfaces.json'}},
        'info':{'metadata':{},'audit':{'manifest_path':'/forbidden'}},'candidate_m1_refs':refs,'decimal_initial_predictions':initial}
    hooks=[];monkeypatch.setattr(common.sys,'addaudithook',hooks.append)
    common._install_read_guard(context,tmp_path/'output',tmp_path/'runtime',candidate_m1_refs=refs,candidate_initial_predictions=initial)
    assert context['candidate_m1_guard_installed'] is True and len(hooks)==1
    hooks[0]('open',(initial['path'],'r',0))
    with pytest.raises(RuntimeError,match='ORIGINAL_WRITE'):hooks[0]('open',(initial['path'],'w',1))
    wrong={**initial,'sha256':'0'*64}
    with pytest.raises(RuntimeError,match='CANDIDATE_INITIAL_BINDING'):
        common._install_read_guard(context,tmp_path/'output',tmp_path/'runtime',candidate_m1_refs=refs,candidate_initial_predictions=wrong)
