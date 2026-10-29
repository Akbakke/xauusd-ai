"""Synthetic checks for the separately qualified five-priority M5 fit."""
import copy
import hashlib
import json
import subprocess
from pathlib import Path
import pytest
from gx1.contracts import entry_native_direction_five_priority_fullfit_v1 as owner
from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as runner


def test_priority_fit_uses_the_price_adapter_without_m1():
    original={'run_scope':prefix.M5_FIVE_PRIORITY_FIT_SCOPE,'candidate_mode':'price_only'}
    assert runner._fit_input_launch(original,prefix)=={**original,'run_scope':prefix.M1_PRICE_PRIVATE_ADAPTER_SCOPE}
    assert 'candidate_m1_refs' not in original


def test_protocol_rejects_unsafe_or_non_matched_experiments(monkeypatch):
    design=json.loads((Path(__file__).resolve().parents[1]/'docs/DIRECTION_M5_FIVE_PRIORITY_DESIGN_20261029.json').read_text())
    recipe={'dense_evaluation_protocol':owner.private.DESIGN,'priority_baseline_initial_report':owner.private.BASELINE_INITIAL_REPORT}
    monkeypatch.setattr(owner.campaign,'require_binding',lambda ref,**kwargs:ref)
    owner.require_protocol(design,recipe)
    for section,key,value in [('candidate','five_minute_weight',11),('candidate','other_horizon_weight',2),
            ('candidate','preserve_model_inputs_targets_and_seed',False),('fit','train_parent_rows',47601),
            ('fit','physical_restart_before_each_training_window',False),('fit','gpu_limit_during_and_after_w',160),
            ('fit','qualification_weights_reused',True),('comparison','report_all_horizons',False),
            ('comparison','no_horizon_or_parameter_search',False)]:
        wrong=copy.deepcopy(design);wrong[section][key]=value
        with pytest.raises(RuntimeError):owner.require_protocol(wrong,recipe)
    wrong=copy.deepcopy(design);wrong['test_data_used']=True
    with pytest.raises(RuntimeError):owner.require_protocol(wrong,recipe)


@pytest.fixture
def evidence(monkeypatch):
    ref=lambda n:{'path':'/'+n,'sha256':n[0]*64}
    qualification=ref('qualification');initial=ref('initial');pinitial=ref('predictions')
    plan_ref=ref('plan');prior_ref=ref('prior');result_ref=ref('result');receipt_ref=ref('receipt');power_ref=ref('power')
    monkeypatch.setattr(owner,'QUALIFICATION',qualification);monkeypatch.setattr(owner,'PRIVATE_INITIAL',pinitial)
    monkeypatch.setattr(owner.m5,'INITIAL',initial)
    monkeypatch.setattr(owner.campaign,'require_binding',lambda ref,**kwargs:ref)
    calls=[];monkeypatch.setattr(owner,'require_source_transition',lambda *a:calls.append(a))
    common={'input_manifest':ref('input'),'fold':{'fold_id':1},'dense_target':ref('target'),
        'dense_target_result':ref('targetresult'),'dense_source_transition_proof':ref('sourceproof'),
        'decimal_input_audit':ref('audit'),'decimal_target_refs':{'corrected':ref('target')},
        'candidate_mode':'price_only','parent_order_algorithm':'fixed','gc_layout_metadata_parity':ref('gc')}
    numerical={'seed':7,'max_updates':2976,'contract_sha256':'x','research_objective_sha256':'new','five_minute_priority':True}
    baseline={**copy.deepcopy(common),'numerical_recipe':{'seed':7,'max_updates':2976,'contract_sha256':'y','research_objective_sha256':'old'}}
    prior={**copy.deepcopy(common),'run_scope':'private_m5','qualification_weights_for_main_fit':False,
        'priority_design':owner.private.DESIGN,'priority_baseline_initial_report':owner.private.BASELINE_INITIAL_REPORT,
        'numerical_recipe':{**numerical,'max_updates':2},'source_owners':{}}
    recipe={**copy.deepcopy(common),'native_dense_qualification':qualification,'decimal_initial_predictions':initial,'numerical_recipe':numerical}
    required=['initial_model_and_buffers_exactly_matched_previous_m5_fit','initial_predictions_256_exactly_matched_previous_m5_fit',
        'all_sixty_heads_have_eligible_labels_and_gradient_evidence','four_price_projection_columns_nonzero_gradient_on_each_step',
        'bit_exact_resume','campaign_task_disabled','source_unchanged']
    review={'schema_version':'gx1_m5_five_priority_private_root_review_v1','status':'PASS_CONSUMED_PRICE_ONLY_FIVE_PRIORITY_PRIVATE_MECHANICS',
        'candidate_mode':'price_only','m1_values_and_masks_as_entry_inputs':False,**{k:True for k in required},
        'private_optimizer_updates':4,'unique_train_parents':32,'configured_gpu_limit_after_w':250,
        'qualification_weights_for_main_fit':False,'test_data_used':False,'plan':plan_ref,'source_commit':'a'*40,
        'result':result_ref,'receipt':receipt_ref,'power_after':power_ref}
    plan={'source_commit':'a'*40,'recipe':prior_ref,'plan_sha256':'z'}
    receipt={'plan_sha256':'z','outcome':'COMPLETE','guard_decision':'PASS','trainer_guard_exit_code':0,'progress_observer_exit_code':0,'test_data_used':False}
    result={'schema_version':'gx1_native_m1_price_private_qualification_result_v1','decision':'PASS_M1_PRICE_PRIVATE_MECHANICS',
        'recipe':prior_ref,'candidate_mode':'price_only','candidate_m1_refs':None,'initial_predictions':pinitial,
        'priority_design':owner.private.DESIGN,'priority_baseline_initial_report':owner.private.BASELINE_INITIAL_REPORT,
        'priority_initial_prediction_and_model_buffer_byte_parity':True,'decimal_prior_against_sign_audit':True,'bit_exact_resume':True,
        'physical_optimizer_updates':4,'source_unchanged':True,'test_data_used':False,'private_weights_for_main_fit':False}
    power={'campaign_task_state':'Disabled','configured_power_limit_w':250,'gpu':'GPU-ID, 250.00, 23.28, 37, 0, 78',
        'keeper_state':'Running','bridge_state':'Running','test_data_used':False}
    initial_report={'role':'INITIAL_TRAIN','test_data_used':False,'fold':common['fold'],
        'predictions':{'sha256':initial['sha256']},'preservation':{'model':'model-sha','all_buffers':'buffer-sha'}}
    docs={qualification['path']:review,plan_ref['path']:plan,prior_ref['path']:prior,result_ref['path']:result,
        receipt_ref['path']:receipt,power_ref['path']:power,owner.private.BASELINE_RECIPE['path']:baseline,
        owner.private.BASELINE_INITIAL_REPORT['path']:initial_report}
    monkeypatch.setattr(owner,'_read',lambda ref,label:docs[ref['path']])
    return locals()


def test_private_qualification_rejects_mismatches(evidence):
    f=evidence
    call=lambda:owner.require_qualification(f['recipe'],{},source_repo=Path('/repo'),source_commit='b'*40,private_scope='private_m5')
    call();assert len(f['calls'])==1
    checks=[(f['review'],'bit_exact_resume',False),(f['review'],'m1_values_and_masks_as_entry_inputs',True),
        (f['result'],'priority_initial_prediction_and_model_buffer_byte_parity',False),(f['receipt'],'outcome','FAILED'),
        (f['power'],'gpu','GPU-ID, 160.00, 23.28, 37, 0, 78'),(f['prior'],'parent_order_algorithm','other'),
        (f['baseline']['numerical_recipe'],'seed',8),(f['prior']['numerical_recipe'],'seed',9)]
    for obj,key,value in checks:
        previous=obj[key];obj[key]=value
        with pytest.raises(RuntimeError):call()
        obj[key]=previous
    f['recipe']['candidate_m1_refs']={}
    with pytest.raises(RuntimeError):call()


def test_source_transition_rejects_unrelated_computation(tmp_path,monkeypatch):
    old={};new={};before={}
    for name,rules in owner.TRANSITIONS.items():
        file=tmp_path/name;file.parent.mkdir(parents=True,exist_ok=True)
        original=('VALUE = 1\n\ndef stable(x):\n    return x + 1\n'+''.join('\ndef '+n+'():\n    pass\n' for n in rules['functions'])).encode()
        current=original+''.join('\n'+n+' = 3\n' for n in rules.get('new_assignments',())).encode()
        file.write_bytes(current);before[name]=original
        old[name]={'path':str(file),'sha256':hashlib.sha256(original).hexdigest()};new[name]={'path':str(file),'sha256':hashlib.sha256(current).hexdigest()}
    new[owner.OWN_SOURCE]={'path':str(tmp_path/owner.OWN_SOURCE),'sha256':'a'*64}
    monkeypatch.setattr(subprocess,'check_output',lambda args,**kwargs:before[args[-1].split(':',1)[1]])
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:type('Result',(),{'returncode':0})())
    owner.require_source_transition(tmp_path,'b'*40,'a'*40,old,new)
    name=next(iter(owner.TRANSITIONS));file=tmp_path/name
    changed=file.read_bytes().replace(b'return x + 1',b'return x + 2');file.write_bytes(changed)
    new[name]['sha256']=hashlib.sha256(changed).hexdigest()
    with pytest.raises(RuntimeError,match='SOURCE_AST'):owner.require_source_transition(tmp_path,'b'*40,'a'*40,old,new)
