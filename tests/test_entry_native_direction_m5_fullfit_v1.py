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


def test_factor_protocol_precommits_three_separate_comparisons(monkeypatch):
    protocol=json.loads((Path(__file__).resolve().parents[1]/'docs/DIRECTION_M5_SIGN_BPS_EXPOSURE_PROTOCOL_20261030.json').read_text())
    monkeypatch.setattr(m5.campaign,'require_binding',lambda ref,**kwargs:ref)
    recipe={'fold':protocol['fold'],'dense_target':protocol['target'],'source_repo':str(Path(__file__).resolve().parents[1]),
        'run_scope':owner.M5_FACTOR_BPS_SCOPE,'runtime_root':m5.FACTOR_RUNTIME_ROOTS['bps_one_pass']}
    m5.require_factor_protocol(protocol,recipe)
    for section,key,value in ((None,'candidate_input_mode','price_and_m1'),(None,'m1_values_or_masks_as_entry_inputs',True),
            (None,'test_data_used',True),('fit','seed',4),('fit','same_source_and_parent_order_required',False),
            ('comparison','primary_minutes',6),('comparison','threshold_or_checkpoint_selection',True),
            ('hardware','gpu_power_limit_w_during_and_after',160),('arms','sign_two_pass',{'passes':3})):
        wrong=copy.deepcopy(protocol);(wrong if section is None else wrong[section])[key]=value
        with pytest.raises(RuntimeError):m5.require_factor_protocol(wrong,recipe)
    wrong=copy.deepcopy(recipe);wrong['runtime_root']=m5.FACTOR_RUNTIME_ROOTS['sign_one_pass']
    with pytest.raises(RuntimeError,match='FACTOR_PROTOCOL_IDENTITY'):m5.require_factor_protocol(protocol,wrong)


def test_factor_power_scope_is_250_only_for_exact_matched_recipe():
    for scope,bps in ((owner.M5_FACTOR_SIGN_SCOPE,False),(owner.M5_FACTOR_BPS_SCOPE,True),(owner.M5_FACTOR_CONTINUATION_SCOPE,False)):
        recipe={'run_scope':scope,'candidate_mode':'price_only','numerical_recipe':{'dense_signed_bps_supervision':bps}}
        assert owner.guarded_prefix_power_limit(recipe)==250
        recipe['numerical_recipe']['dense_signed_bps_supervision']=not bps
        with pytest.raises(RuntimeError,match='FACTOR_POWER_SCOPE'):owner.guarded_prefix_power_limit(recipe)
    assert owner.guarded_prefix_power_limit({'run_scope':owner.M5_PRICE_FIT_SCOPE})==300


def test_factor_input_adapter_preserves_scope_data_without_m1():
    for scope in (owner.M5_FACTOR_SIGN_SCOPE,owner.M5_FACTOR_BPS_SCOPE,owner.M5_FACTOR_CONTINUATION_SCOPE):
        recipe={'run_scope':scope,'candidate_mode':'price_only'}
        assert runner._fit_input_launch(recipe,owner)=={**recipe,'run_scope':owner.M1_PRICE_PRIVATE_ADAPTER_SCOPE}
        assert recipe['run_scope']==scope


@pytest.fixture
def factor_evidence(monkeypatch):
    from gx1.contracts import entry_native_direction_research_campaign_v1 as private
    from gx1.contracts.entry_model_native_train_recipe_v1 import native_direction_research_recipe
    ref=lambda name:{'path':'/'+name,'sha256':name[0]*64}
    review_ref=ref('review');plan_ref=ref('plan');prior_ref=ref('prior');result_ref=ref('result');receipt_ref=ref('receipt');power_ref=ref('power')
    monkeypatch.setattr(m5,'FACTOR_QUALIFICATION',review_ref)
    monkeypatch.setattr(m5.campaign,'require_binding',lambda ref,**kwargs:ref)
    transitions=[];monkeypatch.setattr(m5,'require_factor_source_transition',lambda *args:transitions.append(args))
    common_fields={'input_manifest':ref('input'),'fold':{'fold_id':1},'dense_target':ref('target'),
        'dense_target_result':ref('targetresult'),'dense_source_transition_proof':ref('sourceproof'),
        'decimal_input_audit':ref('audit'),'decimal_target_refs':{'corrected':ref('target')},
        'candidate_mode':'price_only','parent_order_algorithm':'fixed'}
    old_owners={'immutable':ref('immutable')};owners=copy.deepcopy(old_owners)
    prior={**common_fields,'run_scope':owner.M5_DENSE_BPS_PRIVATE_SCOPE,'source_owners':old_owners,
        'numerical_recipe':native_direction_research_recipe(batch_size=16,max_updates=2,dense_minutes=True,dense_bps=True)}
    recipe={**common_fields,'run_scope':owner.M5_FACTOR_BPS_SCOPE,'native_dense_qualification':review_ref,
        'decimal_initial_predictions':private.M5_DENSE_BPS_BASELINE_INITIAL,'factor_initial_report':private.M5_DENSE_BPS_BASELINE_REPORT,
        'dense_bps_scale':private.M5_DENSE_BPS_SCALE,
        'numerical_recipe':native_direction_research_recipe(batch_size=16,max_updates=2976,dense_minutes=True,dense_bps=True)}
    review={'status':'PASS_VERIFIED_PRIVATE_MECHANICS_NOT_EDGE','physical_boot_id':583,'unique_train_parents':32,
        'physical_optimizer_updates':4,'initial_predictions':256,'initial_shared_state_and_rng_parity':True,
        'signed_bps_gradient_rows':60,'bps_only_joint_projection_nonzero':65536,'bit_exact_resume':True,
        'm1_entry_input':False,'private_weights_for_main_fit':False,'test_data_used':False,'source_commit':'a'*40,
        'plan':plan_ref,'result':result_ref,'receipt':receipt_ref,'power_after':power_ref}
    plan={'source_commit':'a'*40,'recipe':prior_ref,'plan_sha256':'hash'}
    receipt={'plan_sha256':'hash','outcome':'COMPLETE','guard_decision':'PASS','trainer_guard_exit_code':0,
        'progress_observer_exit_code':0,'test_data_used':False}
    result={'recipe':prior_ref,'decision':'PASS_M5_DENSE_BPS_PRIVATE_MECHANICS',
        'dense_bps_initial_prediction_and_shared_state_parity':True,'bit_exact_resume':True,'source_unchanged':True,
        'private_weights_for_main_fit':False,'test_data_used':False}
    power={'task_state':'Disabled','configured_power_limit_w':250}
    objects={review_ref['path']:review,plan_ref['path']:plan,prior_ref['path']:prior,result_ref['path']:result,
        receipt_ref['path']:receipt,power_ref['path']:power}
    monkeypatch.setattr(m5,'_read',lambda ref,label:objects[ref['path']])
    return locals()


def test_factor_qualification_binds_inputs_loss_and_real_private_evidence(factor_evidence):
    f=factor_evidence;call=lambda:m5.require_factor_qualification(f['recipe'],f['owners'],source_repo=Path('/repo'),source_commit='b'*40)
    call();assert len(f['transitions'])==1
    for obj,key,value in ((f['review'],'signed_bps_gradient_rows',59),(f['review'],'bps_only_joint_projection_nonzero',0),
        (f['review'],'initial_shared_state_and_rng_parity',False),(f['result'],'bit_exact_resume',False),
        (f['receipt'],'guard_decision','FAILED'),(f['power'],'configured_power_limit_w',160),
        (f['prior'],'parent_order_algorithm','different')):
        previous=obj[key];obj[key]=value
        with pytest.raises(RuntimeError):call()
        obj[key]=previous
    f['recipe']['candidate_m1_refs']={}
    with pytest.raises(RuntimeError,match='FACTOR_QUALIFICATION_BINDING'):call()
    del f['recipe']['candidate_m1_refs']
    f['recipe']['numerical_recipe']['optimizer']['kwargs']['lr']=0.01
    with pytest.raises(RuntimeError,match='FACTOR_CANONICAL_NUMERICS'):call()


def test_factor_second_pass_guard_is_exact_readonly(monkeypatch,tmp_path):
    base=m5.FACTOR_RUNTIME_ROOTS['sign_one_pass']+'/prefix-fit/'
    paths=(base+'states/terminal.pt',base+'states/initial.pt',base+'window-0001/FINAL_TRAIN.npz')
    context={'ancestor_plan':{'payloads':{},'native_cache_dir':'/cache','gc_surfaces':{'path':'/gc/surfaces.json'}},
        'info':{'metadata':{},'audit':{'manifest_path':'/forbidden'}},'factor_continuation_reads':paths}
    hooks=[];monkeypatch.setattr(common.sys,'addaudithook',hooks.append)
    common._install_read_guard(context,tmp_path/'output',tmp_path/'runtime',extra_read_paths=paths)
    for path in paths:
        hooks[0]('open',(path,'r',0))
        with pytest.raises(RuntimeError,match='ORIGINAL_WRITE'):hooks[0]('open',(path,'w',1))
    with pytest.raises(RuntimeError,match='FACTOR_EXTRA_READ_PATHS'):
        common._install_read_guard(context,tmp_path/'output',tmp_path/'runtime',extra_read_paths=paths[:2])
    context['factor_continuation_reads']=('/home/andre2/GX1_RUNS/unrelated',*paths[1:])
    with pytest.raises(RuntimeError,match='FACTOR_EXTRA_READ_PATHS'):
        common._install_read_guard(context,tmp_path/'output',tmp_path/'runtime',extra_read_paths=context['factor_continuation_reads'])


def test_factor_continuation_requires_original_terminal_and_identical_code_owners(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from gx1.contracts.entry_model_native_train_recipe_v1 import native_direction_research_recipe
    ref=lambda name:{'path':'/'+name,'sha256':name[0]*64}
    root=m5.FACTOR_RUNTIME_ROOTS['sign_one_pass'].removesuffix('/RUNTIME')
    review_ref={'path':root+'/ACTUAL_FIT_REVIEW_001.json','sha256':'a'*64}
    names=('launch_recipe','terminal_state','initial_state','initial_measurement','terminal_train_measurement','terminal_train_predictions')
    origin={'review':review_ref,**{name:ref(name) for name in names}}
    owners={'path':ref('source')};fold={'prefix_rows':47602};initial_ref=ref('first_initial');target=ref('target');manifest=ref('manifest');protocol=ref('protocol')
    numerical=native_direction_research_recipe(batch_size=16,max_updates=2976,dense_minutes=True)
    numerical2=native_direction_research_recipe(batch_size=16,max_updates=5952,dense_minutes=True)
    recipe={'continuation_origin':origin,'source_repo':'/repo','input_manifest':manifest,'fold':fold,
        'dense_target':target,'dense_evaluation_protocol':protocol,'factor_initial_report':initial_ref,'numerical_recipe':numerical2}
    old={**{k:recipe[k] for k in ('input_manifest','fold','dense_target','dense_evaluation_protocol','factor_initial_report')},
        'run_scope':owner.M5_FACTOR_SIGN_SCOPE,'source_owners':owners,'numerical_recipe':numerical}
    plan_ref=ref('plan');result_ref=ref('result');receipt_ref=ref('receipt');commit='a'*40
    review={'status':'PASS_CONSUMED_M5_FACTOR_SIGN_ONEPASS','source_commit':commit,'recipe':origin['launch_recipe'],
        'training_source_unchanged':True,'optimizer_updates':2976,'train_parents':47602,'signed_guard_pass':True,
        'test_data_used':False,'private_weights_reused':False,'plan':plan_ref,'result':result_ref,'receipt':receipt_ref}
    plan={'source_commit':commit,'recipe':origin['launch_recipe'],'plan_sha256':'bound'}
    receipt={'plan_sha256':'bound','outcome':'COMPLETE','guard_decision':'PASS','trainer_guard_exit_code':0,
        'progress_observer_exit_code':0,'test_data_used':False}
    result={'recipe':origin['launch_recipe'],'source_unchanged':True,'test_data_used':False,'qualification_weights_reused':False,
        'outcome':'COMPLETE','parent_cursor':47602,'optimizer_updates':2976,'full_state':origin['terminal_state'],
        'own_initial_state':origin['initial_state'],'initial_measurement':origin['initial_measurement'],
        'final_train_measurement':origin['terminal_train_measurement']}
    initial={'role':'INITIAL_TRAIN','own_initial_state':origin['initial_state'],'fold':fold,'test_data_used':False}
    terminal={'role':'FINAL_TRAIN','own_initial_state':origin['initial_state'],'fold':fold,'test_data_used':False,
        'predictions':origin['terminal_train_predictions']}
    objects={review_ref['path']:review,origin['launch_recipe']['path']:old,plan_ref['path']:plan,
        result_ref['path']:result,receipt_ref['path']:receipt,origin['initial_measurement']['path']:initial,
        origin['terminal_train_measurement']['path']:terminal}
    monkeypatch.setattr(m5,'_read',lambda ref,label:objects[ref['path']]);monkeypatch.setattr(m5.campaign,'require_binding',lambda ref,**kwargs:ref)
    monkeypatch.setattr(m5.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=0))
    call=lambda:m5.require_factor_continuation(recipe,owners,source_commit='b'*40)
    call()
    for obj,key,value in ((review,'optimizer_updates',2975),(review,'source_commit','not-a-commit'),
        (plan,'source_commit','b'*40),(receipt,'plan_sha256','different'),(receipt,'guard_decision','FAILED'),
        (result,'parent_cursor',47586),(result,'source_unchanged',False),(terminal,'predictions',ref('other')),
        (old,'source_owners',{}),(old,'run_scope',owner.M5_FACTOR_BPS_SCOPE)):
        previous=obj[key];obj[key]=value
        with pytest.raises(RuntimeError):call()
        obj[key]=previous

    # The real, previously verified review uses fit_result and guard fields.
    initial['predictions']=ref('initial_predictions');receipt['boot']={'boot_id':584}
    actual={'status':'PASS_SIGN_ONEPASS_TERMINAL_CHAIN_NOT_EDGE','fit_result':result_ref,
        'source_unchanged':True,'guard_decision':'PASS','guard_exit_code':0,
        'no_prior_private_weights_reused':True,'forbidden_opens':0,'trainer_processes_remaining':0,
        'terminal_checkpoint':origin['terminal_state'],'initial_predictions':initial['predictions'],
        'boot':receipt['boot'],'oof_parents':17288,'optimizer_updates':2976,'train_parents':47602,
        'test_data_used':False,'plan':plan_ref,'receipt':receipt_ref}
    objects[review_ref['path']]=actual
    call()
    for key,value in (('guard_decision','FAIL'),('guard_exit_code',1),('source_unchanged',False),
        ('no_prior_private_weights_reused',False),('forbidden_opens',1),('oof_parents',17287),
        ('terminal_checkpoint',ref('other')),('initial_predictions',ref('other')),
        ('boot',{'boot_id':585}),('test_data_used',True),('status','OTHER')):
        previous=actual[key];actual[key]=value
        with pytest.raises(RuntimeError):call()
        actual[key]=previous

    # Only the continuation validator may differ; all other owner bytes remain bound.
    control='gx1/contracts/entry_native_direction_m5_fullfit_v1.py';file=tmp_path/control
    file.parent.mkdir(parents=True)
    before=b'def stable():\n    return 1\n\ndef require_factor_continuation(recipe, owners, *, source_commit):\n    return 1\n'
    # Change only the continuation body, not the stable function.
    after=before.replace(b'    return 1\n',b'    return 2\n',2)
    after=after.replace(b'def stable():\n    return 2',b'def stable():\n    return 1',1)
    file.write_bytes(after);path=str(file)
    old_owners={control:{'path':path,'sha256':hashlib.sha256(before).hexdigest()},'other':ref('other')}
    owners={control:{'path':path,'sha256':hashlib.sha256(after).hexdigest()},'other':ref('other')}
    old['source_owners']=old_owners;recipe['source_repo']=str(tmp_path)
    monkeypatch.setattr(m5.subprocess,'check_output',lambda *args,**kwargs:before)
    call()
    changed=owners['other'];owners['other']=ref('changed')
    with pytest.raises(RuntimeError):call()
    owners['other']=changed
    changed_bytes=after.replace(b'def stable():\n    return 1',b'def stable():\n    return 3',1)
    file.write_bytes(changed_bytes);owners[control]['sha256']=hashlib.sha256(changed_bytes).hexdigest()
    with pytest.raises(RuntimeError):call()
    changed_signature=after.replace(b'*, source_commit):',b'*, source_commit, extra=None):')
    file.write_bytes(changed_signature);owners[control]['sha256']=hashlib.sha256(changed_signature).hexdigest()
    with pytest.raises(RuntimeError):call()
    file.write_bytes(after);owners[control]['sha256']='0'*64
    with pytest.raises(RuntimeError):call()
