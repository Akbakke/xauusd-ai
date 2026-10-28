"""Synthetic scope and source-transition checks; no model or market data."""
import copy
import hashlib
import subprocess

import pytest

from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as owner
from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as runner
from tests.test_entry_native_direction_prefix_fit_campaign_v1 import (
    fixture as prefix_fixture, numerical_seal, reseal, write,
)


def test_decimal_fullfit_requires_corrected_private_proof_and_exact_bindings(prefix_fixture, monkeypatch):
    f=prefix_fixture;repo=f['repo'];runtime=f['runtime'];dummy=f['dummy'];private=owner.private
    dec_input='gx1/contracts/entry_native_direction_decimal_input_v1.py'
    dec_data='gx1/contracts/entry_native_direction_decimal_data_v1.py'
    monkeypatch.setattr(private,'DENSE_EXTRA_SOURCE_PATHS',())
    monkeypatch.setattr(private,'DECIMAL_EXTRA_SOURCE_PATHS',(dec_input,dec_data))
    monkeypatch.setattr(private,'REQUIRED_SOURCE_PATHS',('source.py',))
    monkeypatch.setattr(private,'DENSE_INPUT_MANIFEST',f['input_ref'])
    monkeypatch.setattr(private,'BPS_GC_PARITY',dummy)
    owners={'source.py':f['source'],dec_input:write(repo/dec_input,{'owner':'decimal-input'}),
        dec_data:write(repo/dec_data,{'owner':'decimal-data'})}
    original={'path':str(runtime/'original.npz'),'sha256':'a'*64}
    corrected={'path':str(runtime/'corrected.npz'),'sha256':'b'*64}
    refs={'original':original,'original_result':dummy,'corrected':corrected,'corrected_result':dummy}
    monkeypatch.setattr(private,'DENSE_TARGET',original)
    monkeypatch.setattr(private,'DENSE_TARGET_RESULT',dummy)
    monkeypatch.setattr(private,'DECIMAL_INITIAL_PREDICTIONS',dummy)
    audit=write(runtime/'DECIMAL_INPUT_AUDIT.json',{'schema_version':'gx1_decimal_bound_input_actual_audit_v1',
        'status':'COMPLETE','parents':118249,'horizons_minutes':list(range(1,61)),'test_data_used':False,
        'new_model_forwards':0,'optimizer_steps':0,'physical_parent_and_bar_start_hashes_verified':True,
        'corrected_archive_and_receipt_links_verified':True,'original_legacy_anchors_verified':True,
        'immutable_outputs':True,'inputs':refs,'source':owners[dec_input]})
    monkeypatch.setattr(private,'DECIMAL_INPUT_AUDIT',audit)
    monkeypatch.setattr(owner,'DENSE_DECIMAL_REFERENCE_PREDICTIONS',dummy)
    monkeypatch.setattr(owner,'DECIMAL_EVALUATION_SOURCES',('comparison.py',))
    comparison=write(repo/'comparison.py',{'comparison':'fixed'})
    protocol=write(repo/'docs/DIRECTION_DENSE_DECIMAL_EVALUATION_20261028.json',{
        'schema_version':'gx1_dense_decimal_direction_evaluation_protocol_v1',
        'status':'PREDECLARED_BEFORE_DENSE_FIT_NO_EXECUTION_AUTHORITY','fold':f['fold'],
        'corrected_target':corrected,'reference_predictions':dummy,'primary_horizon_minutes':5,
        'comparison_sources':{'comparison.py':comparison},
        'test_data_used':False,'native_launch_authorized':False})
    item=copy.deepcopy(f['recipe']);item['run_scope']=owner.DENSE_DECIMAL_FIT_SCOPE
    item['source_owners']=owners
    num=item['numerical_recipe'];num['dense_minute_supervision']=True;numerical_seal(num)
    prior_num=copy.deepcopy(num);prior_num['max_updates']=2;numerical_seal(prior_num)
    prior=write(runtime/'PRIVATE_RECIPE.json',{'run_scope':private.DECIMAL_PRIVATE_SCOPE,
        'input_manifest':f['input_ref'],'fold':f['fold'],'dense_target':corrected,'dense_target_result':dummy,
        'dense_source_transition_proof':dummy,'decimal_input_audit':audit,'decimal_target_refs':refs,
        'decimal_initial_predictions':dummy,'qualification_weights_for_main_fit':False,
        'numerical_recipe':prior_num,'source_owners':owners})
    private_plan=write(runtime/'PRIVATE_PLAN.json',{'recipe':prior,'source_commit':'a'*40,'plan_sha256':'9'*64})
    result=write(runtime/'PRIVATE_RESULT.json',{'schema_version':'gx1_native_decimal_dense_private_qualification_result_v1',
        'decision':'PASS_DECIMAL_DENSE_NATIVE_MECHANICS','recipe':prior,'decimal_target_refs':refs,
        'decimal_initial_prediction_byte_parity':True,'decimal_prior_against_sign_audit':True,
        'bit_exact_resume':True,'physical_optimizer_updates':4,'source_unchanged':True,'test_data_used':False,
        'private_weights_for_main_fit':False})
    receipt=write(runtime/'PRIVATE_RECEIPT.json',{'outcome':'COMPLETE','guard_decision':'PASS',
        'plan_sha256':'9'*64,'test_data_used':False})
    review_value={'schema_version':'gx1_decimal_dense_private_root_review_v1',
        'status':'PASS_CONSUMED_DECIMAL_PRIVATE_NATIVE_MECHANICS','plan':private_plan,'source_commit':'a'*40,
        'result':result,'receipt':receipt,'idle_restore':dummy,
        'initial_256_logits_parent_and_clock_byte_parity_independently_checked':True,
        'corrected_all_60_train_priors_independently_checked':True,'decimal_target':corrected,
        'gc_projection_and_gate_nonzero_for_all_five_timeframes':True,
        'sixty_horizons_have_eligible_labels_and_gradients':True,'chart_mtf_gradient_first_step_nonzero':True,
        'chart_local_gradient_first_step_zero_and_second_nonzero':True,'bit_exact_resume':True,
        'private_optimizer_updates':4,'qualification_weights_for_main_fit':False,'test_data_used':False}
    review=write(runtime/'PRIVATE_REVIEW.json',review_value)
    monkeypatch.setattr(owner,'DENSE_DECIMAL_NATIVE_QUALIFICATION',review)
    calls=[]
    monkeypatch.setattr(owner,'_require_decimal_source_equivalence',lambda *args:calls.append(args))
    item.update(gc_layout_metadata_parity=dummy,native_dense_qualification=review,dense_source_transition_proof=dummy,
        dense_target_result=dummy,dense_target=corrected,dense_evaluation_protocol=protocol,
        decimal_input_audit=audit,decimal_target_refs=refs,decimal_initial_predictions=dummy)
    reseal(item,'recipe_sha256')
    checked=owner.require_prefix_fit_launch_recipe(write(runtime/'DECIMAL_FULLFIT.json',item),source_repo=repo,source_commit='a'*40)
    assert checked['run_scope']==owner.DENSE_DECIMAL_FIT_SCOPE and len(calls)==1
    assert not (runtime/'original.npz').exists() and not (runtime/'corrected.npz').exists()
    for key,value in (('dense_target',{**corrected,'sha256':'0'*64}),('decimal_input_audit',{**audit,'sha256':'0'*64}),
        ('decimal_initial_predictions',{**dummy,'sha256':'0'*64}),('qualification_weights_for_main_fit',True),
        ('test_data_used',True),('run_scope',owner.DENSE_DECIMAL_PRIVATE_SCOPE)):
        wrong=copy.deepcopy(item);wrong[key]=value;reseal(wrong,'recipe_sha256')
        with pytest.raises(RuntimeError):
            owner.require_prefix_fit_launch_recipe(write(runtime/'BAD_DECIMAL_FULLFIT.json',wrong),source_repo=repo,source_commit='a'*40)
    bad_review=copy.deepcopy(review_value);bad_review['bit_exact_resume']=False
    bad_ref=write(runtime/'BAD_PRIVATE_REVIEW.json',bad_review)
    monkeypatch.setattr(owner,'DENSE_DECIMAL_NATIVE_QUALIFICATION',bad_ref)
    wrong=copy.deepcopy(item);wrong['native_dense_qualification']=bad_ref;reseal(wrong,'recipe_sha256')
    with pytest.raises(RuntimeError,match='DENSE_PRIVATE_QUALIFICATION'):
        owner.require_prefix_fit_launch_recipe(write(runtime/'BAD_DECIMAL_REVIEW.json',wrong),source_repo=repo,source_commit='a'*40)


def test_decimal_scope_reuses_private_adapter_without_mutating_launch():
    launch={'run_scope':owner.DENSE_DECIMAL_FIT_SCOPE,'dense_target':{'path':'/fixed','sha256':'a'*64}}
    adapted=runner._fit_input_launch(launch,owner)
    assert adapted=={**launch,'run_scope':owner.DENSE_DECIMAL_PRIVATE_SCOPE}
    assert launch['run_scope']==owner.DENSE_DECIMAL_FIT_SCOPE
    old={'run_scope':owner.DENSE_SCOPE}
    assert runner._fit_input_launch(old,owner) is old


def test_source_transition_detects_changes_to_qualified_functions(tmp_path,monkeypatch):
    name='owner.py';path=tmp_path/name
    before=b'THRESHOLD = 1\n\ndef _prepare(x):\n    return x + 1\n'
    after=before+b'\ndef new_scope():\n    return True\n'
    path.write_bytes(after)
    old_ref={'path':str(path),'sha256':hashlib.sha256(before).hexdigest()}
    new_ref={'path':str(path),'sha256':hashlib.sha256(after).hexdigest()}
    monkeypatch.setattr(owner,'DECIMAL_TRANSITION_FUNCTIONS',{name:('_prepare',)})
    monkeypatch.setattr(owner,'DECIMAL_TRANSITION_ALLOWED',{name:{'changed_functions':(),
        'new_functions':('new_scope',),'new_assignments':()}})
    monkeypatch.setattr(subprocess,'check_output',lambda args:before)
    owner._require_decimal_source_equivalence(tmp_path,'a'*40,{name:old_ref},{name:new_ref})
    changed=after.replace(b'return x + 1',b'return x + 2');path.write_bytes(changed)
    new_ref['sha256']=hashlib.sha256(changed).hexdigest()
    with pytest.raises(RuntimeError,match='DECIMAL_TRANSITION_FUNCTION'):
        owner._require_decimal_source_equivalence(tmp_path,'a'*40,{name:old_ref},{name:new_ref})
    changed_global=after.replace(b'THRESHOLD = 1',b'THRESHOLD = 2');path.write_bytes(changed_global)
    new_ref['sha256']=hashlib.sha256(changed_global).hexdigest()
    with pytest.raises(RuntimeError,match='DECIMAL_TRANSITION_MODULE'):
        owner._require_decimal_source_equivalence(tmp_path,'a'*40,{name:old_ref},{name:new_ref})
    old_ref['sha256']='0'*64
    with pytest.raises(RuntimeError,match='DECIMAL_TRANSITION_SHA'):
        owner._require_decimal_source_equivalence(tmp_path,'a'*40,{name:old_ref},{name:new_ref})
