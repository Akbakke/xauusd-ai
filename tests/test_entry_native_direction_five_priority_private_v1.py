"""Synthetic tests for private qualification bindings and initial identity."""
import copy
from pathlib import Path
import numpy as np
import pytest
from gx1.contracts import entry_native_direction_five_priority_private_v1 as owner
from gx1.contracts.entry_native_direction_m5_fullfit_v1 import INITIAL
from gx1.scripts.run_entry_native_direction_dense_qualification_v1 import require_five_priority_initial_parity


def test_initial_prediction_and_model_buffer_parity_fail_closed(tmp_path):
    rows=np.arange(256,dtype=np.int64)
    data={'parent_rows':rows,'bar_start_ns':rows*300,'decision_ns':rows*300+300,'logit_up':np.zeros((256,60),dtype=np.float32)}
    p=tmp_path/'initial.npz';np.savez(p,**data)
    ref={'path':str(p),'sha256':'0'*64}
    old={'preservation':{'model':'a'*64,'all_buffers':'b'*64}}
    measured={'arrays':data,'state_preservation':dict(old['preservation'])}
    require_five_priority_initial_parity(measured,old,ref,np=np)
    changed=copy.deepcopy(measured);changed['arrays']['logit_up'][0,4]=1
    with pytest.raises(RuntimeError):require_five_priority_initial_parity(changed,old,ref,np=np)
    changed=copy.deepcopy(measured);changed['state_preservation']['model']='c'*64
    with pytest.raises(RuntimeError):require_five_priority_initial_parity(changed,old,ref,np=np)
    changed=copy.deepcopy(measured);changed['state_preservation']['all_buffers']='d'*64
    with pytest.raises(RuntimeError):require_five_priority_initial_parity(changed,old,ref,np=np)


def setup_recipe(monkeypatch):
    source=Path('/home/andre2/src/GX1_CURRENT');fold={'fold_id':1}
    owners={name:{'path':str(source/name),'sha256':'a'*64} for name in owner.EXTRA_SOURCE_PATHS}
    review_ref={'path':owner.SOURCE_REVIEW_PATH,'sha256':'b'*64}
    recipe={'admission_stage':owner.STAGE,'candidate_mode':'price_only','priority_design':owner.DESIGN,
            'priority_baseline_initial_report':owner.BASELINE_INITIAL_REPORT,'priority_source_review':review_ref,
            'numerical_recipe':{'five_minute_priority':True},'fold':fold}
    design={'schema_version':'gx1_m5_five_priority_candidate_design_v1','status':'PREDECLARED_NO_EXECUTION_AUTHORITY',
            'candidate':{'five_minute_weight':10,'other_horizon_weight':1,'horizon_minutes':list(range(1,61))},
            'baseline':{'recipe':owner.BASELINE_RECIPE},'test_data_used':False}
    initial={'role':'INITIAL_TRAIN','test_data_used':False,'fold':fold,'predictions':{'sha256':INITIAL['sha256']},
             'preservation':{'model':'a'*64,'all_buffers':'b'*64}}
    baseline={'fold':fold}
    review={'schema_version':'gx1_m5_five_priority_source_review_v1','status':'SOURCE_REVIEWED_NO_EXECUTION',
            'baseline_recipe':owner.BASELINE_RECIPE,'candidate_design':owner.DESIGN,
            'baseline_source_commit':'ac0a17671ebb0d66ea5e90da3f3c501e2725b60e','candidate_source_owners':owners,
            'declared_source_changes_reviewed':True,'dense_forward_ast_unchanged':True,
            'all_unmodified_baseline_owners_verified':True,'transition_summary':{'baseline_owner_count':63},'test_data_used':False}
    monkeypatch.setattr(owner,'require_source_transition',lambda *args,**kwargs:{'baseline_owner_count':63})
    docs={owner.DESIGN['path']:design,owner.BASELINE_INITIAL_REPORT['path']:initial,owner.BASELINE_RECIPE['path']:baseline,review_ref['path']:review}
    monkeypatch.setattr(owner,'_read',lambda ref,label:docs[ref['path']])
    return recipe,owners,source,docs


def test_private_recipe_binds_single_exact_m5_design(monkeypatch):
    recipe,owners,source,docs=setup_recipe(monkeypatch)
    assert owner.require_private_recipe(recipe,owners,source_repo=source,source_commit='a'*40)['design']==docs[owner.DESIGN['path']]
    for mutate in ('other_weight','unreviewed','wrong_initial','m1'):
        recipe,owners,source,docs=setup_recipe(monkeypatch)
        if mutate=='other_weight':docs[owner.DESIGN['path']]['candidate']['other_horizon_weight']=2
        if mutate=='unreviewed':docs[owner.SOURCE_REVIEW_PATH]['all_unmodified_baseline_owners_verified']=False
        if mutate=='wrong_initial':docs[owner.BASELINE_INITIAL_REPORT['path']]['predictions']['sha256']='1'*64
        if mutate=='m1':recipe['candidate_m1_refs']={}
        with pytest.raises(RuntimeError):owner.require_private_recipe(recipe,owners,source_repo=source,source_commit='a'*40)


def test_ast_scope_allows_only_declared_function_changes():
    before=b"def old():\n    return 1\n\ndef stable():\n    return 2\n"
    good=b"def old():\n    return 9\n\ndef stable():\n    return 2\n\ndef added():\n    return 3\n"
    bad=good.replace(b"return 2",b"return 7")
    spec={'functions':('old',),'new_functions':('added',)}
    assert owner._unchanged_ast(before,spec,original=True)==owner._unchanged_ast(good,spec,original=False)
    assert owner._unchanged_ast(before,spec,original=True)!=owner._unchanged_ast(bad,spec,original=False)
    with pytest.raises(RuntimeError):owner._unchanged_ast(before,{'functions':('missing',)},original=True)


def test_ast_scope_preserves_dense_forward_and_requires_new_method():
    before=b"class M:\n    def __init__(self): pass\n    def forward(self): return 1\n"
    good=b"class M:\n    def __init__(self): return None\n    def added(self): return True\n    def forward(self): return 1\n"
    bad=good.replace(b"return 1",b"return 2")
    spec={'class_name':'M','methods':('__init__',),'new_methods':('added',)}
    assert owner._unchanged_ast(before,spec,original=True)==owner._unchanged_ast(good,spec,original=False)
    assert owner._unchanged_ast(before,spec,original=True)!=owner._unchanged_ast(bad,spec,original=False)
