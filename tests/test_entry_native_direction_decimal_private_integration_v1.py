"""Synthetic tests for the separate corrected input; no native training or data."""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix_authority
from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
from gx1.scripts import run_entry_native_direction_dense_qualification_v1 as qualification
from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as prefix
from gx1.scripts import run_entry_native_direction_research_window_v1 as common
from tests.test_entry_native_direction_decimal_input_v1 import make_case
from tests.test_entry_native_direction_research_campaign_v1 import fixture as research_fixture, reseal, write


def test_private_recipe_binds_exact_decimal_audit_and_no_full_epoch(research_fixture, monkeypatch):
    f=research_fixture;repo=f['repo'];runtime=f['runtime'];dummy=f['dummy']
    original={'path':str(runtime/'original.npz'),'sha256':'a'*64}
    corrected={'path':str(runtime/'corrected.npz'),'sha256':'b'*64}
    source_key='gx1/contracts/entry_native_direction_decimal_input_v1.py'
    source=write(repo/source_key,{'synthetic':'owner'})
    original_result=write(runtime/'original-result.json',{'synthetic':True})
    corrected_result=write(runtime/'corrected-result.json',{'synthetic':True})
    refs={'legacy':{'path':str(runtime/'legacy.npz'),'sha256':'c'*64},'original':original,
        'original_result':original_result,'corrected':corrected,'corrected_result':corrected_result,
        **{k:write(runtime/(k+'.json'),{'synthetic':k}) for k in ('consumer_plan','consumer_result','materialization_plan','audit_result')}}
    audit={'schema_version':'gx1_decimal_bound_input_actual_audit_v1','status':'COMPLETE','parents':118249,
        'horizons_minutes':list(range(1,61)),'test_data_used':False,'new_model_forwards':0,'optimizer_steps':0,
        'physical_parent_and_bar_start_hashes_verified':True,'corrected_archive_and_receipt_links_verified':True,
        'original_legacy_anchors_verified':True,'immutable_outputs':True,'source':source,'inputs':refs}
    audit_ref=write(runtime/'decimal-input-audit.json',audit)
    proof=write(repo/'docs/DIRECTION_DENSE_SOURCE_TRANSITION_20261028.json',{'synthetic':True})
    for name,value in [('DENSE_INPUT_MANIFEST',dummy),('DENSE_TARGET_RESULT',original_result),('DENSE_TARGET',original),
                       ('BPS_GC_PARITY',dummy),('ORDER_SUPPORTED_DESIGN',dummy),('DENSE_EXTRA_SOURCE_PATHS',()),
                       ('DECIMAL_EXTRA_SOURCE_PATHS',(source_key,)),('DECIMAL_INPUT_AUDIT',audit_ref)]:monkeypatch.setattr(authority,name,value)
    item=copy.deepcopy(f['recipe']);item['run_scope']=authority.DECIMAL_PRIVATE_SCOPE
    item.update(fold=authority.BPS_FOLD,gc_layout_metadata_parity=dummy,qualification_design=dummy,
        dense_source_transition_proof=proof,dense_target_result=corrected_result,dense_target=corrected,
        decimal_input_audit=audit_ref,decimal_target_refs=refs,max_forwards=21)
    item['source_owners'][source_key]=source
    num=item['numerical_recipe'];num.update(batch_size=16,dense_minute_supervision=True)
    num.pop('contract_sha256');num['contract_sha256']=hashlib.sha256(json.dumps(num,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    reseal(item,'recipe_sha256');checked=authority.require_research_launch_recipe(write(runtime/'private.json',item),source_repo=repo,source_commit='a'*40)
    assert checked['run_scope']==authority.DECIMAL_PRIVATE_SCOPE and checked['full_epoch_training_allowed'] is False
    assert not Path(corrected['path']).exists()
    for change in ('reference','audit','full_epoch','source'):
        wrong=copy.deepcopy(item)
        if change=='reference':wrong['decimal_target_refs']['corrected']['sha256']='0'*64
        if change=='audit':wrong['decimal_input_audit']['sha256']='0'*64
        if change=='full_epoch':wrong['full_epoch_training_allowed']=True
        if change=='source':wrong['source_owners'][source_key]['sha256']='0'*64
        reseal(wrong,'recipe_sha256')
        with pytest.raises(RuntimeError):authority.require_research_launch_recipe(write(runtime/'bad.json',wrong),source_repo=repo,source_commit='a'*40)


def test_decimal_private_adapter_is_rejected_as_prefix_fit_scope():
    assert prefix_authority.DENSE_DECIMAL_PRIVATE_SCOPE != prefix_authority.DENSE_SCOPE
    assert prefix_authority.DENSE_DECIMAL_PRIVATE_SCOPE not in (
        prefix_authority.BPS_SCOPE,prefix_authority.CONTINUATION_SCOPE,prefix_authority.ORDER_SCOPE,prefix_authority.DENSE_SCOPE,
        'single_prefix_one_pass_direction_fit')


def test_native_prepare_uses_authenticated_corrected_sign_and_rejects_tampering(tmp_path,monkeypatch):
    from gx1.contracts import entry_native_direction_feature_view_v1 as view
    from gx1.contracts import entry_native_direction_context_report_v1 as report
    refs,axes,legacy,corrected=make_case(tmp_path)
    class Archive:
        files=list(prefix.PRICE_MEMBERS)
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def __getitem__(self,key):
            return np.tile(np.array([60,300,900],np.int64),(2,1)) if key=='elapsed_wall_seconds' else legacy[key]
    np_proxy=SimpleNamespace(load=lambda path,allow_pickle=False:Archive())
    arrays={k:axes[k].copy() for k in ('parent_rows','bar_start_ns','decision_ns')}
    prepared={'arrays':arrays,'gc':{'layout':{},'transition':{}},
        'entry_sequence_source':{'times_ns':np.arange(3,dtype=np.int64),'sample_positions':np.arange(2,dtype=np.int64)}}
    physical=SimpleNamespace(prepare_inputs=lambda *_:prepared,release_pages=lambda *_:None)
    source_refs={k:{'path':'/synthetic/'+k,'sha256':'a'*64} for k in view.SOURCE_REF_KEYS}
    manifest={'targets':refs['legacy'],'view_source_bindings':source_refs,
        'prefix_normalization':{'output':{'path':'/synthetic/prefix','sha256':'a'*64}},
        'constructor_metadata':{'path':'/synthetic/constructor','sha256':'a'*64}}
    context={'physical':physical,'ancestor':object(),'ancestor_plan':{},'info':{'metadata':{}},'manifest':manifest,
        'fitted':{'base_normalization':{'surfaces':{'mtf_m5':{'field_names':[]}}}},'constructor':{},
        'dense_target':refs['corrected'],'decimal_original':refs['original'],'decimal_target_refs':refs}
    owners={p:{'path':'/synthetic/'+p,'sha256':'b'*64} for p in ('gx1/contracts/entry_native_direction_dense_v1.py',
        'gx1/features/entry_specialist_feature_groups_v1.py','gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py')}
    launch={'run_scope':prefix_authority.DENSE_DECIMAL_PRIVATE_SCOPE,'dense_target':refs['corrected'],
        'decimal_target_refs':refs,'fold':{'prefix_rows':2},'source_owners':owners}
    monkeypatch.setattr(common,'_read',lambda *_:{})
    monkeypatch.setattr(prefix,'_qualify_axes_targets',lambda *a,**k:None)
    monkeypatch.setattr(prefix,'_presentation_order',lambda parents,*a,**k:parents.copy())
    monkeypatch.setattr(view,'bind_native_direction_feature_view',lambda **kwargs:kwargs)
    monkeypatch.setattr(report,'native_direction_context_map',lambda **kwargs:{})
    from gx1.scripts.audit_entry_direction_candle_midpoints_v1 import no_native_models
    previous_profile=sys.getprofile();sys.setprofile(no_native_models)
    try:
        _,targets,_,_,_=prefix._prepare(context,launch,np_proxy,dense_minutes=True)
    finally:
        sys.setprofile(previous_profile)
    assert set(targets)==set(prefix.TARGET_MEMBERS) and np.array_equal(targets['sign'],corrected['sign'])
    assert legacy['sign'][0,0]==1 and targets['sign'][0,0]==0
    assert all(not v.flags.writeable for v in targets.values())
    changed=copy.deepcopy(launch);changed['decimal_target_refs']['original']['sha256']='0'*64
    with pytest.raises(RuntimeError,match='DECIMAL_EXACT_SCOPE'):prefix._prepare(context,changed,np_proxy,dense_minutes=True)
    Path(refs['corrected']['path']).write_bytes(b'altered')
    with pytest.raises(RuntimeError,match='SHA-256 mismatch'):prefix._prepare(context,launch,np_proxy,dense_minutes=True)


def test_decimal_original_guard_allows_only_bound_read_and_freeze_detects_change(monkeypatch):
    hooks=[];monkeypatch.setattr(common.sys,'addaudithook',hooks.append)
    old={'path':'/home/andre2/GX1_RUNS/synthetic/old.npz','sha256':'a'*64}
    new={'path':'/home/andre2/GX1_RUNS/synthetic/new.npz','sha256':'b'*64}
    context={'ancestor_plan':{'payloads':{},'native_cache_dir':'/synthetic/cache','gc_surfaces':{'path':'/synthetic/gc/surface'}},
        'info':{'metadata':{},'audit':{'manifest_path':'/synthetic/audit'}},'dense_target':new,'dense_target_stat':{'id':1},
        'decimal_original':old,'decimal_original_stat':{'id':1},'decimal_target_refs':{'original':old}}
    common._install_read_guard(context,'/synthetic/output','/synthetic/runtime',dense_target=new,decimal_original=old)
    hooks[0]('open',(old['path'],'rb',0));hooks[0]('open',(new['path'],'rb',0))
    with pytest.raises(RuntimeError,match='ORIGINAL_WRITE'):hooks[0]('open',(old['path'],'wb',os.O_WRONLY))
    with pytest.raises(RuntimeError,match='UNBOUND_OPEN'):hooks[0]('open',('/home/andre2/GX1_RUNS/synthetic/other.npz','rb',0))
    with pytest.raises(RuntimeError,match='DECIMAL_ORIGINAL_BINDING'):
        common._install_read_guard(context,'/synthetic/output','/synthetic/runtime',dense_target=new,decimal_original={**old,'sha256':'0'*64})
    monkeypatch.setattr(common.subprocess,'check_output',lambda args,**kwargs:'head\n' if 'rev-parse' in args else '')
    monkeypatch.setattr(common,'_stat',lambda _: {'id':1})
    hashes={old['path']:old,new['path']:new,'/synthetic/policy':{'path':'/synthetic/policy','sha256':'c'*64}}
    monkeypatch.setattr(common,'_hash_file',lambda path:hashes[path])
    binding={'source_head':'head','policy':hashes['/synthetic/policy']}
    common._freeze({'source_owners':{}},binding,context)
    hashes[old['path']]={**old,'sha256':'0'*64}
    with pytest.raises(RuntimeError,match='DECIMAL_ORIGINAL_FREEZE'):common._freeze({'source_owners':{}},binding,context)


def test_decimal_prior_matches_all_sixty_independent_confusion_rows():
    rows=[{'horizon_minutes':i+1,'nonzero_sign_flips':0,'observed_mature':8,
        'saved_by_exact_sign_order_minus_zero_plus':[[3,1,0],[0,0,0],[0,1,3]]} for i in range(60)]
    audit={'schema_version':'gx1_direction_decimal_sign_audit_result_v1','status':'COMPLETE','test_data_used':False,
        'new_model_forwards':0,'optimizer_steps':0,'target_source_rows_and_masks_parity':True,
        'vendor_to_canonical_float64_and_clock_parity':True,'confusions':{'fold1_train':rows}}
    prior={'eligible_up':[3]*60,'eligible_down':[3]*60}
    qualification.require_decimal_prefix_prior(prior,audit)
    wrong=copy.deepcopy(prior);wrong['eligible_up'][27]=4
    with pytest.raises(RuntimeError,match='DECIMAL_PRIOR_COUNT:28'):qualification.require_decimal_prefix_prior(wrong,audit)
    wrong=copy.deepcopy(audit);wrong['confusions']['fold1_train'][4]['nonzero_sign_flips']=1
    with pytest.raises(RuntimeError,match='DECIMAL_PRIOR_COUNT:5'):qualification.require_decimal_prefix_prior(prior,wrong)


def test_decimal_launch_metadata_preserves_original_target_and_receipts(monkeypatch):
    from gx1.contracts import entry_native_direction_research_campaign_v1 as private
    def ref(name):return {'path':'/synthetic/'+name,'sha256':(name[0]*64)}
    refs={key:ref(key+'.json' if key not in ('legacy','original','corrected') else key+'.npz') for key in
        ('legacy','original','original_result','corrected','corrected_result','consumer_plan','consumer_result','materialization_plan','audit_result')}
    proof_ref=ref('transition.json');audit_ref=ref('qualification.json');gc_ref=ref('gc.json')
    proof={'schema_version':'gx1_dense_source_transition_proof_v1','status':'SOURCE_ONLY_VERIFIED_NO_DATA_OR_TRAINING',
        'test_data_used':False,'genuine_training_authorized':False,'source_transitions':{'synthetic':'transition'}}
    qualified={'schema_version':'gx1_decimal_bound_input_actual_audit_v1','status':'COMPLETE','test_data_used':False,'inputs':refs}
    target={'schema_version':'gx1_direction_decimal_targets_result_v1','status':'COMPLETE','test_data_used':False,
        'target_identity':'exact_original_decimal_midpoint_v1','parent_rows':118249,'horizons_minutes':list(range(1,61)),'target':refs['corrected']}
    values={proof_ref['path']:proof,audit_ref['path']:qualified,refs['corrected_result']['path']:target,
        **{refs[k]['path']:{'kind':k} for k in ('original_result','consumer_plan','consumer_result','materialization_plan','audit_result')}}
    helper=SimpleNamespace(_read=lambda item,metadata:values[item['path']],_stat=lambda _path:{'id':1},
        _hash_file=lambda path:refs['corrected'] if path==refs['corrected']['path'] else refs['original'])
    monkeypatch.setattr(private,'DECIMAL_INPUT_AUDIT',audit_ref)
    monkeypatch.setattr(private,'DENSE_TARGET',refs['original']);monkeypatch.setattr(private,'DENSE_TARGET_RESULT',refs['original_result'])
    monkeypatch.setattr(prefix,'_metadata',lambda *a,**k:{'info':{'metadata':{}},'manifest':{'targets':refs['legacy']}})
    launch={'run_scope':prefix_authority.DENSE_DECIMAL_PRIVATE_SCOPE,'dense_source_transition_proof':proof_ref,
        'dense_target_result':refs['corrected_result'],'dense_target':refs['corrected'],'decimal_input_audit':audit_ref,
        'decimal_target_refs':refs,'gc_layout_metadata_parity':gc_ref}
    context=prefix._launch_metadata(launch,helper)
    assert context['dense_target']==refs['corrected'] and context['decimal_original']==refs['original']
    assert context['decimal_sign_audit']=={'kind':'audit_result'} and context['decimal_target_refs']==refs
    wrong=copy.deepcopy(launch);wrong['decimal_target_refs']['corrected']['sha256']='0'*64
    with pytest.raises(RuntimeError,match='DECIMAL_TARGET_RESULT'):prefix._launch_metadata(wrong,helper)
    target['target_identity']='wrong'
    with pytest.raises(RuntimeError,match='DECIMAL_TARGET_RESULT'):prefix._launch_metadata(launch,helper)


def test_guarded_dispatcher_selects_new_private_scope_only(monkeypatch,tmp_path):
    from gx1.contracts import gx1_capped_execution_v1 as capped
    calls=[]
    monkeypatch.setattr(capped,'require_guarded_cuda_trainer_execution',lambda:calls.append('guard') or {})
    monkeypatch.setattr(authority,'require_native_direction_research_context',lambda *args:
        (None,None,None,None,{'run_scope':authority.DECIMAL_PRIVATE_SCOPE}))
    monkeypatch.setattr(qualification,'run_dense_qualification',lambda **kwargs:calls.append('decimal-private') or {'scope':'synthetic'})
    result=common.run_native_direction_research_window(policy_path=tmp_path/'policy',policy_file_sha256='0'*64,progress_path=tmp_path/'progress')
    assert result=={'scope':'synthetic'} and calls==['guard','decimal-private'] and list(tmp_path.iterdir())==[]


from tests.test_entry_native_direction_prefix_fit_campaign_v1 import fixture as prefix_fixture


def test_private_decimal_adapter_is_rejected_by_real_prefix_fit_validator(prefix_fixture):
    f=prefix_fixture;recipe=copy.deepcopy(f['recipe']);recipe['run_scope']=prefix_authority.DENSE_DECIMAL_PRIVATE_SCOPE
    reseal(recipe,'recipe_sha256');ref=write(f['runtime']/'decimal-prefix-fit.json',recipe)
    with pytest.raises(RuntimeError,match='RECIPE_SCOPE'):
        prefix_authority.require_prefix_fit_launch_recipe(ref,source_repo=f['repo'],source_commit='a'*40)


def test_data_audit_releases_its_scope_before_exact_owner(tmp_path):
    from gx1.contracts import entry_native_direction_decimal_input_v1 as owner
    from gx1.scripts import audit_direction_decimal_private_input_v1 as audit
    refs,axes,legacy,corrected=make_case(tmp_path)
    with np.load(refs['legacy']['path'],allow_pickle=False) as archive:members=list(archive.files)
    actual,snapshot=audit.load_verified_axes(refs,members)
    assert all(np.array_equal(actual[k],axes[k]) for k in actual)
    targets=owner.load_bound_decimal_direction_targets(refs=refs,axes=axes,legacy=legacy)
    assert np.array_equal(targets['sign'],corrected['sign'])
    snapshot.require_unchanged()
