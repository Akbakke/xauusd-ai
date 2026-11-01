"""Synthetic recovery admission only: no native model, checkpoint or market payload."""
import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as owner
from gx1.contracts import local_random_access_campaign_v2 as campaign
from tests.test_entry_native_direction_prefix_fit_campaign_v1 import (
    fixture as base_fixture, write, reseal, sealed, cursor,
)


@pytest.fixture
def recovery_fixture(base_fixture, monkeypatch):
    f=base_fixture; repo=f['repo']; old=f['runtime']; stage='dynamic_m5_direction_synthetic_original'
    original=copy.deepcopy(f['recipe'])
    original.update(run_scope=owner.FAMILY_SIGN_SCOPE,family_arm='session_regime_encoder',admission_stage=stage)
    reseal(original,'recipe_sha256'); original_ref=write(old/'ORIGINAL.json',original)
    plan=sealed({'source_commit':owner.SCORING_RECOVERY_HEAD,'recipe':original_ref,
        'runtime_root':str(old)},'plan_sha256')
    plan_ref=write(old/'PLAN.json',plan)
    result={'durable_checkpoint':{'contract_sha256':owner.SCORING_RECOVERY_STATE_SEAL,'state':f['dummy']},
        'source_unchanged':True,'error':None}
    result_ref=write(old/'RESULT.json',result)
    saved=cursor(f,recipe_ref=original_ref,parent_cursor=47602,optimizer_updates=2976,
        phase_state='SCORING',result=result_ref)
    saved_ref=write(old/'invocation-evidence/invocation-0001/CHECKPOINT_POINTER.json',saved)
    receipt=sealed({'schema_version':campaign.RECEIPT_SCHEMA,'plan_sha256':plan['plan_sha256'],
        'invocation_number':1,'outcome':'RESUMABLE','guard_decision':'PASS',
        'trainer_guard_exit_code':0,'progress_observer_exit_code':0,'checkpoint_pointer_snapshot':saved_ref},'receipt_sha256')
    receipt_ref=write(old/'RECEIPT.json',receipt)
    execution_ref=write(old/'BINDING.json',{'source_head':owner.SCORING_RECOVERY_HEAD,'plan':plan_ref})
    review={'schema_version':'gx1_family_training_window_review_v1','status':'PASS_TRAINING_COMPLETE_SCORING_PENDING',
        'source_head':owner.SCORING_RECOVERY_HEAD,'family_arm':'session_regime_encoder','optimizer_updates':2976,
        'train_parents':47602,'training_passes':1,'full_train_scored':False,'later_scored':False,
        'test_data_used':False,'execution_binding':execution_ref,'receipt':receipt_ref,
        'cursor':saved_ref,'full_state':f['dummy'],'result':result_ref}
    review_ref=write(old/'REVIEW.json',review)
    integrity={'schema_version':'gx1_session_interrupted_scoring_integrity_v1',
        'status':'TRAINING_STATE_BYTES_PRESERVED_SCORING_INTERRUPTED','source_head':owner.SCORING_RECOVERY_HEAD,
        'native_process_absent':True,'originals_modified':False,'model_or_checkpoint_deserialized':False,
        'completed_training_updates':2976,'training_parents':47602,'full_train_scoring_complete':False,
        'later_scoring_complete':False,'new_model_forwards':0,'new_optimizer_updates':0,'test_data_used':False,
        'training_receipt':receipt_ref,'cursor':saved_ref,'full_state':f['dummy'],'own_initial_state':f['dummy'],
        'training_result':result_ref}
    integrity_ref=write(old/'INTEGRITY.json',integrity)
    peer={'schema_version':'gx1_session_interruption_independent_metadata_peer_review_v1',
        'status':'PASS_SCOPED_METADATA_WORKER_REVIEW','references':{'integrity':integrity_ref}}
    peer_ref=write(old/'PEER.json',peer)
    closed_gate={'status':'INTERRUPTED_CONSUMED_W2_WINDOWS_BUGCHECK_W1_TRAINING_PRESERVED',
        'claimed_invocations':2,'completed_invocations':1,'interrupted_invocations':1,'unclaimed_invocations_revoked':[3,4],
        'recipe':original_ref,**dict.fromkeys(('execution_authorized','training_enabled','native_training_enabled',
            'full_epoch_training_allowed','native_launch_authorized','genuine_updates_authorized','relaunch_allowed'),False)}
    closed_raw=json.dumps({stage:closed_gate}).encode()
    closure={'schema_version':'gx1_family_interrupted_scoring_closure_v1',
        'status':'CLOSED_INTERRUPTED_SCORING_ORIGINALS_PRESERVED','family_arm':'session_regime_encoder',
        'original_source_head':owner.SCORING_RECOVERY_HEAD,'original_relaunch_allowed':False,
        'training_state_preserved':True,'recovery_execution_authorized':False,'test_data_used':False,
        'interrupted_native_exit_code_known':False,'interrupted_native_exit_code':None,
        'active_native_pids':[],'revoked_invocations':[3,4],'state_integrity':integrity_ref,
        'training_window_review':review_ref,'interruption_review':peer_ref,'closure_source_head':'b'*40,
        'closed_stage':stage,'closed_policy_git':{'commit':'b'*40,'path':'NEXT_RUN_POLICY.json',
            'sha256':hashlib.sha256(closed_raw).hexdigest()}}
    closure_ref=write(old/'CLOSURE.json',closure);monkeypatch.setattr(owner,'SCORING_RECOVERY_CLOSURE',closure_ref)
    origin={'schema_version':'gx1_family_session_scoring_recovery_origin_v1','family_arm':'session_regime_encoder',
        'original_source_head':owner.SCORING_RECOVERY_HEAD,'original_recipe':original_ref,
        'original_plan':plan_ref,'training_receipt':receipt_ref,'training_cursor':saved_ref,
        'training_review':review_ref,'state_integrity':integrity_ref,'interruption_review':peer_ref,'closure_review':closure_ref}
    recipe=copy.deepcopy(original); new=old.parent/'recovery'
    recipe.update(scoring_recovery_origin=origin,runtime_root=str(new),admission_stage='dynamic_m5_direction_synthetic_recovery',
        execution_binding_path=str(new/'EXECUTION_BINDING.json'),max_windows=1,training_enabled=False,full_epoch_training_allowed=False)
    reseal(recipe,'recipe_sha256')
    real_git=subprocess.check_output
    def git(args,**kwargs):
        if args[-1]=='b'*40+':NEXT_RUN_POLICY.json':return closed_raw
        return real_git(args,**kwargs)
    monkeypatch.setattr(subprocess,'check_output',git)
    # Source transition is tested separately against exact source bytes below.
    monkeypatch.setattr(owner,'_require_scoring_recovery_source_equivalence',lambda *a:None)
    return locals()


def require(f):
    return owner.require_scoring_recovery_origin(f['recipe'],source_repo=f['repo'],source_commit='c'*40)


def test_original_training_and_new_scoring_have_distinct_lineage(recovery_fixture):
    f=recovery_fixture; checked=require(f)
    assert checked['cursor']['optimizer_updates']==2976
    assert checked['cursor']['phase_state']=='SCORING'
    assert checked['original_recipe']['runtime_root']!=f['recipe']['runtime_root']
    assert f['recipe']['numerical_recipe']==checked['original_recipe']['numerical_recipe']


@pytest.mark.parametrize('field,value',[
    ('family_arm','momentum_flow_encoder'),('original_source_head','d'*40),
    ('schema_version','generic_recovery'),('closure_review',{'path':'/tmp/foreign','sha256':'a'*64}),
])
def test_foreign_or_unclosed_origin_rejected(recovery_fixture,field,value):
    f=recovery_fixture;f['recipe']['scoring_recovery_origin'][field]=value
    with pytest.raises(campaign.RandomAccessCampaignError):require(f)


@pytest.mark.parametrize('field,value',[
    ('numerical_recipe',{'max_updates':2977}),('fold',{'fold_id':2}),('train_passes',2),
    ('run_scope',owner.M5_FACTOR_SIGN_SCOPE),('family_arm','baseline'),
])
def test_cannot_change_inherited_learning_problem(recovery_fixture,field,value):
    f=recovery_fixture; f['recipe'][field]=value
    with pytest.raises(campaign.RandomAccessCampaignError):require(f)


def test_must_use_receipt_immutable_cursor_snapshot(recovery_fixture):
    f=recovery_fixture; alternate=write(f['old']/'RESUME_CURSOR.json',f['saved'])
    f['recipe']['scoring_recovery_origin']['training_cursor']=alternate
    with pytest.raises(campaign.RandomAccessCampaignError,match='TRAINING_RECEIPT'):require(f)


@pytest.mark.parametrize('name', ['closure_ref','integrity_ref','review_ref','receipt_ref','saved_ref','result_ref','peer_ref'])
def test_bound_evidence_byte_tamper_rejected(recovery_fixture,name):
    f=recovery_fixture;Path(f[name]['path']).write_text('{}')
    with pytest.raises(campaign.RandomAccessCampaignError):require(f)


def test_closed_policy_is_verified_from_git_bytes(recovery_fixture,monkeypatch):
    f=recovery_fixture
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:b'{}')
    with pytest.raises(campaign.RandomAccessCampaignError,match='CLOSED_POLICY_BYTES'):require(f)


@pytest.mark.parametrize('nested,key,value',[
    ('integrity','completed_training_updates',2975),('integrity','native_process_absent',False),
    ('integrity','full_train_scoring_complete',True),('review','training_passes',2),
    ('review','later_scored',True),('peer','status','FAIL'),
    ('result','durable_checkpoint',{'contract_sha256':'a'*64,'state':{}}),
])
def test_semantic_evidence_mismatch_rejected(recovery_fixture,monkeypatch,nested,key,value):
    f=recovery_fixture; original_read=campaign.read_bound_json
    target=f[nested+'_ref']['path']
    def read(path,sha):
        value_read=original_read(path,sha)
        if str(path)==target:value_read[key]=value
        return value_read
    monkeypatch.setattr(campaign,'read_bound_json',read)
    with pytest.raises(campaign.RandomAccessCampaignError):require(f)


def test_control_transition_preserves_numerical_bytes(tmp_path,monkeypatch):
    repo=tmp_path; path='control.py'
    before=b'def control():\n    return 1\n\ndef numerical():\n    return 8\n'
    after=b'def control():\n    return 2\n\ndef numerical():\n    return 8\n'
    (repo/path).write_bytes(after)
    def ref(raw):return {'path':str(repo/path),'sha256':hashlib.sha256(raw).hexdigest()}
    original={'source_owners':{path:ref(before)}};current={'source_owners':{path:ref(after)}}
    monkeypatch.setattr(owner,'SCORING_RECOVERY_CONTROL_CHANGES',{path:(('control',),(),())})
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:before)
    owner._require_scoring_recovery_source_equivalence(repo,original,current)
    changed=after.replace(b'return 8',b'return 9');(repo/path).write_bytes(changed)
    current['source_owners'][path]=ref(changed)
    with pytest.raises(campaign.RandomAccessCampaignError,match='NONCONTROL_SOURCE_CHANGED'):
        owner._require_scoring_recovery_source_equivalence(repo,original,current)


def test_noncontrol_owner_cannot_take_hash_exemption(tmp_path,monkeypatch):
    path=tmp_path/'numerical.py';path.write_bytes(b'new')
    before={'path':str(path),'sha256':hashlib.sha256(b'old').hexdigest()}
    after={'path':str(path),'sha256':hashlib.sha256(b'new').hexdigest()}
    monkeypatch.setattr(owner,'SCORING_RECOVERY_CONTROL_CHANGES',{})
    with pytest.raises(campaign.RandomAccessCampaignError,match='NUMERICAL_SOURCE_CHANGED'):
        owner._require_scoring_recovery_source_equivalence(tmp_path,{'source_owners':{'numerical.py':before}},
            {'source_owners':{'numerical.py':after}})


@pytest.mark.parametrize('recovery', [False,True])
def test_invocation_budget_is_zero_only_with_validated_recovery(base_fixture,monkeypatch,recovery):
    f=base_fixture;recipe=copy.deepcopy(f['recipe']);inv=copy.deepcopy(f['invocations'][0])
    if recovery:recipe['scoring_recovery_origin']={'validated_by_separate_origin_tests':True}
    monkeypatch.setattr(owner,'require_prefix_fit_launch_recipe',lambda *a,**k:recipe)
    inv.update(window_count=1,optimizer_step_budget=0,expected_success_outcome='COMPLETE')
    execution=json.loads(Path(inv['execution_manifest']['path']).read_text())
    if recovery:
        owner.require_prefix_fit_invocation_execution(inv,execution,source_repo=f['repo'],source_commit='a'*40)
        assert inv['direction_prefix_fit_scoring_recovery'] is True
    else:
        with pytest.raises(campaign.RandomAccessCampaignError,match='INVOCATION_BINDING'):
            owner.require_prefix_fit_invocation_execution(inv,execution,source_repo=f['repo'],source_commit='a'*40)


@pytest.mark.parametrize('key,value',[('window_count',4),('optimizer_step_budget',1),
    ('expected_success_outcome','RESUMABLE_OR_COMPLETE'),('invocation_number',2)])
def test_recovery_invocation_cannot_expand(base_fixture,monkeypatch,key,value):
    f=base_fixture;recipe={**f['recipe'],'scoring_recovery_origin':{}}
    monkeypatch.setattr(owner,'require_prefix_fit_launch_recipe',lambda *a,**k:recipe)
    inv=copy.deepcopy(f['invocations'][0]);inv.update(window_count=1,optimizer_step_budget=0,expected_success_outcome='COMPLETE')
    inv[key]=value;execution=json.loads(Path(inv['execution_manifest']['path']).read_text())
    with pytest.raises(campaign.RandomAccessCampaignError):
        owner.require_prefix_fit_invocation_execution(inv,execution,source_repo=f['repo'],source_commit='a'*40)


def test_scoring_progress_keeps_lifetime_training_but_zero_new_budget(base_fixture):
    f=base_fixture;inv=copy.deepcopy(f['invocations'][0])
    inv.update(window_count=1,optimizer_step_budget=0,expected_success_outcome='COMPLETE',
        direction_prefix_fit_recipe=f['recipe_ref'],direction_prefix_fit_scoring_recovery=True)
    complete=cursor(f,parent_cursor=47602,optimizer_updates=2976,phase_state='COMPLETE',
        outcome='COMPLETE',oof_cursor=17288,final_train_measurement=f['dummy'])
    progress=owner.build_prefix_fit_progress(window=f['windows'][0],plan=f['plan'],invocation=inv,
        checkpoint_pointer={'path':inv['checkpoint']['pointer_path'],'sha256':'a'*64},cursor=complete)
    assert progress['optimizer_updates']==2976 and progress['oof_cursor']==17288
    progress['outcome']='RESUMABLE';progress['phase_state']='SCORING';reseal(progress,'progress_sha256')
    with pytest.raises(campaign.RandomAccessCampaignError,match='RECOVERY_PROGRESS_SCOPE'):
        owner.require_prefix_fit_progress(progress,plan_sha256=f['plan']['plan_sha256'],
            invocation=inv,expected_selection_receipt_sha256=None,verify_file=False)


def test_scoring_profile_rejects_reverse_and_step_before_call():
    # Execute only the small nested authority guard, never the native runner.
    source=Path(__file__).parents[1]/'gx1/scripts/run_entry_native_direction_prefix_fit_window_v1.py'
    tree=ast.parse(source.read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_native_direction_prefix_fit_window')
    guard=next(n for n in ast.walk(main) if isinstance(n,ast.FunctionDef) and n.name=='scoring_only_profile')
    optimizer_type=type('Optimizer',(),{})
    def require_ok(ok,reason):
        if not ok:raise RuntimeError(reason)
    calls=[];env={'torch':SimpleNamespace(optim=SimpleNamespace(Optimizer=optimizer_type)),
        '_require':require_ok,'legacy_profile':lambda *a:calls.append(a)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[guard],type_ignores=[])),'<synthetic-profile>','exec'),env)
    for name in ('_train_batch','_step','backward','grad','_engine_run_backward','_optimizer_step_with_finite_gradients','step'):
        frame=SimpleNamespace(f_code=SimpleNamespace(co_name=name),f_locals={'self':optimizer_type()})
        with pytest.raises(RuntimeError,match='RECOVERY_FORBIDDEN'):
            env['scoring_only_profile'](frame,'call',None)
    frame=SimpleNamespace(f_code=SimpleNamespace(co_name='_evaluate'),f_locals={})
    env['scoring_only_profile'](frame,'call',None)
    assert len(calls)==1


@pytest.mark.parametrize('fault', [None,'global_training','stage_updates','stage_backward',
    'checkpoint_reads','recovery_authority','repeat_claim'])
def test_readiness_opens_scoring_only(base_fixture,monkeypatch,fault):
    f=base_fixture;recipe={**f['recipe'],'scoring_recovery_origin':{}}
    monkeypatch.setattr(owner,'require_prefix_fit_launch_recipe',lambda *a,**k:recipe)
    policy=copy.deepcopy(f['policy']);gate=policy[recipe['admission_stage']]
    for key in ('training_enabled','native_training_enabled','full_epoch_training_allowed'):
        policy[key]=False;gate[key]=False
    for key in ('genuine_backward_authorized','genuine_updates_authorized','genuine_predictor_fit_authorized'):
        gate[key]=False
    gate.update(max_invocations=1,checkpoint_reads_authorized=True,zero_update_scoring_recovery_authorized=True)
    if fault=='global_training':policy['training_enabled']=True
    if fault=='stage_updates':gate['genuine_updates_authorized']=True
    if fault=='stage_backward':gate['genuine_backward_authorized']=True
    if fault=='checkpoint_reads':gate['checkpoint_reads_authorized']=False
    if fault=='recovery_authority':gate['zero_update_scoring_recovery_authorized']=False
    if fault=='repeat_claim':gate['claimed_invocations']=1
    write(f['repo']/'NEXT_RUN_POLICY.json',policy)
    def check():
        return owner.require_native_direction_prefix_fit_readiness(f['repo'],
            window_path=Path(f['window_refs'][0]['path']),window_file_sha256=f['window_refs'][0]['sha256'])
    if fault:
        with pytest.raises(campaign.RandomAccessCampaignError):check()
    else:
        scope=check()['native_window_scope']
        assert scope['max_windows']==1 and scope['optimizer_step_ceiling']==0
        assert scope['full_epoch_training_allowed'] is False


def test_read_guard_accepts_only_original_state_pair_and_rejects_writes(tmp_path,monkeypatch):
    import os
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    root=str(Path(owner.SCORING_RECOVERY_CLOSURE['path']).parent/'RUNTIME/prefix-fit/states')
    paths=(root+'/w0001_TERMINAL_u002976_p047602.pt',root+'/INITIAL_0U.pt')
    origin={'closure_review':owner.SCORING_RECOVERY_CLOSURE,'original_recipe':{'path':'/synthetic','sha256':'a'*64}}
    saved={'recipe':origin['original_recipe'],'parent_cursor':47602,'optimizer_updates':2976,'phase_state':'SCORING',
        'full_state':{'path':paths[0],'sha256':'b'*64},'own_initial_state':{'path':paths[1],'sha256':'c'*64}}
    context={'ancestor_plan':{'payloads':{},'native_cache_dir':'/synthetic/cache',
        'gc_surfaces':{'path':'/synthetic/gc/surface.npy'}},
        'info':{'metadata':{},'audit':{'manifest_path':'/synthetic/forbidden'}},
        'scoring_recovery_read_authority':{'launch':{'run_scope':owner.FAMILY_SIGN_SCOPE,
            'family_arm':'session_regime_encoder','scoring_recovery_origin':origin},
            'validated_origin':{'origin':origin,'cursor':saved}}}
    hooks=[];monkeypatch.setattr(common.sys,'addaudithook',hooks.append)
    common._install_read_guard(context,tmp_path/'output',tmp_path/'runtime',extra_read_paths=paths)
    assert len(hooks)==1
    hooks[0]('open',(paths[0],'rb',os.O_RDONLY))
    for path in paths:
        with pytest.raises(RuntimeError,match='ORIGINAL_WRITE'):
            hooks[0]('open',(path,'wb',os.O_WRONLY|os.O_TRUNC))
    with pytest.raises(RuntimeError,match='RECOVERY_EXACT_READ_PATHS'):
        common._install_read_guard(context,tmp_path/'output',tmp_path/'runtime',
            extra_read_paths=(paths[0],root+'/foreign.pt'))
