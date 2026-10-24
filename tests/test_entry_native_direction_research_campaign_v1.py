"""Synthetic metadata/transport checks; no native model or market inputs."""
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from gx1.contracts import entry_native_direction_research_campaign_v1 as owner
from gx1.contracts import local_random_access_campaign_v2 as campaign


def sealed(value, key):
    return {**value,key:campaign.canonical_sha256(value)}


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False))
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    repo=tmp_path/'source';repo.mkdir();runtime=tmp_path/'runtime';runtime.mkdir()
    monkeypatch.setattr(owner,'CANONICAL_REPO',repo)
    monkeypatch.setattr(owner,'REQUIRED_SOURCE_PATHS',('source.py',))
    source=write(repo/'source.py',{'synthetic':True});dummy=write(runtime/'input.json',{'synthetic':True})
    numerical={'schema_version':'gx1_native_direction_research_recipe_v1','batch_size':2,'max_updates':2,
        'genuine_updates_authorized':False,'native_launch_authorized':False,'research_objective_sha256':'e'*64}
    numerical['contract_sha256']=hashlib.sha256(json.dumps(numerical,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    recipe=sealed({'schema_version':owner.RECIPE_SCHEMA,'source_repo':str(repo),'runtime_root':str(runtime),
        'admission_stage':'dynamic_m5_direction_synthetic_272','run_scope':'private_four_update_resume_qualification',
        'numerical_recipe':numerical,'input_manifest':dummy,'source_owners':{'source.py':source},
        'execution_binding_path':str(runtime/'EXECUTION_BINDING.json'),'max_private_updates':4,
        'max_model_constructions':3,'max_forwards':4,'max_backwards':4,'max_checkpoint_bytes':1073741824,
        'max_published_json_bytes':33554432,'training_enabled':True,'full_epoch_training_allowed':False,
        'test_data_used':False,'predictive_value_claimed':False,'qualification_weights_for_main_fit':False},'recipe_sha256')
    recipe_ref=write(runtime/'RECIPE.json',recipe)
    window=sealed({'schema_version':owner.WINDOW_SCHEMA,'recipe':recipe_ref,'invocation_number':1,
        'max_invocation_seconds':12000,'progress_path':str(runtime/'progress/invocation-0001.json'),
        'campaign_cursor_path':str(runtime/'research-cursor/RESUME_CURSOR.json'),
        'output_directory':str(runtime/'private-research'),'test_data_used':False},'policy_sha256')
    window_ref=write(runtime/'WINDOW.json',window)
    argv=[str(repo/'scripts/gx1_capped_run.sh'),'--class','trainer','--mem','20G','--swap','512M','--',
        str(repo/'.venv/bin/python'),'-m',owner.MODULE,'--window-policy',window_ref['path'],
        '--window-policy-file-sha256',window_ref['sha256'],'--progress-path',window['progress_path']]
    execution=sealed({'schema_version':campaign.EXECUTION_MANIFEST_SCHEMA,'decision':'PASS_EXACT_INVOCATION',
        'invocation_kind':owner.KIND,'python_module':owner.MODULE,'source_commit':'a'*40,
        'launcher_argv_sha256':campaign.canonical_sha256(argv),'prelaunch_manifest':recipe_ref,
        'prelaunch_manifest_sha256':recipe['recipe_sha256'],'train_session_manifest':window_ref,
        'train_session_manifest_sha256':window['policy_sha256'],'test_data_used':False},'artifact_sha256')
    invocation=sealed({'schema_version':campaign.INVOCATION_SCHEMA,'decision':'PASS','invocation_number':1,
        'invocation_id':'invocation-0001','kind':owner.KIND,'batch_size':2,'epoch_index':None,
        'window_index':0,'window_count':1,'optimizer_step_budget':4,'expected_success_outcome':'COMPLETE',
        'source_commit':'a'*40,'execution_manifest':write(runtime/'EXECUTION.json',execution),
        'launcher_argv':argv,'launcher_argv_sha256':campaign.canonical_sha256(argv),'progress_path':window['progress_path'],
        'guard_log_path':str(runtime/'logs/invocation-0001.signed-guard.log'),
        'checkpoint':{'pointer_path':window['campaign_cursor_path'],'before_mode':'GENESIS',
            'predecessor_invocation_number':None,'write_mode':'ATOMIC_UPDATE'},'maximum_wall_seconds':13800,
        'requires_fresh_windows_boot':True,'signed_guard_only':True,'test_data_used':False},'invocation_sha256')
    boot=sealed({'schema_version':campaign.BOOT_SCHEMA,'computer_name':'SYNTHETIC','boot_id':1,
        'last_boot_utc':'2026-10-24T00:00:00+00:00'},'identity_sha256')
    plan=sealed({'schema_version':campaign.PLAN_SCHEMA,'decision':'PASS_PREPARED','campaign_id':'synthetic272',
        'phase':owner.PHASE,'created_utc':'2026-10-24T01:00:00+00:00','source_repo':str(repo),
        'source_commit':'a'*40,'runtime_root':str(runtime),'gpu_uuid':'GPU-synthetic','prepared_windows_boot':boot,
        'recipe':recipe_ref,'invocations':[write(runtime/'INVOCATION.json',invocation)],
        'signed_guard_sources':{k:dummy for k in ('runner','guard','query','certificate')},
        'controller_sources':{k:dummy for k in ('controller','observer','campaign_cli')},
        'policy':campaign.campaign_hardware_policy(owner.PHASE),
        'authority':{'test':False,'promotion':False,'paper':False,'live':False,'cloud_spend':False},
        'test_data_used':False},'plan_sha256')
    true=('execution_authorized','data_reads_authorized','genuine_constructor_authorized','genuine_forward_authorized',
        'genuine_backward_authorized','genuine_updates_authorized','training_enabled','native_training_enabled')
    false=('genuine_predictor_fit_authorized','native_launch_authorized','normalization_fit_authorized',
        'synthetic_test_execution_authorized','test_data_used','exit_evaluation','broker_access_allowed','relaunch_allowed')
    gate={**dict.fromkeys(true,True),**dict.fromkeys(false,False),'recipe':recipe_ref,'run_root':str(runtime),
        'max_invocations':1,'claimed_invocations':0,'completed_invocations':0}
    policy={'schema_version':'gx1_next_native_run_policy_v1','canonical_source_repo':str(repo),
        'canonical_branch':'work/gx1-current','training_module':owner.MODULE,'training_enabled':True,
        'native_training_enabled':True,'full_epoch_training_allowed':False,'gpu_clock_launcher':dummy,
        recipe['admission_stage']:gate}
    write(repo/'NEXT_RUN_POLICY.json',policy)
    def git(args,**kwargs):
        assert args[:3]==['git','-C',str(repo)]
        return {'rev-parse':'a'*40+'\n','branch':'work/gx1-current\n','status':''}[args[3]]
    monkeypatch.setattr(subprocess,'check_output',git)
    return locals()


def reseal(value,key):
    value[key]=campaign.canonical_sha256({k:v for k,v in value.items() if k!=key})
    return value


def test_complete_metadata_chain_without_legacy_supervision(fixture):
    f=fixture
    plan=campaign.require_plan(f['plan'])
    assert plan['selection_artifact_sha256'] is None
    assert plan['checked_invocations'][0]['direction_research_recipe']==f['recipe_ref']
    assert not {'transitions_per_epoch','entry_pairs_per_epoch','native_recipe'} & set(plan)
    ready=owner.require_native_direction_research_readiness(f['repo'],window_path=Path(f['window_ref']['path']),
        window_file_sha256=f['window_ref']['sha256'])
    assert ready['decision']=='READY_FOR_EXISTING_BOUND_CAMPAIGN_GATES'
    assert ready['native_window_scope']['full_epoch_training_allowed'] is False
    assert campaign.campaign_hardware_policy(owner.PHASE)==campaign.campaign_hardware_policy('native_candidate')


@pytest.mark.parametrize('key,value',[('requires_fresh_windows_boot',False),('signed_guard_only',False),
    ('batch_size',True),('optimizer_step_budget',5),('expected_success_outcome','RESUMABLE'),('kind','native_candidate_window')])
def test_resealed_invocation_cannot_broaden_scope(fixture,key,value):
    f=fixture;item=copy.deepcopy(f['invocation']);item[key]=value;reseal(item,'invocation_sha256')
    with pytest.raises((RuntimeError,ValueError)):
        campaign.require_invocation(item,source_repo=f['repo'],source_commit='a'*40)


@pytest.mark.parametrize('key,value',[('max_invocation_seconds',True),('invocation_number',2),
    ('test_data_used',True),('max_invocation_seconds',12001)])
def test_window_scope_is_not_free_timeout(fixture,key,value):
    item=copy.deepcopy(fixture['window']);item[key]=value;reseal(item,'policy_sha256')
    with pytest.raises(RuntimeError):owner.require_research_window_policy(item)


@pytest.mark.parametrize('key,value',[('training_enabled',False),('native_training_enabled',False),
    ('full_epoch_training_allowed',True)])
def test_global_stop_still_blocks_bound_window(fixture,key,value):
    f=fixture;policy=copy.deepcopy(f['policy']);policy[key]=value;write(f['repo']/'NEXT_RUN_POLICY.json',policy)
    with pytest.raises(RuntimeError):owner.require_native_direction_research_readiness(f['repo'],
        window_path=Path(f['window_ref']['path']),window_file_sha256=f['window_ref']['sha256'])


@pytest.mark.parametrize('key,value',[('claimed_invocations',1),('completed_invocations',1),
    ('genuine_predictor_fit_authorized',True),('broker_access_allowed',True),('genuine_updates_authorized',False),('max_invocations',True)])
def test_consumption_and_closed_gates_remain_binding(fixture,key,value):
    f=fixture;policy=copy.deepcopy(f['policy']);policy[f['recipe']['admission_stage']][key]=value
    write(f['repo']/'NEXT_RUN_POLICY.json',policy)
    with pytest.raises(RuntimeError):owner.require_native_direction_research_readiness(f['repo'],
        window_path=Path(f['window_ref']['path']),window_file_sha256=f['window_ref']['sha256'])


def test_byte_tamper_blocks_metadata_before_model(fixture):
    f=fixture;Path(f['source']['path']).write_text('changed')
    with pytest.raises(RuntimeError):campaign.require_plan(f['plan'])


def test_durable_research_progress_and_shared_receipt_validation(fixture):
    f=fixture;plan=campaign.require_plan(f['plan']);inv=plan['checked_invocations'][0]
    state_ref=write(f['runtime']/'STATE.synthetic.json',{'synthetic_no_model':True})
    result_ref=write(f['runtime']/'RESULT.json',{'synthetic_no_model':True})
    cursor=owner.build_research_cursor(recipe_ref=f['recipe_ref'],outcome='COMPLETE',optimizer_updates=4,
        parent_cursor=4,parent_order_sha256='d'*64,full_state=state_ref,result=result_ref)
    pointer=write(Path(f['window']['campaign_cursor_path']),cursor)
    progress=owner.build_research_progress(window=f['window'],plan=plan,invocation=inv,checkpoint_pointer=pointer,
        optimizer_updates=4,parent_cursor=4,parent_order_sha256='d'*64,outcome='COMPLETE')
    checked=campaign.require_progress(progress,plan_sha256=plan['plan_sha256'],invocation=inv,
        expected_selection_receipt_sha256=None)
    assert checked['parent_cursor']==4 and 'epoch_index' not in checked
    cursor['optimizer_updates']=3;reseal(cursor,'cursor_sha256');new_ref=write(Path(pointer['path']),cursor)
    progress['checkpoint_pointer']=new_ref;reseal(progress,'progress_sha256')
    with pytest.raises(RuntimeError):campaign.require_progress(progress,plan_sha256=plan['plan_sha256'],
        invocation=inv,expected_selection_receipt_sha256=None)


def test_common_campaign_still_requires_new_physical_boot(fixture):
    f=fixture;plan=campaign.require_plan(f['plan'])
    assert campaign.next_action(plan,[],current_boot=f['boot'])['decision']=='REBOOT_REQUIRED'
    newer=copy.deepcopy(f['boot']);newer['boot_id']=2;newer['last_boot_utc']='2026-10-24T02:00:00+00:00';reseal(newer,'identity_sha256')
    action=campaign.next_action(plan,[],current_boot=newer)
    assert action['decision']=='LAUNCH' and action['kind']==owner.KIND


def test_metadata_owner_imports_without_torch():
    import sys
    completed=subprocess.run([sys.executable,'-c',
        'import sys; import gx1.contracts.entry_native_direction_research_campaign_v1; assert "torch" not in sys.modules'],
        capture_output=True,text=True,timeout=30,check=False)
    assert completed.returncode==0,completed.stderr


@pytest.mark.parametrize('fault',[None,'environment','marker','policy_binding','existing_output'])
def test_context_binds_active_marker_frozen_policy_and_fresh_outputs(fixture,monkeypatch,fault):
    f=fixture;plan_ref=write(f['runtime']/'PLAN.json',f['plan'])
    binding={'plan':plan_ref,'source_head':'a'*40,'policy':{'path':str(f['repo']/'NEXT_RUN_POLICY.json'),
        'sha256':hashlib.sha256((f['repo']/'NEXT_RUN_POLICY.json').read_bytes()).hexdigest()}}
    marker=sealed({'schema_version':campaign.ACTIVE_SCHEMA,'plan_sha256':f['plan']['plan_sha256'],
        'invocation_sha256':f['invocation']['invocation_sha256'],'invocation_number':1,
        'invocation_id':'invocation-0001','kind':owner.KIND,
        'launcher_argv_sha256':f['invocation']['launcher_argv_sha256'],'pointer_before_sha256':'GENESIS'},'marker_sha256')
    if fault=='marker':marker['pointer_before_sha256']='d'*64;reseal(marker,'marker_sha256')
    if fault=='policy_binding':binding['policy']['sha256']='d'*64
    if fault=='existing_output':Path(f['window']['output_directory']).mkdir()
    write(f['runtime']/'EXECUTION_BINDING.json',binding);write(f['runtime']/'ACTIVE_INVOCATION.json',marker)
    for key,value in {'GX1_CAMPAIGN_PLAN_PATH':plan_ref['path'],'GX1_CAMPAIGN_PLAN_FILE_SHA256':plan_ref['sha256'],
        'GX1_CAMPAIGN_PLAN_SHA256':f['plan']['plan_sha256'],
        'GX1_CAMPAIGN_INVOCATION_SHA256':f['invocation']['invocation_sha256']}.items():monkeypatch.setenv(key,value)
    if fault=='environment':monkeypatch.setenv('GX1_CAMPAIGN_INVOCATION_SHA256','d'*64)
    clean=[]
    monkeypatch.setattr(campaign,'require_clean_source',lambda value:clean.append(value['plan_sha256']))
    def call():return owner.require_native_direction_research_context(Path(f['window_ref']['path']),
        f['window_ref']['sha256'],Path(f['window']['progress_path']))
    if fault is None:
        context=call();assert context[1]['plan_sha256']==f['plan']['plan_sha256'] and clean==[f['plan']['plan_sha256']]
    else:
        with pytest.raises(RuntimeError):call()


@pytest.mark.parametrize('fault',[False,True])
def test_native_window_dispatch_keeps_guard_first_and_closes_hash_scope(tmp_path,monkeypatch,fault):
    from gx1.scripts import run_unified_exit_native_candidate_window_v1 as native_window
    from gx1.scripts import run_entry_native_direction_research_window_v1 as research_runner
    ref=write(tmp_path/'WINDOW.json',{'schema_version':owner.WINDOW_SCHEMA})
    events=[]
    def guard(**kwargs):
        assert kwargs=={'execution_tier':'canonical'} and campaign._FILE_VERIFICATION_MEMO.get() is None
        events.append('guard')
    def execute(**kwargs):
        assert campaign._FILE_VERIFICATION_MEMO.get() is None
        assert kwargs=={'policy_path':Path(ref['path']),'policy_file_sha256':ref['sha256'],'progress_path':tmp_path/'PROGRESS.json'}
        events.append('research');return {'synthetic_dispatch_only':True}
    monkeypatch.setattr(native_window.native.trainer,'_require_cuda_trainer_guard_execution',guard)
    monkeypatch.setattr(research_runner,'run_native_direction_research_window',execute)
    def call():return native_window.run_window(policy_path=Path(ref['path']),
        policy_file_sha256='0'*64 if fault else ref['sha256'],progress_path=tmp_path/'PROGRESS.json')
    if fault:
        with pytest.raises(RuntimeError):call()
        assert events==['guard']
    else:
        assert call()=={'synthetic_dispatch_only':True} and events==['guard','research']
    assert campaign._FILE_VERIFICATION_MEMO.get() is None
