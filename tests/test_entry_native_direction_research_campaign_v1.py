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


def test_m1_price_internal_adapter_scope_cannot_authorize_private_research(fixture):
    from gx1.contracts.entry_native_direction_prefix_fit_campaign_v1 import M1_PRICE_PRIVATE_ADAPTER_SCOPE
    f=fixture;recipe=copy.deepcopy(f['recipe']);recipe['run_scope']=M1_PRICE_PRIVATE_ADAPTER_SCOPE
    reseal(recipe,'recipe_sha256');ref=write(f['runtime']/'PRIVATE_ADAPTER_RECIPE.json',recipe)
    with pytest.raises(RuntimeError,match='FIELDS'):
        owner.require_research_launch_recipe(ref,source_repo=f['repo'],source_commit='a'*40)


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


def chronological_private_fixture(f,monkeypatch):
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
    numeric={'schema_version':'gx1_native_direction_research_recipe_v1','batch_size':16,'max_updates':2,
        'genuine_updates_authorized':False,'native_launch_authorized':False,'research_objective_sha256':'e'*64}
    numeric['contract_sha256']=hashlib.sha256(json.dumps(numeric,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    original_numeric=copy.deepcopy(numeric);original_numeric['max_updates']=2976
    original_numeric.pop('contract_sha256')
    original_numeric['contract_sha256']=hashlib.sha256(json.dumps(original_numeric,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    old=write(f['runtime']/'ORIGINAL_ORDER_RECIPE.json',{'fold':owner.BPS_FOLD,'input_manifest':f['dummy'],
        'train_passes':1,'run_scope':'single_prefix_one_pass_direction_fit','numerical_recipe':original_numeric})
    initial=write(f['runtime']/'ORIGINAL_ORDER_INITIAL.json',{'role':'INITIAL_TRAIN','fold':owner.BPS_FOLD,'predictions':f['dummy']})
    origin={'launch_recipe':old,'initial_measurement':initial,'initial_predictions':f['dummy']}
    monkeypatch.setattr(prefix,'ORDER_ORIGIN',origin);monkeypatch.setattr(owner,'ORDER_DESIGN',f['dummy'])
    monkeypatch.setattr(owner,'BPS_GC_PARITY',f['dummy'])
    value=copy.deepcopy(f['recipe']);value.update(run_scope=owner.ORDER_PRIVATE_SCOPE,numerical_recipe=numeric,
        fold=copy.deepcopy(owner.BPS_FOLD),gc_layout_metadata_parity=f['dummy'],qualification_design=f['dummy'],
        max_model_constructions=3,max_forwards=21,max_backwards=4)
    for path in owner.ORDER_EXTRA_SOURCE_PATHS:value['source_owners'][path]=write(f['repo']/path,{'synthetic_source':True})
    reseal(value,'recipe_sha256')
    return value


def test_order_private_scope_matches_original_and_limits_work(fixture,monkeypatch):
    f=fixture;value=chronological_private_fixture(f,monkeypatch)
    ref=write(f['runtime']/'PRIVATE_ORDER.json',value)
    checked=owner.require_research_launch_recipe(ref,source_repo=f['repo'],source_commit='a'*40)
    assert checked['max_forwards']==21 and checked['max_private_updates']==4
    assert checked['full_epoch_training_allowed'] is False and checked['qualification_weights_for_main_fit'] is False


@pytest.mark.parametrize('fault',['forwards','constructors','backwards','updates','objective','fold','input','design','weights','test'])
def test_order_private_scope_rejects_expansion_or_different_baseline(fixture,monkeypatch,fault):
    f=fixture;value=chronological_private_fixture(f,monkeypatch)
    if fault=='forwards':value['max_forwards']=22
    elif fault=='constructors':value['max_model_constructions']=4
    elif fault=='backwards':value['max_backwards']=5
    elif fault=='updates':value['max_private_updates']=5
    elif fault=='objective':
        value['numerical_recipe']['signed_bps_supervision']=True
        unsigned={k:v for k,v in value['numerical_recipe'].items() if k!='contract_sha256'}
        value['numerical_recipe']['contract_sha256']=hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    elif fault=='fold':value['fold']['prefix_rows']+=1
    elif fault=='input':value['input_manifest']=write(f['runtime']/'OTHER_INPUT.json',{'synthetic':True})
    elif fault=='design':value['qualification_design']=write(f['runtime']/'OTHER_DESIGN.json',{'synthetic':True})
    elif fault=='weights':value['qualification_weights_for_main_fit']=True
    else:value['test_data_used']=True
    reseal(value,'recipe_sha256');ref=write(f['runtime']/'BAD_PRIVATE_ORDER.json',value)
    with pytest.raises(RuntimeError):owner.require_research_launch_recipe(ref,source_repo=f['repo'],source_commit='a'*40)


def test_supported_private_scope_is_distinct_and_requires_exact_bound_design(fixture,monkeypatch):
    f=fixture;value=chronological_private_fixture(f,monkeypatch)
    monkeypatch.setattr(owner,'ORDER_SUPPORTED_DESIGN',f['dummy'])
    value['run_scope']=owner.ORDER_SUPPORTED_SCOPE;value['qualification_design']=f['dummy'];reseal(value,'recipe_sha256')
    ref=write(f['runtime']/'SUPPORTED_ORDER.json',value)
    checked=owner.require_research_launch_recipe(ref,source_repo=f['repo'],source_commit='a'*40)
    assert checked['run_scope']==owner.ORDER_SUPPORTED_SCOPE and checked['max_private_updates']==4
    assert checked['full_epoch_training_allowed'] is False and checked['qualification_weights_for_main_fit'] is False
    value['qualification_design']=write(f['runtime']/'OTHER_SUPPORTED_DESIGN.json',{'synthetic':True});reseal(value,'recipe_sha256')
    ref=write(f['runtime']/'BAD_SUPPORTED_ORDER.json',value)
    with pytest.raises(RuntimeError):owner.require_research_launch_recipe(ref,source_repo=f['repo'],source_commit='a'*40)


def test_dense_private_recipe_requires_exact_scope_and_keeps_target_unopened(fixture,monkeypatch):
    f=fixture;repo=f['repo'];runtime=f['runtime'];dummy=f['dummy']
    target={'path':str(runtime/'not-opened.npz'),'sha256':'f'*64}
    proof=write(repo/'docs/DIRECTION_DENSE_SOURCE_TRANSITION_20261028.json',{'synthetic':True})
    monkeypatch.setattr(owner,'DENSE_INPUT_MANIFEST',dummy)
    monkeypatch.setattr(owner,'DENSE_TARGET_RESULT',dummy)
    monkeypatch.setattr(owner,'DENSE_TARGET',target)
    monkeypatch.setattr(owner,'BPS_GC_PARITY',dummy)
    monkeypatch.setattr(owner,'ORDER_SUPPORTED_DESIGN',dummy)
    monkeypatch.setattr(owner,'DENSE_EXTRA_SOURCE_PATHS',())
    item=copy.deepcopy(f['recipe']);item['run_scope']=owner.DENSE_PRIVATE_SCOPE
    item.update(fold=owner.BPS_FOLD,gc_layout_metadata_parity=dummy,qualification_design=dummy,
        dense_source_transition_proof=proof,dense_target_result=dummy,dense_target=target,max_forwards=21)
    numerical=item['numerical_recipe'];numerical['batch_size']=16;numerical['dense_minute_supervision']=True
    numerical.pop('contract_sha256');numerical['contract_sha256']=hashlib.sha256(json.dumps(numerical,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    reseal(item,'recipe_sha256');ref=write(runtime/'DENSE_RECIPE.json',item)
    result=owner.require_research_launch_recipe(ref,source_repo=repo,source_commit='a'*40)
    assert result['run_scope']==owner.DENSE_PRIVATE_SCOPE and not Path(target['path']).exists()
    for key,value in (('dense_target',{**target,'sha256':'0'*64}),('max_forwards',22),
                      ('full_epoch_training_allowed',True),('test_data_used',True)):
        wrong=copy.deepcopy(item);wrong[key]=value;reseal(wrong,'recipe_sha256')
        with pytest.raises(RuntimeError):
            owner.require_research_launch_recipe(write(runtime/'BAD_RECIPE.json',wrong),source_repo=repo,source_commit='a'*40)
    wrong=copy.deepcopy(item);wrong['numerical_recipe']['dense_minute_supervision']=False
    num=wrong['numerical_recipe'];num.pop('contract_sha256');num['contract_sha256']=hashlib.sha256(json.dumps(num,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    reseal(wrong,'recipe_sha256')
    with pytest.raises(RuntimeError):
        owner.require_research_launch_recipe(write(runtime/'BAD_RECIPE.json',wrong),source_repo=repo,source_commit='a'*40)


def candidate_price_fixture(f, monkeypatch, mode):
    """Bound metadata only; the raw candidate/target paths do not exist."""
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
    repo,runtime=f['repo'],f['runtime'];dummy=f['dummy']
    names={'owner_profile_source':'gx1/features/entry_specialist_feature_groups_v1.py',
        'native_constructor_source':'gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py',
        'direction_component_source':'gx1/contracts/entry_native_direction_research_v1.py'}
    owners={'source.py':f['source']}
    owners.update({name:write(repo/name,{'synthetic_source':name}) for name in names.values()})
    decimal_owner='gx1/contracts/entry_native_direction_decimal_input_v1.py'
    owners[decimal_owner]=write(repo/decimal_owner,{'synthetic_source':decimal_owner})
    original={key:{'path':owners[name]['path'],'sha256':'a'*64} for key,name in names.items()}
    manifest=write(runtime/'candidate-manifest.json',{'view_source_bindings':original})
    proof=write(repo/'docs/candidate-transition.json',{'schema_version':'gx1_dense_source_transition_proof_v1',
        'status':'SOURCE_ONLY_VERIFIED_NO_DATA_OR_TRAINING','test_data_used':False,'genuine_training_authorized':False,
        'input_manifest':manifest,'source_transitions':{key:{'original':original[key],'current':owners[name]}
            for key,name in names.items()}})
    original_target={'path':str(runtime/'original.npz'),'sha256':'1'*64}
    corrected={'path':str(runtime/'corrected.npz'),'sha256':'2'*64}
    orig_result=write(runtime/'original-result.json',{'synthetic':True})
    refs={'original':original_target,'original_result':orig_result,'corrected':corrected,'corrected_result':dummy}
    audit=write(runtime/'decimal-audit.json',{'schema_version':'gx1_decimal_bound_input_actual_audit_v1',
        'status':'COMPLETE','parents':118249,'horizons_minutes':list(range(1,61)),'test_data_used':False,
        'new_model_forwards':0,'optimizer_steps':0,'physical_parent_and_bar_start_hashes_verified':True,
        'corrected_archive_and_receipt_links_verified':True,'original_legacy_anchors_verified':True,
        'immutable_outputs':True,'inputs':refs,'source':owners[decimal_owner]})
    source={'path':str(runtime/'m1.npz'),'sha256':'3'*64}
    roll={'path':str(runtime/'roll.json'),'sha256':'4'*64}
    normal=write(runtime/'normalization.json',{'schema_version':'gx1_native_direction_m1_ofi_normalization_v1',
        'scope':'observed_closed_minutes_in_strict_train_prefix_only','contract_sha256':'5'*64,
        'prefix_rows':owner.BPS_FOLD['prefix_rows'],'cutoff_ns':owner.BPS_FOLD['train_cutoff_ns'],
        'roll_sha256':roll['sha256'],'model_contract_clock_bound':True,'genuine_fit_authorized':False,
        'native_launch_authorized':False})
    norm_plan=write(runtime/'normalization-plan.json',{'inputs':{'m1_source':source,'roll':roll}})
    norm_result=write(runtime/'normalization-result.json',{'status':'COMPLETE','plan':norm_plan,
        'train_parent_rows':owner.BPS_FOLD['prefix_rows'],'normalization_fit_calls':1,'target_values_used':False,
        'model_forwards':0,'optimizer_steps':0,'test_data_used':False,'normalization':normal,
        'normalization_contract_sha256':'5'*64})
    for key,value in {'REQUIRED_SOURCE_PATHS':tuple(owners),'DENSE_EXTRA_SOURCE_PATHS':(),
        'DECIMAL_EXTRA_SOURCE_PATHS':(),'M1_PRICE_EXTRA_SOURCE_PATHS':(),'DENSE_INPUT_MANIFEST':manifest,
        'DENSE_TARGET_RESULT':orig_result,'DENSE_TARGET':original_target,'BPS_GC_PARITY':dummy,
        'ORDER_SUPPORTED_DESIGN':dummy,'DECIMAL_INPUT_AUDIT':audit,'M1_PRICE_SOURCE_TRANSITION':proof,
        'M1_NORMALIZATION_PLAN':norm_plan,'M1_NORMALIZATION_RESULT':norm_result}.items():
        monkeypatch.setattr(owner,key,value)
    item=copy.deepcopy(f['recipe']);item.update(run_scope=prefix.M1_PRICE_PRIVATE_ADAPTER_SCOPE,
        input_manifest=manifest,source_owners=owners,fold=owner.BPS_FOLD,gc_layout_metadata_parity=dummy,
        qualification_design=dummy,dense_source_transition_proof=proof,dense_target_result=dummy,dense_target=corrected,
        decimal_input_audit=audit,decimal_target_refs=refs,max_forwards=21,candidate_mode=mode,
        parent_order_algorithm=prefix.PARENT_ORDER_ALGORITHM)
    numerical=item['numerical_recipe'];numerical['batch_size']=16;numerical['dense_minute_supervision']=True
    numerical.pop('contract_sha256');numerical['contract_sha256']=hashlib.sha256(json.dumps(numerical,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if mode in ('price_and_m1',prefix.M1_PRICE_COARSENED_MODE):item['candidate_m1_refs']={'source':source,'roll':roll,'normalization':normal}
    reseal(item,'recipe_sha256')
    return item


@pytest.mark.parametrize('mode',['price_only','price_and_m1','price_and_m1_coarsened'])
def test_candidate_qualification_recipe_binds_corrected_targets_and_exact_current_sources(fixture,monkeypatch,mode):
    f=fixture;recipe=candidate_price_fixture(f,monkeypatch,mode)
    checked=owner.require_research_launch_recipe(write(f['runtime']/'candidate-recipe.json',recipe),
        source_repo=f['repo'],source_commit='a'*40)
    assert checked['candidate_mode']==mode and checked['numerical_recipe']['dense_minute_supervision'] is True
    assert checked['qualification_weights_for_main_fit'] is False and checked['test_data_used'] is False
    assert not Path(checked['dense_target']['path']).exists()
    if mode!='price_only':assert not Path(checked['candidate_m1_refs']['source']['path']).exists()


@pytest.mark.parametrize('fault',['mode','order','m1_source','normalization','stale_constructor','wrong_proof',
    'max_updates','test','private_weights','dense_bps','extra_fields'])
def test_candidate_qualification_recipe_rejects_resealed_scope_or_binding_changes(fixture,monkeypatch,fault):
    f=fixture;recipe=candidate_price_fixture(f,monkeypatch,'price_and_m1')
    if fault=='mode':recipe['candidate_mode']='price_only'
    elif fault=='order':recipe['parent_order_algorithm']='unbound-order'
    elif fault=='m1_source':recipe['candidate_m1_refs']['source']['sha256']='9'*64
    elif fault=='normalization':recipe['candidate_m1_refs']['normalization']['sha256']='9'*64
    elif fault=='stale_constructor':recipe['source_owners']['gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py']['sha256']='9'*64
    elif fault=='wrong_proof':recipe['dense_source_transition_proof']=f['dummy']
    elif fault=='max_updates':recipe['max_private_updates']=5
    elif fault=='test':recipe['test_data_used']=True
    elif fault=='private_weights':recipe['qualification_weights_for_main_fit']=True
    elif fault=='dense_bps':
        numeric=recipe['numerical_recipe'];numeric['dense_signed_bps_supervision']=True;numeric.pop('contract_sha256')
        numeric['contract_sha256']=hashlib.sha256(json.dumps(numeric,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    else:recipe['unbound']='extra'
    reseal(recipe,'recipe_sha256')
    with pytest.raises(RuntimeError):owner.require_research_launch_recipe(write(f['runtime']/'bad-candidate.json',recipe),
        source_repo=f['repo'],source_commit='a'*40)


def test_candidate_pair_preflight_requires_same_data_recipe_and_sources(fixture,monkeypatch):
    f=fixture
    coarse=candidate_price_fixture(f,monkeypatch,'price_and_m1_coarsened')
    ordered=candidate_price_fixture(f,monkeypatch,'price_and_m1')
    ordered['admission_stage']+='b';ordered['runtime_root']=str(f['runtime']/'other')
    ordered['execution_binding_path']=str(f['runtime']/'other'/'EXECUTION_BINDING.json')
    reseal(ordered,'recipe_sha256')
    c=write(f['runtime']/'coarse-recipe.json',coarse);o=write(f['runtime']/'ordered-recipe.json',ordered)
    pair=owner.require_m1_structure_matched_recipes(c,o,source_repo=f['repo'],source_commit='a'*40)
    assert pair['coarsened']['candidate_mode']=='price_and_m1_coarsened' and pair['ordered']['candidate_mode']=='price_and_m1'
    wrong=copy.deepcopy(ordered);wrong['numerical_recipe']['research_objective_sha256']='f'*64
    num=wrong['numerical_recipe'];num.pop('contract_sha256');num['contract_sha256']=hashlib.sha256(json.dumps(num,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    reseal(wrong,'recipe_sha256');bad=write(f['runtime']/'bad-pair.json',wrong)
    with pytest.raises(RuntimeError,match='M1_PRICE_MATCHED_PAIR'):
        owner.require_m1_structure_matched_recipes(c,bad,source_repo=f['repo'],source_commit='a'*40)
    with pytest.raises(RuntimeError,match='M1_PRICE_MATCHED_PAIR'):
        owner.require_m1_structure_matched_recipes(o,c,source_repo=f['repo'],source_commit='a'*40)
