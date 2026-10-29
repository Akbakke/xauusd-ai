"""Synthetic metadata only: no native construction, data, loss, or optimizer."""
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as owner
from gx1.contracts import local_random_access_campaign_v2 as campaign


def sealed(value, key):
    return {**value, key:campaign.canonical_sha256(value)}


def reseal(value, key):
    value[key] = campaign.canonical_sha256({k:v for k,v in value.items() if k != key})
    return value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))
    return {'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def numerical_seal(value):
    value.pop('contract_sha256', None)
    value['contract_sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return value


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    repo=tmp_path/'source';repo.mkdir();runtime=tmp_path/'runtime';runtime.mkdir()
    monkeypatch.setattr(owner, 'CANONICAL_REPO', repo)
    monkeypatch.setattr(owner, 'REQUIRED_SOURCE_PATHS', ('source.py',))
    source=write(repo/'source.py', {'synthetic_source':True})
    dummy=write(runtime/'input.json', {'synthetic_metadata':True})
    fold={'fold_id':1,'prefix_rows':47602,'oof_start_row':47602,'oof_end_row':64890,
        'train_cutoff_ns':1735689600000000000,'oof_cutoff_ns':1743465600000000000}
    prefix_record={'fold_id':1,'prefix_parent_rows':47602,'cutoff_ns':fold['train_cutoff_ns'],'output':dummy,
        'base_normalization_sha256':'2'*64,'combined_contract_sha256':'1'*64,'gc_layout_sha256':'3'*64,
        'gc_normalization_sha256':'4'*64,'fresh_support_report':dummy,'original_support_report':dummy}
    input_ref=write(runtime/'INPUT_MANIFEST.json',{'schema_version':owner.INPUT_SCHEMA,'source_design':dummy,
        'physical_provider_plan':dummy,'physical_provider_worker':dummy,'prefix_normalization':prefix_record,
        'constructor_metadata':dummy,'view_source_bindings':dict.fromkeys(owner.VIEW_SOURCE_KEYS,dummy),'targets':dummy,
        'archive_members':sorted(owner.ARCHIVE_MEMBERS),'state_qualification':dummy,
        'native_private_qualification':dummy,'input_qualification':dummy})
    numerical=numerical_seal({'schema_version':'gx1_native_direction_research_recipe_v1','batch_size':16,
        'max_updates':2976,'genuine_updates_authorized':False,'native_launch_authorized':False,
        'research_objective_sha256':'e'*64})
    recipe=sealed({'schema_version':owner.RECIPE_SCHEMA,'source_repo':str(repo),'runtime_root':str(runtime),
        'admission_stage':'dynamic_m5_direction_synthetic_prefix_275','run_scope':'single_prefix_one_pass_direction_fit',
        'numerical_recipe':numerical,'input_manifest':input_ref,'source_owners':{'source.py':source},
        'execution_binding_path':str(runtime/'EXECUTION_BINDING.json'),'fold':fold,'train_passes':1,
        'max_windows':4,'parent_order_algorithm':owner.PARENT_ORDER_ALGORITHM,'initial_measurement_rows':256,
        'eval_batch_size':16,'checkpoint_every_updates':256,'max_checkpoint_bytes':1073741824,
        'max_checkpoint_total_bytes':4294967296,'max_published_json_bytes':67108864,'training_enabled':True,
        'full_epoch_training_allowed':True,'test_data_used':False,'predictive_value_claimed':False,
        'qualification_weights_for_main_fit':False}, 'recipe_sha256')
    recipe_ref=write(runtime/'RECIPE.json', recipe)
    windows=[];window_refs=[];invocations=[];invocation_refs=[]
    for number in range(1,5):
        window=sealed({'schema_version':owner.WINDOW_SCHEMA,'recipe':recipe_ref,'invocation_number':number,
            'max_invocation_seconds':12000,'progress_path':str(runtime/'progress'/f'invocation-{number:04d}.json'),
            'campaign_cursor_path':str(runtime/'prefix-cursor/RESUME_CURSOR.json'),
            'output_directory':str(runtime/'prefix-fit'),'test_data_used':False}, 'policy_sha256')
        window_ref=write(runtime/f'WINDOW_{number}.json', window)
        argv=[str(repo/'scripts/gx1_capped_run.sh'),'--class','trainer','--mem','20G','--swap','512M','--',
            str(repo/'.venv/bin/python'),'-m',owner.MODULE,'--window-policy',window_ref['path'],
            '--window-policy-file-sha256',window_ref['sha256'],'--progress-path',window['progress_path']]
        execution=sealed({'schema_version':campaign.EXECUTION_MANIFEST_SCHEMA,'decision':'PASS_EXACT_INVOCATION',
            'invocation_kind':owner.KIND,'python_module':owner.MODULE,'source_commit':'a'*40,
            'launcher_argv_sha256':campaign.canonical_sha256(argv),'prelaunch_manifest':recipe_ref,
            'prelaunch_manifest_sha256':recipe['recipe_sha256'],'train_session_manifest':window_ref,
            'train_session_manifest_sha256':window['policy_sha256'],'test_data_used':False}, 'artifact_sha256')
        invocation=sealed({'schema_version':campaign.INVOCATION_SCHEMA,'decision':'PASS','invocation_number':number,
            'invocation_id':f'invocation-{number:04d}','kind':owner.KIND,'batch_size':16,'epoch_index':None,
            'window_index':number-1,'window_count':4,'optimizer_step_budget':2976,
            'expected_success_outcome':'RESUMABLE_OR_COMPLETE','source_commit':'a'*40,
            'execution_manifest':write(runtime/f'EXECUTION_{number}.json',execution),'launcher_argv':argv,
            'launcher_argv_sha256':campaign.canonical_sha256(argv),'progress_path':window['progress_path'],
            'guard_log_path':str(runtime/'logs'/f'invocation-{number:04d}.signed-guard.log'),
            'checkpoint':{'pointer_path':window['campaign_cursor_path'],
                'before_mode':'GENESIS' if number==1 else 'PREVIOUS_RECEIPT_AFTER',
                'predecessor_invocation_number':None if number==1 else number-1,'write_mode':'ATOMIC_UPDATE'},
            'maximum_wall_seconds':13800,'requires_fresh_windows_boot':True,'signed_guard_only':True,
            'test_data_used':False}, 'invocation_sha256')
        windows.append(window);window_refs.append(window_ref);invocations.append(invocation)
        invocation_refs.append(write(runtime/f'INVOCATION_{number}.json', invocation))
    boot=sealed({'schema_version':campaign.BOOT_SCHEMA,'computer_name':'SYNTHETIC','boot_id':1,
        'last_boot_utc':'2026-10-25T00:00:00+00:00'}, 'identity_sha256')
    plan=sealed({'schema_version':campaign.PLAN_SCHEMA,'decision':'PASS_PREPARED','campaign_id':'synthetic275fold1',
        'phase':owner.PHASE,'created_utc':'2026-10-25T01:00:00+00:00','source_repo':str(repo),'source_commit':'a'*40,
        'runtime_root':str(runtime),'gpu_uuid':'GPU-synthetic','prepared_windows_boot':boot,'recipe':recipe_ref,
        'invocations':invocation_refs,'signed_guard_sources':{k:dummy for k in ('runner','guard','query','certificate')},
        'controller_sources':{k:dummy for k in ('controller','observer','campaign_cli')},
        'policy':campaign.campaign_hardware_policy(owner.PHASE),
        'authority':{'test':False,'promotion':False,'paper':False,'live':False,'cloud_spend':False},
        'test_data_used':False}, 'plan_sha256')
    true=('execution_authorized','data_reads_authorized','genuine_constructor_authorized','genuine_forward_authorized',
        'genuine_backward_authorized','genuine_updates_authorized','genuine_predictor_fit_authorized',
        'training_enabled','native_training_enabled','full_epoch_training_allowed','native_launch_authorized')
    false=('normalization_fit_authorized','synthetic_test_execution_authorized','test_data_used','exit_evaluation',
        'broker_access_allowed','full_val_allowed','relaunch_allowed')
    gate={**dict.fromkeys(true,True),**dict.fromkeys(false,False),'recipe':recipe_ref,'run_root':str(runtime),
        'max_invocations':4,'claimed_invocations':0,'completed_invocations':0}
    policy={'schema_version':'gx1_next_native_run_policy_v1','canonical_source_repo':str(repo),
        'canonical_branch':'work/gx1-current','training_module':owner.MODULE,'training_enabled':True,
        'native_training_enabled':True,'full_epoch_training_allowed':True,'full_val_allowed':False,
        'gpu_clock_launcher':dummy,recipe['admission_stage']:gate}
    write(repo/'NEXT_RUN_POLICY.json', policy)
    def git(args, **kwargs):
        assert args[:3]==['git','-C',str(repo)]
        return {'rev-parse':'a'*40+'\n','branch':'work/gx1-current\n','status':''}[args[3]]
    monkeypatch.setattr(subprocess, 'check_output', git)
    prefix_identity={'binding':dummy,'contract_sha256':'1'*64,'base_sha256':'2'*64,
        'gc_layout_sha256':'3'*64,'gc_normalization_sha256':'4'*64,'qualification_sha256':'5'*64}
    view_identity={'contract_sha256':'6'*64,'prefix_normalization':prefix_identity,'research_objective_sha256':'e'*64}
    return locals()


def cursor(f, **changes):
    args={'recipe_ref':f['recipe_ref'],'fold':f['fold'],'feature_view_identity':f['view_identity'],
        'phase_state':'TRAINING','outcome':'RESUMABLE','optimizer_updates':1,'parent_cursor':32,
        'parent_order_sha256':'d'*64,'full_state':f['dummy'],'own_initial_state':f['dummy'],
        'initial_measurement':f['dummy'],'final_train_measurement':None,'oof_cursor':0,'result':f['dummy']}
    args.update(changes)
    return owner.build_prefix_fit_cursor(**args)


def test_four_window_chain_is_one_exact_prefix_and_one_pass(fixture):
    f=fixture;plan=campaign.require_plan(f['plan'])
    assert len(plan['checked_invocations'])==4 and plan['selection_artifact_sha256'] is None
    assert [v['checkpoint']['before_mode'] for v in plan['checked_invocations']]==['GENESIS']+['PREVIOUS_RECEIPT_AFTER']*3
    assert all(v['direction_prefix_fit_recipe']==f['recipe_ref'] for v in plan['checked_invocations'])
    assert not {'entry_pairs_per_epoch','transitions_per_epoch','native_recipe'} & set(plan)
    assert campaign.campaign_hardware_policy(owner.PHASE)==campaign.campaign_hardware_policy('native_candidate')
    ready=owner.require_native_direction_prefix_fit_readiness(f['repo'],window_path=Path(f['window_refs'][0]['path']),
        window_file_sha256=f['window_refs'][0]['sha256'])
    assert ready['native_window_scope']['fold']==f['fold'] and ready['native_window_scope']['full_epoch_training_allowed'] is True


@pytest.mark.parametrize('key,value',[('invocation_number',True),('invocation_number',5),('max_invocation_seconds',12001),('test_data_used',True)])
def test_bound_window_cannot_expand_limits(fixture,key,value):
    value_dict=copy.deepcopy(fixture['windows'][0]);value_dict[key]=value;reseal(value_dict,'policy_sha256')
    with pytest.raises(RuntimeError):owner.require_prefix_fit_window_policy(value_dict)


@pytest.mark.parametrize('fault',['fold','batch','updates','order','passes','qualification_weights'])
def test_recipe_rejects_resealed_scope_mutations(fixture,fault):
    f=fixture;recipe=copy.deepcopy(f['recipe'])
    if fault=='fold':recipe['fold']['prefix_rows']+=1
    elif fault=='batch':recipe['numerical_recipe']['batch_size']=2
    elif fault=='updates':recipe['numerical_recipe']['max_updates']=4
    elif fault=='order':recipe['parent_order_algorithm']='last_emitted_or_sampled'
    elif fault=='passes':recipe['train_passes']=2
    else:recipe['qualification_weights_for_main_fit']=True
    numerical_seal(recipe['numerical_recipe']);reseal(recipe,'recipe_sha256');ref=write(f['runtime']/'BAD_RECIPE.json',recipe)
    with pytest.raises(RuntimeError):owner.require_prefix_fit_launch_recipe(ref,source_repo=f['repo'],source_commit='a'*40)


def test_m1_price_internal_adapter_scope_cannot_authorize_prefix_training(fixture):
    f=fixture;recipe=copy.deepcopy(f['recipe']);recipe['run_scope']=owner.M1_PRICE_PRIVATE_ADAPTER_SCOPE
    reseal(recipe,'recipe_sha256');ref=write(f['runtime']/'PRIVATE_ADAPTER_RECIPE.json',recipe)
    with pytest.raises(RuntimeError,match='RECIPE_SCOPE'):
        owner.require_prefix_fit_launch_recipe(ref,source_repo=f['repo'],source_commit='a'*40)


@pytest.mark.parametrize('key,value',[('batch_size',8),('epoch_index',0),('window_count',5),('optimizer_step_budget',4),
    ('expected_success_outcome','COMPLETE'),('requires_fresh_windows_boot',False)])
def test_resealed_invocation_keeps_fixed_budget_and_guards(fixture,key,value):
    f=fixture;item=copy.deepcopy(f['invocations'][0]);item[key]=value;reseal(item,'invocation_sha256')
    with pytest.raises((RuntimeError,ValueError)):campaign.require_invocation(item,source_repo=f['repo'],source_commit='a'*40)


@pytest.mark.parametrize('scope,key,value',[('global','training_enabled',False),('global','full_epoch_training_allowed',False),
    ('global','full_val_allowed',True),('gate','genuine_predictor_fit_authorized',False),('gate','normalization_fit_authorized',True),
    ('gate','broker_access_allowed',True),('gate','claimed_invocations',1),('gate','max_invocations',True)])
def test_readiness_needs_actual_fullfit_admission(fixture,scope,key,value):
    f=fixture;policy=copy.deepcopy(f['policy']);target=policy if scope=='global' else policy[f['recipe']['admission_stage']]
    target[key]=value;write(f['repo']/'NEXT_RUN_POLICY.json',policy)
    with pytest.raises(RuntimeError):owner.require_native_direction_prefix_fit_readiness(f['repo'],
        window_path=Path(f['window_refs'][0]['path']),window_file_sha256=f['window_refs'][0]['sha256'])


def test_cursor_counts_empty_batches_without_fabricating_optimizer_updates(fixture):
    f=fixture;c=cursor(f)
    assert c['parent_cursor']==32 and c['optimizer_updates']==1
    pointer=write(Path(f['windows'][0]['campaign_cursor_path']),c)
    inv=campaign.require_plan(f['plan'])['checked_invocations'][0]
    progress=owner.build_prefix_fit_progress(window=f['windows'][0],plan=f['plan'],invocation=inv,
        checkpoint_pointer=pointer,cursor=c)
    checked=campaign.require_progress(progress,plan_sha256=f['plan']['plan_sha256'],invocation=inv,
        expected_selection_receipt_sha256=None)
    assert checked['completed_units']==32 and checked['total_units']==47602 and checked['optimizer_updates']==1
    c['parent_order_sha256']='b'*64;reseal(c,'cursor_sha256');progress['checkpoint_pointer']=write(Path(pointer['path']),c)
    reseal(progress,'progress_sha256')
    with pytest.raises(RuntimeError):campaign.require_progress(progress,plan_sha256=f['plan']['plan_sha256'],
        invocation=inv,expected_selection_receipt_sha256=None)


@pytest.mark.parametrize('fault',['mid_batch','too_many_updates','early_scoring','early_oof','missing_initial',
    'complete_before_score','wrong_fold','partial_epoch_alias'])
def test_cursor_rejects_inconsistent_resealed_trajectory(fixture,fault):
    f=fixture;c=cursor(f)
    if fault=='mid_batch':c['parent_cursor']=17
    elif fault=='too_many_updates':c['optimizer_updates']=3
    elif fault=='early_scoring':c['phase_state']='SCORING'
    elif fault=='early_oof':c['oof_cursor']=1
    elif fault=='missing_initial':c['own_initial_state']=None
    elif fault=='complete_before_score':c.update(parent_cursor=47602,phase_state='COMPLETE',outcome='COMPLETE',final_train_measurement=f['dummy'])
    elif fault=='wrong_fold':c['fold']['train_cutoff_ns']+=1
    else:c['epoch_index']=0
    reseal(c,'cursor_sha256')
    with pytest.raises(RuntimeError):owner.require_prefix_fit_cursor(c,expected_recipe=f['recipe_ref'],expected_fold=f['fold'])


def test_scoring_resume_and_complete_preserve_final_fixed_state(fixture):
    f=fixture;c=cursor(f,parent_cursor=47602,optimizer_updates=2000,phase_state='SCORING',final_train_measurement=f['dummy'])
    assert c['outcome']=='RESUMABLE' and c['oof_cursor']==0
    done=cursor(f,parent_cursor=47602,optimizer_updates=2000,phase_state='COMPLETE',outcome='COMPLETE',
        final_train_measurement=f['dummy'],oof_cursor=17288)
    assert done['full_state']==c['full_state'] and done['own_initial_state']==c['own_initial_state']
    failed=cursor(f,phase_state='TRAINING',outcome='FAILED',parent_cursor=0,optimizer_updates=0,
        full_state=None,own_initial_state=None,initial_measurement=None,feature_view_identity=None)
    assert failed['full_state'] is None


@pytest.mark.parametrize('complete',[True,False])
def test_shared_next_action_stops_complete_and_blocks_exhausted_resumable(fixture,monkeypatch,complete):
    f=fixture;plan=campaign.require_plan(f['plan'])
    receipts=[{'outcome':'COMPLETE'}] if complete else [{'outcome':'RESUMABLE'}]*4
    monkeypatch.setattr(campaign,'require_receipt_chain',lambda *args,**kwargs:receipts)
    action=campaign.next_action(plan,[],current_boot=f['boot'])
    assert action['decision']==('COMPLETE' if complete else 'BLOCKED_DIRECTION_PREFIX_WINDOWS_EXHAUSTED')


@pytest.mark.parametrize('fault',[None,'marker','policy_binding','existing_output'])
def test_context_checks_genesis_active_and_exact_frozen_binding(fixture,monkeypatch,fault):
    f=fixture;plan_ref=write(f['runtime']/'PLAN.json',f['plan']);inv=f['invocations'][0]
    binding={'plan':plan_ref,'source_head':'a'*40,'policy':{'path':str(f['repo']/'NEXT_RUN_POLICY.json'),
        'sha256':hashlib.sha256((f['repo']/'NEXT_RUN_POLICY.json').read_bytes()).hexdigest()}}
    marker=sealed({'schema_version':campaign.ACTIVE_SCHEMA,'plan_sha256':f['plan']['plan_sha256'],
        'invocation_sha256':inv['invocation_sha256'],'invocation_number':1,'invocation_id':'invocation-0001',
        'kind':owner.KIND,'launcher_argv_sha256':inv['launcher_argv_sha256'],'pointer_before_sha256':'GENESIS'},'marker_sha256')
    if fault=='marker':marker['pointer_before_sha256']='d'*64;reseal(marker,'marker_sha256')
    if fault=='policy_binding':binding['policy']['sha256']='d'*64
    if fault=='existing_output':Path(f['windows'][0]['output_directory']).mkdir()
    write(f['runtime']/'EXECUTION_BINDING.json',binding);write(f['runtime']/'ACTIVE_INVOCATION.json',marker)
    for key,value in {'GX1_CAMPAIGN_PLAN_PATH':plan_ref['path'],'GX1_CAMPAIGN_PLAN_FILE_SHA256':plan_ref['sha256'],
        'GX1_CAMPAIGN_PLAN_SHA256':f['plan']['plan_sha256'],'GX1_CAMPAIGN_INVOCATION_SHA256':inv['invocation_sha256']}.items():monkeypatch.setenv(key,value)
    monkeypatch.setattr(campaign,'require_clean_source',lambda value:None)
    def call():return owner.require_native_direction_prefix_fit_context(Path(f['window_refs'][0]['path']),
        f['window_refs'][0]['sha256'],Path(f['windows'][0]['progress_path']))
    if fault is None:
        result=call();assert result[0]==f['windows'][0] and result[4]['fold']==f['fold']
    else:
        with pytest.raises(RuntimeError):call()


def test_source_byte_tamper_and_duplicate_invocations_fail_closed(fixture):
    f=fixture;plan=copy.deepcopy(f['plan']);plan['invocations'][1]=plan['invocations'][0];reseal(plan,'plan_sha256')
    with pytest.raises(RuntimeError):owner.require_prefix_fit_plan(plan)
    Path(f['source']['path']).write_text('changed')
    with pytest.raises(RuntimeError):campaign.require_plan(f['plan'])


def test_metadata_import_has_no_numerical_dependency():
    import sys
    done=subprocess.run([sys.executable,'-c','import sys; import gx1.contracts.entry_native_direction_prefix_fit_campaign_v1; '
        'assert not {"torch","numpy","sklearn"} & set(sys.modules)'],capture_output=True,text=True,timeout=30,check=False)
    assert done.returncode==0,done.stderr


def bps_fixture(f, monkeypatch):
    source=write(f['repo']/'bps_source.py', {'qualified_numerical_source':True})
    monkeypatch.setattr(owner,'BPS_EXTRA_SOURCE_PATHS',('bps_source.py',))
    monkeypatch.setattr(owner,'BPS_NUMERICAL_SOURCE_PATHS',('bps_source.py',))
    qualified_recipe=write(f['runtime']/'qualified_recipe.json',{'source_owners':{'bps_source.py':source}})
    proof={'status':'PASS_PRIVATE_NATIVE_MECHANICS_CONSUMED',
        'matched_shared_parameters_buffers_rng_initial_direction_predictions':True,
        'bit_exact_native_resume':True,'recipe_source_pins_verified':True,'private_weights_for_main_fit':False,
        'prefix_rows':47602,'prefix_cutoff_ns':f['fold']['train_cutoff_ns'],
        'recipe':qualified_recipe,'own_prefix_bps_scale':f['dummy']}
    proof_ref=write(f['runtime']/'qualified_bps.json',proof)
    monkeypatch.setattr(owner,'BPS_NATIVE_QUALIFICATION',proof_ref)
    monkeypatch.setattr(owner.private,'BPS_GC_PARITY',f['dummy'])
    recipe=copy.deepcopy(f['recipe'])
    recipe.update(run_scope=owner.BPS_SCOPE,native_bps_qualification=proof_ref,gc_layout_metadata_parity=f['dummy'])
    recipe['source_owners']['bps_source.py']=source
    recipe['numerical_recipe']['signed_bps_supervision']=True
    numerical_seal(recipe['numerical_recipe']);reseal(recipe,'recipe_sha256')
    return recipe,proof


def test_qualified_bps_prefix_keeps_one_pass_budget_and_current_numerical_sources(fixture,monkeypatch):
    f=fixture;recipe,_=bps_fixture(f,monkeypatch)
    ref=write(f['runtime']/'BPS_RECIPE.json',recipe)
    actual=owner.require_prefix_fit_launch_recipe(ref,source_repo=f['repo'],source_commit='a'*40)
    assert actual['fold']==f['fold'] and actual['max_windows']==4
    assert actual['numerical_recipe']['max_updates']==2976 and actual['train_passes']==1


@pytest.mark.parametrize('fault',['old_scope','false_mode','nonboolean_mode','later_fold','gc_proof',
    'missing_native_proof','unqualified_resume','changed_qualified_numerics','missing_source'])
def test_bps_recipe_rejects_unqualified_or_expanded_scope(fixture,monkeypatch,fault):
    f=fixture;recipe,proof=bps_fixture(f,monkeypatch)
    if fault=='old_scope':recipe['run_scope']=f['recipe']['run_scope']
    elif fault=='false_mode':recipe['numerical_recipe']['signed_bps_supervision']=False
    elif fault=='nonboolean_mode':recipe['numerical_recipe']['signed_bps_supervision']=1
    elif fault=='later_fold':
        n,end,cut,oof=owner.FOLD_SPECS[2]
        recipe['fold']={'fold_id':2,'prefix_rows':n,'oof_start_row':n,'oof_end_row':end,
            'train_cutoff_ns':cut,'oof_cutoff_ns':oof}
    elif fault=='gc_proof':recipe['gc_layout_metadata_parity']=recipe['native_bps_qualification']
    elif fault=='missing_native_proof':del recipe['native_bps_qualification']
    elif fault=='unqualified_resume':
        proof['bit_exact_native_resume']=False
        ref=write(f['runtime']/'unqualified_bps.json',proof)
        recipe['native_bps_qualification']=ref;monkeypatch.setattr(owner,'BPS_NATIVE_QUALIFICATION',ref)
    elif fault=='changed_qualified_numerics':
        recipe['source_owners']['bps_source.py']=write(f['repo']/'bps_source.py',{'different_numerics':True})
    else:del recipe['source_owners']['bps_source.py']
    numerical_seal(recipe['numerical_recipe']);reseal(recipe,'recipe_sha256')
    ref=write(f['runtime']/'BAD_BPS_RECIPE.json',recipe)
    with pytest.raises(RuntimeError):
        owner.require_prefix_fit_launch_recipe(ref,source_repo=f['repo'],source_commit='a'*40)


def test_old_prefix_scope_cannot_silently_enable_signed_bps(fixture):
    f=fixture;recipe=copy.deepcopy(f['recipe']);recipe['numerical_recipe']['signed_bps_supervision']=True
    numerical_seal(recipe['numerical_recipe']);reseal(recipe,'recipe_sha256')
    ref=write(f['runtime']/'UNADMITTED_BPS_RECIPE.json',recipe)
    with pytest.raises(RuntimeError,match='NUMERICAL_OBJECTIVE_SCOPE'):
        owner.require_prefix_fit_launch_recipe(ref,source_repo=f['repo'],source_commit='a'*40)


@pytest.mark.parametrize('scale_authorized',[False,True])
def test_bps_readiness_requires_explicit_prefix_scale_authority(fixture,monkeypatch,scale_authorized):
    f=fixture;recipe,_=bps_fixture(f,monkeypatch);ref=write(f['runtime']/'BPS_RECIPE.json',recipe)
    window=copy.deepcopy(f['windows'][0]);window['recipe']=ref;reseal(window,'policy_sha256')
    window_ref=write(f['runtime']/'BPS_WINDOW.json',window)
    policy=copy.deepcopy(f['policy']);gate=policy[recipe['admission_stage']]
    gate.update(recipe=ref,signed_bps_scale_fit_authorized=scale_authorized)
    write(f['repo']/'NEXT_RUN_POLICY.json',policy)
    def check():return owner.require_native_direction_prefix_fit_readiness(f['repo'],
        window_path=Path(window_ref['path']),window_file_sha256=window_ref['sha256'])
    if scale_authorized:assert check()['decision']=='READY_FOR_EXISTING_BOUND_CAMPAIGN_GATES'
    else:
        with pytest.raises(RuntimeError,match='READINESS_BPS_SCALE'):check()


def continuation_fixture(f, monkeypatch):
    frozen_path='gx1/contracts/entry_native_direction_frozen_extraction_v1.py'
    frozen=write(f['repo']/frozen_path,{'synthetic_frozen_source':True})
    initial=write(f['runtime']/'INITIAL_FAKE.json',{'role':'INITIAL_TRAIN','fold':f['fold'],'own_initial_state':f['dummy']})
    final=write(f['runtime']/'FINAL_FAKE.json',{'role':'FINAL_TRAIN','fold':f['fold'],'own_initial_state':f['dummy']})
    origin={'launch_recipe':f['recipe_ref'],'terminal_state':f['dummy'],'initial_measurement':initial,
        'initial_state':f['dummy'],'terminal_train_measurement':final}
    monkeypatch.setattr(owner,'CONTINUATION_ORIGIN',origin)
    monkeypatch.setattr(owner.private,'BPS_GC_PARITY',f['dummy'])
    value=copy.deepcopy(f['recipe']);value['run_scope']=owner.CONTINUATION_SCOPE;value['train_passes']=3
    value['numerical_recipe']['max_updates']=8928;numerical_seal(value['numerical_recipe'])
    value['source_owners'][frozen_path]=frozen
    value['continuation_origin']=copy.deepcopy(origin);value['gc_layout_metadata_parity']=f['dummy']
    proof={'status':'PASS_ORIGINAL_NATIVE_CONTINUATION_ZERO_UPDATE','origin':copy.deepcopy(origin),'source_owners':copy.deepcopy(value['source_owners']),
        'numerical_recipe_sha256':value['numerical_recipe']['contract_sha256'],'checkpoint_state_parity':True,
        'terminal_train_prediction_bit_parity':True,'optimizer_steps':0,'test_data_used':False}
    value['native_continuation_qualification']=write(f['runtime']/'CONTINUATION_FAKE_PROOF.json',proof)
    reseal(value,'recipe_sha256')
    return value,proof


def test_three_pass_launch_requires_exact_original_and_zero_update_proof(fixture,monkeypatch):
    f=fixture;value,_=continuation_fixture(f,monkeypatch)
    ref=write(f['runtime']/'CONTINUATION_FAKE_RECIPE.json',value)
    checked=owner.require_prefix_fit_launch_recipe(ref,source_repo=f['repo'],source_commit='a'*40)
    assert checked['train_passes']==3 and checked['numerical_recipe']['max_updates']==8928


@pytest.mark.parametrize('fault',['terminal','old_recipe','passes','budget','qualification','proof_steps','proof_bool_steps','proof_sources','test'])
def test_three_pass_launch_rejects_expansion_or_unqualified_state(fixture,monkeypatch,fault):
    f=fixture;value,proof=continuation_fixture(f,monkeypatch)
    if fault=='terminal':value['continuation_origin']['terminal_state']={'path':'/other','sha256':'0'*64}
    elif fault=='old_recipe':value['numerical_recipe']['research_objective_sha256']='f'*64;numerical_seal(value['numerical_recipe'])
    elif fault=='passes':value['train_passes']=4
    elif fault=='budget':value['numerical_recipe']['max_updates']=8929;numerical_seal(value['numerical_recipe'])
    elif fault=='qualification':del value['native_continuation_qualification']
    elif fault=='proof_steps':
        proof['optimizer_steps']=1;value['native_continuation_qualification']=write(f['runtime']/'BAD_PROOF.json',proof)
    elif fault=='proof_bool_steps':
        proof['optimizer_steps']=False;value['native_continuation_qualification']=write(f['runtime']/'BAD_PROOF.json',proof)
    elif fault=='proof_sources':
        proof['source_owners']={};value['native_continuation_qualification']=write(f['runtime']/'BAD_PROOF.json',proof)
    else:value['test_data_used']=True
    reseal(value,'recipe_sha256');ref=write(f['runtime']/'BAD_CONTINUATION.json',value)
    with pytest.raises(RuntimeError):owner.require_prefix_fit_launch_recipe(ref,source_repo=f['repo'],source_commit='a'*40)


def test_three_pass_cursor_and_progress_track_partial_boundaries(fixture,monkeypatch):
    f=fixture;value,_=continuation_fixture(f,monkeypatch)
    ref=write(f['runtime']/'CONTINUATION_FAKE_RECIPE.json',value)
    state=cursor(f,recipe_ref=ref,train_passes=3,parent_cursor=47602,optimizer_updates=2976)
    assert state['train_passes']==3 and state['parent_cursor']==47602
    pointer=write(Path(f['windows'][0]['campaign_cursor_path']),state)
    window={**f['windows'][0],'recipe':ref}
    invocation={**f['invocations'][0],'direction_prefix_fit_recipe':ref,'optimizer_step_budget':8928}
    progress=owner.build_prefix_fit_progress(window=window,plan=f['plan'],invocation=invocation,
        checkpoint_pointer=pointer,cursor=state)
    checked=owner.require_prefix_fit_progress(progress,plan_sha256=f['plan']['plan_sha256'],invocation=invocation,
        expected_selection_receipt_sha256=None,verify_file=True)
    assert checked['completed_units']==47602 and checked['total_units']==142806
    for invalid in (47603,2*47602+1,3*47602+1):
        with pytest.raises(RuntimeError):cursor(f,recipe_ref=ref,train_passes=3,parent_cursor=invalid,optimizer_updates=2976)
    finished=cursor(f,recipe_ref=ref,train_passes=3,parent_cursor=3*47602,optimizer_updates=8928,
        phase_state='SCORING',final_train_measurement=f['dummy'])
    assert finished['parent_cursor']==142806
    with pytest.raises(RuntimeError):cursor(f,recipe_ref=ref,train_passes=3,parent_cursor=47602,optimizer_updates=2976,
        phase_state='SCORING',final_train_measurement=f['dummy'])


def test_continuation_cursor_verification_can_read_exact_initial_under_worker_guard(fixture,monkeypatch):
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    f=fixture;value,_=continuation_fixture(f,monkeypatch)
    recipe_ref=write(f['runtime']/'GUARDED_CONTINUATION.json',value)
    initial={'path':'/home/andre2/GX1_RUNS/synthetic/INITIAL_0U.pt','sha256':'e'*64}
    terminal={'path':'/home/andre2/GX1_RUNS/synthetic/TERMINAL.pt','sha256':'f'*64}
    owner.CONTINUATION_ORIGIN['initial_state']=initial;owner.CONTINUATION_ORIGIN['terminal_state']=terminal
    hooks=[];monkeypatch.setattr(common.sys,'addaudithook',hooks.append)
    context={'ancestor_plan':{'payloads':{},'native_cache_dir':'/synthetic/cache','gc_surfaces':{'path':'/synthetic/gc/surface'}},
        'info':{'metadata':{},'audit':{'manifest_path':'/synthetic/audit'}}}
    denied=common._install_read_guard(context,'/synthetic/output','/synthetic/runtime',extra_read_paths=(terminal['path'],initial['path']))
    original_regular=campaign._regular;original_sha=campaign.file_sha256
    def regular(path,label):
        return path if str(path)==initial['path'] else original_regular(path,label)
    def sha(path):
        if str(path)==initial['path']:
            hooks[0]('open',(str(path),'rb',0));return initial['sha256']
        return original_sha(path)
    monkeypatch.setattr(campaign,'_regular',regular);monkeypatch.setattr(campaign,'file_sha256',sha)
    state=cursor(f,recipe_ref=recipe_ref,train_passes=3,parent_cursor=47602,optimizer_updates=2976,own_initial_state=initial)
    checked=owner.require_prefix_fit_cursor(state,expected_recipe=recipe_ref,expected_fold=f['fold'],verify_files=True)
    assert checked['own_initial_state']==initial and denied==[]


def chronological_fixture(f,monkeypatch):
    frozen_path='gx1/contracts/entry_native_direction_frozen_extraction_v1.py'
    frozen=write(f['repo']/frozen_path,{'synthetic':True})
    component_path=owner.ORDER_COMPONENT_SOURCE
    component=write(f['repo']/component_path,{'synthetic':True})
    monkeypatch.setattr(owner,'REQUIRED_SOURCE_PATHS',('source.py',component_path))
    manifest=json.loads(Path(f['recipe']['input_manifest']['path']).read_text())
    original_component={'path':component['path'],'sha256':'1'*64}
    manifest['view_source_bindings']['direction_component_source']=original_component
    input_ref=write(f['runtime']/'ORDER_INPUT.json',manifest)
    old=copy.deepcopy(f['recipe']);old['input_manifest']=input_ref;reseal(old,'recipe_sha256')
    old_ref=write(f['runtime']/'ORDER_ORIGINAL_RECIPE.json',old)
    initial=write(f['runtime']/'ORDER_INITIAL_REPORT.json',{'role':'INITIAL_TRAIN','fold':f['fold'],'predictions':f['dummy']})
    origin={'launch_recipe':old_ref,'initial_measurement':initial,'initial_predictions':f['dummy']}
    monkeypatch.setattr(owner,'ORDER_ORIGIN',origin)
    monkeypatch.setattr(owner.private,'BPS_GC_PARITY',f['dummy'])
    value=copy.deepcopy(old);value.update(run_scope=owner.ORDER_SCOPE,parent_order_algorithm=owner.ORDER_ALGORITHM,
        order_origin=copy.deepcopy(origin),gc_layout_metadata_parity=f['dummy'])
    value['source_owners'][component_path]=component;value['source_owners'][frozen_path]=frozen
    proof={'schema_version':'gx1_direction_order_initial_resume_qualification_v1',
        'status':'PASS_SIGN_ONLY_ORDER_INITIAL_RESUME_QUALIFIED','origin':copy.deepcopy(origin),'fold':f['fold'],
        'source_owners':copy.deepcopy(value['source_owners']),'numerical_recipe_sha256':value['numerical_recipe']['contract_sha256'],
        'gc_layout_metadata_parity':f['dummy'],'initial_model_buffers_optimizer_rng_equal':True,
        'initial_predictions_256_bit_equal':True,'resume_bit_exact':True,'private_optimizer_updates':4,
        'qualification_weights_for_main_fit':False,'test_data_used':False,
        'historical_component_transition':{'original':original_component,'current':component}}
    value['native_order_qualification']=write(f['runtime']/'ORDER_FAKE_PROOF.json',proof)
    reseal(value,'recipe_sha256')
    return value,proof


def test_chronological_one_pass_requires_original_and_exact_initial_resume_proof(fixture,monkeypatch):
    f=fixture;value,_=chronological_fixture(f,monkeypatch)
    ref=write(f['runtime']/'ORDER_FAKE_RECIPE.json',value)
    checked=owner.require_prefix_fit_launch_recipe(ref,source_repo=f['repo'],source_commit='a'*40)
    assert checked['parent_order_algorithm']==owner.ORDER_ALGORITHM and checked['train_passes']==1


@pytest.mark.parametrize('fault',['algorithm','origin','old_numerics','proof_missing','proof_initial','proof_prediction',
    'proof_resume','proof_updates','proof_bool_updates','proof_sources','proof_transition','proof_weights','test'])
def test_chronological_launch_rejects_unmatched_or_unqualified_inputs(fixture,monkeypatch,fault):
    f=fixture;value,proof=chronological_fixture(f,monkeypatch)
    if fault=='algorithm':value['parent_order_algorithm']=owner.PARENT_ORDER_ALGORITHM
    elif fault=='origin':value['order_origin']['launch_recipe']={'path':'/other','sha256':'0'*64}
    elif fault=='old_numerics':value['numerical_recipe']['research_objective_sha256']='f'*64;numerical_seal(value['numerical_recipe'])
    elif fault=='proof_missing':del value['native_order_qualification']
    elif fault=='proof_initial':proof['initial_model_buffers_optimizer_rng_equal']=False
    elif fault=='proof_prediction':proof['initial_predictions_256_bit_equal']=False
    elif fault=='proof_resume':proof['resume_bit_exact']=False
    elif fault=='proof_updates':proof['private_optimizer_updates']=0
    elif fault=='proof_bool_updates':proof['private_optimizer_updates']=True
    elif fault=='proof_sources':proof['source_owners']={}
    elif fault=='proof_transition':proof['historical_component_transition']['original']['sha256']='2'*64
    elif fault=='proof_weights':proof['qualification_weights_for_main_fit']=True
    else:value['test_data_used']=True
    if fault.startswith('proof_') and fault!='proof_missing':
        value['native_order_qualification']=write(f['runtime']/'BAD_ORDER_PROOF.json',proof)
    reseal(value,'recipe_sha256');ref=write(f['runtime']/'BAD_ORDER.json',value)
    with pytest.raises(RuntimeError):owner.require_prefix_fit_launch_recipe(ref,source_repo=f['repo'],source_commit='a'*40)


def test_dense_fit_requires_exact_private_qualification_sources_and_protocol(fixture,monkeypatch):
    f=fixture;repo=f['repo'];runtime=f['runtime'];dummy=f['dummy'];private=owner.private
    monkeypatch.setattr(private,'DENSE_EXTRA_SOURCE_PATHS',())
    monkeypatch.setattr(private,'REQUIRED_SOURCE_PATHS',('source.py','gx1/contracts/entry_native_direction_prefix_fit_campaign_v1.py'))
    monkeypatch.setattr(private,'DENSE_INPUT_MANIFEST',f['input_ref'])
    monkeypatch.setattr(private,'BPS_GC_PARITY',dummy)
    monkeypatch.setattr(private,'DENSE_TARGET_RESULT',dummy)
    target={'path':str(runtime/'unopened.npz'),'sha256':'f'*64};monkeypatch.setattr(private,'DENSE_TARGET',target)
    protocol=write(repo/'docs/DIRECTION_DENSE_ONEPASS_EVALUATION_20261028.json',
        {'schema_version':'gx1_dense_onepass_direction_evaluation_protocol_v1',
         'status':'PREDECLARED_BEFORE_DENSE_FIT_NO_EXECUTION_AUTHORITY','fold':f['fold'],
         'test_data_used':False,'native_launch_authorized':False})
    item=copy.deepcopy(f['recipe']);item['run_scope']=owner.DENSE_SCOPE
    num=item['numerical_recipe'];num['dense_minute_supervision']=True;numerical_seal(num)
    prior_num=copy.deepcopy(num);prior_num['max_updates']=2;numerical_seal(prior_num)
    prior=write(runtime/'PRIVATE_RECIPE.json',{'run_scope':private.DENSE_PRIVATE_SCOPE,
        'input_manifest':f['input_ref'],'fold':f['fold'],'dense_target':target,'dense_target_result':dummy,
        'dense_source_transition_proof':dummy,'qualification_weights_for_main_fit':False,
        'numerical_recipe':prior_num,'source_owners':{'source.py':f['source']}})
    old_plan=write(runtime/'PRIVATE_PLAN.json',{'recipe':prior})
    proof=write(runtime/'PRIVATE_REVIEW.json',{'status':'PASS_CONSUMED_PRIVATE_NATIVE_MECHANICS','plan':old_plan,
        'initial_256_parity':True,'sixty_horizons_have_eligible_labels_and_gradients':True,
        'chart_mtf_gradient_first_step_nonzero':True,'chart_local_gradient_first_step_zero_and_second_nonzero':True,
        'bit_exact_resume':True,'private_optimizer_updates':4,'anchor_prefix_priors_equal':True,
        'qualification_weights_for_main_fit':False,'test_data_used':False})
    monkeypatch.setattr(owner,'DENSE_NATIVE_QUALIFICATION',proof)
    item.update(gc_layout_metadata_parity=dummy,native_dense_qualification=proof,dense_source_transition_proof=dummy,
        dense_target_result=dummy,dense_target=target,dense_evaluation_protocol=protocol)
    reseal(item,'recipe_sha256');ref=write(runtime/'DENSE_RECIPE.json',item)
    assert owner.require_prefix_fit_launch_recipe(ref,source_repo=repo,source_commit='a'*40)['run_scope']==owner.DENSE_SCOPE
    assert not Path(target['path']).exists()
    for key,value in (('dense_target',{**target,'sha256':'0'*64}),('qualification_weights_for_main_fit',True),
                      ('test_data_used',True),('parent_order_algorithm','chronological')):
        wrong=copy.deepcopy(item);wrong[key]=value;reseal(wrong,'recipe_sha256')
        with pytest.raises(RuntimeError):
            owner.require_prefix_fit_launch_recipe(write(runtime/'BAD_DENSE.json',wrong),source_repo=repo,source_commit='a'*40)
    bad_num=copy.deepcopy(item);bad_num['numerical_recipe']['dense_minute_supervision']=False
    numerical_seal(bad_num['numerical_recipe']);reseal(bad_num,'recipe_sha256')
    with pytest.raises(RuntimeError):
        owner.require_prefix_fit_launch_recipe(write(runtime/'BAD_DENSE.json',bad_num),source_repo=repo,source_commit='a'*40)
