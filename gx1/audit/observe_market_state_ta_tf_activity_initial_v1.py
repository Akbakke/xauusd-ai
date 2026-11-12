"""One read-only native c171 initial comparison for independent timeframe activity heads."""
from __future__ import annotations
import copy
import hashlib
import io
from pathlib import Path
import signal
import subprocess
import time

from gx1.audit import cache_market_state_ta_direction_v1 as cache
from gx1.audit import observe_market_state_ta_c171_short_heads_v1 as previous
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts import market_state_ta_m5_short_learning_v1 as short

REPO=cache.REPO
STAGE='market_state_ta_tf_activity_initial_parity_20261112_001'
SCOPE='market_state_ta_tf_activity_initial_parity_v1'
ROOT=Path('/home/andre2/GX1_RUNS')/STAGE.upper()
SCHEMA='gx1_ta_tf_activity_initial_parity_plan_v1'
WORKER=REPO/'gx1/audit/observe_market_state_ta_tf_activity_initial_v1.py'
EXTRA_OWNERS=short.EXTRA_OWNERS | {str(WORKER.relative_to(REPO)),str(previous.WORKER.relative_to(REPO))}
NUMERICAL_CHANGES=frozenset({
    'gx1/contracts/market_state_time_query_model_v2.py',
    'gx1/contracts/market_state_time_query_native_F_model_v1.py',
    'gx1/contracts/market_state_time_query_native_F_objective_v1.py',
    'gx1/contracts/market_state_time_query_fold1_fit_state_v1.py',
    'gx1/contracts/market_state_time_query_fold1_fit_v1.py',
    'gx1/contracts/entry_native_direction_feature_view_v1.py',
    'gx1/contracts/market_state_time_query_native_F_batch_v1.py',
    'gx1/audit/verify_market_state_time_query_zero_step_v1.py',
})
INPUTS={**cache.FUSION_REFS,**{key:previous.INPUTS[key] for key in (
    'short_recipe','short_result','short_initial_checkpoint','short_initial_measurement','short_initial_capture')},
    'mechanical_review':{'path':'/home/andre2/GX1_RUNS/MARKET_STATE_TA_TF_ACTIVITY_MECHANICAL_20261112_001/FOCUSED_TESTS.json','sha256':'5251528c1690d08017280fe5dc363c740ce825cc20bbed32d773003654e71edf'}}
BOUNDS=dict(fit_rows=29574,batch_size=16,panel_batch_indices=list(short.PANEL_BATCH_INDICES),
    sampled_rows=512,unique_rows=256,phases=['CURRENT_SHARED_INITIAL','TF_ACTIVITY_INITIAL'],
    max_model_constructions=2,max_checkpoint_loads=1,max_backbone_encodings=32,
    max_optimizer_constructions=0,max_optimizer_updates=0,max_backwards=0,max_autograd_grad_calls=0,
    max_checkpoint_writes=0,max_candidate_writes=0,max_rng_restores=0,
    max_cache_bytes=2*1024**2,max_published_json_bytes=1024**2,wall_seconds=900,device='cuda',
    initial_model_state_sha256=short.MODEL_SHA256,added_parameter_tensors=20,added_parameters=1290,
    cloned_activity_atol=2e-6,cloned_activity_rtol=1e-6,loss_atol=5e-6,
    numerical_mode='original_eval_no_grad_deterministic_fp32_mha_fastpath_false',
    numerical_threads=8,interop_policy='process_default_unchanged_like_short002',
    thread_environment=dict(previous.BOUNDS['thread_environment']),
    parameter_partition='original_fold1_fit_active_true_inactive_false_new_activity_inherits_shared',
    training_enabled=False,test_data_used=False)

def _need(ok,reason):
    if not ok:raise RuntimeError('TA_TF_ACTIVITY_INITIAL_'+reason)

def current_source_bridge(original,current):
    """This new read-only scope binds eight reviewed changes; the old bridge is untouched."""
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
    _need(set(original)==fit.required_source_paths(ta_sf=True) and len(original)==85
          and set(current)==set(original)|EXTRA_OWNERS,'OWNER_SET')
    changed=set();result={}
    for name,old in original.items():
        _need(old['path']==str(REPO/name) and current[name]['path']==str(REPO/name),'OWNER_PATH')
        if not name.startswith('.venv/'):
            raw=subprocess.check_output(['git','show',cache.REVERSED_SOURCE+':'+name],cwd=REPO)
            _need(hashlib.sha256(raw).hexdigest()==old['sha256'],'HISTORICAL_SOURCE:'+name)
        campaign.require_binding(current[name],label='activity current source '+name,verify_file=True)
        if current[name]!=old:changed.add(name)
        _need(name in short.ORCHESTRATION|NUMERICAL_CHANGES or current[name]==old,
              'UNREVIEWED_SOURCE:'+name)
        result[name]=current[name]['sha256']
    _need(changed-short.ORCHESTRATION==NUMERICAL_CHANGES,'EXACT_NUMERICAL_CHANGE_SET')
    for name in EXTRA_OWNERS:
        _need(current[name]['path']==str(REPO/name),'EXTRA_OWNER_PATH')
        campaign.require_binding(current[name],label='activity helper '+name,verify_file=True)
    return result

def require_view_source_transition(historical,current,plan):
    """Only the v2 model source ref and resulting view digest change for defaultFalse."""
    _need(set(historical)==set(current) and historical['contract_sha256']!=current['contract_sha256'],
          'CURRENT_VIEW_KEYS')
    _need(all(current[k]==v for k,v in historical.items() if k not in ('contract_sha256','source_bindings')),
          'CURRENT_PHYSICAL_VIEW')
    before,after=historical['source_bindings'],current['source_bindings']
    key='time_query_v2_model_source';name='gx1/contracts/market_state_time_query_model_v2.py'
    _need(set(before)==set(after) and after[key]==plan['tf_activity_initial_source_owners'][name]
          and before[key]!=after[key] and all(after[k]==v for k,v in before.items() if k!=key),
          'CURRENT_VIEW_SOURCE_ONLY')

def _require_trial_view(source,trial):
    _need(set(trial)==set(source)|{'dedicated_tf_activity'}
          and trial['dedicated_tf_activity'] is True
          and trial['research_objective']['dedicated_tf_activity'] is True
          and trial['contract_sha256']!=source['contract_sha256']
          and all(trial[k]==v for k,v in source.items() if k not in ('research_objective','contract_sha256')),
          'TRIAL_VIEW_HEAD_ONLY')

def _origin(owners,bound):
    from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero
    context=dict(inputs=INPUTS,tf_activity_initial_scope=SCOPE,tf_activity_initial_source_owners=owners)
    q=cache._qualify_reversed_origin(context,bound)
    historical=cache._read(q['origin']['original_recipe']['input_refs']['ta_candidate_view'])
    trial=zero._bind_ta_view(q['receipts'],dedicated_m5_side=True,dedicated_tf_activity=True)
    _require_trial_view(q['view'],trial)
    original,result,measurement=(cache._read(INPUTS[k]) for k in ('short_recipe','short_result','short_initial_measurement'))
    previous._qualify_short(original,result,measurement,original)
    _need(original['expected_view_sha256']==historical['contract_sha256']
          and original['numerical_recipe']==q['recipe']['numerical_recipe'],'ORIGINAL_SHORT_IDENTITY')
    return dict(q,trial_view=trial,historical_view=historical,short_recipe=original,
                initial_measurement=measurement)

def build_plan(source_commit):
    from gx1.contracts.market_state_time_query_native_F_model_v1 import contract as model_contract
    from gx1.contracts.market_state_time_query_native_F_objective_v1 import contract as loss_contract
    recipe=cache._read(INPUTS['reverse_recipe'])
    owners={name:cache._ref(REPO/name) for name in set(recipe['source_owners'])|EXTRA_OWNERS}
    q=_origin(owners,[])
    return dict(schema_version=SCHEMA,stage=STAGE,source_repo=str(REPO),source_base_commit=source_commit,
        worker=cache._ref(WORKER),cap_runner=cache._ref(REPO/cache.CAP_RUNNER),source_owners=owners,
        inputs=INPUTS,source_view_sha256=q['view']['contract_sha256'],trial_view_sha256=q['trial_view']['contract_sha256'],
        source_model_contract_sha256=q['short_recipe']['expected_model_contract_sha256'],
        trial_model_contract_sha256=model_contract(ta_only=True,dedicated_m5_side=True,dedicated_tf_activity=True)['contract_sha256'],
        source_loss_contract_sha256=q['short_recipe']['expected_loss_contract_sha256'],
        trial_loss_contract_sha256=loss_contract(ta_only=True,dedicated_m5_side=True,m5_equal_observation=True,dedicated_tf_activity=True)['contract_sha256'],**BOUNDS)

def _require_plan(plan,plan_path,out_dir):
    dynamic={'schema_version','stage','source_repo','source_base_commit','worker','cap_runner','source_owners',
             'inputs','source_view_sha256','trial_view_sha256','source_model_contract_sha256','trial_model_contract_sha256',
             'source_loss_contract_sha256','trial_loss_contract_sha256'}
    _need(isinstance(plan,dict) and set(plan)==dynamic|set(BOUNDS) and plan['schema_version']==SCHEMA
          and plan['stage']==STAGE and plan['source_repo']==str(REPO)
          and Path(plan_path)==ROOT/'PLAN.json' and Path(out_dir)==ROOT and plan['inputs']==INPUTS
          and all(type(plan[k]) is type(v) and plan[k]==v for k,v in BOUNDS.items())
          and all(type(i) is int for i in plan['panel_batch_indices']), 'EXACT_PLAN')
    _need(isinstance(plan['source_base_commit'],str) and campaign._COMMIT.fullmatch(plan['source_base_commit']),
          'SOURCE_BASE')
    return plan

def _inspect(plan_path,plan_sha,out_dir,*,execute):
    ref=dict(path=str(plan_path),sha256=plan_sha);plan=_require_plan(cache._read(ref),plan_path,out_dir)
    bound=[ref]
    for key,path in (('worker',WORKER),('cap_runner',REPO/cache.CAP_RUNNER)):
        _need(plan[key]['path']==str(path),'SOURCE_PATH')
        campaign.require_binding(plan[key],label='activity '+key,verify_file=True);bound.append(plan[key])
    for value in INPUTS.values():campaign.require_binding(value,label='activity input',verify_file=True);bound.append(value)
    bound.extend(plan['source_owners'].values())
    head=cache._git('rev-parse','HEAD')
    subprocess.run(['git','merge-base','--is-ancestor',plan['source_base_commit'],head],cwd=REPO,check=True)
    expected=build_plan(plan['source_base_commit']);_need(plan==expected,'CURRENT_PLAN_BINDINGS')
    q=_origin(plan['source_owners'],bound)
    policy_ref=cache._ref(REPO/'NEXT_RUN_POLICY.json');policy=cache._read(policy_ref);bound.append(policy_ref)
    _need(all(policy.get(k) is False for k in ('training_enabled','native_training_enabled',
        'native_launch_authorized','full_epoch_training_allowed','full_val_allowed')),'NO_TRAINING')
    _need(not any(ROOT.glob('*.npz')) and not any((ROOT/name).exists() for name in ('CLAIM.json','RESULT.json','TERMINAL.json')),'CONSUMED')
    if execute:
        gate=policy.get(STAGE,{})
        exact=dict(status='ADMITTED',plan=ref,max_invocations=1,remaining_invocations=1,claimed_invocations=0,
            completed_invocations=0,relaunch_allowed=False,execution_authorized=True,data_reads_authorized=True,
            checkpoint_reads_authorized=True,backbone_encodings_authorized=32,head_tensor_imports_authorized=20,
            candidate_weights_writes_authorized=0,backward_calls_authorized=0,optimizer_steps_authorized=0,
            training_authorized=False)
        _need(all(type(gate.get(k)) is type(v) and gate[k]==v for k,v in {**BOUNDS,**exact}.items()),'ADMISSION')
        _need(cache._git('branch','--show-current')=='work/gx1-current','BRANCH')
        campaign.require_clean_source(dict(source_repo=str(REPO),source_commit=head))
    return dict(q,plan=plan,plan_ref=ref,head=head,bound_refs=bound,policy_ref=policy_ref)

def preflight(plan_path,plan_sha,out_dir):
    q=_inspect(plan_path,plan_sha,out_dir,execute=False)
    return dict(status='PASS_METADATA_ONLY_TF_ACTIVITY_INITIAL_PREFLIGHT',stage=STAGE,plan=q['plan_ref'],
        source_view_sha256=q['view']['contract_sha256'],trial_view_sha256=q['trial_view']['contract_sha256'],
        model_constructions=0,array_deserializations=0,checkpoint_loads=0,optimizer_updates=0,claims=0,
        training_enabled=False,test_data_used=False)

def _require_trial_parity(actual,baseline,loss,expected_loss,*,np):
    previous._require_arrays(actual,16,np=np);previous._require_arrays(baseline,16,np=np)
    _need(cache._fusion_exact_arrays({k:v for k,v in actual.items() if not k.endswith('_raw')},
                                   {k:v for k,v in baseline.items() if not k.endswith('_raw')}),'TRIAL_AXES')
    result={}
    for task in ('state','future'):
        a,b=actual[task+'_raw'],baseline[task+'_raw']
        _need(cache._fusion_exact_arrays({'side':a[...,1]}, {'side':b[...,1]}),'UNCHANGED_OUTPUTS')
        delta=a[...,0].astype(np.float64)-b[...,0].astype(np.float64)
        _need(np.allclose(a[...,0],b[...,0],rtol=BOUNDS['cloned_activity_rtol'],atol=BOUNDS['cloned_activity_atol']),
              'CLONED_ACTIVITY_FP32_PARITY')
        result[task]=dict(unchanged_outputs_bitexact=True,max_abs_activity=float(np.abs(delta).max()))
    _need(abs(float(loss)-float(expected_loss))<=BOUNDS['loss_atol'],'CLONED_LOSS_PARITY')
    return dict(tasks=result,loss_delta=float(loss)-float(expected_loss))

def _save(parts,counts,*,partial,np):
    pieces=[p for phase in BOUNDS['phases'] for p in parts[phase]]
    arrays={key:np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
    for phase,index in zip(BOUNDS['phases'],range(2)):
        chosen=arrays['phase']==index;n=int(chosen.sum())
        if n:previous._require_arrays({k:v[chosen] for k,v in arrays.items() if k!='phase'},n,np=np)
    _need(len(arrays['parents'])<=512 and np.all(np.diff(arrays['phase'].astype(int))>=0),'CAPTURE_PHASES')
    if not partial:_need(np.array_equal(arrays['phase'],np.r_[np.zeros(256,np.uint8),np.ones(256,np.uint8)]),'FULL_CAPTURE')
    buf=io.BytesIO();np.savez(buf,**arrays)
    _need(counts['published_npz_bytes']+buf.tell()<=BOUNDS['max_cache_bytes'],'OUTPUT_BYTES')
    return cache._save_fusion_arrays('TRAIN_FUSION_PARTIAL.npz' if partial else 'TRAIN_FUSION.npz',arrays,counts,np=np,root=ROOT)

def run(plan_path,plan_sha,out_dir,device='cuda'):
    started=time.monotonic();_need(device=='cuda','CUDA_ONLY')
    q=_inspect(plan_path,plan_sha,out_dir,execute=True);cap=previous._runtime_cap()
    import numpy as np
    import torch
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as base
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts import entry_native_direction_research_state_v1 as old
    from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
    from gx1.contracts.market_state_time_query_binding_v1 import load_bound_time_query_targets
    from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
    from gx1.contracts.market_state_time_query_native_F_batch_v1 import evaluate_fit_batch
    from gx1.contracts.market_state_time_query_native_F_objective_v1 import require_m5_equal_observation_axes
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as model_sha
    from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import _BATCH_EQUAL_LENGTH_MTF_EVAL
    _need(not torch.cuda.is_initialized() and signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'INIT_ORDER')
    claim=cache._publish('CLAIM.json',dict(plan=q['plan_ref'],source_commit=q['head'],cap=cap,max_invocations=1,
        no_relaunch_under_this_plan=True,test_data_used=False),root=ROOT)
    counts=dict(model_constructions=0,checkpoint_loads=0,backbone_encodings=0,forward_returns=0,
        head_tensor_imports=0,archive_decodes=0,autograd_grad_calls=0,backwards=0,optimizer_constructions=0,
        optimizer_updates=0,checkpoint_writes=0,candidate_writes=0,resume_state_load_calls=0,
        resume_state_load_returns=0,published_npz_bytes=0,sampled_rows=0,matched_batches=0,extra_target_gathers=0,rng_restores=0)
    provider=prepared=None;models=[];parts={phase:[] for phase in BOUNDS['phases']};records=[];artifacts={};before={};passivity={};runtime={}
    active_phase=None;previous_alarm=signal.getsignal(signal.SIGALRM)
    def timeout(*unused):raise TimeoutError('Timeframe activity initial comparison900-second bound')
    def bookend():
        campaign.require_clean_source(dict(source_repo=str(REPO),source_commit=q['head']))
        for value in q['bound_refs']:campaign.require_binding(value,label='activity bookend',verify_file=True)
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,max(1.,900-(time.monotonic()-started)))
    try:
        receipts,view=q['receipts'],q['view'];cuda=torch.device('cuda');numerical=q['short_recipe']['numerical_recipe']
        trainer._set_deterministic(numerical['seed'],cuda,numerical['precision_policy'])
        runtime=previous._runtime_profile(torch);previous._require_runtime_profile(runtime)
        _need(_BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False,'RUNTIME')
        with np.load(INPUTS['short_initial_capture']['path'],allow_pickle=False) as z:baseline={k:z[k].copy() for k in z.files}
        counts['archive_decodes']+=1;previous._require_arrays(baseline,256,np=np)
        provider,provider_plan,prepared,rows,issues=runner._prepared_fold1(receipts,view,np=np)
        require_m5_equal_observation_axes(rows,issues)
        bound,future=load_bound_time_query_targets(),load_bound_future_displacement()
        for model_view in (view,q['trial_view']):
            counts['model_constructions']+=1
            backbone=views.build_native_direction_research_model(model_view,prefix_normalization=receipts['fold1']['normal'],
                constructor_metadata=receipts['constructor'],device=torch.device('cpu'),unobserved_units_profile=receipts['fold1']['profile'])
            models.append(NativeFutureDualModel(backbone))
        source,trial=models;counts['checkpoint_loads']+=1
        saved=common._load_state(INPUTS['short_initial_checkpoint'],torch=torch,maximum=q['short_recipe']['max_checkpoint_bytes'],counters=counts)
        _need(saved['model_state_sha256']==model_sha(saved['model_state'])==short.MODEL_SHA256
              and saved['feature_view_sha256']==q['short_recipe']['expected_view_sha256']
              and (saved['cursor'],saved['optimizer_updates'],saved['skipped_batches'])==(0,0,0)
              and np.array_equal(saved['parent_order'],rows),'INITIAL_CHECKPOINT')
        source.load_state_dict(saved['model_state'],strict=True)
        modes=saved['module_training'];_need(set(modes)==set(dict(source.named_modules())),'SOURCE_MODES')
        for name,module in source.named_modules():module.training=modes[name]
        previous._short_parameter_partition(source,state)
        migration=state.copy_dedicated_m5_weights_to_tf_activity(source,trial,expected_legacy_model_sha256=short.MODEL_SHA256)
        _need(migration['added_parameters']==1290 and len(migration['copied_activity_rows'])==20
              and migration['optimizer_updates']==migration['cursor']==0,'EXACT_MIGRATION')
        counts['head_tensor_imports']=20;del saved
        for model in models:model.to(cuda);model.eval()
        def signature(model):
            return dict(model=model_sha(model.state_dict()),buffers=old._tree_sha(dict(model.named_buffers())),
                rng=old._tree_sha(trainer._attended_session_rng_state(device=cuda)),
                modes={n:m.training for n,m in model.named_modules()},flags={n:p.requires_grad for n,p in model.named_parameters()},
                parameters={n:id(p) for n,p in model.named_parameters()},buffer_ids={n:id(p) for n,p in model.named_buffers()},
                modules={n:id(m) for n,m in model.named_modules()},view=copy.deepcopy(model.state_model.feature_view_identity),
                hooks={n:{k:tuple((i,id(h)) for i,h in getattr(m,k).items()) for k in
                    ('_forward_hooks','_forward_pre_hooks','_backward_hooks','_backward_pre_hooks')} for n,m in model.named_modules()},
                mha_fastpath=torch.backends.mha.get_fastpath_enabled(),runtime=previous._runtime_profile(torch))
        for phase,model in zip(BOUNDS['phases'],models):before[phase]=signature(model)
        _need(before[BOUNDS['phases'][0]]['model']==short.MODEL_SHA256
              and before[BOUNDS['phases'][1]]['model']==migration['model_state_sha256'],'MIGRATED_DIGEST')
        for position,index in enumerate(BOUNDS['panel_batch_indices']):
            left=index*16;sl=slice(position*16,(position+1)*16)
            raw,parents,clocks=runner._feature_batch(prepared,view,receipts,rows,issues,left,left+16,np=np)
            features={k:torch.from_numpy(v).to(cuda) for k,v in raw.items()};reference=None;reference_loss=None
            for number,(phase,model) in enumerate(zip(BOUNDS['phases'],models)):
                active_phase=phase
                with torch.no_grad(),trainer._training_mha_fastpath_context(model.backbone),trainer._training_autocast_context(),trainer._training_model_finite_check_context():
                    _need(not torch.is_grad_enabled() and not model.training and not model.backbone.training
                          and torch.backends.mha.get_fastpath_enabled() is False
                          and _BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False
                          and counts['backbone_encodings']<32,'FORWARD_PROFILE')
                    counts['backbone_encodings']+=1
                    report=evaluate_fit_batch(model,features,bound,future,parents,clocks,fold_id=1,pass_index=0,
                        seed=numerical['seed'],future_weight=1.,m5_equal_observation=True,fit_batch_start=left)
                    counts['forward_returns']+=1
                role='source' if number==0 else 'trial'
                _need(report['skip_update'] is False and not report['loss'].requires_grad
                    and report['model_contract_sha256']==q['plan'][role+'_model_contract_sha256']
                    and report['loss_contract_sha256']==q['plan'][role+'_loss_contract_sha256'],'REPORT_BINDING')
                truths=dict(state=bound.gather_fit_sampled(parents,clocks,fold_id=1,pass_index=0,seed=numerical['seed']),
                    future=future.gather(parents,clocks,fold_id=1,partition='fit'));counts['extra_target_gathers']+=2
                piece=dict(parents=parents.copy(),issues=clocks.copy(),state_query=report['state_query_k'].copy(),
                           future_query=np.broadcast_to(report['future_native_k'],(16,5)).copy())
                for task in ('state','future'):
                    piece[task+'_raw']=report[task+'_activity_side_logits'].detach().cpu().numpy().copy()
                    for name in ('allowed','eligible','mature'):piece[task+'_'+name]=truths[task][name].copy()
                previous._require_arrays(piece,16,np=np);loss=float(report['loss'])
                parts[phase].append(dict(piece,phase=np.full(16,number,np.uint8)));counts['sampled_rows']+=16
                if number==0:
                    reference=piece;reference_loss=loss
                    previous._require_parity(piece,{k:v[sl] for k,v in baseline.items()},loss,
                        q['initial_measurement']['measurement']['canonical_batch_losses'][position],np=np)
                    parity={'historical_all12fields_and_loss_bitexact':True}
                else:parity=_require_trial_parity(piece,reference,loss,reference_loss,np=np)
                counts['matched_batches']+=1;records.append(dict(phase=phase,batch_index=index,rows=16,loss=loss,parity=parity))
                _need(all(p.grad is None for p in model.parameters()),'NO_GRADIENTS');del report
            del features,raw
        for phase,model in zip(BOUNDS['phases'],models):
            passivity[phase+'_signature_unchanged']=signature(model)==before[phase]
            passivity[phase+'_no_parameter_gradients']=all(p.grad is None for p in model.parameters())
        _need(all(passivity.values()),'PASSIVITY')
        exact=dict(model_constructions=2,checkpoint_loads=1,backbone_encodings=32,forward_returns=32,head_tensor_imports=20,
            archive_decodes=1,resume_state_load_calls=1,resume_state_load_returns=1,sampled_rows=512,matched_batches=32,extra_target_gathers=64,
            optimizer_constructions=0,optimizer_updates=0,backwards=0,autograd_grad_calls=0,checkpoint_writes=0,candidate_writes=0,rng_restores=0)
        _need(all(counts[k]==v for k,v in exact.items()),'COUNTS')
        artifacts['initial_parity_capture']=_save(parts,counts,partial=False,np=np)
        release=prepared;prepared=None;base._release_pages(provider,release);provider.validate_metadata(provider_plan);bookend()
        return cache._publish_fusion_outcome(q['plan_ref'],dict(claim=claim,counts=counts,artifacts=artifacts,source_commit=q['head'],
            source_unchanged=True,migration=migration,phase_states={k:{field:v[field] for field in ('model','buffers','rng','flags','modes','view')} for k,v in before.items()},
            batch_records=records,passivity=passivity,numeric_runtime=runtime,elapsed_seconds=time.monotonic()-started,
            historical_initial_measurement=INPUTS['short_initial_measurement'],historical_view_sha256=q['historical_view']['contract_sha256'],
            source_view_sha256=view['contract_sha256'],trial_view_sha256=q['trial_view']['contract_sha256'],
            fresh_actual_initial_measurement=True,source_checkpoint_weights_only=True,optimizer_or_rng_restored=False,
            limitations=['256dependenthistoricalTRAINrows,48previouslyupdated; noheldoutorfullhistoryqualification.',
                'Initialparity only: independentactivityhead learning and edge remainunproved.',
                'No native learning run is admitted by this read-only initial comparison.']),complete=True,root=ROOT,json_bytes_bound=BOUNDS['max_published_json_bytes'])
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL,0);changed=save_error=None
        if any(parts.values()) and not artifacts:
            try:artifacts['partial_initial_parity']=_save(parts,counts,partial=True,np=np)
            except Exception as other:save_error=repr(other)
        try:bookend()
        except Exception as other:changed=repr(other)
        result=cache._publish_fusion_outcome(q['plan_ref'],dict(claim=claim,counts=counts,artifacts=artifacts,source_commit=q['head'],
            source_unchanged=changed is None,active_phase=active_phase,batch_records=records,passivity=passivity,numeric_runtime=runtime,
            error=type(exc).__name__+':'+str(exc),bookend_error=changed,partial_save_error=save_error,interpretation_authorized=False),
            complete=False,root=ROOT,json_bytes_bound=BOUNDS['max_published_json_bytes'])
        raise RuntimeError('TA_TF_ACTIVITY_INITIAL_FAILED_CONSUMED:'+str(result)) from exc
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous_alarm)
        if prepared is not None:base._release_pages(provider,prepared)

def main(argv=None):
    import argparse,json,sys
    supplied=list(sys.argv[1:] if argv is None else argv)
    flags=('--market-state-ta-tf-activity-initial-plan','--market-state-ta-tf-activity-initial-plan-sha256')
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False,add_help=False)
    parser.add_argument(flags[0],type=Path,required=True);parser.add_argument(flags[1],required=True)
    parser.add_argument('--device',choices=('cuda',),required=True);parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    for flag in (*flags,'--device','--out-dir','--preflight'):
        if sum(v.partition('=')[0]==flag for v in supplied)>1:parser.error('duplicate flag: '+flag)
    args=parser.parse_args(supplied)
    if not args.out_dir.is_absolute():parser.error('--out-dir must be absolute')
    kwargs=dict(plan_path=args.market_state_ta_tf_activity_initial_plan,plan_sha=args.market_state_ta_tf_activity_initial_plan_sha256,out_dir=args.out_dir)
    print(json.dumps(preflight(**kwargs) if args.preflight else run(**kwargs,device=args.device),indent=2,sort_keys=True,allow_nan=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
