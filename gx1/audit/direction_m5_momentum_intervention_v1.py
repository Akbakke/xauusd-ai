"""Fixed-model price momentum interventions on qualified M5 TRAIN/development rows.

Only the four XAU MTF momentum columns are zeroed at their normalized, gated
projection input. Other routes, including GC order flow, stay intact. This is
an offline sensitivity test, not an executable model or a trading rule.
"""
from __future__ import annotations
import json
import hashlib
import subprocess
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
import torch

REPO=Path('/home/andre2/src/GX1_CURRENT')
ROOT=Path('/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63')
STAGE='independent_m5_momentum_intervention_20261029_003'
OUTPUT_NAME='DIRECTION_M5_MOMENTUM_INTERVENTION_20261029_003'
BATCH=16
STRIDE=10
ARMS=('rsi14_centered','macd_line_atr','macd_hist_atr','ret_1','all_four')
TIMEFRAMES=('m5','m15','h1','h4','d1')
HEADLINES=(5,60)
REPLICATES=10000
# Five ablations, two horizons, two calendar blocks, three paired metrics.
ALPHA=0.05/(len(ARMS)*len(HEADLINES)*2*3)
SOURCE_PATHS=(
 'gx1/audit/direction_m5_momentum_intervention_v1.py',
 'gx1/audit/direction_gc_mtf_intervention_v1.py',
 'gx1/scripts/validate_entry_model_native_technical_checkpoint_v1.py',
 'gx1/scripts/audit_entry_direction_candle_midpoints_v1.py',
 'gx1/scripts/compare_entry_direction_saved_bps_v1.py',
 'gx1/contracts/entry_direction_block_rank_v1.py',
 'gx1/contracts/entry_direction_probability_reference_v1.py',
 'gx1/contracts/gx1_capped_execution_v1.py',
 'gx1/contracts/local_random_access_campaign_v2.py',
 'gx1/contracts/immutable_event_authority_v1.py',
 'gx1/contracts/entry_model_native_bundle_commit_v1.py',
 'scripts/gx1_capped_run.sh', 'scripts/gx1_guarded_trainer_exec.sh',
 'tests/test_direction_m5_momentum_intervention_v1.py',
 'tests/test_direction_m5_momentum_power_guard_v1.py',
)

def require(value,why):
    if not bool(value):raise RuntimeError('M5_MOMENTUM_INTERVENTION_'+why)

def install_neutralization(layer,columns):
    """Replace selected normalized/gated columns by 0; never mutate input."""
    require(type(layer) is torch.nn.Linear and layer.in_features==4
        and isinstance(columns,tuple) and columns and len(set(columns))==len(columns)
        and all(type(x) is int and 0<=x<4 for x in columns),'NEUTRALIZATION_BINDING')
    calls=[]
    def hook(_module,args):
        require(len(args)==1 and isinstance(args[0],torch.Tensor)
            and args[0].ndim==3 and args[0].shape[-1]==4,'NEUTRALIZATION_INPUT')
        original=args[0];masked=original.clone();masked[...,list(columns)]=0
        calls.append(tuple(original.shape));return (masked,)
    return layer.register_forward_pre_hook(hook),calls

def install_input_observer(layer,fields):
    """Accumulate nonzero/absolute input evidence in actual TF call order."""
    require(type(layer) is torch.nn.Linear and layer.in_features==4 and len(fields)==4
        and len(set(fields))==4,'OBSERVER_BINDING')
    observations={};calls=[]
    def hook(_module,args):
        require(len(args)==1 and isinstance(args[0],torch.Tensor) and args[0].ndim==3
            and args[0].shape[-1]==4 and len(calls)<len(TIMEFRAMES),'OBSERVER_INPUT')
        value=args[0].detach();tf=TIMEFRAMES[len(calls)];calls.append(tf)
        flat=value.reshape(-1,4);nonzero=torch.count_nonzero(flat,dim=0);absolute=flat.abs().sum(dim=0)
        if tf not in observations:observations[tf]={'nonzero':nonzero,'absolute_sum':absolute,'cells_per_field':int(flat.shape[0])}
        else:
            observations[tf]['nonzero']=observations[tf]['nonzero']+nonzero
            observations[tf]['absolute_sum']=observations[tf]['absolute_sum']+absolute
            observations[tf]['cells_per_field']+=int(flat.shape[0])
    return layer.register_forward_pre_hook(hook),calls,observations

def accumulate_observations(total,observed,fields):
    require(set(observed)==set(TIMEFRAMES),'OBSERVATION_TF')
    for tf in TIMEFRAMES:
        row=observed[tf];n=int(row['cells_per_field']);counts=row['nonzero'].cpu().tolist();sums=row['absolute_sum'].cpu().tolist()
        require(n>0 and len(counts)==len(sums)==4,'OBSERVATION_SHAPE')
        bucket=total.setdefault(tf,{name:{'cells':0,'nonzero':0,'absolute_sum':0.0} for name in fields})
        for j,name in enumerate(fields):
            bucket[name]['cells']+=n;bucket[name]['nonzero']+=int(counts[j]);bucket[name]['absolute_sum']+=float(sums[j])

def influence_summary(sign,observed,terminal,baseline,ablated,cutoff):
    n=len(baseline);require(baseline.shape==ablated.shape==sign.shape==observed.shape==terminal.shape==(n,60)
        and np.isfinite(baseline).all() and np.isfinite(ablated).all(),'INFLUENCE_INPUT')
    out=[]
    for j in range(60):
        mask=observed[:,j] & (sign[:,j]!=0) & (terminal[:,j]<cutoff);count=int(mask.sum())
        require(count>0,'INFLUENCE_ELIGIBLE')
        old=baseline[mask,j].astype(np.float64);new=ablated[mask,j].astype(np.float64)
        d=np.abs(new-old);old_dec=old>0;new_dec=new>0;up=sign[mask,j]==1
        out.append({'horizon_minutes':j+1,'eligible':count,'mean_abs_logit_change':float(d.mean()),
            'median_abs_logit_change':float(np.median(d)),'p99_abs_logit_change':float(np.quantile(d,.99)),
            'max_abs_logit_change':float(d.max()),'direction_decisions_changed':int(np.count_nonzero(old_dec!=new_dec)),
            'original_correct_ablated_wrong':int(np.count_nonzero((old_dec==up)&(new_dec!=up))),
            'ablated_correct_original_wrong':int(np.count_nonzero((new_dec==up)&(old_dec!=up)))})
    return out

def design_path_for_stage(stage):
    names={'independent_m5_momentum_intervention_20261029_003':'DIRECTION_M5_MOMENTUM_INTERVENTION_20261029.json',
           'independent_m5_momentum_trace_20261029_002':'DIRECTION_M5_MOMENTUM_TRACE_20261029.json'}
    require(stage in names,'DESIGN_STAGE')
    return REPO/'docs'/names[stage]

def verified_fit_source_transition(preflight, plan, launch, review, historical_runner):
    """Retain full validated fit provenance while pinning the new live runner."""
    runner='scripts/gx1_capped_run.sh';guard='scripts/gx1_guarded_trainer_exec.sh'
    owners=launch['source_owners'];old_runner=owners[runner];new_runner=plan['sources'][runner]
    historical=preflight['historical_and_current_execution_sources_equal']
    require(preflight['status']=='PASS' and preflight['fit_authority_full_validation_passed'] is True
        and preflight['review_and_fit_source_hashes_verified'] is True
        and preflight['fit_source_owners_count']==len(owners)==63
        and preflight['model_forward_calls']==preflight['optimizer_steps']==0
        and preflight['test_data_used'] is False and preflight['recipe']==plan['inputs']['recipe']
        and preflight['review']==plan['inputs']['fit_review']
        and preflight['historical_fit_source_head']==review['source_commit']
        and historical[runner]==old_runner and historical[guard]==owners[guard]
        and plan['sources'][guard]==owners[guard] and new_runner['path']==old_runner['path']
        and new_runner['sha256']!=old_runner['sha256']
        and hashlib.sha256(historical_runner).hexdigest()==old_runner['sha256'], 'FIT_SOURCE_TRANSITION')
    replacement=dict(owners);replacement[runner]=new_runner
    return {**launch,'source_owners':replacement}

def run(*,plan_path,plan_sha256,device,out_dir):
    from gx1.audit import direction_gc_mtf_intervention_v1 as gc
    from gx1.contracts import gx1_capped_execution_v1 as cap,local_random_access_campaign_v2 as binding
    from gx1.contracts import immutable_event_authority_v1 as publication,entry_native_direction_research_state_v1 as state_owner
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as fit_authority
    from gx1.contracts.entry_native_direction_feature_view_v1 import build_native_direction_research_model
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    from gx1.features.entry_direction_momentum_profile_v1 import MOMENTUM_FAMILY,PRICE_MOMENTUM_FIELDS,PROFILE_NAME
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as window
    from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
    sys.dont_write_bytecode=True
    require(device=='cuda' and not torch.cuda.is_initialized(),'EXPLICIT_CUDA')
    require(gc.BATCH==BATCH and gc.BATCH_STRIDE==STRIDE and gc.HEADLINE==HEADLINES and gc.REPLICATES==REPLICATES,'REUSED_METRIC_CONTRACT')
    plan_ref={'path':str(candle.regular(plan_path)),'sha256':plan_sha256};policy_path=REPO/'NEXT_RUN_POLICY.json'
    source_head,branch,dirty=candle.git_state();require(branch=='work/gx1-current' and not dirty,'INITIAL_SOURCE')
    with binding.fresh_file_verification_scope():
        plan=binding.read_bound_json(Path(plan_ref['path']),plan_ref['sha256'])
        require(set(plan)=={'schema_version','stage','design','inputs','sources','fit_source_preflight','output_dir','wall_seconds','batch_size','batch_stride','test_data_used'}
            and plan['schema_version']=='gx1_m5_momentum_intervention_plan_v1' and plan['stage']==STAGE
            and plan['wall_seconds']==3300 and plan['batch_size']==BATCH and plan['batch_stride']==STRIDE
            and plan['test_data_used'] is False,'PLAN')
        require(str(out_dir)==plan['output_dir'] and out_dir.parent==ROOT and out_dir.name==OUTPUT_NAME
            and ROOT.resolve(strict=True)==ROOT and not out_dir.exists() and not out_dir.is_symlink(),'OUTPUT_SCOPE')
        require(plan['design']['path']==str(design_path_for_stage(STAGE)),'DESIGN_PATH')
        design=binding.read_bound_json(Path(plan['design']['path']),plan['design']['sha256'])
        require(design['schema_version']=='gx1_m5_momentum_intervention_design_v1' and design['status']=='PREDECLARED_BEFORE_INTERVENTION'
            and design['inputs']==plan['inputs'] and design['arms']==list(ARMS) and design['timeframes']==list(TIMEFRAMES)
            and design['batch_size']==BATCH and design['development_batch_stride']==STRIDE
            and design['headline_horizons_minutes']==list(HEADLINES) and design['bootstrap_replicates']==REPLICATES
            and design['paired_interval_alpha']==ALPHA and design['test_data_used'] is False,'DESIGN')
        policy_sha=common._hash_file(str(policy_path))['sha256'];policy_ref={'path':str(policy_path),'sha256':policy_sha}
        policy=binding.read_bound_json(policy_path,policy_sha);gate=policy[STAGE]
        require(gate['status']=='ADMITTED' and gate['plan']==plan_ref and gate['execution_authorized'] is True
            and gate['checkpoint_reads_authorized'] is gate['model_forwards_authorized'] is True
            and gate['max_invocations']==1 and gate['claimed_invocations']==gate['completed_invocations']==0
            and gate['relaunch_allowed'] is False,'ONE_SHOT_POLICY')
        require(all(policy[k] is False for k in ('training_enabled','native_training_enabled','full_epoch_training_allowed','full_val_allowed','native_launch_authorized'))
            and all(gate[k] is False for k in ('optimizer_steps_authorized','test_data_used','broker_access_allowed')),'CLOSED_GATES')
        require(set(plan['sources'])==set(SOURCE_PATHS) and all(ref['path']==str(REPO/name) for name,ref in plan['sources'].items()),'SOURCE_SET')
        for ref in plan['sources'].values():candle.regular(ref['path']);binding.require_binding(ref,label='intervention source')
        keys={'recipe','fit_review','fit_result','fit_receipt','checkpoint','oof_report','oof_predictions','train_report','train_predictions',
            'corrected_target','synthetic_test_terminal','synthetic_test_plan','comparison_result','similarity_review'}
        require(set(plan['inputs'])==keys,'INPUT_SET');meta={}
        for key in keys-{'checkpoint','oof_predictions','train_predictions','corrected_target'}:
            ref=plan['inputs'][key];meta[key]=binding.read_bound_json(Path(ref['path']),ref['sha256'])
        # The original complete fit validator ran before the runner was changed.
        # Recheck its immutable receipt, the historical Git blob, all 62 other
        # original source owners, and the currently bound runner separately.
        preflight=binding.read_bound_json(Path(plan['fit_source_preflight']['path']),plan['fit_source_preflight']['sha256'])
        previous_gate=policy['independent_m5_fit_source_preflight_20261029_003']
        require(previous_gate['status']=='COMPLETE_CONSUMED' and previous_gate['result']==plan['fit_source_preflight']
            and previous_gate['execution_authorized'] is False,'FIT_PREFLIGHT_GATE')
        fit=meta['fit_result'];review=meta['fit_review'];receipt=meta['fit_receipt'];oof=meta['oof_report'];train=meta['train_report']
        launch=binding.read_bound_json(Path(plan['inputs']['recipe']['path']),plan['inputs']['recipe']['sha256'])
        require(launch['recipe_sha256']==binding.canonical_sha256({k:v for k,v in launch.items() if k!='recipe_sha256'}),'FIT_RECIPE_SEAL')
        subprocess.run(['git','merge-base','--is-ancestor',review['source_commit'],preflight['source_head']],cwd=REPO,check=True)
        subprocess.run(['git','merge-base','--is-ancestor',preflight['source_head'],source_head],cwd=REPO,check=True)
        historical_runner=subprocess.check_output(['git','show',review['source_commit']+':scripts/gx1_capped_run.sh'],cwd=REPO)
        freeze_launch=verified_fit_source_transition(preflight,plan,launch,review,historical_runner)
        require(launch['run_scope']==fit_authority.M5_PRICE_FIT_SCOPE and launch['candidate_mode']=='price_only'
            and 'candidate_m1_refs' not in launch and launch['dense_target']==plan['inputs']['corrected_target']
            and review['status']=='PASS_CONSUMED_M5_PRICE_ONLY_FIT' and review['recipe']==plan['inputs']['recipe']
            and review['result']==plan['inputs']['fit_result'] and review['receipt']==plan['inputs']['fit_receipt']
            and fit['outcome']=='COMPLETE' and fit['recipe']==plan['inputs']['recipe'] and fit['source_unchanged'] is True
            and fit['qualification_weights_reused'] is False and fit['test_data_used'] is False
            and fit['full_state']==plan['inputs']['checkpoint'] and fit['oof_measurement']==plan['inputs']['oof_report']
            and fit['final_train_measurement']==plan['inputs']['train_report']
            and receipt['outcome']=='COMPLETE' and receipt['guard_decision']=='PASS' and receipt['trainer_guard_exit_code']==0
            and oof['role']=='OOF' and train['role']=='FINAL_TRAIN' and oof['fold']==train['fold']==fit['fold']==launch['fold']
            and oof['predictions']==plan['inputs']['oof_predictions'] and train['predictions']==plan['inputs']['train_predictions']
            and oof['test_data_used'] is train['test_data_used'] is False,'FIT_LINEAGE')
        require(meta['comparison_result']['status']=='COMPLETE' and meta['similarity_review']['status']=='COMPLETE'
            and meta['similarity_review']['comparison']==plan['inputs']['comparison_result']
            and meta['comparison_result']['test_data_used'] is meta['similarity_review']['test_data_used'] is False,'COMPARISON_LINEAGE')
        gc.require_synthetic_receipts(plan['inputs'],fit,meta['synthetic_test_terminal'],meta['synthetic_test_plan'])
        require(len(launch['source_owners'])==63,'FIT_SOURCE_COUNT')
        for name,ref in launch['source_owners'].items():
            if name=='scripts/gx1_capped_run.sh':continue
            candle.regular(ref['path']);binding.require_binding(ref,label='fit source')
        control_snapshot=binding.capture_verified_file_snapshot()
    hardware,lock=cap.require_guarded_cuda_producer_execution(),cap.require_capped_lock_ancestry()
    require(lock['scope']==hardware['cgroup_path'][len('/sys/fs/cgroup'):],'CAPPED_LOCK_SCOPE')
    out_dir.mkdir(exist_ok=False)
    def publish(path,value):publication._publish_bytes(path,binding.canonical_bytes(value))
    publish(out_dir/'INVOCATION_CLAIM.json',{'plan':plan_ref,'policy':policy_ref,'source_head':source_head,'execution':hardware,'lock':lock})
    gc._install_runtime_guard(out_dir,(plan['inputs']['synthetic_test_terminal']['path'],plan['inputs']['synthetic_test_plan']['path']))
    start=time.monotonic();old_profile=sys.getprofile();old_handler=signal.getsignal(signal.SIGALRM);handles=[];stage=out_dir/'.OUTPUT.staging'
    def expired(*_):raise RuntimeError('M5_MOMENTUM_INTERVENTION_WALL_LIMIT')
    def forbid_updates(frame,event,arg):
        if event=='call':
            name,path=frame.f_code.co_name,frame.f_code.co_filename.replace('\\','/')
            require(not (('/torch/optim/' in path and name in ('step','_init_group'))
                or ('/torch/autograd/' in path and name in ('backward','_engine_run_backward'))),'FORBIDDEN_UPDATE')
    signal.signal(signal.SIGALRM,expired);signal.alarm(plan['wall_seconds']);sys.setprofile(forbid_updates)
    try:
        stage.mkdir(exist_ok=False);input_launch=window._fit_input_launch(launch,fit_authority);context=window._launch_metadata(input_launch,common)
        require(context.get('candidate_inputs')=={'mode':'price_only'} and 'candidate_m1_refs' not in context,'M5_ONLY_CONTEXT')
        payloads=[plan['inputs'][k] for k in ('checkpoint','oof_predictions','train_predictions','corrected_target')]
        require(len({ref['path'] for ref in payloads})==len(payloads),'DISTINCT_PAYLOADS')
        for ref in payloads:
            before=common._stat(ref['path']);require(common._hash_file(ref['path'])==ref,'PAYLOAD_SHA')
            context['info']['metadata'][ref['path']]={'ref':dict(ref),'stat':before}
        freeze={'source_head':source_head,'policy':policy_ref};common._freeze(freeze_launch,freeze,context)
        denied=common._install_read_guard(context,out_dir,out_dir,dense_target=context['dense_target'],decimal_original=context['decimal_original'])
        prepared,targets,view,_map,_order=window._prepare(context,input_launch,np,dense_minutes=True)
        names=('parent_rows','bar_start_ns','decision_ns','terminal_time_ns','observed','sign','logit_up');saved={}
        for role in ('train','oof'):
            with np.load(plan['inputs'][role+'_predictions']['path'],allow_pickle=False) as archive:
                require(all(name in archive.files for name in names),'SAVED_FIELDS:'+role);saved[role]={k:archive[k].copy() for k in names}
            arr=saved[role];rows=arr['parent_rows'];at=np.searchsorted(prepared['arrays']['parent_rows'],rows)
            require(np.all(at<len(prepared['arrays']['parent_rows'])) and np.array_equal(prepared['arrays']['parent_rows'][at],rows),'PARENT_JOIN:'+role)
            for name in ('bar_start_ns','decision_ns'):require(np.array_equal(arr[name],prepared['arrays'][name][at]),'AXIS:'+role+':'+name)
            for name in ('terminal_time_ns','observed','sign'):require(np.array_equal(arr[name],targets[name][at]),'TARGET:'+role+':'+name)
            require(arr['logit_up'].dtype==np.float32 and arr['logit_up'].shape==(len(rows),60) and np.isfinite(arr['logit_up']).all(),'LOGITS:'+role)
        fold=launch['fold'];require(len(saved['train']['parent_rows'])==256 and np.all(saved['train']['parent_rows']<fold['prefix_rows'])
            and np.array_equal(saved['oof']['parent_rows'],np.arange(fold['oof_start_row'],fold['oof_end_row'],dtype=np.int64)),'SAMPLE_POPULATIONS')
        selections={'train':[np.arange(i,min(i+BATCH,256),dtype=np.int64) for i in range(0,256,BATCH)],
            'oof':gc.selected_batches(len(saved['oof']['parent_rows']))}
        trainer._set_deterministic(launch['numerical_recipe']['seed'],torch.device('cuda',0),launch['numerical_recipe']['precision_policy'])
        torch.set_num_threads(1);require(torch.get_num_threads()==1,'NUMERICAL_THREADS')
        model=build_native_direction_research_model(view,prefix_normalization=context['fitted'],constructor_metadata=context['constructor'],device=torch.device('cuda',0))
        require(model.feature_view_identity==fit['feature_view_identity'] and model.backbone.entry_execution_profile['name']==PROFILE_NAME,'VIEW_IDENTITY')
        state=torch.load(plan['inputs']['checkpoint']['path'],map_location='cpu',weights_only=True)
        require(isinstance(state,dict) and set(state)==state_owner.STATE_KEYS and state['contract_sha256']==state_owner._tree_sha({k:v for k,v in state.items() if k!='contract_sha256'})
            and state['contract_sha256']==fit['durable_checkpoint']['contract_sha256'] and state['model_state_sha256']==state_owner.canonical_model_state_sha256(state['model_state'])
            and state['binding']['feature_view_identity']==model.feature_view_identity and state['cursor']==47602 and state['optimizer_updates']==2976,'CHECKPOINT_SEAL')
        model.load_state_dict(state['model_state'],strict=True)
        with torch.no_grad():
            for name,buffer in model.named_buffers():buffer.copy_(state['buffer_state'][name].to(buffer.device))
        require(state_owner.canonical_model_state_sha256(model.state_dict())==state['model_state_sha256']
            and state_owner._tree_sha(dict(model.named_buffers()))==state['buffer_state_sha256'],'LOADED_STATE')
        model.backbone.require_input_normalization_state();model._require_binding();model.eval();model_sha,buffer_sha=state['model_state_sha256'],state['buffer_state_sha256'];del state
        layer=model.backbone.mtf_family_proj[MOMENTUM_FAMILY];fields=model.backbone._input_normalization_field_names['mtf_m5'];indices=model.backbone._multi_tf_specialist_index_tuples[MOMENTUM_FAMILY]
        routed=tuple(fields[i] for i in indices);require(len(routed)==4 and set(routed)==set(PRICE_MOMENTUM_FIELDS),'FIELD_ROUTE')
        columns={name:(routed.index(name),) for name in ARMS[:-1]};columns['all_four']=tuple(range(4))
        total_batches=sum(len(v) for v in selections.values());ceiling=total_batches*(1+len(ARMS));counters={'forward_calls':0,'forward_returns':0,'forward_kind_calls':{}}
        paired={};observations={};call_counts={name:0 for name in ARMS};batch_counts={}
        for role in ('train','oof'):
            sample=saved[role];outputs={'original':[],**{name:[] for name in ARMS}};chosen=selections[role];obs_total={}
            for ids in chosen:
                require(time.monotonic()-start<plan['wall_seconds'],'WALL_LIMIT')
                raw,target=window._batch(prepared,targets,view,sample['parent_rows'][ids],np);features,_unused=window._device_batch(raw,target,torch,torch.device('cuda',0))
                observer,calls,obs=install_input_observer(layer,routed);handles=[observer]
                try:z0=window._forward(model,features,kind=role+'_REPLAY',counters=counters,ceiling=ceiling,torch=torch).detach().cpu().numpy().copy()
                finally:observer.remove();handles=[]
                require(tuple(calls)==TIMEFRAMES,'OBSERVED_TF_ORDER');accumulate_observations(obs_total,obs,routed)
                gc.require_exact_replay(z0,np.ascontiguousarray(sample['logit_up'][ids]));outputs['original'].append(z0)
                for name in ARMS:
                    handle,calls=install_neutralization(layer,columns[name]);handles=[handle]
                    try:z=window._forward(model,features,kind=role+'_'+name,counters=counters,ceiling=ceiling,torch=torch).detach().cpu().numpy().copy()
                    finally:handle.remove();handles=[]
                    require(len(calls)==len(TIMEFRAMES) and np.isfinite(z).all(),'ABLATION_CALLS:'+name);call_counts[name]+=len(calls);outputs[name].append(z)
                del raw,target,features,_unused,z0,z
            selected=np.concatenate(chosen);rows={k:np.ascontiguousarray(sample[k][selected]) for k in ('parent_rows','decision_ns','terminal_time_ns','observed','sign')}
            paired[role]={**rows,**{name:np.concatenate(values) for name,values in outputs.items()}};observations[role]=obs_total;batch_counts[role]=len(chosen)
        require(counters['forward_calls']==counters['forward_returns']==ceiling and all(n==total_batches*len(TIMEFRAMES) for n in call_counts.values()) and not denied,'CALL_ACCOUNTING')
        require(state_owner.canonical_model_state_sha256(model.state_dict())==model_sha and state_owner._tree_sha(dict(model.named_buffers()))==buffer_sha
            and all(p.grad is None for p in model.parameters()),'MODEL_STATE_UNCHANGED')
        original_alpha=gc.ALPHA;gc.ALPHA=ALPHA
        try:
            metrics={};effects={}
            for role in ('train','oof'):
                item=paired[role];cutoff=fold['train_cutoff_ns'] if role=='train' else fold['oof_cutoff_ns'];effects[role]={}
                for name in ARMS:
                    effects[role][name]=influence_summary(item['sign'],item['observed'],item['terminal_time_ns'],item['original'],item[name],cutoff)
                    if role=='oof':
                        report=gc.paired_metrics(item['decision_ns'],item['observed'],item['sign'],item['terminal_time_ns'],item['original'],item[name],cutoff=cutoff)
                        report['scope']='Prespecified every-tenth original 16-row OOF batch on one reused development fold; no retraining.'
                        report['limitations']='This zero-input intervention acts on the selected XAU price projection only, retains all GC and other paths, and may create an out-of-distribution hidden state. Calendar-block exchangeability and economic edge are unproved.'
                        metrics[name]=report
        finally:gc.ALPHA=original_alpha
        npz_path=stage/'PAIRED_OUTPUTS.npz'
        with npz_path.open('xb') as handle:
            np.savez_compressed(handle,**{role+'__'+key:value for role,items in paired.items() for key,value in items.items()});handle.flush();os.fsync(handle.fileno())
        common._freeze(freeze_launch,freeze,context);control_snapshot.require_unchanged();require(cap.require_capped_lock_ancestry()==lock and not denied,'FINAL_SOURCE_LOCK')
        result={'schema_version':'gx1_m5_momentum_intervention_result_v1','status':'COMPLETE','plan':plan_ref,'source_head':source_head,
            'fit_recipe':plan['inputs']['recipe'],'fit_source_preflight':plan['fit_source_preflight'],
            'historical_runner':launch['source_owners']['scripts/gx1_capped_run.sh'],'current_runner':plan['sources']['scripts/gx1_capped_run.sh'],
            'checkpoint':plan['inputs']['checkpoint'],'fold':fold,'field_order':list(routed),
            'selection':{'train_saved_fixed_parents':len(paired['train']['parent_rows']),'train_batches':batch_counts['train'],
                'oof_every_tenth_original_batch_parents':len(paired['oof']['parent_rows']),'oof_batches':batch_counts['oof'],'batch_size':BATCH,'batch_stride':STRIDE},
            'exact_saved_train_and_oof_logits_reproduced':True,'intervention':'Each selected XAU price feature is set to zero at its normalized, gated input to the shared momentum projection on all five timeframes; GC, local aliases, other inputs and projection bias remain.',
            'observation_counts':observations,'influence':effects,'paired_oof_metrics':metrics,'paired_interval_alpha':ALPHA,
            'forward_calls':counters['forward_calls'],'optimizer_steps':0,'model_state_unchanged':True,'test_data_used':False,'economic_edge_qualified':False,
            'paired_outputs':{'path':str(out_dir/'OUTPUT/PAIRED_OUTPUTS.npz'),'sha256':common._hash_file(str(npz_path))['sha256']},
            'limitations':'Each zeroed value is an internal intervention that may be out of distribution; correlated or alternative routes remain. Results do not isolate retraining or prove achievable trading edge.'}
        publish(stage/'RESULT.json',result);publish(stage/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'source_unchanged':True,'inputs_unchanged':True,'elapsed_seconds':time.monotonic()-start,'test_data_used':False})
        publish_bundle_directory_noreplace(stage,out_dir/'OUTPUT');publish(out_dir/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'plan':plan_ref,'result_path':str(out_dir/'OUTPUT/RESULT.json')});return result
    except Exception as exc:
        publish(out_dir/'FAILED_TERMINAL.json',{'status':'FAILED','exit_code':1,'error':type(exc).__name__+': '+str(exc),'relaunch_allowed':False});raise
    finally:
        for handle in handles:handle.remove()
        sys.setprofile(old_profile);signal.alarm(0);signal.signal(signal.SIGALRM,old_handler)


# Separate exact TRAIN-only diagnostic; the consumed intervention scopes above
# retain their original no-reverse behavior.
UPSTREAM_ROOT = Path('/home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19')
UPSTREAM_STAGE = 'direction_family_upstream_gradient_diagnostic_20261031_001'
UPSTREAM_PLAN = UPSTREAM_ROOT/'UPSTREAM_GRADIENT_PLAN_20261031_001.json'
UPSTREAM_OUTPUT = UPSTREAM_ROOT/'UPSTREAM_GRADIENT_20261031_001'
UPSTREAM_DESIGN = REPO/'docs/DIRECTION_FAMILY_UPSTREAM_GRADIENT_DESIGN_20261031.json'
UPSTREAM_CHANGED_OWNERS = frozenset((
    'gx1/contracts/entry_native_direction_prefix_fit_campaign_v1.py',
    'gx1/contracts/entry_native_direction_research_campaign_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
    'gx1/scripts/local_random_access_campaign_v2.py',
    'scripts/gx1_capped_run.sh',
))
UPSTREAM_EXTRA_SOURCES = (
    'gx1/audit/direction_m5_momentum_intervention_v1.py',
    'gx1/audit/direction_m5_momentum_trace_v1.py',
    'gx1/audit/direction_gc_mtf_intervention_v1.py',
    'gx1/scripts/validate_entry_model_native_technical_checkpoint_v1.py',
    'gx1/scripts/audit_entry_direction_candle_midpoints_v1.py',
    'tests/test_direction_m5_momentum_intervention_v1.py',
    'tests/test_direction_m5_momentum_trace_v1.py',
    'tests/test_direction_m5_momentum_power_guard_v1.py',
)


def upstream_selected_predictions(initial, terminal, prepared, targets, prefix_rows):
    """Exact ordered INITIAL axes, joined to terminal TRAIN predictions."""
    axes = ('parent_rows','bar_start_ns','decision_ns','terminal_time_ns','observed','sign')
    require(all(k in initial and k in terminal for k in axes)
        and 'logit_up' in terminal, 'UPSTREAM_SAVED_FIELDS')
    rows = initial['parent_rows']
    require(rows.dtype == np.int64 and rows.shape == (256,)
        and len(np.unique(rows)) == 256 and np.all((rows >= 0) & (rows < prefix_rows))
        and np.array_equal(terminal['parent_rows'], np.arange(prefix_rows,dtype=np.int64)),
        'UPSTREAM_TRAIN_POPULATION')
    at = np.searchsorted(prepared['parent_rows'], rows)
    require(np.all(at < len(prepared['parent_rows']))
        and np.array_equal(prepared['parent_rows'][at], rows), 'UPSTREAM_PARENT_JOIN')
    for key in axes[1:]:
        require(np.array_equal(initial[key], terminal[key][rows]), 'UPSTREAM_TERMINAL_JOIN:'+key)
        actual = prepared[key] if key in ('bar_start_ns','decision_ns') else targets[key]
        require(np.array_equal(initial[key], actual[at]), 'UPSTREAM_TARGET_JOIN:'+key)
    logits = terminal['logit_up']
    require(logits.dtype == np.float32 and logits.shape == (prefix_rows,60)
        and np.isfinite(logits).all(), 'UPSTREAM_TERMINAL_LOGITS')
    return {**{k:np.ascontiguousarray(initial[k]) for k in axes},
        'logit_up':np.ascontiguousarray(logits[rows])}


def upstream_source_transition(launch, sources, proof, historical):
    """Bind 65 original owners and explicit control-only transitions."""
    owners=launch['source_owners']
    require(len(owners)==65 and set(historical)==set(owners)
        and set(sources)==set(owners)|set(UPSTREAM_EXTRA_SOURCES)
        and proof['schema_version']=='gx1_family_upstream_source_transition_v1'
        and proof['status']=='SOURCE_ONLY_REVIEWED'
        and proof['original_sources']==owners
        and proof['current_sources']=={name:sources[name] for name in owners}
        and proof['changed_paths']==sorted(UPSTREAM_CHANGED_OWNERS)
        and proof['numerical_model_sources_unchanged'] is True
        and proof['test_data_used'] is False, 'UPSTREAM_SOURCE_TRANSITION')
    changed=set()
    for name, old in owners.items():
        new=sources[name]
        require(old['path']==new['path']==str(REPO/name)
            and hashlib.sha256(historical[name]).hexdigest()==old['sha256'],
            'UPSTREAM_HISTORICAL_SOURCE:'+name)
        if old != new:changed.add(name)
    require(changed==UPSTREAM_CHANGED_OWNERS,'UPSTREAM_SOURCE_DELTA')
    return {**launch,'source_owners':{name:sources[name] for name in owners}}


def upstream_profile_budget(codes, phase, counts):
    """Count real Python boundaries; reverse is admitted only in this scope."""
    def profile(frame,event,arg):
        code=frame.f_code;item=codes.get(code)
        if item is not None:
            name,allowed,ceiling=item
            if event=='call':
                require(phase['name']==allowed,'UPSTREAM_CALL_PHASE:'+name)
                counts[name+'_calls']=counts.get(name+'_calls',0)+1
                require(counts[name+'_calls']<=ceiling,'UPSTREAM_CALL_BUDGET:'+name)
            elif event=='return':
                counts[name+'_returns']=counts.get(name+'_returns',0)+1
        if event=='call':
            path=code.co_filename.replace('\\','/');name=code.co_name
            forbidden=(('/torch/optim/' in path and name in ('__init__','step','_init_group'))
                or ('/torch/' in path and name=='backward')
                or (path.endswith('/torch/serialization.py') and name=='save')
                or ('/site-packages/sklearn/' in path and (name=='fit' or name.startswith('fit_')))
                or (path.startswith(str(REPO)+'/') and name in
                    ('_optimizer_step_with_finite_gradients','_save_state','_save_snapshot')))
            require(not forbidden,'UPSTREAM_FORBIDDEN_OPERATION:'+name)
            if path.endswith('/torch/nn/modules/module.py') and name=='__init__':
                require(phase['name']=='CONSTRUCT','UPSTREAM_EXTRA_MODULE')
    return profile


def run_upstream(*,plan_path,plan_sha256,device,out_dir):
    """One frozen family-baseline checkpoint; 256 TRAIN, no optimizer."""
    from gx1.audit import direction_gc_mtf_intervention_v1 as gc
    from gx1.audit.direction_m5_momentum_trace_v1 import UpstreamGradientRecorder
    from gx1.contracts import gx1_capped_execution_v1 as cap,local_random_access_campaign_v2 as binding
    from gx1.contracts import immutable_event_authority_v1 as publication,entry_native_direction_research_state_v1 as state_owner
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as authority
    from gx1.contracts.entry_native_direction_feature_view_v1 import build_native_direction_research_model
    from gx1.contracts.entry_native_direction_dense_v1 import DenseNativeEntryDirectionResearch
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as window
    from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
    sys.dont_write_bytecode=True
    require(device=='cuda' and not torch.cuda.is_initialized(),'UPSTREAM_EXPLICIT_CUDA')
    require(Path(plan_path)==UPSTREAM_PLAN and Path(out_dir)==UPSTREAM_OUTPUT,'UPSTREAM_EXACT_PATHS')
    out_dir=Path(out_dir);plan_ref={'path':str(candle.regular(plan_path)),'sha256':plan_sha256}
    source_head,branch,dirty=candle.git_state()
    require(branch=='work/gx1-current' and not dirty,'UPSTREAM_INITIAL_SOURCE')
    policy_path=REPO/'NEXT_RUN_POLICY.json'
    with binding.fresh_file_verification_scope():
        plan=binding.read_bound_json(Path(plan_ref['path']),plan_ref['sha256'])
        require(set(plan)=={'schema_version','stage','design','inputs','sources','output_dir','wall_seconds','batch_size','train_parents','forward_calls','reverse_calls','test_data_used'}
            and plan['schema_version']=='gx1_family_upstream_gradient_plan_v1' and plan['stage']==UPSTREAM_STAGE
            and plan['output_dir']==str(out_dir) and plan['wall_seconds']==3300
            and plan['batch_size']==16 and plan['train_parents']==256
            and plan['forward_calls']==16 and plan['reverse_calls']==32 and plan['test_data_used'] is False,'UPSTREAM_PLAN')
        require(out_dir.parent.resolve(strict=True)==UPSTREAM_ROOT
            and not out_dir.exists() and not out_dir.is_symlink(),'UPSTREAM_OUTPUT')
        require(plan['design']['path']==str(UPSTREAM_DESIGN),'UPSTREAM_DESIGN_PATH')
        design=binding.read_bound_json(UPSTREAM_DESIGN,plan['design']['sha256'])
        require(design['schema_version']=='gx1_family_upstream_gradient_design_v1'
            and design['inputs']==plan['inputs'] and design['stage']==UPSTREAM_STAGE
            and design['train_parents']==256 and design['batch_size']==16
            and design['forward_calls']==16 and design['reverse_calls']==32
            and design['optimizer_constructions']==design['optimizer_updates']==0
            and design['test_data_used'] is False,'UPSTREAM_DESIGN')
        policy_ref=common._hash_file(str(policy_path));policy=binding.read_bound_json(policy_path,policy_ref['sha256']);gate=policy[UPSTREAM_STAGE]
        require(gate['status']=='ADMITTED' and gate['plan']==plan_ref
            and all(gate[k] is True for k in ('execution_authorized','genuine_constructor_authorized','genuine_forward_authorized','genuine_backward_authorized'))
            and gate['max_invocations']==1 and gate['claimed_invocations']==gate['completed_invocations']==0
            and gate['relaunch_allowed'] is False,'UPSTREAM_ONE_SHOT')
        require(all(policy[k] is False for k in ('training_enabled','native_training_enabled','full_epoch_training_allowed','full_val_allowed','native_launch_authorized'))
            and all(gate[k] is False for k in ('genuine_updates_authorized','training_enabled','native_launch_authorized','test_data_used','exit_evaluation','broker_access_allowed')),'UPSTREAM_CLOSED_GATES')
        refs=plan['inputs']
        expected={'recipe','fit_review','fit_result','fit_receipt','baseline_terminal','checkpoint','initial_report','diagnostic_axes',
            'train_report','terminal_predictions','synthetic_test_terminal','synthetic_test_plan','source_transition'}
        require(set(refs)==expected,'UPSTREAM_INPUT_SET')
        meta={k:binding.read_bound_json(Path(refs[k]['path']),refs[k]['sha256'])
            for k in expected-{'checkpoint','diagnostic_axes','terminal_predictions'}}
        launch=meta['recipe'];review=meta['fit_review'];fit=meta['fit_result'];receipt=meta['fit_receipt']
        baseline=policy['dynamic_m5_direction_family_baseline_onepass_fit_20261030_001']
        require(baseline['status']=='COMPLETE_CONSUMED_BASELINE_FULLTRAIN_LATER_NO_EDGE_QUALIFIED'
            and baseline['recipe']==refs['recipe'] and baseline['root_review']==refs['fit_review']
            and baseline['verified_terminal']==refs['baseline_terminal']
            and meta['baseline_terminal']['root_review']==refs['fit_review']
            and meta['baseline_terminal']['status']=='COMPLETE' and meta['baseline_terminal']['exit_code']==0,'UPSTREAM_BASELINE_POLICY')
        require(launch['recipe_sha256']==binding.canonical_sha256({k:v for k,v in launch.items() if k!='recipe_sha256'})
            and launch['run_scope']==authority.FAMILY_SIGN_SCOPE and launch['family_arm']=='baseline'
            and launch['candidate_mode']=='price_only' and 'candidate_m1_refs' not in launch
            and review['status']=='PASS_BASELINE_ONEPASS_FULLTRAIN_LATER_TERMINAL'
            and review['result']==refs['fit_result'] and review['receipt']==refs['fit_receipt']
            and review['full_state']==refs['checkpoint'] and review['full_train_predictions']==refs['terminal_predictions']
            and review['initial_measurement']==refs['initial_report']
            and fit['outcome']=='COMPLETE' and fit['recipe']==refs['recipe'] and fit['source_unchanged'] is True
            and fit['qualification_weights_reused'] is False and fit['full_state']==refs['checkpoint']
            and fit['initial_measurement']==refs['initial_report'] and fit['final_train_measurement']==refs['train_report']
            and receipt['outcome']=='COMPLETE' and receipt['guard_decision']=='PASS' and receipt['trainer_guard_exit_code']==0
            and fit['optimizer_updates']==2976 and fit['parent_cursor']==47602
            and all(meta[k]['test_data_used'] is False for k in ('fit_result','fit_review','baseline_terminal','initial_report','train_report')),
            'UPSTREAM_FIT_LINEAGE')
        for key,role,payload in (('initial_report','INITIAL_TRAIN','diagnostic_axes'),('train_report','FINAL_TRAIN','terminal_predictions')):
            record=meta[key]
            require(record['role']==role and record['predictions']==refs[payload]
                and record['fold']==launch['fold']==fit['fold']
                and record['feature_view_identity']==fit['feature_view_identity'],'UPSTREAM_MEASUREMENT:'+key)
        synthetic=meta['synthetic_test_terminal']
        require(synthetic['status']=='COMPLETE_SYNTHETIC_ONLY_NO_GENUINE_ADMISSION'
            and synthetic['plan']==refs['synthetic_test_plan'] and synthetic['exit_code']==0
            and synthetic['tests_passed']>0 and synthetic['genuine_payloads_or_fit_admitted'] is False
            and meta['synthetic_test_plan']['sources']==plan['sources'],'UPSTREAM_SYNTHETIC')
        proof=meta['source_transition']
        require(proof['baseline_source_head']==review['source_head']
            and proof['recipe']==refs['recipe'] and proof['baseline_review']==refs['fit_review'],'UPSTREAM_SOURCE_ORIGIN')
        subprocess.run(['git','merge-base','--is-ancestor',review['source_head'],source_head],cwd=REPO,check=True)
        historical={name:(subprocess.check_output(['git','show',review['source_head']+':'+name],cwd=REPO)
            if name in UPSTREAM_CHANGED_OWNERS else Path(ref['path']).read_bytes())
            for name,ref in launch['source_owners'].items()}
        freeze_launch=upstream_source_transition(launch,plan['sources'],proof,historical)
        for name,ref in plan['sources'].items():
            require(ref['path']==str(REPO/name),'UPSTREAM_SOURCE_PATH')
            candle.regular(ref['path']);binding.require_binding(ref,label='upstream diagnostic source')
        control_snapshot=binding.capture_verified_file_snapshot()
    hardware,lock=cap.require_guarded_cuda_producer_execution(),cap.require_capped_lock_ancestry()
    require(lock['scope']==hardware['cgroup_path'][len('/sys/fs/cgroup'):],'UPSTREAM_CAPPED_LOCK')
    out_dir.mkdir(exist_ok=False)
    def publish(path,value):publication._publish_bytes(path,binding.canonical_bytes(value))
    publish(out_dir/'INVOCATION_CLAIM.json',{'plan':plan_ref,'policy':policy_ref,'source_head':source_head,'execution':hardware,'lock':lock})
    gc._install_runtime_guard(out_dir,(refs['synthetic_test_terminal']['path'],refs['synthetic_test_plan']['path']))
    start=time.monotonic();old_profile=sys.getprofile();old_handler=signal.getsignal(signal.SIGALRM)
    phase={'name':'PREPARE'};counts={};model=None;modes=None;identity=None
    codes={fn.__code__:(name,p,n) for fn,name,p,n in (
        (build_native_direction_research_model,'model_constructor','CONSTRUCT',1),
        (torch.load,'checkpoint_load','LOAD',1),
        (EntryV10CtxHybridTransformer.forward,'backbone_forward','FORWARD',16),
        (DenseNativeEntryDirectionResearch.forward,'research_forward','FORWARD',16),
        (trainer._model_forward_fp32,'trainer_forward','FORWARD',16),
        (torch.autograd.grad,'autograd_grad','REVERSE',32),
        (torch.autograd.graph._engine_run_backward,'engine_reverse','REVERSE',32))}
    def expired(*_):raise RuntimeError('UPSTREAM_GRADIENT_WALL_LIMIT')
    signal.signal(signal.SIGALRM,expired);signal.alarm(plan['wall_seconds'])
    sys.setprofile(upstream_profile_budget(codes,phase,counts))
    stage=out_dir/'.OUTPUT.staging'
    try:
        stage.mkdir(exist_ok=False)
        input_launch=window._fit_input_launch(launch,authority);context=window._launch_metadata(input_launch,common)
        require(context.get('candidate_inputs')=={'mode':'price_only'},'UPSTREAM_PRICE_CONTEXT')
        for key in ('checkpoint','diagnostic_axes','terminal_predictions'):
            ref=refs[key];before=common._stat(ref['path'])
            require(common._hash_file(ref['path'])==ref,'UPSTREAM_PAYLOAD_SHA:'+key)
            context['info']['metadata'][ref['path']]={'ref':dict(ref),'stat':before}
        freeze={'source_head':source_head,'policy':policy_ref};common._freeze(freeze_launch,freeze,context)
        denied=common._install_read_guard(context,out_dir,out_dir,dense_target=context['dense_target'],decimal_original=context['decimal_original'])
        prepared,targets,view,_map,_order=window._prepare(context,input_launch,np,dense_minutes=True)
        saved={}
        axes=('parent_rows','bar_start_ns','decision_ns','terminal_time_ns','observed','sign')
        for key,names in (('diagnostic_axes',axes),('terminal_predictions',axes+('logit_up',))):
            with np.load(refs[key]['path'],allow_pickle=False) as archive:
                require(all(name in archive.files for name in names),'UPSTREAM_ARCHIVE_FIELDS')
                saved[key]={name:archive[name].copy() for name in names}
        sample=upstream_selected_predictions(saved['diagnostic_axes'],saved['terminal_predictions'],
            prepared['arrays'],targets,launch['fold']['prefix_rows']);del saved
        dev=torch.device('cuda',0)
        trainer._set_deterministic(launch['numerical_recipe']['seed'],dev,launch['numerical_recipe']['precision_policy'])
        torch.set_num_threads(1);require(torch.get_num_threads()==1,'UPSTREAM_THREADS')
        phase['name']='CONSTRUCT'
        model=build_native_direction_research_model(view,prefix_normalization=context['fitted'],constructor_metadata=context['constructor'],device=dev)
        require(model.feature_view_identity==fit['feature_view_identity'],'UPSTREAM_VIEW')
        phase['name']='LOAD';state=torch.load(refs['checkpoint']['path'],map_location='cpu',weights_only=True)
        require(isinstance(state,dict) and set(state)==state_owner.STATE_KEYS
            and state['contract_sha256']==state_owner._tree_sha({k:v for k,v in state.items() if k!='contract_sha256'})
            and state['contract_sha256']==fit['durable_checkpoint']['contract_sha256']
            and state['model_state_sha256']==state_owner.canonical_model_state_sha256(state['model_state'])
            and state['binding']['feature_view_identity']==model.feature_view_identity
            and state['cursor']==47602 and state['optimizer_updates']==2976,'UPSTREAM_CHECKPOINT')
        model.load_state_dict(state['model_state'],strict=True)
        with torch.no_grad():
            for name,buffer in model.named_buffers():buffer.copy_(state['buffer_state'][name].to(buffer.device))
        require(state_owner.canonical_model_state_sha256(model.state_dict())==state['model_state_sha256']
            and state_owner._tree_sha(dict(model.named_buffers()))==state['buffer_state_sha256'],'UPSTREAM_LOADED_STATE')
        # Restore exactly the checkpoint's active parameter names without AdamW.
        pairs,active,inactive=state_owner._parameters(model)
        require([name for name,_ in active]==state['binding']['active_parameter_names']
            and [name for name,_ in inactive]==state['binding']['inactive_parameter_names']
            and [[name for name,_ in active]]==state['binding']['optimizer_group_parameter_names'],
            'UPSTREAM_ACTIVE_PARAMETERS')
        for _,param in active:param.requires_grad_(True)
        for _,param in inactive:param.requires_grad_(False)
        model.backbone.require_input_normalization_state();model._require_binding();del state
        require(all(p.grad is None for p in model.parameters()),'UPSTREAM_INITIAL_GRADS')
        identity=window._online_identity(model,dev,state_owner._tree_sha,trainer)
        modes=[(module,module.training) for module in model.modules()];model.eval()
        recorder=UpstreamGradientRecorder();forward={'forward_calls':0,'forward_returns':0,'forward_kind_calls':{}}
        for offset in range(0,256,16):
            raw,target=window._batch(prepared,targets,view,sample['parent_rows'][offset:offset+16],np)
            features,target_device=window._device_batch(raw,target,torch,dev)
            phase['name']='FORWARD'
            with recorder.capture(model):
                logits=window._forward(model,features,kind='TRAIN_UPSTREAM_DIAGNOSTIC',counters=forward,ceiling=16,torch=torch,grad=True)
            gc.require_exact_replay(logits.detach().cpu().numpy(),sample['logit_up'][offset:offset+16])
            phase['name']='REVERSE'
            recorder.measure(logits,target_device,launch['fold']['train_cutoff_ns'])
            del raw,target,features,target_device,logits
        phase['name']='FINAL'
        for module,training in modes:module.training=training
        require(window._online_identity(model,dev,state_owner._tree_sha,trainer)==identity
            and all(p.grad is None for p in model.parameters()),'UPSTREAM_STATE_PRESERVATION')
        for name,_p,n in codes.values():
            require(counts.get(name+'_calls')==counts.get(name+'_returns')==n,'UPSTREAM_EXACT_CALLS:'+name)
        require(forward['forward_calls']==forward['forward_returns']==16 and not denied,'UPSTREAM_FORWARD_READS')
        geometry=recorder.result()
        require(geometry['batch_count']==16 and geometry['parents']==256 and geometry['reverse_calls']==32,'UPSTREAM_GEOMETRY_COUNTS')
        common._freeze(freeze_launch,freeze,context);control_snapshot.require_unchanged()
        require(cap.require_capped_lock_ancestry()==lock,'UPSTREAM_FINAL_LOCK')
        result={'schema_version':'gx1_family_upstream_gradient_result_v1','status':'COMPLETE','plan':plan_ref,
            'source_head':source_head,'checkpoint':refs['checkpoint'],'train_parents':256,'batch_size':16,
            'forward_calls':16,'reverse_calls':32,'call_counts':counts,'optimizer_constructions':0,'optimizer_updates':0,
            'saved_terminal_logits_exact':True,'axes_and_targets_exact':True,'state_preservation':identity,
            'model_buffers_rng_modes_unchanged':True,'parameter_grads_all_none':True,'geometry':geometry,
            'test_data_used':False,'later_rows_forwarded_or_scored':0,'economic_edge_qualified':False,
            'limitations':design['limitations']}
        publish(stage/'RESULT.json',result)
        publish(stage/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'elapsed_seconds':time.monotonic()-start,
            'source_unchanged':True,'inputs_unchanged':True,'test_data_used':False})
        publish_bundle_directory_noreplace(stage,out_dir/'OUTPUT')
        publish(out_dir/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'plan':plan_ref,'result_path':str(out_dir/'OUTPUT/RESULT.json')})
        return result
    except Exception as exc:
        publish(out_dir/'FAILED_TERMINAL.json',{'status':'FAILED','exit_code':1,'error':type(exc).__name__+': '+str(exc),'relaunch_allowed':False})
        raise
    finally:
        if modes is not None:
            for module,training in modes:module.training=training
        sys.setprofile(old_profile);signal.alarm(0);signal.signal(signal.SIGALRM,old_handler)
