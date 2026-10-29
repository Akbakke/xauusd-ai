"""Fixed-model price momentum interventions on qualified M5 TRAIN/development rows.

Only the four XAU MTF momentum columns are zeroed at their normalized, gated
projection input. Other routes, including GC order flow, stay intact. This is
an offline sensitivity test, not an executable model or a trading rule.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
import torch

REPO=Path('/home/andre2/src/GX1_CURRENT')
ROOT=Path('/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63')
STAGE='independent_m5_momentum_intervention_20261029_001'
OUTPUT_NAME='DIRECTION_M5_MOMENTUM_INTERVENTION_20261029_001'
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
        require(set(plan)=={'schema_version','stage','design','inputs','sources','output_dir','wall_seconds','batch_size','batch_stride','test_data_used'}
            and plan['schema_version']=='gx1_m5_momentum_intervention_plan_v1' and plan['stage']==STAGE
            and plan['wall_seconds']==3300 and plan['batch_size']==BATCH and plan['batch_stride']==STRIDE
            and plan['test_data_used'] is False,'PLAN')
        require(str(out_dir)==plan['output_dir'] and out_dir.parent==ROOT and out_dir.name==OUTPUT_NAME
            and ROOT.resolve(strict=True)==ROOT and not out_dir.exists() and not out_dir.is_symlink(),'OUTPUT_SCOPE')
        require(plan['design']['path']==str(REPO/'docs/DIRECTION_M5_MOMENTUM_INTERVENTION_20261029.json'),'DESIGN_PATH')
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
        launch=fit_authority.require_prefix_fit_launch_recipe(plan['inputs']['recipe'],source_repo=REPO,source_commit=source_head)
        fit=meta['fit_result'];review=meta['fit_review'];receipt=meta['fit_receipt'];oof=meta['oof_report'];train=meta['train_report']
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
        for ref in launch['source_owners'].values():candle.regular(ref['path']);binding.require_binding(ref,label='fit source')
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
        freeze={'source_head':source_head,'policy':policy_ref};common._freeze(launch,freeze,context)
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
        common._freeze(launch,freeze,context);control_snapshot.require_unchanged();require(cap.require_capped_lock_ancestry()==lock and not denied,'FINAL_SOURCE_LOCK')
        result={'schema_version':'gx1_m5_momentum_intervention_result_v1','status':'COMPLETE','plan':plan_ref,'source_head':source_head,
            'fit_recipe':plan['inputs']['recipe'],'checkpoint':plan['inputs']['checkpoint'],'fold':fold,'field_order':list(routed),
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
