"""Fixed INITIAL/FINAL context-path interventions; no fitting or production change."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import io
from pathlib import Path
import signal
import subprocess
import time
from gx1.audit import observe_market_state_ta_current_updates_v1 as parent
from gx1.audit.ta_current_context_intervention_v1 import ARMS, build_spec, intervene
cache,campaign,previous,REPO=parent.cache,parent.campaign,parent.previous,parent.REPO
STAGE='market_state_ta_context_drift_observation_20261113_001'
ROOT=Path('/home/andre2/GX1_RUNS')/STAGE.upper()
WORKER=REPO/'gx1/audit/observe_market_state_ta_context_drift_v1.py'
PHASES=('INITIAL','FINAL');MONTHS=('2024-10','2024-12');TF=('M5','M15','H1','H4','D1')
MODELS=dict(INITIAL='bef35dafc43f0b3450cca161049e9daacee55d92d7f1c3187bd82c9c4de972fc',FINAL='c34e754b0dba9cfe58e7b37d617b6273bdb34841921ba552d5dc1c0160cc38d7')
NATIVE_ROOT=ROOT.parent/'MARKET_STATE_TA_FULL_PREFIX_ORDER_20261113_001'
NATIVE_RESULT=dict(path=str(NATIVE_ROOT/'RESULT.json'),sha256='a3ff7bdb4341d89140621025c26192d0e6ea5ef84abbb1f8785aade8498d863d')
DRIFT_RESULT=dict(path=str(ROOT.parent/'MARKET_STATE_TA_QUERY_INPUT_DRIFT_20261113_001/RESULT.json'),sha256='2c87d24c9b366f7d4d110bd64cf9604ddf804a3c60edc70e724955a64db8cb32')
HELPERS=('gx1/audit/ta_current_context_intervention_v1.py','gx1/audit/observe_market_state_ta_current_updates_v1.py','gx1/audit/observe_market_state_ta_c171_short_heads_v1.py','gx1/audit/cache_market_state_ta_direction_v1.py','gx1/audit/market_state_retained_accuracy_v1.py')
COUNTS=dict(model_constructions=1,checkpoint_loads=2,resume_state_load_calls=2,resume_state_load_returns=2,
    archive_decodes=2,input_sidecar_archive_decodes=12,feature_batches=32,checkpoint_weight_transitions=1,
    forwards=128,native_reference_matches=32,sampled_rows=512,captured_rows=2048,phase_capture_writes=2,
    deterministic_initialization_calls=1,optimizer_constructions=0,optimizer_updates=0,backwards=0,
    autograd_grad_calls=0,checkpoint_writes=0,candidate_writes=0,new_target_gathers=0)
BOUNDS=dict(counts=COUNTS,batch_size=16,batches_per_month=8,unique_rows=256,wall_seconds=600,
    memory_bytes=12*1024**3,numerical_threads=8,max_published_json_bytes=8*1024**2,
    max_capture_bytes=4*1024**2,raw_rtol=1e-6,raw_atol=2e-6,training_enabled=False,test_data_used=False)
SCORING=dict(path='/home/andre2/GX1_RUNS/MARKET_STATE_TA_TF_ACTIVITY_SHORT_LEARNING_20261112_001/INDEPENDENT_ACTIVITY_LEARNING_REVIEW.py',sha256='939b3d3bad4ba32e47a8f24dd1e54488fcbe1e93d8fd1b7a8d39c54533fee2ba')

def need(ok,why):
    if not bool(ok):raise RuntimeError('TA_CONTEXT_DRIFT_'+why)

def select_slots(session_counts):
    """Eight intact B16 groups in each declared month; clocks/counts only."""
    offset=0;selected=[]
    for month in ('2024-10','2024-11','2024-12'):
        count=sum(session_counts['LATER/'+month]);first=(offset+15)//16;stop=(offset+count)//16
        if month in MONTHS:
            need(stop-first>=8,'MONTH_SUPPORT')
            selected.extend(dict(month=month,panel_slot=first+(stop-first-1)*i//7) for i in range(8))
        offset+=count
    need(offset==4096 and len(selected)==16 and len({x['panel_slot'] for x in selected})==16,'SLOT_POPULATION')
    return selected

def _metadata():
    from gx1.contracts import market_state_ta_full_prefix_order_v1 as scope
    native=cache._read(NATIVE_RESULT);recipe=cache._read(native['recipe']);drift=cache._read(DRIFT_RESULT)
    need(native['outcome']=='COMPLETE' and native['initial_model_state_sha256']==MODELS['INITIAL'] and native['arms']['FULL_PREFIX']['final_model_state_sha256']==MODELS['FINAL'] and native['optimizer_updates']==3698 and native['source_unchanged'] is True,'COMPLETE_NATIVE')
    need(drift['status']=='COMPLETE_QUERY_INPUT_DRIFT' and drift['counts']['feature_issue_rows']==8192 and drift['counts']['models']==0,'DRIFT_EVIDENCE')
    owners=dict(recipe['source_owners']);owners[cache.CAP_RUNNER]=cache._ref(REPO/cache.CAP_RUNNER)
    context=copy.deepcopy(recipe);context['source_owners']=owners;origin=scope.read_origin(context)
    replay=cache._read(parent.PARAMETER_RESULT);provenance=cache._read(replay['plan'])
    views={phase:cache._read(v) for phase,v in provenance['feature_views'].items()}
    need(views['FINAL']==origin['view'] and parent._leaves(views['INITIAL'],views['FINAL'])==provenance['expected_feature_view_differences'],'UNCHANGED_VIEW_PROVENANCE')
    inputs=dict(native_result=NATIVE_RESULT,native_recipe=native['recipe'],drift_result=DRIFT_RESULT,
        provenance_plan=replay['plan'],normalization=drift['normalization'],scoring=SCORING)
    normal=cache._read(inputs['normalization']);need(normal['contract_sha256']==origin['view']['prefix_normalization']['ta_base_normalization_sha256'],'ACTUAL_NORMALIZATION')
    spec=build_spec(normal);slots=select_slots(drift['session_counts']);measurements={}
    for phase,label,updates in (('INITIAL','INITIAL_LATER',0),('FINAL','FULL_PREFIX_PASS2_LATER',3698)):
        wrapper=cache._read(native['measurement_refs'][label]);m=wrapper['measurement']
        need(wrapper['evaluated_model_state_sha256']==MODELS[phase] and wrapper['optimizer_updates']==updates and wrapper['capture']==native['panel_captures'][label] and wrapper['evaluation_state_unchanged'] is True and m['rows']==4096 and m['batches']==256,'SAVED_MEASUREMENT')
        measurements[phase]=wrapper
        inputs.update({phase+'_capture':wrapper['capture'],phase+'_measurement':native['measurement_refs'][label],
            phase+'_checkpoint':provenance['states']['INITIAL'] if phase=='INITIAL' else native['arms']['FULL_PREFIX']['evidence_capsule'],phase+'_view':provenance['feature_views'][phase]})
        need(0<Path(inputs[phase+'_checkpoint']['path']).stat().st_size<128*1024**2,'STATE_SIZE')
    need(measurements['INITIAL']['panel_batch_indices']==measurements['FINAL']['panel_batch_indices'],'MATCHED_BATCHES')
    for item in slots:item['original_batch_index']=measurements['INITIAL']['panel_batch_indices'][item['panel_slot']]
    for binding in (*inputs.values(),*owners.values()):campaign.require_binding(binding,label='context observation',verify_file=True)
    return dict(origin,recipe=recipe,native=native,views=views,inputs=inputs,source_owners=owners,spec=spec,slots=slots,measurements=measurements)

def build_plan(source_base_commit):
    q=_metadata()
    return dict(schema_version='gx1_ta_context_drift_observation_plan_v1',stage=STAGE,source_repo=str(REPO),source_base_commit=source_base_commit,
        worker=cache._ref(WORKER),source_owners=q['source_owners'],helpers={name:cache._ref(REPO/name) for name in HELPERS},inputs=q['inputs'],
        model_digests=MODELS,spec=q['spec'],slots=q['slots'],arms=list(ARMS),bounds=BOUNDS,
        primary='All-row mean conditionalUP and logits of F/M15,H1,H4,D1: each intervention-minus-baseline, FINAL-minus-INITIAL effect, December and October shown separately; all arms/batches retained.',
        interpretation='Private input-path intervention using fixed TRAIN bounds. Current-context aliases move together; independently normalized MTF history stays genuine. Altered vectors need not describe a physically coherent market. No feature removal, production change, threshold fit, candidate selection, training, or economic edge claim.')

def gate_fields(ref):
    return dict(status='ADMITTED',plan=ref,max_invocations=1,remaining_invocations=1,claimed_invocations=0,completed_invocations=0,
        execution_authorized=True,relaunch_allowed=False,data_reads_authorized=True,checkpoint_reads_authorized=True,training_authorized=False,bounds=BOUNDS)

def _inspect(plan_path,plan_sha,out_dir,*,execute):
    need(Path(plan_path)==ROOT/'PLAN.json' and Path(out_dir)==ROOT,'EXACT_PATHS');binding=dict(path=str(plan_path),sha256=plan_sha);plan=cache._read(binding)
    need(not any((ROOT/name).exists() for name in ('CLAIM.json','RESULT.json','TERMINAL.json')),'ONE_USE')
    need(plan==build_plan(plan['source_base_commit']),'EXACT_PLAN');head=cache._git('rev-parse','HEAD')
    subprocess.run(['git','merge-base','--is-ancestor',plan['source_base_commit'],head],cwd=REPO,check=True)
    q=_metadata();policy_ref=cache._ref(REPO/'NEXT_RUN_POLICY.json');policy=cache._read(policy_ref)
    need(all(policy.get(k) is False for k in ('training_enabled','native_training_enabled','native_launch_authorized','full_epoch_training_allowed','full_val_allowed')),'NO_TRAINING')
    if execute:
        need(all(type(policy.get(STAGE,{}).get(k)) is type(v) and policy[STAGE][k]==v for k,v in gate_fields(binding).items()),'ADMISSION')
        need(cache._git('branch','--show-current')=='work/gx1-current','BRANCH');campaign.require_clean_source(dict(source_repo=str(REPO),source_commit=head))
    refs=[binding,policy_ref,plan['worker'],*plan['helpers'].values(),*plan['inputs'].values(),*plan['source_owners'].values()]
    return dict(q,plan=plan,plan_ref=binding,head=head,bound_refs=refs)

def preflight(plan_path,plan_sha,out_dir):
    q=_inspect(plan_path,plan_sha,out_dir,execute=False)
    return dict(status='PASS_METADATA_ONLY_CONTEXT_DRIFT',plan=q['plan_ref'],model_constructions=0,checkpoint_loads=0,array_decodes=0,claims=0)

def _raw_digest(raw):
    return {k:(str(v.dtype),v.shape,hashlib.sha256(v.tobytes()).hexdigest()) for k,v in raw.items()}

def _mean_prob(z,np):return np.exp(-np.logaddexp(0.,-z.astype(np.float64))).mean(axis=0).tolist()

def run(plan_path,plan_sha,out_dir,device='cuda'):
    started=time.monotonic();need(device=='cuda','CUDA_ONLY');q=_inspect(plan_path,plan_sha,out_dir,execute=True);cap=previous._runtime_cap()
    import numpy as np,torch
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts import entry_native_direction_research_state_v1 as old
    from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
    from gx1.contracts import market_state_ta_full_prefix_order_v1 as scope
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as model_sha
    from gx1.contracts.market_state_time_query_native_F_objective_v1 import require_m5_equal_observation_axes
    from gx1.audit.run_market_state_ta_m5_objective_benchmark_v1 import _prepare_later
    from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import _BATCH_EQUAL_LENGTH_MTF_EVAL
    need(not torch.cuda.is_initialized() and signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'INIT_ORDER')
    counts={k:0 for k in COUNTS};claim=cache._publish('CLAIM.json',dict(plan=q['plan_ref'],source_commit=q['head'],cap=cap,no_relaunch_under_this_plan=True,test_data_used=False),root=ROOT)
    provider=prepared=model=None;phase=None;records=[];artifacts={};passivity={};pieces={p:[] for p in PHASES};before={}
    prior_alarm=signal.getsignal(signal.SIGALRM)
    def timeout(*unused):raise TimeoutError('Context observation600second bound')
    def bookend():
        campaign.require_clean_source(dict(source_repo=str(REPO),source_commit=q['head']))
        for binding in q['bound_refs']:campaign.require_binding(binding,label='context bookend',verify_file=True)
    def signature():
        return dict(model=model_sha(model.state_dict()),buffers=old._tree_sha(dict(model.named_buffers())),modes={n:m.training for n,m in model.named_modules()},
            flags={n:p.requires_grad for n,p in model.named_parameters()},parameters={n:id(p) for n,p in model.named_parameters()},
            modules={n:id(m) for n,m in model.named_modules()},view=copy.deepcopy(model.state_model.feature_view_identity),
            hooks={n:{k:tuple((i,id(h)) for i,h in getattr(m,k).items()) for k in ('_forward_hooks','_forward_pre_hooks','_backward_hooks','_backward_pre_hooks')} for n,m in model.named_modules()},
            rng=old._tree_sha(trainer._attended_session_rng_state(device=cuda)),runtime=previous._runtime_profile(torch),mha_fastpath=torch.backends.mha.get_fastpath_enabled(),mtf_batch_equal=_BATCH_EQUAL_LENGTH_MTF_EVAL.get())
    def persist(which):
        if which in PHASES and pieces[which] and which not in artifacts:
            joined={k:np.concatenate([piece[k] for piece in pieces[which]]) for k in pieces[which][0]};stream=io.BytesIO();np.savez(stream,**joined);blob=stream.getvalue();need(len(blob)<=BOUNDS['max_capture_bytes'],'CAPTURE_SIZE')
            from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
            path=ROOT/(which+'_CAPTURE.npz');_publish_bytes(path,blob);artifacts[which]=cache._ref(path);counts['phase_capture_writes']+=1
    def publish(complete,**extra):
        return cache._publish_fusion_outcome(q['plan_ref'],dict(claim=claim,source_commit=q['head'],counts=counts,artifacts=artifacts,batch_records=records,passivity=passivity,model_digests=MODELS,interpretation=q['plan']['interpretation'],**extra),complete=complete,root=ROOT,json_bytes_bound=BOUNDS['max_published_json_bytes'])
    try:
        signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,BOUNDS['wall_seconds'])
        recipe,view,receipts=q['recipe'],q['view'],q['receipts'];cuda=torch.device('cuda');numerical=recipe['numerical_recipe']
        trainer._set_deterministic(numerical['seed'],cuda,numerical['precision_policy']);counts['deterministic_initialization_calls']+=1
        previous._require_runtime_profile(previous._runtime_profile(torch));need(_BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False and torch.backends.mha.get_fastpath_enabled() is True,'RUNTIME')
        captures={};states={}
        for name in PHASES:
            with np.load(q['inputs'][name+'_capture']['path'],allow_pickle=False) as z:captures[name]={k:z[k].copy() for k in z.files if not k.endswith('_representation')}
            counts['archive_decodes']+=1;need(all(len(a)==4096 for a in captures[name].values()),'CAPTURE_ROWS')
            counts['checkpoint_loads']+=1;states[name]=common._load_state(q['inputs'][name+'_checkpoint'],torch=torch,maximum=128*1024**2,counters=counts)
        need(all(np.array_equal(v,captures['FINAL'][k]) for k,v in captures['INITIAL'].items() if not k.endswith('_logits')),'PAIRED_TARGET_AXES')
        need(np.all(np.diff(captures['INITIAL']['issues'])>0),'CHRONOLOGICAL_PANEL')
        provider,provider_plan,prepared,rows,issues=scope.full.anchor.prepare_fold1(receipts,view,np=np,counters=counts)
        require_m5_equal_observation_axes(rows,issues);order=torch.from_numpy(rows.copy());parent._require_parent_order(order,states,torch=torch)
        later=_prepare_later(q,prepared,receipts,view,counters=counts,np=np)
        counts['model_constructions']+=1;model=NativeFutureDualModel(views.build_native_direction_research_model(view,prefix_normalization=receipts['fold1']['normal'],constructor_metadata=receipts['constructor'],device=torch.device('cpu'),unobserved_units_profile=receipts['fold1']['profile']))
        previous._short_parameter_partition(model,state)
        need(all(states['INITIAL']['binding'][k]==states['FINAL']['parameter_binding'][k] for k in ('parameter_inventory','active_parameter_names','inactive_parameter_names','optimizer_group_parameter_names','research_objective')),'PAIRED_PARAMETER_BINDINGS')
        for phase in PHASES:
            parent._load_endpoint(model,states[phase],phase,q,order,torch=torch,old=old,state=state,model_sha=model_sha,model_digests=MODELS)
            model.to(cuda);model.eval();counts['checkpoint_weight_transitions']+=int(phase=='FINAL');before[phase]=signature();capture=captures[phase]
            for selected in q['slots']:
                slot=selected['panel_slot'];left=16*slot;right=left+16
                need(np.all(capture['issues'][left:right].astype('datetime64[ns]').astype('datetime64[M]').astype(str)==selected['month']),'EXACT_CLOCK_SELECTION')
                raw,parents,clocks=runner._feature_batch(later,view,receipts,capture['parents'],capture['issues'],left,right,np=np);counts['feature_batches']+=1;counts['sampled_rows']+=16;original=_raw_digest(raw)
                baseline=None
                for arm_index,arm in enumerate(ARMS):
                    changed,change_counts=intervene(raw,q['spec'],arm);need(_raw_digest(raw)==original,'RAW_MUTATION')
                    features={k:torch.from_numpy(v).to(cuda) for k,v in changed.items()};query=torch.tensor([1,3,12,48,288],dtype=torch.int64,device=cuda)[None,:].expand(16,-1)
                    with torch.inference_mode():
                        need(not any(m.training for m in model.modules()) and not torch.is_grad_enabled(),'PASSIVE_FORWARD')
                        output=model(**features,query_k=query);counts['forwards']+=1
                        piece={task+'_raw':output[task+'_activity_side_logits'].cpu().numpy().copy() for task in ('state','future')}
                    need(all(v.shape==(16,5,2) and v.dtype==np.float32 and np.isfinite(v).all() for v in piece.values()),'FINITE_OUTPUT')
                    parity={}
                    if arm=='BASELINE':
                        for task in ('state','future'):
                            actual=piece[task+'_raw'];expected=capture[task+'_activity_side_logits'][left:right];parity[task]=dict(bitexact=np.array_equal(actual,expected),max_abs=float(np.max(np.abs(actual-expected))),within_tolerance=bool(np.allclose(actual,expected,rtol=BOUNDS['raw_rtol'],atol=BOUNDS['raw_atol'])))
                        baseline={k:v.copy() for k,v in piece.items()}
                    elif change_counts['total_changed_cells']==0:need(all(np.array_equal(v,baseline[k]) for k,v in piece.items()),'NOOP_OUTPUT')
                    for task in ('state','future'):
                        for key in ('allowed','eligible','mature'):piece[task+'_'+key]=capture[task+'_'+key][left:right].copy()
                    piece.update(parents=parents.copy(),issues=clocks.copy(),query_native=capture['query_native'][left:right].copy(),arm_index=np.full(16,arm_index,np.int8),panel_slot=np.full(16,slot,np.int16))
                    pieces[phase].append(piece);counts['captured_rows']+=16
                    record=dict(phase=phase,month=selected['month'],panel_slot=slot,original_batch_index=selected['original_batch_index'],arm=arm,changes=change_counts,baseline_reference=parity,
                        probabilities={task:dict(mean_p_up=_mean_prob(piece[task+'_raw'][:,:,1],np),mean_p_active=_mean_prob(piece[task+'_raw'][:,:,0],np),mean_side_logit=piece[task+'_raw'][:,:,1].astype(np.float64).mean(0).tolist()) for task in ('state','future')})
                    records.append(record)
                    if arm=='BASELINE':
                        need(all(v['within_tolerance'] for v in parity.values()),'RETAINED_REFERENCE:'+phase+':slot'+str(slot))
                        counts['native_reference_matches']+=1
                    need(_raw_digest(raw)==original and all(p.grad is None for p in model.parameters()),'NO_INPUT_OR_GRAD_MUTATION');del features,changed,output
            passivity[phase]=dict(signature_unchanged=signature()==before[phase],no_parameter_gradients=all(p.grad is None for p in model.parameters()))
            need(all(passivity[phase].values()),'PHASE_PASSIVITY');persist(phase);print(phase+'16 matchedB16groups/4arms complete',flush=True)
        need(counts==COUNTS,'EXACT_COUNTS');scope.full.anchor.release_inputs(provider,prepared);prepared=None;provider.validate_metadata(provider_plan);bookend()
        return publish(True,source_unchanged=True,elapsed_seconds=time.monotonic()-started,raw_input_mutations=0,production_input_changes=0,selection_authorized=False)
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL,0);bookend_error=passivity_error=None
        try:
            if phase in before:passivity[phase]=dict(signature_unchanged=signature()==before[phase],no_parameter_gradients=all(p.grad is None for p in model.parameters()))
        except Exception as other:passivity_error=repr(other)
        try:persist(phase);bookend()
        except Exception as other:bookend_error=repr(other)
        publish(False,source_unchanged=bookend_error is None,active_phase=phase,error=type(exc).__name__+':'+str(exc),bookend_error=bookend_error,passivity_error=passivity_error,interpretation_authorized=False,elapsed_seconds=time.monotonic()-started)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,prior_alarm)
        if prepared is not None:scope.full.anchor.release_inputs(provider,prepared)

def main(argv=None):
    import argparse,json,sys
    supplied=list(sys.argv[1:] if argv is None else argv);flags=('--market-state-ta-context-drift-plan','--market-state-ta-context-drift-plan-sha256')
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False,add_help=False)
    parser.add_argument(flags[0],type=Path,required=True);parser.add_argument(flags[1],required=True)
    parser.add_argument('--device',choices=('cuda',),required=True);parser.add_argument('--out-dir',type=Path,required=True);parser.add_argument('--preflight',action='store_true')
    for flag in (*flags,'--device','--out-dir','--preflight'):
        if sum(v.partition('=')[0]==flag for v in supplied)>1:parser.error('duplicate flag:'+flag)
    if '--preflight' in supplied and supplied[-1]!='--preflight':parser.error('--preflight must be last')
    args=parser.parse_args(supplied);kwargs=dict(plan_path=args.market_state_ta_context_drift_plan,plan_sha=args.market_state_ta_context_drift_plan_sha256,out_dir=args.out_dir)
    print(json.dumps(preflight(**kwargs) if args.preflight else run(**kwargs,device=args.device),sort_keys=True,allow_nan=False));return 0
if __name__=='__main__':raise SystemExit(main())
