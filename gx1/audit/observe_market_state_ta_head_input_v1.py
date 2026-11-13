"""Observe native TF head inputs at fixed endpoints; no training or model edits."""
from __future__ import annotations
import copy
import io
from pathlib import Path
import signal
import subprocess
import time
from gx1.audit import observe_market_state_ta_context_drift_v1 as base
from gx1.audit.ta_head_representation_accounting_v1 import account
cache,campaign,previous,REPO=base.cache,base.campaign,base.previous,base.REPO
STAGE='market_state_ta_head_input_observation_20261113_001'
ROOT=Path('/home/andre2/GX1_RUNS')/STAGE.upper()
WORKER=REPO/'gx1/audit/observe_market_state_ta_head_input_v1.py'
PHASES,MODELS,TF=base.PHASES,base.MODELS,base.TF
HELPERS=(*base.HELPERS,'gx1/audit/observe_market_state_ta_context_drift_v1.py','gx1/audit/ta_head_representation_accounting_v1.py')
COUNTS=dict(model_constructions=1,checkpoint_loads=2,resume_state_load_calls=2,resume_state_load_returns=2,
    archive_decodes=2,input_sidecar_archive_decodes=12,feature_batches=32,checkpoint_weight_transitions=1,
    forwards=32,native_reference_matches=32,sampled_rows=512,captured_rows=512,phase_capture_writes=2,
    projection_input_hook_calls=64,representation_hook_calls=64,head_parameter_capture_sets=2,affine_accounting_cells=30,
    deterministic_initialization_calls=1,optimizer_constructions=0,optimizer_updates=0,backwards=0,
    autograd_grad_calls=0,checkpoint_writes=0,candidate_writes=0,new_target_gathers=0)
BOUNDS=dict(counts=COUNTS,batch_size=16,batches_per_month=8,unique_rows=256,wall_seconds=600,
    memory_bytes=12*1024**3,numerical_threads=8,max_published_json_bytes=8*1024**2,
    max_capture_bytes=4*1024**2,raw_rtol=1e-6,raw_atol=2e-6,training_enabled=False,test_data_used=False)

def need(ok,why):
    if not bool(ok):raise RuntimeError('TA_HEAD_INPUT_'+why)

def build_plan(source_base_commit):
    q=base._metadata()
    need(q['view']['dedicated_m5_side'] is True and q['view']['dedicated_tf_activity'] is True and q['view'].get('dedicated_non_m5_side',False) is False,'CURRENT_HEAD_DESIGN')
    return dict(schema_version='gx1_ta_head_input_observation_plan_v1',stage=STAGE,source_repo=str(REPO),source_base_commit=source_base_commit,
        worker=cache._ref(WORKER),source_owners=q['source_owners'],helpers={name:cache._ref(REPO/name) for name in HELPERS},inputs=q['inputs'],
        model_digests=MODELS,slots=q['slots'],bounds=BOUNDS,
        primary='All-row S/F side-logit change INITIAL to current order FINAL, separately October/December and combined: both ordered and symmetric head-weight/representation/bias accounting for all five TF. Primary question is non-M5 common UP drift; no cell or mixed endpoint is selected.',
        interpretation='Algebraic endpoint accounting using actual native Linear inputs. Head-input change includes backbone and shared query projection. Crossed affine calculations are diagnostic, not fitted candidates or historical causal shares. Both input and post-projection vectors are retained privately from the same native forwards. No feature intervention, production change, candidate selection, training, or edge claim.')

def gate_fields(ref):
    return dict(status='ADMITTED',plan=ref,max_invocations=1,remaining_invocations=1,claimed_invocations=0,completed_invocations=0,
        execution_authorized=True,relaunch_allowed=False,data_reads_authorized=True,checkpoint_reads_authorized=True,training_authorized=False,bounds=BOUNDS)

def _inspect(plan_path,plan_sha,out_dir,*,execute):
    need(Path(plan_path)==ROOT/'PLAN.json' and Path(out_dir)==ROOT,'EXACT_PATHS');binding=dict(path=str(plan_path),sha256=plan_sha);plan=cache._read(binding)
    need(not any((ROOT/name).exists() for name in ('CLAIM.json','RESULT.json','TERMINAL.json')),'ONE_USE')
    need(plan==build_plan(plan['source_base_commit']),'EXACT_PLAN');head=cache._git('rev-parse','HEAD')
    subprocess.run(['git','merge-base','--is-ancestor',plan['source_base_commit'],head],cwd=REPO,check=True)
    q=base._metadata();policy_ref=cache._ref(REPO/'NEXT_RUN_POLICY.json');policy=cache._read(policy_ref)
    need(all(policy.get(k) is False for k in ('training_enabled','native_training_enabled','native_launch_authorized','full_epoch_training_allowed','full_val_allowed')),'NO_TRAINING')
    if execute:
        need(all(type(policy.get(STAGE,{}).get(k)) is type(v) and policy[STAGE][k]==v for k,v in gate_fields(binding).items()),'ADMISSION')
        need(cache._git('branch','--show-current')=='work/gx1-current','BRANCH');campaign.require_clean_source(dict(source_repo=str(REPO),source_commit=head))
    refs=[binding,policy_ref,plan['worker'],*plan['helpers'].values(),*plan['inputs'].values(),*plan['source_owners'].values()]
    return dict(q,plan=plan,plan_ref=binding,head=head,bound_refs=refs)

def preflight(plan_path,plan_sha,out_dir):
    q=_inspect(plan_path,plan_sha,out_dir,execute=False)
    return dict(status='PASS_METADATA_ONLY_HEAD_INPUT',plan=q['plan_ref'],model_constructions=0,checkpoint_loads=0,array_decodes=0,claims=0)

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
    provider=prepared=model=None;phase=None;records=[];artifacts={};passivity={};pieces={p:[] for p in PHASES};before={};heads={};joined={};handles=[];accounting=[]
    prior_alarm=signal.getsignal(signal.SIGALRM)
    def timeout(*unused):raise TimeoutError('Head input observation600second bound')
    def bookend():
        campaign.require_clean_source(dict(source_repo=str(REPO),source_commit=q['head']))
        for binding in q['bound_refs']:campaign.require_binding(binding,label='head input bookend',verify_file=True)
    def signature():
        return dict(model=model_sha(model.state_dict()),buffers=old._tree_sha(dict(model.named_buffers())),modes={n:m.training for n,m in model.named_modules()},
            flags={n:p.requires_grad for n,p in model.named_parameters()},parameters={n:id(p) for n,p in model.named_parameters()},
            modules={n:id(m) for n,m in model.named_modules()},view=copy.deepcopy(model.state_model.feature_view_identity),
            hooks={n:{k:tuple((i,id(h)) for i,h in getattr(m,k).items()) for k in ('_forward_hooks','_forward_pre_hooks','_backward_hooks','_backward_pre_hooks')} for n,m in model.named_modules()},
            rng=old._tree_sha(trainer._attended_session_rng_state(device=cuda)),runtime=previous._runtime_profile(torch),mha_fastpath=torch.backends.mha.get_fastpath_enabled(),mtf_batch_equal=_BATCH_EQUAL_LENGTH_MTF_EVAL.get())
    def persist(which):
        if which in PHASES and pieces[which] and which not in artifacts:
            joined[which]={k:np.concatenate([piece[k] for piece in pieces[which]]) for k in pieces[which][0]}
            stream=io.BytesIO();np.savez(stream,**joined[which],**heads[which]);blob=stream.getvalue();need(len(blob)<=BOUNDS['max_capture_bytes'],'CAPTURE_SIZE')
            from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
            path=ROOT/(which+'_CAPTURE.npz');_publish_bytes(path,blob);artifacts[which]=cache._ref(path);counts['phase_capture_writes']+=1
    def publish(complete,**extra):
        return cache._publish_fusion_outcome(q['plan_ref'],dict(claim=claim,source_commit=q['head'],counts=counts,artifacts=artifacts,batch_records=records,passivity=passivity,model_digests=MODELS,accounting=accounting,interpretation=q['plan']['interpretation'],**extra),complete=complete,root=ROOT,json_bytes_bound=BOUNDS['max_published_json_bytes'])
    try:
        signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,BOUNDS['wall_seconds'])
        recipe,view,receipts=q['recipe'],q['view'],q['receipts'];cuda=torch.device('cuda');numerical=recipe['numerical_recipe']
        trainer._set_deterministic(numerical['seed'],cuda,numerical['precision_policy']);counts['deterministic_initialization_calls']+=1
        previous._require_runtime_profile(previous._runtime_profile(torch));need(_BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False and torch.backends.mha.get_fastpath_enabled() is True,'RUNTIME')
        captures={};states={}
        for name in PHASES:
            with np.load(q['inputs'][name+'_capture']['path'],allow_pickle=False) as z:captures[name]={k:z[k].copy() for k in z.files}
            counts['archive_decodes']+=1;need(all(len(a)==4096 for a in captures[name].values()),'CAPTURE_ROWS')
            counts['checkpoint_loads']+=1;states[name]=common._load_state(q['inputs'][name+'_checkpoint'],torch=torch,maximum=128*1024**2,counters=counts)
        need(all(np.array_equal(v,captures['FINAL'][k]) for k,v in captures['INITIAL'].items() if not (k.endswith('_logits') or k.endswith('_representation'))),'PAIRED_TARGET_AXES')
        provider,provider_plan,prepared,rows,issues=scope.full.anchor.prepare_fold1(receipts,view,np=np,counters=counts)
        require_m5_equal_observation_axes(rows,issues);order=torch.from_numpy(rows.copy());base.parent._require_parent_order(order,states,torch=torch)
        later=_prepare_later(q,prepared,receipts,view,counters=counts,np=np)
        counts['model_constructions']+=1;model=NativeFutureDualModel(views.build_native_direction_research_model(view,prefix_normalization=receipts['fold1']['normal'],constructor_metadata=receipts['constructor'],device=torch.device('cpu'),unobserved_units_profile=receipts['fold1']['profile']))
        previous._short_parameter_partition(model,state)
        need(all(states['INITIAL']['binding'][k]==states['FINAL']['parameter_binding'][k] for k in ('parameter_inventory','active_parameter_names','inactive_parameter_names','optimizer_group_parameter_names','research_objective')),'PAIRED_PARAMETER_BINDINGS')
        need(model.state_model._hidden_width==128 and model.state_model._dedicated_non_m5_side is False,'ACTUAL_SHARED_NON_M5_HEAD')
        for phase in PHASES:
            base.parent._load_endpoint(model,states[phase],phase,q,order,torch=torch,old=old,state=state,model_sha=model_sha,model_digests=MODELS)
            model.to(cuda);model.eval();counts['checkpoint_weight_transitions']+=int(phase=='FINAL');before[phase]=signature();capture=captures[phase]
            heads[phase]={}
            for task,shared,m5 in (('state',model.state_model.query_head,model.state_model.m5_side_head),('future',model.future_head,model.future_m5_side_head)):
                heads[phase][task+'_weights']=torch.cat((m5.weight,shared.weight[1:2].expand(4,-1))).detach().cpu().numpy().copy()
                heads[phase][task+'_biases']=torch.cat((m5.bias,shared.bias[1:2].expand(4))).detach().cpu().numpy().copy()
            # These small fixed decoder tensors allow later algebra without decoding checkpoints again.
            projection=model.state_model.query_projection[0]
            heads[phase]['projection_weight']=projection.weight.detach().cpu().numpy().copy();heads[phase]['projection_bias']=projection.bias.detach().cpu().numpy().copy()
            counts['head_parameter_capture_sets']+=1;observed={}
            def capture_input(module,args):
                need(len(args)==1,'PROJECTION_ARGUMENT');x=args[0];task='state' if x.ndim==4 else 'future'
                need(x.shape==((16,1,5,134) if task=='state' else (16,5,134)) and task+'_hidden' not in observed,'NATIVE_PROJECTION_INPUT')
                a=x.detach().cpu().numpy().copy();a=a[:,0] if task=='state' else a
                need(np.array_equal(a[:,:,:128],np.broadcast_to(a[:,0:1,:128],(16,5,128))) and np.array_equal(a[:,:,128:133],np.broadcast_to(np.eye(5,dtype=np.float32),(16,5,5))) and np.array_equal(a[:,:,133],np.ones((16,5),np.float32)),'NATIVE_QUERY_COORDINATES')
                observed[task+'_hidden']=a[:,0,:128].copy();counts['projection_input_hook_calls']+=1
            def capture_representation(module,args,output):
                task='state' if output.ndim==4 else 'future'
                need(output.shape==((16,1,5,128) if task=='state' else (16,5,128)) and task+'_representation' not in observed,'NATIVE_PROJECTION_OUTPUT')
                a=output.detach().cpu().numpy().copy();observed[task+'_representation']=a[:,0] if task=='state' else a;counts['representation_hook_calls']+=1
            handles=[model.state_model.query_projection.register_forward_pre_hook(capture_input),model.state_model.query_projection.register_forward_hook(capture_representation)]
            for selected in q['slots']:
                slot=selected['panel_slot'];left=16*slot;right=left+16
                need(np.all(capture['issues'][left:right].astype('datetime64[ns]').astype('datetime64[M]').astype(str)==selected['month']),'EXACT_CLOCK_SELECTION')
                raw,parents,clocks=runner._feature_batch(later,view,receipts,capture['parents'],capture['issues'],left,right,np=np);counts['feature_batches']+=1;counts['sampled_rows']+=16;original=base._raw_digest(raw);observed.clear()
                features={k:torch.from_numpy(v).to(cuda) for k,v in raw.items()};query=torch.tensor([1,3,12,48,288],dtype=torch.int64,device=cuda)[None,:].expand(16,-1)
                with torch.inference_mode():
                    need(not any(m.training for m in model.modules()) and not torch.is_grad_enabled(),'PASSIVE_FORWARD')
                    output=model(**features,query_k=query);counts['forwards']+=1
                    piece={task+'_raw':output[task+'_activity_side_logits'].cpu().numpy().copy() for task in ('state','future')}
                need(set(observed)=={'state_hidden','future_hidden','state_representation','future_representation'} and np.array_equal(observed['state_hidden'],observed['future_hidden']),'ONE_SHARED_HIDDEN')
                piece.update(hidden=observed['state_hidden'],state_representation=observed['state_representation'],future_representation=observed['future_representation'])
                parity={}
                for task in ('state','future'):
                    actual=piece[task+'_raw'];expected=capture[task+'_activity_side_logits'][left:right]
                    need(actual.shape==(16,5,2) and actual.dtype==np.float32 and np.isfinite(actual).all(),'FINITE_OUTPUT')
                    parity[task]=dict(bitexact=np.array_equal(actual,expected),max_abs=float(np.max(np.abs(actual-expected))),within_tolerance=bool(np.allclose(actual,expected,rtol=BOUNDS['raw_rtol'],atol=BOUNDS['raw_atol'])),m5_representation_bitexact=np.array_equal(piece[task+'_representation'][:,0],capture[task+'_representation'][left:right]))
                    for key in ('allowed','eligible','mature'):piece[task+'_'+key]=capture[task+'_'+key][left:right].copy()
                piece.update(parents=parents.copy(),issues=clocks.copy(),query_native=capture['query_native'][left:right].copy(),panel_slot=np.full(16,slot,np.int16))
                pieces[phase].append(piece);counts['captured_rows']+=16
                records.append(dict(phase=phase,month=selected['month'],panel_slot=slot,baseline_reference=parity,mean_p_up={task:base._mean_prob(piece[task+'_raw'][:,:,1],np) for task in ('state','future')}))
                need(all(v['within_tolerance'] and v['m5_representation_bitexact'] for v in parity.values()),'RETAINED_REFERENCE:'+phase+':slot'+str(slot));counts['native_reference_matches']+=1
                need(base._raw_digest(raw)==original and all(p.grad is None for p in model.parameters()),'NO_INPUT_OR_GRAD_MUTATION');del features,output
            for handle in handles:handle.remove()
            handles=[];passivity[phase]=dict(signature_unchanged=signature()==before[phase],no_parameter_gradients=all(p.grad is None for p in model.parameters()))
            need(all(passivity[phase].values()),'PHASE_PASSIVITY');persist(phase);print(phase+'16 matchedB16groups/allTFhead inputs complete',flush=True)
        a,b=joined['INITIAL'],joined['FINAL'];need(all(np.array_equal(a[k],b[k]) for k in ('parents','issues','query_native','panel_slot')),'MATCHED_CAPTURE_AXES')
        months=a['issues'].astype('datetime64[ns]').astype('datetime64[M]').astype(str)
        for subset,mask in [('ALL',np.ones(256,bool)),('2024-10',months=='2024-10'),('2024-12',months=='2024-12')]:
            for task in ('state','future'):
                for i,timeframe in enumerate(TF):
                    value=account(a[task+'_representation'][mask,i],b[task+'_representation'][mask,i],heads['INITIAL'][task+'_weights'][i],heads['FINAL'][task+'_weights'][i],heads['INITIAL'][task+'_biases'][i:i+1],heads['FINAL'][task+'_biases'][i:i+1],a[task+'_raw'][mask,i,1],b[task+'_raw'][mask,i,1])
                    accounting.append(dict(subset=subset,rows=int(mask.sum()),task=task,timeframe=timeframe,result=value));counts['affine_accounting_cells']+=1
        need(counts==COUNTS,'EXACT_COUNTS');scope.full.anchor.release_inputs(provider,prepared);prepared=None;provider.validate_metadata(provider_plan);bookend()
        return publish(True,source_unchanged=True,elapsed_seconds=time.monotonic()-started,raw_input_mutations=0,production_input_changes=0,selection_authorized=False)
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL,0)
        for handle in handles:handle.remove()
        bookend_error=passivity_error=None
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
    supplied=list(sys.argv[1:] if argv is None else argv);flags=('--market-state-ta-head-input-plan','--market-state-ta-head-input-plan-sha256')
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False,add_help=False)
    parser.add_argument(flags[0],type=Path,required=True);parser.add_argument(flags[1],required=True)
    parser.add_argument('--device',choices=('cuda',),required=True);parser.add_argument('--out-dir',type=Path,required=True);parser.add_argument('--preflight',action='store_true')
    for flag in (*flags,'--device','--out-dir','--preflight'):
        if sum(v.partition('=')[0]==flag for v in supplied)>1:parser.error('duplicate flag:'+flag)
    if '--preflight' in supplied and supplied[-1]!='--preflight':parser.error('--preflight must be last')
    args=parser.parse_args(supplied);kwargs=dict(plan_path=args.market_state_ta_head_input_plan,plan_sha=args.market_state_ta_head_input_plan_sha256,out_dir=args.out_dir)
    print(json.dumps(preflight(**kwargs) if args.preflight else run(**kwargs,device=args.device),sort_keys=True,allow_nan=False));return 0
if __name__=='__main__':raise SystemExit(main())
