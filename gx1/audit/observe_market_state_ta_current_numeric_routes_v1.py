"""Three passive first-B16 forwards to isolate current evaluation-route mismatch."""
from __future__ import annotations

import contextlib
import copy
import io
import math
from pathlib import Path
import signal
import subprocess
import time

from gx1.audit import observe_market_state_ta_current_updates_v1 as parent

cache, campaign, full, previous, REPO = parent.cache, parent.campaign, parent.full, parent.previous, parent.REPO
STAGE = 'market_state_ta_current_numeric_routes_20261113_001'
ROOT = Path('/home/andre2/GX1_RUNS') / STAGE.upper()
WORKER = REPO/'gx1/audit/observe_market_state_ta_current_numeric_routes_v1.py'
FAILED_ROOT = ROOT.parent/'MARKET_STATE_TA_CURRENT_UPDATE_OBSERVATION_20261113_002'
FAILED_TERMINAL = dict(path=str(FAILED_ROOT/'TERMINAL.json'),sha256='05e86ad22bad5645aff0b831b16abe7f59ea22b52c02f4c0f7bc8f03d027d2bc')
ROUTES = ('NATIVE_NO_GRAD','DYNAMIC_NO_GRAD','DYNAMIC_GRAD')
HELPERS = ('gx1/audit/observe_market_state_ta_current_updates_v1.py',*parent.HELPERS,
    '.venv/lib/python3.10/site-packages/torch/backends/mha/__init__.py',
    '.venv/lib/python3.10/site-packages/torch/nn/modules/transformer.py')
BOUNDS = dict(batch_size=16,original_batch_index=0,panel_slot=0,unique_rows=16,
    model_constructions=1,checkpoint_loads=1,resume_state_load_calls=1,resume_state_load_returns=1,
    archive_decodes=1,input_sidecar_archive_decodes=6,forwards=3,grad_forwards=1,
    route_capture_writes=3,autograd_grad_calls=0,optimizer_constructions=0,optimizer_updates=0,
    backwards=0,rng_restores=0,checkpoint_writes=0,candidate_writes=0,
    wall_seconds=300,memory_bytes=12*1024**3,max_capture_bytes=1024**2,max_published_json_bytes=2*1024**2,
    numerical_threads=8,raw_rtol=parent.BOUNDS['raw_rtol'],raw_atol=parent.BOUNDS['raw_atol'],
    loss_atol=parent.BOUNDS['loss_atol'],training_enabled=False,test_data_used=False)


def _need(ok,reason):
    if not bool(ok): raise RuntimeError('TA_CURRENT_NUMERIC_ROUTES_'+reason)


def compare_pieces(actual,reference,loss,reference_loss,*,np):
    """Preserve measured deviations; an over-tolerance comparison is not a PASS."""
    _need(set(actual)==set(reference)==set(previous.BASE_KEYS),'KEYS')
    for name,value in actual.items():
        expected=reference[name]
        _need(isinstance(value,np.ndarray) and isinstance(expected,np.ndarray)
            and value.shape==expected.shape and value.dtype==expected.dtype,'LAYOUT:'+name)
        if name.endswith('_raw'):
            _need(value.ndim==3 and value.shape[1:]==(5,2) and value.dtype==np.float32
                and bool(np.isfinite(value).all()) and bool(np.isfinite(expected).all()),'RAW_DOMAIN:'+name)
        else: _need(value.tobytes()==expected.tobytes(),'AXES_TARGETS:'+name)
    raw={}
    for name in ('state_raw','future_raw'):
        a,b=actual[name].astype(np.float64),reference[name].astype(np.float64)
        delta=a-b;absolute=np.abs(delta);scale=BOUNDS['raw_atol']+BOUNDS['raw_rtol']*np.abs(b)
        pa,pb=np.exp(-np.logaddexp(0.,-a)),np.exp(-np.logaddexp(0.,-b))
        raw[name]=dict(max_abs=float(absolute.max()),max_scaled=float((absolute/scale).max()),
            bitexact=actual[name].tobytes()==reference[name].tobytes(),
            max_probability_abs=float(np.abs(pa-pb).max()),mean_abs=float(absolute.mean()),
            max_abs_by_tf_channel=absolute.max(axis=0).tolist(),mean_logit_delta_by_tf_channel=delta.mean(axis=0).tolist(),
            max_probability_abs_by_tf_channel=np.abs(pa-pb).max(axis=0).tolist())
    error=abs(float(loss)-float(reference_loss));_need(math.isfinite(error),'LOSS_FINITE')
    return dict(within_existing_tolerance=all(v['max_scaled']<=1. for v in raw.values()) and error<=BOUNDS['loss_atol'],
        axes_targets_bitexact=True,loss_max_abs=error,raw=raw,
        tolerances={k:BOUNDS[k] for k in ('raw_rtol','raw_atol','loss_atol')})


def _origin():
    q=parent._metadata();terminal=cache._read(FAILED_TERMINAL);failed=cache._read(terminal['result'])
    _need(terminal['status']==failed['status']=='FAILED_CONSUMED' and terminal['exit_code']==1
        and failed['error']=='RuntimeError:TA_CURRENT_UPDATE_OBSERVATION_RAW_PARITY:state_raw'
        and failed['counts']['forwards']==1 and failed['counts']['grad_forwards']==failed['counts']['autograd_grad_calls']==0
        and failed['source_unchanged'] is True,'PREVIOUS_PARITY_STOP')
    return dict(q,failed_terminal=FAILED_TERMINAL,failed_result=terminal['result'])


def build_plan(source_base_commit):
    q=_origin()
    return dict(schema_version='gx1_ta_current_numeric_routes_plan_v1',stage=STAGE,source_repo=str(REPO),
        source_base_commit=source_base_commit,worker=cache._ref(WORKER),source_owners=q['source_owners'],
        helpers={name:cache._ref(REPO/name) for name in HELPERS},inputs=q['inputs'],
        previous_failed_terminal=q['failed_terminal'],previous_failed_result=q['failed_result'],
        model_state_sha256=parent.MODELS['INITIAL'],feature_view_sha256=q['view']['contract_sha256'],
        numerical_recipe_sha256=q['recipe']['numerical_recipe']['contract_sha256'],routes=list(ROUTES),bounds=BOUNDS,
        comparison_scope='Same actual INITIAL weights/features/targets/queries on first fixed B16 TRAIN batch. Native retained control first; explicit dynamic no_grad and grad-enabled counterpart measured separately.',
        interpretation='Completed comparison does not certify cross-route equivalence; unchanged tolerances are reported for every pair. No autograd calls, training, trajectory or edge inference.')


def gate_fields(ref):
    return dict(status='ADMITTED',plan=ref,max_invocations=1,remaining_invocations=1,claimed_invocations=0,
        completed_invocations=0,relaunch_allowed=False,execution_authorized=True,training_authorized=False,bounds=BOUNDS)


def _inspect(plan_path,plan_sha,out_dir,*,execute):
    _need(Path(plan_path)==ROOT/'PLAN.json' and Path(out_dir)==ROOT,'EXACT_PATHS')
    ref=dict(path=str(plan_path),sha256=plan_sha);plan=cache._read(ref)
    _need(not any((ROOT/name).exists() for name in ('CLAIM.json','RESULT.json','TERMINAL.json')),'CONSUMED')
    _need(plan==build_plan(plan['source_base_commit']),'CURRENT_BINDINGS')
    head=cache._git('rev-parse','HEAD');subprocess.run(['git','merge-base','--is-ancestor',plan['source_base_commit'],head],cwd=REPO,check=True)
    q=_origin();policy_ref=cache._ref(REPO/'NEXT_RUN_POLICY.json');policy=cache._read(policy_ref)
    _need(all(policy.get(k) is False for k in ('training_enabled','native_training_enabled','native_launch_authorized','full_epoch_training_allowed','full_val_allowed')),'NO_TRAINING')
    if execute:
        gate=policy.get(STAGE,{})
        _need(all(type(gate.get(k)) is type(v) and gate[k]==v for k,v in gate_fields(ref).items()),'ADMISSION')
        _need(cache._git('branch','--show-current')=='work/gx1-current','BRANCH')
        campaign.require_clean_source(dict(source_repo=str(REPO),source_commit=head))
    bound=[ref,policy_ref,plan['worker'],q['failed_terminal'],q['failed_result'],
        *plan['helpers'].values(),*plan['inputs'].values(),*plan['source_owners'].values()]
    return dict(q,plan=plan,plan_ref=ref,head=head,bound_refs=bound)


def preflight(plan_path,plan_sha,out_dir):
    q=_inspect(plan_path,plan_sha,out_dir,execute=False)
    return dict(status='PASS_METADATA_ONLY_CURRENT_NUMERIC_ROUTES_PREFLIGHT',plan=q['plan_ref'],
        model_constructions=0,checkpoint_loads=0,array_decodes=0,claims=0,training_enabled=False,test_data_used=False)


def _save_route(route,piece,counts,*,np):
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    _need(route in ROUTES,'ROUTE_NAME');buffer=io.BytesIO();np.savez(buffer,**piece)
    _need(0<buffer.tell() and counts['published_capture_bytes']+buffer.tell()<=BOUNDS['max_capture_bytes'],'CAPTURE_SIZE')
    _publish_bytes(ROOT/(route+'.npz'),buffer.getvalue())
    counts['route_capture_writes']+=1;counts['published_capture_bytes']+=buffer.tell()
    return cache._ref(ROOT/(route+'.npz'))


def run(plan_path,plan_sha,out_dir,device='cuda'):
    started=time.monotonic();_need(device=='cuda','CUDA_ONLY')
    q=_inspect(plan_path,plan_sha,out_dir,execute=True);cap=previous._runtime_cap()
    import numpy as np
    import torch
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
    claim=cache._publish('CLAIM.json',dict(plan=q['plan_ref'],source_commit=q['head'],cap=cap,
        no_relaunch_under_this_plan=True,test_data_used=False),root=ROOT)
    counts={k:0 for k in ('model_constructions','checkpoint_loads','resume_state_load_calls','resume_state_load_returns',
        'archive_decodes','input_sidecar_archive_decodes','forwards','grad_forwards','route_capture_writes',
        'published_capture_bytes','autograd_grad_calls','optimizer_constructions','optimizer_updates',
        'backwards','rng_restores','checkpoint_writes','candidate_writes')}
    provider=prepared=model=None;before=None;route=None;records=[];artifacts={};pieces={};losses={};runtime={};passivity={}
    old_alarm=signal.getsignal(signal.SIGALRM)
    def timeout(*unused):raise TimeoutError('Current numerical routes300-second bound')
    def bookend():
        campaign.require_clean_source(dict(source_repo=str(REPO),source_commit=q['head']))
        for ref in q['bound_refs']:campaign.require_binding(ref,label='numeric routes bookend',verify_file=True)
    def signature():
        return dict(model=model_sha(model.state_dict()),buffers=old._tree_sha(dict(model.named_buffers())),
            rng=old._tree_sha(trainer._attended_session_rng_state(device=cuda)),
            modes={n:m.training for n,m in model.named_modules()},flags={n:p.requires_grad for n,p in model.named_parameters()},
            parameters={n:id(p) for n,p in model.named_parameters()},buffer_ids={n:id(p) for n,p in model.named_buffers()},
            modules={n:id(m) for n,m in model.named_modules()},view=copy.deepcopy(model.state_model.feature_view_identity),
            hooks={n:{k:tuple((i,id(h)) for i,h in getattr(m,k).items()) for k in
                ('_forward_hooks','_forward_pre_hooks','_backward_hooks','_backward_pre_hooks')} for n,m in model.named_modules()},
            mha_fastpath=torch.backends.mha.get_fastpath_enabled(),runtime=previous._runtime_profile(torch))
    def publish(complete,**extra):
        return cache._publish_fusion_outcome(q['plan_ref'],dict(claim=claim,source_commit=q['head'],counts=counts,
            route_records=records,artifacts=artifacts,numeric_runtime=runtime,passivity=passivity,
            model_state_sha256=parent.MODELS['INITIAL'],**extra),complete=complete,root=ROOT,
            json_bytes_bound=BOUNDS['max_published_json_bytes'])
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,max(1.,300-(time.monotonic()-started)))
    try:
        recipe,view,receipts=q['recipe'],q['view'],q['receipts'];numerical=recipe['numerical_recipe'];cuda=torch.device('cuda')
        trainer._set_deterministic(numerical['seed'],cuda,numerical['precision_policy'])
        runtime=previous._runtime_profile(torch);previous._require_runtime_profile(runtime)
        _need(torch.backends.mha.get_fastpath_enabled() is True and _BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False,'ORIGINAL_DEFAULT_PROFILE')
        with np.load(q['inputs']['INITIAL_capture']['path'],allow_pickle=False) as z:
            _need(set(z.files)==set(previous.BASE_KEYS),'CAPTURE_KEYS');baseline={k:z[k][:16].copy() for k in z.files}
        counts['archive_decodes']+=1
        counts['checkpoint_loads']+=1
        saved=common._load_state(q['inputs']['INITIAL_checkpoint'],torch=torch,maximum=128*1024**2,counters=counts)
        provider,provider_plan,prepared,rows,issues=full.anchor.prepare_fold1(receipts,view,np=np,counters=counts)
        require_m5_equal_observation_axes(rows,issues);order=torch.from_numpy(rows.copy())
        bound,future=load_bound_time_query_targets(),load_bound_future_displacement()
        counts['model_constructions']+=1
        model=NativeFutureDualModel(views.build_native_direction_research_model(view,
            prefix_normalization=receipts['fold1']['normal'],constructor_metadata=receipts['constructor'],
            device=torch.device('cpu'),unobserved_units_profile=receipts['fold1']['profile']))
        previous._short_parameter_partition(model,state)
        parent._load_endpoint(model,saved,'INITIAL',q,order,torch=torch,old=old,state=state,model_sha=model_sha)
        del saved;model.to(cuda);model.eval();before=signature()
        _need(before['model']==parent.MODELS['INITIAL'] and all(p.grad is None for p in model.parameters()),'INITIAL_READY')
        raw,parents,clocks=runner._feature_batch(prepared,view,receipts,rows,issues,0,16,np=np)
        features={k:torch.from_numpy(v).to(cuda) for k,v in raw.items()}
        feature_signature={k:old._tree_sha(v) for k,v in features.items()}
        truths=dict(state=bound.gather_fit_sampled(parents,clocks,fold_id=1,pass_index=0,seed=numerical['seed']),
            future=future.gather(parents,clocks,fold_id=1,partition='fit'))
        for route in ROUTES:
            grad_enabled=route=='DYNAMIC_GRAD';dynamic=route!='NATIVE_NO_GRAD'
            _need(counts['forwards']<3 and time.monotonic()-started<300,'FORWARD_BOUND')
            with contextlib.ExitStack() as contexts:
                contexts.enter_context(torch.set_grad_enabled(grad_enabled))
                if dynamic:
                    contexts.enter_context(trainer._training_mha_fastpath_context(model.backbone))
                    contexts.enter_context(trainer._training_autocast_context())
                    contexts.enter_context(trainer._training_model_finite_check_context())
                profile=dict(grad_enabled=torch.is_grad_enabled(),inference_mode=torch.is_inference_mode_enabled(),
                    mha_fastpath=torch.backends.mha.get_fastpath_enabled(),batch_equal_length_mtf_eval=_BATCH_EQUAL_LENGTH_MTF_EVAL.get())
                _need(profile==dict(grad_enabled=grad_enabled,inference_mode=False,mha_fastpath=not dynamic,batch_equal_length_mtf_eval=False)
                    and all(not m.training for m in model.modules()),'ROUTE_PROFILE')
                counts['forwards']+=1;counts['grad_forwards']+=int(grad_enabled)
                report=evaluate_fit_batch(model,features,bound,future,parents,clocks,fold_id=1,pass_index=0,
                    seed=numerical['seed'],future_weight=1.,m5_equal_observation=True,fit_batch_start=0)
                _need(report['skip_update'] is False and report['loss'].requires_grad==grad_enabled
                    and report['model_contract_sha256']==recipe['expected_model_contract_sha256']
                    and report['loss_contract_sha256']==recipe['expected_loss_contract_sha256'],'REPORT_BINDING')
                piece=dict(parents=parents.copy(),issues=clocks.copy(),state_query=report['state_query_k'].copy(),
                    future_query=np.broadcast_to(report['future_native_k'],(16,5)).copy())
                for task in ('state','future'):
                    _need(np.array_equal(truths[task]['parent_rows'],parents) and np.array_equal(truths[task]['issue_ns'],clocks)
                        and np.array_equal(truths[task]['allowed'],report[task+'_allowed'])
                        and np.array_equal(truths[task]['mature'],report[task+'_mature']),'TARGET_AXES')
                    piece[task+'_raw']=report[task+'_activity_side_logits'].detach().cpu().numpy().copy()
                    for name in ('allowed','eligible','mature'):piece[task+'_'+name]=truths[task][name].copy()
                loss=float(report['loss'].detach());del report
            artifacts[route]=_save_route(route,piece,counts,np=np);pieces[route]=piece;losses[route]=loss
            record=dict(route=route,profile=profile,loss=loss)
            records.append(record)
            record['retained_comparison']=compare_pieces(piece,baseline,loss,
                q['measurements']['INITIAL']['measurement']['canonical_batch_losses'][0],np=np)
            if route=='NATIVE_NO_GRAD':_need(record['retained_comparison']['within_existing_tolerance'],'NATIVE_RETAINED_CONTROL')
            else:
                record['native_route_comparison']=compare_pieces(piece,pieces['NATIVE_NO_GRAD'],loss,losses['NATIVE_NO_GRAD'],np=np)
            if route=='DYNAMIC_GRAD':
                record['dynamic_no_grad_comparison']=compare_pieces(piece,pieces['DYNAMIC_NO_GRAD'],loss,losses['DYNAMIC_NO_GRAD'],np=np)
            passivity[route]=dict(signature_unchanged=signature()==before,no_parameter_gradients=all(p.grad is None for p in model.parameters()),
                features_unchanged={k:old._tree_sha(v) for k,v in features.items()}==feature_signature)
            _need(all(passivity[route].values()),'PASSIVITY')
        for key,value in counts.items():
            if key!='published_capture_bytes':_need(value==BOUNDS[key],'COUNTS:'+key)
        full.anchor.release_inputs(provider,prepared);prepared=None;provider.validate_metadata(provider_plan);bookend()
        return publish(True,source_unchanged=True,elapsed_seconds=time.monotonic()-started,
            native_retained_control_qualified=records[0]['retained_comparison']['within_existing_tolerance'],
            dynamic_grad_vs_no_grad_qualified=records[2]['dynamic_no_grad_comparison']['within_existing_tolerance'],
            cross_route_equivalence_qualified=all(x['retained_comparison']['within_existing_tolerance'] for x in records),
            limitations=[q['plan']['interpretation'],'One outcome-blind first B16 batch at INITIAL only; no FINAL, population, historical gradient or economic claim.'])
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL,0);changed=state_error=None
        try:
            if before is not None:passivity['failure']=dict(signature_unchanged=signature()==before,no_parameter_gradients=all(p.grad is None for p in model.parameters()))
        except Exception as other:state_error=repr(other)
        try:bookend()
        except Exception as other:changed=repr(other)
        result=publish(False,source_unchanged=changed is None,error=type(exc).__name__+':'+str(exc),
            active_route=route,bookend_error=changed,passivity_error=state_error,interpretation_authorized=False)
        raise RuntimeError('TA_CURRENT_NUMERIC_ROUTES_FAILED_CONSUMED:'+str(result)) from exc
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_alarm)
        if prepared is not None:full.anchor.release_inputs(provider,prepared)


def main(argv=None):
    import argparse,json,sys
    supplied=list(sys.argv[1:] if argv is None else argv)
    flags=('--market-state-ta-current-numeric-routes-plan','--market-state-ta-current-numeric-routes-plan-sha256')
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False,add_help=False)
    parser.add_argument(flags[0],type=Path,required=True);parser.add_argument(flags[1],required=True)
    parser.add_argument('--device',choices=('cuda',),required=True);parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    for flag in (*flags,'--device','--out-dir','--preflight'):
        if sum(v.partition('=')[0]==flag for v in supplied)>1:parser.error('duplicate flag: '+flag)
    args=parser.parse_args(supplied)
    kwargs=dict(plan_path=args.market_state_ta_current_numeric_routes_plan,
        plan_sha=args.market_state_ta_current_numeric_routes_plan_sha256,out_dir=args.out_dir)
    print(json.dumps(preflight(**kwargs) if args.preflight else run(**kwargs,device=args.device),sort_keys=True,allow_nan=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
