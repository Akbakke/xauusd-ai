"""Bounded paired current train/eval forwards; no gradients or parameter updates."""
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
from gx1.audit.observe_market_state_ta_current_numeric_routes_v1 import compare_pieces

cache, campaign, full, previous, REPO = parent.cache, parent.campaign, parent.full, parent.previous, parent.REPO
STAGE = 'market_state_ta_current_mode_observation_20261113_001'
ROOT = Path('/home/andre2/GX1_RUNS') / STAGE.upper()
WORKER = REPO/'gx1/audit/observe_market_state_ta_current_modes_v1.py'
PHASES, MODELS, PANEL_SLOTS, BATCHES = parent.PHASES, parent.MODELS, parent.PANEL_SLOTS, parent.BATCHES
ROUTES = ('NATIVE_NO_GRAD', 'EVAL_DYNAMIC_GRAD', 'TRAIN_ZERO_DROPOUT_GRAD',
          'TRAIN_DROPOUT_0', 'TRAIN_DROPOUT_1', 'TRAIN_DROPOUT_2', 'TRAIN_DROPOUT_3')
OBSERVER_SEED = 20261113
QUALIFICATION_ROOT = ROOT.parent/'MARKET_STATE_TA_CURRENT_UPDATE_OBSERVATION_20261113_003'
QUALIFICATION_REFS = {
    'result': dict(path=str(QUALIFICATION_ROOT/'RESULT.json'), sha256='d5efba0db7e6958c8cd902ea5ef4211fe22d39f4e8310c95b9384207dad2a055'),
    'terminal': dict(path=str(QUALIFICATION_ROOT/'TERMINAL.json'), sha256='81432923c80298de97f62cf6505ad3549896a46358725324b9e9c4a1ef8f1624'),
    'validation': dict(path=str(QUALIFICATION_ROOT/'REVIEW_VALIDATION.json'), sha256='cc4d547faaa45b7896fd8c7cf697c41964f63d0326d401f119b73e8e0ef0a04c'),
}
HELPERS = tuple(dict.fromkeys((str(parent.WORKER.relative_to(REPO)), *parent.HELPERS,
    '.venv/lib/python3.10/site-packages/torch/nn/modules/activation.py',
    '.venv/lib/python3.10/site-packages/torch/nn/modules/dropout.py',
    '.venv/lib/python3.10/site-packages/torch/nn/functional.py')))
BOUNDS = dict(batch_size=16, panel_slots=PANEL_SLOTS, original_batch_indices=BATCHES,
    unique_rows=256, sampled_rows=512, captured_rows=3584, capture_records=224,
    model_constructions=1, checkpoint_loads=2, resume_state_load_calls=2, resume_state_load_returns=2,
    archive_decodes=2, input_sidecar_archive_decodes=6, checkpoint_weight_transitions=1,
    forwards=224, grad_forwards=192, native_reference_matches=32, extra_target_gathers=64,
    deterministic_initialization_calls=2, observer_rng_seed_initializations=1, rng_restores=1,
    phase_capture_writes=2, autograd_grad_calls=0, backwards=0, optimizer_constructions=0,
    optimizer_updates=0, checkpoint_writes=0, candidate_writes=0,
    wall_seconds=600, memory_bytes=12*1024**3, numerical_threads=8,
    max_capture_bytes=4*1024**2, max_published_json_bytes=8*1024**2,
    raw_rtol=1e-6, raw_atol=2e-6, loss_atol=5e-6, training_enabled=False, test_data_used=False)
COUNT_KEYS = ('model_constructions','checkpoint_loads','resume_state_load_calls','resume_state_load_returns',
    'archive_decodes','input_sidecar_archive_decodes','checkpoint_weight_transitions','forwards','grad_forwards',
    'native_reference_matches','extra_target_gathers','deterministic_initialization_calls',
    'observer_rng_seed_initializations','rng_restores','phase_capture_writes','autograd_grad_calls',
    'backwards','optimizer_constructions','optimizer_updates','checkpoint_writes','candidate_writes',
    'sampled_rows','captured_rows','capture_records')


def _need(ok, reason):
    if not bool(ok): raise RuntimeError('TA_CURRENT_MODES_' + reason)


def _metadata():
    q = parent._metadata()
    result, terminal, validation = (cache._read(QUALIFICATION_REFS[k]) for k in ('result','terminal','validation'))
    prior_plan = cache._read(result['plan'])
    _need(result['status'] == terminal['status'] == 'COMPLETE' and terminal['exit_code'] == 0
        and result['source_unchanged'] is terminal['source_unchanged'] is True
        and terminal['result'] == validation['result'] == QUALIFICATION_REFS['result']
        and validation['terminal'] == QUALIFICATION_REFS['terminal']
        and result['plan'] == terminal['plan'] == validation['plan']
        and result['model_digests'] == MODELS
        and validation['status'] == 'PASS_SOURCE_COUNTS_REFERENCE_GATES_PASSIVITY_AND_SCALAR_CLOSURES'
        and validation['passivity_verified'] is True
        and validation['native_retained_bitexact_batches'] == validation['dynamic_grad_no_grad_bitexact_batches'] == 32
        and validation['unchanged_original_tolerances'] == {k:BOUNDS[k] for k in ('raw_rtol','raw_atol','loss_atol')}
        and result['counts']['forwards'] == 96 and result['counts']['autograd_grad_calls'] == 256
        and result['counts']['optimizer_updates'] == 0, 'COMPLETED_CURRENT_QUALIFICATION')
    campaign.require_binding(prior_plan['worker'], label='unchanged current observer helpers', verify_file=True)
    _need(prior_plan['worker']['path'] == str(parent.WORKER), 'PARENT_OWNER')
    inputs = dict(q['inputs'], **{'qualified_current_'+k:v for k,v in QUALIFICATION_REFS.items()},
                  qualified_current_plan=result['plan'])
    return dict(q, inputs=inputs)


def build_plan(source_base_commit):
    q = _metadata()
    return dict(schema_version='gx1_ta_current_mode_observation_plan_v1', stage=STAGE, source_repo=str(REPO),
        source_base_commit=source_base_commit, worker=cache._ref(WORKER), source_owners=q['source_owners'],
        helpers={name:cache._ref(REPO/name) for name in HELPERS}, inputs=q['inputs'],
        model_digests=MODELS, feature_view_sha256=q['view']['contract_sha256'],
        numerical_recipe_sha256=q['recipe']['numerical_recipe']['contract_sha256'],
        target_query_seed=q['recipe']['numerical_recipe']['seed'], observer_seed=OBSERVER_SEED,
        routes=list(ROUTES), bounds=BOUNDS,
        rng_policy='Fresh observer seed after INITIAL load; sequential four dropout draws per batch; one exact FINAL restore to that observer start. No historical-mask claim.',
        interpretation='Fixed TRAIN subset at two fixed weights. Mode differences and four stochastic draws are observations, not expected-dropout, historical training, generalization or causal edge proofs.')


def gate_fields(ref):
    return dict(status='ADMITTED', plan=ref, max_invocations=1, remaining_invocations=1,
        claimed_invocations=0, completed_invocations=0, relaunch_allowed=False, execution_authorized=True,
        data_reads_authorized=True, checkpoint_reads_authorized=True, training_authorized=False, bounds=BOUNDS)


def _inspect(plan_path, plan_sha, out_dir, *, execute):
    _need(Path(plan_path) == ROOT/'PLAN.json' and Path(out_dir) == ROOT, 'EXACT_PATHS')
    ref = dict(path=str(plan_path), sha256=plan_sha); plan = cache._read(ref)
    _need(not any((ROOT/name).exists() for name in ('CLAIM.json','RESULT.json','TERMINAL.json')), 'CONSUMED')
    _need(plan == build_plan(plan['source_base_commit']), 'CURRENT_PLAN_BINDINGS')
    head = cache._git('rev-parse','HEAD')
    subprocess.run(['git','merge-base','--is-ancestor',plan['source_base_commit'],head], cwd=REPO, check=True)
    q = _metadata(); policy_ref = cache._ref(REPO/'NEXT_RUN_POLICY.json'); policy = cache._read(policy_ref)
    _need(all(policy.get(k) is False for k in ('training_enabled','native_training_enabled',
        'native_launch_authorized','full_epoch_training_allowed','full_val_allowed')), 'NO_TRAINING')
    if execute:
        gate = policy.get(STAGE,{})
        _need(all(type(gate.get(k)) is type(v) and gate[k] == v for k,v in gate_fields(ref).items()), 'ADMISSION')
        _need(cache._git('branch','--show-current') == 'work/gx1-current', 'BRANCH')
        campaign.require_clean_source(dict(source_repo=str(REPO),source_commit=head))
    refs = [ref,policy_ref,plan['worker'],*plan['helpers'].values(),*plan['inputs'].values(),*plan['source_owners'].values()]
    return dict(q,plan=plan,plan_ref=ref,head=head,bound_refs=refs)


def preflight(plan_path, plan_sha, out_dir):
    q = _inspect(plan_path,plan_sha,out_dir,execute=False)
    return dict(status='PASS_METADATA_ONLY_CURRENT_MODES_PREFLIGHT', stage=STAGE, plan=q['plan_ref'],
        model_constructions=0, checkpoint_loads=0, array_decodes=0, claims=0,
        training_enabled=False, test_data_used=False)


def _dropout_probabilities(model, *, torch):
    values = {}
    for name,module in model.named_modules():
        field = 'p' if isinstance(module,torch.nn.Dropout) else (
            'dropout' if isinstance(module,torch.nn.MultiheadAttention) else None)
        if field is not None:
            value = getattr(module,field)
            _need(type(value) in (int,float) and math.isfinite(value) and 0 <= value < 1, 'DROPOUT_DOMAIN:'+name)
            values[name] = dict(field=field, probability=value)
    return values


@contextlib.contextmanager
def temporary_mode(model, *, training, zero_dropout, torch):
    """Restore every mode and both FF/residual and attention probabilities on error."""
    _need(type(training) is bool and type(zero_dropout) is bool and (not zero_dropout or training), 'MODE_REQUEST')
    modules = dict(model.named_modules()); modes = {n:m.training for n,m in modules.items()}
    probabilities = _dropout_probabilities(model,torch=torch)
    try:
        model.train(training)
        if zero_dropout:
            for name,value in probabilities.items(): setattr(modules[name],value['field'],0.)
        yield
    finally:
        for name,value in probabilities.items(): setattr(modules[name],value['field'],value['probability'])
        for name,mode in modes.items(): modules[name].training = mode


def describe_route(route, piece, reference, loss, reference_loss, *, np, eval_piece=None, eval_loss=None):
    _need(route in ROUTES, 'ROUTE')
    comparison = compare_pieces(piece,reference,loss,reference_loss,np=np)
    value = dict(route=route,loss=float(loss),retained_comparison=comparison,
        required_reference_qualified=comparison['within_existing_tolerance'] if route == ROUTES[0] else None,
        outputs={})
    for task in ('state','future'):
        raw = piece[task+'_raw'].astype(np.float64)
        value['outputs'][task] = dict(mean_raw_by_tf_channel=raw.mean(axis=0).tolist(),
            mean_probability_by_tf_channel=np.exp(-np.logaddexp(0.,-raw)).mean(axis=0).tolist())
    if route.startswith('TRAIN_'):
        _need(eval_piece is not None and eval_loss is not None, 'DYNAMIC_EVAL_REFERENCE')
        value['dynamic_eval_comparison'] = compare_pieces(piece,eval_piece,loss,eval_loss,np=np)
    return value


def require_paired_rng(initial, final):
    _need(isinstance(initial,list) and isinstance(final,list) and len(initial) == len(final) == len(BATCHES), 'RNG_BATCH_COUNT')
    for index,(a,b) in enumerate(zip(initial,final)):
        _need(a['panel_slot'] == PANEL_SLOTS[index] and a['original_batch_index'] == BATCHES[index]
            and [r['route'] for r in a['routes']] == list(ROUTES)
            and all(type(a[k]) is str and len(a[k]) == 64 for k in ('before','after'))
            and a == b, 'PAIRED_RNG_BATCH:'+str(index))
        _need(a['before'] == a['routes'][0]['before'] and a['after'] == a['routes'][-1]['after']
            and all(left['after'] == right['before'] for left,right in zip(a['routes'],a['routes'][1:])), 'RNG_ROUTE_CHAIN')
        if index: _need(initial[index-1]['after'] == a['before'], 'RNG_BATCH_CHAIN')


def _save_phase(phase, pieces, counts, *, np):
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    _need(phase in PHASES and 0 < len(pieces) <= len(BATCHES)*len(ROUTES), 'CAPTURE_PIECES')
    arrays = {key:np.concatenate([p[key] for p in pieces],axis=0) for key in pieces[0]}
    buffer = io.BytesIO(); np.savez(buffer,**arrays)
    _need(0 < buffer.tell() and counts['published_capture_bytes']+buffer.tell() <= BOUNDS['max_capture_bytes'], 'CAPTURE_BOUND')
    path = ROOT/(phase+'_MODE_ROUTES.npz'); _publish_bytes(path,buffer.getvalue())
    counts['phase_capture_writes'] += 1; counts['published_capture_bytes'] += buffer.tell()
    return dict(ref=cache._ref(path),rows=len(arrays['parents']),route_records=len(pieces),
        complete_phase=len(pieces) == len(BATCHES)*len(ROUTES))


def _summary(records, *, np):
    result = {}
    for route in ROUTES:
        selected = [r for r in records if r['route'] == route]
        if not selected: continue
        result[route] = dict(batches=len(selected),rows=16*len(selected),
            mean_original_loss=float(np.mean([r['loss'] for r in selected],dtype=np.float64)),
            outputs={task:{key:np.mean([r['outputs'][task][key] for r in selected],axis=0,dtype=np.float64).tolist()
                for key in ('mean_raw_by_tf_channel','mean_probability_by_tf_channel')} for task in ('state','future')})
    return result


def run(plan_path, plan_sha, out_dir, device='cuda'):
    started = time.monotonic(); _need(device == 'cuda','CUDA_ONLY')
    q = _inspect(plan_path,plan_sha,out_dir,execute=True); cap = previous._runtime_cap()
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
    _need(not torch.cuda.is_initialized() and signal.getitimer(signal.ITIMER_REAL) == (0.,0.), 'INIT_ORDER')
    old_alarm = signal.getsignal(signal.SIGALRM)
    claim = cache._publish('CLAIM.json',dict(plan=q['plan_ref'],source_commit=q['head'],cap=cap,
        no_relaunch_under_this_plan=True,test_data_used=False),root=ROOT)
    counts = {k:0 for k in COUNT_KEYS}; counts['published_capture_bytes'] = 0
    provider=prepared=model=None; phase=slot=None; records=[]; pieces={p:[] for p in PHASES}
    artifacts={}; summaries={}; passivity={}; before={}; runtime={}; transition={}; rng_phases={}; rng_batches={}
    observer_rng_sha=None; paired_rng=False; dropout_inventory={}
    def timeout(*unused): raise TimeoutError('Current modes600-second bound')
    def rng_sha(): return old._tree_sha(trainer._attended_session_rng_state(device=cuda))
    def signature():
        return dict(model=model_sha(model.state_dict()),buffers=old._tree_sha(dict(model.named_buffers())),
            modes={n:m.training for n,m in model.named_modules()},flags={n:p.requires_grad for n,p in model.named_parameters()},
            parameters={n:id(p) for n,p in model.named_parameters()},buffer_ids={n:id(p) for n,p in model.named_buffers()},
            modules={n:id(m) for n,m in model.named_modules()},view=copy.deepcopy(model.state_model.feature_view_identity),
            hooks={n:{k:tuple((i,id(h)) for i,h in getattr(m,k).items()) for k in
                ('_forward_hooks','_forward_pre_hooks','_backward_hooks','_backward_pre_hooks')} for n,m in model.named_modules()},
            dropout=_dropout_probabilities(model,torch=torch),mha_fastpath=torch.backends.mha.get_fastpath_enabled(),
            mtf_batch_equal=_BATCH_EQUAL_LENGTH_MTF_EVAL.get(),runtime=previous._runtime_profile(torch))
    def bookend():
        campaign.require_clean_source(dict(source_repo=str(REPO),source_commit=q['head']))
        for ref in q['bound_refs']: campaign.require_binding(ref,label='current modes bookend',verify_file=True)
    def persist(active_phase):
        if active_phase in PHASES and pieces[active_phase] and active_phase not in artifacts:
            artifacts[active_phase] = _save_phase(active_phase,pieces[active_phase],counts,np=np)
    def publish(complete, **extra):
        return cache._publish_fusion_outcome(q['plan_ref'],dict(claim=claim,source_commit=q['head'],counts=counts,
            artifacts=artifacts,batch_records=records,phase_summary=summaries,passivity=passivity,
            model_digests=MODELS,numeric_runtime=runtime,checkpoint_transition=transition,dropout_inventory=dropout_inventory,
            observer_seed=OBSERVER_SEED,target_query_seed=q['recipe']['numerical_recipe']['seed'],
            observer_start_rng_sha256=observer_rng_sha,rng_phases=rng_phases,rng_batches=rng_batches,
            paired_observer_rng_exact=paired_rng,historical_training_masks_matched=False,**extra),
            complete=complete,root=ROOT,json_bytes_bound=BOUNDS['max_published_json_bytes'])
    try:
        signal.signal(signal.SIGALRM,timeout); signal.setitimer(signal.ITIMER_REAL,max(1.,600-(time.monotonic()-started)))
        recipe,view,receipts = q['recipe'],q['view'],q['receipts']; numerical=recipe['numerical_recipe']; cuda=torch.device('cuda')
        counts['deterministic_initialization_calls'] += 1
        trainer._set_deterministic(numerical['seed'],cuda,numerical['precision_policy'])
        runtime=previous._runtime_profile(torch); previous._require_runtime_profile(runtime)
        _need(_BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False and torch.backends.mha.get_fastpath_enabled() is True,'RUNTIME')
        captures={}; states={}
        for load_phase in PHASES:
            counts['archive_decodes'] += 1
            with np.load(q['inputs'][load_phase+'_capture']['path'],allow_pickle=False) as z: captures[load_phase]={k:z[k].copy() for k in z.files}
            _need(set(captures[load_phase]) == set(previous.BASE_KEYS) and all(len(a)==4096 for a in captures[load_phase].values()),'CAPTURE_LAYOUT')
            counts['checkpoint_loads'] += 1
            states[load_phase]=common._load_state(q['inputs'][load_phase+'_checkpoint'],torch=torch,maximum=128*1024**2,counters=counts)
        provider,provider_plan,prepared,rows,issues=full.anchor.prepare_fold1(receipts,view,np=np,counters=counts)
        require_m5_equal_observation_axes(rows,issues); order=torch.from_numpy(rows.copy())
        parent._require_parent_order(order,states,torch=torch)
        bound,future=load_bound_time_query_targets(),load_bound_future_displacement()
        counts['model_constructions'] += 1
        model=NativeFutureDualModel(views.build_native_direction_research_model(view,
            prefix_normalization=receipts['fold1']['normal'],constructor_metadata=receipts['constructor'],
            device=torch.device('cpu'),unobserved_units_profile=receipts['fold1']['profile']))
        previous._short_parameter_partition(model,state)
        initial,final=states['INITIAL'],states['FINAL']
        _need(model_sha(final['model_state']) == final['model_state_sha256'] == MODELS['FINAL']
            and final['parameter_binding']['feature_view_identity'] == parent._view_identity(q['views']['FINAL'])
            and all(initial['binding'][k] == final['parameter_binding'][k] for k in
                ('parameter_inventory','active_parameter_names','inactive_parameter_names','optimizer_group_parameter_names','research_objective')), 'PAIRED_STATE_BINDINGS')
        parent._load_endpoint(model,initial,'INITIAL',q,order,torch=torch,old=old,state=state,model_sha=model_sha)
        model.to(cuda); model.eval()
        probabilities=_dropout_probabilities(model,torch=torch)
        _need(probabilities and model.backbone.cfg.dropout == receipts['constructor']['dropout'] == .05, 'CONFIGURED_DROPOUT_PROFILE')
        dropout_inventory=dict(configured_probability=float(model.backbone.cfg.dropout),modules=probabilities,
            module_counts={field:sum(v['field'] == field for v in probabilities.values()) for field in ('p','dropout')})
        # This is a new observation stream, separate from immutable target-query seed and saved training RNG.
        counts['deterministic_initialization_calls'] += 1; counts['observer_rng_seed_initializations'] += 1
        trainer._set_deterministic(OBSERVER_SEED,cuda,numerical['precision_policy'])
        observer_rng=copy.deepcopy(trainer._attended_session_rng_state(device=cuda)); observer_rng_sha=old._tree_sha(observer_rng)
        for phase in PHASES:
            if phase == 'FINAL':
                prior=signature(); prior_rng=rng_sha()
                parent._load_endpoint(model,final,phase,q,order,torch=torch,old=old,state=state,model_sha=model_sha)
                counts['checkpoint_weight_transitions'] += 1; after=signature()
                _need(all(prior[k] == after[k] for k in prior if k not in ('model','buffers')) and rng_sha()==prior_rng, 'WEIGHTS_ONLY_TRANSITION')
                transition=dict(initial_model=prior['model'],final_model=after['model'],
                    initial_buffers=prior['buffers'],final_buffers=after['buffers'],non_weight_state_unchanged=True)
                counts['rng_restores'] += 1
                trainer._restore_attended_session_rng_state(observer_rng,device=cuda)
            _need(rng_sha() == observer_rng_sha,'PHASE_RNG_START')
            before[phase]=signature(); rng_phases[phase]=dict(before=rng_sha()); rng_batches[phase]=[]
            for slot,batch in zip(PANEL_SLOTS,BATCHES):
                raw,parents,clocks=runner._feature_batch(prepared,view,receipts,rows,issues,batch*16,batch*16+16,np=np)
                features={k:torch.from_numpy(v).to(cuda) for k,v in raw.items()}
                feature_signature={k:old._tree_sha(v) for k,v in features.items()}
                counts['extra_target_gathers'] += 2
                truths=dict(state=bound.gather_fit_sampled(parents,clocks,fold_id=1,pass_index=0,seed=numerical['seed']),
                    future=future.gather(parents,clocks,fold_id=1,partition='fit'))
                retained={k:v[slot*16:slot*16+16] for k,v in captures[phase].items()}
                retained_loss=q['measurements'][phase]['measurement']['canonical_batch_losses'][slot]
                rng_batch=dict(panel_slot=slot,original_batch_index=batch,before=rng_sha(),routes=[])
                rng_batches[phase].append(rng_batch); eval_piece=eval_loss=None
                for route_index,route in enumerate(ROUTES):
                    grad_enabled=route != 'NATIVE_NO_GRAD'; training=route.startswith('TRAIN_'); zero_dropout=route == 'TRAIN_ZERO_DROPOUT_GRAD'
                    _need(counts['forwards'] < BOUNDS['forwards'] and time.monotonic()-started < BOUNDS['wall_seconds'],'FORWARD_BOUND')
                    rng_before=rng_sha()
                    with temporary_mode(model,training=training,zero_dropout=zero_dropout,torch=torch), contextlib.ExitStack() as contexts:
                        contexts.enter_context(torch.set_grad_enabled(grad_enabled))
                        if grad_enabled:
                            contexts.enter_context(trainer._training_mha_fastpath_context(model.backbone))
                            contexts.enter_context(trainer._training_autocast_context())
                            contexts.enter_context(trainer._training_model_finite_check_context())
                        _need(all(m.training is training for m in model.modules()) and torch.is_grad_enabled() is grad_enabled
                            and not torch.is_inference_mode_enabled() and torch.backends.mha.get_fastpath_enabled() is (not grad_enabled)
                            and _BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False,'FORWARD_PROFILE')
                        counts['forwards'] += 1; counts['grad_forwards'] += int(grad_enabled)
                        report=evaluate_fit_batch(model,features,bound,future,parents,clocks,fold_id=1,pass_index=0,
                            seed=numerical['seed'],future_weight=1.,m5_equal_observation=True,fit_batch_start=batch*16)
                        _need(report['skip_update'] is False and report['loss'].requires_grad is grad_enabled
                            and report['model_contract_sha256'] == recipe['expected_model_contract_sha256']
                            and report['loss_contract_sha256'] == recipe['expected_loss_contract_sha256'],'REPORT_BINDING')
                        piece=dict(parents=parents.copy(),issues=clocks.copy(),state_query=report['state_query_k'].copy(),
                            future_query=np.broadcast_to(report['future_native_k'],(16,5)).copy())
                        for task in ('state','future'):
                            _need(np.array_equal(truths[task]['parent_rows'],parents) and np.array_equal(truths[task]['issue_ns'],clocks)
                                and np.array_equal(truths[task]['allowed'],report[task+'_allowed'])
                                and np.array_equal(truths[task]['mature'],report[task+'_mature']),'TARGET_AXES')
                            piece[task+'_raw']=report[task+'_activity_side_logits'].detach().cpu().numpy().copy()
                            for key in ('allowed','eligible','mature'): piece[task+'_'+key]=truths[task][key].copy()
                        loss=float(report['loss'].detach()); del report
                        _need(math.isfinite(loss),'LOSS_FINITE')
                    rng_after=rng_sha(); rng_batch['routes'].append(dict(route=route,before=rng_before,after=rng_after))
                    pieces[phase].append(dict(piece,route_index=np.full(16,route_index,dtype=np.int8),
                        panel_slot=np.full(16,slot,dtype=np.int16),original_batch_index=np.full(16,batch,dtype=np.int32),
                        batch_loss=np.full(16,loss,dtype=np.float64)))
                    counts['capture_records'] += 1; counts['captured_rows'] += 16
                    record=dict(phase=phase,panel_slot=slot,original_batch_index=batch,route=route,
                        grad_enabled=grad_enabled,training=training,zero_dropout=zero_dropout,rng_before=rng_before,rng_after=rng_after,loss=loss)
                    records.append(record)
                    record.update(describe_route(route,piece,retained,loss,retained_loss,np=np,eval_piece=eval_piece,eval_loss=eval_loss))
                    _need(record['required_reference_qualified'] is not False,'NATIVE_REFERENCE_PARITY')
                    if route == 'NATIVE_NO_GRAD': counts['native_reference_matches'] += 1
                    if route == 'EVAL_DYNAMIC_GRAD': eval_piece,eval_loss=piece,loss
                    if route_index < 3: _need(rng_before==rng_after,'NONSTOCHASTIC_ROUTE_RNG')
                    record['features_unchanged']={k:old._tree_sha(v) for k,v in features.items()} == feature_signature
                    _need(record['features_unchanged'] and all(p.grad is None for p in model.parameters()),'PASSIVE_INPUTS_AND_PARAMETERS')
                rng_batch['after']=rng_sha(); counts['sampled_rows'] += 16
                del features,raw,eval_piece
            rng_phases[phase]['after']=rng_sha()
            passivity[phase]=dict(non_rng_signature_unchanged=signature()==before[phase],
                no_parameter_gradients=all(p.grad is None for p in model.parameters()))
            _need(all(passivity[phase].values()),'PHASE_PASSIVITY')
            summaries[phase]=_summary([r for r in records if r['phase']==phase],np=np); persist(phase)
        require_paired_rng(rng_batches['INITIAL'],rng_batches['FINAL'])
        _need(rng_phases['INITIAL']==rng_phases['FINAL'],'PAIRED_PHASE_RNG'); paired_rng=True
        for key in COUNT_KEYS: _need(counts[key] == BOUNDS[key],'COUNTS:'+key)
        full.anchor.release_inputs(provider,prepared); prepared=None; provider.validate_metadata(provider_plan); bookend()
        return publish(True,source_unchanged=True,elapsed_seconds=time.monotonic()-started,limits=[q['plan']['interpretation'],
            '16 intact dependent B16 groups per endpoint; four dropout draws are not the full distribution of masks.',
            'All inputs, original sampled-S/native-F targets and losses are unchanged. Only native-route retained parity is required; other route differences are reported, never relabeled equivalent.',
            'Means cover all rows; probability means are means of sigmoid(raw), not sigmoid(mean raw). Conditional-side interpretation requires the retained known-label masks.',
            'RNG evolves intentionally during train/dropout routes; only the fresh paired observer streams are matched. No historical-mask or optimizer-path reconstruction.'])
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL,0); changed=passivity_error=capture_error=None
        try:
            if phase in before:
                passivity[phase]=dict(non_rng_signature_unchanged=signature()==before[phase],no_parameter_gradients=all(p.grad is None for p in model.parameters()))
                rng_phases[phase]['after']=rng_sha()
        except Exception as other: passivity_error=repr(other)
        try: persist(phase)
        except Exception as other: capture_error=repr(other)
        try: bookend()
        except Exception as other: changed=repr(other)
        result=publish(False,source_unchanged=changed is None,active_phase=phase,active_panel_slot=slot,
            error=type(exc).__name__+':'+str(exc),bookend_error=changed,passivity_error=passivity_error,
            capture_error=capture_error,interpretation_authorized=False,elapsed_seconds=time.monotonic()-started)
        raise RuntimeError('TA_CURRENT_MODES_FAILED_CONSUMED:'+str(result)) from exc
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,old_alarm)
        if prepared is not None: full.anchor.release_inputs(provider,prepared)


def main(argv=None):
    import argparse,json,sys
    supplied=list(sys.argv[1:] if argv is None else argv)
    flags=('--market-state-ta-current-modes-plan','--market-state-ta-current-modes-plan-sha256')
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False,add_help=False)
    parser.add_argument(flags[0],type=Path,required=True); parser.add_argument(flags[1],required=True)
    parser.add_argument('--device',choices=('cuda',),required=True); parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    for flag in (*flags,'--device','--out-dir','--preflight'):
        if sum(v.partition('=')[0] == flag for v in supplied) > 1: parser.error('duplicate flag: '+flag)
    if '--preflight' in supplied and supplied[-1] != '--preflight': parser.error('--preflight must be last')
    args=parser.parse_args(supplied)
    kwargs=dict(plan_path=args.market_state_ta_current_modes_plan,plan_sha=args.market_state_ta_current_modes_plan_sha256,out_dir=args.out_dir)
    print(json.dumps(preflight(**kwargs) if args.preflight else run(**kwargs,device=args.device),sort_keys=True,allow_nan=False))
    return 0


if __name__ == '__main__': raise SystemExit(main())
