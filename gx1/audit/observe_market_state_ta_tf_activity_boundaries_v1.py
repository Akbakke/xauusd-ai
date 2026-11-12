"""One passive paired full-prefix activity-model boundary observation."""
from __future__ import annotations

import copy
import io
import math
from pathlib import Path
import signal
import subprocess
import time

from gx1.audit import observe_market_state_ta_tf_activity_initial_inputs_v1 as initial
from gx1.audit.ta_c171_joint_boundary_capture_v1 import capture_joint_boundaries

cache, previous, campaign, short = initial.cache, initial.previous, initial.campaign, initial.short
REPO = initial.REPO
STAGE = 'market_state_ta_tf_activity_boundaries_20261112_001'
SCOPE = 'market_state_ta_tf_activity_boundaries_v1'
ROOT = Path('/home/andre2/GX1_RUNS')/STAGE.upper()
SCHEMA = 'gx1_ta_tf_activity_boundaries_plan_v1'
WORKER = REPO/'gx1/audit/observe_market_state_ta_tf_activity_boundaries_v1.py'
PHASES = ('INITIAL_TRAIN','FINAL_TRAIN')
FINAL_MODEL = 'f418c9194f94dcafd4ae18efd6266e25417a25e725e5b2d8f1d12b55ead88ca6'
INITIAL_INPUT_RESULT = dict(path=str(initial.ROOT/'RESULT.json'),sha256='6aac91031b1dfdef16ddac4ff311c378ab04601cfec1f1e95ebf52b793535780')
INITIAL_INPUT_TERMINAL = dict(path=str(initial.ROOT/'TERMINAL.json'),sha256='84433f8c936a4aba446eccd69d77c80af93a69f38ed4a91fd2f23bba22ae5450')
HELPERS = {
    'initial_inputs':dict(path=str(initial.WORKER),sha256='eab78c1d362348d245c60d62e6de80254620ba8b5406bd89a6cbdc77a27f6f0a'),
    'joint_capture':dict(path=str(REPO/'gx1/audit/ta_c171_joint_boundary_capture_v1.py'),sha256='791c4283b021d9ff071119e3292a8134c80e7696f4aade310a8e5afb049d1094'),
}
BOUNDS = dict(initial.BOUNDS, sampled_rows=8192, unique_rows=4096, phases=list(PHASES),
    max_checkpoint_loads=2, max_backbone_encodings=512, max_extra_target_gathers=1024,
    registered_head_hooks=20, max_head_input_observations=5120, max_joint_source_captures=512,
    max_joint_hidden_captures=512, max_query_input_observations=1024,
    max_joint_hook_contexts=512, max_checkpoint_weight_transitions=1, final_model_state_sha256=FINAL_MODEL)


def _need(ok,reason):
    if not ok: raise RuntimeError('TA_TF_ACTIVITY_BOUNDARIES_'+reason)


def _input_refs():
    refs=initial._input_refs(); result=cache._read(refs['actual_result']); heads=cache._read(INITIAL_INPUT_RESULT)
    return dict(refs, final_checkpoint=result['final_state'],final_measurement=result['measurement_refs']['FINAL_TRAIN'],
        final_capture=result['panel_captures']['FINAL_TRAIN'],initial_inputs_result=INITIAL_INPUT_RESULT,
        initial_inputs_terminal=INITIAL_INPUT_TERMINAL,initial_inputs_plan=heads['plan'],
        initial_inputs_capture=heads['artifacts']['initial_readout_inputs'])


def _origin(owners,inputs,bound):
    _need(inputs == _input_refs(),'ACTUAL_INPUT_REFS')
    q=initial._origin(owners,{k:inputs[k] for k in initial._input_refs()},bound)
    for key,ref in inputs.items():
        campaign.require_binding(ref,label='paired boundary '+key,verify_file=True)
    for key,ref in HELPERS.items(): campaign.require_binding(ref,label='unchanged helper '+key,verify_file=True)
    expected=dict(final_checkpoint=initial.ACTUAL_ROOT/'RUNTIME/private-research/FINAL_C171_SHORT_STATE.pt',
        final_measurement=initial.ACTUAL_ROOT/'FINAL_TRAIN.json',final_capture=initial.ACTUAL_ROOT/'FINAL_TRAIN.npz',
        initial_inputs_capture=initial.ROOT/'TRAIN_FUSION.npz',initial_inputs_plan=initial.ROOT/'PLAN.json')
    _need(all(inputs[k]['path'] == str(path) for k,path in expected.items()),'ENDPOINT_PATHS')
    result,recipe=q['actual_result'],q['actual_recipe']
    terminal,measurement,heads,head_terminal,head_plan=(cache._read(inputs[k]) for k in
        ('actual_terminal','final_measurement','initial_inputs_result','initial_inputs_terminal','initial_inputs_plan'))
    _need(result['final_model_state_sha256'] == FINAL_MODEL and result['final_state_validated'] is True
        and terminal['final_state'] == inputs['final_checkpoint']
        and terminal['measurement_refs']['FINAL_TRAIN'] == inputs['final_measurement'],'ACTUAL_FINAL')
    m=measurement['measurement']
    _need(measurement['capture'] == inputs['final_capture'] and measurement['optimizer_updates'] == 256
        and measurement['evaluated_model_state_sha256'] == FINAL_MODEL and measurement['evaluation_state_unchanged'] is True
        and measurement['test_data_used'] is False and measurement['panel_batch_indices'] == BOUNDS['batch_indices']
        and (m['rows'],m['batches'],m['fold_id'],m['pass_index']) == (4096,256,1,0)
        and m['seed'] == recipe['numerical_recipe']['seed'] and m['feature_view_sha256'] == recipe['expected_view_sha256']
        and m['model_contract_sha256'] == recipe['expected_model_contract_sha256']
        and m['loss_contract_sha256'] == recipe['expected_loss_contract_sha256']
        and len(m['canonical_batch_losses']) == 256
        and all(type(v) is float and math.isfinite(v) for v in m['canonical_batch_losses']),'FINAL_MEASUREMENT')
    _need(heads['status'] == head_terminal['status'] == 'COMPLETE' and head_terminal['exit_code'] == 0
        and head_terminal['result'] == inputs['initial_inputs_result'] and head_terminal['plan'] == heads['plan']
        and head_plan['worker'] == HELPERS['initial_inputs'] and heads['source_unchanged'] is True
        and heads['all_initial_fields_and_loss_bitexact'] is True
        and heads['passivity'] == dict(signature_unchanged=True,no_parameter_gradients=True)
        and heads['candidate_model_state_sha256'] == short.MODEL_SHA256
        and heads['initial_checkpoint'] == inputs['initial_checkpoint']
        and heads['initial_measurement'] == inputs['initial_measurement']
        and heads['feature_view_sha256'] == recipe['expected_view_sha256']
        and heads['numerical_recipe_sha256'] == recipe['numerical_recipe']['contract_sha256']
        and heads['parent_order_sha256'] == recipe['parent_order_sha256']
        and heads['counts']['backbone_encodings'] == heads['counts']['matched_batches'] == 256
        and heads['counts']['sampled_rows'] == 4096 and heads['counts']['head_input_observations'] == 2560
        and heads['counts']['optimizer_updates'] == heads['counts']['autograd_grad_calls'] == 0,'QUALIFIED_INITIAL_INPUTS')
    return dict(q,final_measurement=measurement)


def build_plan(source_commit):
    inputs=_input_refs(); recipe=cache._read(inputs['actual_recipe'])
    owners={name:cache._ref(REPO/name) for name in recipe['source_owners']};q=_origin(owners,inputs,[])
    return dict(schema_version=SCHEMA,stage=STAGE,run_scope=SCOPE,source_repo=str(REPO),source_base_commit=source_commit,
        worker=cache._ref(WORKER),cap_runner=cache._ref(REPO/cache.CAP_RUNNER),source_owners=owners,
        helpers=HELPERS,inputs=inputs,feature_view_sha256=q['view']['contract_sha256'],
        numerical_recipe_sha256=recipe['numerical_recipe']['contract_sha256'],parent_order_sha256=recipe['parent_order_sha256'],**BOUNDS)


def _require_plan(plan,plan_path,out_dir):
    dynamic={'schema_version','stage','run_scope','source_repo','source_base_commit','worker','cap_runner',
        'source_owners','helpers','inputs','feature_view_sha256','numerical_recipe_sha256','parent_order_sha256'}
    _need(isinstance(plan,dict) and set(plan) == dynamic|set(BOUNDS) and plan['schema_version'] == SCHEMA
        and plan['stage'] == STAGE and plan['run_scope'] == SCOPE and plan['source_repo'] == str(REPO)
        and Path(plan_path) == ROOT/'PLAN.json' and Path(out_dir) == ROOT and plan['helpers'] == HELPERS
        and all(type(plan[k]) is type(v) and plan[k] == v for k,v in BOUNDS.items())
        and all(type(i) is int for i in plan['batch_indices']),'EXACT_PLAN')
    _need(type(plan['source_base_commit']) is str and campaign._COMMIT.fullmatch(plan['source_base_commit']),'SOURCE_BASE')
    return plan


def _inspect(plan_path,plan_sha,out_dir,*,execute):
    ref=dict(path=str(plan_path),sha256=plan_sha);plan=_require_plan(cache._read(ref),plan_path,out_dir)
    _need(not any(ROOT.glob('*.npz')) and not any((ROOT/name).exists() for name in
        ('CLAIM.json','RESULT.json','TERMINAL.json')),'CONSUMED')
    bound=[ref]
    for key,path in (('worker',WORKER),('cap_runner',REPO/cache.CAP_RUNNER)):
        _need(plan[key]['path'] == str(path),'SOURCE_PATH');campaign.require_binding(plan[key],label=key,verify_file=True);bound.append(plan[key])
    head=cache._git('rev-parse','HEAD')
    subprocess.run(['git','merge-base','--is-ancestor',plan['source_base_commit'],head],cwd=REPO,check=True)
    _need(plan == build_plan(plan['source_base_commit']),'CURRENT_PLAN_BINDINGS')
    bound.extend(plan['inputs'].values());bound.extend(plan['source_owners'].values());bound.extend(plan['helpers'].values())
    q=_origin(plan['source_owners'],plan['inputs'],bound)
    policy_ref=cache._ref(REPO/'NEXT_RUN_POLICY.json');policy=cache._read(policy_ref);bound.append(policy_ref)
    _need(all(policy.get(k) is False for k in ('training_enabled','native_training_enabled',
        'native_launch_authorized','full_epoch_training_allowed','full_val_allowed')),'NO_TRAINING')
    if execute:
        exact=dict(status='ADMITTED',plan=ref,max_invocations=1,remaining_invocations=1,claimed_invocations=0,
            completed_invocations=0,relaunch_allowed=False,execution_authorized=True,data_reads_authorized=True,
            checkpoint_reads_authorized=True,backbone_encodings_authorized=512,head_input_observations_authorized=5120,
            joint_source_captures_authorized=512,query_input_observations_authorized=1024,
            candidate_weights_writes_authorized=0,backward_calls_authorized=0,optimizer_steps_authorized=0,training_authorized=False)
        gate=policy.get(STAGE,{})
        _need(all(type(gate.get(k)) is type(v) and gate[k] == v for k,v in {**BOUNDS,**exact}.items()),'ADMISSION')
        _need(cache._git('branch','--show-current') == 'work/gx1-current','BRANCH')
        campaign.require_clean_source(dict(source_repo=str(REPO),source_commit=head))
    return dict(q,plan=plan,plan_ref=ref,head=head,bound_refs=bound)


def preflight(plan_path,plan_sha,out_dir):
    q=_inspect(plan_path,plan_sha,out_dir,execute=False)
    return dict(status='PASS_METADATA_ONLY_TF_ACTIVITY_BOUNDARIES_PREFLIGHT',stage=STAGE,plan=q['plan_ref'],
        initial_model_state_sha256=short.MODEL_SHA256,final_model_state_sha256=FINAL_MODEL,
        feature_view_sha256=q['view']['contract_sha256'],model_constructions=0,array_deserializations=0,
        checkpoint_loads=0,optimizer_updates=0,claims=0,actual_feature_normalization_identity_checked=True,
        training_enabled=False,test_data_used=False)


def _load_endpoint_weights(model,saved,*,phase,recipe,order,torch,old,state,model_sha):
    """Strict weight/buffer copy only; leave runtime modes, flags and RNG alone."""
    _need(phase in PHASES,'ENDPOINT_PHASE')
    final=phase == PHASES[1];expected=FINAL_MODEL if final else short.MODEL_SHA256
    _need(saved['schema_version'] == state.TA_STATE_SCHEMA and saved['source_commit'] == initial.ACTUAL_SOURCE
        and saved['model_state_sha256'] == model_sha(saved['model_state']) == expected
        and saved['buffer_state_sha256'] == old._tree_sha(saved['buffer_state'])
        and saved['feature_view_sha256'] == recipe['expected_view_sha256']
        and (saved['cursor'],saved['optimizer_updates'],saved['skipped_batches']) == ((4096,256,0) if final else (0,0,0))
        and torch.equal(saved['parent_order'],order) and saved['binding']['numerical_recipe'] == recipe['numerical_recipe']
        and model.state_model.feature_view_identity == saved['binding']['feature_view_identity'],'ENDPOINT_CHECKPOINT')
    _need(set(saved['module_training']) == set(dict(model.named_modules()))
        and all(type(v) is bool for v in saved['module_training'].values()),'ENDPOINT_MODES')
    pairs,active,inactive=state._parameters(model)
    _need([n for n,_ in active] == saved['binding']['active_parameter_names']
        and [n for n,_ in inactive] == saved['binding']['inactive_parameter_names']
        and all(p.requires_grad for _,p in active) and all(not p.requires_grad for _,p in inactive)
        and all(p.grad is None for _,p in pairs),'ENDPOINT_PARTITION')
    old._tensor_map(saved['model_state'],model.state_dict(),'ENDPOINT_MODEL')
    old._tensor_map(saved['buffer_state'],dict(model.named_buffers()),'ENDPOINT_BUFFER')
    model.load_state_dict(saved['model_state'],strict=True)
    _need(model_sha(model.state_dict()) == expected
        and old._tree_sha(dict(model.named_buffers())) == saved['buffer_state_sha256'],'ENDPOINT_LOADED_DIGEST')
    state._numerical(recipe['numerical_recipe'],model)
    return dict(model=expected,buffers=saved['buffer_state_sha256'])


def _require_transition(before,after,expected):
    _need(set(before) == set(after) and before['model'] == short.MODEL_SHA256
        and after['model'] == expected['model'] == FINAL_MODEL and after['buffers'] == expected['buffers']
        and all(after[k] == before[k] for k in before if k not in ('model','buffers')),'WEIGHTS_ONLY_TRANSITION')
    return dict(initial_model=before['model'],final_model=after['model'],initial_buffers=before['buffers'],
        final_buffers=after['buffers'],non_weight_state_unchanged=True,optimizer_or_rng_restored=False)


def _require_arrays(arrays,*,np,complete=False):
    _need(set(arrays) == set(initial.BASE_KEYS)|{'state_representation','future_representation','joint_source','joint_hidden','phase'},'ARRAY_FIELDS')
    n=len(arrays['parents']);_need(0 < n <= 8192 and n%16 == 0 and (not complete or n == 8192),'ARRAY_ROWS')
    phase=arrays['phase'];expected=np.r_[np.zeros(min(n,4096),np.uint8),np.ones(max(0,n-4096),np.uint8)]
    _need(phase.dtype == np.uint8 and phase.shape == (n,) and phase.tobytes() == expected.tobytes(),'PHASE_ORDER')
    for i in (0,1):
        chosen=phase == i
        if chosen.any(): initial._require_arrays({k:arrays[k][chosen] for k in
            (*initial.BASE_KEYS,'state_representation','future_representation')},int(chosen.sum()),np=np,representations=True)
    for name,width in (('joint_source',512),('joint_hidden',128)):
        value=arrays[name];_need(isinstance(value,np.ndarray) and value.dtype == np.float32
            and value.shape == (n,width) and bool(np.isfinite(value).all()),'BOUNDARY_ARRAY:'+name)
    _need(sum(a.nbytes for a in arrays.values()) <= BOUNDS['max_cache_bytes'],'ARRAY_BYTES')


def _require_parity(piece,baseline,loss,expected_loss,*,initial_heads,np):
    previous._require_parity({k:piece[k] for k in initial.BASE_KEYS},baseline,loss,expected_loss,np=np)
    if initial_heads is not None:
        _need(cache._fusion_exact_arrays({k:piece[k] for k in initial_heads},initial_heads),'INITIAL_ALL_TF_HEAD_INPUT_PARITY')


def _write_arrays(name,arrays,counts,*,np):
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    _need(name in ('TRAIN_FUSION.npz','TRAIN_FUSION_PARTIAL.npz','FAILED_BATCH_BOUNDARIES.npz') and arrays,'OUTPUT_NAME')
    buf=io.BytesIO();np.savez(buf,**arrays)
    _need(0 < buf.tell() and counts['published_npz_bytes']+buf.tell() <= BOUNDS['max_cache_bytes'],'OUTPUT_BYTES')
    _publish_bytes(ROOT/name,buf.getvalue());counts['published_npz_bytes']+=buf.tell();return cache._ref(ROOT/name)


def _save(parts,counts,*,partial,np):
    arrays={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};_require_arrays(arrays,np=np,complete=not partial)
    return _write_arrays('TRAIN_FUSION_PARTIAL.npz' if partial else 'TRAIN_FUSION.npz',arrays,counts,np=np)


def _require_counts(counts):
    exact=dict(model_constructions=1,model_construction_returns=1,checkpoint_loads=2,resume_state_load_calls=2,
        resume_state_load_returns=2,checkpoint_weight_transitions=1,backbone_encodings=512,forward_returns=512,
        head_hook_registrations=20,head_input_observations=5120,joint_hook_contexts=512,joint_source_captures=512,
        joint_hidden_captures=512,query_input_observations=1024,extra_target_gathers=1024,archive_decodes=3,
        sampled_rows=8192,matched_batches=512,autograd_grad_calls=0,backwards=0,optimizer_constructions=0,
        optimizer_updates=0,rng_restores=0,checkpoint_writes=0,candidate_writes=0)
    _need(all(type(counts.get(k)) is int and counts[k] == v for k,v in exact.items()),'COUNTS')


def run(plan_path,plan_sha,out_dir,device='cuda'):
    started=time.monotonic();_need(device == 'cuda','CUDA_ONLY')
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
    _need(not torch.cuda.is_initialized() and signal.getitimer(signal.ITIMER_REAL) == (0.,0.),'INIT_ORDER')
    claim=cache._publish('CLAIM.json',dict(plan=q['plan_ref'],source_commit=q['head'],cap=cap,max_invocations=1,
        no_relaunch_under_this_plan=True,test_data_used=False),root=ROOT)
    counts={k:0 for k in ('model_constructions','model_construction_returns','checkpoint_loads','resume_state_load_calls',
        'resume_state_load_returns','checkpoint_weight_transitions','backbone_encodings','forward_returns',
        'head_hook_registrations','head_input_observations','joint_hook_contexts','joint_source_captures',
        'joint_hidden_captures','query_input_observations','extra_target_gathers','archive_decodes','sampled_rows',
        'matched_batches','autograd_grad_calls','backwards','optimizer_constructions','optimizer_updates',
        'rng_restores','checkpoint_writes','candidate_writes','published_npz_bytes')}
    provider=prepared=model=None;parts=[];captures=[];boundary={};records=[];artifacts={};before={};passivity={};runtime={};transition={}
    active_phase=None;active_index=None;batch_axes=None;previous_alarm=signal.getsignal(signal.SIGALRM)
    def timeout(*unused): raise TimeoutError('Paired boundary observation900-second bound')
    def bookend():
        campaign.require_clean_source(dict(source_repo=str(REPO),source_commit=q['head']))
        for ref in q['bound_refs']:campaign.require_binding(ref,label='paired-boundary bookend',verify_file=True)
    def signature():
        return dict(model=model_sha(model.state_dict()),buffers=old._tree_sha(dict(model.named_buffers())),
            rng=old._tree_sha(trainer._attended_session_rng_state(device=cuda)),
            modes={n:m.training for n,m in model.named_modules()},flags={n:p.requires_grad for n,p in model.named_parameters()},
            parameters={n:id(p) for n,p in model.named_parameters()},buffer_ids={n:id(p) for n,p in model.named_buffers()},
            modules={n:id(m) for n,m in model.named_modules()},view=copy.deepcopy(model.state_model.feature_view_identity),
            hooks={n:{k:tuple((i,id(h)) for i,h in getattr(m,k).items()) for k in
                ('_forward_hooks','_forward_pre_hooks','_backward_hooks','_backward_pre_hooks')} for n,m in model.named_modules()},
            mha_fastpath=torch.backends.mha.get_fastpath_enabled(),runtime=previous._runtime_profile(torch))
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,max(1.,900-(time.monotonic()-started)))
    try:
        receipts,view,recipe=q['receipts'],q['view'],q['actual_recipe'];numerical=recipe['numerical_recipe'];cuda=torch.device('cuda')
        trainer._set_deterministic(numerical['seed'],cuda,numerical['precision_policy'])
        runtime=previous._runtime_profile(torch);previous._require_runtime_profile(runtime)
        _need(_BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False,'RUNTIME')
        baselines={}
        for key in ('initial_capture','final_capture','initial_inputs_capture'):
            with np.load(q['plan']['inputs'][key]['path'],allow_pickle=False) as z:baselines[key]={k:z[k].copy() for k in z.files}
            counts['archive_decodes']+=1
            initial._require_arrays(baselines[key],4096,np=np,representations=key == 'initial_inputs_capture')
        _need(cache._fusion_exact_arrays({k:baselines['initial_inputs_capture'][k] for k in initial.BASE_KEYS},
            baselines['initial_capture']),'INITIAL_BASELINE_CROSSCHECK')
        provider,provider_plan,prepared,rows,issues=runner._prepared_fold1(receipts,view,np=np)
        require_m5_equal_observation_axes(rows,issues);order=torch.from_numpy(rows.copy())
        _need(old._tree_sha(order) == recipe['parent_order_sha256'],'PARENT_ORDER')
        bound,future=load_bound_time_query_targets(),load_bound_future_displacement()
        counts['model_constructions']+=1
        model=NativeFutureDualModel(views.build_native_direction_research_model(view,
            prefix_normalization=receipts['fold1']['normal'],constructor_metadata=receipts['constructor'],
            device=torch.device('cpu'),unobserved_units_profile=receipts['fold1']['profile']))
        counts['model_construction_returns']+=1;previous._short_parameter_partition(model,state)
        for number,phase in enumerate(PHASES):
            active_phase=phase;active_index=None;captures=[];boundary={};batch_axes=None
            before_transition=signature() if number else None
            _need(counts['checkpoint_loads'] < 2,'CHECKPOINT_BOUND');counts['checkpoint_loads']+=1
            endpoint='final' if number else 'initial'
            saved=common._load_state(q['plan']['inputs'][endpoint+'_checkpoint'],torch=torch,
                maximum=recipe['max_checkpoint_bytes'],counters=counts)
            if number:counts['checkpoint_weight_transitions']+=1
            identity=_load_endpoint_weights(model,saved,phase=phase,recipe=recipe,order=order,
                torch=torch,old=old,state=state,model_sha=model_sha)
            del saved
            if number:transition=_require_transition(before_transition,signature(),identity)
            else:model.to(cuda);model.eval()
            before[phase]=signature()
            _need(before[phase]['model'] == identity['model'] and all(p.grad is None for p in model.parameters()),'LOADED_MODEL')
            baseline=baselines[endpoint+'_capture'];measurement=q[endpoint+'_measurement']
            with initial.capture_activity_inputs(model,counts=counts,torch=torch) as captures:
                for index in BOUNDS['batch_indices']:
                    active_index=index;left=index*16;selected=slice(left,left+16);boundary={}
                    _need(counts['backbone_encodings'] < 512 and time.monotonic()-started < 900,'FORWARD_BOUND')
                    raw,parents,clocks=runner._feature_batch(prepared,view,receipts,rows,issues,left,left+16,np=np)
                    batch_axes=dict(parents=parents.copy(),issues=clocks.copy())
                    features={k:torch.from_numpy(v).to(cuda) for k,v in raw.items()}
                    counts['joint_hook_contexts']+=1
                    with capture_joint_boundaries(model,counts=counts,torch=torch) as boundary:
                        with torch.no_grad(),trainer._training_mha_fastpath_context(model.backbone),trainer._training_autocast_context(),trainer._training_model_finite_check_context():
                            _need(not torch.is_grad_enabled() and not model.training and not model.backbone.training
                                and torch.backends.mha.get_fastpath_enabled() is False
                                and _BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False,'FORWARD_PROFILE')
                            counts['backbone_encodings']+=1
                            report=evaluate_fit_batch(model,features,bound,future,parents,clocks,fold_id=1,pass_index=0,
                                seed=numerical['seed'],future_weight=1.,m5_equal_observation=True,fit_batch_start=left)
                            counts['forward_returns']+=1
                    _need(report['skip_update'] is False and not report['loss'].requires_grad
                        and report['model_contract_sha256'] == recipe['expected_model_contract_sha256']
                        and report['loss_contract_sha256'] == recipe['expected_loss_contract_sha256'],'REPORT_BINDING')
                    truths=dict(state=bound.gather_fit_sampled(parents,clocks,fold_id=1,pass_index=0,seed=numerical['seed']),
                        future=future.gather(parents,clocks,fold_id=1,partition='fit'));counts['extra_target_gathers']+=2
                    piece=dict(batch_axes,state_query=report['state_query_k'].copy(),
                        future_query=np.broadcast_to(report['future_native_k'],(16,5)).copy(),**boundary,phase=np.full(16,number,np.uint8))
                    _need(len(captures) == index+1 and tuple(captures[index]) == initial.HEAD_KEYS,'BATCH_HEAD_CAPTURE')
                    for task in ('state','future'):
                        piece[task+'_raw']=report[task+'_activity_side_logits'].detach().cpu().numpy().copy()
                        _need(np.array_equal(truths[task]['parent_rows'],parents) and np.array_equal(truths[task]['issue_ns'],clocks)
                            and np.array_equal(truths[task]['allowed'],report[task+'_allowed'])
                            and np.array_equal(truths[task]['mature'],report[task+'_mature']),'ACTUAL_TARGET_AXES')
                        for name in ('allowed','eligible','mature'):piece[task+'_'+name]=truths[task][name].copy()
                        piece[task+'_representation']=np.stack([captures[index][task+'_'+tf] for tf in initial.TIMEFRAMES],axis=1)
                    # The per-phase base validator avoids resetting the paired phase axis for FINAL batches.
                    initial._require_arrays({k:piece[k] for k in (*initial.BASE_KEYS,'state_representation','future_representation')},16,np=np,representations=True)
                    loss=float(report['loss']);parts.append(piece);counts['sampled_rows']+=16
                    _require_parity(piece,{k:v[selected] for k,v in baseline.items()},loss,
                        measurement['measurement']['canonical_batch_losses'][index],
                        initial_heads=None if number else {k:v[selected] for k,v in baselines['initial_inputs_capture'].items()},np=np)
                    counts['matched_batches']+=1;records.append(dict(phase=phase,batch_index=index,rows=16,loss=loss,
                        all12fields_and_loss_bitexact=True,all_initial_head_inputs_bitexact=True if not number else None))
                    _need(all(p.grad is None for p in model.parameters()),'NO_GRADIENTS');del report,features,raw
            passivity[phase]=dict(signature_unchanged=signature() == before[phase],no_parameter_gradients=all(p.grad is None for p in model.parameters()))
            _need(all(passivity[phase].values()),'PHASE_PASSIVITY')
        _require_counts(counts)
        artifacts['paired_boundaries']=_save(parts,counts,partial=False,np=np)
        release=prepared;prepared=None;base._release_pages(provider,release);provider.validate_metadata(provider_plan);bookend()
        return cache._publish_fusion_outcome(q['plan_ref'],dict(claim=claim,counts=counts,artifacts=artifacts,
            source_commit=q['head'],source_unchanged=True,candidate_model_state_sha256=short.MODEL_SHA256,
            final_model_state_sha256=FINAL_MODEL,feature_view_sha256=view['contract_sha256'],
            numerical_recipe_sha256=numerical['contract_sha256'],parent_order_sha256=recipe['parent_order_sha256'],
            phase_names=list(PHASES),batch_indices=BOUNDS['batch_indices'],batch_records=records,passivity=passivity,
            phase_states={phase:{k:v[k] for k in ('model','buffers','rng','modes','flags','view')} for phase,v in before.items()},
            checkpoint_transition=transition,numeric_runtime=runtime,source_checkpoint_weights_only=True,
            optimizer_or_rng_restored=False,all_endpoint_fields_and_loss_bitexact=True,all_initial_head_inputs_bitexact=True,
            original_initial_measurement=q['plan']['inputs']['initial_measurement'],original_final_measurement=q['plan']['inputs']['final_measurement'],
            elapsed_seconds=time.monotonic()-started,limitations=[
                'Two reproduced eval endpoints on the same4096dependent historically usedTRAIN rows; no stored historical hidden identity claim.',
                'One explicit weight/buffer transition; per-phase modes/flags/RNG/hooks unchanged, no optimizer/RNG restoration or fitting.',
                'Boundary magnitudes do not establish family utility, historical task attribution, generalization or economic edge.']),
            complete=True,root=ROOT,json_bytes_bound=BOUNDS['max_published_json_bytes'])
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL,0);changed=save_error=passivity_error=None
        try:
            if active_phase in before:
                passivity[active_phase]=dict(signature_unchanged=signature() == before[active_phase],no_parameter_gradients=all(p.grad is None for p in model.parameters()))
        except Exception as other:passivity_error=repr(other)
        try:
            if parts and not artifacts:artifacts['partial_paired_boundaries']=_save(parts,counts,partial=True,np=np)
            partial=dict(batch_axes or {},**boundary)
            if active_index is not None and active_index < len(captures):partial.update(captures[active_index])
            if partial:artifacts['failed_batch_boundaries']=_write_arrays('FAILED_BATCH_BOUNDARIES.npz',partial,counts,np=np)
        except Exception as other:save_error=repr(other)
        try:bookend()
        except Exception as other:changed=repr(other)
        result=cache._publish_fusion_outcome(q['plan_ref'],dict(claim=claim,counts=counts,artifacts=artifacts,
            source_commit=q['head'],source_unchanged=changed is None,active_phase=active_phase,active_batch_index=active_index,
            batch_records=records,passivity=passivity,passivity_error=passivity_error,checkpoint_transition=transition,
            numeric_runtime=runtime,error=type(exc).__name__+':'+str(exc),bookend_error=changed,
            partial_save_error=save_error,interpretation_authorized=False),complete=False,root=ROOT,
            json_bytes_bound=BOUNDS['max_published_json_bytes'])
        raise RuntimeError('TA_TF_ACTIVITY_BOUNDARIES_FAILED_CONSUMED:'+str(result)) from exc
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous_alarm)
        if prepared is not None:base._release_pages(provider,prepared)


def main(argv=None):
    import argparse,json,sys
    supplied=list(sys.argv[1:] if argv is None else argv)
    flags=('--market-state-ta-tf-activity-boundaries-plan','--market-state-ta-tf-activity-boundaries-plan-sha256')
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False,add_help=False)
    parser.add_argument(flags[0],type=Path,required=True);parser.add_argument(flags[1],required=True)
    parser.add_argument('--device',choices=('cuda',),required=True);parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    for flag in (*flags,'--device','--out-dir','--preflight'):
        if sum(v.partition('=')[0] == flag for v in supplied) > 1:parser.error('duplicate flag: '+flag)
    args=parser.parse_args(supplied)
    if not args.out_dir.is_absolute():parser.error('--out-dir must be absolute')
    kwargs=dict(plan_path=args.market_state_ta_tf_activity_boundaries_plan,
        plan_sha=args.market_state_ta_tf_activity_boundaries_plan_sha256,out_dir=args.out_dir)
    print(json.dumps(preflight(**kwargs) if args.preflight else run(**kwargs,device=args.device),indent=2,sort_keys=True,allow_nan=False))
    return 0


if __name__ == '__main__':raise SystemExit(main())
