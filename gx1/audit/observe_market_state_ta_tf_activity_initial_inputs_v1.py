"""One passive full-prefix INITIAL readout-input capture; no learning authority."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import io
import math
from pathlib import Path
import signal
import subprocess
import time

from gx1.audit import cache_market_state_ta_direction_v1 as cache
from gx1.audit import observe_market_state_ta_c171_short_heads_v1 as previous
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts import market_state_ta_tf_activity_short_learning_v1 as short

REPO = cache.REPO
STAGE = 'market_state_ta_tf_activity_initial_inputs_20261112_001'
SCOPE = 'market_state_ta_tf_activity_initial_inputs_v1'
ROOT = Path('/home/andre2/GX1_RUNS') / STAGE.upper()
SCHEMA = 'gx1_ta_tf_activity_initial_inputs_plan_v1'
WORKER = REPO / 'gx1/audit/observe_market_state_ta_tf_activity_initial_inputs_v1.py'
ACTUAL_ROOT = short.ROOT
ACTUAL_SOURCE = '7dcddb37586cdcf297ffac2e8b1f643912f4c50a'
RESULT_REF = dict(path=str(ACTUAL_ROOT/'RESULT.json'), sha256='aca65dc41d267a47528c38c036f777a084ed7255b87bf0a6305fa70db18d9901')
TERMINAL_REF = dict(path=str(ACTUAL_ROOT/'TERMINAL.json'), sha256='039d6a5b1acf5281a6a93f21782ca124000f3213e722480c7377ab10e9e1b60c')
PEER_REF = dict(path=str(ACTUAL_ROOT/'INDEPENDENT_ACTIVITY_LEARNING_REVIEW.json'), sha256='f79b828f10cac0198da8f6d9b70218e062e8b3da57498fd18b4f0ef92a9ac70b')
TIMEFRAMES = ('M5', 'M15', 'H1', 'H4', 'D1')
HEAD_KEYS = tuple(task+'_'+tf for task in ('state', 'future') for tf in TIMEFRAMES)
BASE_KEYS = previous.BASE_KEYS
BOUNDS = dict(fit_rows=29574, sampled_rows=4096, batch_size=16, batch_indices=list(range(256)),
    max_model_constructions=1, max_checkpoint_loads=1, max_backbone_encodings=256,
    max_extra_target_gathers=512, registered_head_hooks=10, max_head_input_observations=2560,
    max_optimizer_constructions=0, max_optimizer_updates=0, max_backwards=0,
    max_autograd_grad_calls=0, max_rng_restores=0, max_checkpoint_writes=0,
    max_candidate_writes=0, max_cache_bytes=64*1024**2, max_published_json_bytes=2*1024**2,
    wall_seconds=900, device='cuda', initial_model_state_sha256=short.MODEL_SHA256,
    numerical_mode='original_eval_no_grad_deterministic_fp32_mha_fastpath_false',
    numerical_threads=8, interop_policy='process_default_unchanged_like_short002',
    thread_environment=dict(previous.BOUNDS['thread_environment']),
    parameter_partition='original_fold1_fit_active_true_inactive_false',
    training_enabled=False, test_data_used=False)


def _need(ok, reason):
    if not ok:
        raise RuntimeError('TA_TF_ACTIVITY_INITIAL_INPUTS_'+reason)


def _input_refs():
    result = cache._read(RESULT_REF)
    return dict(actual_result=RESULT_REF, actual_terminal=TERMINAL_REF, retained_review=PEER_REF,
        actual_recipe=result['recipe'], initial_checkpoint=result['initial_state'],
        initial_measurement=result['measurement_refs']['INITIAL_TRAIN'],
        initial_capture=result['panel_captures']['INITIAL_TRAIN'])


def _metadata(inputs):
    _need(inputs == _input_refs(), 'ACTUAL_INPUT_REFS')
    expected_paths = dict(actual_recipe=ACTUAL_ROOT/'RECIPE.json',
        initial_checkpoint=ACTUAL_ROOT/'RUNTIME/private-research/INITIAL_C171_STATE.pt',
        initial_measurement=ACTUAL_ROOT/'INITIAL_TRAIN.json', initial_capture=ACTUAL_ROOT/'INITIAL_TRAIN.npz')
    for key, ref in inputs.items():
        campaign.require_binding(ref, label='initial inputs '+key, verify_file=True)
        if key in expected_paths:
            _need(ref['path'] == str(expected_paths[key]), 'ACTUAL_INPUT_PATH:'+key)
    result, terminal, recipe, measurement, peer = (cache._read(inputs[k]) for k in
        ('actual_result', 'actual_terminal', 'actual_recipe', 'initial_measurement', 'retained_review'))
    _need(result['outcome'] == terminal['status'] == 'COMPLETE' and terminal['exit_code'] == 0
        and terminal['result'] == inputs['actual_result'] and terminal['recipe'] == inputs['actual_recipe']
        and terminal['initial_state'] == inputs['initial_checkpoint']
        and terminal['measurement_refs']['INITIAL_TRAIN'] == inputs['initial_measurement']
        and result['source_commit'] == ACTUAL_SOURCE and result['source_unchanged'] is True
        and terminal['source_unchanged'] is True and result['initial_restore_exact'] is True
        and result['evaluation_state_unchanged'] is True and result['all_panels_complete'] is True
        and result['dedicated_tf_activity'] is True and result['original_training_rng_endpoints_matched'] is True
        and (result['optimizer_updates'], result['parent_cursor'], result['skipped_batches']) == (256,4096,0)
        and result['initial_model_state_sha256'] == short.MODEL_SHA256
        and result['panel_batch_indices'] == BOUNDS['batch_indices'] and result['test_data_used'] is False,
        'COMPLETED_ACTUAL_ORIGIN')
    _need(recipe['run_scope'] == short.SCOPE and recipe['admission_stage'] == short.STAGE
        and recipe['recipe_sha256'] == campaign.canonical_sha256({k:v for k,v in recipe.items() if k!='recipe_sha256'})
        and recipe['dedicated_tf_activity'] is True and recipe.get('dedicated_non_m5_side',False) is False
        and recipe['expected_initial_model_state_sha256'] == short.MODEL_SHA256
        and recipe['candidate_inputs'] == short.INPUTS
        and recipe['parent_order_sha256'] == result['parent_order_sha256'], 'ACTUAL_RECIPE')
    m = measurement['measurement']
    _need(measurement['capture'] == inputs['initial_capture'] and measurement['optimizer_updates'] == 0
        and measurement['evaluated_model_state_sha256'] == short.MODEL_SHA256
        and measurement['evaluation_state_unchanged'] is True and measurement['test_data_used'] is False
        and measurement['panel_batch_indices'] == BOUNDS['batch_indices']
        and m['rows'] == 4096 and m['batches'] == 256 and m['fold_id'] == 1 and m['pass_index'] == 0
        and m['seed'] == recipe['numerical_recipe']['seed']
        and m['feature_view_sha256'] == recipe['expected_view_sha256']
        and m['model_contract_sha256'] == recipe['expected_model_contract_sha256']
        and m['loss_contract_sha256'] == recipe['expected_loss_contract_sha256']
        and len(m['canonical_batch_losses']) == 256
        and all(type(v) is float and math.isfinite(v) for v in m['canonical_batch_losses']), 'INITIAL_MEASUREMENT')
    _need(peer['status'] == 'PASS_INDEPENDENT_RETAINED_ACTIVITY_LEARNING'
        and peer['source_unchanged'] is True and peer['test_data_used'] is False
        and peer['inputs']['activity_result'] == inputs['actual_result']
        and peer['inputs']['activity_INITIAL_TRAIN_capture'] == inputs['initial_capture']
        and peer['inputs']['activity_INITIAL_TRAIN_measurement'] == inputs['initial_measurement']
        and peer['native_evidence']['activity']['initial_model'] == short.MODEL_SHA256
        and peer['parity']['activity']['all_axes_targets_byteexact'] is True, 'RETAINED_EVIDENCE')
    return dict(actual_result=result, actual_recipe=recipe, initial_measurement=measurement)


def _require_sources(owners, recipe):
    historical = recipe['source_owners']
    _need(set(owners) == set(historical) and len(owners) == 92, 'OWNER_SET')
    for name, old in historical.items():
        _need(old['path'] == owners[name]['path'] == str(REPO/name), 'OWNER_PATH:'+name)
        if not name.startswith('.venv/'):
            raw = subprocess.check_output(['git','show',ACTUAL_SOURCE+':'+name], cwd=REPO)
            _need(hashlib.sha256(raw).hexdigest() == old['sha256'], 'ACTUAL_SOURCE_SNAPSHOT:'+name)
        campaign.require_binding(owners[name], label='initial-input owner '+name, verify_file=True)
        # Only the dedicated cap selector may differ from the completed trial.
        _need(name == cache.CAP_RUNNER or owners[name] == old, 'ACTUAL_NUMERICAL_SOURCE_CHANGED:'+name)


def _origin(owners, inputs, bound):
    from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero
    from gx1.contracts.market_state_time_query_fold1_fit_v1 import ta_sf_numerical_recipe
    meta = _metadata(inputs)
    _require_sources(owners, meta['actual_recipe'])
    context = dict(inputs=short.INPUTS, tf_activity_learning_scope=short.SCOPE,
        tf_activity_learning_source_owners=owners)
    # Metadata/input qualification only. No old scope gate, claim or runner is used.
    q = cache._qualify_reversed_origin(context, bound)
    view = zero._bind_ta_view(q['receipts'], dedicated_m5_side=True, dedicated_tf_activity=True)
    short.initial._require_trial_view(q['view'], view)
    recipe = meta['actual_recipe']
    _need(view['contract_sha256'] == recipe['expected_view_sha256']
        and recipe['numerical_recipe'] == ta_sf_numerical_recipe(dedicated_m5_side=True,
            m5_equal_observation=True, dedicated_tf_activity=True), 'ACTUAL_VIEW_NUMERICAL_IDENTITY')
    return dict(q, **meta, view=view)


def build_plan(source_commit):
    inputs = _input_refs(); recipe = cache._read(inputs['actual_recipe'])
    owners = {name:cache._ref(REPO/name) for name in recipe['source_owners']}
    q = _origin(owners, inputs, [])
    return dict(schema_version=SCHEMA, stage=STAGE, run_scope=SCOPE, source_repo=str(REPO),
        source_base_commit=source_commit, worker=cache._ref(WORKER), cap_runner=cache._ref(REPO/cache.CAP_RUNNER),
        source_owners=owners, inputs=inputs, feature_view_sha256=q['view']['contract_sha256'],
        numerical_recipe_sha256=recipe['numerical_recipe']['contract_sha256'],
        parent_order_sha256=recipe['parent_order_sha256'], **BOUNDS)


def _require_plan(plan, plan_path, out_dir):
    dynamic = {'schema_version','stage','run_scope','source_repo','source_base_commit','worker',
        'cap_runner','source_owners','inputs','feature_view_sha256','numerical_recipe_sha256','parent_order_sha256'}
    _need(isinstance(plan,dict) and set(plan) == dynamic|set(BOUNDS)
        and plan['schema_version'] == SCHEMA and plan['stage'] == STAGE and plan['run_scope'] == SCOPE
        and plan['source_repo'] == str(REPO) and Path(plan_path) == ROOT/'PLAN.json' and Path(out_dir) == ROOT
        and all(type(plan[k]) is type(v) and plan[k] == v for k,v in BOUNDS.items())
        and all(type(i) is int for i in plan['batch_indices']), 'EXACT_PLAN')
    _need(type(plan['source_base_commit']) is str and campaign._COMMIT.fullmatch(plan['source_base_commit']), 'SOURCE_BASE')
    return plan


def _inspect(plan_path, plan_sha, out_dir, *, execute):
    ref = dict(path=str(plan_path),sha256=plan_sha)
    plan = _require_plan(cache._read(ref), plan_path, out_dir)
    _need(not any(ROOT.glob('*.npz')) and not any((ROOT/name).exists() for name in
        ('CLAIM.json','RESULT.json','TERMINAL.json')), 'CONSUMED')
    bound = [ref]
    for key,path in (('worker',WORKER),('cap_runner',REPO/cache.CAP_RUNNER)):
        _need(plan[key]['path'] == str(path), 'SOURCE_PATH')
        campaign.require_binding(plan[key],label=key,verify_file=True); bound.append(plan[key])
    head = cache._git('rev-parse','HEAD')
    subprocess.run(['git','merge-base','--is-ancestor',plan['source_base_commit'],head],cwd=REPO,check=True)
    _need(plan == build_plan(plan['source_base_commit']), 'CURRENT_PLAN_BINDINGS')
    bound.extend(plan['inputs'].values()); bound.extend(plan['source_owners'].values())
    q = _origin(plan['source_owners'], plan['inputs'], bound)
    policy_ref = cache._ref(REPO/'NEXT_RUN_POLICY.json'); policy = cache._read(policy_ref); bound.append(policy_ref)
    _need(all(policy.get(k) is False for k in ('training_enabled','native_training_enabled',
        'native_launch_authorized','full_epoch_training_allowed','full_val_allowed')), 'NO_TRAINING')
    if execute:
        exact = dict(status='ADMITTED',plan=ref,max_invocations=1,remaining_invocations=1,claimed_invocations=0,
            completed_invocations=0,relaunch_allowed=False,execution_authorized=True,data_reads_authorized=True,
            checkpoint_reads_authorized=True,backbone_encodings_authorized=256,
            head_input_observations_authorized=2560,candidate_weights_writes_authorized=0,
            backward_calls_authorized=0,optimizer_steps_authorized=0,training_authorized=False)
        gate = policy.get(STAGE,{})
        _need(all(type(gate.get(k)) is type(v) and gate[k] == v for k,v in {**BOUNDS,**exact}.items()), 'ADMISSION')
        _need(cache._git('branch','--show-current') == 'work/gx1-current', 'BRANCH')
        campaign.require_clean_source(dict(source_repo=str(REPO),source_commit=head))
    return dict(q, plan=plan, plan_ref=ref, head=head, bound_refs=bound)


def preflight(plan_path, plan_sha, out_dir):
    q = _inspect(plan_path,plan_sha,out_dir,execute=False)
    return dict(status='PASS_METADATA_ONLY_TF_ACTIVITY_INITIAL_INPUTS_PREFLIGHT',stage=STAGE,plan=q['plan_ref'],
        candidate_model_state_sha256=short.MODEL_SHA256,feature_view_sha256=q['view']['contract_sha256'],
        model_constructions=0,array_deserializations=0,checkpoint_loads=0,optimizer_updates=0,claims=0,
        actual_feature_normalization_identity_checked=True,training_enabled=False,test_data_used=False)


@contextmanager
def capture_activity_inputs(model, *, counts, torch, batches=256):
    """Ten passive pre-hooks, ordered S then F; owned FP32 copies survive failure."""
    _need(type(batches) is int and 0 < batches <= 256, 'CAPTURE_BATCH_BOUND')
    state = getattr(model,'state_model',None)
    groups = (getattr(state,'tf_activity_heads',None), getattr(model,'future_tf_activity_heads',None))
    _need(getattr(state,'_dedicated_tf_activity',False) is True
        and getattr(state,'_dedicated_m5_side',False) is True
        and getattr(state,'_dedicated_non_m5_side',True) is False
        and all(isinstance(g,torch.nn.ModuleDict) and tuple(g) == TIMEFRAMES for g in groups), 'HEAD_PROFILE')
    modules = [g[tf] for g in groups for tf in TIMEFRAMES]
    _need(len({id(m) for m in modules}) == 10 and all(type(m) is torch.nn.Linear
        and (m.in_features,m.out_features) == (128,1) and m.bias is not None for m in modules), 'HEAD_LAYOUT')
    _need(len({p.data_ptr() for m in modules for p in m.parameters()}) == 20, 'HEAD_STORAGE')
    for key in ('head_input_observations','head_hook_registrations'):
        _need(type(counts.get(key,0)) is int and counts.get(key,0) >= 0, 'HOOK_COUNTS')
        counts.setdefault(key,0)
    captured=[]; handles=[]; observed=0; failure=None
    def observer(index):
        def hook(module,args):
            nonlocal observed,failure
            call = observed; observed += 1; counts['head_input_observations'] += 1
            try:
                _need(call < batches*10 and call%10 == index, 'HEAD_CALL_ORDER')
                if index == 0: captured.append({})
                _need(isinstance(args,tuple) and len(args) == 1, 'HEAD_ARGS')
                value = args[0]; shape = (16,1,128) if index < 5 else (16,128)
                _need(isinstance(value,torch.Tensor) and tuple(value.shape) == shape
                    and value.dtype == torch.float32 and value.device == module.weight.device, 'HEAD_INPUT:'+HEAD_KEYS[index])
                _need(bool(torch.isfinite(value.detach()).all().item()), 'HEAD_NONFINITE:'+HEAD_KEYS[index])
                copied = value.detach().cpu().numpy().reshape(16,128).copy()
                captured[-1][HEAD_KEYS[index]] = copied
            except BaseException as exc:
                failure = exc
                raise
            # No input replacement, mode change, RNG use or graph participation.
            return None
        return hook
    try:
        for i,module in enumerate(modules):
            handles.append(module.register_forward_pre_hook(observer(i)))
            counts['head_hook_registrations'] += 1
        yield captured
        _need(failure is None and observed == batches*10 and len(captured) == batches
            and all(tuple(p) == HEAD_KEYS for p in captured), 'INCOMPLETE_HEAD_CAPTURE')
    finally:
        for handle in handles: handle.remove()


def _require_arrays(arrays, rows, *, np, representations=False):
    _need(type(rows) is int and 0 < rows <= 4096 and rows%16 == 0, 'ARRAY_ROWS')
    layout = previous._layout(rows,np=np,prefix=True); layout.pop('phase')
    if representations:
        layout.update({task+'_representation':((rows,5,128),np.float32) for task in ('state','future')})
    _need(set(arrays) == set(layout), 'ARRAY_FIELDS')
    for key,(shape,dtype) in layout.items():
        a = arrays[key]
        _need(isinstance(a,np.ndarray) and a.shape == shape and a.dtype == dtype
            and bool(np.isfinite(a).all()), 'ARRAY:'+key)
    _need(sum(a.nbytes for a in arrays.values()) <= BOUNDS['max_cache_bytes'], 'ARRAY_BYTES')
    _need(bool(np.all(np.diff(arrays['parents']) > 0)) and bool(np.all(np.diff(arrays['issues']) > 0)), 'ARRAY_CHRONOLOGY')


def _write_arrays(name, arrays, counts, *, np):
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    _need(name in ('TRAIN_FUSION.npz','TRAIN_FUSION_PARTIAL.npz','FAILED_BATCH_INPUTS.npz') and arrays, 'OUTPUT_NAME')
    buf=io.BytesIO(); np.savez(buf,**arrays)
    _need(0 < buf.tell() and counts['published_npz_bytes']+buf.tell() <= BOUNDS['max_cache_bytes'], 'OUTPUT_BYTES')
    _publish_bytes(ROOT/name,buf.getvalue()); counts['published_npz_bytes'] += buf.tell()
    return cache._ref(ROOT/name)


def _save(parts, counts, *, partial, np):
    arrays = {key:np.concatenate([p[key] for p in parts]) for key in parts[0]}
    rows = len(arrays['parents']); _require_arrays(arrays,rows,np=np,representations=True)
    _need(partial or rows == 4096, 'FULL_CAPTURE')
    return _write_arrays('TRAIN_FUSION_PARTIAL.npz' if partial else 'TRAIN_FUSION.npz',arrays,counts,np=np)


def run(plan_path, plan_sha, out_dir, device='cuda'):
    started=time.monotonic(); _need(device == 'cuda','CUDA_ONLY')
    q=_inspect(plan_path,plan_sha,out_dir,execute=True); cap=previous._runtime_cap()
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
    counts=dict(model_constructions=0,model_construction_returns=0,checkpoint_loads=0,backbone_encodings=0,
        forward_returns=0,archive_decodes=0,head_hook_registrations=0,head_input_observations=0,
        extra_target_gathers=0,sampled_rows=0,matched_batches=0,resume_state_load_calls=0,resume_state_load_returns=0,
        autograd_grad_calls=0,backwards=0,optimizer_constructions=0,optimizer_updates=0,rng_restores=0,
        checkpoint_writes=0,candidate_writes=0,published_npz_bytes=0)
    provider=prepared=model=None; parts=[]; captures=[]; records=[]; artifacts={}; before=None; passivity={}; runtime={}
    active_index=None; batch_axes=None; previous_alarm=signal.getsignal(signal.SIGALRM)
    def timeout(*unused): raise TimeoutError('Initial readout-input observation900-second bound')
    def bookend():
        campaign.require_clean_source(dict(source_repo=str(REPO),source_commit=q['head']))
        for ref in q['bound_refs']: campaign.require_binding(ref,label='initial-input bookend',verify_file=True)
    def signature():
        return dict(model=model_sha(model.state_dict()),buffers=old._tree_sha(dict(model.named_buffers())),
            rng=old._tree_sha(trainer._attended_session_rng_state(device=cuda)),
            modes={n:m.training for n,m in model.named_modules()},flags={n:p.requires_grad for n,p in model.named_parameters()},
            parameters={n:id(p) for n,p in model.named_parameters()},buffer_ids={n:id(p) for n,p in model.named_buffers()},
            modules={n:id(m) for n,m in model.named_modules()},view=copy.deepcopy(model.state_model.feature_view_identity),
            hooks={n:{k:tuple((i,id(h)) for i,h in getattr(m,k).items()) for k in
                ('_forward_hooks','_forward_pre_hooks','_backward_hooks','_backward_pre_hooks')} for n,m in model.named_modules()},
            mha_fastpath=torch.backends.mha.get_fastpath_enabled(),runtime=previous._runtime_profile(torch))
    signal.signal(signal.SIGALRM,timeout); signal.setitimer(signal.ITIMER_REAL,max(1.,900-(time.monotonic()-started)))
    try:
        receipts,view,recipe=q['receipts'],q['view'],q['actual_recipe']; numerical=recipe['numerical_recipe']; cuda=torch.device('cuda')
        trainer._set_deterministic(numerical['seed'],cuda,numerical['precision_policy'])
        runtime=previous._runtime_profile(torch); previous._require_runtime_profile(runtime)
        _need(_BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False,'RUNTIME')
        with np.load(q['plan']['inputs']['initial_capture']['path'],allow_pickle=False) as z:
            baseline={k:z[k].copy() for k in z.files}
        counts['archive_decodes']+=1; _require_arrays(baseline,4096,np=np)
        provider,provider_plan,prepared,rows,issues=runner._prepared_fold1(receipts,view,np=np)
        require_m5_equal_observation_axes(rows,issues)
        order=torch.from_numpy(rows.copy()); _need(old._tree_sha(order) == recipe['parent_order_sha256'],'PARENT_ORDER')
        bound,future=load_bound_time_query_targets(),load_bound_future_displacement()
        counts['model_constructions']+=1
        model=NativeFutureDualModel(views.build_native_direction_research_model(view,
            prefix_normalization=receipts['fold1']['normal'],constructor_metadata=receipts['constructor'],
            device=torch.device('cpu'),unobserved_units_profile=receipts['fold1']['profile']))
        counts['model_construction_returns']+=1; counts['checkpoint_loads']+=1
        saved=common._load_state(q['plan']['inputs']['initial_checkpoint'],torch=torch,
            maximum=recipe['max_checkpoint_bytes'],counters=counts)
        _need(saved['schema_version'] == state.TA_STATE_SCHEMA and saved['source_commit'] == ACTUAL_SOURCE
            and saved['model_state_sha256'] == model_sha(saved['model_state']) == short.MODEL_SHA256
            and saved['buffer_state_sha256'] == old._tree_sha(saved['buffer_state'])
            and saved['feature_view_sha256'] == view['contract_sha256']
            and (saved['cursor'],saved['optimizer_updates'],saved['skipped_batches']) == (0,0,0)
            and torch.equal(saved['parent_order'],order)
            and saved['binding']['numerical_recipe'] == numerical,'ACTUAL_INITIAL_CHECKPOINT')
        model.load_state_dict(saved['model_state'],strict=True)
        _need(old._tree_sha(dict(model.named_buffers())) == saved['buffer_state_sha256'],'ACTUAL_BUFFERS')
        modes=saved['module_training']; _need(set(modes) == set(dict(model.named_modules()))
            and all(type(v) is bool for v in modes.values()),'ACTUAL_MODES')
        for name,module in model.named_modules(): module.training=modes[name]
        flags=previous._short_parameter_partition(model,state)
        _need([n for n,v in flags.items() if v] == saved['binding']['active_parameter_names']
            and set(n for n,v in flags.items() if not v) == set(saved['binding']['inactive_parameter_names'])
            and model.state_model.feature_view_identity == saved['binding']['feature_view_identity'], 'ACTUAL_BINDING')
        state._numerical(numerical,model)
        del saved,order
        model.to(cuda); model.eval(); before=signature()
        _need(before['model'] == short.MODEL_SHA256 and all(p.grad is None for p in model.parameters()),'LOADED_MODEL')
        with capture_activity_inputs(model,counts=counts,torch=torch) as captures:
            for index in BOUNDS['batch_indices']:
                active_index=index; left=index*16; selected=slice(left,left+16)
                _need(counts['backbone_encodings'] < 256 and time.monotonic()-started < 900,'FORWARD_BOUND')
                raw,parents,clocks=runner._feature_batch(prepared,view,receipts,rows,issues,left,left+16,np=np)
                batch_axes=dict(parents=parents.copy(),issues=clocks.copy())
                features={k:torch.from_numpy(v).to(cuda) for k,v in raw.items()}
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
                    future=future.gather(parents,clocks,fold_id=1,partition='fit')); counts['extra_target_gathers']+=2
                piece=dict(batch_axes,state_query=report['state_query_k'].copy(),
                    future_query=np.broadcast_to(report['future_native_k'],(16,5)).copy())
                _need(len(captures) == index+1 and tuple(captures[index]) == HEAD_KEYS,'BATCH_HEAD_CAPTURE')
                for task in ('state','future'):
                    piece[task+'_raw']=report[task+'_activity_side_logits'].detach().cpu().numpy().copy()
                    _need(np.array_equal(truths[task]['parent_rows'],parents)
                        and np.array_equal(truths[task]['issue_ns'],clocks)
                        and np.array_equal(truths[task]['allowed'],report[task+'_allowed'])
                        and np.array_equal(truths[task]['mature'],report[task+'_mature']),'ACTUAL_TARGET_AXES')
                    for name in ('allowed','eligible','mature'): piece[task+'_'+name]=truths[task][name].copy()
                    piece[task+'_representation']=np.stack([captures[index][task+'_'+tf] for tf in TIMEFRAMES],axis=1)
                _require_arrays(piece,16,np=np,representations=True)
                loss=float(report['loss']); parts.append(piece); counts['sampled_rows']+=16
                previous._require_parity({k:piece[k] for k in BASE_KEYS},{k:v[selected] for k,v in baseline.items()},
                    loss,q['initial_measurement']['measurement']['canonical_batch_losses'][index],np=np)
                records.append(dict(batch_index=index,rows=16,loss=loss,all12fields_and_loss_bitexact=True))
                counts['matched_batches']+=1
                _need(all(p.grad is None for p in model.parameters()),'NO_GRADIENTS')
                del report,features,raw
        passivity=dict(signature_unchanged=signature()==before,no_parameter_gradients=all(p.grad is None for p in model.parameters()))
        _need(all(passivity.values()),'PASSIVITY')
        exact=dict(model_constructions=1,model_construction_returns=1,checkpoint_loads=1,resume_state_load_calls=1,
            resume_state_load_returns=1,backbone_encodings=256,forward_returns=256,head_hook_registrations=10,
            head_input_observations=2560,extra_target_gathers=512,archive_decodes=1,sampled_rows=4096,matched_batches=256,
            autograd_grad_calls=0,backwards=0,optimizer_constructions=0,optimizer_updates=0,rng_restores=0,
            checkpoint_writes=0,candidate_writes=0)
        _need(all(counts[k] == v for k,v in exact.items()),'COUNTS')
        artifacts['initial_readout_inputs']=_save(parts,counts,partial=False,np=np)
        release=prepared; prepared=None; base._release_pages(provider,release); provider.validate_metadata(provider_plan); bookend()
        return cache._publish_fusion_outcome(q['plan_ref'],dict(claim=claim,counts=counts,artifacts=artifacts,
            source_commit=q['head'],source_unchanged=True,candidate_model_state_sha256=short.MODEL_SHA256,
            feature_view_sha256=view['contract_sha256'],numerical_recipe_sha256=numerical['contract_sha256'],
            parent_order_sha256=recipe['parent_order_sha256'],initial_checkpoint=q['plan']['inputs']['initial_checkpoint'],
            initial_measurement=q['plan']['inputs']['initial_measurement'],batch_records=records,passivity=passivity,
            initial_state={k:before[k] for k in ('model','buffers','rng','modes','flags','view')},numeric_runtime=runtime,
            source_checkpoint_weights_only=True,optimizer_or_rng_restored=False,all_initial_fields_and_loss_bitexact=True,
            elapsed_seconds=time.monotonic()-started,limitations=[
                'Reproduced INITIAL eval inputs on4096dependent historically usedTRAIN rows; not stored historical hidden identities.',
                'No FINAL model, new learning, generalization or predictive utility measured.',
                'Head/representation attribution requires separately qualified retained arithmetic.']),
            complete=True,root=ROOT,json_bytes_bound=BOUNDS['max_published_json_bytes'])
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL,0); changed=save_error=passivity_error=None
        try:
            if before is not None:
                passivity=dict(signature_unchanged=signature()==before,no_parameter_gradients=all(p.grad is None for p in model.parameters()))
        except Exception as other: passivity_error=repr(other)
        try:
            if parts and not artifacts: artifacts['partial_initial_readout_inputs']=_save(parts,counts,partial=True,np=np)
            if active_index is not None and active_index < len(captures) and captures[active_index]:
                artifacts['failed_batch_inputs']=_write_arrays('FAILED_BATCH_INPUTS.npz',
                    dict(batch_axes or {},**captures[active_index]),counts,np=np)
        except Exception as other: save_error=repr(other)
        try: bookend()
        except Exception as other: changed=repr(other)
        result=cache._publish_fusion_outcome(q['plan_ref'],dict(claim=claim,counts=counts,artifacts=artifacts,
            source_commit=q['head'],source_unchanged=changed is None,active_batch_index=active_index,batch_records=records,
            passivity=passivity,passivity_error=passivity_error,numeric_runtime=runtime,error=type(exc).__name__+':'+str(exc),
            bookend_error=changed,partial_save_error=save_error,interpretation_authorized=False),
            complete=False,root=ROOT,json_bytes_bound=BOUNDS['max_published_json_bytes'])
        raise RuntimeError('TA_TF_ACTIVITY_INITIAL_INPUTS_FAILED_CONSUMED:'+str(result)) from exc
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous_alarm)
        if prepared is not None: base._release_pages(provider,prepared)


def main(argv=None):
    import argparse,json,sys
    supplied=list(sys.argv[1:] if argv is None else argv)
    flags=('--market-state-ta-tf-activity-initial-inputs-plan','--market-state-ta-tf-activity-initial-inputs-plan-sha256')
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False,add_help=False)
    parser.add_argument(flags[0],type=Path,required=True); parser.add_argument(flags[1],required=True)
    parser.add_argument('--device',choices=('cuda',),required=True); parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    for flag in (*flags,'--device','--out-dir','--preflight'):
        if sum(v.partition('=')[0] == flag for v in supplied) > 1: parser.error('duplicate flag: '+flag)
    args=parser.parse_args(supplied)
    if not args.out_dir.is_absolute(): parser.error('--out-dir must be absolute')
    kwargs=dict(plan_path=args.market_state_ta_tf_activity_initial_inputs_plan,
        plan_sha=args.market_state_ta_tf_activity_initial_inputs_plan_sha256,out_dir=args.out_dir)
    print(json.dumps(preflight(**kwargs) if args.preflight else run(**kwargs,device=args.device),
        indent=2,sort_keys=True,allow_nan=False))
    return 0


if __name__ == '__main__': raise SystemExit(main())
