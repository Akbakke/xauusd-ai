"""One read-only paired c171 initial/final joint-boundary observation on TRAIN.

No numerical model, feature transform, loss, fit or checkpoint writer lives here.
A retained native/fastpath cache is not assumed to equal these short-mode inputs.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
import signal
import subprocess
import time

from gx1.audit import cache_market_state_ta_direction_v1 as cache
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts import market_state_ta_m5_short_learning_v1 as short

REPO = cache.REPO
STAGE = 'market_state_ta_c171_short_joint_boundaries_20261111_001'
ROOT = Path('/home/andre2/GX1_RUNS') / STAGE.upper()
SCHEMA = 'gx1_ta_c171_short_joint_boundaries_plan_v1'
WORKER = REPO / 'gx1/audit/observe_market_state_ta_c171_short_heads_v1.py'
HELPER = REPO / 'gx1/audit/ta_c171_joint_boundary_capture_v1.py'
MATCHED_ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_C171_SHORT_INITIAL_HEADS_20261111_002')
SHORT_ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_C171_SHORT_LEARNING_20261111_002')
FINAL_MODEL_SHA256 = '5f54bbe47df0f84c60e475d0e9fcf83ad70b8c569d6ebea6ac802aebda655c82'
INPUTS = {
    **cache.FUSION_REFS,
    'matched_initial_heads': {'path': str(MATCHED_ROOT/'TRAIN_FUSION.npz'), 'sha256': 'ead6516a52299ccaa228e54735a290a7f54140e57f46b28b0aebd6cca9ee8db4'},
    'matched_initial_result': {'path': str(MATCHED_ROOT/'RESULT.json'), 'sha256': 'f6e0c8a3edf50a2f10ad52417f1d73f597594fd1cf26124fac1f49f652184ed7'},
    'short_recipe': {'path': str(SHORT_ROOT/'RECIPE.json'), 'sha256': '7e66e231594358e72ad5b6efc8d88e62053bdf24dd8cf4131d3c65d571357b20'},
    'short_result': {'path': str(SHORT_ROOT/'RESULT.json'), 'sha256': '1b25f095c3f610fc9c42325840f6f683cf3e9a50006fcd6249250e07e753724a'},
    'short_initial_measurement': {'path': str(SHORT_ROOT/'INITIAL_TRAIN.json'), 'sha256': '5b20a6fbe2885b5f9ad1998a66852f6773300003c290016e2b40bf2eaf18c099'},
    'short_initial_capture': {'path': str(SHORT_ROOT/'INITIAL_TRAIN.npz'), 'sha256': 'c80969b27b741a477d6b7231cf6646216e36f1829cbaca73c08c6757b1b2b66f'},
    'short_final_measurement': {'path': str(SHORT_ROOT/'FINAL_TRAIN.json'), 'sha256': 'e7ac26da179665e05f7c7c09f884a2a11c2332b6b8e1582d3de0917df5b8f7b7'},
    'short_final_capture': {'path': str(SHORT_ROOT/'FINAL_TRAIN.npz'), 'sha256': 'ff2d9af697a1df167795926601d339808827a610688f930a6bc1f84781404973'},
    'short_final_checkpoint': {'path': str(SHORT_ROOT/'RUNTIME/private-research/FINAL_C171_SHORT_STATE.pt'), 'sha256': '5b4bffaeb10ff296fbb8ece944def51b1087584c52b355c23edaa256f10f3f6b'},
    'short_trace': {'path': str(SHORT_ROOT/'TRACE.npz'), 'sha256': '4bdcb761e01e19677af4f5d758dd203348640b5feadc6df3016edc95cb6b3455'},
}
BOUNDS = dict(fit_rows=29574, batch_size=16, decoder_width=128, joint_source_width=512,
    panel_batch_indices=list(short.PANEL_BATCH_INDICES), sampled_rows=512, unique_rows=256,
    phases=['INITIAL_TRAIN','FINAL_TRAIN'], max_backbone_encodings=32,
    max_model_constructions=1, max_checkpoint_loads=2, head_tensor_imports=4,
    max_checkpoint_weight_transitions=1, max_autograd_grad_calls=0, max_backwards=0,
    max_optimizer_constructions=0, max_optimizer_updates=0, max_checkpoint_writes=0,
    max_candidate_writes=0, max_cache_bytes=2*1024**2, max_published_json_bytes=1024**2,
    wall_seconds=900, device='cuda', candidate_model_state_sha256=short.MODEL_SHA256,
    final_model_state_sha256=FINAL_MODEL_SHA256,
    numerical_mode='eval_no_grad_deterministic_fp32_training_mha_fastpath_false',
    boundary_observations_authorized=True,
    numerical_threads=8, interop_policy='process_default_unchanged_like_short002',
    thread_environment={name:'8' for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
        'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')},
    parameter_partition='original_fold1_fit_active_true_inactive_false',
    training_enabled=False, test_data_used=False)
BASE_KEYS = ('parents','issues','state_query','future_query','state_raw','future_raw',
    'state_allowed','state_eligible','state_mature','future_allowed','future_eligible','future_mature')
HEAD_KEYS = ('state_representation','future_representation')


def _need(ok, reason):
    if not ok:
        raise RuntimeError('TA_C171_SHORT_HEADS_' + reason)


def build_plan(source_commit):
    """Read-only qualification; never writes/re-admits the consumed short recipe."""
    recipe = short.build_recipe(source_commit=source_commit)
    return dict(schema_version=SCHEMA, stage=STAGE, source_repo=str(REPO),
        source_base_commit=source_commit, worker=cache._ref(WORKER),
        dispatch=cache._ref(WORKER), cap_runner=cache._ref(REPO/cache.CAP_RUNNER),
        boundary_helper=cache._ref(HELPER),
        source_owners=recipe['source_owners'], inputs=INPUTS, **BOUNDS)


def _require_plan(plan, plan_path, out_dir):
    fields = {'schema_version','stage','source_repo','source_base_commit','worker',
              'dispatch','cap_runner','boundary_helper','source_owners','inputs'} | set(BOUNDS)
    _need(isinstance(plan,dict) and set(plan)==fields and plan['schema_version']==SCHEMA
        and plan['stage']==STAGE and plan['source_repo']==str(REPO)
        and Path(plan_path)==ROOT/'PLAN.json' and Path(out_dir)==ROOT
        and plan['inputs']==INPUTS
        and all(type(plan[k]) is type(v) and plan[k]==v for k,v in BOUNDS.items())
        and all(type(i) is int for i in plan['panel_batch_indices']), 'EXACT_PLAN')
    _need(isinstance(plan['source_base_commit'],str)
        and campaign._COMMIT.fullmatch(plan['source_base_commit']), 'SOURCE_BASE')
    return plan


def _qualify_short(transient, result, measurement, original):
    # Only current orchestration/source bindings and mechanical commit may differ.
    ignored = {'source_owners','trace_worker','mechanical_source_commit','recipe_sha256'}
    _need(set(original)==set(transient)
        and {k:v for k,v in original.items() if k not in ignored}
            == {k:v for k,v in transient.items() if k not in ignored}, 'SHORT_RECIPE_IDENTITY')
    _need(result['schema_version']==short.RESULT_SCHEMA and result['outcome']=='COMPLETE'
        and result['recipe']==INPUTS['short_recipe'] and result['source_unchanged'] is True
        and result['test_data_used'] is False and result['initial_restore_exact'] is True
        and result['initial_model_state_sha256']==short.MODEL_SHA256
        and result['optimizer_updates']==256 and result['parent_cursor']==4096
        and result['skipped_batches']==0 and result['all_panels_complete'] is True
        and result['evaluation_state_unchanged'] is True
        and result['panel_batch_indices']==BOUNDS['panel_batch_indices']
        and result['trace']==INPUTS['short_trace']
        and result['measurement_refs']['INITIAL_TRAIN']==INPUTS['short_initial_measurement']
        and result['panel_captures']['INITIAL_TRAIN']==INPUTS['short_initial_capture']
        and result['measurements']['INITIAL_TRAIN']==measurement['measurement'], 'SHORT_COMPLETE')
    m = measurement['measurement']
    _need(measurement['capture']==INPUTS['short_initial_capture']
        and measurement['evaluated_model_state_sha256']==short.MODEL_SHA256
        and measurement['evaluation_state_unchanged'] is True
        and measurement['optimizer_updates']==0 and measurement['test_data_used'] is False
        and measurement['panel_batch_indices']==BOUNDS['panel_batch_indices']
        and m['rows']==256 and m['batches']==16 and len(m['canonical_batch_losses'])==16
        and m['feature_view_sha256']==transient['expected_view_sha256']
        and m['model_contract_sha256']==transient['expected_model_contract_sha256']
        and m['loss_contract_sha256']==transient['expected_loss_contract_sha256'], 'SHORT_INITIAL')


def _qualify_final(transient, result, measurement):
    _need(result['final_model_state_sha256']==FINAL_MODEL_SHA256
        and result['final_state']==INPUTS['short_final_checkpoint']
        and result['retained_final_state']==INPUTS['short_final_checkpoint']
        and result['final_state_validated'] is True
        and result['measurement_refs']['FINAL_TRAIN']==INPUTS['short_final_measurement']
        and result['panel_captures']['FINAL_TRAIN']==INPUTS['short_final_capture']
        and result['measurements']['FINAL_TRAIN']==measurement['measurement'], 'SHORT_FINAL_RESULT')
    m=measurement['measurement']
    _need(measurement['capture']==INPUTS['short_final_capture']
        and measurement['evaluated_model_state_sha256']==FINAL_MODEL_SHA256
        and measurement['evaluation_state_unchanged'] is True
        and measurement['optimizer_updates']==256 and measurement['test_data_used'] is False
        and measurement['panel_batch_indices']==BOUNDS['panel_batch_indices']
        and m['rows']==256 and m['batches']==16 and len(m['canonical_batch_losses'])==16
        and m['feature_view_sha256']==transient['expected_view_sha256']
        and m['model_contract_sha256']==transient['expected_model_contract_sha256']
        and m['loss_contract_sha256']==transient['expected_loss_contract_sha256'], 'SHORT_FINAL')


def _inspect(plan_path, plan_sha, out_dir, *, execute):
    ref = dict(path=str(plan_path),sha256=plan_sha)
    plan = _require_plan(cache._read(ref),plan_path,out_dir)
    bound = [ref]
    for key,path in (('worker',WORKER),('dispatch',WORKER),('cap_runner',REPO/cache.CAP_RUNNER),('boundary_helper',HELPER)):
        _need(plan[key]['path']==str(path), 'SOURCE_PATH:'+key)
        campaign.require_binding(plan[key],label='short heads '+key,verify_file=True);bound.append(plan[key])
    for key,value in plan['inputs'].items():
        campaign.require_binding(value,label='short heads input '+key,verify_file=True);bound.append(value)
    head = cache._git('rev-parse','HEAD')
    subprocess.run(['git','merge-base','--is-ancestor',plan['source_base_commit'],head],cwd=REPO,check=True)
    transient = short.build_recipe(source_commit=head)
    _need(plan['source_owners']==transient['source_owners'], 'CURRENT_SOURCES')
    bound.extend(transient['source_owners'].values())
    bound.extend(transient['input_refs'].values())
    q = short.read_origin(transient)
    original,result,measurement = (cache._read(INPUTS[k]) for k in
        ('short_recipe','short_result','short_initial_measurement'))
    _qualify_short(transient,result,measurement,original)
    final_measurement=cache._read(INPUTS['short_final_measurement'])
    _qualify_final(transient,result,final_measurement)
    matched=cache._read(INPUTS['matched_initial_result'])
    _need(matched['status']=='COMPLETE' and matched['source_unchanged'] is True
        and all(matched['passivity'].values()) and matched['counts']['matched_batches']==16
        and matched['candidate_model_state_sha256']==short.MODEL_SHA256
        and matched['artifacts']['head_inputs']==INPUTS['matched_initial_heads']
        and matched['original_initial_measurement']==INPUTS['short_initial_measurement'], 'MATCHED_HEAD_BASELINE')
    _need(q['view']['contract_sha256']==transient['expected_view_sha256']
        and q['recipe']['numerical_recipe']==transient['numerical_recipe']
        and transient['numerical_recipe']['precision_policy']=='deterministic_fp32', 'VIEW_NUMERICS')
    policy_ref=cache._ref(REPO/'NEXT_RUN_POLICY.json');policy=cache._read(policy_ref);bound.append(policy_ref)
    _need(all(policy.get(k) is False for k in ('training_enabled','native_training_enabled',
        'native_launch_authorized','full_epoch_training_allowed')), 'NO_TRAINING')
    _need(not any(ROOT.glob('*.npz')) and not any((ROOT/n).exists()
        for n in ('CLAIM.json','RESULT.json','TERMINAL.json')), 'CONSUMED')
    if execute:
        gate = policy.get(STAGE,{})
        expected = dict(status='ADMITTED',plan=ref,max_invocations=1,remaining_invocations=1,
            claimed_invocations=0,completed_invocations=0,relaunch_allowed=False,execution_authorized=True,
            data_reads_authorized=True,checkpoint_reads_authorized=True,backbone_encodings_authorized=32,
            head_tensor_imports_authorized=4,candidate_weights_writes_authorized=0,
            backward_calls_authorized=0,optimizer_steps_authorized=0,training_authorized=False)
        _need(all(type(gate.get(k)) is type(v) and gate[k]==v for k,v in {**BOUNDS,**expected}.items()), 'ADMISSION')
        _need(cache._git('branch','--show-current')=='work/gx1-current','BRANCH')
        campaign.require_clean_source(dict(source_repo=str(REPO),source_commit=head))
    return dict(q,plan=plan,plan_ref=ref,head=head,bound_refs=bound,policy_ref=policy_ref,
        transient_recipe=transient,initial_measurement=measurement,final_measurement=final_measurement)


def preflight(plan_path, plan_sha, out_dir):
    q=_inspect(plan_path,plan_sha,out_dir,execute=False)
    return dict(status='PASS_READ_ONLY_SHORT_HEADS_PREFLIGHT',stage=STAGE,plan=q['plan_ref'],
        candidate_model_state_sha256=short.MODEL_SHA256,model_constructions=0,
        array_deserializations=0,checkpoint_loads=0,optimizer_updates=0,claims=0,
        actual_feature_normalization_identity_checked=True,test_data_used=False)


def _layout(rows, *, np, boundaries=False):
    _need(type(boundaries) is bool and type(rows) is int
        and 0 < rows <= (512 if boundaries else 256) and rows % 16 == 0, 'ROWS')
    result={k:((rows,),np.int64) for k in ('parents','issues')}
    for task in ('state','future'):
        result[task+'_query']=((rows,5),np.int64)
        result[task+'_raw']=((rows,5,2),np.float32)
        result[task+'_representation']=((rows,128),np.float32)
        for name,dtype in (('allowed',np.uint8),('eligible',np.bool_),('mature',np.bool_)):
            result[task+'_'+name]=((rows,5),dtype)
    if boundaries:
        result.update(joint_source=((rows,512),np.float32),joint_hidden=((rows,128),np.float32),
            phase=((rows,),np.uint8))
    return result


def _require_arrays(arrays, rows, *, np, heads=True, boundaries=False, phase=None):
    layout=_layout(rows,np=np,boundaries=boundaries)
    if not heads:layout={k:layout[k] for k in BASE_KEYS}
    _need(set(arrays)==set(layout), 'ARRAY_FIELDS')
    for name,(shape,dtype) in layout.items():
        value=arrays[name]
        _need(isinstance(value,np.ndarray) and value.shape==shape and value.dtype==dtype
            and bool(np.isfinite(value).all()), 'ARRAY:'+name)
    if boundaries:
        _need(phase is None or type(phase) is int and phase in (0,1), 'PHASE')
        expected=(np.r_[np.zeros(min(rows,256),np.uint8),np.ones(max(0,rows-256),np.uint8)]
            if phase is None else np.full(rows,phase,np.uint8))
        _need(np.array_equal(arrays['phase'],expected), 'PHASE_ORDER')
    _need(sum(x.nbytes for x in arrays.values()) <= BOUNDS['max_cache_bytes'], 'ARRAY_BYTES')


def _require_parity(actual, baseline, loss, expected_loss, *, np, match_heads=False):
    _require_arrays(actual,16,np=np)
    _require_arrays(baseline,16,np=np,heads=match_heads)
    keys=BASE_KEYS+HEAD_KEYS if match_heads else BASE_KEYS
    _need(cache._fusion_exact_arrays({k:actual[k] for k in keys},baseline), 'INITIAL_ARRAY_PARITY')
    # JSON retained the exact Python float conversion of each original FP32 loss.
    _need(type(expected_loss) is float and float(loss)==expected_loss, 'INITIAL_LOSS_PARITY')


def _save(parts, counts, *, partial, np):
    arrays={k:np.concatenate([piece[k] for piece in parts],axis=0) for k in parts[0]}
    _require_arrays(arrays,len(arrays['parents']),np=np,boundaries=True)
    # Existing immutable publisher allows named fusion artifacts and a 16 MiB cap;
    # enforce this route's stricter aggregate 2 MiB cap before calling it.
    buf=io.BytesIO();np.savez(buf,**arrays)
    _need(counts['published_npz_bytes']+buf.tell()<=BOUNDS['max_cache_bytes'],'OUTPUT_BYTES')
    return cache._save_fusion_arrays('TRAIN_FUSION_PARTIAL.npz' if partial else 'TRAIN_FUSION.npz',
        arrays,counts,np=np,root=ROOT)



def _runtime_profile(torch):
    return dict(intra_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
        thread_environment={name:os.environ.get(name) for name in BOUNDS['thread_environment']})


def _require_runtime_profile(value):
    _need(value['intra_threads']==BOUNDS['numerical_threads']
        and type(value['interop_threads']) is int and value['interop_threads']>0
        and value['thread_environment']==BOUNDS['thread_environment'], 'RUNTIME_THREADS')


def _short_parameter_partition(model, state):
    """Same existing active/inactive flags as build_fold1_fit_optimizer; no optimizer."""
    pairs,active,inactive=state._parameters(model)
    _need(all(p.grad is None for _,p in pairs), 'PENDING_GRADIENT')
    for _,parameter in active:parameter.requires_grad_(True)
    for _,parameter in inactive:parameter.requires_grad_(False)
    expected={name:True for name,_ in active};expected.update({name:False for name,_ in inactive})
    _need({name:p.requires_grad for name,p in pairs}==expected,'PARAMETER_PARTITION')
    return expected


def _runtime_cap():
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_producer_execution, require_capped_lock_ancestry
    cap=require_guarded_cuda_producer_execution();cap['lock']=require_capped_lock_ancestry()
    _need(cap['cgroup_path']=='/sys/fs/cgroup'+cap['lock']['scope']
        and cap['memory_max']==cap['memory_high']==12*1024**3 and cap['swap']==512*1024**2,'CAP')
    for key,value in (('MAX_POWER_LIMIT_W','250'),('MAX_POWER_DRAW_W','170'),('MAX_CORE_TEMP_C','65'),('MAX_MEMORY_TEMP_C','80')):
        _need(os.environ.get('GX1_TRAINER_GPU_'+key)==value,'HARDWARE:'+key)
    return cap


def run(plan_path, plan_sha, out_dir, device='cuda'):
    started=time.monotonic();_need(device=='cuda','CUDA_ONLY')
    q=_inspect(plan_path,plan_sha,out_dir,execute=True)
    cap=_runtime_cap()
    import numpy as np
    import torch
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as base
    from gx1.audit.run_market_state_ta_m5_full_reverse_order_v1 import build_reverse_order
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts import entry_native_direction_research_state_v1 as old
    from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
    from gx1.contracts.market_state_time_query_binding_v1 import load_bound_time_query_targets
    from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
    from gx1.contracts.market_state_time_query_native_F_batch_v1 import evaluate_fit_batch, _m5_readout_hooks
    from gx1.contracts.market_state_time_query_native_F_objective_v1 import require_m5_equal_observation_axes
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as model_sha
    from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.audit.ta_c171_joint_boundary_capture_v1 import capture_joint_boundaries
    from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import _BATCH_EQUAL_LENGTH_MTF_EVAL
    _need(not torch.cuda.is_initialized() and signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'INIT_ORDER')
    claim=cache._publish('CLAIM.json',dict(plan=q['plan_ref'],source_commit=q['head'],cap=cap,
        max_invocations=1,no_relaunch_under_this_plan=True,test_data_used=False),root=ROOT)
    counts=dict(model_constructions=0,checkpoint_loads=0,backbone_encodings=0,forward_returns=0,
        head_tensor_imports=0,archive_decodes=0,autograd_grad_calls=0,backwards=0,
        optimizer_constructions=0,optimizer_updates=0,checkpoint_writes=0,candidate_writes=0,
        resume_state_load_calls=0,resume_state_load_returns=0,published_npz_bytes=0,
        sampled_rows=0,matched_batches=0,head_capture_returns=0,extra_target_gathers=0,
        checkpoint_weight_transitions=0,joint_source_captures=0,joint_hidden_captures=0,query_input_observations=0)
    provider=prepared=model=None;parts=[];artifacts={};records=[];candidate=None;passivity={};runtime={};trainability={}
    phase_states={};transition={};active_phase=None
    previous_alarm=signal.getsignal(signal.SIGALRM)
    def timeout(*unused):raise TimeoutError('Short initial heads 900-second bound')
    def bookend():
        campaign.require_clean_source(dict(source_repo=str(REPO),source_commit=q['head']))
        for ref in q['bound_refs']:campaign.require_binding(ref,label='short heads bookend',verify_file=True)
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,max(1.,900-(time.monotonic()-started)))
    try:
        recipe,receipts,view=(q[k] for k in ('recipe','receipts','view'))
        short_recipe=q['transient_recipe'];numerical=short_recipe['numerical_recipe'];cuda=torch.device('cuda')
        trainer._set_deterministic(numerical['seed'],cuda,numerical['precision_policy'])
        runtime=_runtime_profile(torch);_require_runtime_profile(runtime)
        _need(_BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False,'RUNTIME')
        baselines={}
        for phase,key in [('INITIAL_TRAIN','matched_initial_heads'),('FINAL_TRAIN','short_final_capture')]:
            with np.load(INPUTS[key]['path'],allow_pickle=False) as z:
                expected_keys=BASE_KEYS+HEAD_KEYS if phase=='INITIAL_TRAIN' else BASE_KEYS
                _need(set(z.files)==set(expected_keys),'PANEL_FIELDS')
                baselines[phase]={k:z[k].copy() for k in expected_keys}
            counts['archive_decodes']+=1
            _require_arrays(baselines[phase],256,np=np,heads=phase=='INITIAL_TRAIN')
        _need(all(cache._fusion_exact_arrays({k:baselines['INITIAL_TRAIN'][k]},
            {k:baselines['FINAL_TRAIN'][k]}) for k in BASE_KEYS if not k.endswith('_raw')), 'PAIRED_PANEL_AXES')

        provider,provider_plan,prepared,rows,issues=runner._prepared_fold1(receipts,view,np=np)
        require_m5_equal_observation_axes(rows,issues)
        bound,future=load_bound_time_query_targets(),load_bound_future_displacement()
        counts['model_constructions']+=1
        backbone=views.build_native_direction_research_model(view,prefix_normalization=receipts['fold1']['normal'],
            constructor_metadata=receipts['constructor'],device=torch.device('cpu'),unobserved_units_profile=receipts['fold1']['profile'])
        model=NativeFutureDualModel(backbone)
        fit.require_checkpoint_view(model.state_model.feature_view_identity,recipe=q['origin']['original_recipe'],verify_files=True,ta_sf=True)
        counts['checkpoint_loads']+=1
        saved=common._load_state(INPUTS['reversed_evidence'],torch=torch,maximum=recipe['max_checkpoint_bytes'],counters=counts)
        order=torch.from_numpy(build_reverse_order(rows,recipe['batch_indices'])[0].copy())
        source_identity=cache._load_reversed_weights(model,saved,q,order,torch=torch);del saved,order
        with np.load(INPUTS['folded_coefficients']['path'],allow_pickle=False) as z:
            expected={prefix+suffix for prefix in ('state_model.m5_side_head','future_m5_side_head') for suffix in ('.weight','.bias')}
            _need(set(z.files)==expected,'COEFFICIENT_KEYS');coefficients={name:torch.from_numpy(z[name].copy()) for name in z.files}
        counts['archive_decodes']+=1
        import_proof=state.import_equal_weight_m5_side_heads(model,coefficients,expected_model_state_sha256=cache.REVERSED_MODEL,
            coefficient_artifact=INPUTS['folded_coefficients'],coefficient_recipe=INPUTS['candidate_plan'])
        counts['head_tensor_imports']=4;del coefficients
        model.to(cuda)
        trainability=_short_parameter_partition(model,state)
        def signature():
            return dict(model=model_sha(model.state_dict()),buffers=old._tree_sha(dict(model.named_buffers())),
                rng=old._tree_sha(trainer._attended_session_rng_state(device=cuda)),
                modes={n:bool(m.training) for n,m in model.named_modules()},
                trainability={n:p.requires_grad for n,p in model.named_parameters()},
                parameter_identity={n:id(p) for n,p in model.named_parameters()},
                buffer_identity={n:id(p) for n,p in model.named_buffers()},
                module_identity={n:id(m) for n,m in model.named_modules()},
                hooks={n:{key:tuple((i,id(h)) for i,h in getattr(m,key).items()) for key in
                    ('_forward_hooks','_forward_pre_hooks','_backward_hooks','_backward_pre_hooks')}
                    for n,m in model.named_modules()},mha_fastpath=torch.backends.mha.get_fastpath_enabled(),
                numeric_runtime=_runtime_profile(torch))
        candidate=signature()
        _need(candidate['model']==short.MODEL_SHA256 and not any(candidate['modes'].values())
            and candidate['trainability']==trainability and all(p.grad is None for p in model.parameters()),'C171_INITIAL_PROFILE')
        for phase_number,phase in enumerate(BOUNDS['phases']):
            active_phase=phase
            if phase_number:
                before=signature()
                _need(before==candidate, 'INITIAL_BOOKEND_BEFORE_TRANSITION')
                counts['checkpoint_loads']+=1
                saved=common._load_state(INPUTS['short_final_checkpoint'],torch=torch,
                    maximum=recipe['max_checkpoint_bytes'],counters=counts)
                _need(saved['model_state_sha256']==FINAL_MODEL_SHA256
                    and model_sha(saved['model_state'])==FINAL_MODEL_SHA256
                    and (saved['cursor'],saved['optimizer_updates'],saved['skipped_batches'])==(4096,256,0),
                    'FINAL_CHECKPOINT_IDENTITY')
                counts['checkpoint_weight_transitions']+=1
                model.load_state_dict(saved['model_state'],strict=True);del saved
                candidate=signature()
                _need(candidate['model']==FINAL_MODEL_SHA256
                    and all(candidate[k]==before[k] for k in before if k not in ('model','buffers')),
                    'EXPLICIT_FINAL_WEIGHT_TRANSITION')
                transition=dict(source=INPUTS['short_final_checkpoint'],from_model=short.MODEL_SHA256,
                    to_model=candidate['model'],nonnumerical_signature_unchanged=True,
                    optimizer_or_rng_restored=False,checkpoint_weights_only=True)
            phase_states[phase]=dict(model=candidate['model'],buffers=candidate['buffers'])
            baseline=baselines[phase]
            measurement=q['initial_measurement' if phase_number==0 else 'final_measurement']['measurement']
            for group,index in enumerate(BOUNDS['panel_batch_indices']):
                left=index*16
                raw,parents,clocks=runner._feature_batch(prepared,view,receipts,rows,issues,left,left+16,np=np)
                expected={k:v[group*16:(group+1)*16] for k,v in baseline.items()}
                _need(np.array_equal(parents,expected['parents']) and np.array_equal(clocks,expected['issues']),'PANEL_ROWS')
                features={k:torch.from_numpy(v).to(cuda) for k,v in raw.items()}
                captured,handles=_m5_readout_hooks(model,16)
                try:
                    with capture_joint_boundaries(model,counts=counts,torch=torch) as boundaries:
                        with torch.no_grad(),trainer._training_mha_fastpath_context(model.backbone),trainer._training_autocast_context(),trainer._training_model_finite_check_context():
                            _need(not torch.is_grad_enabled() and torch.backends.mha.get_fastpath_enabled() is False
                                and _BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False
                                and counts['backbone_encodings']<BOUNDS['max_backbone_encodings'],'FORWARD_PROFILE')
                            counts['backbone_encodings']+=1
                            report=evaluate_fit_batch(model,features,bound,future,parents,clocks,fold_id=1,pass_index=0,
                                seed=numerical['seed'],future_weight=1.,m5_equal_observation=True,fit_batch_start=left)
                            counts['forward_returns']+=1
                finally:
                    for handle in handles:handle.remove()
                _need(set(captured)==set(HEAD_KEYS) and set(boundaries)=={'joint_source','joint_hidden'},'BOUNDARY_CAPTURE')
                counts['head_capture_returns']+=2
                _need(report['skip_update'] is False and report['model_contract_sha256']==short_recipe['expected_model_contract_sha256']
                    and report['loss_contract_sha256']==short_recipe['expected_loss_contract_sha256']
                    and bool(torch.isfinite(report['loss'])),'REPORT_CONTRACT')
                truths=dict(state=bound.gather_fit_sampled(parents,clocks,fold_id=1,pass_index=0,seed=numerical['seed']),
                    future=future.gather(parents,clocks,fold_id=1,partition='fit'))
                counts['extra_target_gathers']+=2
                piece=dict(captured,**boundaries,phase=np.full(16,phase_number,np.uint8),
                    parents=parents.copy(),issues=clocks.copy(),state_query=report['state_query_k'].copy(),
                    future_query=np.broadcast_to(report['future_native_k'],(16,5)).copy())
                for task in ('state','future'):
                    piece[task+'_raw']=report[task+'_activity_side_logits'].detach().cpu().numpy().copy()
                    for name in ('allowed','eligible','mature'):piece[task+'_'+name]=truths[task][name].copy()
                _require_arrays(piece,16,np=np,boundaries=True,phase=phase_number)
                parts.append(piece);counts['sampled_rows']+=16
                loss=float(report['loss'].detach())
                records.append(dict(phase=phase,batch_index=index,rows=16,loss=loss,parity=False))
                _require_parity({k:piece[k] for k in BASE_KEYS+HEAD_KEYS},expected,loss,
                    measurement['canonical_batch_losses'][group],np=np,match_heads=phase_number==0)
                records[-1]['parity']=True;counts['matched_batches']+=1
                _need(all(p.grad is None for p in model.parameters()),'NO_GRADIENTS')
                del report,features,raw
            passivity[phase+'_complete_signature_unchanged']=signature()==candidate
            passivity[phase+'_original_arrays_and_losses_bitexact']=all(v['parity'] for v in records if v['phase']==phase)
            _need(all(passivity.values()),'PHASE_PASSIVITY')
        passivity['no_parameter_gradients']=all(p.grad is None for p in model.parameters())
        passivity['explicit_transition_preserved_nonnumerical_state']=transition['nonnumerical_signature_unchanged']
        _need(all(passivity.values()),'PASSIVITY')
        exact=dict(model_constructions=1,checkpoint_loads=2,backbone_encodings=32,forward_returns=32,
            head_tensor_imports=4,archive_decodes=3,resume_state_load_calls=2,resume_state_load_returns=2,
            sampled_rows=512,matched_batches=32,head_capture_returns=64,extra_target_gathers=64,
            checkpoint_weight_transitions=1,joint_source_captures=32,joint_hidden_captures=32,
            query_input_observations=64,autograd_grad_calls=0,backwards=0)
        _need(all(counts[k]==v for k,v in exact.items()),'COUNTS')
        artifacts['paired_boundaries']=_save(parts,counts,partial=False,np=np)
        release=prepared;prepared=None;base._release_pages(provider,release);provider.validate_metadata(provider_plan);bookend()
        return cache._publish_fusion_outcome(q['plan_ref'],dict(claim=claim,counts=counts,artifacts=artifacts,
            source_commit=q['head'],source_identity=source_identity,import_proof=import_proof,
            candidate_model_state_sha256=short.MODEL_SHA256,final_model_state_sha256=FINAL_MODEL_SHA256,
            phase_states=phase_states,checkpoint_transition=transition,phase_names=BOUNDS['phases'],
            batch_indices=BOUNDS['panel_batch_indices'],batch_records=records,
            passivity=passivity,numeric_runtime=runtime,
            parameter_partition=dict(profile=BOUNDS['parameter_partition'],active_tensors=sum(trainability.values()),
                inactive_tensors=len(trainability)-sum(trainability.values())),
            source_unchanged=True,evaluation_state_unchanged=True,
            original_initial_measurement=INPUTS['short_initial_measurement'],
            original_final_measurement=INPUTS['short_final_measurement'],elapsed_seconds=time.monotonic()-started,
            limitations=['Same256dependent historical TRAIN rows at two actual retained endpoint weight states; no heldout/edge claim.',
                'Explicit checkpoint-weight transition is read-only research; no optimizer/RNG restore, fitting or candidate write.',
                'Exact endpoint outputs and initial-head inputs do not prove equality to unrecorded historical intermediate vectors.',
                'Capture isolates paired boundary changes for subsequent retained arithmetic; no selected intervention or historical task attribution.']),
            complete=True,root=ROOT,json_bytes_bound=BOUNDS['max_published_json_bytes'])
    except BaseException as exc:
        counts['backwards']=counts['autograd_grad_calls']
        signal.setitimer(signal.ITIMER_REAL,0)
        changed=save_error=None
        if parts and not artifacts:
            try:artifacts['partial_head_inputs']=_save(parts,counts,partial=True,np=np)
            except Exception as other:save_error=repr(other)
        if candidate is not None:
            try:passivity['complete_signature_unchanged']=signature()==candidate
            except Exception as other:passivity['signature_error']=repr(other)
        try:bookend()
        except Exception as other:changed=repr(other)
        result=cache._publish_fusion_outcome(q['plan_ref'],dict(claim=claim,counts=counts,artifacts=artifacts,
            active_phase=active_phase,phase_states=phase_states,checkpoint_transition=transition,
            batch_records=records,passivity=passivity,numeric_runtime=runtime,error=type(exc).__name__+':'+str(exc),
            source_unchanged=changed is None,bookend_error=changed,partial_save_error=save_error,
            interpretation_authorized=False),complete=False,root=ROOT,json_bytes_bound=BOUNDS['max_published_json_bytes'])
        raise RuntimeError('TA_C171_SHORT_HEADS_FAILED_CONSUMED:'+str(result)) from exc
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous_alarm)
        if prepared is not None:base._release_pages(provider,prepared)


def main(argv=None):
    """Exact standalone observation route; the historical dispatcher stays sealed."""
    import argparse
    import json
    import sys
    supplied=list(sys.argv[1:] if argv is None else argv)
    options=("--market-state-ta-c171-short-heads-plan",
             "--market-state-ta-c171-short-heads-plan-sha256")
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False,add_help=False)
    parser.add_argument(options[0],type=Path,required=True)
    parser.add_argument(options[1],required=True)
    parser.add_argument("--device",choices=("cuda",),required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--preflight",action="store_true")
    for option in (*options,"--device","--out-dir","--preflight"):
        if sum(value.partition("=")[0]==option for value in supplied)>1:
            parser.error(f"duplicate flag: {option}")
    args=parser.parse_args(supplied)
    if not args.out_dir.is_absolute():
        parser.error("--out-dir must be an absolute path")
    kwargs=dict(plan_path=args.market_state_ta_c171_short_heads_plan,
                plan_sha=args.market_state_ta_c171_short_heads_plan_sha256,out_dir=args.out_dir)
    report=preflight(**kwargs) if args.preflight else run(**kwargs,device="cuda")
    print(json.dumps(report,indent=2,sort_keys=True,allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
