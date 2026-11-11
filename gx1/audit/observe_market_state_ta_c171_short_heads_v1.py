"""One read-only c171 initial/final train-mode observation with bound RNG streams.

No numerical model, feature transform, loss, fit or checkpoint writer lives here.
A retained native/fastpath cache is not assumed to equal these short-mode inputs.
"""
from __future__ import annotations

import copy
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
STAGE = 'market_state_ta_c171_matched_train_mode_20261111_001'
ROOT = Path('/home/andre2/GX1_RUNS') / STAGE.upper()
SCHEMA = 'gx1_ta_c171_matched_train_mode_plan_v1'
WORKER = REPO / 'gx1/audit/observe_market_state_ta_c171_short_heads_v1.py'
SHORT_ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_C171_SHORT_LEARNING_20261111_002')
FINAL_MODEL_SHA256 = '5f54bbe47df0f84c60e475d0e9fcf83ad70b8c569d6ebea6ac802aebda655c82'
INITIAL_RNG_SHA256 = '67daad3d6238c809023f066b4599c7634db931c73d9229990a3f76a1845289ae'
FINAL_RNG_SHA256 = 'b06f3644120220289779083e7591cd3b066fa048c0b55d389c834855c0986a60'
INPUTS = {
    **cache.FUSION_REFS,
    'short_initial_checkpoint': {'path': str(SHORT_ROOT/'RUNTIME/private-research/INITIAL_C171_STATE.pt'), 'sha256': '60f13b07832a2da6a236426ed2d88858eb4f5cb2d55668acea06127a8cccee6b'},
    'zero_update_mode_review': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_C171_FULL_USED_PREFIX_20261111_001/TRACE_ZERO_UPDATE_MODE_REVIEW.json', 'sha256': '11734549fab2b03d44526a1b8b63f44efa1c347b7a8c62b644175a625afb2bca'},
    'short_recipe': {'path': str(SHORT_ROOT/'RECIPE.json'), 'sha256': '7e66e231594358e72ad5b6efc8d88e62053bdf24dd8cf4131d3c65d571357b20'},
    'short_result': {'path': str(SHORT_ROOT/'RESULT.json'), 'sha256': '1b25f095c3f610fc9c42325840f6f683cf3e9a50006fcd6249250e07e753724a'},
    'short_initial_measurement': {'path': str(SHORT_ROOT/'INITIAL_TRAIN.json'), 'sha256': '5b20a6fbe2885b5f9ad1998a66852f6773300003c290016e2b40bf2eaf18c099'},
    'short_initial_capture': {'path': str(SHORT_ROOT/'INITIAL_TRAIN.npz'), 'sha256': 'c80969b27b741a477d6b7231cf6646216e36f1829cbaca73c08c6757b1b2b66f'},
    'short_final_measurement': {'path': str(SHORT_ROOT/'FINAL_TRAIN.json'), 'sha256': 'e7ac26da179665e05f7c7c09f884a2a11c2332b6b8e1582d3de0917df5b8f7b7'},
    'short_final_capture': {'path': str(SHORT_ROOT/'FINAL_TRAIN.npz'), 'sha256': 'ff2d9af697a1df167795926601d339808827a610688f930a6bc1f84781404973'},
    'short_final_checkpoint': {'path': str(SHORT_ROOT/'RUNTIME/private-research/FINAL_C171_SHORT_STATE.pt'), 'sha256': '5b4bffaeb10ff296fbb8ece944def51b1087584c52b355c23edaa256f10f3f6b'},
    'short_trace': {'path': str(SHORT_ROOT/'TRACE.npz'), 'sha256': '4bdcb761e01e19677af4f5d758dd203348640b5feadc6df3016edc95cb6b3455'},
}
BOUNDS = dict(fit_rows=29574, batch_size=16, used_prefix_rows=4096,
    panel_batch_indices=list(short.PANEL_BATCH_INDICES), evaluation_batch_indices=list(range(256)),
    control_batch_indices=[0], sampled_rows=8192, unique_rows=4096,
    phases=['INITIAL_TRAIN_MODE','FINAL_TRAIN_MODE'], max_backbone_encodings=512,
    max_model_constructions=1, max_checkpoint_loads=2, head_tensor_imports=0,
    max_checkpoint_weight_transitions=2, max_rng_restores=2, max_grad_enabled_forwards=512, max_autograd_grad_calls=0, max_backwards=0,
    max_optimizer_constructions=0, max_optimizer_updates=0, max_checkpoint_writes=0,
    max_candidate_writes=0, max_cache_bytes=2*1024**2, max_published_json_bytes=1024**2,
    wall_seconds=900, device='cuda', candidate_model_state_sha256=short.MODEL_SHA256,
    final_model_state_sha256=FINAL_MODEL_SHA256,
    numerical_mode='original_train_enable_grad_deterministic_fp32_mha_fastpath_false_no_backward',
    initial_rng_state_sha256=INITIAL_RNG_SHA256, terminal_rng_state_sha256=FINAL_RNG_SHA256,
    original_dropout=0.05, matched_train_mode_observation_authorized=True,
    numerical_threads=8, interop_policy='process_default_unchanged_like_short002',
    thread_environment={name:'8' for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
        'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')},
    parameter_partition='original_fold1_fit_active_true_inactive_false',
    training_enabled=False, test_data_used=False)
BASE_KEYS = ('parents','issues','state_query','future_query','state_raw','future_raw',
    'state_allowed','state_eligible','state_mature','future_allowed','future_eligible','future_mature')
TRACE_KEYS = tuple(k for k in BASE_KEYS if not k.endswith('_raw'))
TRACE_COUNTS = tuple(task+'_'+kind+'_count' for task in ('state','future') for kind in ('known','up'))


def _need(ok, reason):
    if not ok:
        raise RuntimeError('TA_C171_SHORT_HEADS_' + reason)


def build_plan(source_commit):
    """Read-only qualification; never writes/re-admits the consumed short recipe."""
    recipe = short.build_recipe(source_commit=source_commit)
    return dict(schema_version=SCHEMA, stage=STAGE, source_repo=str(REPO),
        source_base_commit=source_commit, worker=cache._ref(WORKER),
        dispatch=cache._ref(WORKER), cap_runner=cache._ref(REPO/cache.CAP_RUNNER),
        source_owners=recipe['source_owners'], inputs=INPUTS, **BOUNDS)


def _require_plan(plan, plan_path, out_dir):
    fields = {'schema_version','stage','source_repo','source_base_commit','worker',
              'dispatch','cap_runner','source_owners','inputs'} | set(BOUNDS)
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
        and result['initial_state']==INPUTS['short_initial_checkpoint']
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
    for key,path in (('worker',WORKER),('dispatch',WORKER),('cap_runner',REPO/cache.CAP_RUNNER)):
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
    mode=cache._read(INPUTS['zero_update_mode_review'])
    _need(mode['status']=='COMPLETE_RETAINED_ZERO_UPDATE_MODE_COMPARISON'
        and mode['short_result']==INPUTS['short_result']
        and mode['zero_update']['updates']==0 and mode['zero_update']['loss_bitexact'] is False
        and mode['checkpoint_metadata']['INITIAL']['checkpoint']==INPUTS['short_initial_checkpoint']
        and mode['checkpoint_metadata']['FINAL']['checkpoint']==INPUTS['short_final_checkpoint']
        and mode['checkpoint_metadata']['INITIAL']['rng_sha256']==INITIAL_RNG_SHA256
        and mode['checkpoint_metadata']['FINAL']['rng_sha256']==FINAL_RNG_SHA256, 'MODE_GAP_BINDING')
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
            data_reads_authorized=True,checkpoint_reads_authorized=True,backbone_encodings_authorized=512,
            head_tensor_imports_authorized=0,candidate_weights_writes_authorized=0,
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


def _layout(rows, *, np, prefix=False):
    _need(type(prefix) is bool and type(rows) is int
        and 0 < rows <= (8192 if prefix else 256) and rows % 16 == 0, 'ROWS')
    result={k:((rows,),np.int64) for k in ('parents','issues')}
    for task in ('state','future'):
        result[task+'_query']=((rows,5),np.int64)
        result[task+'_raw']=((rows,5,2),np.float32)
        for name,dtype in (('allowed',np.uint8),('eligible',np.bool_),('mature',np.bool_)):
            result[task+'_'+name]=((rows,5),dtype)
    if prefix:result['phase']=((rows,),np.uint8)
    return result


def _require_arrays(arrays, rows, *, np, prefix=False, phase=None):
    layout=_layout(rows,np=np,prefix=prefix)
    _need(set(arrays)==set(layout), 'ARRAY_FIELDS')
    for name,(shape,dtype) in layout.items():
        value=arrays[name]
        _need(isinstance(value,np.ndarray) and value.shape==shape and value.dtype==dtype
            and bool(np.isfinite(value).all()), 'ARRAY:'+name)
    if prefix:
        _need(phase is None or type(phase) is int and phase in (0,1), 'PHASE')
        expected=(np.r_[np.zeros(min(rows,4096),np.uint8),np.ones(max(0,rows-4096),np.uint8)]
            if phase is None else np.full(rows,phase,np.uint8))
        _need(np.array_equal(arrays['phase'],expected), 'PHASE_ORDER')
    _need(sum(x.nbytes for x in arrays.values()) <= BOUNDS['max_cache_bytes'], 'ARRAY_BYTES')


def _require_parity(actual, baseline, loss, expected_loss, *, np):
    _require_arrays(actual,16,np=np)
    _require_arrays(baseline,16,np=np)
    _need(cache._fusion_exact_arrays(actual,baseline), 'INITIAL_ARRAY_PARITY')
    _need(type(expected_loss) is float and float(loss)==expected_loss, 'INITIAL_LOSS_PARITY')


def _require_trace(trace, *, np):
    _need(set(trace)==set(TRACE_KEYS+TRACE_COUNTS),'TRACE_FIELDS')
    layout=_layout(16,np=np)
    for key in TRACE_KEYS+TRACE_COUNTS:
        shape,dtype=((5,),np.int64) if key in TRACE_COUNTS else layout[key]
        value=trace[key]
        _need(isinstance(value,np.ndarray) and value.shape==(256,*shape)
            and value.dtype==dtype,'TRACE_LAYOUT:'+key)
    _need(bool(np.all(np.diff(trace['parents'].reshape(-1))>0))
        and bool(np.all(np.diff(trace['issues'].reshape(-1))>0)),'TRACE_CHRONOLOGY')


def _require_trace_batch(piece, trace, index, *, np):
    _need(type(index) is int and 0<=index<256,'TRACE_INDEX')
    actual={k:piece[k] for k in TRACE_KEYS}
    for task in ('state','future'):
        known=piece[task+'_eligible'] & piece[task+'_mature'] & np.isin(piece[task+'_allowed'],(1,4))
        actual[task+'_known_count']=known.sum(0,dtype=np.int64)
        actual[task+'_up_count']=(known & (piece[task+'_allowed']==4)).sum(0,dtype=np.int64)
    _need(cache._fusion_exact_arrays(actual,{k:trace[k][index] for k in TRACE_KEYS+TRACE_COUNTS}),
        'TRACE_FIXED_AXES_PARITY')


def _save(parts, counts, *, partial, np):
    arrays={k:np.concatenate([piece[k] for piece in parts],axis=0) for k in parts[0]}
    _require_arrays(arrays,len(arrays['parents']),np=np,prefix=True)
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


def _require_phase_signature(before, after):
    _need(set(before)==set(after) and 'rng' in before
        and before['rng']==INITIAL_RNG_SHA256 and after['rng']==FINAL_RNG_SHA256
        and all(after[k]==before[k] for k in before if k!='rng'), 'PHASE_STATE_AND_RNG')


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
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts import entry_native_direction_research_state_v1 as old
    from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
    from gx1.contracts.market_state_time_query_binding_v1 import load_bound_time_query_targets
    from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
    from gx1.contracts.market_state_time_query_native_F_batch_v1 import evaluate_fit_batch
    from gx1.contracts.market_state_time_query_native_F_objective_v1 import require_m5_equal_observation_axes
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as model_sha
    from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.audit.trace_market_state_ta_m5_original_cuda_prefix_v1 import _short_loss_components
    from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import _BATCH_EQUAL_LENGTH_MTF_EVAL
    _need(not torch.cuda.is_initialized() and signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'INIT_ORDER')
    claim=cache._publish('CLAIM.json',dict(plan=q['plan_ref'],source_commit=q['head'],cap=cap,
        max_invocations=1,no_relaunch_under_this_plan=True,test_data_used=False),root=ROOT)
    counts=dict(model_constructions=0,checkpoint_loads=0,backbone_encodings=0,forward_returns=0,
        head_tensor_imports=0,archive_decodes=0,autograd_grad_calls=0,backwards=0,
        optimizer_constructions=0,optimizer_updates=0,checkpoint_writes=0,candidate_writes=0,
        resume_state_load_calls=0,resume_state_load_returns=0,published_npz_bytes=0,
        sampled_rows=0,matched_batches=0,trace_matched_batches=0,extra_target_gathers=0,
        checkpoint_weight_transitions=0,loss_component_owner_calls=0,rng_restores=0,
        grad_enabled_forwards=0,paired_rng_batches=0)
    provider=prepared=model=None;parts=[];artifacts={};records=[];candidate=None;passivity={};runtime={};trainability={}
    phase_states={};transition={};active_phase=None
    previous_alarm=signal.getsignal(signal.SIGALRM)
    def timeout(*unused):raise TimeoutError('Matched train-mode 900-second bound')
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
        with np.load(INPUTS['short_trace']['path'],allow_pickle=False) as z:
            trace_keys=BASE_KEYS+TRACE_COUNTS+('loss',)
            _need(set(trace_keys)<=set(z.files),'TRACE_FIELDS')
            trace={k:z[k].copy() for k in trace_keys}
        counts['archive_decodes']+=1
        _require_trace({k:trace[k] for k in TRACE_KEYS+TRACE_COUNTS},np=np)
        _need(trace['loss'].shape==(256,) and trace['loss'].dtype==np.float64
            and bool(np.isfinite(trace['loss']).all()),'TRACE_LOSSES')
        for key in ('state_raw','future_raw'):
            _need(trace[key].shape==(256,16,5,2) and trace[key].dtype==np.float32
                and bool(np.isfinite(trace[key]).all()),'TRACE_RAW')

        provider,provider_plan,prepared,rows,issues=runner._prepared_fold1(receipts,view,np=np)
        require_m5_equal_observation_axes(rows,issues)
        bound,future=load_bound_time_query_targets(),load_bound_future_displacement()
        counts['model_constructions']+=1
        backbone=views.build_native_direction_research_model(view,prefix_normalization=receipts['fold1']['normal'],
            constructor_metadata=receipts['constructor'],device=torch.device('cpu'),unobserved_units_profile=receipts['fold1']['profile'])
        model=NativeFutureDualModel(backbone)
        fit.require_checkpoint_view(model.state_model.feature_view_identity,recipe=q['origin']['original_recipe'],verify_files=True,ta_sf=True)
        counts['checkpoint_loads']+=1
        saved=common._load_state(INPUTS['short_initial_checkpoint'],torch=torch,
            maximum=recipe['max_checkpoint_bytes'],counters=counts)
        _need(saved['model_state_sha256']==short.MODEL_SHA256
            and model_sha(saved['model_state'])==short.MODEL_SHA256
            and (saved['cursor'],saved['optimizer_updates'],saved['skipped_batches'])==(0,0,0)
            and old._tree_sha(saved['rng_state'])==saved['rng_state_sha256']==INITIAL_RNG_SHA256,
            'INITIAL_CHECKPOINT_IDENTITY')
        initial_rng=copy.deepcopy(saved['rng_state']);original_modes=copy.deepcopy(saved['module_training'])
        counts['checkpoint_weight_transitions']+=1
        model.load_state_dict(saved['model_state'],strict=True);del saved
        model.to(cuda)
        _need(set(original_modes)==set(dict(model.named_modules()))
            and all(type(v) is bool for v in original_modes.values()),'ORIGINAL_MODES')
        for name,module in model.named_modules():module.training=original_modes[name]
        trainability=_short_parameter_partition(model,state)
        old._rng(initial_rng,device=cuda,seed=numerical['seed'])
        source_identity=dict(source=INPUTS['short_initial_checkpoint'],model_state_sha256=short.MODEL_SHA256,
            optimizer_or_rng_restored_by_checkpoint_loader=False,checkpoint_weights_only=True)
        import_proof=dict(actual_initial_weights_only=True,head_tensor_imports=0,
            optimizer_state_not_imported=True,original_module_modes=True)
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
        _need(candidate['model']==short.MODEL_SHA256 and candidate['modes']==original_modes
            and model.training and model.state_model.training and model.backbone.training
            and candidate['trainability']==trainability and all(p.grad is None for p in model.parameters()),'C171_INITIAL_PROFILE')
        for phase_number,phase in enumerate(BOUNDS['phases']):
            active_phase=phase
            if phase_number:
                before=signature()
                _require_phase_signature(candidate,before)
                counts['checkpoint_loads']+=1
                saved=common._load_state(INPUTS['short_final_checkpoint'],torch=torch,
                    maximum=recipe['max_checkpoint_bytes'],counters=counts)
                _need(saved['model_state_sha256']==FINAL_MODEL_SHA256
                    and model_sha(saved['model_state'])==FINAL_MODEL_SHA256
                    and (saved['cursor'],saved['optimizer_updates'],saved['skipped_batches'])==(4096,256,0)
                    and saved['module_training']==original_modes
                    and old._tree_sha(saved['rng_state'])==saved['rng_state_sha256']==FINAL_RNG_SHA256,
                    'FINAL_CHECKPOINT_IDENTITY')
                counts['checkpoint_weight_transitions']+=1
                model.load_state_dict(saved['model_state'],strict=True);del saved
                candidate=signature()
                _need(candidate['model']==FINAL_MODEL_SHA256
                    and all(candidate[k]==before[k] for k in before if k not in ('model','buffers')),
                    'EXPLICIT_FINAL_WEIGHT_TRANSITION')
                transition=dict(source=INPUTS['short_final_checkpoint'],from_model=short.MODEL_SHA256,
                    to_model=candidate['model'],nonnumerical_signature_unchanged=True,
                    optimizer_or_rng_restored_by_checkpoint_loader=False,checkpoint_weights_only=True)
            before_reset=signature()
            _need(counts['rng_restores']<BOUNDS['max_rng_restores'],'RNG_RESTORE_BOUND')
            counts['rng_restores']+=1
            trainer._restore_attended_session_rng_state(initial_rng,device=cuda)
            candidate=signature()
            _need(candidate['rng']==INITIAL_RNG_SHA256
                and all(candidate[k]==before_reset[k] for k in candidate if k!='rng'),'EXPLICIT_RNG_RESET')
            phase_states[phase]=dict(model=candidate['model'],buffers=candidate['buffers'],initial_rng=candidate['rng'])
            for index in BOUNDS['evaluation_batch_indices']:
                left=index*16
                raw,parents,clocks=runner._feature_batch(prepared,view,receipts,rows,issues,left,left+16,np=np)
                _need(np.array_equal(parents,trace['parents'][index])
                    and np.array_equal(clocks,trace['issues'][index]),'PREFIX_ROWS')
                features={k:torch.from_numpy(v).to(cuda) for k,v in raw.items()}
                rng_before=old._tree_sha(trainer._attended_session_rng_state(device=cuda))
                with torch.enable_grad(),trainer._training_mha_fastpath_context(model.backbone),trainer._training_autocast_context(),trainer._training_model_finite_check_context():
                    _need(torch.is_grad_enabled() and model.training and torch.backends.mha.get_fastpath_enabled() is False
                        and _BATCH_EQUAL_LENGTH_MTF_EVAL.get() is False
                        and counts['backbone_encodings']<BOUNDS['max_backbone_encodings'],'FORWARD_PROFILE')
                    counts['backbone_encodings']+=1;counts['grad_enabled_forwards']+=1
                    report=evaluate_fit_batch(model,features,bound,future,parents,clocks,fold_id=1,pass_index=0,
                        seed=numerical['seed'],future_weight=1.,m5_equal_observation=True,fit_batch_start=left)
                    counts['forward_returns']+=1
                _need(report['skip_update'] is False and report['model_contract_sha256']==short_recipe['expected_model_contract_sha256']
                    and report['loss_contract_sha256']==short_recipe['expected_loss_contract_sha256']
                    and report['loss'].requires_grad and report['loss'].grad_fn is not None
                    and bool(torch.isfinite(report['loss'])),'REPORT_CONTRACT')
                truths=dict(state=bound.gather_fit_sampled(parents,clocks,fold_id=1,pass_index=0,seed=numerical['seed']),
                    future=future.gather(parents,clocks,fold_id=1,partition='fit'))
                counts['extra_target_gathers']+=2
                piece=dict(phase=np.full(16,phase_number,np.uint8),
                    parents=parents.copy(),issues=clocks.copy(),state_query=report['state_query_k'].copy(),
                    future_query=np.broadcast_to(report['future_native_k'],(16,5)).copy())
                for task in ('state','future'):
                    piece[task+'_raw']=report[task+'_activity_side_logits'].detach().cpu().numpy().copy()
                    for name in ('allowed','eligible','mature'):piece[task+'_'+name]=truths[task][name].copy()
                _require_arrays(piece,16,np=np,prefix=True,phase=phase_number)
                parts.append(piece);counts['sampled_rows']+=16
                loss=float(report['loss'].detach())
                rng_after=old._tree_sha(trainer._attended_session_rng_state(device=cuda))
                records.append(dict(phase=phase,batch_index=index,rows=16,loss=loss,trace_parity=False,
                    control_parity=None,rng_before=rng_before,rng_after=rng_after))
                _require_trace_batch(piece,trace,index,np=np)
                counts['trace_matched_batches']+=1;records[-1]['trace_parity']=True
                with torch.no_grad():components=_short_loss_components(report,truths,torch=torch)
                counts['loss_component_owner_calls']+=1;records[-1]['loss_components']=components
                if phase_number==0 and index==0:
                    expected={k:trace[k][0] for k in BASE_KEYS}
                    _require_parity({k:piece[k] for k in BASE_KEYS},expected,loss,float(trace['loss'][0]),np=np)
                    records[-1]['control_parity']=True;counts['matched_batches']+=1
                if phase_number==1:
                    previous=records[index]
                    _need(previous['phase']==BOUNDS['phases'][0] and previous['batch_index']==index
                        and (previous['rng_before'],previous['rng_after'])==(rng_before,rng_after),'PAIRED_RNG_STREAM')
                    counts['paired_rng_batches']+=1
                _need(all(p.grad is None for p in model.parameters()),'NO_GRADIENTS')
                del report,features,raw
            terminal_signature=signature()
            _require_phase_signature(candidate,terminal_signature)
            phase_states[phase]['terminal_rng']=terminal_signature['rng']
            passivity[phase+'_nonrng_signature_unchanged']=all(terminal_signature[k]==candidate[k]
                for k in candidate if k!='rng')
            passivity[phase+'_rng_matches_original_terminal']=terminal_signature['rng']==FINAL_RNG_SHA256
            passivity[phase+'_all_used_rows_match_trace_fixed_fields']=all(v['trace_parity'] for v in records if v['phase']==phase)
            _need(all(passivity.values()),'PHASE_PASSIVITY')
        passivity['initial_zero_update_arrays_and_loss_bitexact']=records[0]['control_parity'] is True
        passivity['all256paired_batch_rng_boundaries_bitexact']=counts['paired_rng_batches']==256
        passivity['no_parameter_gradients']=all(p.grad is None for p in model.parameters())
        passivity['explicit_transition_preserved_nonnumerical_state']=transition['nonnumerical_signature_unchanged']
        _need(all(passivity.values()),'PASSIVITY')
        exact=dict(model_constructions=1,checkpoint_loads=2,backbone_encodings=512,forward_returns=512,
            head_tensor_imports=0,archive_decodes=1,resume_state_load_calls=2,resume_state_load_returns=2,
            sampled_rows=8192,matched_batches=1,trace_matched_batches=512,extra_target_gathers=1024,
            checkpoint_weight_transitions=2,loss_component_owner_calls=512,autograd_grad_calls=0,backwards=0,
            rng_restores=2,grad_enabled_forwards=512,paired_rng_batches=256)
        _need(all(counts[k]==v for k,v in exact.items()),'COUNTS')
        artifacts['paired_train_mode_prefix']=_save(parts,counts,partial=False,np=np)
        release=prepared;prepared=None;base._release_pages(provider,release);provider.validate_metadata(provider_plan);bookend()
        return cache._publish_fusion_outcome(q['plan_ref'],dict(claim=claim,counts=counts,artifacts=artifacts,
            source_commit=q['head'],source_identity=source_identity,import_proof=import_proof,
            candidate_model_state_sha256=short.MODEL_SHA256,final_model_state_sha256=FINAL_MODEL_SHA256,
            phase_states=phase_states,checkpoint_transition=transition,phase_names=BOUNDS['phases'],
            batch_indices=BOUNDS['evaluation_batch_indices'],control_batch_indices=BOUNDS['control_batch_indices'],batch_records=records,
            passivity=passivity,numeric_runtime=runtime,
            parameter_partition=dict(profile=BOUNDS['parameter_partition'],active_tensors=sum(trainability.values()),
                inactive_tensors=len(trainability)-sum(trainability.values())),
            source_unchanged=True,nonnumerical_state_unchanged_except_declared_rng=True,
            original_initial_measurement=INPUTS['short_initial_measurement'],
            original_final_measurement=INPUTS['short_final_measurement'],elapsed_seconds=time.monotonic()-started,
            limitations=['Exact4096dependent TRAINrows at two fixed weights with originaltrainmode and one paired RNGrealization; no fitting or edge claim.',
                'INITIALbatch0 matches actualTRACE before any update. Both phases share all256batch RNGboundaries and end at originalFINAL RNG.',
                'Originalperbatch RNGstates/masks were not retained. Matching endpoints and source paths do not independently prove all historical masks.',
                'Originaldropout remains0.05; enable_grad constructs discarded graphs, but no backward, parameter gradients or optimizer exists.',
                'This is not an expectation over dropout masks or a replay of the historical changing-weight path.']),
            complete=True,root=ROOT,json_bytes_bound=BOUNDS['max_published_json_bytes'])
    except BaseException as exc:
        counts['backwards']=counts['autograd_grad_calls']
        signal.setitimer(signal.ITIMER_REAL,0)
        changed=save_error=None
        if parts and not artifacts:
            try:artifacts['partial_used_prefix']=_save(parts,counts,partial=True,np=np)
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
