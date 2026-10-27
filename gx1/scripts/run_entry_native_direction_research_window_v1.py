"""Guarded private CUDA4U pure-direction resume qualification; never a fit.

The public entry repeats the canonical CUDA guard before context/numerical imports.
Immutable267/264 are physical ancestry only; no old run/claim/frozen policy is
executed. Caller-bound current sources and this job's metadata/payload bookends
remain authoritative. Direct boundary counters are not independent profiling.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import stat
import subprocess
import sys
import tempfile
import time

REPO = Path('/home/andre2/src/GX1_CURRENT')
MANIFEST_KEYS = frozenset(('schema_version','qualified_input_design','physical_provider_plan','physical_provider_worker',
    'prefix_normalization','constructor_metadata','view_source_bindings','parent_rows','state_qualification',
    'zero_update_qualification','input_qualification'))
INPUT_DESIGN = {'path':'/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_NATIVE_ZERO_UPDATE_QUALIFICATION_20261024_269/DESIGN_INPUT_COVERAGE_REPAIR.json',
    'sha256':'2534575f89f38546865df061fb492c153cb572fc262fffb35905ec194153f1a0'}
STATE_QUALIFICATION = {'path':'/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_RESEARCH_STATE_NONFINITE_TEST_REPAIR_20261024_271/ACTUAL_QUALIFICATION_REVIEW.json',
    'sha256':'662f5dda61432d212d100adf2ca1635d5783200087951d4fb14e00742cfa3258'}
ZERO_QUALIFICATION = {'path':'/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_NATIVE_ZERO_UPDATE_QUALIFICATION_20261024_269/ACTUAL_QUALIFICATION_REVIEW.json',
    'sha256':'7469b56734102442eae9e855eec17cc3a86a65c5e572300733194e84e5f482f1'}
PARENTS = (47595,47596,47597,47598)
STAT_KEYS = ('st_dev','st_ino','st_mode','st_size','st_mtime_ns','st_ctime_ns')


def _require(ok, reason):
    if not ok:raise RuntimeError('NATIVE_DIRECTION_RESEARCH_WINDOW_'+reason)


def _stat(path):
    path=Path(path)
    _require(path.is_absolute() and '..' not in path.parts and not any(p.is_symlink() for p in (path,*path.parents)),'REGULAR_PATH')
    value=path.lstat();_require(stat.S_ISREG(value.st_mode),'REGULAR_FILE')
    return {name:int(getattr(value,name)) for name in STAT_KEYS}


def _hash_file(path):
    before=_stat(path);digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    _require(_stat(path)==before,'HASH_BOOKEND')
    return {'path':str(path),'sha256':digest.hexdigest()}


def _read(ref, metadata):
    _require(isinstance(ref,dict) and set(ref)=={'path','sha256'} and _stat(ref['path'])['st_size']<=16*1024**2,'SMALL_PLAINREF')
    before=_stat(ref['path']);_require(_hash_file(ref['path'])==ref,'INPUT_SHA')
    value=json.loads(Path(ref['path']).read_text());_require(_stat(ref['path'])==before,'INPUT_BOOKEND')
    metadata[ref['path']]={'ref':dict(ref),'stat':before};return value


def _module(ref):
    _require(_hash_file(ref['path'])==ref,'PHYSICAL_WORKER_SHA')
    spec=importlib.util.spec_from_file_location('gx1_research_physical267',ref['path'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    _require(_hash_file(ref['path'])==ref,'PHYSICAL_WORKER_BOOKEND');return module


def _validate_metadata(recipe):
    metadata={};manifest=_read(recipe['input_manifest'],metadata)
    _require(set(manifest)==MANIFEST_KEYS and manifest['schema_version']=='gx1_native_direction_research_input_manifest_v1'
        and manifest['qualified_input_design']==INPUT_DESIGN and manifest['state_qualification']==STATE_QUALIFICATION
        and manifest['zero_update_qualification']==ZERO_QUALIFICATION and manifest['parent_rows']==list(PARENTS),'FIXED_INPUT_MANIFEST')
    design=_read(INPUT_DESIGN,metadata)
    for key in ('physical_provider_plan','physical_provider_worker','prefix_normalization','constructor_metadata','view_source_bindings','input_qualification'):
        _require(manifest[key]==design[key],'INPUT_PROVENANCE:'+key)
    _read(STATE_QUALIFICATION,metadata);_read(ZERO_QUALIFICATION,metadata);_read(manifest['input_qualification'],metadata)
    physical=_module(manifest['physical_provider_worker']);ancestor=physical.provider_module()
    old_plan=_read(manifest['physical_provider_plan'],metadata)
    _old_design,ancestral_plan,info=physical.validate_inputs(old_plan,ancestor)
    fitted=_read(manifest['prefix_normalization']['output'],metadata);constructor=_read(manifest['constructor_metadata'],metadata)
    _require(fitted['contract_sha256']==manifest['prefix_normalization']['combined_contract_sha256']
        and fitted['base_normalization']['contract_sha256']==manifest['prefix_normalization']['base_normalization_sha256']
        and fitted['gc_mtf_history']['layout']['contract_sha256']==manifest['prefix_normalization']['gc_layout_sha256']
        and fitted['gc_mtf_history']['normalization']['contract_sha256']==manifest['prefix_normalization']['gc_normalization_sha256'],'PREFIX_IDENTITIES')
    for ref in manifest['view_source_bindings'].values():_read(ref,metadata) if ref['path'].endswith('.json') else _require(_hash_file(ref['path'])==ref,'VIEW_OWNER_SHA')
    info['metadata'].update(metadata)
    return {'manifest':manifest,'design':design,'physical':physical,'ancestor':ancestor,'ancestor_plan':ancestral_plan,
        'info':info,'fitted':fitted,'constructor':constructor}


def _freeze(recipe, binding, context):
    _require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==binding['source_head']
        and subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)=='','SOURCE_FREEZE')
    _require(_hash_file(binding['policy']['path'])==binding['policy'],'POLICY_FREEZE')
    for ref in recipe['source_owners'].values():_require(_hash_file(ref['path'])==ref,'CURRENT_SOURCE_OWNER')
    for path,item in context['ancestor_plan']['payloads'].items():_require(_stat(path)==item['stat'],'PAYLOAD_STAT:'+path)
    for path,item in context['info']['metadata'].items():
        _require(_stat(path)==item['stat'] and _hash_file(path)==item['ref'],'METADATA_FREEZE:'+path)


def _install_read_guard(context, output, runtime):
    plan=context['ancestor_plan'];info=context['info']
    allowed={p for p,v in plan['payloads'].items() if v['verification'] not in ('inherited_stat_only_no_open','reused_original051_stat_only_no_open')}|set(info['metadata'])
    directories={plan['native_cache_dir'],str(Path(plan['gc_surfaces']['path']).parent)}
    relative={Path(p).name:p for p in allowed if p.endswith('.npy')};denied=[]
    def guard(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
        path=os.fsdecode(args[0])
        if path.startswith('/proc/self/fd/'):
            try:path=os.readlink(path)
            except OSError:return
        elif '/' not in path and path in relative:path=relative[path]
        if path in directories:
            flags=args[2] if len(args)>2 else None
            _require(isinstance(flags,int) and flags&os.O_DIRECTORY and not flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC),'READONLY_EXACT_DIRECTORY');return
        if path.startswith(('/home/andre2/GX1_DATA/','/home/andre2/GX1_RUNS/')):
            permitted=path in allowed or path.startswith(str(output)+'/') or path.startswith(str(runtime)+'/')
            if not permitted or path==info['audit']['manifest_path']:
                if len(denied)<16:denied.append(path)
                raise RuntimeError('NATIVE_DIRECTION_RESEARCH_WINDOW_UNBOUND_OPEN:'+path)
    sys.addaudithook(guard);return denied


def _qualify_fixed_targets(arrays, target, *, cutoff_ns, np):
    decisions=arrays['decision_ns'];parents=arrays['parent_rows'];stop=int(np.searchsorted(decisions,cutoff_ns,side='left'))
    _require(stop==47602 and int(decisions[stop-1])==1735682100000000000
        and np.array_equal(parents[stop-7:stop-3],np.asarray(PARENTS,np.int64)),'FIXED_PREFIX_CLOCK_AND_PARENT_SLICE')
    rows=len(parents)
    _require(target['terminal_time_ns'].dtype==np.int64 and target['terminal_time_ns'].shape==(rows,3)
        and target['observed'].dtype==np.bool_ and target['observed'].shape==(rows,3)
        and target['sign'].dtype==np.int8 and target['sign'].shape==(rows,3),'TARGET_HEADERS')
    minute=60_000_000_000
    _require(np.all(decisions<=np.iinfo(np.int64).max-15*minute)
        and np.array_equal(target['terminal_time_ns'],decisions[:,None]+np.asarray([1,5,15],np.int64)[None,:]*minute)
        and np.isin(target['sign'],(-1,0,1)).all() and np.all(target['sign'][~target['observed']]==0),'TARGET_CLOCK_DOMAIN')
    indices=np.searchsorted(parents,np.asarray(PARENTS,np.int64));selected=decisions[indices]
    _require(np.array_equal(parents[indices],np.asarray(PARENTS,np.int64)) and np.all(selected<cutoff_ns)
        and np.all(selected+15*minute<=decisions[stop-1]),'PREDECLARED_CLOCK_MARGIN')
    batches=[];report=[]
    for left in (0,2):
        at=indices[left:left+2];batch={key:np.ascontiguousarray(value[at]) for key,value in target.items()}
        batch['decision_ns']=np.ascontiguousarray(decisions[at])
        eligible=batch['observed']&(batch['sign']!=0)&(batch['terminal_time_ns']<cutoff_ns)
        counts=eligible.sum(axis=0).astype(np.int64).tolist()
        _require(all(number>0 for number in counts),'ALL_THREE_HORIZONS_PER_FIXED_BATCH_NO_SEARCH')
        batches.append(batch);report.append({'parent_rows':list(PARENTS[left:left+2]),'decision_ns':selected[left:left+2].tolist(),
            'eligible_cells_per_horizon':counts,'observed_cells_per_horizon':batch['observed'].sum(axis=0).astype(np.int64).tolist()})
    return batches,report


def _gc_support(raw, layout, np):
    report={}
    for tf in layout['timeframes']:
        values=raw['features']['gc_seq_'+tf.lower()];report[tf]={}
        for family,indices in layout['family_indices'][tf].items():
            if not indices:continue
            pairs=[(v,m) for v,m in layout['value_mask_pairs'][tf] if v in set(indices)];_require(pairs,'GC_OWN_MASK_PAIRS')
            vi,mi=map(list,zip(*pairs));masks=values[:,:,mi];source=values[:,:,vi]
            _require(np.isfinite(masks).all() and np.isin(masks,(0,1)).all() and np.all(source[masks==0]==0),'GC_MASK_STORAGE')
            counts=np.count_nonzero(masks==1,axis=(1,2)).astype(np.int64).tolist();_require(sum(counts)>0,'GC_TF_FAMILY_SUPPORT_NO_SEARCH')
            report[tf][family]={'observed_cells_per_parent':counts,'raw_nonzero_values_per_parent':np.count_nonzero(source,axis=(1,2)).astype(np.int64).tolist()}
    _require(len(report)==5 and all(len(item)==5 for item in report.values()),'GC_TWENTY_FIVE_GROUPS');return report


def _prepare_batches(context, np):
    from gx1.contracts import entry_native_direction_feature_view_v1 as feature
    from gx1.features.htf_features import discard_multi_tf_v4_cache_pages
    physical,ancestor,plan,info=(context[k] for k in ('physical','ancestor','ancestor_plan','info'))
    prepared=physical.prepare_inputs(ancestor,plan,info,np)
    physical.release_pages(ancestor,prepared,discard_multi_tf_v4_cache_pages)
    design=context['design'];manifest=context['manifest']
    _require(prepared['gc']['layout']==_read(manifest['view_source_bindings']['raw_gc_layout'],info['metadata'])
        and prepared['gc']['transition']==_read(manifest['view_source_bindings']['raw_gc_transition'],info['metadata']),'QUALIFIED_RAW_DOMAIN')
    with np.load(design['targets']['path'],allow_pickle=False) as archive:
        _require(archive.files==design['archive_members'],'EXACT_ELEVEN_ARCHIVE_NAMES')
        target={key:archive[key] for key in ('terminal_time_ns','observed','sign')}
    targets,report=_qualify_fixed_targets(prepared['arrays'],target,cutoff_ns=manifest['prefix_normalization']['cutoff_ns'],np=np)
    for value in prepared['arrays'].values():value.setflags(write=False)
    for key in ('times_ns','sample_positions'):prepared['entry_sequence_source'][key].setflags(write=False)
    view=feature.bind_native_direction_feature_view(prefix_normalization=context['fitted'],prefix_normalization_ref=manifest['prefix_normalization']['output'],
        constructor_metadata=context['constructor'],constructor_metadata_ref=manifest['constructor_metadata'],source_bindings=manifest['view_source_bindings'],max_batch_rows=2)
    raw=[]
    for left in (0,2):
        batch=feature.gather_native_direction_feature_batch(view,parent_rows=prepared['arrays']['parent_rows'],bar_start_ns=prepared['arrays']['bar_start_ns'],
            decision_ns=prepared['arrays']['decision_ns'],entry_sequence_source=prepared['entry_sequence_source'],snap_x=prepared['current']['snap'],
            ctx_cont=prepared['current']['ctx_cont'],ctx_cat=prepared['current']['ctx_cat'],multi_tf_sources=prepared['cache'],gc_surfaces=prepared['gc']['surfaces'],
            gc_layout=prepared['gc']['layout'],batch_parent_rows=np.asarray(PARENTS[left:left+2],np.int64))
        _require(len(batch['features'])==14 and sum(v.nbytes for v in batch['features'].values())==5013032,'EXACT_RAW_BATCH')
        report[left//2].update(witness=batch['witness'],gc_support=_gc_support(batch,prepared['gc']['layout'],np));raw.append(batch['features'])
    physical.release_pages(ancestor,prepared,discard_multi_tf_v4_cache_pages)
    return view,raw,targets,report


def _json(path, value, counters, limit):
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    _require(counters['json_bytes']+len(data)<=limit,'JSON_OUTPUT_BUDGET')
    Path(path).parent.mkdir(parents=True,exist_ok=True);_publish_bytes(Path(path),data);counters['json_bytes']+=len(data)
    return _hash_file(path)


def _save_state(path, state, *, torch, maximum, counters):
    from gx1.models.entry_v10.entry_v10_ctx_train_v3 import _fsync_directory
    from gx1.scripts.run_unified_exit_random_access_full_train_v1 import _dynamic_m5_technical_tree_sha256 as tree_sha
    path=Path(path);_require(not path.exists() and not path.is_symlink(),'CHECKPOINT_EXISTS')
    fd,temporary=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:torch.save(state,handle);handle.flush();os.fsync(handle.fileno())
        _require(0<_stat(temporary)['st_size']<=maximum,'CHECKPOINT_BYTES')
        staged=_hash_file(temporary);before=_stat(temporary)
        counters['staged_state_load_calls']+=1
        loaded=torch.load(temporary,map_location='cpu',weights_only=True);counters['staged_state_load_returns']+=1
        _require(_stat(temporary)==before and _hash_file(temporary)==staged and tree_sha(loaded)==tree_sha(state),'CHECKPOINT_STAGED_ROUNDTRIP')
        del loaded
        os.link(temporary,path);path.chmod(0o444);_fsync_directory(path.parent)
    finally:
        if Path(temporary).exists():Path(temporary).unlink()
    return _hash_file(path)


def _load_state(ref, *, torch, maximum, counters):
    _require(0<_stat(ref['path'])['st_size']<=maximum and _hash_file(ref['path'])==ref,'CHECKPOINT_LOAD_SHA_SIZE')
    before=_stat(ref['path']);counters['resume_state_load_calls']+=1
    value=torch.load(ref['path'],map_location='cpu',weights_only=True);counters['resume_state_load_returns']+=1
    _require(_stat(ref['path'])==before,'CHECKPOINT_LOAD_BOOKEND');return value


def _fresh(context, view, recipe, device, counters, *, max_constructions=3):
    from gx1.contracts.entry_native_direction_feature_view_v1 import build_native_direction_research_model
    from gx1.contracts.entry_native_direction_research_state_v1 import build_native_direction_research_optimizer
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    trainer._set_deterministic(recipe['seed'],device,recipe['precision_policy'])
    _require(type(max_constructions) is int and max_constructions in (3,4),'CONSTRUCTOR_PROFILE')
    counters['model_constructions']+=1;_require(counters['model_constructions']<=max_constructions,'CONSTRUCTOR_BUDGET')
    model=build_native_direction_research_model(view,prefix_normalization=context['fitted'],constructor_metadata=context['constructor'],device=device)
    counters['model_construction_returns']+=1;model.train()
    optimizer,binding=build_native_direction_research_optimizer(model,recipe=recipe);counters['optimizer_constructions']+=1
    return model,optimizer,binding


def _gradient_evidence(model, torch):
    groups={'direction_head':list(model.direction_head.parameters()),'entry_q_joint':list(model.backbone.entry_q_joint_in.parameters())+list(model.backbone.entry_q_joint_norm.parameters())}
    for tf in ('m5','m15','h1','h4','d1'):
        for label,modules in (('gc_projection',model.backbone.gc_mtf_family_proj),('gc_gate',model.backbone.gc_mtf_feature_context_gate)):
            keys=[key for key in modules if key.startswith(tf+'__')];_require(len(keys)==5,'GC_FAMILY_COUNT')
            groups[label+'_'+tf]=[p for key in keys for p in modules[key].parameters()]
    if getattr(model,'signed_bps',False):groups['bps_mean_head']=list(model.bps_mean_head.parameters())
    claimed={id(p) for values in groups.values() for p in values}
    inactive=[p for name,p in model.named_parameters() if not p.requires_grad]
    groups['base_backbone']=[p for p in model.parameters() if p.requires_grad and id(p) not in claimed]
    _require(all(p.grad is None for p in inactive),'INACTIVE_GRADIENT')
    _require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'FINITE_GRADIENTS')
    evidence={name:sum(int(torch.count_nonzero(p.grad)) for p in values if p.grad is not None) for name,values in groups.items()}
    _require(all(value>0 for value in evidence.values()),'REQUIRED_GRADIENT_GROUP')
    _require(model.direction_head.weight.grad is not None and model.direction_head.bias.grad is not None,'DIRECT_HEAD_GRADIENTS')
    evidence['direction_head_weight_rows']=[int(torch.count_nonzero(row)) for row in model.direction_head.weight.grad]
    evidence['direction_head_bias_rows']=[int(value!=0) for value in model.direction_head.bias.grad]
    _require(all(weight or bias for weight,bias in zip(evidence['direction_head_weight_rows'],evidence['direction_head_bias_rows'])),'ALL_THREE_SUPERVISED_HEAD_ROWS');return evidence


def _step(model, optimizer, features, target, *, cutoff, counters, torch):
    from gx1.contracts.entry_native_direction_research_v1 import native_direction_binary_loss
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as hybrid
    _require(model.training and all(module.training for module in model.modules()),'TRAIN_MODE')
    counters['forwards']+=1;_require(counters['forwards']<=4,'FORWARD_BUDGET')
    with trainer._training_mha_fastpath_context(model.backbone):
        _require(not torch.backends.mha.get_fastpath_enabled() and not hybrid._BATCH_EQUAL_LENGTH_MTF_EVAL.get(),'CANONICAL_FP32_MHA')
        out=trainer._model_forward_fp32(model,**features)
    counters['forward_returns']+=1
    loss=native_direction_binary_loss(out['logit_up'],**target,cutoff_ns=cutoff)
    _require(loss['update_allowed'] and all(row['eligible_cells']>0 for row in loss['per_horizon_counts']),'ALL_HORIZON_LOSS_NO_SEARCH')
    counters['backwards']+=1;_require(counters['backwards']<=4,'BACKWARD_BUDGET');loss['loss'].backward();counters['backward_returns']+=1
    gradients=_gradient_evidence(model,torch)
    counters['optimizer_step_calls']+=1;_require(counters['optimizer_step_calls']<=4,'OPTIMIZER_STEP_BUDGET')
    norm=trainer._optimizer_step_with_finite_gradients(model=model,optimizer=optimizer)
    counters['optimizer_updates']+=1;_require(counters['optimizer_updates']<=4,'UPDATE_BUDGET')
    return {'loss':float(loss['loss'].detach()),'counts':loss['counts'],'per_horizon_counts':loss['per_horizon_counts'],
        'gradient_nonzero_counts':gradients,'gradient_norm':float(norm)}


def _deny_legacy_work(previous, counters):
    forbidden=frozenset(('fit_native_direction_prefix_normalization','qualify_native_direction_prefix_population',
        'fit_surface_normalization','fit_gc_mtf_history_normalization','fit_ctx_cat_contract','build_dynamic_m5_batch_targets',
        '_materialize_dynamic_m5_batch_targets','_dynamic_m5_optimizer_task_losses','read_dynamic_m5_target_source',
        'query_dynamic_m1_prefix','_run_bound_full_train_candidate'))
    def guard(frame,event,arg):
        if event=='call' and frame.f_code.co_name in forbidden:
            counters['forbidden_legacy_calls']+=1
            raise RuntimeError('NATIVE_DIRECTION_RESEARCH_WINDOW_FORBIDDEN_LEGACY:'+frame.f_code.co_name)
        if previous is not None:previous(frame,event,arg)
    return guard


def _io():
    values={key:int(number) for key,number in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}
    values['peak_process_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024;return values


def run_native_direction_research_window(*, policy_path: Path, policy_file_sha256: str, progress_path: Path) -> dict:
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    limits=require_guarded_cuda_trainer_execution()
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    window,plan,invocation,marker,launch=authority.require_native_direction_research_context(policy_path,policy_file_sha256,progress_path)
    if launch['run_scope'] == authority.BPS_SCOPE:
        from gx1.scripts.run_entry_native_direction_bps_qualification_v1 import run_bps_qualification
        return run_bps_qualification(policy_path=policy_path,policy_file_sha256=policy_file_sha256,progress_path=progress_path)
    sys.dont_write_bytecode=True
    import numpy as np
    import torch
    from gx1.contracts.entry_model_native_train_recipe_v1 import require_native_direction_research_recipe
    from gx1.contracts import entry_native_direction_research_state_v1 as state_owner
    from gx1.scripts.run_unified_exit_random_access_full_train_v1 import _dynamic_m5_technical_tree_sha256 as tree_sha
    recipe=require_native_direction_research_recipe(launch['numerical_recipe']);_require(recipe['batch_size']==recipe['max_updates']==2,'FIXED_PRIVATE_RECIPE')
    binding_ref=_hash_file(launch['execution_binding_path']);extra={};binding=_read(binding_ref,extra)
    context=_validate_metadata(launch);context['info']['metadata'].update(extra)
    _read(window['recipe'],context['info']['metadata']);_read(binding['plan'],context['info']['metadata'])
    _read({'path':str(policy_path),'sha256':policy_file_sha256},context['info']['metadata'])
    _freeze(launch,binding,context)
    output=Path(window['output_directory']);output.mkdir(parents=True)
    denied=_install_read_guard(context,output,Path(launch['runtime_root']))
    counters={name:0 for name in ('model_constructions','model_construction_returns','optimizer_constructions','forwards','forward_returns','backwards','backward_returns','optimizer_step_calls','optimizer_updates','staged_state_load_calls','staged_state_load_returns','resume_state_load_calls','resume_state_load_returns','forbidden_legacy_calls','json_bytes')}
    before=_io();started=time.monotonic();full_state=None;cursor=0;error=None;initial=None;initial_identity=None;steps=[];parity=False;unchanged=False
    order=torch.tensor(PARENTS,dtype=torch.int64);order_sha=tree_sha(order);device=torch.device('cuda',0)
    _json(output/'CLAIM.json',{'recipe':window['recipe'],'plan_sha256':plan['plan_sha256'],'limits':limits},counters,launch['max_published_json_bytes'])
    previous_profile=sys.getprofile()
    try:
        sys.setprofile(_deny_legacy_work(previous_profile,counters))
        view,raw,targets,input_report=_prepare_batches(context,np)
        _json(output/'INPUT_QUALIFICATION.json',{'view':view,'batches':input_report,'parent_order_sha256':order_sha,
            'selection':'fixed_prefix_N-7:N-3_before_labels','observed_zero_feature_values_valid':True},counters,launch['max_published_json_bytes'])
        batches=[({key:torch.from_numpy(value).to(device) for key,value in features.items()},
            {key:torch.from_numpy(value).to(device) for key,value in target.items()}) for features,target in zip(raw,targets)]
        batch_sha=tree_sha(batches);initial_inactive=None
        def capture(model,optimizer,offset,updates):
            return state_owner.capture_native_direction_research_state(model,optimizer,recipe=recipe,feature_view_identity=model.feature_view_identity,
                parent_order=order,cursor=offset,optimizer_updates=updates,device=device)
        for trajectory in ('reference','interrupted','resumed'):
            cursor=0
            model,optimizer,optimizer_binding=_fresh(context,view,recipe,device,counters)
            fresh=capture(model,optimizer,0,0)
            identity=tree_sha({key:fresh[key] for key in ('binding','model_state','buffer_state','optimizer_state','rng_state','module_training')})
            if initial_identity is None:
                initial_identity=identity;initial=_save_state(output/'INITIAL_STATE.pt',fresh,torch=torch,maximum=launch['max_checkpoint_bytes'],counters=counters)
                initial_inactive=tree_sha({name:p for name,p in model.named_parameters() if not p.requires_grad})
            else:_require(identity==initial_identity,'FRESH_CUDA_INITIAL_IDENTITY')
            del fresh
            if trajectory=='resumed':
                loaded=_load_state(interrupted_ref,torch=torch,maximum=launch['max_checkpoint_bytes'],counters=counters)
                restored=state_owner.restore_native_direction_research_state(loaded,model,optimizer,recipe=recipe,feature_view_identity=model.feature_view_identity,parent_order=order,device=device)
                _require(restored['cursor']==2 and restored['optimizer_updates']==1 and capture(model,optimizer,2,1)['contract_sha256']==loaded['contract_sha256'],'ACTUAL_SERIALIZED_RESTORE')
                del loaded;cursor=2
            logical=(0,1) if trajectory=='reference' else (0,) if trajectory=='interrupted' else (1,)
            for index in logical:
                _require(time.monotonic()-started<window['max_invocation_seconds'],'DEADLINE')
                evidence=_step(model,optimizer,*batches[index],cutoff=context['manifest']['prefix_normalization']['cutoff_ns'],counters=counters,torch=torch)
                steps.append({'trajectory':trajectory,'logical_step':index+1,**evidence});cursor=2*(index+1)
                _require(tree_sha({name:p for name,p in model.named_parameters() if not p.requires_grad})==initial_inactive,'INACTIVE_WEIGHTS_UNCHANGED')
            saved=capture(model,optimizer,cursor,2 if trajectory!='interrupted' else 1)
            if trajectory=='reference':reference_sha=saved['contract_sha256'];reference_fields={key:tree_sha(value) for key,value in saved.items() if key!='contract_sha256'}
            elif trajectory=='interrupted':interrupted_ref=_save_state(output/'INTERRUPTED_STEP1.pt',saved,torch=torch,maximum=launch['max_checkpoint_bytes'],counters=counters)
            else:
                _require(saved['contract_sha256']==reference_sha and {key:tree_sha(value) for key,value in saved.items() if key!='contract_sha256'}==reference_fields,'BIT_EXACT_FULL_RESUME')
                full_state=_save_state(output/'FINAL_PRIVATE_STATE.pt',saved,torch=torch,maximum=launch['max_checkpoint_bytes'],counters=counters);parity=True
            del saved,model,optimizer;gc.collect();torch.cuda.empty_cache()
        _require(counters['model_constructions']==counters['model_construction_returns']==counters['optimizer_constructions']==launch['max_model_constructions']==3
            and counters['forwards']==counters['forward_returns']==launch['max_forwards']==4
            and counters['backwards']==counters['backward_returns']==launch['max_backwards']==4
            and counters['optimizer_step_calls']==counters['optimizer_updates']==launch['max_private_updates']==4
            and counters['staged_state_load_calls']==counters['staged_state_load_returns']==3
            and counters['resume_state_load_calls']==counters['resume_state_load_returns']==1
            and counters['forbidden_legacy_calls']==0 and tree_sha(batches)==batch_sha,'EXACT_WORK_AND_RAW_BATCH_PRESERVATION')
        _freeze(launch,binding,context);unchanged=True
    except BaseException as exc:
        error=type(exc).__name__+':'+str(exc)
        try:_freeze(launch,binding,context);unchanged=True
        except Exception:unchanged=False
    finally:sys.setprofile(previous_profile)
    after=_io()
    if after['rchar']>context['design']['budget']['max_rchar_bytes']:error='RCHAR_BUDGET_EXCEEDED'
    passed=parity and unchanged and error is None
    outcome='COMPLETE' if passed else 'FAILED'
    result=_json(output/'RESULT.json',{'schema_version':'gx1_native_direction_research_private_resume_result_v1',
        'decision':'PASS_PRIVATE_NATIVE_DIRECTION_RESUME' if passed else 'FAILED','recipe':window['recipe'],'execution_binding':binding_ref,
        'counts':{key:value for key,value in counters.items() if key!='json_bytes'},'json_bytes_before_result':counters['json_bytes'],'counter_scope':'direct admitted caller boundaries; delegated loader internals not independently recounted',
        'initial_state':initial,'initial_cuda_identity_sha256':initial_identity,'final_state':full_state,'bit_exact_resume':parity,
        'steps':steps,'parent_order':list(PARENTS),'parent_order_sha256':order_sha,'parent_cursor':cursor,'physical_optimizer_updates':counters['optimizer_updates'],
        'source_unchanged':unchanged,'metadata_bookends':context['info']['metadata'],'payload_stat_bookends':context['ancestor_plan']['payloads'],
        'forbidden_open_paths':denied,'limits':limits,'elapsed_seconds':time.monotonic()-started,'io_before':before,'io_after':after,
        'rchar_delta':after['rchar']-before['rchar'],'rchar_scope':'proc_read_counter_excludes_mmap_fault_bytes',
        'all_three_horizons_supervised':passed,'ema_created':False,'scheduler_created':False,'target_model_created':False,
        'private_weights_for_main_fit':False,'predictive_value_qualified':False,'test_data_used':False,'error':error},counters,launch['max_published_json_bytes'])
    cursor_path=Path(window['campaign_cursor_path'])
    checkpoint=_json(cursor_path,authority.build_research_cursor(recipe_ref=window['recipe'],outcome=outcome,
        optimizer_updates=counters['optimizer_updates'],parent_cursor=cursor,parent_order_sha256=order_sha,full_state=full_state,result=result),counters,launch['max_published_json_bytes'])
    progress=authority.build_research_progress(window=window,plan=plan,invocation=invocation,checkpoint_pointer=checkpoint,
        optimizer_updates=counters['optimizer_updates'],parent_cursor=cursor,parent_order_sha256=order_sha,outcome=outcome)
    _json(progress_path,progress,counters,launch['max_published_json_bytes'])
    authority.require_research_progress(progress,plan_sha256=plan['plan_sha256'],invocation=invocation,expected_selection_receipt_sha256=None,verify_file=True)
    _freeze(launch,binding,context)
    if not passed:raise RuntimeError('NATIVE_DIRECTION_RESEARCH_WINDOW_FAILED:'+str(error))
    return {'decision':'PASS_PRIVATE_NATIVE_DIRECTION_RESUME','result':result,'progress_path':str(progress_path),'counts':counters,'checkpoint_pointer':checkpoint}
