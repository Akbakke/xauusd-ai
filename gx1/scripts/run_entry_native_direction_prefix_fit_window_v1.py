"""One guarded, resumable pure-direction prefix trajectory; no launch authority.

The canonical CUDA guard precedes strict context and numerical imports.  Immutable
267/264 providers supply raw physical inputs only.  Every prefix starts its own
combined ONLINE and measured0U baseline; private273 weights are never loaded.
The fixed shuffled prefix is consumed once, including missing/tied/immature rows.
Only the separate260 loss decides eligible cells.  Context tags are report-only.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

SOURCE_DESIGN = {'path':'/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_PREFIX_FIT_SOURCE_20261025_275/DESIGN.json',
    'sha256':'f73b5a516494ec82cab64e3d43526f337d10438dc11d46dcbb59f724c8c80e19'}
INPUT_QUALIFICATION = {'path':'/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_GC_BOOLEAN_PHYSICAL_PREFIX_NORMALIZATION_20261024_267/ACTUAL_QUALIFICATION_REVIEW.json',
    'sha256':'8b3e0b953e6e760e787f51e44bb8c701dd22e8e06ee2d7f934f50f86f5cf84c4'}
PRIVATE_QUALIFICATION = {'path':'/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_CANONICAL_PRIVATE_GPU_RESUME_20261024_273/ACTUAL_QUALIFICATION_REVIEW.json',
    'sha256':'59a2e9918ec7df392b483d840b07b48beeb01a35fb0220483b41bce301242d0a'}
MANIFEST_KEYS = frozenset(('schema_version','source_design','physical_provider_plan','physical_provider_worker',
    'prefix_normalization','constructor_metadata','view_source_bindings','targets','archive_members',
    'state_qualification','native_private_qualification','input_qualification'))
PRICE_MEMBERS = ('parent_rows','bar_start_ns','dynamic_entry_m1_row','decision_ns','terminal_time_ns',
    'elapsed_wall_seconds','terminal_source_row','observed','mid_move','mid_return_bps','sign')
TARGET_MEMBERS = ('terminal_time_ns','observed','sign')
SEED = 20260911
BATCH_SIZE = 16
MINUTE_NS = 60_000_000_000


def _require(ok, reason):
    if not ok:raise RuntimeError('NATIVE_DIRECTION_PREFIX_WINDOW_'+reason)


def _metadata(launch, common):
    metadata={};manifest=common._read(launch['input_manifest'],metadata)
    _require(set(manifest)==MANIFEST_KEYS and manifest['schema_version']=='gx1_native_direction_prefix_fit_input_manifest_v1'
        and manifest['source_design']==SOURCE_DESIGN and manifest['state_qualification']==common.STATE_QUALIFICATION
        and manifest['native_private_qualification']==PRIVATE_QUALIFICATION and manifest['input_qualification']==INPUT_QUALIFICATION
        and manifest['archive_members']==list(PRICE_MEMBERS),'INPUT_MANIFEST')
    design=common._read(SOURCE_DESIGN,metadata);fold=launch['fold']
    _require(fold in design['folds'],'SOURCE_DESIGN_FOLD')
    qualified=common._read(INPUT_QUALIFICATION,metadata);common._read(PRIVATE_QUALIFICATION,metadata)
    common._read(manifest['state_qualification'],metadata)
    item=manifest['prefix_normalization']
    _require(item in qualified['normalizations'] and item['fold_id']==fold['fold_id']
        and item['prefix_parent_rows']==fold['prefix_rows'] and item['cutoff_ns']==fold['train_cutoff_ns'],'QUALIFIED_PREFIX_RECORD')
    physical=common._module(manifest['physical_provider_worker']);ancestor=physical.provider_module()
    old_plan=common._read(manifest['physical_provider_plan'],metadata)
    _old_design,ancestral_plan,info=physical.validate_inputs(old_plan,ancestor)
    _require(manifest['targets']==ancestral_plan['axes']['source'] and old_plan['worker']==manifest['physical_provider_worker'],
        'SAME_PHYSICAL_AXES_AND_PROVIDER')
    fitted=common._read(item['output'],metadata);constructor=common._read(manifest['constructor_metadata'],metadata)
    _require(fitted['contract_sha256']==item['combined_contract_sha256']
        and fitted['base_normalization']['contract_sha256']==item['base_normalization_sha256']
        and fitted['gc_mtf_history']['layout']['contract_sha256']==item['gc_layout_sha256']
        and fitted['gc_mtf_history']['normalization']['contract_sha256']==item['gc_normalization_sha256'],'PREFIX_IDENTITIES')
    for ref in manifest['view_source_bindings'].values():
        if ref['path'].endswith('.json'):common._read(ref,metadata)
        else:_require(common._hash_file(ref['path'])==ref,'VIEW_SOURCE_SHA')
    info['metadata'].update(metadata)
    return {'manifest':manifest,'design':design,'physical':physical,'ancestor':ancestor,'ancestor_plan':ancestral_plan,
        'info':info,'fitted':fitted,'constructor':constructor}


def _qualify_axes_targets(axes, targets, fold, np):
    parents,starts,decisions=(axes[k] for k in ('parent_rows','bar_start_ns','decision_ns'))
    size=len(parents)
    _require(size==118249 and parents.dtype==starts.dtype==decisions.dtype==np.int64
        and parents.shape==starts.shape==decisions.shape==(size,) and np.all(parents[1:]>parents[:-1])
        and np.all(decisions[1:]>decisions[:-1]) and np.all(starts<=np.iinfo(np.int64).max-300_000_000_000)
        and np.array_equal(decisions,starts+300_000_000_000),'FULL_AXES')
    _require(int(np.searchsorted(decisions,fold['train_cutoff_ns'],side='left'))==fold['prefix_rows']
        and fold['oof_start_row']==fold['prefix_rows']
        and int(np.searchsorted(decisions,fold['oof_cutoff_ns'],side='left'))==fold['oof_end_row'],'FOLD_CLOCK_AND_ROW_COHORT')
    _require(set(targets)==set(TARGET_MEMBERS) and targets['terminal_time_ns'].dtype==np.int64
        and targets['terminal_time_ns'].shape==(size,3) and targets['observed'].dtype==np.bool_
        and targets['observed'].shape==(size,3) and targets['sign'].dtype==np.int8 and targets['sign'].shape==(size,3),
        'TARGET_HEADERS')
    _require(np.all(decisions>0) and np.all(decisions<=np.iinfo(np.int64).max-15*MINUTE_NS)
        and np.array_equal(targets['terminal_time_ns'],decisions[:,None]+np.asarray([1,5,15],np.int64)[None,:]*MINUTE_NS)
        and np.isin(targets['sign'],(-1,0,1)).all() and np.all(targets['sign'][~targets['observed']]==0),'TARGET_CLOCK_AND_DOMAIN')


def _fixed_order(parents, prefix_rows, seed, np):
    _require(type(prefix_rows) is int and 256<=prefix_rows<=len(parents) and seed==SEED,'ORDER_ARGUMENTS')
    return np.ascontiguousarray(parents[:prefix_rows][np.random.RandomState(seed).permutation(prefix_rows)],dtype=np.int64)


def _prepare(context, launch, np):
    from gx1.contracts import entry_native_direction_feature_view_v1 as view_owner
    from gx1.contracts import entry_native_direction_context_report_v1 as report_owner
    from gx1.features.htf_features import discard_multi_tf_v4_cache_pages
    physical,ancestor,plan,info=(context[k] for k in ('physical','ancestor','ancestor_plan','info'))
    prepared=physical.prepare_inputs(ancestor,plan,info,np)
    physical.release_pages(ancestor,prepared,discard_multi_tf_v4_cache_pages)
    manifest=context['manifest']
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    _require(prepared['gc']['layout']==common._read(manifest['view_source_bindings']['raw_gc_layout'],info['metadata'])
        and prepared['gc']['transition']==common._read(manifest['view_source_bindings']['raw_gc_transition'],info['metadata']),
        'CURRENT_QUALIFIED_RAW_GC_DOMAIN')
    with np.load(manifest['targets']['path'],allow_pickle=False) as archive:
        _require(archive.files==list(PRICE_MEMBERS),'EXACT_ELEVEN_ARCHIVE_NAMES')
        targets={key:archive[key] for key in TARGET_MEMBERS}
    _qualify_axes_targets(prepared['arrays'],targets,launch['fold'],np)
    for value in prepared['arrays'].values():value.setflags(write=False)
    for value in targets.values():value.setflags(write=False)
    for key in ('times_ns','sample_positions'):prepared['entry_sequence_source'][key].setflags(write=False)
    view=view_owner.bind_native_direction_feature_view(prefix_normalization=context['fitted'],
        prefix_normalization_ref=manifest['prefix_normalization']['output'],constructor_metadata=context['constructor'],
        constructor_metadata_ref=manifest['constructor_metadata'],source_bindings=manifest['view_source_bindings'],max_batch_rows=BATCH_SIZE)
    context_map=report_owner.native_direction_context_map(gc_layout=prepared['gc']['layout'],
        mtf_field_names=context['fitted']['base_normalization']['surfaces']['mtf_m5']['field_names'])
    order=_fixed_order(prepared['arrays']['parent_rows'],launch['fold']['prefix_rows'],SEED,np)
    return prepared,targets,view,context_map,order


def _batch(prepared, targets, view, selected, np):
    from gx1.contracts.entry_native_direction_feature_view_v1 import gather_native_direction_feature_batch
    raw=gather_native_direction_feature_batch(view,parent_rows=prepared['arrays']['parent_rows'],
        bar_start_ns=prepared['arrays']['bar_start_ns'],decision_ns=prepared['arrays']['decision_ns'],
        entry_sequence_source=prepared['entry_sequence_source'],snap_x=prepared['current']['snap'],
        ctx_cont=prepared['current']['ctx_cont'],ctx_cat=prepared['current']['ctx_cat'],multi_tf_sources=prepared['cache'],
        gc_surfaces=prepared['gc']['surfaces'],gc_layout=prepared['gc']['layout'],batch_parent_rows=selected)
    at=np.searchsorted(prepared['arrays']['parent_rows'],selected)
    _require(np.array_equal(prepared['arrays']['parent_rows'][at],selected),'TARGET_EXACT_PARENT_JOIN')
    target={key:np.ascontiguousarray(value[at]) for key,value in targets.items()}
    target['decision_ns']=np.ascontiguousarray(prepared['arrays']['decision_ns'][at])
    return raw,target


def _device_batch(raw, target, torch, device):
    return ({key:torch.from_numpy(value).to(device) for key,value in raw['features'].items()},
        {key:torch.from_numpy(value).to(device) for key,value in target.items()})


def _prefix_prior(axes, targets, fold, np):
    n=fold['prefix_rows'];eligible=targets['observed'][:n]&(targets['sign'][:n]!=0)&(targets['terminal_time_ns'][:n]<fold['train_cutoff_ns'])
    up=np.count_nonzero(eligible&(targets['sign'][:n]==1),axis=0);down=np.count_nonzero(eligible&(targets['sign'][:n]==-1),axis=0)
    _require(np.all(up>0) and np.all(down>0),'OWN_PREFIX_BOTH_CLASSES')
    return {'eligible_up':up.astype(np.int64).tolist(),'eligible_down':down.astype(np.int64).tolist(),
        'probability_up':(up/(up+down)).tolist(),'logit_up':np.log(up.astype(np.float64)/down).tolist(),
        'scope':'own mature observed nonzero prefix labels; no weights fit or OOF use'}


def _forward(model, features, *, kind, counters, ceiling, torch, grad=False):
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as hybrid
    counters['forward_calls']+=1;counters['forward_kind_calls'][kind]=counters['forward_kind_calls'].get(kind,0)+1
    _require(counters['forward_calls']<=ceiling,'DERIVED_FORWARD_CEILING')
    with trainer._training_mha_fastpath_context(model.backbone), (torch.enable_grad() if grad else torch.no_grad()):
        _require(not torch.backends.mha.get_fastpath_enabled() and not hybrid._BATCH_EQUAL_LENGTH_MTF_EVAL.get(),'CANONICAL_FP32_MHA')
        output=trainer._model_forward_fp32(model,**features)
    counters['forward_returns']+=1
    return output['logit_up']


def _online_identity(model, device, tree_sha, trainer):
    return {'model':tree_sha(model.state_dict()),'all_buffers':tree_sha(dict(model.named_buffers())),
        'rng':tree_sha(trainer._attended_session_rng_state(device=device)),
        'modes':{name:bool(module.training) for name,module in model.named_modules()}}


def _evaluate(model, prepared, targets, view, context_map, selected, *, cutoff, kind, counters, ceiling,
        device, torch, np, tree_sha, deadline=None, parity=False):
    from gx1.contracts import entry_native_direction_context_report_v1 as reporter
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    before=_online_identity(model,device,tree_sha,trainer);modes=[(module,module.training) for module in model.modules()]
    predictions=[];tags=[];decisions=[];sign=[];observed=[];terminal=[];starts=[];completed=0;parity_passed=not parity;context_header=None
    _require(all(p.grad is None for p in model.parameters()),'EVAL_PARTIAL_GRADIENT')
    try:
        model.eval()
        for left in range(0,len(selected),BATCH_SIZE):
            if deadline is not None and time.monotonic()>=deadline:break
            ids=selected[left:left+BATCH_SIZE];raw,target=_batch(prepared,targets,view,ids,np)
            context=reporter.tag_native_direction_contexts(raw['features'],context_map=context_map,witness=raw['witness'])
            header={key:value for key,value in context.items() if key not in ('parent_rows','decision_ns','tags')}
            if context_header is None:context_header=header
            else:_require(header==context_header,'CONTEXT_HEADER_PRESERVATION')
            _require(np.array_equal(context['parent_rows'],ids) and np.array_equal(context['decision_ns'],target['decision_ns']),
                'REPORT_CONTEXT_EXACT_INPUT_JOIN')
            features,_unused=_device_batch(raw,target,torch,device)
            logits=_forward(model,features,kind=kind,counters=counters,ceiling=ceiling,torch=torch)
            if parity and left==0:
                with_grad=_forward(model,features,kind='INITIAL_PARITY_GRAD',counters=counters,ceiling=ceiling,torch=torch,grad=True)
                _require(torch.equal(logits.detach(),with_grad.detach()) and all(p.grad is None for p in model.parameters()),
                    'OWN_INITIAL_GRAD_NOGRAD_LOGIT_PARITY')
                del with_grad;parity_passed=True
            predictions.append(logits.detach().cpu().numpy().copy());tags.append(context['tags'])
            decisions.append(target['decision_ns']);sign.append(target['sign']);observed.append(target['observed']);terminal.append(target['terminal_time_ns'])
            starts.append(np.asarray(raw['witness']['bar_start_ns'],np.int64));completed+=len(ids)
            del logits,features,_unused,raw,target,context
    finally:
        for module,training in modes:module.training=training
        _require(_online_identity(model,device,tree_sha,trainer)==before and all(p.grad is None for p in model.parameters()),
            'EVAL_ONLINE_BUFFERS_RNG_MODES_UNCHANGED')
    if completed!=len(selected):return None,completed
    arrays={'parent_rows':np.ascontiguousarray(selected),'bar_start_ns':np.concatenate(starts),
        'decision_ns':np.concatenate(decisions),'terminal_time_ns':np.concatenate(terminal),
        'observed':np.concatenate(observed),'sign':np.concatenate(sign),'logit_up':np.concatenate(predictions)}
    tag_arrays={key:np.concatenate([item[key] for item in tags]) for key in tags[0]}
    context={**context_header,'parent_rows':arrays['parent_rows'],
        'decision_ns':arrays['decision_ns'],'tags':tag_arrays}
    report=reporter.score_native_direction_logits(arrays['logit_up'],sign=arrays['sign'],observed=arrays['observed'],
        decision_ns=arrays['decision_ns'],terminal_time_ns=arrays['terminal_time_ns'],cutoff_ns=cutoff,context_tags=context)
    arrays.update({'context__'+key:value for key,value in tag_arrays.items()})
    return {'arrays':arrays,'report':report,'context_tags':context,'state_preservation':before,'initial_parity_passed':parity_passed},completed


def _save_predictions(path, arrays, *, np, common, tree_sha):
    path=Path(path);_require(not path.exists() and not path.is_symlink(),'PREDICTION_EXISTS')
    _require(arrays and all(isinstance(a,np.ndarray) and a.flags.c_contiguous and a.dtype.kind in 'biuf'
        and np.isfinite(a).all() for a in arrays.values()),'PREDICTION_ARRAY_DOMAIN')
    fd,temporary=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:np.savez(handle,**arrays);handle.flush();os.fsync(handle.fileno())
        before=common._stat(temporary);staged=common._hash_file(temporary)
        with np.load(temporary,allow_pickle=False) as archive:
            _require(archive.files==list(arrays) and tree_sha({k:archive[k] for k in archive.files})==tree_sha(arrays),
                'PREDICTION_STAGED_ROUNDTRIP')
        _require(common._stat(temporary)==before and common._hash_file(temporary)==staged and before['st_size']<=sum(a.nbytes for a in arrays.values())+65536,
            'PREDICTION_STAGED_BOUND')
        os.link(temporary,path);path.chmod(0o444)
        from gx1.models.entry_v10.entry_v10_ctx_train_v3 import _fsync_directory
        _fsync_directory(path.parent)
    finally:
        if Path(temporary).exists():Path(temporary).unlink()
    return common._hash_file(path)


def _publish_measurement(name, measured, *, output, view, prior, own_initial_state, fresh_identity,
        context_map, counters, launch, np, common, tree_sha):
    from gx1.contracts.entry_native_direction_context_report_v1 import score_native_direction_logits
    arrays=measured['arrays'];prediction=_save_predictions(output/(name+'.npz'),arrays,np=np,common=common,tree_sha=tree_sha)
    prior_logits=np.broadcast_to(np.asarray(prior['logit_up'],np.float64),(len(arrays['parent_rows']),3)).copy()
    prior_report=score_native_direction_logits(prior_logits,sign=arrays['sign'],observed=arrays['observed'],
        decision_ns=arrays['decision_ns'],terminal_time_ns=arrays['terminal_time_ns'],
        cutoff_ns=launch['fold']['train_cutoff_ns'] if name!='OOF' else launch['fold']['oof_cutoff_ns'],context_tags=measured['context_tags'])
    return common._json(output/(name+'.json'),{'schema_version':'gx1_native_direction_prefix_measurement_v1',
        'role':name,'fold':launch['fold'],'feature_view_identity':view,'context_map':context_map,
        'predictions':prediction,'model_metrics':measured['report'],'own_prefix_prior':prior,'prior_metrics':prior_report,
        'own_initial_state':own_initial_state,'fresh_identity_sha256':fresh_identity,'preservation':measured['state_preservation'],
        'initial_grad_nograd_parity':measured['initial_parity_passed'],'test_data_used':False,
        'scope':'reused TRAIN development periods; binary direction conditional observed nonzero; no trading or generalization authority'},
        counters,launch['max_published_json_bytes'])


def _gradient_summary(model, torch):
    _require(all(p.grad is None for p in model.parameters() if not p.requires_grad),'INACTIVE_GRADIENT')
    _require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'FINITE_GRADIENT')
    head=model.direction_head
    out={'head_weight_rows':[int(torch.count_nonzero(row)) for row in head.weight.grad],
        'head_bias_rows':[int(value!=0) for value in head.bias.grad]}
    for tf in ('m5','m15','h1','h4','d1'):
        for label,modules in (('gc_projection',model.backbone.gc_mtf_family_proj),('gc_gate',model.backbone.gc_mtf_feature_context_gate)):
            out[label+'_'+tf]=sum(int(torch.count_nonzero(p.grad)) for key in modules if key.startswith(tf+'__')
                for p in modules[key].parameters() if p.grad is not None)
    return out


def _train_batch(model, optimizer, features, target, *, cutoff, counters, ceiling, torch, diagnostics):
    from gx1.contracts.entry_native_direction_research_v1 import native_direction_binary_loss
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    _require(model.training and all(module.training for module in model.modules()),'TRAIN_MODE')
    logits=_forward(model,features,kind='TRAIN',counters=counters,ceiling=ceiling,torch=torch,grad=True)
    loss=native_direction_binary_loss(logits,**target,cutoff_ns=cutoff)
    record={'eligible_cells':loss['counts']['eligible_cells'],'counts':loss['counts'],'per_horizon_counts':loss['per_horizon_counts']}
    if not loss['update_allowed']:
        _require(loss['loss'] is None and all(p.grad is None for p in model.parameters()),'EMPTY_BATCH_NO_UPDATE')
        counters['empty_batches']+=1;return record,False
    counters['backward_calls']+=1;loss['loss'].backward();counters['backward_returns']+=1
    if diagnostics:record['gradient_nonzero_counts']=_gradient_summary(model,torch)
    counters['optimizer_step_calls']+=1
    norm=trainer._optimizer_step_with_finite_gradients(model=model,optimizer=optimizer)
    counters['optimizer_step_returns']+=1
    record.update(loss=float(loss['loss'].detach()),gradient_norm=float(norm));return record,True


def _save_snapshot(model, optimizer, *, context, recipe, order, parent_cursor, updates, state_directory,
        name, launch, counters, torch, device, common, state_owner):
    used=sum(common._stat(p)['st_size'] for p in state_directory.glob('*.pt'))
    remaining=launch['max_checkpoint_total_bytes']-used
    _require(remaining>0,'PREFIX_CHECKPOINT_TOTAL_BUDGET')
    state=state_owner.capture_native_direction_research_state(model,optimizer,recipe=recipe,
        feature_view_identity=model.feature_view_identity,parent_order=order,cursor=parent_cursor,optimizer_updates=updates,device=device)
    ref=common._save_state(state_directory/(name+'.pt'),state,torch=torch,
        maximum=min(launch['max_checkpoint_bytes'],remaining),counters=counters)
    seal=state['contract_sha256'];del state
    return ref,seal


def _latest_cursor(path, value, *, common, counters, limit):
    from gx1.models.entry_v10.entry_v10_ctx_train_v3 import _attended_session_atomic_write_json, _fsync_directory
    data=(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)+'\n').encode()
    _require(counters['json_bytes']+len(data)<=limit,'PREFIX_JSON_BUDGET')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    _attended_session_atomic_write_json(path,value);_fsync_directory(path.parent);counters['json_bytes']+=len(data)
    return common._hash_file(path)


def run_native_direction_prefix_fit_window(*, policy_path: Path, policy_file_sha256: str, progress_path: Path) -> dict:
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    limits=require_guarded_cuda_trainer_execution()
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as authority
    window,plan,invocation,marker,launch=authority.require_native_direction_prefix_fit_context(policy_path,policy_file_sha256,progress_path)
    sys.dont_write_bytecode=True
    import numpy as np
    import torch
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.contracts.entry_model_native_train_recipe_v1 import require_native_direction_research_recipe
    from gx1.contracts import entry_native_direction_research_state_v1 as state_owner
    from gx1.contracts.entry_native_direction_feature_view_v1 import build_native_direction_research_model
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.features.htf_features import discard_multi_tf_v4_cache_pages
    from gx1.scripts.run_unified_exit_random_access_full_train_v1 import _dynamic_m5_technical_tree_sha256 as tree_sha
    recipe=require_native_direction_research_recipe(launch['numerical_recipe']);fold=launch['fold'];number=window['invocation_number']
    binding_ref=common._hash_file(launch['execution_binding_path']);extra={};binding=common._read(binding_ref,extra)
    context=_metadata(launch,common);context['info']['metadata'].update(extra)
    for ref in (window['recipe'],binding['plan'],{'path':str(policy_path),'sha256':policy_file_sha256}):common._read(ref,context['info']['metadata'])
    common._freeze(launch,binding,context)
    output=Path(window['output_directory']);output.mkdir(parents=True,exist_ok=True)
    work=output/f'window-{number:04d}';work.mkdir();state_directory=output/'states';state_directory.mkdir(exist_ok=True)
    denied=common._install_read_guard(context,output,Path(launch['runtime_root']))
    counters={k:0 for k in ('model_constructions','model_construction_returns','optimizer_constructions','forward_calls','forward_returns',
        'backward_calls','backward_returns','optimizer_step_calls','optimizer_step_returns','empty_batches',
        'staged_state_load_calls','staged_state_load_returns','resume_state_load_calls','resume_state_load_returns','forbidden_legacy_calls')}
    counters.update(json_bytes=sum(common._stat(p)['st_size'] for p in output.rglob('*.json'))
        +sum(common._stat(p)['st_size'] for folder in ('prefix-cursor','progress') for p in (Path(launch['runtime_root'])/folder).glob('*.json')),forward_kind_calls={})
    started=time.monotonic();deadline=started+window['max_invocation_seconds'];before=common._io();device=torch.device('cuda',0)
    initial_ref=initial_measurement=final_measurement=full_state=None;view_identity=None;parent_cursor=updates=oof_cursor=0
    phase='TRAINING';outcome='FAILED';error=None;source_unchanged=False;checkpoint_refs=[];first_diagnostic=last_diagnostic=None
    prepared=model=optimizer=None;order_sha='0'*64;prior_cursor=None;fresh_identity=None;prior=None;initial_inactive=None
    ceiling=(fold['prefix_rows']+15)//16+33+(fold['oof_end_row']-fold['oof_start_row']+15)//16
    durable=None;observed_progress=None
    common._json(work/'CLAIM.json',{'recipe':window['recipe'],'fold':fold,'invocation_number':number,
        'source_head':binding['source_head'],'plan_sha256':plan['plan_sha256'],'limits':limits},counters,launch['max_published_json_bytes'])
    previous_profile=sys.getprofile()
    try:
        sys.setprofile(common._deny_legacy_work(previous_profile,counters))
        prepared,targets,view,context_map,ordered=_prepare(context,launch,np)
        order=torch.from_numpy(ordered.copy());order_sha=tree_sha(order)
        prior=_prefix_prior(prepared['arrays'],targets,fold,np)
        if number>1:
            cursor_ref={'path':window['campaign_cursor_path'],'sha256':marker['pointer_before_sha256']}
            saved_cursor=json.loads(Path(cursor_ref['path']).read_text())
            _require(common._hash_file(cursor_ref['path'])==cursor_ref,'PREVIOUS_CURSOR_SHA')
            prior_cursor=authority.require_prefix_fit_cursor(saved_cursor,expected_recipe=window['recipe'],expected_fold=fold,verify_files=True)
            _require(prior_cursor['parent_order_sha256']==order_sha,'SAME_FIXED_PERMUTATION')
        trainer._set_deterministic(recipe['seed'],device,recipe['precision_policy'])
        counters['model_constructions']+=1
        model=build_native_direction_research_model(view,prefix_normalization=context['fitted'],constructor_metadata=context['constructor'],device=device)
        counters['model_construction_returns']+=1;model.train()
        optimizer,_optimizer_binding=state_owner.build_native_direction_research_optimizer(model,recipe=recipe);counters['optimizer_constructions']+=1
        view_identity=model.feature_view_identity
        fresh=state_owner.capture_native_direction_research_state(model,optimizer,recipe=recipe,feature_view_identity=view_identity,
            parent_order=order,cursor=0,optimizer_updates=0,device=device)
        fresh_identity=tree_sha({key:fresh[key] for key in ('binding','model_state','buffer_state','optimizer_state','rng_state','module_training')})
        initial_inactive=tree_sha({name:p for name,p in model.named_parameters() if not p.requires_grad});del fresh
        if prior_cursor is None:
            initial_ref,_seal=_save_snapshot(model,optimizer,context=context,recipe=recipe,order=order,parent_cursor=0,updates=0,
                state_directory=state_directory,name='INITIAL_0U',launch=launch,counters=counters,torch=torch,device=device,common=common,state_owner=state_owner)
            measured,_=_evaluate(model,prepared,targets,view,context_map,ordered[:256],cutoff=fold['train_cutoff_ns'],kind='INITIAL_TRAIN',
                counters=counters,ceiling=ceiling,device=device,torch=torch,np=np,tree_sha=tree_sha,parity=True)
            initial_measurement=_publish_measurement('INITIAL_TRAIN',measured,output=work,view=view_identity,prior=prior,
                own_initial_state=initial_ref,fresh_identity=fresh_identity,context_map=context_map,counters=counters,launch=launch,np=np,common=common,tree_sha=tree_sha)
            del measured
        else:
            _require(prior_cursor['feature_view_identity']==view_identity,'RESUME_VIEW_IDENTITY')
            initial_ref=prior_cursor['own_initial_state'];initial_measurement=prior_cursor['initial_measurement']
            original=common._read(initial_measurement,context['info']['metadata'])
            _require(original['fresh_identity_sha256']==fresh_identity and original['own_prefix_prior']==prior,'SAME_OWN_FRESH_BASELINE')
            loaded=common._load_state(prior_cursor['full_state'],torch=torch,maximum=launch['max_checkpoint_bytes'],counters=counters)
            restored=state_owner.restore_native_direction_research_state(loaded,model,optimizer,recipe=recipe,
                feature_view_identity=view_identity,parent_order=order,device=device)
            _require(restored['cursor']==prior_cursor['parent_cursor'] and restored['optimizer_updates']==prior_cursor['optimizer_updates'],
                'RESTORED_PROGRESS')
            restored_state=state_owner.capture_native_direction_research_state(model,optimizer,recipe=recipe,feature_view_identity=view_identity,
                parent_order=order,cursor=restored['cursor'],optimizer_updates=restored['optimizer_updates'],device=device)
            _require(restored_state['contract_sha256']==loaded['contract_sha256'],'EXACT_SERIALIZED_FULL_RESUME');del restored_state
            del loaded;parent_cursor=restored['cursor'];updates=restored['optimizer_updates'];phase=prior_cursor['phase_state']
            final_measurement=prior_cursor['final_train_measurement'];full_state=prior_cursor['full_state']
        durable={'state':full_state or initial_ref,'parent_cursor':parent_cursor,'optimizer_updates':updates}
        while parent_cursor<len(ordered) and time.monotonic()<deadline:
            right=min(parent_cursor+BATCH_SIZE,len(ordered));raw,target=_batch(prepared,targets,view,ordered[parent_cursor:right],np)
            features,device_target=_device_batch(raw,target,torch,device)
            diagnostic=(counters['optimizer_step_returns']==0 or right==len(ordered))
            record,did_update=_train_batch(model,optimizer,features,device_target,cutoff=fold['train_cutoff_ns'],
                counters=counters,ceiling=ceiling,torch=torch,diagnostics=diagnostic)
            parent_cursor=right;updates+=int(did_update)
            if did_update and first_diagnostic is None:first_diagnostic={'next_parent_cursor':parent_cursor,**record}
            if right==len(ordered):last_diagnostic={'next_parent_cursor':parent_cursor,**record}
            _require(updates<=recipe['max_updates'],'PREFIX_UPDATE_CEILING')
            del raw,target,features,device_target
            if did_update and updates%launch['checkpoint_every_updates']==0:
                ref,seal=_save_snapshot(model,optimizer,context=context,recipe=recipe,order=order,parent_cursor=parent_cursor,updates=updates,
                    state_directory=state_directory,name=f'w{number:04d}_u{updates:06d}_p{parent_cursor:06d}',launch=launch,
                    counters=counters,torch=torch,device=device,common=common,state_owner=state_owner)
                durable={'state':ref,'contract_sha256':seal,'optimizer_updates':updates,'parent_cursor':parent_cursor}
                checkpoint_refs.append(durable)
                common._json(work/f'CHECKPOINT_u{updates:06d}_p{parent_cursor:06d}.json',{'recipe':window['recipe'],'fold':fold,
                    'feature_view_identity':view_identity,'parent_order_sha256':order_sha,**durable},counters,launch['max_published_json_bytes'])
        if parent_cursor==len(ordered):phase='SCORING'
        if phase=='SCORING' and final_measurement is None and time.monotonic()<deadline:
            measured,_=_evaluate(model,prepared,targets,view,context_map,ordered[:256],cutoff=fold['train_cutoff_ns'],kind='FINAL_TRAIN',
                counters=counters,ceiling=ceiling,device=device,torch=torch,np=np,tree_sha=tree_sha)
            final_measurement=_publish_measurement('FINAL_TRAIN',measured,output=work,view=view_identity,prior=prior,
                own_initial_state=initial_ref,fresh_identity=fresh_identity,context_map=context_map,counters=counters,launch=launch,np=np,common=common,tree_sha=tree_sha)
            del measured
        oof_ref=None
        if phase=='SCORING' and final_measurement is not None:
            selected=prepared['arrays']['parent_rows'][fold['oof_start_row']:fold['oof_end_row']]
            measured,oof_cursor=_evaluate(model,prepared,targets,view,context_map,selected,cutoff=fold['oof_cutoff_ns'],kind='OOF',
                counters=counters,ceiling=ceiling,device=device,torch=torch,np=np,tree_sha=tree_sha,deadline=deadline)
            if measured is not None:
                oof_ref=_publish_measurement('OOF',measured,output=work,view=view_identity,prior=prior,own_initial_state=initial_ref,
                    fresh_identity=fresh_identity,context_map=context_map,counters=counters,launch=launch,np=np,common=common,tree_sha=tree_sha)
                del measured;phase='COMPLETE'
        _require(tree_sha({name:p for name,p in model.named_parameters() if not p.requires_grad})==initial_inactive,'INACTIVE_WEIGHTS_UNCHANGED')
        full_state,seal=_save_snapshot(model,optimizer,context=context,recipe=recipe,order=order,parent_cursor=parent_cursor,updates=updates,
            state_directory=state_directory,name=f'w{number:04d}_TERMINAL_u{updates:06d}_p{parent_cursor:06d}',launch=launch,
            counters=counters,torch=torch,device=device,common=common,state_owner=state_owner)
        durable={'state':full_state,'contract_sha256':seal,'optimizer_updates':updates,'parent_cursor':parent_cursor};checkpoint_refs.append(durable)
        _require(counters['model_constructions']==counters['model_construction_returns']==counters['optimizer_constructions']==1
            and counters['forward_calls']==counters['forward_returns'] and counters['backward_calls']==counters['backward_returns']
            ==counters['optimizer_step_calls']==counters['optimizer_step_returns'] and counters['forbidden_legacy_calls']==0,'ACTUAL_CALL_RETURNS')
        common._freeze(launch,binding,context);source_unchanged=True
        outcome='COMPLETE' if phase=='COMPLETE' else 'RESUMABLE'
    except BaseException as exc:
        error=type(exc).__name__+':'+str(exc)
        observed_progress={'parent_cursor':parent_cursor,'optimizer_updates':updates,'oof_cursor':oof_cursor}
        if durable is not None:
            full_state=durable['state'];parent_cursor=durable['parent_cursor'];updates=durable['optimizer_updates']
            phase='SCORING' if parent_cursor==fold['prefix_rows'] else 'TRAINING'
            if phase=='TRAINING':final_measurement=None;oof_cursor=0
        else:parent_cursor=updates=oof_cursor=0;phase='TRAINING';full_state=initial_ref
        try:common._freeze(launch,binding,context);source_unchanged=True
        except Exception:source_unchanged=False
    finally:
        sys.setprofile(previous_profile)
        if prepared is not None:context['physical'].release_pages(context['ancestor'],prepared,discard_multi_tf_v4_cache_pages)
    after=common._io()
    if after['rchar']>context['ancestor_plan']['budget']['max_rchar_bytes']:
        outcome='FAILED';error=(error+';' if error else '')+'INHERITED_PHYSICAL_RCHAR_CEILING'
        if phase=='COMPLETE':phase='SCORING'
    result=common._json(work/'RESULT.json',{'schema_version':'gx1_native_direction_prefix_window_result_v1','outcome':outcome,'phase_state':phase,
        'recipe':window['recipe'],'fold':fold,'feature_view_identity':view_identity,'parent_order_sha256':order_sha,
        'parent_cursor':parent_cursor,'optimizer_updates':updates,'oof_cursor':oof_cursor,'own_initial_state':initial_ref,
        'initial_measurement':initial_measurement,'final_train_measurement':final_measurement,'oof_measurement':locals().get('oof_ref'),
        'full_state':full_state,'durable_checkpoint':durable,'observed_progress_before_failure':observed_progress,'checkpoints':checkpoint_refs,'first_update_diagnostic':first_diagnostic,'last_batch_diagnostic':last_diagnostic,
        'counts':{k:v for k,v in counters.items() if k!='json_bytes'},'counter_scope':'actual admitted caller boundaries; no independent delegated-loader callback claim',
        'source_unchanged':source_unchanged,'metadata_bookends':context['info']['metadata'],'payload_stat_bookends':context['ancestor_plan']['payloads'],
        'forbidden_open_paths':denied,'limits':limits,'elapsed_seconds':time.monotonic()-started,'io_before':before,'io_after':after,
        'rchar_delta':after['rchar']-before['rchar'],'rchar_scope':'process reads; mmap fault bytes excluded','error':error,
        'test_data_used':False,'qualification_weights_reused':False,'predictive_value_qualified':False},counters,launch['max_published_json_bytes'])
    cursor=authority.build_prefix_fit_cursor(recipe_ref=window['recipe'],fold=fold,feature_view_identity=view_identity,phase_state=phase,outcome=outcome,
        optimizer_updates=updates,parent_cursor=parent_cursor,parent_order_sha256=order_sha,full_state=full_state,own_initial_state=initial_ref,
        initial_measurement=initial_measurement,final_train_measurement=final_measurement,oof_cursor=oof_cursor,result=result)
    common._json(work/'CURSOR.json',cursor,counters,launch['max_published_json_bytes'])
    pointer=_latest_cursor(window['campaign_cursor_path'],cursor,common=common,counters=counters,limit=launch['max_published_json_bytes'])
    progress=authority.build_prefix_fit_progress(window=window,plan=plan,invocation=invocation,checkpoint_pointer=pointer,cursor=cursor)
    common._json(progress_path,progress,counters,launch['max_published_json_bytes'])
    authority.require_prefix_fit_progress(progress,plan_sha256=plan['plan_sha256'],invocation=invocation,
        expected_selection_receipt_sha256=None,verify_file=True)
    common._freeze(launch,binding,context)
    if error is not None:raise RuntimeError('NATIVE_DIRECTION_PREFIX_WINDOW_FAILED:'+error)
    return {'decision':outcome,'result':result,'checkpoint_pointer':pointer,'counts':counters,'progress_path':str(progress_path)}
