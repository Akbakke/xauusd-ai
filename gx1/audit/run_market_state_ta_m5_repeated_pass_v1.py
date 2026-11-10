"""Bounded E1 -> E2 -> E3 continuation on the same 4096 intact TRAIN rows.

Only a paired CONTROL evidence capsule may start this diagnostic. AdamW and RNG
continue exactly; the resulting private capsule grants no canonical resume.
"""
from __future__ import annotations
import copy
import gc
import io
from pathlib import Path
import subprocess
import time
from gx1.audit import run_market_state_ta_m5_paired_batch_order_v1 as paired
from gx1.audit import trace_market_state_ta_m5_original_cuda_prefix_v1 as original_trace
from gx1.contracts import market_state_native_F_private_campaign_v1 as scope

TASKS=('state','future')
CAPSULE_SCHEMA='gx1_ta_m5_repeated_pass_evidence_v1'
CONTROL_KEYS=frozenset(('schema_version','arm','recipe','source_commit','original_checkpoint',
    'canonical_resume_authority','model_state','buffer_state','optimizer_state','rng_state','module_training',
    'parent_order','parent_cursor','physical_optimizer_updates','pending_gradients','parameter_binding',
    'original_numerical_recipe_reference','method_state','model_state_sha256','buffer_state_sha256',
    'optimizer_state_sha256','rng_state_sha256','test_data_used','evidence_sha256'))


def _need(ok,reason):
    if not ok:raise RuntimeError('TA_M5_REPEATED_PASS_'+reason)


def restore_control_evidence(saved,model,optimizer,*,recipe,origin,parent_order,feature_view_identity,device):
    """Validate the complete private E1 capsule before any state mutation."""
    import torch
    from gx1.contracts import entry_native_direction_research_state_v1 as old
    from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as model_sha
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.scripts.run_unified_exit_random_access_full_train_v1 import _dynamic_m5_technical_tree_sha256 as tree_sha
    numerical=state._numerical(recipe['numerical_recipe'],model)
    old._device(model,device);old._no_partial_gradients(model)
    binding=state._binding(model,optimizer,numerical,feature_view_identity)
    order=old._order(parent_order);pairs,_,_=state._parameters(model)
    parameter_ids={name:id(p) for name,p in pairs}
    expected=origin['paired_result']['arm_progress']['CONTROL']
    _need(isinstance(saved,dict) and set(saved)==CONTROL_KEYS
          and saved['schema_version']==paired.CAPSULE_SCHEMA and saved['arm']=='CONTROL'
          and saved['recipe']==recipe['input_refs']['paired_recipe']
          and saved['source_commit']==origin['paired_result']['source_commit']
          and saved['original_checkpoint']==origin['paired_recipe']['input_refs']['original_initial_checkpoint']
          and saved['canonical_resume_authority'] is False and saved['test_data_used'] is False
          and type(saved['parent_cursor']) is int and saved['parent_cursor']==4096
          and type(saved['physical_optimizer_updates']) is int and saved['physical_optimizer_updates']==256
          and type(saved['pending_gradients']) is int and saved['pending_gradients']==0,
          'CONTROL_CAPSULE_IDENTITY')
    _need(saved['evidence_sha256']==tree_sha({k:v for k,v in saved.items() if k!='evidence_sha256'}),
          'CONTROL_EVIDENCE_SEAL')
    _need(saved['parameter_binding']==binding and saved['original_numerical_recipe_reference']==numerical,
          'CONTROL_BINDING')
    _need(isinstance(saved['parent_order'],torch.Tensor) and saved['parent_order'].dtype==torch.int64
          and torch.equal(saved['parent_order'].cpu(),order)
          and tree_sha(order)==expected['parent_order_sha256'],'CONTROL_FULL_ORDER')
    indices=origin['paired_recipe']['arm_batch_indices']['CONTROL']
    _need(saved['method_state']==dict(arm_batch_indices=indices,
        batch_order_sha256=expected['batch_order_sha256'],full_order_sha256=expected['parent_order_sha256'],
        original_fit_batch_starts=[16*b for b in indices],unselected_suffix_consumed=False,
        loss_and_optimizer_unchanged=True),'CONTROL_METHOD')
    old._tensor_map(saved['model_state'],model.state_dict(),'CONTROL_MODEL')
    old._tensor_map(saved['buffer_state'],dict(model.named_buffers()),'CONTROL_BUFFERS')
    for name,value in saved['buffer_state'].items():
        if name in saved['model_state']:
            _need(torch.equal(value.cpu(),saved['model_state'][name].cpu()),'CONTROL_PERSISTENT_BUFFER:'+name)
    immutable=frozenset({'native_F_objective_sha256'}|
        {'state_model.'+name for name in old.IMMUTABLE_BUFFER_NAMES|state._QUERY_BUFFERS})
    prefixes=tuple('state_model.'+prefix for prefix in old.IMMUTABLE_BUFFER_PREFIXES)
    for name,value in model.named_buffers():
        if name in immutable or name.startswith(prefixes):
            _need(torch.equal(saved['buffer_state'][name].cpu(),value.cpu()),'CONTROL_IMMUTABLE_BUFFER:'+name)
    modules=dict(model.named_modules());modes=saved['module_training']
    _need(isinstance(modes,dict) and set(modes)==set(modules)
          and all(type(v) is bool for v in modes.values()),'CONTROL_MODULE_MODES')
    old._optimizer_state(saved['optimizer_state'],optimizer,model,256)
    old._rng(saved['rng_state'],device=device,seed=numerical['seed'])
    for key in ('model_state','buffer_state','optimizer_state','rng_state'):
        digest=model_sha if key=='model_state' else tree_sha
        _need(digest(saved[key])==saved[key+'_sha256'],'CONTROL_DIGEST:'+key)
    _need(saved['model_state_sha256']==origin['measurements']['CONTROL_TRAIN']['evaluated_model_state_sha256']
          ==origin['measurements']['CONTROL_LATER']['evaluated_model_state_sha256'],'CONTROL_MODEL_IDENTITY')
    # Every payload/binding check above precedes the first copy. No rollback is claimed.
    model.load_state_dict(saved['model_state'],strict=True)
    with torch.no_grad():
        for name,value in model.named_buffers():value.copy_(saved['buffer_state'][name].to(device))
    optimizer.load_state_dict(copy.deepcopy(saved['optimizer_state']))
    for name,value in modes.items():modules[name].training=value
    model.backbone.require_input_normalization_state();state._require_model_binding(model)
    trainer._restore_attended_session_rng_state(saved['rng_state'],device=device)
    actual=dict(model_state=copy.deepcopy(model.state_dict()),
        buffer_state=copy.deepcopy(dict(model.named_buffers())),
        optimizer_state=copy.deepcopy(optimizer.state_dict()),
        rng_state=copy.deepcopy(trainer._attended_session_rng_state(device=device)))
    digests={}
    for key,value in actual.items():
        digest=model_sha if key=='model_state' else tree_sha
        digests[key+'_sha256']=digest(value)
        _need(digests[key+'_sha256']==saved[key+'_sha256'],'RESTORED_EXACT:'+key)
    _need({n:bool(m.training) for n,m in model.named_modules()}==modes
          and {n:id(p) for n,p in state._parameters(model)[0]}==parameter_ids
          and state._binding(model,optimizer,numerical,feature_view_identity)==binding
          and all(p.grad is None for _,p in pairs),'RESTORED_MODES_PARAMETERS_BINDING')
    return dict(all_numerical_state_exact=True,**digests,
        module_training_sha256=tree_sha(modes),parent_order_sha256=tree_sha(order),
        parameter_binding_sha256=binding['contract_sha256'],parameter_objects_preserved=True,
        optimizer_parameter_states=len(saved['optimizer_state']['state']),
        source_commit=saved['source_commit'],source_evidence=recipe['input_refs']['control_evidence'],
        canonical_resume_authority=False)


def compare_fixed_pass(measurements,epoch,*,baseline_score_atol):
    """E2 is descriptive; E3 is primary, independently of either score sign."""
    _need(epoch in ('E2','E3'),'FIXED_MEASUREMENT_PASS')
    comparison={label+'_'+panel:measurements[(epoch if label=='INTERLEAVED' else label)+'_'+panel]
                for panel in ('TRAIN','LATER') for label in ('INITIAL','CONTROL','INTERLEAVED')}
    gate=paired.compare_m5_measurements(comparison,baseline_score_atol=baseline_score_atol)
    for cell in gate['cells'].values():cell[epoch]=cell.pop('INTERLEAVED')
    gate.update(candidate=epoch,CONTROL='cached_E1',primary=epoch=='E3',checkpoint_selection=False)
    return gate


def preflight_read_only(recipe,*,source_commit=None):
    """Exact metadata and feature-view checks; no genuine arrays or model."""
    from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero
    from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
    if source_commit is None:
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=scope.CANONICAL_REPO,text=True).strip()
    scope.require_ta_m5_repeated_pass_recipe(recipe,source_repo=scope.CANONICAL_REPO,
                                            source_commit=source_commit,verify_files=True)
    origin=scope.read_ta_m5_repeated_pass_origin(recipe)
    receipts=zero.load_ta_input_receipts(current_source_owners={k:v['sha256'] for k,v in recipe['source_owners'].items()})
    view=zero._bind_ta_view(receipts,dedicated_m5_side=True)
    ref=origin['original_recipe']['input_refs']['ta_candidate_view']
    _need(view==scope.campaign.read_bound_json(Path(ref['path']),ref['sha256'])
          and view['contract_sha256']==recipe['expected_view_sha256'],'ORIGINAL_VIEW')
    ref=recipe['input_refs']['endpoint_priors']
    priors=scope.campaign.read_bound_json(Path(ref['path']),ref['sha256'])
    objective.require_endpoint_priors(priors,fold_id=1,ta_only=True,dedicated_m5_side=True)
    _need(recipe['numerical_recipe']==origin['paired_recipe']['numerical_recipe']
          ==origin['original_recipe']['numerical_recipe']
          and recipe['numerical_recipe']['grad_accum_steps']==1
          and recipe['numerical_recipe']['grad_clip_norm']==recipe['numerical_recipe']['future_weight']==1.
          and recipe['numerical_recipe']['m5_equal_observation'] is True,'UNCHANGED_NUMERICAL')
    synthetic=paired._synthetic_checks(priors,origin['paired_recipe'])
    return dict(origin=origin,receipts=receipts,view=view,priors=priors,synthetic=synthetic)


def run_repeated_pass(*,policy_path:Path,policy_file_sha256:str,progress_path:Path)->dict:
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    limits=require_guarded_cuda_trainer_execution()
    window,plan,invocation,marker,recipe=authority.require_native_direction_research_context(
        policy_path,policy_file_sha256,progress_path)
    _need(recipe['run_scope']==scope.TA_M5_REPEATED_PASS_SCOPE
          and recipe['max_forwards']==1050 and recipe['max_private_updates']==recipe['max_backwards']==512
          and recipe['max_checkpoint_loads']==recipe['max_full_state_restores']==1
          and recipe['max_candidate_evidence_writes']==1 and recipe['max_checkpoint_writes']==0
          and recipe['max_model_seconds']==2400 and recipe['passes']==[2,3]
          and recipe['baseline_parity_forwards']==2 and recipe['max_training_forwards']==512
          and recipe['max_evaluation_forwards']==538,'ADMITTED_BOUNDS')
    qualified=preflight_read_only(recipe,source_commit=plan['source_commit'])
    origin,receipts,view,priors=(qualified[k] for k in ('origin','receipts','view','priors'))
    numerical=recipe['numerical_recipe'];refs=recipe['input_refs']
    root=scope.TA_M5_REPEATED_PASS_ROOT;output=Path(window['output_directory'])
    output.mkdir(parents=True)
    _need(not any((root/n).exists() for n in ('RESULT.json','TERMINAL.json','E3_EVIDENCE.pt')),'OUTPUT_EXISTS')
    counters={k:0 for k in ('model_constructions','model_construction_returns','optimizer_constructions',
        'forwards','forward_returns','training_forwards','evaluation_forwards','backwards','backward_returns',
        'optimizer_step_calls','optimizer_step_returns','optimizer_updates','checkpoint_loads',
        'resume_state_load_calls','resume_state_load_returns','strict_full_state_restore_calls',
        'strict_full_state_restore_returns','checkpoint_writes','candidate_evidence_writes','json_bytes',
        'extra_truth_gathers')}
    measurements={k:v['measurement'] for k,v in origin['measurements'].items()}
    measurement_refs={k:refs[k.lower()+'_measurement'] for k in measurements}
    capture_refs={k:refs[k.lower()+'_capture'] for k in measurements}
    started=time.monotonic();model_started=None
    provider=prepared=model=optimizer=saved=None;error=None;evidence_ref=None
    current_pass,batch_index=2,0;restore_proof={};baseline_parity={};score_gates={}
    step_records=[];timings={};evaluation_unchanged=all_measured=source_unchanged=False

    def clock_guard():
        _need(time.monotonic()-started<window['max_invocation_seconds'],'INVOCATION_DEADLINE')
        if model_started is not None:_need(time.monotonic()-model_started<2400,'MODEL_DEADLINE')

    try:
        original_trace._bookend(recipe,plan,policy_path,policy_file_sha256)
        common._json(output/'CLAIM.json',dict(recipe=window['recipe'],plan_sha256=plan['plan_sha256'],
            source_commit=plan['source_commit'],limits=limits,test_data_used=False),
            counters,recipe['max_published_json_bytes'])
        import numpy as np
        import torch
        from gx1.contracts import entry_native_direction_feature_view_v1 as views
        from gx1.contracts import entry_native_direction_research_state_v1 as old
        from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
        from gx1.contracts import market_state_time_query_fold1_fit_v1 as full_scope
        from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
        from gx1.contracts.market_state_time_query_binding_v1 import load_bound_time_query_targets,sampled_query_k
        from gx1.contracts.market_state_five_endpoint_target_binding_v1 import load_bound_five_endpoint_targets
        from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
        from gx1.contracts.market_state_time_query_native_F_batch_v1 import evaluate_fit_batch,evaluate_endpoint_batch
        from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
        from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as model_sha
        from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
        from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
        from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner
        from gx1.scripts.run_unified_exit_random_access_full_train_v1 import _dynamic_m5_technical_tree_sha256 as tree_sha
        _need(torch.cuda.is_available(),'CUDA_REQUIRED');device=torch.device('cuda',0)
        bound,endpoint_bound,future_bound=load_bound_time_query_targets(),load_bound_five_endpoint_targets(),load_bound_future_displacement()
        provider,_,prepared,rows,issues=runner._prepared_fold1(receipts,view,np=np)
        objective.require_m5_equal_observation_axes(rows,issues)
        orders,positions=paired.build_execution_orders(rows,origin['paired_recipe']['arm_batch_indices'])
        order=torch.from_numpy(orders['CONTROL']);selected=positions['CONTROL']
        batch_indices=recipe['batch_indices']
        _need(batch_indices==origin['paired_recipe']['arm_batch_indices']['CONTROL'],'UNCHANGED_BATCH_INDICES')
        _need(tree_sha(order)==origin['paired_result']['arm_progress']['CONTROL']['parent_order_sha256'],
              'CONTROL_ORDER')
        axes=prepared['axes'];sl=slice(full_scope.FOLD['oof_start_row'],full_scope.FOLD['oof_end_row'])
        later_rows,later_issues=np.asarray(axes['parent_rows'][sl],np.int64),np.asarray(axes['decision_ns'][sl],np.int64)
        expected=endpoint_bound.mature_oof_axes(fold_id=1)
        _need(len(later_rows)==17753 and np.array_equal(later_rows,expected['score_parent_rows'])
              and np.array_equal(later_issues,expected['score_decision_ns']),'LATER_AXIS')
        later_positions=np.concatenate([np.arange(l,r,dtype=np.int64)
                                        for l,r in origin['paired_recipe']['score_eval_ranges']])
        with np.load(refs['control_train_capture']['path'],allow_pickle=False) as archive:
            baseline={k:archive[k].copy() for k in ('parents','issues','state_query','future_query','state_raw','future_raw',
                'state_allowed','state_eligible','state_mature','future_allowed','future_eligible','future_mature')}
        with np.load(refs['control_later_capture']['path'],allow_pickle=False) as archive:
            later_baseline={k:archive[k].copy() for k in ('parents','issues','query_native',
                'state_activity_side_logits','future_activity_side_logits','state_allowed','state_eligible',
                'state_mature','future_allowed','future_eligible','future_mature')}
        _need(np.array_equal(baseline['parents'],rows[selected]) and np.array_equal(baseline['issues'],issues[selected])
              and np.array_equal(baseline['state_query'],sampled_query_k(issues[selected],pass_index=0,seed=numerical['seed']))
              and np.array_equal(baseline['future_query'],np.tile(np.array([1,3,12,48,288],np.int64),(4096,1)))
              and np.array_equal(later_baseline['parents'],later_rows[later_positions])
              and np.array_equal(later_baseline['issues'],later_issues[later_positions]),'CACHED_AXES_QUERIES')
        timings['input_setup_seconds']=time.monotonic()-started
        clock_guard();model_started=time.monotonic()
        trainer._set_deterministic(numerical['seed'],device,numerical['precision_policy'])
        _need(trainer._GRAD_CLIP_NORM==numerical['grad_clip_norm']==1.,'CANONICAL_CLIP')
        counters['model_constructions']+=1
        backbone=views.build_native_direction_research_model(view,prefix_normalization=receipts['fold1']['normal'],
            constructor_metadata=receipts['constructor'],device=device,unobserved_units_profile=receipts['fold1']['profile'])
        model=NativeFutureDualModel(backbone).to(device);counters['model_construction_returns']+=1
        view_identity=full_scope.require_checkpoint_view(model.state_model.feature_view_identity,
            recipe=origin['original_recipe'],verify_files=True,ta_sf=True)
        _need(model.research_objective['contract_sha256']==recipe['expected_model_contract_sha256'],'MODEL_CONTRACT')
        model.train();counters['optimizer_constructions']+=1
        optimizer,_=state.build_fold1_fit_optimizer(model,numerical_recipe=numerical)
        saved=common._load_state(refs['control_evidence'],torch=torch,maximum=recipe['max_checkpoint_bytes'],counters=counters)
        counters['checkpoint_loads']+=1;saved_sha=tree_sha(saved)
        counters['strict_full_state_restore_calls']+=1
        restore_proof=restore_control_evidence(saved,model,optimizer,recipe=recipe,origin=origin,
            parent_order=order,feature_view_identity=view_identity,device=device)
        counters['strict_full_state_restore_returns']+=1
        pairs,_,_=state._parameters(model)

        def signature():
            return dict(model=model_sha(model.state_dict()),buffers=tree_sha(dict(model.named_buffers())),
                optimizer=tree_sha(optimizer.state_dict()),rng=tree_sha(trainer._attended_session_rng_state(device=device)),
                modes={n:bool(m.training) for n,m in model.named_modules()})

        def fit_report(slot,training):
            original_start=16*batch_indices[slot]
            _need(np.array_equal(order[slot*16:(slot+1)*16].numpy(),rows[original_start:original_start+16]),'INTACT_BATCH')
            raw,parents,clocks=runner._feature_batch(prepared,view,receipts,rows,issues,original_start,original_start+16,np=np)
            features={k:torch.from_numpy(v).to(device) for k,v in raw.items()}
            _need(counters['forwards']<1050,'FORWARD_BUDGET')
            counters['forwards']+=1;counters['training_forwards' if training else 'evaluation_forwards']+=1
            if training:
                with trainer._training_mha_fastpath_context(model.backbone),trainer._training_autocast_context(), \
                     trainer._training_model_finite_check_context():
                    report=evaluate_fit_batch(model,features,bound,future_bound,parents,clocks,fold_id=1,
                        pass_index=0,seed=numerical['seed'],future_weight=1.,m5_equal_observation=True,
                        fit_batch_start=original_start)
            else:
                with torch.inference_mode():
                    report=evaluate_fit_batch(model,features,bound,future_bound,parents,clocks,fold_id=1,
                        pass_index=0,seed=numerical['seed'],future_weight=1.,m5_equal_observation=True,
                        fit_batch_start=original_start)
            counters['forward_returns']+=1
            _need(report['model_contract_sha256']==recipe['expected_model_contract_sha256']
                  and report['loss_contract_sha256']==recipe['expected_loss_contract_sha256']
                  and report['skip_update'] is False and bool(torch.isfinite(report['loss'])),'CANONICAL_LOSS')
            sl=slice(slot*16,(slot+1)*16)
            _need(np.array_equal(parents,baseline['parents'][sl]) and np.array_equal(clocks,baseline['issues'][sl])
                  and np.array_equal(report['state_query_k'],baseline['state_query'][sl])
                  and np.array_equal(report['future_native_k'],baseline['future_query'][slot*16]),'FIT_AXIS_QUERY')
            for task in TASKS:
                for key in ('allowed','mature'):
                    _need(np.array_equal(report[task+'_'+key],baseline[task+'_'+key][sl]),'FIT_LABELS')
            if not training:
                truths={'state':bound.gather_fit_sampled(parents,clocks,fold_id=1,pass_index=0,seed=numerical['seed']),
                        'future':future_bound.gather(parents,clocks,fold_id=1,partition='fit')}
                counters['extra_truth_gathers']+=2
                for task,truth in truths.items():
                    for key in ('allowed','eligible','mature'):
                        _need(np.array_equal(truth[key],baseline[task+'_'+key][sl]),'FIT_ELIGIBILITY')
            return report

        def later_report(left):
            raw,parents,clocks=runner._feature_batch(prepared,view,receipts,later_rows[later_positions],
                                                    later_issues[later_positions],left,left+16,np=np)
            features={k:torch.from_numpy(v).to(device) for k,v in raw.items()}
            _need(counters['forwards']<1050,'FORWARD_BUDGET')
            counters['forwards']+=1;counters['evaluation_forwards']+=1
            report=evaluate_endpoint_batch(model,features,endpoint_bound,future_bound,parents,clocks,
                fold_id=1,partition='score',future_weight=1.,frozen_priors=priors,capture_m5_readout=True)
            counters['forward_returns']+=1
            for key,value in later_baseline.items():
                if not key.endswith('_logits'):
                    _need(np.array_equal(report['m5_readout'][key],value[left:left+16]),'LATER_AXES_MASKS')
            return report

        before=signature();model.eval()
        try:
            report=fit_report(0,False)
            raw_fit={t:report[t+'_activity_side_logits'].detach().cpu().numpy().copy() for t in TASKS};del report
            report=later_report(0)
            raw_later={t:report[t+'_activity_side_logits'] for t in TASKS};del report
        finally:
            for n,m in model.named_modules():m.training=before['modes'][n]
        _need(signature()==before and all(p.grad is None for _,p in pairs),'BASELINE_EVAL_STATE')
        tolerance=recipe['baseline_parity']
        for panel,raws in (('TRAIN',raw_fit),('LATER',raw_later)):
            baseline_parity[panel]={}
            for task,actual in raws.items():
                target=baseline[task+'_raw'][:16] if panel=='TRAIN' else later_baseline[task+'_activity_side_logits'][:16]
                raw_error=float(np.max(np.abs(actual.astype(np.float64)-target.astype(np.float64))))
                probability_error=float(np.max(np.abs(np.exp(-np.logaddexp(0.,-actual[:,0,1].astype(np.float64)))-
                                                       np.exp(-np.logaddexp(0.,-target[:,0,1].astype(np.float64))))))
                _need(raw_error<=tolerance['raw_atol'] and probability_error<=tolerance['probability_atol'],
                      'BASELINE_OUTPUT_PARITY')
                baseline_parity[panel][task]=dict(raw_max_abs_error=raw_error,M5_probability_max_abs_error=probability_error)
        del raw_fit,raw_later

        def publish_measurement(name,measured,arrays,digest,updates):
            stream=io.BytesIO();np.savez_compressed(stream,**arrays)
            _need(stream.tell()<=recipe['max_trace_bytes'],'CAPTURE_BYTES')
            _publish_bytes(root/(name+'_READOUT.npz'),stream.getvalue())
            capture_refs[name]=common._hash_file(root/(name+'_READOUT.npz'))
            payload=dict(schema_version='gx1_ta_m5_repeated_pass_measurement_wrapper_v1',measurement=measured,
                capture=capture_refs[name],evaluated_model_state_sha256=digest,
                cumulative_optimizer_updates=updates,query_pass_index=0,test_data_used=False)
            measurement_refs[name]=common._json(root/(name+'.json'),payload,counters,recipe['max_published_json_bytes'])
            measurements[name]=measured

        def measure(epoch):
            before=signature();model.eval();arrays={k:v.copy() for k,v in baseline.items()};losses=[]
            try:
                for slot in range(256):
                    clock_guard();report=fit_report(slot,False)
                    for task in TASKS:arrays[task+'_raw'][slot*16:(slot+1)*16]=report[task+'_activity_side_logits'].cpu().numpy()
                    losses.append(float(report['loss']));del report
                measured=paired.sampled_fit_measurement(arrays,losses,priors,
                    model_contract_sha256=recipe['expected_model_contract_sha256'],
                    loss_contract_sha256=recipe['expected_loss_contract_sha256'],feature_view_sha256=recipe['expected_view_sha256'])
                _need(measured['observation_chain_sha256']==measurements['CONTROL_TRAIN']['observation_chain_sha256'],
                      'TRAIN_MATCHED')
                publish_measurement(epoch+'_TRAIN',measured,arrays,before['model'],256+counters['optimizer_updates'])
                arrays={k:np.empty(shape,dtype) for k,(shape,dtype) in runner._ta_m5_capture_layout(192).items()}
                total=None
                for left in range(0,192,16):
                    clock_guard();report=later_report(left)
                    total=objective.merge_endpoint_measurements(total,report['measurement'])
                    runner._ta_m5_capture_batch(arrays,report['m5_readout'],left=left,right=left+16,
                        parents=later_rows[later_positions][left:left+16],issues=later_issues[later_positions][left:left+16])
                    del report
                measured=objective.finalize_endpoint_measurement(total)
                runner._ta_require_matched_observations(measurements['CONTROL_LATER'],measured)
                publish_measurement(epoch+'_LATER',measured,arrays,before['model'],256+counters['optimizer_updates'])
            finally:
                for n,m in model.named_modules():m.training=before['modes'][n]
            _need(signature()==before and all(p.grad is None for _,p in pairs),'EVALUATION_FULL_STATE')
            score_gates[epoch]=compare_fixed_pass(measurements,epoch,baseline_score_atol=tolerance['score_atol'])

        for epoch_number in recipe['passes']:
            current_pass=epoch_number;batch_index=0
            fit_started=time.monotonic()
            for slot in range(256):
                clock_guard();optimizer.zero_grad(set_to_none=True);report=fit_report(slot,True)
                loss=float(report['loss'].detach());counters['backwards']+=1
                report['loss'].backward();counters['backward_returns']+=1
                counters['optimizer_step_calls']+=1
                norm=trainer._optimizer_step_with_finite_gradients(model=model,optimizer=optimizer)
                counters['optimizer_step_returns']+=1;counters['optimizer_updates']+=1;batch_index=slot+1
                optimizer.zero_grad(set_to_none=True)
                step_records.append(dict(pass_number=current_pass,batch_index=batch_index,
                    original_fit_batch_start=16*batch_indices[slot],query_pass_index=0,
                    cumulative_optimizer_updates=256+counters['optimizer_updates'],loss=loss,gradient_norm=float(norm)))
                del report
                if batch_index%64==0:print('E'+str(current_pass)+' updates='+str(batch_index),flush=True)
            timings['E'+str(current_pass)+'_fit_seconds']=time.monotonic()-fit_started
            evaluation_started=time.monotonic();measure('E'+str(current_pass))
            timings['E'+str(current_pass)+'_evaluation_seconds']=time.monotonic()-evaluation_started
        evaluation_unchanged=all_measured=True
        _need(all(p.grad is None and bool(torch.isfinite(p).all()) for _,p in pairs)
              and state._binding(model,optimizer,numerical,view_identity)==saved['parameter_binding'],'FINAL_BINDING')
        model_state=copy.deepcopy(model.state_dict());buffers=copy.deepcopy(dict(model.named_buffers()))
        optimizer_state=copy.deepcopy(optimizer.state_dict());rng=copy.deepcopy(trainer._attended_session_rng_state(device=device))
        old._tensor_map(model_state,model.state_dict(),'E3_MODEL');old._tensor_map(buffers,dict(model.named_buffers()),'E3_BUFFERS')
        old._optimizer_state(optimizer_state,optimizer,model,768);old._rng(rng,device=device,seed=numerical['seed'])
        _need(tree_sha(saved)==saved_sha,'SOURCE_CAPSULE_MUTATED')
        capsule=dict(schema_version=CAPSULE_SCHEMA,recipe=window['recipe'],source_commit=plan['source_commit'],
            source_control_evidence=refs['control_evidence'],canonical_resume_authority=False,test_data_used=False,
            model_state=model_state,buffer_state=buffers,optimizer_state=optimizer_state,rng_state=rng,
            module_training={n:bool(m.training) for n,m in model.named_modules()},parent_order=order.clone(),
            parent_cursor=None,pending_gradients=0,parameter_binding=saved['parameter_binding'],
            original_numerical_recipe_reference=numerical,current_pass=3,batch_index=256,
            optimizer_updates=512,cumulative_optimizer_updates=768,unique_fit_rows=4096,
            selected_parent_presentations=12288,method_state=dict(batch_indices=batch_indices,query_pass_index=0,
                order_sha256=tree_sha(order),unselected_suffix_consumed=False,AdamW_state_reset=False,
                original_fit_batch_starts=[16*b for b in batch_indices],fixed_measurement_passes=[2,3]),
            model_state_sha256=model_sha(model_state),buffer_state_sha256=tree_sha(buffers),
            optimizer_state_sha256=tree_sha(optimizer_state),rng_state_sha256=tree_sha(rng))
        capsule['evidence_sha256']=tree_sha(capsule)
        stream=io.BytesIO();torch.save(capsule,stream)
        _need(0<stream.tell()<=recipe['max_candidate_evidence_bytes'],'EVIDENCE_BYTES');clock_guard()
        _publish_bytes(root/'E3_EVIDENCE.pt',stream.getvalue());counters['candidate_evidence_writes']+=1
        evidence_ref=common._hash_file(root/'E3_EVIDENCE.pt')
        _need(all(counters[k]==v for k,v in scope.TA_M5_REPEATED_PASS_COMPLETE_COUNTS.items())
              and counters['forward_returns']==1050 and counters['backward_returns']==512
              and counters['optimizer_step_returns']==512 and counters['strict_full_state_restore_returns']==1,
              'EXACT_EXECUTION_COUNTS')
        original_trace._bookend(recipe,plan,policy_path,policy_file_sha256);clock_guard();source_unchanged=True
    except BaseException as exc:error=type(exc).__name__+': '+str(exc)
    finally:
        if prepared is not None:
            try:
                from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as base
                base._release_pages(provider,prepared)
            except BaseException as exc:error=(error+'; ' if error else '')+'cleanup: '+type(exc).__name__+': '+str(exc)
        prepared=provider=model=optimizer=saved=None;gc.collect()
    outcome='COMPLETE' if error is None and source_unchanged and all_measured else 'FAILED'
    payload=dict(schema_version=scope.TA_M5_REPEATED_PASS_RESULT_SCHEMA,outcome=outcome,
        decision='PASS_TA_M5_REPEATED_PASS_TECHNICAL' if outcome=='COMPLETE' else 'FAIL_TA_M5_REPEATED_PASS_TECHNICAL',
        recipe=window['recipe'],source_commit=plan['source_commit'],source_control_evidence=refs['control_evidence'],
        optimizer_updates=counters['optimizer_updates'],physical_optimizer_updates=counters['optimizer_updates'],
        cumulative_optimizer_updates=256+counters['optimizer_updates'],current_pass=current_pass,batch_index=batch_index,
        parent_cursor=None,parent_order_sha256=None,full_state=None,final_state=None,skipped_batches=0,
        candidate_evidence=evidence_ref,canonical_resume_authority=False,checkpoint_writes=0,
        control_restore_exact=restore_proof.get('all_numerical_state_exact',False),
        baseline_output_parity=bool(baseline_parity) and sum(len(v) for v in baseline_parity.values())==4,
        evaluation_state_unchanged=evaluation_unchanged,all_measurements_complete=all_measured,
        optimizer_state_continued=restore_proof.get('all_numerical_state_exact',False),
        restore_proof=restore_proof,baseline_parity=baseline_parity,measurement_refs=measurement_refs,captures=capture_refs,
        score_gates=score_gates,primary_comparison='E3_vs_cached_E1',fixed_intermediate_measurement='E2',
        counts=counters,step_records=step_records,timings=dict(timings,elapsed_seconds=time.monotonic()-started),
        source_unchanged=source_unchanged,error=error,test_data_used=False,promotion_authorized=False,
        limits='Two further passes on the same4096TRAIN rows, original sampled queries and AdamW state. E2/E3 measured unconditionally; E3 is primary, no best-checkpoint selection. Reused later TRAIN is development data. Private evidence is not canonical resume authority; unselected full-order suffix is never consumed.')
    result=common._json(root/'RESULT.json',payload,counters,recipe['max_published_json_bytes'])
    common._json(root/'TERMINAL.json',dict(status=outcome,exit_code=0 if outcome=='COMPLETE' else 1,
        result=result,recipe=window['recipe'],source_unchanged=source_unchanged,error=error,
        test_data_used=False,no_relaunch_under_this_plan=True),counters,recipe['max_published_json_bytes'])
    cursor=scope.build_ta_m5_repeated_pass_cursor(recipe_ref=window['recipe'],outcome=outcome,
        optimizer_updates=counters['optimizer_updates'],current_pass=current_pass,batch_index=batch_index,
        cumulative_optimizer_updates=256+counters['optimizer_updates'],result=result,batch_size=16)
    cursor_ref=common._json(Path(window['campaign_cursor_path']),cursor,counters,recipe['max_published_json_bytes'])
    progress=authority.build_research_progress(window=window,plan=plan,invocation=invocation,
        checkpoint_pointer=cursor_ref,optimizer_updates=counters['optimizer_updates'],parent_cursor=None,
        parent_order_sha256=None,outcome=outcome)
    common._json(progress_path,progress,counters,recipe['max_published_json_bytes'])
    authority.require_research_progress(progress,plan_sha256=plan['plan_sha256'],invocation=invocation,
        expected_selection_receipt_sha256=None,verify_file=True)
    if error is not None:raise RuntimeError(error)
    return dict(result=result,checkpoint_pointer=cursor_ref,progress_path=str(progress_path))
