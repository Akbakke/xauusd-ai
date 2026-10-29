"""One source-bound private 60-minute native qualification; no fitted model.

Four physical updates compare two uninterrupted steps with one plus exact
serialized resume. All 32 parents are fixed before this qualification. Private
weights cannot be used by a later fit; this worker conveys no execution rights.
"""
from __future__ import annotations

import gc
import signal
import sys
import time
from pathlib import Path


def _require(ok, reason):
    if not ok:
        raise RuntimeError('NATIVE_DENSE_QUALIFICATION_' + reason)


def require_m1_value_gradient_contact(evidence):
    """Require observed M1 value contact after the zero-residual first step."""
    gradients=evidence.get('gradient_nonzero_counts') if isinstance(evidence,dict) else None
    _require(isinstance(gradients,dict), 'M1_VALUE_GRADIENT_EVIDENCE')
    counts=[gradients.get('m1_ofi_value_minute_'+str(i)+'_weight_nonzero') for i in range(1,6)]
    _require(all(type(value) is int and value>=0 for value in counts), 'M1_VALUE_GRADIENT_EVIDENCE')
    _require(any(counts), 'M1_VALUE_GRADIENT_NOT_OBSERVED')


def select_qualification_parents(ordered, diagnostic_ids, design, *, np):
    parents=design.get('parents_sorted') if isinstance(design,dict) else None
    _require(isinstance(parents,list) and len(parents)==32 and all(type(x) is int for x in parents)
        and len(ordered)>=32 and len(diagnostic_ids)>=32, 'PARENT_TYPES')
    selected=np.ascontiguousarray(np.sort(diagnostic_ids[:32]),dtype=np.int64)
    _require(np.array_equal(selected,np.asarray(parents,dtype=np.int64))
        and np.all(selected[1:]>selected[:-1]) and np.isin(selected,ordered).all(), 'PARENT_IDENTITY')
    return selected


def require_decimal_prefix_prior(prior, audit):
    """Match all 60 corrected TRAIN counts to the independent decimal audit."""
    _require(isinstance(prior,dict) and isinstance(audit,dict)
        and audit.get('schema_version')=='gx1_direction_decimal_sign_audit_result_v1'
        and audit.get('status')=='COMPLETE' and audit.get('test_data_used') is False
        and audit.get('new_model_forwards')==audit.get('optimizer_steps')==0
        and audit.get('target_source_rows_and_masks_parity') is True
        and audit.get('vendor_to_canonical_float64_and_clock_parity') is True, 'DECIMAL_PRIOR_AUDIT')
    rows=audit.get('confusions',{}).get('fold1_train')
    _require(isinstance(rows,list) and len(rows)==60
        and all(isinstance(prior.get(k),list) and len(prior[k])==60 for k in ('eligible_up','eligible_down')),
        'DECIMAL_PRIOR_ROWS')
    for i,row in enumerate(rows):
        _require(isinstance(row,dict), 'DECIMAL_PRIOR_ROW:'+str(i+1))
        matrix=row.get('saved_by_exact_sign_order_minus_zero_plus')
        _require(row.get('horizon_minutes')==i+1 and row.get('nonzero_sign_flips')==0
            and isinstance(matrix,list) and len(matrix)==3
            and all(isinstance(r,list) and len(r)==3 and all(type(v) is int and v>=0 for v in r) for r in matrix)
            and matrix[0][2]==matrix[1][0]==matrix[1][2]==matrix[2][0]==0
            and type(row.get('observed_mature')) is int and row['observed_mature']==sum(map(sum,matrix))
            and type(prior['eligible_down'][i]) is int and type(prior['eligible_up'][i]) is int
            and prior['eligible_down'][i]==matrix[0][0] and prior['eligible_up'][i]==matrix[2][2],
            'DECIMAL_PRIOR_COUNT:'+str(i+1))


def require_decimal_initial_prediction_parity(arrays, ref, *, np):
    """Match the prior 60-head 0U predictions, without reading its weights."""
    keys=('parent_rows','bar_start_ns','decision_ns','logit_up')
    _require(isinstance(arrays,dict) and isinstance(ref,dict) and set(ref)=={'path','sha256'}, 'DECIMAL_INITIAL_INPUT')
    with np.load(ref['path'],allow_pickle=False) as archive:
        for key in keys:
            _require(key in archive.files and key in arrays, 'DECIMAL_INITIAL_FIELD:'+key)
            old=archive[key];new=arrays[key]
            _require(isinstance(new,np.ndarray) and old.dtype==new.dtype and old.shape==new.shape
                and np.ascontiguousarray(old).tobytes()==np.ascontiguousarray(new).tobytes(), 'DECIMAL_INITIAL_PARITY:'+key)
    _require(arrays['parent_rows'].shape==(256,) and arrays['logit_up'].shape==(256,60), 'DECIMAL_INITIAL_SHAPE')


def run_dense_qualification(*, policy_path:Path, policy_file_sha256:str, progress_path:Path):
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    limits=require_guarded_cuda_trainer_execution()
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix_authority
    window,plan,invocation,marker,launch=authority.require_native_direction_research_context(
        policy_path,policy_file_sha256,progress_path)
    candidate=launch['run_scope']==prefix_authority.M1_PRICE_PRIVATE_ADAPTER_SCOPE
    decimal=launch['run_scope']==authority.DECIMAL_PRIVATE_SCOPE or candidate
    _require(launch['run_scope'] in (authority.DENSE_PRIVATE_SCOPE,authority.DECIMAL_PRIVATE_SCOPE,
        prefix_authority.M1_PRICE_PRIVATE_ADAPTER_SCOPE) and invocation['batch_size']==16
        and (not candidate or launch.get('candidate_mode') in ('price_only','price_and_m1',prefix_authority.M1_PRICE_COARSENED_MODE)), 'EXACT_SCOPE')
    sys.dont_write_bytecode=True
    import numpy as np
    import torch
    from gx1.contracts.entry_model_native_train_recipe_v1 import require_native_direction_research_recipe
    from gx1.contracts import entry_native_direction_research_state_v1 as state_owner
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as prefix
    from gx1.scripts.run_unified_exit_random_access_full_train_v1 import _dynamic_m5_technical_tree_sha256 as tree_sha

    recipe=require_native_direction_research_recipe(launch['numerical_recipe'])
    _require(recipe['batch_size']==16 and recipe['max_updates']==2
        and recipe.get('dense_minute_supervision') is True and recipe.get('signed_bps_supervision',False) is False
        and recipe.get('magnitude_weighted_direction',False) is False, 'DENSE_RECIPE')
    binding_ref=common._hash_file(launch['execution_binding_path']); extra={}; binding=common._read(binding_ref,extra)
    io_budget=common._read(common.INPUT_DESIGN,extra)['budget']['max_rchar_bytes']
    _require(io_budget==137438953472, 'IO_CEILING')
    initial_report=None if decimal else common._read(prefix_authority.ORDER_ORIGIN['initial_measurement'],extra)
    design=common._read(launch['qualification_design'],extra)
    _require(design.get('schema_version')=='gx1_order_supported_qualification_design_v1'
        and design.get('status')=='DESIGN_ONLY_NO_EXECUTION_AUTHORITY'
        and design.get('seed')==prefix.SEED and design.get('prefix_rows')==launch['fold']['prefix_rows']
        and design.get('batch_size')==16 and design.get('test_data_used') is False, 'PARENT_DESIGN')
    for key in ('prior_failure_review','prior_support_report','prior_support_run_result','prior_support_root_review'):
        common._read(design[key],extra)
    proof=common._read(launch['gc_layout_metadata_parity'],extra)
    _require(proof.get('status')=='COMPLETE_CONSUMED' and proof.get('metadata_exact_parity') is True
        and proof.get('test_cases_passed')==6 and isinstance(proof.get('counters'),dict)
        and set(proof['counters'])=={'cuda_initializations','fit_calls','native_model_calls','optimizer_steps','payload_loads'}
        and all(v==0 for v in proof['counters'].values()), 'GC_METADATA_PROOF')
    actual_launch={**launch,'run_scope':(prefix_authority.M1_PRICE_PRIVATE_ADAPTER_SCOPE if candidate
        else prefix_authority.DENSE_DECIMAL_PRIVATE_SCOPE if decimal else prefix_authority.DENSE_SCOPE)}
    context=prefix._launch_metadata(actual_launch,common);context['info']['metadata'].update(extra)
    if decimal and not candidate:
        ref=launch['decimal_initial_predictions']
        _require(ref==authority.DECIMAL_INITIAL_PREDICTIONS, 'DECIMAL_INITIAL_REFERENCE')
        context['decimal_initial_predictions']=dict(ref)
        context['decimal_initial_predictions_stat']=common._stat(ref['path'])
        _require(common._hash_file(ref['path'])==ref, 'DECIMAL_INITIAL_SHA')
    for ref in (window['recipe'],binding['plan'],{'path':str(policy_path),'sha256':policy_file_sha256},launch['qualification_design']):
        common._read(ref,context['info']['metadata'])
    common._freeze(launch,binding,context)
    output=Path(window['output_directory']);output.mkdir(parents=True)
    denied=common._install_read_guard(context,output,Path(launch['runtime_root']),dense_target=context['dense_target'],
        decimal_original=context.get('decimal_original') if decimal else None,
        decimal_initial_predictions=context.get('decimal_initial_predictions') if decimal and not candidate else None,
        candidate_m1_refs=context.get('candidate_m1_refs') if candidate else None)
    counters={key:0 for key in ('model_constructions','model_construction_returns','optimizer_constructions',
        'forwards','forward_returns','backwards','backward_returns','optimizer_step_calls','optimizer_updates',
        'staged_state_load_calls','staged_state_load_returns','resume_state_load_calls','resume_state_load_returns',
        'forbidden_legacy_calls','json_bytes')}
    evaluation={'forward_calls':0,'forward_returns':0,'forward_kind_calls':{}}
    before=common._io();started=time.monotonic();previous=sys.getprofile();device=torch.device('cuda',0)
    full_state=initial=initial_identity=None;error=None;initial_checked=parity=unchanged=False;cursor=0
    steps=[];batch_support=[];order_sha='0'*64;prepared=model=optimizer=None;prior=None;decimal_prior_checked=False;decimal_initial_checked=False
    common._json(output/'CLAIM.json',{'recipe':window['recipe'],'plan_sha256':plan['plan_sha256'],
        'limits':limits,'additional_worker_ceiling_seconds':1200},counters,launch['max_published_json_bytes'])

    def expired(*_args):
        raise RuntimeError('NATIVE_DENSE_QUALIFICATION_1200_SECOND_CEILING')
    old_handler=signal.signal(signal.SIGALRM,expired);signal.alarm(1200)
    try:
        sys.setprofile(common._deny_legacy_work(previous,counters))
        prepared,targets,view,context_map,ordered=prefix._prepare(context,actual_launch,np,dense_minutes=True)
        prior=prefix._prefix_prior(prepared['arrays'],targets,launch['fold'],np)
        if decimal:
            require_decimal_prefix_prior(prior,context['decimal_sign_audit']);decimal_prior_checked=True
        else:
            original_prior=initial_report['own_prefix_prior'];anchors=(0,4,14)
            for key in ('eligible_up','eligible_down','probability_up','logit_up'):
                _require([prior[key][i] for i in anchors]==original_prior[key], 'ORIGINAL_ANCHOR_PRIOR:'+key)
        diagnostic_ids=prefix._fixed_order(prepared['arrays']['parent_rows'],launch['fold']['prefix_rows'],prefix.SEED,np)[:256]
        selected=select_qualification_parents(ordered,diagnostic_ids,design,np=np)
        order=torch.from_numpy(selected);order_sha=tree_sha(order);batches=[]
        for left in (0,16):
            raw,target=prefix._batch(prepared,targets,view,selected[left:left+16],np)
            support=common._gc_support(raw,prepared['gc']['layout'],np)
            _require(all(sum(item['raw_nonzero_values_per_parent'])>0 for families in support.values()
                for item in families.values()),'BOUND_GC_NONZERO_INPUTS')
            batch_support.append({'parent_rows':selected[left:left+16].tolist(),'gc':support})
            batches.append(prefix._device_batch(raw,target,torch,device))
        batch_sha=tree_sha(batches)

        def capture(m,opt,offset,updates):
            return state_owner.capture_native_direction_research_state(m,opt,recipe=recipe,
                feature_view_identity=m.feature_view_identity,parent_order=order,cursor=offset,
                optimizer_updates=updates,device=device)

        for trajectory in ('reference','interrupted','resumed'):
            cursor=0;model,optimizer,_=common._fresh(context,view,recipe,device,counters,max_constructions=3)
            fresh=capture(model,optimizer,0,0)
            identity=tree_sha({k:fresh[k] for k in ('binding','model_state','buffer_state',
                'optimizer_state','rng_state','module_training')})
            if initial_identity is None:
                initial_identity=identity
                measured,count=prefix._evaluate(model,prepared,targets,view,context_map,diagnostic_ids,
                    cutoff=launch['fold']['train_cutoff_ns'],kind='INITIAL_TRAIN',counters=evaluation,
                    ceiling=17,device=device,torch=torch,np=np,tree_sha=tree_sha,parity=True)
                _require(count==256 and measured['initial_parity_passed']
                    and measured['arrays']['logit_up'].shape==(256,60), 'FRESH_DENSE_INITIAL')
                if decimal and not candidate:
                    require_decimal_initial_prediction_parity(measured['arrays'],context['decimal_initial_predictions'],np=np)
                    decimal_initial_checked=True
                initial_prediction=prefix._save_predictions(output/'INITIAL_TRAIN.npz',measured['arrays'],np=np,common=common,tree_sha=tree_sha)
                del measured
                initial=common._save_state(output/'INITIAL_STATE.pt',fresh,torch=torch,
                    maximum=launch['max_checkpoint_bytes'],counters=counters)
                initial_inactive=tree_sha({name:p for name,p in model.named_parameters() if not p.requires_grad})
                initial_checked=True
            else:
                _require(identity==initial_identity and initial_checked, 'SAME_FRESH_IDENTITY')
            del fresh
            if trajectory=='resumed':
                loaded=common._load_state(interrupted_ref,torch=torch,maximum=launch['max_checkpoint_bytes'],counters=counters)
                restored=state_owner.restore_native_direction_research_state(loaded,model,optimizer,recipe=recipe,
                    feature_view_identity=model.feature_view_identity,parent_order=order,device=device)
                _require(restored['cursor']==16 and restored['optimizer_updates']==1
                    and capture(model,optimizer,16,1)['contract_sha256']==loaded['contract_sha256'], 'SERIALIZED_RESTORE')
                del loaded;cursor=16
            logical=(0,1) if trajectory=='reference' else (0,) if trajectory=='interrupted' else (1,)
            for index in logical:
                _require(initial_checked and time.monotonic()-started<1200, 'INITIAL_AND_DEADLINE')
                evidence=common._step(model,optimizer,*batches[index],cutoff=launch['fold']['train_cutoff_ns'],counters=counters,torch=torch)
                _require(evidence['gradient_nonzero_counts']['chart_mtf_weight_nonzero']>0, 'CHART_MTF_GRADIENT')
                if index==1:
                    _require(evidence['gradient_nonzero_counts']['chart_local_weight_nonzero']>0, 'CHART_LOCAL_AFTER_ONE_UPDATE')
                cursor=16*(index+1);steps.append({'trajectory':trajectory,'logical_step':index+1,**evidence})
                if candidate and launch['candidate_mode'] in ('price_and_m1',prefix_authority.M1_PRICE_COARSENED_MODE) and index==1:
                    require_m1_value_gradient_contact(evidence)
                _require(tree_sha({name:p for name,p in model.named_parameters() if not p.requires_grad})==initial_inactive,
                    'INACTIVE_WEIGHTS_UNCHANGED')
            saved=capture(model,optimizer,cursor,1 if trajectory=='interrupted' else 2)
            if trajectory=='reference':reference_sha=saved['contract_sha256']
            elif trajectory=='interrupted':interrupted_ref=common._save_state(output/'INTERRUPTED_STEP1.pt',saved,torch=torch,
                maximum=launch['max_checkpoint_bytes'],counters=counters)
            else:
                _require(saved['contract_sha256']==reference_sha,'BIT_EXACT_COMPLETE_RESUME')
                full_state=common._save_state(output/'RESUMED_STEP2.pt',saved,torch=torch,
                    maximum=launch['max_checkpoint_bytes'],counters=counters);parity=True
            del saved,model,optimizer;model=optimizer=None;gc.collect();torch.cuda.empty_cache()
        expected={'model_constructions':3,'model_construction_returns':3,'optimizer_constructions':3,
            'forwards':4,'forward_returns':4,'backwards':4,'backward_returns':4,'optimizer_step_calls':4,
            'optimizer_updates':4,'staged_state_load_calls':3,'staged_state_load_returns':3,
            'resume_state_load_calls':1,'resume_state_load_returns':1,'forbidden_legacy_calls':0}
        _require(all(counters[k]==v for k,v in expected.items()) and evaluation['forward_calls']==17
            and evaluation['forward_returns']==17 and tree_sha(batches)==batch_sha,'EXACT_WORK')
        common._freeze(launch,binding,context);unchanged=True
    except BaseException as exc:
        error=type(exc).__name__+':'+str(exc)
        try:common._freeze(launch,binding,context);unchanged=True
        except Exception:unchanged=False
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_handler);sys.setprofile(previous)
    after=common._io()
    if after['rchar']>io_budget:error='RCHAR_BUDGET_EXCEEDED'
    passed=initial_checked and parity and unchanged and error is None and (not decimal or (decimal_prior_checked and (candidate or decimal_initial_checked)))
    outcome='COMPLETE' if passed else 'FAILED'
    result=common._json(output/'RESULT.json',{'schema_version':('gx1_native_m1_price_private_qualification_result_v1' if candidate
        else 'gx1_native_decimal_dense_private_qualification_result_v1' if decimal else 'gx1_native_dense_private_qualification_result_v1'),
        'decision':(('PASS_M1_PRICE_PRIVATE_MECHANICS' if candidate else 'PASS_DECIMAL_DENSE_NATIVE_MECHANICS' if decimal
            else 'PASS_DENSE_NATIVE_MECHANICS') if passed else 'FAILED'),
        **({'decimal_input_audit':launch['decimal_input_audit'],'decimal_target_refs':launch['decimal_target_refs'],
            'decimal_initial_predictions':launch['decimal_initial_predictions'],
            'decimal_initial_prediction_byte_parity':decimal_initial_checked,
            'decimal_prior_against_sign_audit':decimal_prior_checked} if decimal and not candidate else {}),
        **({'candidate_mode':launch['candidate_mode'],'candidate_m1_refs':launch.get('candidate_m1_refs'),
            'own_candidate_initial_measured':initial_checked,'previous_model_initial_used':False,
            'decimal_input_audit':launch['decimal_input_audit'],'decimal_target_refs':launch['decimal_target_refs'],
            'decimal_prior_against_sign_audit':decimal_prior_checked} if candidate else {}),'recipe':window['recipe'],
        'execution_binding':binding_ref,'counts':{k:v for k,v in counters.items() if k!='json_bytes'},
        'initial_evaluation_counts':evaluation,'initial_state':initial,'initial_identity_sha256':initial_identity,
        'initial_predictions':locals().get('initial_prediction'),'own_prefix_prior':prior,
        'dense_target':context['dense_target'],'source_transition_proof':launch['dense_source_transition_proof'],
        'final_state':full_state,'initial_checked':initial_checked,'bit_exact_resume':parity,
        'steps':steps,'batch_gc_support':batch_support,'parent_order_sha256':order_sha,'parent_cursor':cursor,
        'planned_unique_parents':32,'planned_logical_optimizer_steps':2,'physical_optimizer_updates':counters['optimizer_updates'],
        'source_unchanged':unchanged,'metadata_bookends':context['info']['metadata'],
        'payload_stat_bookends':context['ancestor_plan']['payloads'],'forbidden_open_paths':denied,
        'limits':limits,'elapsed_seconds':time.monotonic()-started,'io_before':before,'io_after':after,
        'private_weights_for_main_fit':False,'predictive_value_qualified':False,'test_data_used':False,
        'learning_boundary':'Two fixed B16 steps and exact resume, no fit or later predictive-value claim.','error':error},
        counters,launch['max_published_json_bytes'])
    checkpoint=common._json(Path(window['campaign_cursor_path']),authority.build_research_cursor(
        recipe_ref=window['recipe'],outcome=outcome,optimizer_updates=counters['optimizer_updates'],parent_cursor=cursor,
        parent_order_sha256=order_sha,full_state=full_state,result=result,batch_size=16),counters,launch['max_published_json_bytes'])
    progress=authority.build_research_progress(window=window,plan=plan,invocation=invocation,
        checkpoint_pointer=checkpoint,optimizer_updates=counters['optimizer_updates'],parent_cursor=cursor,
        parent_order_sha256=order_sha,outcome=outcome)
    common._json(progress_path,progress,counters,launch['max_published_json_bytes'])
    authority.require_research_progress(progress,plan_sha256=plan['plan_sha256'],invocation=invocation,
        expected_selection_receipt_sha256=None,verify_file=True)
    common._freeze(launch,binding,context)
    if not passed:raise RuntimeError('NATIVE_DENSE_QUALIFICATION_FAILED:'+str(error))
    return progress
