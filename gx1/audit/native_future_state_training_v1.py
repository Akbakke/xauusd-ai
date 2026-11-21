"""Source-fixed paired learning and separately admitted one-epoch future-state TRAIN.

No old raw/CPU/CUDA authority is reopened. Checkpoint trajectory bindings are
independent of an execution's boot/policy so a separately admitted recovery can
validate an authentic original checkpoint without pretending it is a new fit.
"""
from __future__ import annotations
import argparse,copy,hashlib,io,json,signal,subprocess,time
from pathlib import Path
import numpy as np
from gx1.audit import qualify_native_m5_full_gc_direction_v1 as core

REPO=core.REPO;POLICY=core.POLICY
ROOT=Path('/home/andre2/GX1_RUNS/GX1_FUTURE_STATE_TRAIN_READINESS_20261121_001')
PARENT=core.FUTURE_CUDA_PARENT
MODULE='gx1.audit.native_future_state_training_v1'
SELF='gx1/audit/native_future_state_training_v1.py';TEST='tests/test_native_future_state_training_v1.py'
PAIR_PLAN=REPO/'configs/research/NATIVE_FUTURE_STATE_PAIRED_LEARNING_20261121_001.json'
PAIR_ROOT=Path('/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_PAIRED_LEARNING_20261121_001')
MAIN_PLAN=REPO/'configs/research/NATIVE_FUTURE_STATE_ONE_EPOCH_20261121_001.json'
MAIN_ROOT=Path('/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_ONE_EPOCH_20261121_001')
RECOVERY_PLAN=REPO/'configs/research/NATIVE_FUTURE_STATE_RECOVERY_PROBE_20261121_001.json'
RECOVERY_ROOT=Path('/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_RECOVERY_PROBE_20261121_001')
RESUME_PLAN=REPO/'configs/research/NATIVE_FUTURE_STATE_ONE_EPOCH_RESUME_20261121_001.json'
RESUME_ROOT=Path('/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_ONE_EPOCH_RESUME_20261121_001')
DESIGN={'path':str(ROOT/'PAIRED_LEARNING_DESIGN001.json'),'sha256':'fb7f221c9fb93bdfa110c5fd106817d51f762422425d8e25d6edbbc5bda147af'}
INITIAL_RESULT={'path':'/home/andre2/GX1_RUNS/NATIVE_M5_FUTURE_STATE_CUDA_INITIAL_20261121_001/RESULT.json','sha256':'a1b7716abc27047eaf3d940f83474159ce05abf61396e037690c62529720c1e1'}
INITIAL_TERMINAL={'path':'/home/andre2/GX1_RUNS/NATIVE_M5_FUTURE_STATE_CUDA_INITIAL_20261121_001/TERMINAL.json','sha256':'1a6372bb1356a32eebdeb3d1b0956bd40633d66839de30eabe52afef2ec97419'}
TRAIN_ROWS=354550;TRAIN_SHA=core.FUTURE_CPU_TRAIN_SHA;UPPER=core.FUTURE_CPU_TRAIN_UPPER
START=core.FUTURE_CPU_TRAIN_START;DAY=86400000000000;ORIGIN=79200000000000
TF=('M5','M15','H1','H4','D1');TRUTH=core.FUTURE_CPU_TRUTH_KEYS
TRAIN_BATCHES=(64,96,128);EVAL_BATCHES=(64,256)
LANES={'joint':None,'joint_replay':None,'m5_primary':'M5_primary','m5_primary_replay':'M5_primary','main':None,'recovery_probe':None,'main_resume':None}
GUARD_PATHS={'gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh'}
read,ref,publish,sha=core.future_cpu_read,core.ref,core.publish,core.sha


def need(ok,reason):
    if not bool(ok):raise ValueError('NATIVE_FUTURE_TRAIN_'+reason)


def seal(value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return {**value,'contract_sha256':hashlib.sha256(raw).hexdigest()}


def array_sha(value):return core.digest(value)


def select_study_clock_rows(context):
    """Clock-only phase. This function never accesses context['truth']."""
    provider=context['provider'];full=np.flatnonzero((provider.issue_ns>=START)&(provider.issue_ns<UPPER)).astype(np.int64)
    issues=provider.issue_ns[full]
    need(len(full)==TRAIN_ROWS and issues.dtype==np.int64 and np.all(np.diff(issues)>0)
        and np.all(issues%300000000000==0)and array_sha(issues)==TRAIN_SHA and provider.input_ready[full].all(),'WHOLE_UNFILTERED_TRAIN_CLOCK')
    if 'rows'in context:need(np.array_equal(context['rows'],full),'CONTEXT_FULL_CLOCK')
    buckets=(issues-ORIGIN)//DAY;begins=np.r_[0,np.flatnonzero(np.diff(buckets))+1];ends=np.r_[begins[1:],len(full)]
    ranks=np.concatenate([a+np.linspace(0,b-a-1,min(16,b-a),dtype=np.int64)for a,b in zip(begins,ends)])
    need(len(ranks)>0 and np.all(np.diff(ranks)>0),'SOURCE_SELECTION_UNIQUE_ORDER')
    positions=full[ranks];source=np.asarray(provider.source_row[positions],dtype=np.int64);chosen=issues[ranks]
    need(np.all(np.diff(source)>0),'SOURCE_SELECTION_GLOBAL_ORDER')
    manifest=seal({'schema_version':'gx1_native_future_state_source_clock_selection_v1','design':DESIGN,
        'whole_TRAIN_rows':TRAIN_ROWS,'whole_TRAIN_issue_ns_sha256':TRAIN_SHA,
        'selector':'bucket=floor((issue_ns-22h)/24h);min16_linspace_int64_including_endpoints',
        'bucket_origin_ns':ORIGIN,'bucket_duration_ns':DAY,'nonempty_buckets':len(begins),'rows':len(ranks),
        'selected_positions_sha256':array_sha(positions),'selected_ranks_sha256':array_sha(ranks),
        'selected_source_rows_sha256':array_sha(source),'selected_issue_ns_sha256':array_sha(chosen),
        'first_issue_ns':int(chosen[0]),'last_issue_ns':int(chosen[-1]),
        'bucket_sizes':[int(b-a)for a,b in zip(begins,ends)],
        'selected_per_bucket':[min(16,int(b-a))for a,b in zip(begins,ends)],
        'target_filter':False,'class_or_observability_quotas':False,'top_up':False})
    for a in (positions,source,chosen):a.flags.writeable=False
    return {'manifest':manifest,'selected_positions':positions,'selected_source_rows':source,'selected_issue_ns':chosen}


def support_by_tf(truth,target_contract):
    from gx1.contracts.market_state_ta_integrated_objective_v1 import _native_future_state_truth
    masks=_native_future_state_truth(truth,rows=len(truth['issue_ns']),fold_upper_ns=UPPER,target_contract=target_contract)
    issue=truth['issue_ns'];days=issue.astype('datetime64[ns]').astype('datetime64[D]').astype(np.int64)
    years=issue.astype('datetime64[ns]').astype('datetime64[Y]').astype(np.int64)+1970;weekdays=(days+3)%7
    def counts(j,take):
        a=masks['activity'][:,j]&take;s=masks['side'][:,j]&take;one=masks['singleton'][:,j]&take
        return {'rows':int(take.sum()),'activity_known':int(a.sum()),'activity_unknown':int((take&~masks['activity'][:,j]).sum()),
            'ACTIVE':int((a&truth['activity'][:,j]).sum()),'FLAT':int((a&~truth['activity'][:,j]).sum()),
            'side_known':int(s.sum()),'UP':int((s&truth['side_up'][:,j]).sum()),'DOWN':int((s&~truth['side_up'][:,j]).sum()),
            'active_side_unknown':int((a&truth['activity'][:,j]&~s).sum()),'singleton':int(one.sum()),
            'unique_activity_targets':int(np.unique(truth['target_start_ns'][a,j]).size),
            'unique_side_targets':int(np.unique(truth['target_start_ns'][s,j]).size),
            'unique_singleton_targets':int(np.unique(truth['target_start_ns'][one,j]).size)}
    result={}
    for j,tf in enumerate(TF):
        base=counts(j,np.ones(len(issue),np.bool_));base['by_UTC_issue_year_weekday']={}
        for y in np.unique(years):
            for w in range(7):
                take=(years==y)&(weekdays==w)
                if take.any():base['by_UTC_issue_year_weekday'][str(int(y))+'|'+str(w)]=counts(j,take)
        present=truth['target_present'][:,j];times=truth['target_start_ns'][:,j]
        td=times.astype('datetime64[ns]').astype('datetime64[D]').astype(np.int64);ty=times.astype('datetime64[ns]').astype('datetime64[Y]').astype(np.int64)+1970
        base['by_UTC_target_year_weekday']={}
        for y in np.unique(ty[present]):
            for w in range(7):
                take=present&(ty==y)&((td+3)%7==w)
                if take.any():base['by_UTC_target_year_weekday'][str(int(y))+'|'+str(w)]=counts(j,take)
        result[tf]=base
    return result


def make_schedule(source_rows,issue_ns,*,batch_rows,passes,known_activity_by_batch):
    need(type(batch_rows)is int and batch_rows in TRAIN_BATCHES and type(passes)is int and passes in (1,2),'SCHEDULE_GEOMETRY')
    n=len(source_rows);batches=core.all_score_batches(np.arange(n,dtype=np.int64),batch_rows)
    need(len(known_activity_by_batch)==len(batches)and all(type(v)is int and v>=0 for v in known_activity_by_batch),'SCHEDULE_SUPPORT')
    update=[v>0 for v in known_activity_by_batch]*passes
    return seal({'schema_version':'gx1_native_future_state_training_schedule_v1','rows':n,'passes':passes,
        'batch_rows':batch_rows,'batches_per_pass':len(batches),'logical_batches':len(batches)*passes,
        'source_rows_sha256':array_sha(source_rows),'issue_ns_sha256':array_sha(issue_ns),
        'batch_source_rows_sha256':[array_sha(source_rows[b])for b in batches],
        'batch_issue_ns_sha256':[array_sha(issue_ns[b])for b in batches],
        'known_activity_by_batch':known_activity_by_batch,'update_flags':update,
        'optimizer_updates':sum(update),'skipped_batches':len(update)-sum(update),'shuffle':False,'drop_last':False})


def prepare_study_cohort_from_context(context,*,clock_selection,frozen_selection_ref,target_contract):
    """Second phase: caller durably publishes the clock manifest before this call."""
    from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
    from gx1.contracts.market_state_native_m5_targets_v1 import require_fully_future_state_contract
    check=select_study_clock_rows(context)
    need(check['manifest']==clock_selection['manifest']and all(np.array_equal(check[k],clock_selection[k])for k in
        ('selected_positions','selected_source_rows','selected_issue_ns')),'FROZEN_CLOCK_SELECTION')
    need(type(frozen_selection_ref)is dict and set(frozen_selection_ref)=={'path','sha256'}
        and Path(frozen_selection_ref['path']).is_absolute()and Path(frozen_selection_ref['path']).suffix=='.json'
        and len(frozen_selection_ref['sha256'])==64,'SELECTION_PUBLICATION_REFERENCE')
    target=require_fully_future_state_contract(target_contract);factory=objective.require_native_future_state_train_prior(context['prior'],target_contract=target)
    pop=context['normalization']['fit_population'];p=factory['fit_population']
    need(pop['issue_rows']==p['issue_rows']==TRAIN_ROWS and pop['selected_issue_ns_sha256']==p['issue_ns_sha256']==TRAIN_SHA
        and pop['cutoff_ns']==p['cutoff_ns']==UPPER,'FACTORY_PRIOR_WHOLE_TRAIN_ONLY')
    full=context['truth'];whole=context['rows'];positions=np.searchsorted(whole,check['selected_positions'])
    need(np.array_equal(whole[positions],check['selected_positions'])and np.array_equal(full['issue_ns'],context['provider'].issue_ns[whole])
        and objective._native_prior_array_sha(full)==factory['fit_population']['truth_sha256'],'FULL_TRUTH_AXIS_FACTORY_PRIOR')
    truth={k:np.ascontiguousarray(full[k][positions]).copy()for k in TRUTH}
    masks=objective._native_future_state_truth(truth,rows=len(positions),fold_upper_ns=UPPER,target_contract=target)
    geometries={}
    for b in TRAIN_BATCHES:
        batches=core.all_score_batches(np.arange(len(positions),dtype=np.int64),b)
        known=[int(masks['activity'][take].sum())for take in batches]
        geometries[str(b)]={'batch_rows':b,'batches':len(batches),'last_rows':len(batches[-1]),
            'known_activity_by_batch':known,'updates_per_pass':sum(v>0 for v in known),
            'skips_per_pass':sum(v==0 for v in known),'first_two_supported':len(known)>=2 and all(v>0 for v in known[:2]),
            'schedule':make_schedule(check['selected_source_rows'],check['selected_issue_ns'],batch_rows=b,passes=2,known_activity_by_batch=known)}
    full_masks=objective._native_future_state_truth(full,rows=len(whole),fold_upper_ns=UPPER,target_contract=target)
    full_source=context['provider'].source_row[whole];full_issue=context['provider'].issue_ns[whole]
    need(array_sha(full_source)==pop['selected_source_rows_sha256'],'WHOLE_TRAIN_SOURCE_ROWS')
    whole_geometries={}
    for b in TRAIN_BATCHES:
        batches=core.all_score_batches(np.arange(len(whole),dtype=np.int64),b)
        known=[int(full_masks['activity'][take].sum())for take in batches]
        whole_geometries[str(b)]={'batch_rows':b,'batches':len(batches),'last_rows':len(batches[-1]),
            'known_activity_by_batch':known,'updates_per_pass':sum(v>0 for v in known),'skips_per_pass':sum(v==0 for v in known),
            'schedule':make_schedule(full_source,full_issue,batch_rows=b,passes=1,known_activity_by_batch=known)}
    matched=objective.fit_native_future_state_train_prior(truth=truth,fold_upper_ns=UPPER,target_contract=target)
    constants={}
    for name,prior in (('whole_TRAIN_factory',factory),('matched_selected_SCORING_ONLY',matched)):
        constants[name]=constant_prior_scores(prior,truth,target)
    for a in truth.values():a.flags.writeable=False
    return {'selection':check['manifest'],'selection_ref':frozen_selection_ref,
        **{k:check[k]for k in ('selected_positions','selected_source_rows','selected_issue_ns')},
        'truth':truth,'support':support_by_tf(truth,target),'batch_geometries':geometries,
        'whole_TRAIN_batch_geometries':whole_geometries,
        'evaluation_geometries':{str(b):{'batches':(len(positions)+b-1)//b,'last_rows':(len(positions)-1)%b+1}for b in EVAL_BATCHES},
        'matched_scoring_prior':matched,'factory_prior':factory,'constant_reference_scores':constants,
        'truth_sha256':objective._native_prior_array_sha(truth),'matched_scoring_prior_fits':1,'factory_prior_fits':0}


def constant_prior_scores(prior,truth,target):
    """Score identified primitives; zero-support fitted probabilities stay unknown."""
    import torch
    from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
    prior=objective.require_native_future_state_train_prior(prior,target_contract=target)
    missing=[(task,tf)for task in ('activity','side')for tf in TF
        if prior['per_task'][task][tf]['fit_status']!='FITTED' or prior['per_task'][task][tf]['count']==0]
    if not missing:
        logits=objective.native_future_state_train_prior_logits(prior,rows=len(truth['issue_ns']),target_contract=target)
        return objective.native_future_state_score_sums(model_output=logits,truth=truth,fold_upper_ns=UPPER,target_contract=target)
    # Finite internal placeholders enable reuse of masked proper-score algebra.
    # They are not estimated priors and no dependent full-class score survives.
    logits={}
    for task in ('activity','side'):
        values=np.zeros((len(truth['issue_ns']),len(TF)),np.float32)
        for j,tf in enumerate(TF):
            probabilities=prior['per_task'][task][tf]['probabilities']
            if (task,tf)not in missing:values[:,j]=np.float32(np.log(probabilities[1])-np.log(probabilities[0]))
        logits[task+'_logits']=torch.from_numpy(values)
    scores=objective.native_future_state_score_sums(model_output=logits,truth=truth,fold_upper_ns=UPPER,target_contract=target)
    for task,tf in missing:
        cell=scores['per_tf'][tf][task]
        cell.update(log_loss_sum=None,brier_sum=None,estimated_probability_identified=False,
            undefined_reason='UNFITTED_NO_PRIOR_OBSERVATIONS',fit_status=prior['per_task'][task][tf]['fit_status'])
        if cell['count']:
            scores['per_tf'][tf]['partial_log_loss_sum']=None
    for tf in TF:
        if any((task,tf)in missing for task in ('activity','side')):
            scores['per_tf'][tf]['three_class'].update(log_loss_sum=None,brier_sum=None,
                undefined_reason='UNFITTED_PRIMITIVE_DISTRIBUTION_NOT_IDENTIFIED')
    scores['UNFITTED_tasks']=[task+'.'+tf for task,tf in missing]
    scores['unfitted_placeholders_are_not_estimates']=True
    return scores


def binary_rank_metrics(logits,labels):
    x=np.asarray(logits);y=np.asarray(labels)
    need(x.ndim==y.ndim==1 and len(x)==len(y)and y.dtype==np.bool_ and np.isfinite(x).all(),'BINARY_METRIC_DOMAIN')
    n=len(x);positive=int(y.sum());negative=n-positive;auc=None
    if positive and negative:
        order=np.argsort(x,kind='mergesort');sorted_x=x[order];sorted_y=y[order];rank_sum=0.
        begins=np.r_[0,np.flatnonzero(sorted_x[1:]!=sorted_x[:-1])+1];ends=np.r_[begins[1:],n]
        for a,b in zip(begins,ends):rank_sum+=float(sorted_y[a:b].sum())*((a+1+b)/2.)
        auc=(rank_sum-positive*(positive+1)/2.)/(positive*negative)
    return {'count':n,'positive':positive,'negative':negative,'AUC':None if auc is None else float(auc),
        'AUC_undefined_reason':None if auc is not None else'one_or_zero_classes',
        'unique_float32_logits':int(np.unique(x.astype(np.float32)).size),'nonconstant':bool(n and np.unique(x.astype(np.float32)).size>1),
        'minimum':float(x.min())if n else None,'maximum':float(x.max())if n else None,
        'standard_deviation':float(x.astype(np.float64).std())if n else None}


def rank_metrics(output,truth,target):
    from gx1.contracts.market_state_ta_integrated_objective_v1 import _native_future_state_truth
    masks=_native_future_state_truth(truth,rows=len(truth['issue_ns']),fold_upper_ns=UPPER,target_contract=target)
    report={}
    for j,tf in enumerate(TF):
        report[tf]={}
        for task,label in (('activity','activity'),('side','side_up')):
            take=masks[task][:,j];report[tf][task]=binary_rank_metrics(output[task+'_logits'][take,j],truth[label][take,j])
    a=output['activity_logits'].astype(np.float64);z=output['side_logits'].astype(np.float64)
    loga=-np.logaddexp(0.,-a);logflat=-np.logaddexp(0.,a)
    classes=np.stack((loga-np.logaddexp(0.,z),logflat,loga-np.logaddexp(0.,-z)),axis=-1)
    for j,tf in enumerate(TF):
        values=classes[masks['singleton'][:,j],j];maximum=values.max(1,keepdims=True)if len(values)else values
        ties=(values==maximum).sum(1)if len(values)else np.empty(0,np.int64)
        unique=ties==1
        report[tf]['singleton_argmax']={'count':len(values),'ties_without_direction_credit':int((ties>1).sum()),
            'unique_argmax_DOWN_FLAT_UP':np.bincount(values[unique].argmax(1),minlength=3).tolist()if unique.any()else[0,0,0]}
    return report



def verify_initial_source_bridge(inputs):
    need(inputs['initial_result']==INITIAL_RESULT and inputs['initial_terminal']==INITIAL_TERMINAL,'EXACT_INITIAL_REFERENCES')
    result,terminal=core.future_cuda_complete(INITIAL_RESULT,INITIAL_TERMINAL,'COMPLETE_NATIVE_FUTURE_STATE_CUDA_INITIAL')
    plan=read(result['plan']);owners=plan['source_owners']
    need(type(owners)is dict and len(owners)==58,'INITIAL_SOURCE_INVENTORY')
    for name,digest in owners.items():
        path=Path(name)
        need(type(name)is str and not path.is_absolute()and '..'not in path.parts and str(path)==name
            and path.suffix in ('.py','.sh')and type(digest)is str and len(digest)==64
            and all(c in '0123456789abcdef'for c in digest),'INITIAL_SOURCE_SCHEMA')
    from gx1.contracts.gx1_capped_execution_v1 import native_future_state_guard_source_bridge
    proof=native_future_state_guard_source_bridge(owners,producer_commit=plan['source_base_commit'])
    need(GUARD_PATHS<=set(owners)and all(sha(REPO/k)==v for k,v in owners.items()if k not in GUARD_PATHS),
        'INITIAL_OTHER56_SOURCE_UNCHANGED')
    need(result['report']['whole_TRAIN_initialbaseline_qualified']is True
        and result['report']['initial']['rows']==TRAIN_ROWS and result['report']['initial']['initial_prior_exact']is True
        and result['report']['counts']['optimizer_steps']==0,'COMPLETE_WHOLE_TRAIN_INITIAL')
    return {'result':INITIAL_RESULT,'terminal':INITIAL_TERMINAL,'plan':result['plan'],
        'guard_bridge':proof,'source_owners':owners,'metadata':plan['metadata'],
        'initial_state_sha256':result['report']['initial_state_sha256'],
        'initial_score':{k:v for k,v in result['report']['initial'].items()if k!='batches'}}


def release_phase_allocator_cache(model,optimizer,*,label):
    """Release unused CUDA cache between returned phases, preserving live state."""
    import torch
    need(type(label)is str and bool(label),'ALLOCATOR_PHASE_LABEL')
    parameter=next(model.parameters(),None)
    if parameter is None or parameter.device.type!='cuda':
        return {'label':label,'applied':False,'reason':'MODEL_NOT_CUDA'}
    device=parameter.device
    need(all(p.grad is None for p in model.parameters()),'ALLOCATOR_PHASE_GRAD_NONE')
    def memory():
        return {'allocated_bytes':int(torch.cuda.memory_allocated(device)),
            'reserved_bytes':int(torch.cuda.memory_reserved(device)),
            'process_max_allocated_bytes':int(torch.cuda.max_memory_allocated(device)),
            'process_max_reserved_bytes':int(torch.cuda.max_memory_reserved(device))}
    def proof():
        return {'model':core.state_sha({'state_dict':model.state_dict(),'training':model.training}),
            'AdamW':core.state_sha(optimizer.state_dict()),'full_rng':core.state_sha(core.training_rng())}
    torch.cuda.synchronize(device);before=memory();state_before=proof()
    torch.cuda.empty_cache();torch.cuda.synchronize(device)
    after=memory();state_after=proof()
    need(after['allocated_bytes']==before['allocated_bytes'],'ALLOCATOR_PHASE_LIVE_ALLOCATION_UNCHANGED')
    need(after['reserved_bytes']<=before['reserved_bytes'],'ALLOCATOR_PHASE_RESERVED_NONINCREASE')
    need(state_after==state_before and all(p.grad is None for p in model.parameters()),'ALLOCATOR_PHASE_FUNCTION_STATE_UNCHANGED')
    return {'label':label,'applied':True,'device':str(device),'before':before,'after':after,
        'state_before':state_before,'state_after':state_after,'peak_counters_reset':False}


def own_numeric_ast():
    import ast
    tree=ast.parse((REPO/SELF).read_text());nodes={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    names=('select_study_clock_rows','prepare_study_cohort_from_context','make_schedule','diagnostic_panels',
        'constant_prior_scores','constant_rank_metrics','binary_rank_metrics','rank_metrics','paired_final_deltas','score_with_capture','score_point','packet',
        'load_context','training_step','position','make_checkpoint','checkpoint_proof','strict_restore',
        'runtime_compatible','initial_identity','release_phase_allocator_cache','run_training_loop','physical','recovery_physical')
    result={n:hashlib.sha256(ast.dump(nodes[n],include_attributes=False).encode()).hexdigest()for n in names}
    result['numeric_constants']=seal({'TRAIN_ROWS':TRAIN_ROWS,'TRAIN_SHA':TRAIN_SHA,'UPPER':UPPER,'START':START,
        'DAY':DAY,'ORIGIN':ORIGIN,'TF':list(TF),'TRUTH':list(TRUTH),'TRAIN_BATCHES':list(TRAIN_BATCHES),'EVAL_BATCHES':list(EVAL_BATCHES)})['contract_sha256']
    return result


def trajectory_bindings(plan,arm):
    policy=None if arm=='joint'else'M5_primary';meta=plan['metadata']
    keys=('source_snapshot','normalization','input_binding_ref','input_binding','prior','dataset',
        'native_manifest','target_contract','architecture')
    return seal({'schema_version':'gx1_native_future_state_training_trajectory_v1','mode':plan['mode'],'arm':arm,
        'backbone_gradient_policy':policy,'function_inputs':{k:meta[k]for k in keys},
        'model_contract':meta['arm_model_contracts'][arm],'cohort':plan['cohort'],
        'recipe':plan['recipe'],'schedule':plan['schedule'],
        'runtime_compatibility':{k:meta['runtime'][k]for k in ('library_versions','gpu')},
        'numeric_source_owners':{k:v for k,v in plan['baseline_bridge']['source_owners'].items()if k not in GUARD_PATHS},
        'training_leaf_AST':plan['training_leaf_AST']})


def runtime_compatible(original,current):
    core.future_cuda_require_runtime(current)
    need(set(original)=={'library_versions','gpu'}and all(original[k]==current[k]for k in original),
        'RECOVERY_RUNTIME_COMPATIBILITY')
    # Boot, observation, policy and execution-plan identities are provenance,
    # never silently substituted for immutable function/trajectory identity.
    return True


def position(schedule,cursor):
    need(type(cursor)is int and 0<=cursor<=schedule['logical_batches'],'LOGICAL_CURSOR')
    return {'logical_cursor':cursor,'completed_passes':cursor//schedule['batches_per_pass'],
        'next_batch_in_pass':cursor%schedule['batches_per_pass'],
        'optimizer_updates':sum(schedule['update_flags'][:cursor]),
        'skipped_batches':cursor-sum(schedule['update_flags'][:cursor])}


def make_checkpoint(model,optimizer,*,trajectory,original_plan_ref,lane,cursor,steps,scores,producer_runtime):
    return {'schema_version':'gx1_native_future_state_training_checkpoint_v1','trajectory':trajectory,
        'producer':{'original_plan':original_plan_ref,'lane':lane,'runtime':producer_runtime},
        'position':position(trajectory['schedule'],cursor),'model_state':model.state_dict(),
        'optimizer_state':optimizer.state_dict(),'scheduler_state':None,'rng_state':core.training_rng(),
        'training':model.training,'optimizer_parameter_names':[n for n,_ in model.named_parameters()],
        'steps':steps,'scores':scores}


def checkpoint_proof(checkpoint):
    keys=('trajectory','position','model_state','optimizer_state','scheduler_state','rng_state',
        'training','optimizer_parameter_names','steps','scores')
    return {k:core.state_sha(checkpoint[k])for k in keys}


def strict_restore(model,optimizer,checkpoint,*,trajectory,original_plan_ref,expected_cursor,current_runtime,expected_state_proof):
    """Validate authentic original trajectory before any parameter copy.

    A new execution may have a new boot/policy/plan. Its admission must bind the
    original plan and checkpoint hash; those are never replaced with new refs.
    """
    import torch
    keys={'schema_version','trajectory','producer','position','model_state','optimizer_state','scheduler_state',
        'rng_state','training','optimizer_parameter_names','steps','scores'}
    need(type(checkpoint)is dict and set(checkpoint)==keys
        and checkpoint['schema_version']=='gx1_native_future_state_training_checkpoint_v1'
        and checkpoint['trajectory']==trajectory and checkpoint['producer']['original_plan']==original_plan_ref
        and checkpoint['position']==position(trajectory['schedule'],expected_cursor)
        and all(type(v)is int for v in checkpoint['position'].values())
        and checkpoint['scheduler_state']is None and checkpoint['training']is True
        and len(checkpoint['steps'])==expected_cursor,'CHECKPOINT_TRAJECTORY_CURSOR')
    need(type(expected_state_proof)is dict and checkpoint_proof(checkpoint)==expected_state_proof,
        'AUTHENTIC_ORIGINAL_CHECKPOINT_STATE_PROOF_BEFORE_COPY')
    runtime_compatible(trajectory['runtime_compatibility'],current_runtime)
    need(checkpoint['optimizer_parameter_names']==[n for n,_ in model.named_parameters()],
        'CHECKPOINT_PARAMETER_NAMES')
    expected=model.state_dict();state=checkpoint['model_state']
    from collections.abc import Mapping
    need(isinstance(state,Mapping)and set(state)==set(expected),'CHECKPOINT_MODEL_KEYS')
    for name,value in state.items():
        need(isinstance(value,torch.Tensor)and value.shape==expected[name].shape and value.dtype==expected[name].dtype
            and value.device.type!='meta'and bool(torch.isfinite(value).all()),'CHECKPOINT_MODEL_HEADER')
    for name,value in model.named_buffers():
        if name in expected:need(torch.equal(state[name].detach().cpu(),value.detach().cpu()),'CHECKPOINT_MODEL_IDENTITY')
    opt=checkpoint['optimizer_state'];fresh=optimizer.state_dict();parameters=list(model.parameters())
    need(type(opt)is dict and set(opt)=={'state','param_groups'}and len(opt['param_groups'])==1
        and opt['param_groups']==fresh['param_groups'],'CHECKPOINT_OPTIMIZER_GROUPS')
    ids=opt['param_groups'][0]['params'];updates=checkpoint['position']['optimizer_updates']
    need(len(ids)==len(parameters)and len(set(ids))==len(ids)and all(type(i)is int for i in ids)
        and all(type(i)is int for i in opt['state'])and set(opt['state'])<=set(ids),
        'CHECKPOINT_OPTIMIZER_IDS')
    for index,item in opt['state'].items():
        parameter=parameters[ids.index(index)]
        need(type(item)is dict and set(item)=={'step','exp_avg','exp_avg_sq'}
            and all(isinstance(v,torch.Tensor)and v.device.type=='cpu'and bool(torch.isfinite(v).all())for v in item.values())
            and item['step'].numel()==1 and item['step'].dtype==torch.float32
            and float(item['step']).is_integer()and 1<=float(item['step'])<=updates
            and all(item[k].shape==parameter.shape and item[k].dtype==parameter.dtype for k in ('exp_avg','exp_avg_sq')),
            'CHECKPOINT_OPTIMIZER_SLOTS')
        need(bool((item['exp_avg_sq']>=0).all()),'CHECKPOINT_NONNEGATIVE_SECOND_MOMENT')
    heads={ids[i]for i,(name,p)in enumerate(model.named_parameters())if name.startswith('readout.')}
    need((not updates and not opt['state'])or(heads and heads<=set(opt['state'])
        and all(float(opt['state'][i]['step'])==updates for i in heads)),'CHECKPOINT_READOUT_STEP_POSITION')
    rng=checkpoint['rng_state'];before=core.training_rng()
    need(type(rng)is dict and set(rng)==set(before)and type(rng['torch_cuda'])is list
        and len(rng['torch_cuda'])==len(before['torch_cuda'])
        and all(isinstance(v,torch.Tensor)and v.dtype==torch.uint8 and v.ndim==1 and v.device.type=='cpu'
            for v in rng['torch_cuda']),'CHECKPOINT_RNG_HEADER')
    try:
        core.restore_training_rng(rng);need(core.state_sha(core.training_rng())==core.state_sha(rng),'CHECKPOINT_RNG_ROUNDTRIP')
    finally:core.restore_training_rng(before)
    model.load_state_dict(state,strict=True);optimizer.load_state_dict(opt);model.train(True)
    for p,item in optimizer.state.items():
        need(item['step'].device.type=='cpu'and item['exp_avg'].device==item['exp_avg_sq'].device==p.device,
            'CHECKPOINT_RESTORED_SLOT_DEVICES')
    core.restore_training_rng(rng)
    need(core.state_sha(model.state_dict())==core.state_sha(state)
        and core.state_sha(optimizer.state_dict())==core.state_sha(opt)
        and core.state_sha(core.training_rng())==core.state_sha(rng),'CHECKPOINT_FULL_RESTORED_STATE')
    return checkpoint['position']


def score_with_capture(model,packets,plan,factory_prior,scoring_prior,truth,*,initial):
    from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
    captured={}
    def hook(module,args,output):
        for k,v in output.items():captured.setdefault(k,[]).append(v.detach().cpu().numpy().copy())
    handle=model.register_forward_hook(hook)
    try:base=core.future_cuda_score(model,packets,plan,factory_prior,initial=initial,equal_tf=True)
    finally:handle.remove()
    output={k:np.concatenate(v)for k,v in captured.items()}
    need(len(next(iter(output.values())))==len(truth['issue_ns']),'SCORE_CAPTURE_AXIS')
    constants={}
    for name,prior in (('whole_TRAIN_factory',factory_prior),('matched_SCORING_ONLY',scoring_prior)):
        constants[name]=constant_prior_scores(prior,truth,plan['metadata']['target_contract'])
    return {**base,'rank_and_variation':rank_metrics(output,truth,plan['metadata']['target_contract']),
        'constant_references':constants,
        'constant_rank_and_variation':{name:constant_rank_metrics(prior,truth,plan['metadata']['target_contract'])
            for name,prior in (('whole_TRAIN_factory',factory_prior),('matched_SCORING_ONLY',scoring_prior))},
        'capture_used_existing_forwards':True}


def constant_rank_metrics(prior,truth,target):
    from gx1.contracts.market_state_ta_integrated_objective_v1 import _native_future_state_truth
    masks=_native_future_state_truth(truth,rows=len(truth['issue_ns']),fold_upper_ns=UPPER,target_contract=target)
    result={}
    for j,tf in enumerate(TF):
        result[tf]={}
        for task,label in (('activity','activity'),('side','side_up')):
            take=masks[task][:,j];p=prior['per_task'][task][tf]['probabilities']
            cell=prior['per_task'][task][tf]
            if cell['fit_status']!='FITTED' or cell['count']==0:result[tf][task]={'count':int(take.sum()),'AUC':None,'nonconstant':None,'undefined_reason':'UNFITTED_PRIOR_NOT_ESTIMATED'}
            else:result[tf][task]=binary_rank_metrics(np.full(int(take.sum()),np.float32(np.log(p[1])-np.log(p[0]))),truth[label][take,j])
    return result


def paired_final_deltas(joint,primary):
    need(joint['rows']==primary['rows'],'PAIRED_IDENTICAL_ROW_COUNT');report={}
    for tf in TF:
        report[tf]={}
        for task in ('activity','side'):
            a=joint['scores']['per_tf'][tf][task];b=primary['scores']['per_tf'][tf][task]
            ra=joint['rank_and_variation'][tf][task];rb=primary['rank_and_variation'][tf][task]
            need(a['count']==b['count']and a['unique_target_starts']==b['unique_target_starts']
                and(ra['positive'],ra['negative'])==(rb['positive'],rb['negative']),'PAIRED_IDENTICAL_MASK_SUPPORT')
            n=a['count']
            report[tf][task]={'count':n,'NLL_M5primary_minus_joint':(b['log_loss_sum']-a['log_loss_sum'])/n if n else None,
                'Brier_M5primary_minus_joint':(b['brier_sum']-a['brier_sum'])/n if n else None,
                'AUC_M5primary_minus_joint':rb['AUC']-ra['AUC']if ra['AUC']is not None and rb['AUC']is not None else None,
                'joint_nonconstant':ra['nonconstant'],'M5primary_nonconstant':rb['nonconstant']}
    return {'primary_M5_conditional_side':report['M5']['side'],'per_tf':report,
        'point':'fixed_final_after_two_passes','automatic_policy_winner':False,'generalization_or_edge_proven':False}


def training_step(model,optimizer,packet,plan,counts,logical_index):
    from gx1.contracts.market_state_ta_integrated_objective_v1 import _native_future_state_truth
    masks=_native_future_state_truth(packet['truth'],rows=len(packet['truth']['issue_ns']),fold_upper_ns=UPPER,
        target_contract=plan['metadata']['target_contract'])
    if not masks['activity'].any():
        import torch
        optimizer.zero_grad(set_to_none=True);model.train()
        output,_,_=core.future_cuda_forward(model,packet['features'])
        from gx1.contracts.market_state_ta_integrated_objective_v1 import native_future_state_loss
        loss=native_future_state_loss(model_output=output,truth=packet['truth'],fold_upper_ns=UPPER,target_contract=plan['metadata']['target_contract'])
        need(loss['skip_update']and float(loss['loss'])==0.,'UNKNOWN_BATCH_NO_GUESSED_CLASS')
        counts['skipped_batches']+=1
        return {'logical_step':logical_index+1,'skip_update':True,'loss':0.,'support':{k:v.cpu().tolist()for k,v in loss['counts'].items()},
            'output_sha256':{k:array_sha(v.detach().cpu().numpy())for k,v in output.items()}}
    result=core.future_cuda_step(model,optimizer,packet,plan,counts,logical_index)
    return {**result,'skip_update':False}



def expected_source_names():
    plan=read(read(INITIAL_RESULT)['plan'])
    return set(plan['source_owners'])|{SELF,TEST,'gx1/audit/measure_native_future_state_throughput_v1.py',
        'tests/test_native_future_state_throughput_v1.py'}


def require_source_inventory(owners):
    need(type(owners)is dict and set(owners)==expected_source_names(),'EXACT_SOURCE_INVENTORY')
    for name,value in owners.items():
        p=Path(name)
        need(type(name)is str and not p.is_absolute()and '..'not in p.parts and str(p)==name
            and p.suffix in ('.py','.sh')and type(value)is str and len(value)==64
            and all(c in '0123456789abcdef'for c in value),'SOURCE_SCHEMA')
        path=REPO/p;need(path.is_file()and not path.is_symlink()and path.resolve()==path,'SOURCE_FILE')


def validate_training_flags(policy,*,main,admitted):
    expected={k:False for k in ('training_enabled','native_training_enabled','native_launch_authorized','full_epoch_training_allowed','full_val_allowed')}
    if admitted:
        for k in ('training_enabled','native_training_enabled'):expected[k]=True
        if main:
            expected['native_launch_authorized']=True;expected['full_epoch_training_allowed']=True
    need(all(policy.get(k)is v for k,v in expected.items()),'EXACT_MODE_TRAINING_AUTHORITY')


def selected_policy_evidence(inputs,policy,meta,recipe):
    arm='joint'if policy is None else'm5_primary'
    a,_=core.future_cuda_complete(inputs['selected_arm_result'],inputs['selected_arm_terminal'],'COMPLETE_NATIVE_FUTURE_STATE_PAIRED_LEARNING')
    b,_=core.future_cuda_complete(inputs['selected_arm_replay_result'],inputs['selected_arm_replay_terminal'],'COMPLETE_NATIVE_FUTURE_STATE_PAIRED_LEARNING')
    need(inputs['selected_arm_result']['path']==str(PAIR_ROOT/arm/'RESULT.json')
        and inputs['selected_arm_replay_result']['path']==str(PAIR_ROOT/(arm+'_replay')/'RESULT.json')
        and a['plan']==b['plan']and a['report']['scores']['INITIAL']['initial_prior_exact']is True
        and b['report']['cold_replay_bit_exact']is True
        and a['report']['trajectory']==b['report']['trajectory'],'SELECTED_POLICY_OWN_REPLAY')
    original=read(a['plan']);decision=read(inputs['policy_decision'])
    need(a['report']['fresh_initial']['policy_normalized_comparison_sha256']==original['baseline_bridge']['initial_state_sha256']
        and original['baseline_bridge']['result']==INITIAL_RESULT,'SELECTED_POLICY_FORWARD_INITIAL_IDENTITY')
    need(original['metadata']['arm_model_contracts'][arm]==meta['arm_model_contracts'][arm]
        and original['recipe']['train_batch_rows']==recipe['train_batch_rows']
        and original['recipe']['eval_batch_rows']==recipe['eval_batch_rows']
        and decision['schema_version']=='gx1_native_future_state_gradient_policy_decision_v1'
        and decision['backbone_gradient_policy']==policy and decision['paired_plan']==a['plan']
        and decision['selected_arm_result']==inputs['selected_arm_result']
        and decision['selected_arm_replay_result']==inputs['selected_arm_replay_result']
        and type(decision['main_epochs'])is int and decision['main_epochs']==1 and decision['CONTROL_VAL_TEST_used']is False,
        'EXPLICIT_REVIEWED_ONE_EPOCH_POLICY')
    return {'decision':inputs['policy_decision'],'arm_result':inputs['selected_arm_result'],
        'arm_replay_result':inputs['selected_arm_replay_result'],'paired_plan':a['plan']}


def metadata(inputs,*,mode,train_batch_rows,eval_batch_rows,backbone_gradient_policy):
    from gx1.audit.measure_native_future_state_throughput_v1 import require_throughput_result
    from gx1.contracts.market_state_ta_integrated_model_v1 import native_future_state_model_contract
    from gx1.contracts.market_state_ta_integrated_objective_v1 import require_native_future_state_train_prior
    keys=set(core.FUTURE_CUDA_INPUTS)|{'initial_result','initial_terminal','cohort_result','cohort_terminal',
        'throughput_result','throughput_terminal','design'}
    if mode=='whole_train_one_epoch':keys|={'selected_arm_result','selected_arm_terminal','selected_arm_replay_result','selected_arm_replay_terminal','policy_decision'}
    need(type(inputs)is dict and set(inputs)==keys and inputs['design']==DESIGN,'INPUT_ROLES_DESIGN')
    design=read(DESIGN);need(design['selection_frozen_before_new_cohort_target_or_support_inspection']is True,'FROZEN_SELECTOR_DESIGN')
    bridge=verify_initial_source_bridge(inputs)
    meta,oldrecipe=core.future_cuda_metadata({k:inputs[k]for k in core.FUTURE_CUDA_INPUTS},'mechanics',None)
    static=('source_snapshot','normalization','input_binding_ref','input_binding','prior','dataset','native_manifest','target_contract','architecture')
    need(all(meta[k]==bridge['metadata'][k]for k in static),'INITIAL_SAME_INFERENCE_INPUT_FUNCTION')
    runtime_compatible({k:bridge['metadata']['runtime'][k]for k in ('library_versions','gpu')},meta['runtime'])
    q=require_throughput_result(inputs['throughput_result'],inputs['throughput_terminal'])
    need(train_batch_rows==q['train_batch_rows']and eval_batch_rows==q['eval_batch_rows']
        and train_batch_rows in TRAIN_BATCHES and eval_batch_rows in EVAL_BATCHES,'ONE_PREQUALIFIED_BATCH_RECIPE')
    need(all(q['function_metadata'][k]==meta[k]for k in static),'THROUGHPUT_SAME_CURRENT_FUNCTION')
    runtime_compatible({k:q['function_metadata']['runtime'][k]for k in ('library_versions','gpu')},meta['runtime'])
    cohort,terminal=core.future_cuda_complete(inputs['cohort_result'],inputs['cohort_terminal'],'COMPLETE_NATIVE_FUTURE_STATE_STUDY_COHORT')
    need(inputs['cohort_result']['path']==str(ROOT/'PAIRED_COHORT_RESULT001.json'),'COHORT_RESULT_PATH')
    r=cohort['report'];snapshot=read(r['cohort_snapshot']);selection=read(r['selection'])
    need(snapshot['schema_version']=='gx1_native_future_state_study_cohort_v1'and snapshot['arrays']==r['arrays']
        and snapshot['selection']==r['selection']and snapshot['rows']==r['rows']==selection['rows']
        and snapshot['truth_sha256']==r['truth_sha256']and selection['design']==DESIGN
        and selection['whole_TRAIN_issue_ns_sha256']==TRAIN_SHA,'COHORT_SNAPSHOT_CLOCK_PROOF')
    need(r['source_snapshot']==meta['source_snapshot']and r['normalization']==meta['normalization']
        and r['input_binding']==meta['input_binding_ref']and r['factory_prior']==meta['prior'],'COHORT_CURRENT_INPUT_PRIOR')
    matched=require_native_future_state_train_prior(read(r['matched_scoring_prior']),target_contract=meta['target_contract'])
    need(matched['fit_population']['issue_rows']==r['rows']and matched['fit_population']['truth_sha256']==r['truth_sha256']
        and matched['fit_population']['issue_ns_sha256']==selection['selected_issue_ns_sha256'],'MATCHED_SCORING_PRIOR_ONLY')
    field='whole_TRAIN_batch_geometries'if mode=='whole_train_one_epoch'else'batch_geometries'
    geometry=r[field][str(train_batch_rows)];schedule=geometry['schedule']
    need(schedule['passes']==(1 if mode=='whole_train_one_epoch'else 2)
        and schedule['batch_rows']==train_batch_rows and schedule['rows']==(TRAIN_ROWS if mode=='whole_train_one_epoch'else r['rows']),
        'FROZEN_SCHEDULE_SCOPE')
    need(schedule==seal({k:v for k,v in schedule.items()if k!='contract_sha256'}),'SCHEDULE_HASH')
    if mode=='paired_learning':need(geometry['first_two_supported']is True and schedule['batches_per_pass']>=2,'FIRST_TWO_SUPPORTED_LOGICAL_BATCHES')
    recipe={k:oldrecipe[k]for k in ('optimizer','gradient_clip_norm','scheduler','seed')}
    recipe.update(train_batch_rows=train_batch_rows,eval_batch_rows=eval_batch_rows,passes=schedule['passes'],checkpoint_every_optimizer_updates=256,
        allocator_phase_policy={'function':'release_phase_allocator_cache','unused_cache_only':True,'per_step':False,
            'peak_reset':False,'boundaries':'score_to_train_train_to_score_score_to_final_diagnostics_and_changed_diagnostic_shape'})
    factory=read(meta['prior']);meta['arm_model_contracts']={}
    for arm,policy in (('joint',None),('m5_primary','M5_primary')):
        meta['arm_model_contracts'][arm]=native_future_state_model_contract(meta['input_binding'],architecture=meta['architecture'],
            train_prior=factory,target_contract=meta['target_contract'],backbone_gradient_policy=policy)
    meta['cohort_report']=r;meta['cohort_snapshot']=r['cohort_snapshot'];meta['matched_scoring_prior']=r['matched_scoring_prior']
    if mode=='whole_train_one_epoch':meta['selected_policy_evidence']=selected_policy_evidence(inputs,backbone_gradient_policy,meta,recipe)
    return meta,recipe,schedule,bridge,q


def execution(lane):
    replay=lane.endswith('_replay');outer=12000 if lane in ('main','main_resume')else 1800 if replay or lane=='recovery_probe'else 5400
    return {**core.future_cuda_execution('initial'),'outer_seconds':outer,'internal_seconds':outer-30}


def lane_root(lane):
    need(lane in LANES,'LANE')
    if lane=='main':return MAIN_ROOT
    if lane=='main_resume':return RESUME_ROOT
    if lane=='recovery_probe':return RECOVERY_ROOT
    return PAIR_ROOT/lane


def diagnostic_panels(schedule,*,main=False):
    full=schedule['rows']//schedule['batch_rows']
    indices=[]if main or not full else np.linspace(0,full-1,min(3,full),dtype=np.int64).tolist()
    empty=[i for i in indices if schedule['known_activity_by_batch'][i]==0]
    return {'selection':'first_middle_last_FULL_source_batch_integer_linspace_dedup_no_replacement',
        'candidate_batch_indices':indices,'skipped_empty_batch_indices':empty,
        'supported_panels':len(indices)-len(empty)}


def limits_for(schedule,eval_batch_rows,*,replay=False,main=False):
    t=schedule['batches_per_pass'];e=(schedule['rows']+eval_batch_rows-1)//eval_batch_rows
    updates=schedule['optimizer_updates'];logical=schedule['logical_batches'];panels=diagnostic_panels(schedule,main=main)
    extra=panels['supported_panels']
    if replay:return {'model_constructions':1,'optimizer_constructions':1,'model_forwards':1,'backbone_forwards':1,
        'backward_calls':1,'optimizer_steps':1,'skipped_batches':0,'autograd_calls':0,'diagnostic_skips':0,'checkpoint_loads':1,'checkpoint_writes':1,
        'checkpoint_readbacks':1,'checkpoint_receipt_writes':1,'normalization_fits':0,'prior_fits':0,'CONTROL_VAL_TEST_rows':0}
    writes=2+(max(0,updates-1)//256 if main else 0)
    fw=logical+(1 if main else 3)*e+extra
    return {'model_constructions':1,'optimizer_constructions':1,'model_forwards':fw,'backbone_forwards':fw,
        'backward_calls':updates,'optimizer_steps':updates,'skipped_batches':logical-updates,
        'autograd_calls':3*extra,'diagnostic_skips':len(panels['skipped_empty_batch_indices']),
        'checkpoint_loads':0,'checkpoint_writes':writes,'checkpoint_readbacks':writes,'checkpoint_receipt_writes':writes,
        'normalization_fits':0,'prior_fits':0,'CONTROL_VAL_TEST_rows':0}


def _build_plan(*,inputs,mode,backbone_gradient_policy,train_batch_rows,eval_batch_rows,source_base_commit=None):
    need(mode in ('paired_learning','whole_train_one_epoch')and backbone_gradient_policy in (None,'M5_primary'),'PLAN_MODE_POLICY')
    owners=dict.fromkeys(expected_source_names(),'0'*64);require_source_inventory(owners)
    meta,recipe,schedule,bridge,q=metadata(inputs,mode=mode,train_batch_rows=train_batch_rows,eval_batch_rows=eval_batch_rows,backbone_gradient_policy=backbone_gradient_policy)
    main=mode=='whole_train_one_epoch';path=MAIN_PLAN if main else PAIR_PLAN;root=MAIN_ROOT if main else PAIR_ROOT
    base=source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    need(base==meta['runtime']['source_commit'],'CURRENT_SOURCE_COMMIT_RUNTIME')
    score_passes=1 if main else 3
    report=q['report'];estimate=int(np.ceil(schedule['rows']*(recipe['passes']*report['train'][str(train_batch_rows)]['elapsed_seconds']/768.
        +score_passes*report['eval'][str(eval_batch_rows)]['elapsed_seconds']/768.)))
    need(estimate<execution('main'if main else'joint')['internal_seconds'],'MEASURED_COMPUTE_ESTIMATE_WITHIN_HARD_WALL')
    lanes=('main',)if main else('joint','joint_replay','m5_primary','m5_primary_replay')
    return {'schema_version':'gx1_native_future_state_training_plan_v1','status':'PREPARED_NOT_ADMITTED',
        'mode':mode,'stage':'native_future_state_one_epoch_20261121_001'if main else'native_future_state_paired_learning_20261121_001',
        'plan_path':str(path),'output_root':str(root),'source_base_commit':base,
        'source_owners':{k:sha(REPO/k)for k in owners},'inputs':inputs,'metadata':meta,
        'baseline_bridge':bridge,'training_leaf_AST':own_numeric_ast(),'recipe':recipe,'schedule':schedule,
        'cohort':{'scope':'whole354550TRAIN'if main else'source_fixed_16_per22UTCday',
            'issue_ns_sha256':schedule['issue_ns_sha256'],'source_rows_sha256':schedule['source_rows_sha256'],
            'rows':schedule['rows'],'cohort_result':inputs['cohort_result']},
        'backbone_gradient_policy':backbone_gradient_policy,'diagnostic_panels':diagnostic_panels(schedule,main=main),
        'limits':{lane:limits_for(schedule,eval_batch_rows,replay=lane.endswith('_replay'),main=main)for lane in lanes},
        'execution':{lane:execution(lane)for lane in lanes},'measured_compute_estimate_seconds':estimate,
        'wall_estimate_scope':'train_score_compute_only_optimizer_checkpoint_and_post_final_diagnostic_runtime_review_required_not_runtime_guarantee',
        'main_execution_requires_separate_exact_admission':True,'training_authorized':False,'CONTROL_VAL_TEST_used':False,
        'production_gradient_policy_selected':False,'automatic_continuation':False}


def build_paired_plan(*,inputs,train_batch_rows,eval_batch_rows,source_base_commit=None):
    return _build_plan(inputs=inputs,mode='paired_learning',backbone_gradient_policy=None,
        train_batch_rows=train_batch_rows,eval_batch_rows=eval_batch_rows,source_base_commit=source_base_commit)


def build_whole_train_one_epoch_plan(*,inputs,backbone_gradient_policy,train_batch_rows,eval_batch_rows,source_base_commit=None):
    return _build_plan(inputs=inputs,mode='whole_train_one_epoch',backbone_gradient_policy=backbone_gradient_policy,
        train_batch_rows=train_batch_rows,eval_batch_rows=eval_batch_rows,source_base_commit=source_base_commit)


def gate_expected(plan_ref,*,main=False,admitted=False,kind=None):
    kind=kind or('main_one_epoch'if main else'paired_learning')
    names={'paired_learning':('joint','joint_replay','m5_primary','m5_primary_replay'),
        'main_one_epoch':('main',),'recovery_probe':('recovery_probe',),'main_resume':('main_resume',)}
    need(kind in names,'GATE_KIND');lanes=names[kind]
    status={'paired_learning':'ADMITTED_EXACT_PAIRED_LEARNING','main_one_epoch':'ADMITTED_EXACT_WHOLE_TRAIN_ONE_EPOCH',
        'recovery_probe':'ADMITTED_EXACT_RECOVERY_PROBE','main_resume':'ADMITTED_EXACT_MAIN_RESUME'}[kind]
    return {'status':status if admitted else'PREPARED_NOT_ADMITTED','plan':plan_ref,
        'execution_authorized':admitted,'relaunch_allowed':False,
        'lanes':{k:{'execution_authorized':admitted,'remaining_invocations':1 if admitted else 0}for k in lanes}}


def inspect(plan_sha,lane,*,admitted):
    need(lane in LANES and Path.cwd().resolve()==REPO,'CWD_LANE');main=lane in ('main','main_resume')
    paths={'main':MAIN_PLAN,'main_resume':RESUME_PLAN,'recovery_probe':RECOVERY_PLAN}
    pr={'path':str(paths.get(lane,PAIR_PLAN)),'sha256':plan_sha};plan=read(pr)
    require_source_inventory(plan['source_owners'])
    need(subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='work/gx1-current','CANONICAL_BRANCH')
    if lane in ('main_resume','recovery_probe'):
        expected=build_recovery_plan(inputs=plan['inputs'],main_resume=lane=='main_resume',source_base_commit=plan['source_base_commit'])
    else:expected=_build_plan(inputs=plan['inputs'],mode=plan['mode'],backbone_gradient_policy=plan['backbone_gradient_policy'],
        train_batch_rows=plan['recipe']['train_batch_rows'],eval_batch_rows=plan['recipe']['eval_batch_rows'],source_base_commit=plan['source_base_commit'])
    modes={'main':'whole_train_one_epoch','main_resume':'main_resume','recovery_probe':'recovery_probe'}
    need(plan==expected and plan['mode']==modes.get(lane,'paired_learning'),'EXACT_PLAN')
    pref=ref(POLICY);policy=read(pref)
    kind=lane if lane in ('main_resume','recovery_probe')else'main_one_epoch'if main else'paired_learning';key=kind+'_001'
    need(json.dumps(policy[PARENT][key],sort_keys=True)==json.dumps(gate_expected(pr,main=main,admitted=admitted,kind=kind),sort_keys=True),'EXACT_ONEUSE_GATE')
    validate_training_flags(policy,main=main,admitted=admitted)
    need(not lane_root(lane).exists()and not lane_root(lane).is_symlink(),'FRESH_LANE')
    live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['metadata']['runtime'],live)
    if lane.endswith('_replay'):
        arm=lane[:-7];terminal=read(ref(lane_root(arm)/'TERMINAL.json'));claim=read(terminal['claim'])
        need(terminal['status']=='COMPLETE'and terminal['exit_code']==0 and terminal['plan']==pr
            and claim['policy_at_claim']==pref and claim['lane']==arm,'REPLAY_AUTHENTIC_ORIGINAL_CLAIM_POLICY')
        result=read(terminal['result']);core.future_cpu_parent_stopped(result['process_identity'])
    return plan,pr,pref,live



def load_context(plan):
    from gx1.audit.build_native_m5_ta_feature_cache_v1 import load_snapshot
    from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
    meta=plan['metadata'];provider=load_snapshot(meta['source_snapshot']);norm=read(meta['normalization'])
    factory=objective.require_native_future_state_train_prior(read(meta['prior']),target_contract=meta['target_contract'])
    whole=np.flatnonzero((provider.issue_ns>=START)&(provider.issue_ns<UPPER)).astype(np.int64)
    need(len(whole)==TRAIN_ROWS and array_sha(provider.issue_ns[whole])==TRAIN_SHA and provider.input_ready[whole].all()
        and provider.input_binding(norm)==meta['input_binding']
        and array_sha(provider.source_row[whole])==norm['fit_population']['selected_source_rows_sha256'],
        'CURRENT_PHYSICAL_WHOLE_TRAIN_INPUTS')
    main=plan['mode']=='whole_train_one_epoch'
    if main:
        data=read(meta['dataset']);binding=data['arrays'];need(ref(binding['path'])==binding,'FULL_TARGET_ARCHIVE')
        truth={}
        with np.load(binding['path'],allow_pickle=False)as archive:
            for k in TRUTH:
                a=archive[k];h=data['array_headers'][k]
                need(a.dtype.str==h['dtype']and list(a.shape)==h['shape']and array_sha(a)==h['sha256'],'FULL_TRUTH_BYTES')
                truth[k]=a
        positions=whole;source=provider.source_row[whole];scoring=factory
        need(objective._native_prior_array_sha(truth)==factory['fit_population']['truth_sha256'],'FULL_TRUTH_FACTORY_PRIOR')
    else:
        report=meta['cohort_report'];data=read(report['cohort_snapshot']);binding=data['arrays']
        need(ref(binding['path'])==binding,'COHORT_ARCHIVE')
        expected={'selected_positions','source_rows','issue_ns'}|{'truth__'+k for k in TRUTH};arrays={}
        with np.load(binding['path'],allow_pickle=False)as archive:
            need(set(archive.files)==set(data['array_headers'])==expected,'COHORT15_INVENTORY')
            for k in sorted(expected):
                a=archive[k];h=data['array_headers'][k]
                need(a.dtype.str==h['dtype']and list(a.shape)==h['shape']and array_sha(a)==h['sha256'],'COHORT_ARRAY_BYTES:'+k)
                arrays[k]=a
        positions=arrays['selected_positions'];source=arrays['source_rows'];truth={k:arrays['truth__'+k]for k in TRUTH}
        need(np.array_equal(arrays['issue_ns'],truth['issue_ns'])and np.array_equal(provider.source_row[positions],source)
            and np.array_equal(provider.issue_ns[positions],truth['issue_ns'])and provider.input_ready[positions].all(),
            'COHORT_PHYSICAL_SOURCE_CLOCK')
        scoring=objective.require_native_future_state_train_prior(read(meta['matched_scoring_prior']),target_contract=meta['target_contract'])
        need(objective._native_prior_array_sha(truth)==report['truth_sha256']==scoring['fit_population']['truth_sha256'],
            'COHORT_SCORING_PRIOR_TRUTH')
    need(array_sha(source)==plan['schedule']['source_rows_sha256']and array_sha(truth['issue_ns'])==plan['schedule']['issue_ns_sha256']
        and len(source)==plan['schedule']['rows'],'FROZEN_PHYSICAL_SCHEDULE')
    masks=objective._native_future_state_truth(truth,rows=len(source),fold_upper_ns=UPPER,target_contract=meta['target_contract'])
    batches=core.all_score_batches(np.arange(len(source),dtype=np.int64),plan['recipe']['train_batch_rows'])
    known=[int(masks['activity'][b].sum())for b in batches]
    need(make_schedule(source,truth['issue_ns'],batch_rows=plan['recipe']['train_batch_rows'],
        passes=plan['recipe']['passes'],known_activity_by_batch=known)==plan['schedule'],'PHYSICAL_SCHEDULE_SUPPORT')
    return {'provider':provider,'normalization':norm,'prior':factory,'scoring_prior':scoring,
        'positions':positions,'source_rows':source,'truth':truth,'batches':batches}


def packet(context,take):
    import torch
    raw=context['provider'].gather(context['source_rows'][take],max_rows=len(take))
    return {'features':{k:torch.from_numpy(v)for k,v in raw.items()},'truth':{k:v[take]for k,v in context['truth'].items()}}


def score_point(model,context,plan,*,initial):
    batches=core.all_score_batches(np.arange(len(context['source_rows']),dtype=np.int64),plan['recipe']['eval_batch_rows'])
    return score_with_capture(model,(packet(context,b)for b in batches),plan,context['prior'],context['scoring_prior'],
        context['truth'],initial=initial)


def initial_identity(model,plan):
    import torch
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    # Comparison only: never alter a load-state dict to cross policy identities.
    state=dict(model.state_dict());baseline=plan['baseline_bridge']['metadata']['model_contract']['contract_sha256']
    state['future_state_model_sha256']=torch.tensor(list(bytes.fromhex(baseline)),dtype=torch.uint8)
    need(canonical_model_state_sha256(state)==plan['baseline_bridge']['initial_state_sha256'],
        'FRESH_FACTORY_SAME_INITIAL_PARAMETERS_EXCEPT_EXPLICIT_POLICY_IDENTITY')
    return {'full_state_sha256':canonical_model_state_sha256(model.state_dict()),
        'policy_normalized_comparison_sha256':canonical_model_state_sha256(state),
        'rng_sha256':core.state_sha(core.training_rng())}


def run_training_loop(model,optimizer,context,plan,plan_ref,root,lane,live,counts,*,resume=None,execution_ref=None):
    """One fixed schedule for paired two-pass and preparation-only main recipes."""
    main=plan['mode']=='whole_train_one_epoch';arm=('joint'if plan['backbone_gradient_policy']is None else'm5_primary')if main else lane.removesuffix('_replay')
    trajectory=trajectory_bindings(plan,arm);fresh=initial_identity(model,plan);replay=lane.endswith('_replay')
    steps=[];scores={};cp1=None;cursor2=None;start_cursor=0;allocator_boundaries=[]
    if resume is not None:
        need(main and not replay and resume['original_plan']==plan_ref,'EXPLICIT_MAIN_RECOVERY_ORIGINAL_TRAJECTORY')
        start_cursor=resume['expected_cursor']
        need(0<=start_cursor<plan['schedule']['logical_batches'],'RECOVERY_REMAINING_EPOCH_SCOPE')
        strict_restore(model,optimizer,resume['checkpoint'],trajectory=trajectory,original_plan_ref=plan_ref,
            expected_cursor=start_cursor,current_runtime=live,expected_state_proof=resume['expected_state_proof'])
        steps=list(resume['checkpoint']['steps']);scores=dict(resume['checkpoint']['scores'])
    if replay:
        terminal=read(ref(lane_root(arm)/'TERMINAL.json'));reference=read(terminal['result'])
        need(reference['plan']==plan_ref and reference['report']['fresh_initial']==fresh,'REPLAY_FRESH_FACTORY_STATE_RNG')
        receipt=read(reference['report']['cursor1_receipt']);binding=receipt['checkpoint']
        need(binding['path']==str(lane_root(arm)/'CURSOR001.pt')and ref(binding['path'])==binding
            and receipt['trajectory_sha256']==trajectory['contract_sha256'],'AUTHENTIC_SAME_ARM_CURSOR1')
        import torch
        cp=torch.load(binding['path'],map_location='cpu',weights_only=True);counts['checkpoint_loads']+=1
        need(checkpoint_proof(cp)==receipt['proof'],'CURSOR1_DISK_PROOF')
        strict_restore(model,optimizer,cp,trajectory=trajectory,original_plan_ref=plan_ref,expected_cursor=1,current_runtime=live,expected_state_proof=receipt['proof'])
        steps=list(cp['steps']);scores=dict(cp['scores'])
        step=training_step(model,optimizer,packet(context,context['batches'][1]),plan,counts,1);steps.append(step)
        current=make_checkpoint(model,optimizer,trajectory=trajectory,original_plan_ref=plan_ref,lane=lane,
            cursor=2,steps=steps,scores=scores,producer_runtime=live)
        expected=read(reference['report']['cursor2_proof'])
        need(checkpoint_proof(current)==expected['proof'],'OWN_POLICY_COLD_STEP2_BITEXACT')
        receipt=publish_checkpoint_receipt(root/'CURSOR002.pt',current,counts,execution_ref=execution_ref or plan_ref)
        binding=read(receipt)['checkpoint']
        return {'lane':lane,'fresh_initial':fresh,'cold_replay_bit_exact':True,'counts':counts,'cursor':2,
            'checkpoint':binding,'proof':checkpoint_proof(current),'trajectory':trajectory}
    if main and resume is None:
        # Qualified unchanged full inference baseline is reused, never refitted.
        scores['INITIAL']={'reused_qualified_result':INITIAL_RESULT,'score':plan['baseline_bridge']['initial_score']}
    elif not main:
        scores['INITIAL']=score_point(model,context,plan,initial=True)
        allocator_boundaries.append(release_phase_allocator_cache(model,optimizer,label='INITIAL_SCORE_TO_TRAIN'))
    if main and resume is None:
        cp=make_checkpoint(model,optimizer,trajectory=trajectory,original_plan_ref=plan_ref,lane=lane,cursor=0,
            steps=steps,scores=scores,producer_runtime=live)
        publish_checkpoint_receipt(root/'INITIAL.pt',cp,counts,execution_ref=execution_ref or plan_ref)
    n=plan['schedule']['logical_batches'];offset=position(plan['schedule'],start_cursor);last_updates=offset['optimizer_updates']
    for index in range(start_cursor,n):
        take=context['batches'][index%len(context['batches'])]
        steps.append(training_step(model,optimizer,packet(context,take),plan,counts,index));cursor=index+1
        need(counts['optimizer_steps']+offset['optimizer_updates']==position(plan['schedule'],cursor)['optimizer_updates']
            and counts['skipped_batches']+offset['skipped_batches']==position(plan['schedule'],cursor)['skipped_batches'],'LOGICAL_VS_ACTUAL_UPDATE_COUNTS')
        if not main and cursor in (1,2):
            cp=make_checkpoint(model,optimizer,trajectory=trajectory,original_plan_ref=plan_ref,lane=lane,cursor=cursor,
                steps=list(steps),scores=copy.deepcopy(scores),producer_runtime=live)
            if cursor==1:
                cp1=publish_checkpoint_receipt(root/'CURSOR001.pt',cp,counts,execution_ref=execution_ref or plan_ref)
            else:cursor2=publish(root/'CURSOR002_PROOF.json',{'proof':checkpoint_proof(cp)})
        if cursor%len(context['batches'])==0:
            label='PASS'+str(cursor//len(context['batches']))
            allocator_boundaries.append(release_phase_allocator_cache(model,optimizer,label='TRAIN_TO_'+label+'_SCORE'))
            scores[label]=score_point(model,context,plan,initial=False)
            allocator_boundaries.append(release_phase_allocator_cache(model,optimizer,
                label=label+('_SCORE_TO_TRAIN'if cursor<n else'_SCORE_TO_FINAL_DIAGNOSTICS_OR_CHECKPOINT')))
        updates=counts['optimizer_steps']+offset['optimizer_updates']
        if main and updates and updates%256==0 and updates!=last_updates and updates<plan['schedule']['optimizer_updates']:
            cp=make_checkpoint(model,optimizer,trajectory=trajectory,original_plan_ref=plan_ref,lane=lane,cursor=cursor,
                steps=list(steps),scores=copy.deepcopy(scores),producer_runtime=live)
            publish_checkpoint_receipt(root/('UPDATE'+str(updates).zfill(6)+'.pt'),cp,counts,execution_ref=execution_ref or plan_ref)
        last_updates=updates
    diagnostics=[]
    before_optimizer=core.state_sha(optimizer.state_dict());previous_panel_shape=None
    for index in plan['diagnostic_panels']['candidate_batch_indices']:
        if index in plan['diagnostic_panels']['skipped_empty_batch_indices']:
            counts['diagnostic_skips']+=1;diagnostics.append({'batch_index':index,'skipped_empty_support':True});continue
        panel_packet=packet(context,context['batches'][index])
        panel_shape=tuple((k,tuple(v.shape))for k,v in sorted(panel_packet['features'].items()))
        if previous_panel_shape is not None and panel_shape!=previous_panel_shape:
            allocator_boundaries.append(release_phase_allocator_cache(model,optimizer,label='FINAL_DIAGNOSTIC_SHAPE_CHANGE_'+str(index)))
        balance=core.future_cuda_gradient_probe(model,panel_packet,plan,counts);previous_panel_shape=panel_shape
        balance['role']='post_final_two_pass_data_gradients_no_optimal_weight_or_universal_side_coverage_claim'
        diagnostics.append({'batch_index':index,'source_rows_sha256':plan['schedule']['batch_source_rows_sha256'][index],'report':balance})
    need(core.state_sha(optimizer.state_dict())==before_optimizer,'DIAGNOSTICS_OPTIMIZER_UNCHANGED')
    cp=make_checkpoint(model,optimizer,trajectory=trajectory,original_plan_ref=plan_ref,lane=lane,cursor=n,
        steps=steps,scores=scores,producer_runtime=live)
    final_receipt=publish_checkpoint_receipt(root/'FINAL.pt',cp,counts,execution_ref=execution_ref or plan_ref)
    binding=read(final_receipt)['checkpoint']
    return {'lane':lane,'fresh_initial':fresh,'trajectory':trajectory,'scores':scores,'steps':steps,'counts':counts,
        'cursor1_receipt':cp1,'cursor2_proof':cursor2,'final_checkpoint':binding,'final_checkpoint_receipt':final_receipt,'final_proof':checkpoint_proof(cp),
        'logical_batches':n,'execution_start_cursor':start_cursor,
        'actual_optimizer_updates':counts['optimizer_steps'],'skipped_batches':counts['skipped_batches'],
        'global_position':position(plan['schedule'],n),
        'passes':plan['recipe']['passes'],'candidate_replay_qualified':False,
        'post_final_gradient_balance':diagnostics,'allocator_phase_boundaries':allocator_boundaries,
        'scope':'whole_TRAIN_one_epoch'if main else'fixed_source_selected_TRAIN_two_passes_no_generalization_or_policy_winner_claim'}


def physical(plan,plan_ref,root,lane,live):
    import torch
    from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
    context=load_context(plan);core.future_cuda_seed_and_runtime(plan);m=plan['metadata']
    policy=plan['backbone_gradient_policy']if lane=='main'else LANES[lane]
    model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=m['input_binding'],
        architecture=m['architecture'],device=torch.device('cuda'),train_prior=context['prior'],target_contract=m['target_contract'],
        backbone_gradient_policy=policy)
    options={**plan['recipe']['optimizer'],'betas':tuple(plan['recipe']['optimizer']['betas'])}
    optimizer=torch.optim.AdamW(model.parameters(),**options);need(not optimizer.state,'FRESH_ADAMW')
    counts=dict.fromkeys(plan['limits'][lane],0);counts['model_constructions']=counts['optimizer_constructions']=1
    def count(key):
        def hook(*args):
            counts[key]+=1;need(counts[key]<=plan['limits'][lane][key],'EXACT_FORWARD_BUDGET')
        return hook
    hooks=[model.register_forward_pre_hook(count('model_forwards')),model.backbone.register_forward_pre_hook(count('backbone_forwards'))]
    try:
        report=run_training_loop(model,optimizer,context,plan,plan_ref,root,lane,live,counts)
        need(counts==plan['limits'][lane],'EXACT_PHYSICAL_COUNTS')
        need(torch.cuda.max_memory_reserved()<=core.LARGER_VRAM_MAX,'CUDA_VRAM_CAP')
        return report
    finally:
        for hook in hooks:hook.remove()


def run(plan_sha,lane):
    from gx1.contracts.gx1_capped_execution_v1 import require_native_future_state_training_execution
    began=time.monotonic()
    # Exact mode-specific policy admission precedes the execution guard/CLAIM.
    plan,pr,pref,live=inspect(plan_sha,lane,admitted=True)
    guard=require_native_future_state_training_execution(plan_sha256=plan_sha,lane=lane)
    root=lane_root(lane);expected=execution(lane)
    need(guard['plan']==pr and guard['output_root']==str(root)and guard['outer_seconds']==expected['outer_seconds']
        and guard['internal_seconds']==expected['internal_seconds'],'EXACT_EXECUTION_GUARD')
    root.mkdir(parents=True,exist_ok=False);process=core.future_cpu_process_identity()
    claim=publish(root/'CLAIM.json',{'stage':plan['stage'],'plan':pr,'lane':lane,'policy_at_claim':pref,
        'runtime_at_claim':live,'process_identity':process,'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False})
    old=signal.getsignal(signal.SIGALRM);result=None
    def expired(*args):raise TimeoutError('NATIVE_FUTURE_STATE_TRAINING_INTERNAL_WALL')
    try:
        remaining=expected['internal_seconds']-(time.monotonic()-began);need(remaining>0,'WALL_BEFORE_INPUTS')
        signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
        report=recovery_physical(plan,pr,root,lane,live)if lane in ('recovery_probe','main_resume')else physical(plan,pr,root,lane,live)
        need(all(sha(REPO/k)==v for k,v in plan['source_owners'].items())and ref(POLICY)==pref and ref(pr['path'])==pr,
            'SOURCE_POLICY_PLAN_BOOKEND')
        for binding in plan['inputs'].values():read(binding)
        function_meta=plan['metadata']['original']['metadata']if lane in ('recovery_probe','main_resume')else plan['metadata']
        for k in ('source_snapshot','normalization','input_binding_ref','prior','dataset','native_manifest'):read(function_meta[k])
        snapshot=read(function_meta['source_snapshot']);data=read(function_meta['dataset'])
        need(ref(snapshot['archive']['path'])==snapshot['archive']and ref(data['arrays']['path'])==data['arrays'],
            'FROZEN_CACHE_TARGET_BYTES_BOOKEND')
        cohort=function_meta['cohort_report'];need(ref(cohort['arrays']['path'])==cohort['arrays'],'FROZEN_COHORT_BYTES_BOOKEND')
        need(subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==live['source_commit']
            and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==live['wsl_boot_id']
            and core.physical_windows_boot_id()==live['physical_windows_boot_id'],'EXECUTION_BOOT_SOURCE_BOOKEND')
        need(core.common.firewall_state()['denied_attempts']==0,'CLEAN_FIREWALL')
        result=publish(root/'RESULT.json',{'stage':plan['stage'],'status':{'main':'COMPLETE_NATIVE_FUTURE_STATE_ONE_EPOCH','main_resume':'COMPLETE_NATIVE_FUTURE_STATE_MAIN_RESUME','recovery_probe':'COMPLETE_NATIVE_FUTURE_STATE_RECOVERY_PROBE'}.get(lane,'COMPLETE_NATIVE_FUTURE_STATE_PAIRED_LEARNING'),
            'plan':pr,'claim':claim,'process_identity':process,'report':report,'guard':guard,
            'source_unchanged':True,'test_data_used':False,'elapsed_seconds':time.monotonic()-began})
        terminal=publish(root/'TERMINAL.json',{'stage':plan['stage'],'status':'COMPLETE','exit_code':0,
            'plan':pr,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,
            'source_unchanged':True,'test_data_used':False})
        return {'result':result,'terminal':terminal}
    except BaseException as error:
        publish(root/'TERMINAL.json',{'stage':plan['stage'],'status':'FAILED_CONSUMED','exit_code':1,'plan':pr,
            'claim':claim,'result':result,'error':type(error).__name__+':'+str(error)[:2000],
            'remaining_invocations':0,'relaunch_allowed':False,'source_unchanged':False,'test_data_used':False});raise
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)


def publish_checkpoint_receipt(path,checkpoint,counts,*,execution_ref):
    binding=core.future_cuda_publish_checkpoint(path,checkpoint,counts)
    receipt={'schema_version':'gx1_native_future_state_training_checkpoint_receipt_v1','checkpoint':binding,
        'original_plan':checkpoint['producer']['original_plan'],'execution_plan':execution_ref,
        'producer_lane':checkpoint['producer']['lane'],'producer_runtime':checkpoint['producer']['runtime'],
        'logical_cursor':checkpoint['position']['logical_cursor'],
        'optimizer_updates':checkpoint['position']['optimizer_updates'],
        'trajectory_sha256':checkpoint['trajectory']['contract_sha256'],'proof':checkpoint_proof(checkpoint)}
    value=publish(Path(path).with_suffix('.json'),receipt);counts['checkpoint_receipt_writes']+=1
    need(read(value)==receipt,'CHECKPOINT_RECEIPT_DISK_READBACK');return value


def require_recovery_ref(value,*,expected_path=None,suffix='.json'):
    need(type(value)is dict and set(value)=={'path','sha256'}
        and type(value['path'])is str and type(value['sha256'])is str
        and len(value['sha256'])==64 and all(c in '0123456789abcdef'for c in value['sha256']),
        'RECOVERY_REFERENCE_SCHEMA')
    path=Path(value['path'])
    need(path.is_absolute()and str(path)==value['path']and '..'not in path.parts and path.suffix==suffix
        and(expected_path is None or path==expected_path),'RECOVERY_REFERENCE_PATH')
    return path


def require_recovery_input_refs(inputs,*,main_resume):
    keys={'original_plan','checkpoint_receipt','runtime_proof'}|(set()if main_resume else{'original_cursor2_proof'})
    need(type(inputs)is dict and set(inputs)==keys,'RECOVERY_INPUT_ROLES')
    require_recovery_ref(inputs['original_plan'],expected_path=MAIN_PLAN if main_resume else PAIR_PLAN)
    name=require_recovery_ref(inputs['checkpoint_receipt'])
    if main_resume:
        need(name.parent==MAIN_ROOT and(name.name in ('INITIAL.json','FINAL.json')or
            (len(name.stem)==12 and name.stem.startswith('UPDATE')and name.stem[6:].isascii()and name.stem[6:].isdigit())),
            'FIXED_RECOVERY_RECEIPT_PATH')
    else:
        need(name in tuple(PAIR_ROOT/a/'CURSOR001.json'for a in ('joint','m5_primary')),'FIXED_RECOVERY_RECEIPT_PATH')
        require_recovery_ref(inputs['original_cursor2_proof'],expected_path=name.parent/'CURSOR002_PROOF.json')
    runtime=require_recovery_ref(inputs['runtime_proof'])
    need(runtime.parent==ROOT and len(runtime.stem)==16 and runtime.stem.startswith('RUNTIME_PROOF')
        and runtime.stem[13:].isascii()and runtime.stem[13:].isdigit(),'FIXED_RECOVERY_RUNTIME_PATH')
    need(not runtime.is_symlink(),'RECOVERY_RUNTIME_SYMLINK')


def require_recovery_checkpoint_proof(proof):
    keys={'trajectory','position','model_state','optimizer_state','scheduler_state','rng_state',
        'training','optimizer_parameter_names','steps','scores'}
    need(type(proof)is dict and set(proof)==keys and all(type(v)is str and len(v)==64
        and all(c in '0123456789abcdef'for c in v)for v in proof.values()),'RECOVERY_CHECKPOINT_PROOF_SCHEMA')
    return proof


def require_physical_recovery_boot(original,producer,current):
    for value in (original,producer,current):core.future_cuda_require_runtime(value)
    need(current['physical_windows_boot_id']>max(original['physical_windows_boot_id'],producer['physical_windows_boot_id'])
        and all(current['wsl_boot_id']!=v['wsl_boot_id']for v in (original,producer)),
        'RECOVERY_PROBE_REQUIRES_PHYSICAL_NEW_BOOT')
    return True


def require_recovery_receipt(receipt,*,receipt_ref,original_plan_ref,original,trajectory,arm,main_resume):
    keys={'schema_version','checkpoint','original_plan','execution_plan','producer_lane','producer_runtime',
        'logical_cursor','optimizer_updates','trajectory_sha256','proof'}
    need(type(receipt)is dict and set(receipt)==keys,'RECOVERY_RECEIPT_SCHEMA')
    name=require_recovery_ref(receipt_ref)
    require_recovery_ref(receipt['checkpoint'],expected_path=name.with_suffix('.pt'),suffix='.pt')
    for key in ('original_plan','execution_plan'):
        require_recovery_ref(receipt[key],expected_path=MAIN_PLAN if main_resume else PAIR_PLAN)
        need(receipt[key]==original_plan_ref,'RECOVERY_RECEIPT_ORIGINAL_EXECUTION_PLAN')
    proof=require_recovery_checkpoint_proof(receipt['proof']);cursor=receipt['logical_cursor']
    need(type(cursor)is int and 0<=cursor<original['schedule']['logical_batches']and(main_resume or cursor==1),
        'RECOVERY_EXACT_CURSOR_REMAINING_EPOCH')
    need(receipt['schema_version']=='gx1_native_future_state_training_checkpoint_receipt_v1'
        and receipt['producer_lane']==('main'if main_resume else arm)
        and type(receipt['optimizer_updates'])is int
        and receipt['optimizer_updates']==position(original['schedule'],cursor)['optimizer_updates']
        and receipt['trajectory_sha256']==trajectory['contract_sha256']
        and proof['trajectory']==core.state_sha(trajectory)
        and proof['position']==core.state_sha(position(original['schedule'],cursor)),
        'AUTHENTIC_ORIGINAL_CHECKPOINT_RECEIPT')
    if main_resume:
        need((name.name=='INITIAL.json'and cursor==0)or
            (name.stem.startswith('UPDATE')and int(name.stem[6:])==receipt['optimizer_updates']
                and receipt['optimizer_updates']>0 and receipt['optimizer_updates']%256==0),
            'RECOVERY_MAIN_CHECKPOINT_CADENCE')
    producer=receipt['producer_runtime'];core.future_cuda_require_runtime(original['metadata']['runtime']);core.future_cuda_require_runtime(producer)
    runtime_compatible(trajectory['runtime_compatibility'],producer)
    need(all(producer[k]==original['metadata']['runtime'][k]for k in ('physical_windows_boot_id','wsl_boot_id')),
        'RECOVERY_RECEIPT_ORIGINAL_PRODUCER_BOOT')
    return cursor


def build_recovery_plan(*,inputs,main_resume=False,source_base_commit=None):
    require_recovery_input_refs(inputs,main_resume=main_resume)
    original=read(inputs['original_plan']);receipt=read(inputs['checkpoint_receipt']);live=read(inputs['runtime_proof'])
    core.future_cuda_require_runtime(live)
    need(inputs['original_plan']['path']==str(MAIN_PLAN if main_resume else PAIR_PLAN)
        and original['mode']==('whole_train_one_epoch'if main_resume else'paired_learning'),'FIXED_ORIGINAL_TRAJECTORY_PLAN')
    arm=('joint'if original['backbone_gradient_policy']is None else'm5_primary')if main_resume else receipt['producer_lane']
    need(arm in ('joint','m5_primary'),'RECOVERY_ORIGINAL_ARM')
    trajectory=trajectory_bindings(original,arm)
    cursor=require_recovery_receipt(receipt,receipt_ref=inputs['checkpoint_receipt'],original_plan_ref=inputs['original_plan'],
        original=original,trajectory=trajectory,arm=arm,main_resume=main_resume)
    if not main_resume:require_physical_recovery_boot(original['metadata']['runtime'],receipt['producer_runtime'],live)
    need(original['training_leaf_AST']==own_numeric_ast(),'RECOVERY_NUMERIC_LEAF_UNCHANGED')
    require_source_inventory(original['source_owners'])
    need(all(sha(REPO/k)==v for k,v in original['source_owners'].items()if k not in GUARD_PATHS),'RECOVERY_NUMERIC_SOURCES_UNCHANGED')
    verify_initial_source_bridge(original['inputs']);runtime_compatible(trajectory['runtime_compatibility'],live)
    if not main_resume:
        need(inputs['checkpoint_receipt']['path']==str(PAIR_ROOT/arm/'CURSOR001.json')
            and inputs['original_cursor2_proof']['path']==str(PAIR_ROOT/arm/'CURSOR002_PROOF.json'),
            'FIXED_RECOVERY_PROBE_CURSOR1_2')
        expected=read(inputs['original_cursor2_proof'])
        need(type(expected)is dict and set(expected)=={'proof'},'ORIGINAL_CURSOR2_PROOF_SCHEMA')
        require_recovery_checkpoint_proof(expected['proof'])
        need(expected['proof']['trajectory']==core.state_sha(trajectory)
            and expected['proof']['position']==core.state_sha(position(original['schedule'],2)),'ORIGINAL_CURSOR2_TRAJECTORY')
    lane='main_resume'if main_resume else'recovery_probe';schedule=original['schedule']
    limit=limits_for(schedule,original['recipe']['eval_batch_rows'],replay=not main_resume,main=main_resume)
    if main_resume:
        before=position(schedule,cursor);remaining=schedule['logical_batches']-cursor;updates=schedule['optimizer_updates']-before['optimizer_updates']
        periodic=max(0,(max(0,schedule['optimizer_updates']-1)//256)-(before['optimizer_updates']//256))
        e=(schedule['rows']+original['recipe']['eval_batch_rows']-1)//original['recipe']['eval_batch_rows']
        limit.update(model_forwards=remaining+e,backbone_forwards=remaining+e,backward_calls=updates,optimizer_steps=updates,
            skipped_batches=remaining-updates,checkpoint_loads=1,checkpoint_writes=periodic+1,
            checkpoint_readbacks=periodic+1,checkpoint_receipt_writes=periodic+1)
    owners=dict.fromkeys(expected_source_names(),'0'*64);require_source_inventory(owners)
    base=source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    need(base==live['source_commit'],'RECOVERY_CURRENT_EXECUTION_SOURCE')
    return {'schema_version':'gx1_native_future_state_recovery_execution_plan_v1','status':'PREPARED_NOT_ADMITTED',
        'mode':lane,'stage':'native_future_state_'+lane+'_20261121_001','source_base_commit':base,
        'source_owners':{k:sha(REPO/k)for k in owners},'inputs':inputs,'output_root':str(lane_root(lane)),
        'metadata':{'original':original,'original_plan':inputs['original_plan'],'receipt':receipt,
            'trajectory':trajectory,'arm':arm,'runtime':live},'limits':{lane:limit},
        'execution':{lane:execution(lane)},'original_epoch_not_extended':True,'training_authorized':False,
        'CONTROL_VAL_TEST_used':False,'relaunch_allowed':False}


def build_recovery_probe_plan(*,inputs,source_base_commit=None):
    return build_recovery_plan(inputs=inputs,main_resume=False,source_base_commit=source_base_commit)


def build_main_resume_plan(*,inputs,source_base_commit=None):
    return build_recovery_plan(inputs=inputs,main_resume=True,source_base_commit=source_base_commit)


def recovery_physical(plan,execution_ref,root,lane,live):
    import torch
    from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
    meta=plan['metadata'];original=meta['original'];arm=meta['arm'];trajectory=meta['trajectory']
    context=load_context(original);core.future_cuda_seed_and_runtime(original);m=original['metadata']
    policy=None if arm=='joint'else'M5_primary'
    model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=m['input_binding'],
        architecture=m['architecture'],device=torch.device('cuda'),train_prior=context['prior'],target_contract=m['target_contract'],backbone_gradient_policy=policy)
    options={**original['recipe']['optimizer'],'betas':tuple(original['recipe']['optimizer']['betas'])}
    optimizer=torch.optim.AdamW(model.parameters(),**options);need(not optimizer.state,'RECOVERY_FRESH_ADAMW')
    fresh=initial_identity(model,original);counts=dict.fromkeys(plan['limits'][lane],0)
    counts['model_constructions']=counts['optimizer_constructions']=1;receipt=meta['receipt'];binding=receipt['checkpoint']
    need(ref(binding['path'])==binding,'ORIGINAL_CHECKPOINT_FILE_HASH')
    checkpoint=torch.load(binding['path'],map_location='cpu',weights_only=True);counts['checkpoint_loads']+=1
    need(checkpoint['producer']=={'original_plan':meta['original_plan'],'lane':receipt['producer_lane'],
        'runtime':receipt['producer_runtime']},'AUTHENTIC_CHECKPOINT_PRODUCER_READBACK')
    if lane=='recovery_probe':require_physical_recovery_boot(original['metadata']['runtime'],checkpoint['producer']['runtime'],live)
    def count(key):
        def hook(*args):counts[key]+=1;need(counts[key]<=plan['limits'][lane][key],'RECOVERY_EXACT_FORWARD_BUDGET')
        return hook
    hooks=[model.register_forward_pre_hook(count('model_forwards')),model.backbone.register_forward_pre_hook(count('backbone_forwards'))]
    try:
        if lane=='main_resume':
            report=run_training_loop(model,optimizer,context,original,meta['original_plan'],root,'main',live,counts,
                resume={'original_plan':meta['original_plan'],'checkpoint':checkpoint,'expected_cursor':receipt['logical_cursor'],
                    'expected_state_proof':receipt['proof']},execution_ref=execution_ref)
        else:
            strict_restore(model,optimizer,checkpoint,trajectory=trajectory,original_plan_ref=meta['original_plan'],
                expected_cursor=1,current_runtime=live,expected_state_proof=receipt['proof'])
            steps=list(checkpoint['steps']);steps.append(training_step(model,optimizer,packet(context,context['batches'][1]),original,counts,1))
            cp=make_checkpoint(model,optimizer,trajectory=trajectory,original_plan_ref=meta['original_plan'],lane=lane,cursor=2,
                steps=steps,scores=checkpoint['scores'],producer_runtime=live)
            expected=read(plan['inputs']['original_cursor2_proof'])
            need(checkpoint_proof(cp)==expected['proof'],'PHYSICAL_RECOVERY_FULL_CURSOR2_BITEXACT')
            value=publish_checkpoint_receipt(root/'CURSOR002.pt',cp,counts,execution_ref=execution_ref)
            report={'lane':lane,'fresh_initial':fresh,'checkpoint_receipt':value,'proof':checkpoint_proof(cp),
                'cold_recovery_bit_exact':True,'physical_boot_changed':True,'verified_physical_recovery':True}
        need(counts==plan['limits'][lane]and torch.cuda.max_memory_reserved()<=core.LARGER_VRAM_MAX,'RECOVERY_COUNTS_RESOURCES')
        return {**report,'counts':counts,'original_plan':meta['original_plan'],'execution_plan':execution_ref}
    finally:
        for hook in hooks:hook.remove()


def main():
    import sys
    parser=argparse.ArgumentParser();parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--lane',choices=tuple(LANES),required=True);parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args();need(sys.argv[1:]==['--plan-sha256',args.plan_sha256,'--lane',args.lane]+(['--preflight']if args.preflight else []),'EXACT_CLI')
    if args.preflight:
        _,pr,pref,live=inspect(args.plan_sha256,args.lane,admitted=False)
        print(json.dumps({'status':'PREFLIGHT_NATIVE_FUTURE_STATE_TRAINING_METADATA_ONLY','plan':pr,'policy':pref,
            'lane':args.lane,'cuda_initialized':False,'models':0,'optimizer_steps':0,'claim_created':False}))
    else:print(json.dumps(run(args.plan_sha256,args.lane),sort_keys=True))


if __name__=='__main__':
    from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
    install_native_m5_direction_io_firewall();main()
