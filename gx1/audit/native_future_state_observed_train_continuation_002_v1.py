"""One locked observed TRAIN continuation FINAL648 -> FINAL864; never a fit series."""
from pathlib import Path
import argparse,copy,hashlib,json,signal,subprocess,sys,time
REPO=Path(__file__).resolve().parents[2]
if str(REPO)not in sys.path:sys.path.insert(0,str(REPO))
from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
_FIREWALL=install_native_m5_direction_io_firewall()
from gx1.audit import native_future_state_observed_train_continuation_v1 as prior
q=prior.q
from gx1.audit import native_future_state_step_observer_cold_recovery_v1 as recovery
from gx1.contracts import gx1_capped_execution_v1 as guard
o=q.o;core=q.core;train=q.train;need=q.need;read=q.read;ref=q.ref;E=o.E
NAME='NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_20261128_002';STAGE=NAME.lower();PLAN=REPO/'configs/research'/(NAME+'.json');ROOT=Path('/home/andre2/GX1_RUNS')/NAME
SNAPSHOT=E/'V3_OBSERVED_TRAIN_CONTINUATION_ADMITTED_POLICY_20261128_002.json'
DESIGN={'path':str(E/'V3_OBSERVED_TRAIN_CONTINUATION_DESIGN_20261128_002.json'),'sha256':'f52d056a1715f13180b6aee375ae17fd45eb992644bf447f38a0239ebf34bf30'}
ANCHOR='ba00b1e121c3601494092c2f866bd6b912410e6c';SELF='gx1/audit/native_future_state_observed_train_continuation_002_v1.py';TEST='tests/test_native_future_state_observed_train_continuation_002_v1.py';GUARD_TEST='tests/test_native_observed_train_continuation_002_guard_v1.py'
PROFILE='gx1_native_observed_train_continuation_002_trainer20g_v1';STATUS_PREFLIGHT='PREFLIGHT_V3_OBSERVED_TRAIN_CONTINUATION_002_METADATA_ONLY'


def gate(binding,admitted=False):
    return {'status':'ADMITTED_EXACT_V3_OBSERVED_TRAIN_CONTINUATION_002'if admitted else'PREPARED_CLOSED_NOT_ADMITTED','plan':binding,'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':admitted,'test_authorized':False,'spending_authorized':False}


def require_gate(value,binding,admitted=False):
    expected=gate(binding,admitted);need(type(value)is dict and type(value.get('remaining_invocations'))is int and all(type(value.get(k))is bool for k in('execution_authorized','relaunch_allowed','training_authorized','test_authorized','spending_authorized'))and all(value.get(k)==v for k,v in expected.items()),'OBSERVED_CONTINUATION_TYPED_GATE');return value


def merge_inputs(original,extra):
    need(all(k not in original or original[k]==v for k,v in extra.items()),'OBSERVED_CONTINUATION_INPUT_ROLE_COLLISION');return{**copy.deepcopy(original),**copy.deepcopy(extra)}


def snapshot(binding,pref):
    sr=ref(SNAPSHOT);policy=read(sr);need(pref=={'path':str(o.POLICY),'sha256':sr['sha256']}and all(policy.get(k)is(k in o.DEV_FLAGS[:2])for k in o.DEV_FLAGS),'OBSERVED_CONTINUATION_SNAPSHOT_FLAGS');require_gate(policy.get(STAGE),binding,True);need(policy[STAGE].get('design')==DESIGN,'OBSERVED_CONTINUATION_SNAPSHOT_DESIGN');return sr


def continuation_schedule(parent,original_producer,design):
    source=parent['schedule'];original=original_producer['schedule'];order=design['order'];need(source['contract_sha256']==order['source648_schedule_sha256']and original['contract_sha256']==order['original432_schedule_sha256']and source['logical_batches']==source['optimizer_updates']==648 and source['passes']==3 and source['batches_per_pass']==216 and source['rows']==20727 and source['batch_rows']==96 and order['repeat_original_take_indices_start_inclusive']==216 and order['repeat_original_take_indices_stop_exclusive']==432,'OBSERVED_CONTINUATION_002_AUTHENTIC_648_SCHEDULE')
    appended=original['cohort_takes'][216:432];take=o.np.concatenate([o.np.asarray(x,dtype=o.np.int64)for x in appended]);need(o.np.array_equal(o.np.sort(take),o.np.arange(20727,dtype=o.np.int64))and all(all(type(i)is int for i in x)and len(x)>0 and all(a<b for a,b in zip(x,x[1:]))for x in appended),'OBSERVED_CONTINUATION_002_FULL_BIJECTION')
    schedule=copy.deepcopy(source);schedule.pop('contract_sha256')
    for key in('cohort_takes','logical_step_rows','cohort_take_state_sha256_by_logical_batch','update_flags'):schedule[key]+=copy.deepcopy(original[key][216:432])
    for key in('raw_permutation_state_sha256_by_pass','actual_sorted_partition_stream_state_sha256_by_pass'):schedule[key].append(original[key][1])
    schedule.update(schema_version='gx1_native_v3_observed_train_continuation_002_schedule_v1',logical_batches=864,optimizer_updates=864,passes=4,skipped_batches=0,original_prefix_schedule_sha256=source['contract_sha256'],original_take_indices_appended=list(range(216,432)),order_mode='repeat_locked_original_second_pass_once')
    need(len(schedule['cohort_takes'])==len(schedule['logical_step_rows'])==len(schedule['update_flags'])==864 and all(schedule['update_flags'])and all(len(t)==n for t,n in zip(schedule['cohort_takes'],schedule['logical_step_rows'])),'OBSERVED_CONTINUATION_002_EXACT_UPDATE_VECTORS');return train.seal(schedule)


def continuation_trajectory(parent,schedule):
    value=copy.deepcopy(parent['metadata']['trajectory']);value.pop('contract_sha256');value.update(schema_version='gx1_native_v3_observed_train_continuation_002_trajectory_v1',schedule=schedule,observed_continuation_design=DESIGN,original_prefix_trajectory_sha256=parent['metadata']['trajectory']['contract_sha256']);return train.seal(value)


def operation_limits():return prior.operation_limits()


def validate_start_receipt(parent,receipt,design,report):
    need(receipt['checkpoint']==design['authentic_start_FINAL648_checkpoint']and report['checkpoint_receipts']['648']==design['authentic_start_FINAL648_receipt']and receipt['original_plan']==receipt['execution_plan']==design['producer_plan']and receipt['producer_lane']=='continuation'and type(receipt['logical_cursor'])is type(receipt['optimizer_updates'])is int and receipt['logical_cursor']==receipt['optimizer_updates']==648 and receipt['trajectory_sha256']==parent['metadata']['trajectory']['contract_sha256'],'OBSERVED_CONTINUATION_002_OWN_FINAL648_LINEAGE')
    proof=train.require_recovery_checkpoint_proof(receipt['proof']);need(proof==report['full_FINAL648_proof']and proof['trajectory']==core.state_sha(parent['metadata']['trajectory'])and proof['position']==core.state_sha(train.position(parent['schedule'],648))and proof['training']==core.state_sha(True)and proof['scheduler_state']==core.state_sha(None),'OBSERVED_CONTINUATION_002_FINAL648_FULL10_METADATA');return proof


def parent_metadata():
    design=read(DESIGN);parent=read(design['producer_plan']);result=read(design['producer_result']);terminal=read(design['producer_terminal']);receipt=read(design['authentic_start_FINAL648_receipt']);claim=read(result['claim']);acceptance=read(design['producer_acceptance'])
    need(result['plan']==terminal['plan']==claim['plan']==design['producer_plan']and terminal['result']==design['producer_result']and terminal['claim']==result['claim']and result['status']=='COMPLETE_V3_OBSERVED_TRAIN_CONTINUATION'and terminal['status']=='COMPLETE'and terminal['exit_code']==0 and result['error']is None and set(result['bookends'])=={'plan','sources','inputs','policy','boots','HEAD'}and all(v is True for v in result['bookends'].values())and result['report']['ending_cursor']==648 and result['report']['new_optimizer_updates']==216,'OBSERVED_CONTINUATION_002_CLOSED_GENUINE_PARENT')
    need(terminal['remaining_invocations']==claim['remaining_invocations']==0 and terminal['relaunch_allowed']is claim['relaunch_allowed']is False and acceptance['status']=='COMPLETE'and acceptance['plan']==design['producer_plan']and acceptance['result']==design['producer_result']and acceptance['terminal']==design['producer_terminal']and acceptance['claim']==result['claim']and acceptance['all_bookends_pass']is acceptance['hardware_guard_accepted']is acceptance['actual_process_gone']is True,'OBSERVED_CONTINUATION_002_PARENT_NO_RELAUNCH')
    core.future_cpu_parent_stopped(claim['process_identity'])
    validate_start_receipt(parent,receipt,design,result['report'])
    closed,original_producer,original,fr,original_receipt,identity=q.original_metadata();need(closed['plan']==design['unchanged_original_recipe_model_input_owner']and parent['producer_plan']==closed['plan']and parent['metadata']['original']==original and parent['metadata']['producer']==original_producer and parent['physical_input_identity']==identity and parent['model_contract']==original['model_contract']and parent['recipe']==original['recipe'],'OBSERVED_CONTINUATION_002_FRESH_UNCHANGED_PHYSICAL_IDENTITY')
    baseline=read(design['BASELINE648_score']);numeric=read(design['BASELINE648_numeric_owner_score']);expected_scores=copy.deepcopy(closed['value']['report']['scores']);expected_scores['FINAL648']=design['BASELINE648_score'];need(receipt['proof']['scores']==core.state_sha(expected_scores)and baseline['actual_phase']=='FINAL648'and baseline['global_cursor']==648 and baseline['numeric_owner_score']==design['BASELINE648_numeric_owner_score']and baseline['numeric_owner_logits']==numeric['logits']==design['BASELINE648_logits']and numeric['rows']==20727 and numeric['phase']=='PASS2'and parent['inputs']['cohort_prior']==design['frozen_COHORT_prior']and original['metadata']['prior']==design['frozen_factory_prior'],'OBSERVED_CONTINUATION_002_AUTHENTIC_BASELINE648_PRIORS')
    core.future_cuda_runtime_matches(parent['metadata']['runtime'],receipt['producer_runtime'])
    refs={'observed_continuation_002_design':DESIGN,'observed_continuation_002_parent_plan':design['producer_plan'],'observed_continuation_002_original_plan':design['unchanged_original_recipe_model_input_owner'],'observed_continuation_002_parent_claim':result['claim'],'observed_continuation_002_parent_policy_snapshot':claim['admitted_policy_snapshot'],**{'observed_continuation_002_'+k:design[k]for k in('producer_result','producer_terminal','producer_acceptance','authentic_start_FINAL648_receipt','authentic_start_FINAL648_checkpoint','BASELINE648_score','BASELINE648_numeric_owner_score','BASELINE648_logits','original_BASELINE432_score','original_factory_INITIAL_score','basis','endpoint_assessment')}}
    for binding in refs.values():
        if Path(binding['path']).suffix=='.json':read(binding)
    return design,parent,original_producer,original,receipt,identity,refs


def build_plan(runtime,source_commit=None):
    design,parent,original_producer,original,receipt,identity,extra=parent_metadata();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();need(source_commit in(None,head)and runtime['source_commit']==head,'OBSERVED_CONTINUATION_002_FROZEN_SOURCE_RUNTIME');core.future_cuda_require_runtime(runtime);train.runtime_compatible(parent['metadata']['trajectory']['runtime_compatibility'],runtime)
    owners={}
    for name,digest in parent['source_owners'].items():
        current=ref(REPO/name)['sha256'];need(name in o.GUARDS or current==digest,'OBSERVED_CONTINUATION_002_UNCHANGED_NUMERIC_SOURCE:'+name);owners[name]=current
    owners.update({name:ref(REPO/name)['sha256']for name in(SELF,TEST,GUARD_TEST)});old={name:hashlib.sha256(subprocess.check_output(['git','show',ANCHOR+':'+name],cwd=REPO)).hexdigest()for name in o.GUARDS};bridge=guard.native_v3_observed_train_continuation_002_guard_source_bridge(old,producer_commit=ANCHOR)
    inputs=merge_inputs(parent['inputs'],extra);roles=sorted(k for k,v in inputs.items()if Path(v['path']).suffix not in('.npz','.npy','.pt','.pth','.parquet','.safetensors','.pkl','.pickle','.joblib'));schedule=continuation_schedule(parent,original_producer,design);limits=operation_limits();need(limits==design['limits'],'OBSERVED_CONTINUATION_002_EXACT_REUSED_LIMITS')
    return {'schema_version':'gx1_native_v3_observed_train_continuation_plan_v2','stage':STAGE,'source_commit':head,'output_root':str(ROOT),'source_owners':owners,'guard_source_bridge':bridge,'design':DESIGN,'producer_plan':design['producer_plan'],'original_producer_plan':design['unchanged_original_recipe_model_input_owner'],'authentic_final_receipt':design['authentic_start_FINAL648_receipt'],'authentic_final_checkpoint':receipt['checkpoint'],'metadata':{'parent_plan':design['producer_plan'],'original_plan':design['unchanged_original_recipe_model_input_owner'],'receipt':receipt,'trajectory':continuation_trajectory(parent,schedule),'runtime':copy.deepcopy(runtime)},'schedule':schedule,'recipe':parent['recipe'],'model_contract':parent['model_contract'],'backbone_gradient_policy':None,'physical_input_identity':identity,'inputs':inputs,'metadata_input_roles':roles,'payload_stat':{k:{'size':Path(v['path']).stat().st_size,'mtime_ns':Path(v['path']).stat().st_mtime_ns}for k,v in inputs.items()if k not in roles},'starting_cursor':648,'ending_cursor':864,'checkpoint_cursors':[649,864],'observer_enabled':True,'limits':limits,'phase_alias':design['phase_alias'],'execution':guard._native_v3_development_execution('warm'),'execution_profile':PROFILE,'scope':design['scope']}


def inspect(sha,admitted=False):
    bound=guard._native_v3_observed_train_continuation_002_bound_plan(sha);binding=bound['plan'];plan=read(binding);need(plan==build_plan(plan['metadata']['runtime'],plan['source_commit']),'OBSERVED_CONTINUATION_EXACT_PLAN');o.recovery_check_inputs(plan);pref=ref(o.POLICY);require_gate(read(pref).get(STAGE),binding,admitted);live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['metadata']['runtime'],live)
    if admitted:snapshot(binding,pref)
    return plan,binding,pref,live


def compare_reviews(before,after):
    value=prior.compare_reviews(before,after)
    def alias(item):
        if type(item)is dict:return{alias(k):alias(v)for k,v in item.items()}
        if type(item)is list:return[alias(v)for v in item]
        if type(item)is str:return item.replace('BASELINE432','BASELINE648').replace('FINAL648','FINAL864')
        return item
    return alias(value)


def physical(plan,binding,root,live):
    import torch
    from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
    from gx1.audit.native_future_state_step_observer_v1 import NativeFutureStateStepObserver
    parent=read(plan['metadata']['parent_plan']);producer=read(plan['metadata']['original_plan']);original=parent['metadata']['original'];receipt=plan['metadata']['receipt'];context=o.development_nn_context(original);stream=o.issue_mix_actual_stream(context,producer);need(stream==read(plan['inputs']['observed_continuation_producer_stream_binding']),'OBSERVED_CONTINUATION_GENUINE_SOURCE_ISSUE_TRUTH_STREAM');stream_ref=o.development_publish(root,'ORIGINAL_STREAM_BINDING.json',stream);counts=dict.fromkeys(plan['limits'],0);counts['feature_cache_payload_loads']=counts['target_archive_loads']=1
    appended=o.np.concatenate([o.np.asarray(t,dtype=o.np.int64)for t in producer['schedule']['cohort_takes'][216:432]]);need(o.np.array_equal(o.np.sort(appended),o.np.arange(20727,dtype=o.np.int64))and plan['schedule']==continuation_schedule(parent,producer,read(DESIGN)),'OBSERVED_CONTINUATION_ACTUAL_FULL_BIJECTION_BEFORE_NN');core.future_cuda_seed_and_runtime({'recipe':original['recipe'],'metadata':{'runtime':live}})
    model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=context['input_binding'],architecture=original['metadata']['architecture'],device=torch.device('cuda'),train_prior=context['prior'],target_contract=context['target_contract'],backbone_gradient_policy=None);counts['model_constructions']=1;need(model.research_objective==original['model_contract']and core.state_sha(model.state_dict())==original['initial_reference']['model_state_sha256']and core.state_sha(core.training_rng())==original['initial_reference']['rng_sha256'],'OBSERVED_CONTINUATION_FACTORY_FUNCTION_RNG');optimizer=torch.optim.AdamW(model.parameters(),**original['recipe']['optimizer']);counts['optimizer_constructions']=1;need(not optimizer.state,'OBSERVED_CONTINUATION_FRESH_OPTIMIZER');cp_ref=plan['authentic_final_checkpoint'];need(cp_ref==receipt['checkpoint']and ref(cp_ref['path'])==cp_ref,'OBSERVED_CONTINUATION_002_ONLY_GENUINE_FINAL648_LOAD');cp=torch.load(cp_ref['path'],map_location='cpu',weights_only=True);counts['checkpoint_loads']=1;need(cp['producer']=={'original_plan':plan['producer_plan'],'lane':'continuation','runtime':receipt['producer_runtime']},'OBSERVED_CONTINUATION_002_ACTUAL_FINAL648_PRODUCER');train.strict_restore(model,optimizer,cp,trajectory=parent['metadata']['trajectory'],original_plan_ref=plan['producer_plan'],expected_cursor=648,current_runtime=live,expected_state_proof=receipt['proof']);counts['model_restores']=1;steps=copy.deepcopy(cp['steps']);scores=copy.deepcopy(cp['scores']);del cp
    original_proof=train.checkpoint_proof(train.make_checkpoint(model,optimizer,trajectory=parent['metadata']['trajectory'],original_plan_ref=plan['producer_plan'],lane='continuation',cursor=648,steps=steps,scores=scores,producer_runtime=live));need(original_proof==receipt['proof']and all(p.grad is None for p in model.parameters()),'OBSERVED_CONTINUATION_AUTHENTIC_FULL10_START');observer=NativeFutureStateStepObserver();receipts={}
    def counter(key):
        def hook(*_):counts[key]+=1;need(counts[key]<=plan['limits'][key],'OBSERVED_CONTINUATION_FORWARD_BOUND')
        return hook
    hooks=[model.register_forward_pre_hook(counter('model_forwards')),model.backbone.register_forward_pre_hook(counter('backbone_forwards')),model.readout.register_forward_pre_hook(counter('readout_forwards'))]
    def make(cursor):return train.make_checkpoint(model,optimizer,trajectory=plan['metadata']['trajectory'],original_plan_ref=binding,lane='continuation',cursor=cursor,steps=copy.deepcopy(steps),scores=copy.deepcopy(scores),producer_runtime=live)
    try:
        train.release_phase_allocator_cache(model,optimizer,label='OBSERVED_CONTINUATION_RESTORE_TO_TRAIN')
        for index in range(648,864):
            original_index=index-432;take=o.np.asarray(producer['schedule']['cohort_takes'][original_index],dtype=o.np.int64);packet=o.adapter.packet(context,take);record=core.future_cuda_step(model,optimizer,packet,original,counts,index,observer=observer);record['skip_update']=False;record['observed_continuation_binding']={'global_index':index,'original_batch_index':original_index,'binding':o.issue_mix_step_binding(producer['schedule'],stream,original_index)};steps.append(record)
            if index==648:receipts['649']=o.development_checkpoint(root,'CURSOR649.pt',make(649),counts,binding)
        need(len(observer.reports)==216 and observer._pending is None,'OBSERVED_CONTINUATION_ALL216_OBSERVATIONS');counts['observer_records']=len(observer.reports);observations=o.development_publish(root,'OBSERVATIONS.json',{'actual_start':'BASELINE648','actual_end':'FINAL864','reports':observer.reports,'selection':'ALL216steps_ALL5TF','new_diagnostic_forwards':0})
        train.release_phase_allocator_cache(model,optimizer,label='OBSERVED_CONTINUATION_TRAIN_TO_FINAL864_EVAL');before_eval=train.checkpoint_proof(make(864));modes={n:m.training for n,m in model.named_modules()};namespace=root/'FINAL864';namespace.mkdir(exist_ok=False);numeric_score=o.development_score(model,context,original,namespace,'PASS2',counts);need(train.checkpoint_proof(make(864))==before_eval and {n:m.training for n,m in model.named_modules()}==modes and all(p.grad is None for p in model.parameters()),'OBSERVED_CONTINUATION_EVAL_MODEL_OPTIMIZER_MODE_RNG_UNCHANGED');current=read(numeric_score);baseline=read(plan['inputs']['observed_continuation_002_BASELINE648_numeric_owner_score']);comparison=o.development_publish(root,'FINAL864_MINUS_BASELINE648.json',compare_reviews(baseline['review'],current['review']));wrapper=o.development_publish(root,'FINAL864_SCORE.json',{'actual_phase':'FINAL864','global_cursor':864,'actual_before':'BASELINE648','numeric_owner_phase':'PASS2','phase_alias':plan['phase_alias'],'numeric_owner_score':numeric_score,'numeric_owner_logits':current['logits'],'BASELINE648_score':plan['inputs']['observed_continuation_002_BASELINE648_score'],'BASELINE648_numeric_owner_score':plan['inputs']['observed_continuation_002_BASELINE648_numeric_owner_score'],'comparison':comparison,'all5_TRAIN_verdict':current['review']['candidate_TRAIN_comparison_verdict'],'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'});scores['FINAL864']=wrapper
        receipts['864']=o.development_checkpoint(root,'FINAL864.pt',make(864),counts,binding);need(counts==plan['limits']and torch.cuda.max_memory_reserved()<=plan['execution']['cuda_peak_reserved_bytes_max'],'OBSERVED_CONTINUATION_EXACT_COUNTS_VRAM');return {'counts':counts,'checkpoint_receipts':receipts,'observations':observations,'final_score':wrapper,'numeric_owner_score':numeric_score,'comparison':comparison,'starting_cursor':648,'ending_cursor':864,'full_FINAL864_proof':train.checkpoint_proof(make(864)),'original_start_proof':original_proof,'original_stream_binding':stream_ref,'actual_candidate':'FINAL864','new_optimizer_updates':216,'all5_TRAIN_verdict':current['review']['candidate_TRAIN_comparison_verdict'],'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'}
    finally:
        for hook in hooks:hook.remove()


def payload_allowlist(plan,root):
    allowed={v['path']for v in plan['inputs'].values()if Path(v['path']).suffix in('.npz','.npy','.pt','.pth','.parquet','.safetensors','.pkl','.pickle','.joblib')};allowed.update(str(root/name)for name in('CURSOR649.pt','FINAL864.pt'));allowed.add(str(root/'FINAL864'/'PASS2_LOGITS.npz'));return allowed


def bookends(plan,binding,pref,boots):
    observed=recovery.observe_boots();checks={'plan':lambda:ref(PLAN)==binding,'sources':lambda:all(ref(REPO/n)['sha256']==v for n,v in plan['source_owners'].items()),'inputs':lambda:all(ref(v['path'])==v for v in plan['inputs'].values())and ref(SNAPSHOT)['sha256']==pref['sha256'],'policy':lambda:ref(o.POLICY)==pref,'boots':lambda:recovery.boot_matches(boots,observed),'HEAD':lambda:subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_commit']};book={};errors={}
    for name,check in checks.items():
        try:book[name]=bool(check())
        except BaseException as exc:book[name]=False;errors[name]={'type':type(exc).__name__,'message':str(exc)[:2400]}
    return book,{'claimed_boots':boots,'boot_observation':observed,'check_errors':errors}


def run(sha):
    execution=guard.require_native_v3_observed_train_continuation_002_execution(plan_sha256=sha);plan,binding,pref,live=inspect(sha,True);root=ROOT;root.mkdir(exist_ok=False);boots={k:live[k]for k in('physical_windows_boot_id','wsl_boot_id')};started=time.monotonic();claim=o.development_publish(root,'CLAIM.json',{'plan':binding,'policy_at_claim':pref,'admitted_policy_snapshot':snapshot(binding,pref),'execution':execution,'boots':boots,'process_identity':core.future_cpu_process_identity(),'remaining_invocations':0,'relaunch_allowed':False});report=None;error=None
    def stop(*_):raise TimeoutError('OBSERVED_CONTINUATION_INTERNAL_WALL_OR_SIGNAL')
    handlers={sig:signal.getsignal(sig)for sig in(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    for sig in handlers:signal.signal(sig,stop)
    signal.setitimer(signal.ITIMER_REAL,plan['execution']['internal_seconds'])
    try:
        allowed=payload_allowlist(plan,root);sys.addaudithook(lambda e,a:o.development_io_audit(allowed,e,a));o.recovery_check_inputs(plan,physical=True);report=physical(plan,binding,root,live);need(core.common.firewall_state()['denied_attempts']==0,'OBSERVED_CONTINUATION_ZERO_FIREWALL_DENIALS')
    except BaseException as exc:error=type(exc).__name__+':'+str(exc)[:2400]
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);book,evidence=bookends(plan,binding,pref,boots);error=error or(None if all(book.values())else'OBSERVED_CONTINUATION_BOOKEND_FAILED');rr=o.development_publish(root,'RESULT.json',{'status':'COMPLETE_V3_OBSERVED_TRAIN_CONTINUATION_002'if error is None else'FAILED_CONSUMED','plan':binding,'claim':claim,'report':report,'error':error,'bookends':book,'bookend_evidence':evidence,'elapsed_seconds':time.monotonic()-started,'actual_candidate':'FINAL864','test_data_used':False,'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'});tr=o.development_publish(root,'TERMINAL.json',{'status':'COMPLETE'if error is None else'FAILED_CONSUMED','exit_code':0 if error is None else 1,'plan':binding,'claim':claim,'result':rr,'remaining_invocations':0,'relaunch_allowed':False})
        for sig,handler in handlers.items():signal.signal(sig,handler)
    if error:raise RuntimeError(error)
    return {'result':rr,'terminal':tr}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--v3-observed-train-continuation-002-plan-sha256',required=True);parser.add_argument('--preflight',action='store_true');args=parser.parse_args();sha=args.v3_observed_train_continuation_002_plan_sha256;need(sys.argv[1:]==['--v3-observed-train-continuation-002-plan-sha256',sha]+(['--preflight']if args.preflight else[]),'OBSERVED_CONTINUATION_EXACT_CLI')
    if args.preflight:
        guard.require_native_v3_observed_train_continuation_002_execution(plan_sha256=sha);plan,binding,pref,live=inspect(sha,False);import torch;need(not torch.cuda.is_initialized(),'OBSERVED_CONTINUATION_PREFLIGHT_NO_CUDA');print(json.dumps({'status':STATUS_PREFLIGHT,'plan':binding,'policy':pref,'physical_input_identity':plan['physical_input_identity'],'models':0,'payloads_decoded':0,'CUDA_initialized':False,'CLAIM_created':ROOT.exists()}))
    else:print(json.dumps(run(sha),sort_keys=True))

if __name__=='__main__':main()
