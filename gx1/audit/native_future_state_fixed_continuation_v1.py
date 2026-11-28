"""Execute one authenticated fixed216-step continuation; no automatic next run."""
from pathlib import Path
import argparse,copy,hashlib,json,signal,subprocess,sys,time
REPO=Path(__file__).resolve().parents[2]
if str(REPO)not in sys.path:sys.path.insert(0,str(REPO))
from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
_FIREWALL=install_native_m5_direction_io_firewall()
from gx1.audit import native_future_state_observed_train_continuation_v1 as prior
from gx1.contracts import gx1_capped_execution_v1 as guard
q=prior.q;o=q.o;core=q.core;train=q.train;need=q.need;read=q.read;ref=q.ref;E=o.E;recovery=prior.recovery
SELF='gx1/audit/native_future_state_fixed_continuation_v1.py';TEST='tests/test_native_future_state_fixed_continuation_v1.py';GUARD_TEST='tests/test_native_fixed_continuation_guard_v1.py'
ANCHOR='180ef9e01f2e6ed73420928bda52b8222fff2f49';PROFILE='gx1_native_fixed_continuation_trainer20g_v1';STATUS_PREFLIGHT='PREFLIGHT_V3_FIXED_CONTINUATION_METADATA_ONLY'
SOURCE_REFERENCE_PLAN={'path':str(REPO/'configs/research/NATIVE_V3_FROZEN_FINAL864_MODE_GAP_20261128_001.json'),'sha256':'0739ca250b1629a5dac9400aff0a4ce1ee16626a23e1a6ccd4863e1ebd9cd388'}


def run_paths(run_id):
    value=guard.native_fixed_continuation_run_paths(run_id);return {k:value[k]for k in('name','stage','plan','root','snapshot')}|{'design':value['designref']}


def input_prefix(run_id):return 'fixed_continuation_'+'_'.join(run_id.split('_')[-2:])+'_'


def gate(binding,admitted=False):
    return {'status':'ADMITTED_EXACT_V3_FIXED_CONTINUATION'if admitted else'PREPARED_CLOSED_NOT_ADMITTED','plan':binding,'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':admitted,'test_authorized':False,'spending_authorized':False}


def require_gate(value,binding,admitted=False):
    expected=gate(binding,admitted);need(type(value)is dict and type(value.get('remaining_invocations'))is int and all(type(value.get(k))is bool for k in('execution_authorized','relaunch_allowed','training_authorized','test_authorized','spending_authorized'))and all(value.get(k)==v for k,v in expected.items()),'FIXED_CONTINUATION_TYPED_GATE');return value


merge_inputs=prior.merge_inputs
operation_limits=prior.operation_limits


def validate_design(run_id,design):
    start=design['scope']['starting_global_cursor'];end=design['scope']['locked_final_global_cursor'];scope=design['scope'];need(design['experiment_id']==run_id and design['stage']==run_id.lower()and design['execution_owner']['module']=='gx1.audit.native_future_state_fixed_continuation_v1'and design['execution_owner']['CLI']==['--v3-fixed-continuation-run-id',run_id,'--v3-fixed-continuation-plan-sha256','EXACT_PLAN_SHA256'],'FIXED_CONTINUATION_DESIGN_RUN_BINDING')
    need(type(start)is type(end)is int and start>=864 and start%216==0 and end==start+216 and design['checkpoint_cursors']==[start+1,end]and design['limits']==operation_limits()and all(scope[k]==v for k,v in{'new_optimizer_updates':216,'training_forwards':216,'fixed_terminal_EVAL_forwards':81,'total_model_forwards':297,'observer_records':216,'selected_TRAIN_rows':20727,'batch_rows':96,'evaluation_batch_rows':256,'genuine_families':8,'D1_macro_preserved':True,'backbone_gradient_policy':None,'timeframes':list(train.TF),'years':[2021,2022,2023,2024,2025]}.items()),'FIXED_CONTINUATION_FIXED_BUDGET')
    need(design['phase_alias']['actual_before']=='BASELINE'+str(start)and design['phase_alias']['actual_candidate']=='FINAL'+str(end)and design['phase_alias']['numeric_owner_phase']=='PASS2'and design['phase_alias']['numeric_owner_spec']==design['frozen_numeric_owner_spec']and design['order']['repeat_original_take_indices_start_inclusive']==216 and design['order']['repeat_original_take_indices_stop_exclusive']==432,'FIXED_CONTINUATION_FIXED_ORDER_PHASE');return start,end


def snapshot(run_id,binding,pref):
    paths=run_paths(run_id);sr=ref(paths['snapshot']);policy=read(sr);need(pref=={'path':str(o.POLICY),'sha256':sr['sha256']}and all(policy.get(k)is(k in o.DEV_FLAGS[:2])for k in o.DEV_FLAGS),'FIXED_CONTINUATION_SNAPSHOT_FLAGS');require_gate(policy.get(paths['stage']),binding,True);need(policy[paths['stage']].get('design')==paths['design'],'FIXED_CONTINUATION_SNAPSHOT_DESIGN');return sr


def validate_start_receipt(parent,receipt,design,report,start):
    final='FINAL'+str(start);need(receipt['checkpoint']==design['authentic_start_'+final+'_checkpoint']and report['checkpoint_receipts'][str(start)]==design['authentic_start_'+final+'_receipt']and receipt['original_plan']==receipt['execution_plan']==design['producer_plan']and receipt['producer_lane']=='continuation'and type(receipt['logical_cursor'])is type(receipt['optimizer_updates'])is int and receipt['logical_cursor']==receipt['optimizer_updates']==start and receipt['trajectory_sha256']==parent['metadata']['trajectory']['contract_sha256'],'FIXED_CONTINUATION_OWN_PARENT_LINEAGE')
    schema=parent['schema_version']
    if schema=='gx1_native_v3_observed_train_continuation_plan_v2':need(start==864,'FIXED_CONTINUATION_LEGACY_PARENT_CURSOR');expected=report['full_FINAL864_proof']
    elif schema=='gx1_native_v3_fixed_continuation_plan_v1':expected=report['full_final_state_proof']
    else:raise ValueError('FIXED_CONTINUATION_UNQUALIFIED_PARENT_SCHEMA')
    proof=train.require_recovery_checkpoint_proof(receipt['proof']);need(proof==expected and proof['trajectory']==core.state_sha(parent['metadata']['trajectory'])and proof['position']==core.state_sha(train.position(parent['schedule'],start))and proof['training']==core.state_sha(True)and proof['scheduler_state']==core.state_sha(None),'FIXED_CONTINUATION_AUTHENTIC_FULL10_PARENT');return proof


def final_metadata(run_id):
    paths=run_paths(run_id);design=read(paths['design']);start,end=validate_design(run_id,design);baseline='BASELINE'+str(start);final='FINAL'+str(start);parent=read(design['producer_plan']);result=read(design['producer_result']);terminal=read(design['producer_terminal']);claim=read(result['claim']);acceptance=read(design['producer_acceptance']);receipt=read(design['authentic_start_'+final+'_receipt'])
    statuses={'gx1_native_v3_observed_train_continuation_plan_v2':'COMPLETE_V3_OBSERVED_TRAIN_CONTINUATION_002','gx1_native_v3_fixed_continuation_plan_v1':'COMPLETE_V3_FIXED_CONTINUATION'};need(parent['schema_version']in statuses and result['status']==statuses[parent['schema_version']]and result['plan']==terminal['plan']==claim['plan']==design['producer_plan']and terminal['result']==design['producer_result']and terminal['claim']==result['claim']and terminal['status']==acceptance['status']=='COMPLETE'and terminal['exit_code']==0 and result['error']is None and result['report']['ending_cursor']==start and set(result['bookends'])=={'plan','sources','inputs','policy','boots','HEAD'}and all(v is True for v in result['bookends'].values()),'FIXED_CONTINUATION_CLOSED_PARENT')
    need(acceptance['plan']==design['producer_plan']and acceptance['result']==design['producer_result']and acceptance['terminal']==design['producer_terminal']and acceptance['claim']==result['claim']and acceptance['all_bookends_pass']is acceptance['hardware_guard_accepted']is acceptance['actual_process_gone']is True and terminal['remaining_invocations']==claim['remaining_invocations']==0 and terminal['relaunch_allowed']is claim['relaunch_allowed']is False,'FIXED_CONTINUATION_ACCEPTED_CONSUMED_PARENT');core.future_cpu_parent_stopped(claim['process_identity']);validate_start_receipt(parent,receipt,design,result['report'],start)
    closed,original_producer,original,fr,original_receipt,identity=q.original_metadata();need(closed['plan']==parent['original_producer_plan']==design['original_producer_plan']and parent['model_contract']==original['model_contract']and parent['recipe']==original['recipe']and parent['physical_input_identity']==identity,'FIXED_CONTINUATION_FRESH_PHYSICAL_IDENTITY');wrapper=read(design[baseline+'_score']);numeric=read(design[baseline+'_numeric_owner_score']);cohort=read(design['frozen_COHORT_prior']);factory=read(design['frozen_factory_prior']);spec=read(design['frozen_numeric_owner_spec'])
    need(wrapper['actual_phase']==final and wrapper['global_cursor']==start and wrapper['numeric_owner_score']==design[baseline+'_numeric_owner_score']and wrapper['numeric_owner_logits']==numeric['logits']==design[baseline+'_logits']and numeric['phase']=='PASS2'and numeric['rows']==20727 and numeric['batches']==81 and numeric['review']['COHORT_score_prior_sha256']==cohort['contract_sha256']and numeric['review']['factory_prior_sha256']==factory['contract_sha256']and numeric['review']['spec_sha256']==spec['contract_sha256']and spec['cohort_prior']==cohort and spec['whole_TRAIN_spec']['prior_contract_sha256']==factory['contract_sha256']and original['inputs']['cohort_prior']==design['frozen_COHORT_prior']and original['metadata']['prior']==design['frozen_factory_prior']and original['inputs']['cohort_spec']==design['frozen_numeric_owner_spec'],'FIXED_CONTINUATION_UNCHANGED_BASELINE_CONTRACTS')
    core.future_cuda_runtime_matches(parent['metadata']['runtime'],receipt['producer_runtime']);prefix=input_prefix(run_id);refs={prefix+'design':paths['design'],prefix+'parent_plan':design['producer_plan'],prefix+'original_plan':design['original_producer_plan'],prefix+'start_receipt':design['authentic_start_'+final+'_receipt'],prefix+'start_checkpoint':receipt['checkpoint'],prefix+'baseline_score':design[baseline+'_score'],prefix+'baseline_numeric_owner_score':design[baseline+'_numeric_owner_score'],prefix+'baseline_logits':design[baseline+'_logits'],prefix+'parent_claim':result['claim'],prefix+'parent_policy_snapshot':claim['admitted_policy_snapshot'],prefix+'source_reference_plan':SOURCE_REFERENCE_PLAN,**{prefix+k:design[k]for k in('producer_result','producer_terminal','producer_acceptance','basis','mode_gap_actual_review','mode_gap_assessment')}}
    return design,parent,original_producer,original,receipt,identity,refs


def continuation_schedule(parent,original_producer,design):
    start=design['scope']['starting_global_cursor'];end=design['scope']['locked_final_global_cursor'];source=parent['schedule'];original=original_producer['schedule'];need(source['contract_sha256']==design['order']['source'+str(start)+'_schedule_sha256']and original['contract_sha256']==design['order']['original432_schedule_sha256']and source['logical_batches']==source['optimizer_updates']==start and source['batches_per_pass']==216 and source['rows']==20727 and source['batch_rows']==96,'FIXED_CONTINUATION_EXACT_SOURCE_SCHEDULE')
    appended=original['cohort_takes'][216:432];take=o.np.concatenate([o.np.asarray(t,dtype=o.np.int64)for t in appended]);need(o.np.array_equal(o.np.sort(take),o.np.arange(20727,dtype=o.np.int64))and all(all(type(i)is int for i in t)and len(t)>0 and all(a<b for a,b in zip(t,t[1:]))for t in appended),'FIXED_CONTINUATION_FULL_BIJECTION');schedule=copy.deepcopy(source);schedule.pop('contract_sha256')
    for key in('cohort_takes','logical_step_rows','cohort_take_state_sha256_by_logical_batch','update_flags'):schedule[key]+=copy.deepcopy(original[key][216:432])
    for key in('raw_permutation_state_sha256_by_pass','actual_sorted_partition_stream_state_sha256_by_pass'):schedule[key].append(original[key][1])
    schedule.update(schema_version='gx1_native_v3_fixed_continuation_schedule_v1',logical_batches=end,optimizer_updates=end,passes=source['passes']+1,skipped_batches=0,original_prefix_schedule_sha256=source['contract_sha256'],original_take_indices_appended=list(range(216,432)),order_mode='repeat_locked_original_second_pass_once');need(len(schedule['cohort_takes'])==len(schedule['logical_step_rows'])==len(schedule['update_flags'])==end and all(schedule['update_flags'])and all(len(t)==n for t,n in zip(schedule['cohort_takes'],schedule['logical_step_rows'])),'FIXED_CONTINUATION_EXACT_UPDATE_VECTORS');return train.seal(schedule)


def continuation_trajectory(parent,schedule,design_ref):
    value=copy.deepcopy(parent['metadata']['trajectory']);value.pop('contract_sha256');value.update(schema_version='gx1_native_v3_fixed_continuation_trajectory_v1',schedule=schedule,fixed_continuation_design=design_ref,original_prefix_trajectory_sha256=parent['metadata']['trajectory']['contract_sha256']);return train.seal(value)


def build_plan(run_id,runtime,source_commit=None):
    paths=run_paths(run_id);design,parent,original_producer,original,receipt,identity,extra=final_metadata(run_id);start,end=validate_design(run_id,design);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();need(source_commit in(None,head)and runtime['source_commit']==head,'FIXED_CONTINUATION_FROZEN_SOURCE_RUNTIME');core.future_cuda_require_runtime(runtime);train.runtime_compatible(parent['metadata']['trajectory']['runtime_compatibility'],runtime);source=read(SOURCE_REFERENCE_PLAN);owners={}
    for name,digest in source['source_owners'].items():
        current=ref(REPO/name)['sha256'];need(name in o.GUARDS or current==digest,'FIXED_CONTINUATION_UNCHANGED_QUALIFIED_SOURCE:'+name);owners[name]=current
    owners.update({name:ref(REPO/name)['sha256']for name in(SELF,TEST,GUARD_TEST)});need(all(name in o.GUARDS or owners.get(name)==digest for name,digest in parent['source_owners'].items()),'FIXED_CONTINUATION_UNCHANGED_PARENT_SOURCE');old={name:hashlib.sha256(subprocess.check_output(['git','show',ANCHOR+':'+name],cwd=REPO)).hexdigest()for name in o.GUARDS};bridge=guard.native_v3_fixed_continuation_guard_source_bridge(old,producer_commit=ANCHOR);inputs=merge_inputs(parent['inputs'],extra);roles=sorted(k for k,v in inputs.items()if Path(v['path']).suffix not in('.npz','.npy','.pt','.pth','.parquet','.safetensors','.pkl','.pickle','.joblib'));schedule=continuation_schedule(parent,original_producer,design)
    return {'schema_version':'gx1_native_v3_fixed_continuation_plan_v1','run_id':run_id,'stage':paths['stage'],'source_commit':head,'output_root':str(paths['root']),'source_owners':owners,'source_reference_plan':SOURCE_REFERENCE_PLAN,'guard_source_bridge':bridge,'design':paths['design'],'producer_plan':design['producer_plan'],'original_producer_plan':design['original_producer_plan'],'authentic_final_receipt':design['authentic_start_FINAL'+str(start)+'_receipt'],'authentic_final_checkpoint':receipt['checkpoint'],'metadata':{'parent_plan':design['producer_plan'],'original_plan':design['original_producer_plan'],'receipt':receipt,'trajectory':continuation_trajectory(parent,schedule,paths['design']),'runtime':copy.deepcopy(runtime)},'schedule':schedule,'recipe':parent['recipe'],'model_contract':parent['model_contract'],'physical_input_identity':identity,'backbone_gradient_policy':None,'starting_cursor':start,'ending_cursor':end,'checkpoint_cursors':[start+1,end],'observer_enabled':True,'input_prefix':input_prefix(run_id),'inputs':inputs,'metadata_input_roles':roles,'payload_stat':{k:{'size':Path(v['path']).stat().st_size,'mtime_ns':Path(v['path']).stat().st_mtime_ns}for k,v in inputs.items()if k not in roles},'limits':operation_limits(),'phase_alias':design['phase_alias'],'execution':guard._native_v3_development_execution('warm'),'execution_profile':PROFILE,'scope':design['scope']}


def inspect(run_id,sha,admitted=False):
    paths=run_paths(run_id);bound=guard._native_v3_fixed_continuation_bound_plan(run_id,sha);binding=bound['plan'];plan=read(binding);need(plan==build_plan(run_id,plan['metadata']['runtime'],plan['source_commit']),'FIXED_CONTINUATION_EXACT_PLAN');o.recovery_check_inputs(plan);pref=ref(o.POLICY);require_gate(read(pref).get(paths['stage']),binding,admitted);live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['metadata']['runtime'],live)
    if admitted:snapshot(run_id,binding,pref)
    return plan,binding,pref,live


def compare_reviews(before,after,baseline,candidate):
    value=prior.compare_reviews(before,after)
    def alias(item):
        if type(item)is dict:return{alias(k):alias(v)for k,v in item.items()}
        if type(item)is list:return[alias(v)for v in item]
        if type(item)is str:return item.replace('BASELINE432',baseline).replace('FINAL648',candidate)
        return item
    return alias(value)


def physical(plan,binding,root,live):
    import torch
    from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
    from gx1.audit.native_future_state_step_observer_v1 import NativeFutureStateStepObserver
    paths=run_paths(plan['run_id']);design=read(paths['design']);start,end=validate_design(plan['run_id'],design);baseline='BASELINE'+str(start);candidate='FINAL'+str(end);prefix=plan['input_prefix'];parent=read(plan['metadata']['parent_plan']);producer=read(plan['metadata']['original_plan']);original=producer['metadata']['original'];receipt=plan['metadata']['receipt'];context=o.development_nn_context(original);stream=o.issue_mix_actual_stream(context,producer);need(stream==read(plan['inputs']['observed_continuation_producer_stream_binding'])and plan['schedule']==continuation_schedule(parent,producer,design),'FIXED_CONTINUATION_ACTUAL_STREAM_AND_FULL_BIJECTION');stream_ref=o.development_publish(root,'ORIGINAL_STREAM_BINDING.json',stream);counts=dict.fromkeys(plan['limits'],0);counts['feature_cache_payload_loads']=counts['target_archive_loads']=1;core.future_cuda_seed_and_runtime({'recipe':original['recipe'],'metadata':{'runtime':live}})
    model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=context['input_binding'],architecture=original['metadata']['architecture'],device=torch.device('cuda'),train_prior=context['prior'],target_contract=context['target_contract'],backbone_gradient_policy=None);counts['model_constructions']=1;need(model.research_objective==original['model_contract']and core.state_sha(model.state_dict())==original['initial_reference']['model_state_sha256']and core.state_sha(core.training_rng())==original['initial_reference']['rng_sha256'],'FIXED_CONTINUATION_FACTORY_FUNCTION_RNG');optimizer=torch.optim.AdamW(model.parameters(),**original['recipe']['optimizer']);counts['optimizer_constructions']=1;need(not optimizer.state,'FIXED_CONTINUATION_FRESH_OPTIMIZER');cp_ref=plan['authentic_final_checkpoint'];need(cp_ref==receipt['checkpoint']and ref(cp_ref['path'])==cp_ref,'FIXED_CONTINUATION_ONLY_GENUINE_START_LOAD');cp=torch.load(cp_ref['path'],map_location='cpu',weights_only=True);counts['checkpoint_loads']=1;need(cp['producer']=={'original_plan':plan['producer_plan'],'lane':'continuation','runtime':receipt['producer_runtime']},'FIXED_CONTINUATION_OWN_START_PRODUCER');train.strict_restore(model,optimizer,cp,trajectory=parent['metadata']['trajectory'],original_plan_ref=plan['producer_plan'],expected_cursor=start,current_runtime=live,expected_state_proof=receipt['proof']);counts['model_restores']=1;steps=copy.deepcopy(cp['steps']);scores=copy.deepcopy(cp['scores']);del cp
    original_proof=train.checkpoint_proof(train.make_checkpoint(model,optimizer,trajectory=parent['metadata']['trajectory'],original_plan_ref=plan['producer_plan'],lane='continuation',cursor=start,steps=steps,scores=scores,producer_runtime=live));need(original_proof==receipt['proof']and all(p.grad is None for p in model.parameters()),'FIXED_CONTINUATION_AUTHENTIC_FULL10_START');observer=NativeFutureStateStepObserver();receipts={}
    def counter(key):
        def hook(*_):counts[key]+=1;need(counts[key]<=plan['limits'][key],'FIXED_CONTINUATION_FORWARD_BOUND')
        return hook
    hooks=[model.register_forward_pre_hook(counter('model_forwards')),model.backbone.register_forward_pre_hook(counter('backbone_forwards')),model.readout.register_forward_pre_hook(counter('readout_forwards'))]
    def make(cursor):return train.make_checkpoint(model,optimizer,trajectory=plan['metadata']['trajectory'],original_plan_ref=binding,lane='continuation',cursor=cursor,steps=copy.deepcopy(steps),scores=copy.deepcopy(scores),producer_runtime=live)
    try:
        train.release_phase_allocator_cache(model,optimizer,label='FIXED_CONTINUATION_RESTORE_TO_TRAIN')
        for index in range(start,end):
            original_index=216+(index-start);take=o.np.asarray(producer['schedule']['cohort_takes'][original_index],dtype=o.np.int64);packet=o.adapter.packet(context,take);record=core.future_cuda_step(model,optimizer,packet,original,counts,index,observer=observer);record['skip_update']=False;record['fixed_continuation_binding']={'run_id':plan['run_id'],'global_index':index,'original_batch_index':original_index,'binding':o.issue_mix_step_binding(producer['schedule'],stream,original_index)};steps.append(record)
            if index==start:receipts[str(start+1)]=o.development_checkpoint(root,'CURSOR'+str(start+1)+'.pt',make(start+1),counts,binding)
        need(len(observer.reports)==216 and observer._pending is None,'FIXED_CONTINUATION_ALL216_OBSERVATIONS');counts['observer_records']=216;observations=o.development_publish(root,'OBSERVATIONS.json',{'actual_start':baseline,'actual_end':candidate,'reports':observer.reports,'selection':'ALL216steps_ALL5TF','new_diagnostic_forwards':0});train.release_phase_allocator_cache(model,optimizer,label='FIXED_CONTINUATION_TRAIN_TO_EVAL');before_eval=train.checkpoint_proof(make(end));modes={n:m.training for n,m in model.named_modules()};namespace=root/candidate;namespace.mkdir(exist_ok=False);numeric_score=o.development_score(model,context,original,namespace,'PASS2',counts);need(train.checkpoint_proof(make(end))==before_eval and {n:m.training for n,m in model.named_modules()}==modes and all(p.grad is None for p in model.parameters()),'FIXED_CONTINUATION_EVAL_STATE_MODE_RNG_UNCHANGED');current=read(numeric_score);before=read(plan['inputs'][prefix+'baseline_numeric_owner_score']);comparison=o.development_publish(root,candidate+'_MINUS_'+baseline+'.json',compare_reviews(before['review'],current['review'],baseline,candidate));wrapper=o.development_publish(root,candidate+'_SCORE.json',{'actual_phase':candidate,'global_cursor':end,'actual_before':baseline,'numeric_owner_phase':'PASS2','phase_alias':plan['phase_alias'],'numeric_owner_score':numeric_score,'numeric_owner_logits':current['logits'],'baseline_score':plan['inputs'][prefix+'baseline_score'],'baseline_numeric_owner_score':plan['inputs'][prefix+'baseline_numeric_owner_score'],'comparison':comparison,'all5_TRAIN_verdict':current['review']['candidate_TRAIN_comparison_verdict'],'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'});scores[candidate]=wrapper;receipts[str(end)]=o.development_checkpoint(root,candidate+'.pt',make(end),counts,binding);need(counts==plan['limits']and torch.cuda.max_memory_reserved()<=plan['execution']['cuda_peak_reserved_bytes_max'],'FIXED_CONTINUATION_EXACT_COUNTS_VRAM');return {'counts':counts,'checkpoint_receipts':receipts,'observations':observations,'final_score':wrapper,'numeric_owner_score':numeric_score,'comparison':comparison,'full_final_state_proof':train.checkpoint_proof(make(end)),'original_start_proof':original_proof,'original_stream_binding':stream_ref,'starting_cursor':start,'ending_cursor':end,'actual_candidate':candidate,'new_optimizer_updates':216,'all5_TRAIN_verdict':current['review']['candidate_TRAIN_comparison_verdict'],'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'}
    finally:
        for hook in hooks:hook.remove()


def payload_allowlist(plan,root):
    allowed={v['path']for v in plan['inputs'].values()if Path(v['path']).suffix in('.npz','.npy','.pt','.pth','.parquet','.safetensors','.pkl','.pickle','.joblib')};allowed.update(str(root/name)for name in('CURSOR'+str(plan['starting_cursor']+1)+'.pt','FINAL'+str(plan['ending_cursor'])+'.pt'));allowed.add(str(root/('FINAL'+str(plan['ending_cursor']))/'PASS2_LOGITS.npz'));return allowed


def bookends(plan,binding,pref,boots):
    observed=recovery.observe_boots();checks={'plan':lambda:ref(Path(binding['path']))==binding,'sources':lambda:all(ref(REPO/n)['sha256']==v for n,v in plan['source_owners'].items()),'inputs':lambda:all(ref(v['path'])==v for v in plan['inputs'].values())and snapshot(plan['run_id'],binding,pref)['sha256']==pref['sha256'],'policy':lambda:ref(o.POLICY)==pref,'boots':lambda:recovery.boot_matches(boots,observed),'HEAD':lambda:subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_commit']};book={};errors={}
    for name,check in checks.items():
        try:book[name]=bool(check())
        except BaseException as exc:book[name]=False;errors[name]={'type':type(exc).__name__,'message':str(exc)[:2400]}
    return book,{'claimed_boots':boots,'boot_observation':observed,'check_errors':errors}


def run(run_id,sha):
    paths=run_paths(run_id);execution=guard.require_native_v3_fixed_continuation_execution(run_id=run_id,plan_sha256=sha);plan,binding,pref,live=inspect(run_id,sha,True);root=paths['root'];root.mkdir(exist_ok=False);boots={k:live[k]for k in('physical_windows_boot_id','wsl_boot_id')};started=time.monotonic();claim=o.development_publish(root,'CLAIM.json',{'plan':binding,'run_id':run_id,'policy_at_claim':pref,'admitted_policy_snapshot':snapshot(run_id,binding,pref),'execution':execution,'boots':boots,'process_identity':core.future_cpu_process_identity(),'remaining_invocations':0,'relaunch_allowed':False});report=None;error=None
    def stop(*_):raise TimeoutError('FIXED_CONTINUATION_INTERNAL_WALL_OR_SIGNAL')
    handlers={sig:signal.getsignal(sig)for sig in(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    for sig in handlers:signal.signal(sig,stop)
    signal.setitimer(signal.ITIMER_REAL,plan['execution']['internal_seconds'])
    try:
        allowed=payload_allowlist(plan,root);sys.addaudithook(lambda e,a:o.development_io_audit(allowed,e,a));o.recovery_check_inputs(plan,physical=True);report=physical(plan,binding,root,live);need(core.common.firewall_state()['denied_attempts']==0,'FIXED_CONTINUATION_ZERO_FIREWALL_DENIALS')
    except BaseException as exc:error=type(exc).__name__+':'+str(exc)[:2400]
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);book,evidence=bookends(plan,binding,pref,boots);error=error or(None if all(book.values())else'FIXED_CONTINUATION_BOOKEND_FAILED');rr=o.development_publish(root,'RESULT.json',{'status':'COMPLETE_V3_FIXED_CONTINUATION'if error is None else'FAILED_CONSUMED','plan':binding,'run_id':run_id,'claim':claim,'report':report,'error':error,'bookends':book,'bookend_evidence':evidence,'elapsed_seconds':time.monotonic()-started,'actual_candidate':plan['phase_alias']['actual_candidate'],'test_data_used':False,'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'});tr=o.development_publish(root,'TERMINAL.json',{'status':'COMPLETE'if error is None else'FAILED_CONSUMED','exit_code':0 if error is None else 1,'plan':binding,'claim':claim,'result':rr,'remaining_invocations':0,'relaunch_allowed':False})
        for sig,handler in handlers.items():signal.signal(sig,handler)
    if error:raise RuntimeError(error)
    return {'result':rr,'terminal':tr}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--v3-fixed-continuation-run-id',required=True);parser.add_argument('--v3-fixed-continuation-plan-sha256',required=True);parser.add_argument('--preflight',action='store_true');args=parser.parse_args();run_id=args.v3_fixed_continuation_run_id;sha=args.v3_fixed_continuation_plan_sha256;need(sys.argv[1:]==['--v3-fixed-continuation-run-id',run_id,'--v3-fixed-continuation-plan-sha256',sha]+(['--preflight']if args.preflight else[]),'FIXED_CONTINUATION_EXACT_CLI')
    if args.preflight:
        paths=run_paths(run_id);guard.require_native_v3_fixed_continuation_execution(run_id=run_id,plan_sha256=sha);plan,binding,pref,live=inspect(run_id,sha,False);import torch;need(not torch.cuda.is_initialized(),'FIXED_CONTINUATION_PREFLIGHT_NO_CUDA');print(json.dumps({'status':STATUS_PREFLIGHT,'run_id':run_id,'plan':binding,'policy':pref,'physical_input_identity':plan['physical_input_identity'],'models':0,'payloads_decoded':0,'CUDA_initialized':False,'CLAIM_created':paths['root'].exists()}))
    else:print(json.dumps(run(run_id,sha),sort_keys=True))

if __name__=='__main__':main()
