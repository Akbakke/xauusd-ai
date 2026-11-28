"""One locked observed TRAIN continuation FINAL432 -> FINAL648; never a fit series."""
from pathlib import Path
import argparse,copy,hashlib,json,signal,subprocess,sys,time
REPO=Path(__file__).resolve().parents[2]
if str(REPO)not in sys.path:sys.path.insert(0,str(REPO))
from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
_FIREWALL=install_native_m5_direction_io_firewall()
from gx1.audit import native_future_state_step_observer_qualify_v1 as q
from gx1.audit import native_future_state_step_observer_cold_recovery_v1 as recovery
from gx1.contracts import gx1_capped_execution_v1 as guard
o=q.o;core=q.core;train=q.train;need=q.need;read=q.read;ref=q.ref;E=o.E
NAME='NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_20261128_001';STAGE=NAME.lower();PLAN=REPO/'configs/research'/(NAME+'.json');ROOT=Path('/home/andre2/GX1_RUNS')/NAME
SNAPSHOT=E/'V3_OBSERVED_TRAIN_CONTINUATION_ADMITTED_POLICY_20261128_001.json'
DESIGN={'path':str(E/'V3_OBSERVED_TRAIN_CONTINUATION_DESIGN_20261128_001.json'),'sha256':'5b04f35e2765c45c2cf331cada40f0d96497b1561e69183add7a38be6a5eb269'}
ANCHOR='4c6124d988e977143006559840adb5ddffca9955';SELF='gx1/audit/native_future_state_observed_train_continuation_v1.py';TEST='tests/test_native_future_state_observed_train_continuation_v1.py';GUARD_TEST='tests/test_native_observed_train_continuation_guard_v1.py'
PROFILE='gx1_native_observed_train_continuation_trainer20g_v1';STATUS_PREFLIGHT='PREFLIGHT_V3_OBSERVED_TRAIN_CONTINUATION_METADATA_ONLY'


def gate(binding,admitted=False):
    return {'status':'ADMITTED_EXACT_V3_OBSERVED_TRAIN_CONTINUATION'if admitted else'PREPARED_CLOSED_NOT_ADMITTED','plan':binding,'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':admitted,'test_authorized':False,'spending_authorized':False}


def require_gate(value,binding,admitted=False):
    expected=gate(binding,admitted);need(type(value)is dict and type(value.get('remaining_invocations'))is int and all(type(value.get(k))is bool for k in('execution_authorized','relaunch_allowed','training_authorized','test_authorized','spending_authorized'))and all(value.get(k)==v for k,v in expected.items()),'OBSERVED_CONTINUATION_TYPED_GATE');return value


def merge_inputs(original,extra):
    need(all(k not in original or original[k]==v for k,v in extra.items()),'OBSERVED_CONTINUATION_INPUT_ROLE_COLLISION');return{**copy.deepcopy(original),**copy.deepcopy(extra)}


def snapshot(binding,pref):
    sr=ref(SNAPSHOT);policy=read(sr);need(pref=={'path':str(o.POLICY),'sha256':sr['sha256']}and all(policy.get(k)is(k in o.DEV_FLAGS[:2])for k in o.DEV_FLAGS),'OBSERVED_CONTINUATION_SNAPSHOT_FLAGS');require_gate(policy.get(STAGE),binding,True);need(policy[STAGE].get('design')==DESIGN,'OBSERVED_CONTINUATION_SNAPSHOT_DESIGN');return sr


def continuation_schedule(producer,design):
    source=producer['schedule'];order=design['order'];need(source['contract_sha256']==order['source_schedule_sha256']and source['logical_batches']==source['optimizer_updates']==432 and source['passes']==2 and source['batches_per_pass']==216 and source['rows']==20727 and source['batch_rows']==96 and order['reuse_original_take_indices_start_inclusive']==216 and order['reuse_original_take_indices_stop_exclusive']==432,'OBSERVED_CONTINUATION_EXACT_ORIGINAL_SCHEDULE')
    appended=source['cohort_takes'][216:432];take=o.np.concatenate([o.np.asarray(x,dtype=o.np.int64)for x in appended]);need(o.np.array_equal(o.np.sort(take),o.np.arange(20727,dtype=o.np.int64))and all(all(type(i)is int for i in x)and len(x)>0 and all(a<b for a,b in zip(x,x[1:]))for x in appended),'OBSERVED_CONTINUATION_FULL_COHORT_BIJECTION')
    schedule=copy.deepcopy(source);schedule.pop('contract_sha256')
    for key in('cohort_takes','logical_step_rows','cohort_take_state_sha256_by_logical_batch','update_flags'):schedule[key]+=copy.deepcopy(source[key][216:432])
    for key in('raw_permutation_state_sha256_by_pass','actual_sorted_partition_stream_state_sha256_by_pass'):schedule[key].append(source[key][1])
    schedule.update(schema_version='gx1_native_v3_observed_train_continuation_schedule_v1',logical_batches=648,optimizer_updates=648,passes=3,skipped_batches=0,original_prefix_schedule_sha256=source['contract_sha256'],original_take_indices_appended=list(range(216,432)),order_mode='repeat_locked_original_second_pass_once')
    need(len(schedule['cohort_takes'])==len(schedule['logical_step_rows'])==len(schedule['update_flags'])==648 and all(schedule['update_flags'])and all(len(t)==n for t,n in zip(schedule['cohort_takes'],schedule['logical_step_rows'])),'OBSERVED_CONTINUATION_EXACT_UPDATE_VECTORS');return train.seal(schedule)


def continuation_trajectory(producer,schedule):
    value=copy.deepcopy(producer['metadata']['trajectory']);value.pop('contract_sha256');value.update(schema_version='gx1_native_v3_observed_train_continuation_trajectory_v1',schedule=schedule,observed_continuation_design=DESIGN,original_prefix_trajectory_sha256=producer['metadata']['trajectory']['contract_sha256']);return train.seal(value)


def operation_limits():
    value=dict.fromkeys(o.development_limits('fit'),0);value.update(model_constructions=1,optimizer_constructions=1,feature_cache_payload_loads=1,target_archive_loads=1,checkpoint_loads=1,model_restores=1,model_forwards=297,backbone_forwards=297,readout_forwards=297,backward_calls=216,optimizer_steps=216,checkpoint_writes=2,checkpoint_readbacks=2,checkpoint_prepublication_loads=2,checkpoint_receipt_writes=2,logit_archive_writes=1,logit_archive_readbacks=1,observer_records=216);return value


def original_metadata():
    design=read(DESIGN);closed,producer,original,fr,receipt,identity=q.original_metadata();need(closed['plan']==design['producer_fit_plan']and fr==design['authentic_start_FINAL432_receipt']and receipt['checkpoint']==design['authentic_start_FINAL432_checkpoint']and closed['value']['report']['scores']['PASS2']==design['BASELINE432_score'],'OBSERVED_CONTINUATION_GENUINE_FINAL432')
    baseline=read(design['BASELINE432_score']);need(baseline['logits']==design['BASELINE432_logits']and baseline['rows']==20727 and baseline['phase']=='PASS2'and original['inputs']['cohort_prior']==design['frozen_COHORT_prior']and original['inputs']['cohort_spec']==design['frozen_cohort_review_spec']and original['metadata']['prior']==design['frozen_factory_prior']and closed['value']['report']['scores']['INITIAL']==design['original_factory_INITIAL_score'],'OBSERVED_CONTINUATION_UNCHANGED_BASELINE_PRIORS')
    qualified=read(design['observer_operational_qualification']);need(qualified['observer_operational_qualification_complete']is qualified['all10_C434_proofs_equal']is qualified['all6_bookends_pass']is True and qualified['technical_weights_never_initializer']is True,'OBSERVED_CONTINUATION_QUALIFIED_OBSERVER');parent=read(qualified['plan']);terminal=read(qualified['terminal']);result=read(qualified['result']);need(terminal['status']=='COMPLETE'and terminal['exit_code']==0 and terminal['result']==qualified['result']and result['error']is None and all(v is True for v in result['bookends'].values()),'OBSERVED_CONTINUATION_CLOSED_OBSERVER_QUALIFICATION')
    refs={'observed_continuation_design':DESIGN,'observed_continuation_producer_plan':closed['plan'],'observed_continuation_original_FINAL432_receipt':fr,'observed_continuation_original_FINAL432_checkpoint':receipt['checkpoint'],'observed_continuation_producer_result':closed['result'],'observed_continuation_producer_terminal':closed['terminal'],'observed_continuation_producer_claim':closed['claim'],'observed_continuation_producer_admitted_policy':closed['admitted_policy_snapshot'],'observed_continuation_producer_stream_binding':closed['value']['report']['stream_binding'],'observed_continuation_qualified_observer_plan':qualified['plan'],'observed_continuation_qualified_observer_result':qualified['result'],'observed_continuation_qualified_observer_terminal':qualified['terminal'],**{'observed_continuation_'+k:design[k]for k in('BASELINE432_score','BASELINE432_logits','original_factory_INITIAL_score','observer_source_qualification','observer_operational_qualification')}}
    for binding in refs.values():
        if Path(binding['path']).suffix=='.json':read(binding)
    return design,closed,producer,original,receipt,identity,parent,refs


def build_plan(runtime,source_commit=None):
    design,closed,producer,original,receipt,identity,parent,extra=original_metadata();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();need(source_commit in(None,head)and runtime['source_commit']==head,'OBSERVED_CONTINUATION_FROZEN_SOURCE_RUNTIME');core.future_cuda_require_runtime(runtime);train.runtime_compatible(producer['metadata']['trajectory']['runtime_compatibility'],runtime)
    owners={}
    for name,digest in parent['source_owners'].items():
        current=ref(REPO/name)['sha256'];need(name in o.GUARDS or current==digest,'OBSERVED_CONTINUATION_UNCHANGED_NUMERIC_SOURCE:'+name);owners[name]=current
    owners.update({name:ref(REPO/name)['sha256']for name in(SELF,TEST,GUARD_TEST)});old={name:hashlib.sha256(subprocess.check_output(['git','show',ANCHOR+':'+name],cwd=REPO)).hexdigest()for name in o.GUARDS};bridge=guard.native_v3_observed_train_continuation_guard_source_bridge(old,producer_commit=ANCHOR)
    inputs=merge_inputs(producer['inputs'],extra);roles=sorted(k for k,v in inputs.items()if Path(v['path']).suffix not in('.npz','.npy','.pt','.pth','.parquet','.safetensors','.pkl','.pickle','.joblib'));schedule=continuation_schedule(producer,design)
    return {'schema_version':'gx1_native_v3_observed_train_continuation_plan_v1','stage':STAGE,'source_commit':head,'output_root':str(ROOT),'source_owners':owners,'guard_source_bridge':bridge,'design':DESIGN,'producer_plan':closed['plan'],'authentic_final_receipt':extra['observed_continuation_original_FINAL432_receipt'],'authentic_final_checkpoint':receipt['checkpoint'],'metadata':{'original':original,'producer':producer,'receipt':receipt,'trajectory':continuation_trajectory(producer,schedule),'runtime':copy.deepcopy(runtime)},'schedule':schedule,'recipe':original['recipe'],'model_contract':original['model_contract'],'backbone_gradient_policy':None,'physical_input_identity':identity,'inputs':inputs,'metadata_input_roles':roles,'payload_stat':{k:{'size':Path(v['path']).stat().st_size,'mtime_ns':Path(v['path']).stat().st_mtime_ns}for k,v in inputs.items()if k not in roles},'starting_cursor':432,'ending_cursor':648,'checkpoint_cursors':[433,648],'observer_enabled':True,'limits':operation_limits(),'phase_alias':design['phase_alias'],'execution':guard._native_v3_development_execution('warm'),'execution_profile':PROFILE,'scope':design['scope']}


def inspect(sha,admitted=False):
    bound=guard._native_v3_observed_train_continuation_bound_plan(sha);binding=bound['plan'];plan=read(binding);need(plan==build_plan(plan['metadata']['runtime'],plan['source_commit']),'OBSERVED_CONTINUATION_EXACT_PLAN');o.recovery_check_inputs(plan);pref=ref(o.POLICY);require_gate(read(pref).get(STAGE),binding,admitted);live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['metadata']['runtime'],live)
    if admitted:snapshot(binding,pref)
    return plan,binding,pref,live


def compare_reviews(before,after):
    need(before['truth_sha256']==after['truth_sha256']and before['support']==after['support']and before['selection_ref']==after['selection_ref']and before['COHORT_score_prior_sha256']==after['COHORT_score_prior_sha256']and before['factory_prior_sha256']==after['factory_prior_sha256'],'OBSERVED_CONTINUATION_SAME_SCORE_POPULATIONS')
    def delta(a,b):return {'BASELINE432':a,'FINAL648':b,'FINAL648_minus_BASELINE432':None if a is None or b is None else b-a}
    def panel(a,b):
        need(all(a[k]==b[k]for k in('count','unique_target_starts','true_positive_count','true_negative_count')),'OBSERVED_CONTINUATION_SAME_PANEL_SUPPORT');result={k:a[k]for k in('count','unique_target_starts','true_positive_count','true_negative_count')}
        for weighting,x,y in(('row',a['model'],b['model']),('equal_target',(a['equal_target_weight']or{}).get('model'),(b['equal_target_weight']or{}).get('model'))):
            result[weighting]={key:delta(None if x is None else x.get(key),None if y is None else y.get(key))for key in('NLL','brier','mean_positive_probability','mean_probability_minus_truth_pp')}
        result['row_AUC']=delta(a['rank_and_variation']['AUC'],b['rank_and_variation']['AUC'])
        if 'calibration'in a:
            result['calibration_mean_error_pp']={w:delta(None if a['calibration'][w]is None else a['calibration'][w]['mean_probability_minus_truth_pp'],None if b['calibration'][w]is None else b['calibration'][w]['mean_probability_minus_truth_pp'])for w in('row','equal_target')}
        return result
    pooled={tf:{task:panel(before['all_TF'][tf][task],after['all_TF'][tf][task])for task in('activity','side')}for tf in train.TF};years={tf:{str(year):{task:panel(before['by_UTC_target_year'][tf][str(year)][task],after['by_UTC_target_year'][tf][str(year)][task])for task in('activity','side')}for year in(2021,2022,2023,2024,2025)}for tf in train.TF}
    return {'actual_before':'BASELINE432','actual_candidate':'FINAL648','all_TF':pooled,'by_UTC_target_year':years,'BASELINE432_verdict':before['candidate_TRAIN_comparison_verdict'],'FINAL648_verdict':after['candidate_TRAIN_comparison_verdict'],'criterion':'Unchanged all5TF proper-score/rank/UTC-year criteria; every improvement/worsening/UNKNOWN retained.','later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'}


def physical(plan,binding,root,live):
    import torch
    from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
    from gx1.audit.native_future_state_step_observer_v1 import NativeFutureStateStepObserver
    original=plan['metadata']['original'];producer=plan['metadata']['producer'];receipt=plan['metadata']['receipt'];context=o.development_nn_context(original);stream=o.issue_mix_actual_stream(context,producer);need(stream==read(plan['inputs']['observed_continuation_producer_stream_binding']),'OBSERVED_CONTINUATION_GENUINE_SOURCE_ISSUE_TRUTH_STREAM');stream_ref=o.development_publish(root,'ORIGINAL_STREAM_BINDING.json',stream);counts=dict.fromkeys(plan['limits'],0);counts['feature_cache_payload_loads']=counts['target_archive_loads']=1
    appended=o.np.concatenate([o.np.asarray(t,dtype=o.np.int64)for t in producer['schedule']['cohort_takes'][216:432]]);need(o.np.array_equal(o.np.sort(appended),o.np.arange(20727,dtype=o.np.int64))and plan['schedule']==continuation_schedule(producer,read(DESIGN)),'OBSERVED_CONTINUATION_ACTUAL_FULL_BIJECTION_BEFORE_NN');core.future_cuda_seed_and_runtime({'recipe':original['recipe'],'metadata':{'runtime':live}})
    model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=context['input_binding'],architecture=original['metadata']['architecture'],device=torch.device('cuda'),train_prior=context['prior'],target_contract=context['target_contract'],backbone_gradient_policy=None);counts['model_constructions']=1;need(model.research_objective==original['model_contract']and core.state_sha(model.state_dict())==original['initial_reference']['model_state_sha256']and core.state_sha(core.training_rng())==original['initial_reference']['rng_sha256'],'OBSERVED_CONTINUATION_FACTORY_FUNCTION_RNG');optimizer=torch.optim.AdamW(model.parameters(),**original['recipe']['optimizer']);counts['optimizer_constructions']=1;need(not optimizer.state,'OBSERVED_CONTINUATION_FRESH_OPTIMIZER');cp_ref=plan['authentic_final_checkpoint'];need(cp_ref==receipt['checkpoint']and ref(cp_ref['path'])==cp_ref,'OBSERVED_CONTINUATION_ONLY_GENUINE_FINAL432_LOAD');cp=torch.load(cp_ref['path'],map_location='cpu',weights_only=True);counts['checkpoint_loads']=1;o.issue_mix_validate_checkpoint(cp,producer,plan['producer_plan'],432,stream);train.strict_restore(model,optimizer,cp,trajectory=producer['metadata']['trajectory'],original_plan_ref=plan['producer_plan'],expected_cursor=432,current_runtime=live,expected_state_proof=receipt['proof']);counts['model_restores']=1;steps=copy.deepcopy(cp['steps']);scores=copy.deepcopy(cp['scores']);del cp
    original_proof=train.checkpoint_proof(train.make_checkpoint(model,optimizer,trajectory=producer['metadata']['trajectory'],original_plan_ref=plan['producer_plan'],lane='fit',cursor=432,steps=steps,scores=scores,producer_runtime=live));need(original_proof==receipt['proof']and all(p.grad is None for p in model.parameters()),'OBSERVED_CONTINUATION_AUTHENTIC_FULL10_START');observer=NativeFutureStateStepObserver();receipts={}
    def counter(key):
        def hook(*_):counts[key]+=1;need(counts[key]<=plan['limits'][key],'OBSERVED_CONTINUATION_FORWARD_BOUND')
        return hook
    hooks=[model.register_forward_pre_hook(counter('model_forwards')),model.backbone.register_forward_pre_hook(counter('backbone_forwards')),model.readout.register_forward_pre_hook(counter('readout_forwards'))]
    def make(cursor):return train.make_checkpoint(model,optimizer,trajectory=plan['metadata']['trajectory'],original_plan_ref=binding,lane='continuation',cursor=cursor,steps=copy.deepcopy(steps),scores=copy.deepcopy(scores),producer_runtime=live)
    try:
        train.release_phase_allocator_cache(model,optimizer,label='OBSERVED_CONTINUATION_RESTORE_TO_TRAIN')
        for index in range(432,648):
            original_index=index-216;take=o.np.asarray(producer['schedule']['cohort_takes'][original_index],dtype=o.np.int64);packet=o.adapter.packet(context,take);record=core.future_cuda_step(model,optimizer,packet,original,counts,index,observer=observer);record['skip_update']=False;record['observed_continuation_binding']={'global_index':index,'original_batch_index':original_index,'binding':o.issue_mix_step_binding(producer['schedule'],stream,original_index)};steps.append(record)
            if index==432:receipts['433']=o.development_checkpoint(root,'CURSOR433.pt',make(433),counts,binding)
        need(len(observer.reports)==216 and observer._pending is None,'OBSERVED_CONTINUATION_ALL216_OBSERVATIONS');counts['observer_records']=len(observer.reports);observations=o.development_publish(root,'OBSERVATIONS.json',{'actual_start':'BASELINE432','actual_end':'FINAL648','reports':observer.reports,'selection':'ALL216steps_ALL5TF','new_diagnostic_forwards':0})
        train.release_phase_allocator_cache(model,optimizer,label='OBSERVED_CONTINUATION_TRAIN_TO_FINAL648_EVAL');before_eval=train.checkpoint_proof(make(648));modes={n:m.training for n,m in model.named_modules()};namespace=root/'FINAL648';namespace.mkdir(exist_ok=False);numeric_score=o.development_score(model,context,original,namespace,'PASS2',counts);need(train.checkpoint_proof(make(648))==before_eval and {n:m.training for n,m in model.named_modules()}==modes and all(p.grad is None for p in model.parameters()),'OBSERVED_CONTINUATION_EVAL_MODEL_OPTIMIZER_MODE_RNG_UNCHANGED');current=read(numeric_score);baseline=read(plan['inputs']['observed_continuation_BASELINE432_score']);comparison=o.development_publish(root,'FINAL648_MINUS_BASELINE432.json',compare_reviews(baseline['review'],current['review']));wrapper=o.development_publish(root,'FINAL648_SCORE.json',{'actual_phase':'FINAL648','global_cursor':648,'actual_before':'BASELINE432','numeric_owner_phase':'PASS2','phase_alias':plan['phase_alias'],'numeric_owner_score':numeric_score,'numeric_owner_logits':current['logits'],'BASELINE432_score':plan['inputs']['observed_continuation_BASELINE432_score'],'comparison':comparison,'all5_TRAIN_verdict':current['review']['candidate_TRAIN_comparison_verdict'],'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'});scores['FINAL648']=wrapper
        receipts['648']=o.development_checkpoint(root,'FINAL648.pt',make(648),counts,binding);need(counts==plan['limits']and torch.cuda.max_memory_reserved()<=plan['execution']['cuda_peak_reserved_bytes_max'],'OBSERVED_CONTINUATION_EXACT_COUNTS_VRAM');return {'counts':counts,'checkpoint_receipts':receipts,'observations':observations,'final_score':wrapper,'numeric_owner_score':numeric_score,'comparison':comparison,'starting_cursor':432,'ending_cursor':648,'full_FINAL648_proof':train.checkpoint_proof(make(648)),'original_start_proof':original_proof,'original_stream_binding':stream_ref,'actual_candidate':'FINAL648','new_optimizer_updates':216,'all5_TRAIN_verdict':current['review']['candidate_TRAIN_comparison_verdict'],'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'}
    finally:
        for hook in hooks:hook.remove()


def payload_allowlist(plan,root):
    allowed={v['path']for v in plan['inputs'].values()if Path(v['path']).suffix in('.npz','.npy','.pt','.pth','.parquet','.safetensors','.pkl','.pickle','.joblib')};allowed.update(str(root/name)for name in('CURSOR433.pt','FINAL648.pt'));allowed.add(str(root/'FINAL648'/'PASS2_LOGITS.npz'));return allowed


def bookends(plan,binding,pref,boots):
    observed=recovery.observe_boots();checks={'plan':lambda:ref(PLAN)==binding,'sources':lambda:all(ref(REPO/n)['sha256']==v for n,v in plan['source_owners'].items()),'inputs':lambda:all(ref(v['path'])==v for v in plan['inputs'].values())and ref(SNAPSHOT)['sha256']==pref['sha256'],'policy':lambda:ref(o.POLICY)==pref,'boots':lambda:recovery.boot_matches(boots,observed),'HEAD':lambda:subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_commit']};book={};errors={}
    for name,check in checks.items():
        try:book[name]=bool(check())
        except BaseException as exc:book[name]=False;errors[name]={'type':type(exc).__name__,'message':str(exc)[:2400]}
    return book,{'claimed_boots':boots,'boot_observation':observed,'check_errors':errors}


def run(sha):
    execution=guard.require_native_v3_observed_train_continuation_execution(plan_sha256=sha);plan,binding,pref,live=inspect(sha,True);root=ROOT;root.mkdir(exist_ok=False);boots={k:live[k]for k in('physical_windows_boot_id','wsl_boot_id')};started=time.monotonic();claim=o.development_publish(root,'CLAIM.json',{'plan':binding,'policy_at_claim':pref,'admitted_policy_snapshot':snapshot(binding,pref),'execution':execution,'boots':boots,'process_identity':core.future_cpu_process_identity(),'remaining_invocations':0,'relaunch_allowed':False});report=None;error=None
    def stop(*_):raise TimeoutError('OBSERVED_CONTINUATION_INTERNAL_WALL_OR_SIGNAL')
    handlers={sig:signal.getsignal(sig)for sig in(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    for sig in handlers:signal.signal(sig,stop)
    signal.setitimer(signal.ITIMER_REAL,plan['execution']['internal_seconds'])
    try:
        allowed=payload_allowlist(plan,root);sys.addaudithook(lambda e,a:o.development_io_audit(allowed,e,a));o.recovery_check_inputs(plan,physical=True);report=physical(plan,binding,root,live);need(core.common.firewall_state()['denied_attempts']==0,'OBSERVED_CONTINUATION_ZERO_FIREWALL_DENIALS')
    except BaseException as exc:error=type(exc).__name__+':'+str(exc)[:2400]
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);book,evidence=bookends(plan,binding,pref,boots);error=error or(None if all(book.values())else'OBSERVED_CONTINUATION_BOOKEND_FAILED');rr=o.development_publish(root,'RESULT.json',{'status':'COMPLETE_V3_OBSERVED_TRAIN_CONTINUATION'if error is None else'FAILED_CONSUMED','plan':binding,'claim':claim,'report':report,'error':error,'bookends':book,'bookend_evidence':evidence,'elapsed_seconds':time.monotonic()-started,'actual_candidate':'FINAL648','test_data_used':False,'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'});tr=o.development_publish(root,'TERMINAL.json',{'status':'COMPLETE'if error is None else'FAILED_CONSUMED','exit_code':0 if error is None else 1,'plan':binding,'claim':claim,'result':rr,'remaining_invocations':0,'relaunch_allowed':False})
        for sig,handler in handlers.items():signal.signal(sig,handler)
    if error:raise RuntimeError(error)
    return {'result':rr,'terminal':tr}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--v3-observed-train-continuation-plan-sha256',required=True);parser.add_argument('--preflight',action='store_true');args=parser.parse_args();sha=args.v3_observed_train_continuation_plan_sha256;need(sys.argv[1:]==['--v3-observed-train-continuation-plan-sha256',sha]+(['--preflight']if args.preflight else[]),'OBSERVED_CONTINUATION_EXACT_CLI')
    if args.preflight:
        guard.require_native_v3_observed_train_continuation_execution(plan_sha256=sha);plan,binding,pref,live=inspect(sha,False);import torch;need(not torch.cuda.is_initialized(),'OBSERVED_CONTINUATION_PREFLIGHT_NO_CUDA');print(json.dumps({'status':STATUS_PREFLIGHT,'plan':binding,'policy':pref,'physical_input_identity':plan['physical_input_identity'],'models':0,'payloads_decoded':0,'CUDA_initialized':False,'CLAIM_created':ROOT.exists()}))
    else:print(json.dumps(run(sha),sort_keys=True))

if __name__=='__main__':main()
