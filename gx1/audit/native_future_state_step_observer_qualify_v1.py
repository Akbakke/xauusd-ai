"""Bounded observer-off/reference and observer-on/cold parity. Never a fit."""
from pathlib import Path
import argparse,copy,hashlib,json,signal,subprocess,sys,time
REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
_FIREWALL=install_native_m5_direction_io_firewall()
from gx1.audit import native_provider_hlc_learning_v1 as o
from gx1.contracts import gx1_capped_execution_v1 as guard
core=o.core;train=o.train;need=o.need;read=o.read;ref=o.ref
NAME='NATIVE_V3_ACTUAL_STEP_OBSERVER_QUALIFICATION_20261128_001'
STAGE='native_v3_actual_step_observer_20261128_001'
ANCHOR='25cab1bf29f241275b2624110d736e85fcd3bc0e'
SELF='gx1/audit/native_future_state_step_observer_qualify_v1.py'
OBSERVER='gx1/audit/native_future_state_step_observer_v1.py'
TEST='tests/test_native_future_state_step_observer_v1.py'
CORE='gx1/audit/qualify_native_m5_full_gc_direction_v1.py'
PROTOCOL=o.E/'V3_ACTUAL_STEP_OBSERVER_QUALIFICATION_PROTOCOL_20261128_001.json'
PLANS={lane:REPO/'configs/research'/(NAME+'_'+lane.upper()+'.json') for lane in ('reference','cold')}
ROOTS={lane:Path('/home/andre2/GX1_RUNS')/(NAME+'_'+lane) for lane in PLANS}
SNAPS={lane:o.E/('V3_ACTUAL_STEP_OBSERVER_'+lane.upper()+'_ADMITTED_POLICY_20261128_001.json') for lane in PLANS}
PROFILE='gx1_native_actual_step_observer_qualification_trainer20g_v1'

def original_metadata():
    closed=o.issue_mix_closed('fit');producer=read(closed['plan']);original=producer['metadata']['original'];receipt_ref=closed['value']['report']['final_checkpoint_receipt'];receipt=read(receipt_ref);trajectory=producer['metadata']['trajectory']
    need(receipt['original_plan']==receipt['execution_plan']==closed['plan'] and receipt['producer_lane']=='fit' and receipt['logical_cursor']==receipt['optimizer_updates']==432,'OBSERVER_AUTHENTIC_FINAL432')
    proof=train.require_recovery_checkpoint_proof(receipt['proof']);need(proof['trajectory']==core.state_sha(trajectory) and proof['position']==core.state_sha(train.position(trajectory['schedule'],432)) and proof['scores']==core.state_sha(closed['value']['report']['scores']) and proof['scheduler_state']==core.state_sha(None) and proof['training']==core.state_sha(True),'OBSERVER_ORIGINAL_FULL10_METADATA')
    meta=original['metadata'];binding=read(meta['input_binding_ref']);normalization=read(meta['normalization'])
    need(binding==meta['input_binding'] and binding['input_normalization_sha256']==normalization['contract_sha256'] and binding['feature_provider_contract_sha256']==normalization['feature_provider_contract_sha256'],'OBSERVER_PHYSICAL_FEATURE_NORMALIZATION')
    from gx1.contracts.market_state_ta_integrated_model_v1 import native_future_state_model_contract
    contract=native_future_state_model_contract(binding,architecture=meta['architecture'],train_prior=read(meta['prior']),target_contract=meta['target_contract'],backbone_gradient_policy=None)
    need(contract==original['model_contract']==producer['model_contract'],'OBSERVER_CURRENT_PHYSICAL_MODEL_CONTRACT')
    identity={'input_binding_ref':meta['input_binding_ref'],'normalization_ref':meta['normalization'],'model_contract_sha256':contract['contract_sha256'],'target_contract_sha256':meta['target_contract']['contract_sha256'],'actual_metadata_contract_recomputed':True,'payloads_decoded':0}
    need(identity==producer['physical_input_identity'],'OBSERVER_FRESH_INPUT_IDENTITY')
    return closed,producer,original,receipt_ref,receipt,identity

def technical_trajectory(producer,protocol):
    original=producer['schedule'];takes=protocol['technical_takes'];need(takes==[0,1] and original['logical_batches']==432,'OBSERVER_EXACT_APPEND')
    schedule={k:copy.deepcopy(original[k]) for k in ('rows','batch_rows','batches_per_pass','drop_last','source_rows_sha256','issue_ns_sha256','cohort_takes','logical_step_rows','cohort_take_state_sha256_by_logical_batch','update_flags')}
    schedule.update(schema_version='gx1_v3_actual_step_observer_technical_schedule_v1',logical_batches=434,optimizer_updates=434,skipped_batches=0,original_prefix_schedule_sha256=original['contract_sha256'])
    for key in ('cohort_takes','logical_step_rows','cohort_take_state_sha256_by_logical_batch'):
        schedule[key] += copy.deepcopy([original[key][j] for j in takes])
    schedule['update_flags'] += [True,True]
    schedule['technical_append']={'producer_plan':protocol['producer_plan'],'authentic_final_receipt':protocol['authentic_final_receipt'],'original_take_indices':takes,'scope':'Original432prefix plus exactly2technicaltakes; not a fulladditionalpass.'};schedule=train.seal(schedule)
    trajectory=copy.deepcopy(producer['metadata']['trajectory']);trajectory.pop('contract_sha256');trajectory.update(schema_version='gx1_v3_actual_step_observer_technical_trajectory_v1',schedule=schedule,observer_qualification_protocol=ref(PROTOCOL),original_prefix_trajectory_sha256=producer['metadata']['trajectory']['contract_sha256'])
    return train.seal(trajectory)

def closed_reference():
    root=ROOTS['reference'];tr=ref(root/'TERMINAL.json');terminal=read(tr);rr=terminal['result'];value=read(rr);plan_ref=ref(PLANS['reference']);claim=read(value['claim'])
    need(terminal['plan']==value['plan']==claim['plan']==plan_ref and terminal['status']=='COMPLETE' and terminal['exit_code']==0 and value['status']=='COMPLETE_V3_ACTUAL_STEP_OBSERVER_REFERENCE' and value['error'] is None and all(value['bookends'].values()),'OBSERVER_REFERENCE_CLOSED')
    core.future_cpu_parent_stopped(claim['process_identity'])
    return {'plan':plan_ref,'result':rr,'terminal':tr,'claim':value['claim'],'value':value}

def build_plan(lane,runtime,source_commit=None):
    need(lane in PLANS,'OBSERVER_LANE');pr=ref(PROTOCOL);protocol=read(pr);closed,producer,original,fr,receipt,identity=original_metadata();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    need(source_commit in (None,head) and runtime['source_commit']==head,'OBSERVER_FROZEN_HEAD_RUNTIME');core.future_cuda_require_runtime(runtime);train.runtime_compatible(producer['metadata']['trajectory']['runtime_compatibility'],runtime)
    need(protocol['schema_version']=='gx1_v3_actual_step_observer_qualification_protocol_v1' and protocol['producer_plan']==closed['plan'] and protocol['authentic_final_receipt']==fr and protocol['authentic_final_checkpoint']==receipt['checkpoint'],'OBSERVER_PROTOCOL_AUTHENTIC_LINEAGE')
    old_core={CORE:hashlib.sha256(subprocess.check_output(['git','show',ANCHOR+':'+CORE],cwd=REPO)).hexdigest()};core_bridge=guard.native_v3_actual_step_observer_source_bridge(old_core,producer_commit=ANCHOR)
    old_guards={n:hashlib.sha256(subprocess.check_output(['git','show',ANCHOR+':'+n],cwd=REPO)).hexdigest() for n in ('gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh')}
    guard_bridge=guard.native_v3_actual_step_observer_guard_source_bridge(old_guards,producer_commit=ANCHOR)
    owner_bridge=o.issue_mix_source_bridge();need(all(owner_bridge[k]==v for k,v in producer['owner_source_bridge'].items() if k!='current_sha256'),'OBSERVER_OLD_LEAF_NUMERIC_BRIDGE')
    changed={CORE,*o.GUARDS,o.SELF,o.DEV_TEST,o.DEV_GUARD_TEST};owners={}
    for n,digest in producer['source_owners'].items():
        current=ref(REPO/n)['sha256'];need(n in changed or current==digest,'OBSERVER_UNCHANGED_NUMERIC_OWNER:'+n);owners[n]=current
    owners.update({n:ref(REPO/n)['sha256'] for n in (SELF,OBSERVER,TEST)})
    inputs=copy.deepcopy(producer['inputs']);inputs.update(protocol=pr,design=protocol['design'],producer_plan=closed['plan'],authentic_final_receipt=fr,authentic_final_checkpoint=receipt['checkpoint'],producer_result=closed['result'],producer_terminal=closed['terminal'],producer_claim=closed['claim'],producer_admitted_policy=closed['admitted_policy_snapshot'],producer_stream_binding=protocol['original_stream_binding'])
    parents={}
    if lane=='cold':
        previous=closed_reference();parent=read(previous['plan']);need(parent['source_owners']==owners and parent['source_commit']==head and parent['protocol']==pr,'OBSERVER_FROZEN_REFERENCE_GENERATION');parents={'reference':{k:previous[k] for k in ('plan','result','terminal','claim')}}
        inputs.update(reference_plan=previous['plan'],reference_result=previous['result'],reference_terminal=previous['terminal'],reference_cursor1_receipt=previous['value']['report']['checkpoint_receipts']['433'],reference_cursor2_receipt=previous['value']['report']['checkpoint_receipts']['434'])
        inputs['reference_checkpoint']=read(inputs['reference_cursor1_receipt'])['checkpoint']
    metadata_roles=sorted(k for k,v in inputs.items() if Path(v['path']).suffix not in ('.npz','.npy','.pt','.pth','.parquet','.safetensors','.pkl','.pickle','.joblib'))
    return {'schema_version':'gx1_native_v3_actual_step_observer_qualification_plan_v1','stage':STAGE,'lane':lane,'source_commit':head,'output_root':str(ROOTS[lane]),'source_owners':owners,'core_source_bridge':core_bridge,'guard_source_bridge':guard_bridge,'owner_source_bridge':owner_bridge,'protocol':pr,'design':protocol['design'],'producer_plan':closed['plan'],'authentic_final_receipt':fr,'authentic_final_checkpoint':receipt['checkpoint'],'metadata':{'original':original,'producer':producer,'receipt':receipt,'trajectory':technical_trajectory(producer,protocol),'runtime':copy.deepcopy(runtime)},'inputs':inputs,'metadata_input_roles':metadata_roles,'payload_stat':{k:{'size':Path(v['path']).stat().st_size,'mtime_ns':Path(v['path']).stat().st_mtime_ns} for k,v in inputs.items() if k not in metadata_roles},'parents':parents,'recipe':original['recipe'],'model_contract':original['model_contract'],'backbone_gradient_policy':None,'physical_input_identity':identity,'starting_cursor':432 if lane=='reference' else 433,'ending_cursor':434,'checkpoint_cursors':[433,434] if lane=='reference' else [434],'observer_enabled':lane=='cold','technical_weights_never_initializer':True,'limits':protocol['limits'][lane],'execution':guard._native_v3_development_execution('warm'),'execution_profile':PROFILE}

def inspect(sha,admitted):
    bound=guard._native_v3_actual_step_observer_bound_plan(sha);lane=bound['lane'];binding=bound['plan'];plan=read(binding)
    need(plan==build_plan(lane,plan['metadata']['runtime'],plan['source_commit']),'OBSERVER_EXACT_PLAN');o.recovery_check_inputs(plan);pref=ref(o.POLICY)
    live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['metadata']['runtime'],live)
    if admitted:need(ref(SNAPS[lane])['sha256']==pref['sha256'],'OBSERVER_EXACT_ADMITTED_SNAPSHOT')
    return plan,binding,pref,live

def physical(plan,binding,root,live):
    import torch
    from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
    from gx1.audit.native_future_state_step_observer_v1 import NativeFutureStateStepObserver
    original=plan['metadata']['original'];producer=plan['metadata']['producer'];lane=plan['lane'];context=o.development_nn_context(original);stream=o.issue_mix_actual_stream(context,producer)
    need(stream==read(plan['inputs']['producer_stream_binding']),'OBSERVER_ACTUAL_ORIGINAL_STREAM');sr=o.development_publish(root,'ORIGINAL_STREAM_BINDING.json',stream);counts=dict.fromkeys(plan['limits'],0);counts['feature_cache_payload_loads']=counts['target_archive_loads']=1
    core.future_cuda_seed_and_runtime({'recipe':original['recipe'],'metadata':{'runtime':live}})
    model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=context['input_binding'],architecture=original['metadata']['architecture'],device=torch.device('cuda'),train_prior=context['prior'],target_contract=context['target_contract'],backbone_gradient_policy=None);counts['model_constructions']=1
    initial=original['initial_reference'];need(model.research_objective==original['model_contract'] and core.state_sha(model.state_dict())==initial['model_state_sha256'] and core.state_sha(core.training_rng())==initial['rng_sha256'],'OBSERVER_FACTORY_FUNCTION_RNG_PARITY')
    optimizer=torch.optim.AdamW(model.parameters(),**original['recipe']['optimizer']);counts['optimizer_constructions']=1;need(not optimizer.state,'OBSERVER_FRESH_OPTIMIZER')
    cp_ref=plan['authentic_final_checkpoint'] if lane=='reference' else plan['inputs']['reference_checkpoint'];receipt=plan['metadata']['receipt'] if lane=='reference' else read(plan['inputs']['reference_cursor1_receipt']);need(ref(cp_ref['path'])==cp_ref and receipt['checkpoint']==cp_ref,'OBSERVER_EXACT_CHECKPOINT_LOAD');cp=torch.load(cp_ref['path'],map_location='cpu',weights_only=True);counts['checkpoint_loads']=1
    if lane=='reference':
        o.issue_mix_validate_checkpoint(cp,producer,plan['producer_plan'],432,stream);trajectory=producer['metadata']['trajectory'];original_ref=plan['producer_plan']
    else:
        trajectory=plan['metadata']['trajectory'];original_ref=plan['parents']['reference']['plan'];need(cp['producer']['original_plan']==original_ref and receipt['original_plan']==receipt['execution_plan']==original_ref and receipt['logical_cursor']==433,'OBSERVER_OWN_REFERENCE_C1_PRODUCER')
    train.strict_restore(model,optimizer,cp,trajectory=trajectory,original_plan_ref=original_ref,expected_cursor=plan['starting_cursor'],current_runtime=live,expected_state_proof=receipt['proof']);counts['model_restores']=1;steps=copy.deepcopy(cp['steps']);scores=copy.deepcopy(cp['scores']);del cp
    observer=NativeFutureStateStepObserver() if plan['observer_enabled'] else None;receipts={}
    def counter(key):
        def hook(*_):counts[key]+=1;need(counts[key]<=plan['limits'][key],'OBSERVER_FORWARD_BOUND')
        return hook
    hooks=[model.register_forward_pre_hook(counter('model_forwards')),model.backbone.register_forward_pre_hook(counter('backbone_forwards')),model.readout.register_forward_pre_hook(counter('readout_forwards'))]
    def make(cursor):
        own_reference=binding if lane=='reference' else plan['parents']['reference']['plan']
        return train.make_checkpoint(model,optimizer,trajectory=plan['metadata']['trajectory'],original_plan_ref=own_reference,lane=lane,cursor=cursor,steps=copy.deepcopy(steps),scores=copy.deepcopy(scores),producer_runtime=live)
    try:
        train.release_phase_allocator_cache(model,optimizer,label='OBSERVER_RESTORE_TO_TECHNICAL')
        for index in range(plan['starting_cursor'],434):
            local=index-432;take=o.np.asarray(producer['schedule']['cohort_takes'][local],dtype=o.np.int64);packet=o.adapter.packet(context,take)
            if observer is None:record=train.training_step(model,optimizer,packet,original,counts,index)
            else:
                record=core.future_cuda_step(model,optimizer,packet,original,counts,index,observer=observer);record['skip_update']=False
            record['observer_qualification_binding']={'global_index':index,'original_batch_index':local,'binding':o.issue_mix_step_binding(producer['schedule'],stream,local)};steps.append(record)
            cp=make(index+1)
            if lane=='cold':need(train.checkpoint_proof(cp)==read(plan['inputs']['reference_cursor2_receipt'])['proof'],'OBSERVER_OFF_ON_COLD_C2_ALL10')
            receipts[str(index+1)]=o.development_checkpoint(root,'CURSOR'+str(index+1).zfill(3)+'.pt',cp,counts,binding)
        reports=[] if observer is None else observer.reports;counts['observer_records']=len(reports)
        observations=o.development_publish(root,'OBSERVATIONS.json',{'observer_enabled':plan['observer_enabled'],'reports':reports,'technical_only':True,'new_candidate':False})
        need(counts==plan['limits'] and torch.cuda.max_memory_reserved()<=plan['execution']['cuda_peak_reserved_bytes_max'],'OBSERVER_EXACT_COUNTS_VRAM')
        return {'counts':counts,'checkpoint_receipts':receipts,'observations':observations,'starting_cursor':plan['starting_cursor'],'ending_cursor':434,'full_cursor434_proof':train.checkpoint_proof(make(434)),'original_stream_binding':sr,'observer_off_on_all10_equal':lane=='cold','technical_weights_never_initializer':True,'new_candidate':False,'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'}
    finally:
        for hook in hooks:hook.remove()

def run(sha):
    execution=guard.require_native_v3_actual_step_observer_execution(plan_sha256=sha);plan,binding,pref,live=inspect(sha,True);root=ROOTS[plan['lane']];root.mkdir(exist_ok=False);boots={k:live[k] for k in ('physical_windows_boot_id','wsl_boot_id')};started=time.monotonic();claim=o.development_publish(root,'CLAIM.json',{'plan':binding,'lane':plan['lane'],'policy_at_claim':pref,'admitted_policy_snapshot':ref(SNAPS[plan['lane']]),'execution':execution,'boots':boots,'process_identity':core.future_cpu_process_identity(),'remaining_invocations':0,'relaunch_allowed':False});report=None;error=None
    def stop(*_):raise TimeoutError('OBSERVER_INTERNAL_WALL_OR_SIGNAL')
    handlers={sig:signal.getsignal(sig) for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    for sig in handlers:signal.signal(sig,stop)
    signal.setitimer(signal.ITIMER_REAL,plan['execution']['internal_seconds'])
    try:
        allowed={v['path'] for v in plan['inputs'].values() if Path(v['path']).suffix in ('.npz','.npy','.pt','.pth','.parquet','.safetensors','.pkl','.pickle','.joblib')};allowed.update(str(root/('CURSOR'+str(cursor).zfill(3)+'.pt')) for cursor in plan['checkpoint_cursors']);sys.addaudithook(lambda e,a:o.development_io_audit(allowed,e,a));o.recovery_check_inputs(plan,physical=True);report=physical(plan,binding,root,live);need(core.common.firewall_state()['denied_attempts']==0,'OBSERVER_ZERO_FIREWALL_DENIALS')
    except BaseException as exc:error=type(exc).__name__+':'+str(exc)[:2400]
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);checks={'plan':lambda:ref(PLANS[plan['lane']])==binding,'sources':lambda:all(ref(REPO/n)['sha256']==v for n,v in plan['source_owners'].items()),'inputs':lambda:all(ref(v['path'])==v for v in plan['inputs'].values()) and ref(SNAPS[plan['lane']])['sha256']==pref['sha256'],'policy':lambda:ref(o.POLICY)==pref,'boots':lambda:{'physical_windows_boot_id':core.physical_windows_boot_id(),'wsl_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}==boots,'HEAD':lambda:subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_commit']};book={}
        for key,check in checks.items():
            try:book[key]=bool(check())
            except BaseException:book[key]=False
        error=error or (None if all(book.values()) else 'OBSERVER_BOOKEND_FAILED');rr=o.development_publish(root,'RESULT.json',{'status':'COMPLETE_V3_ACTUAL_STEP_OBSERVER_'+plan['lane'].upper() if error is None else 'FAILED_CONSUMED','plan':binding,'claim':claim,'lane':plan['lane'],'report':report,'error':error,'bookends':book,'elapsed_seconds':time.monotonic()-started,'new_candidate':False,'test_data_used':False,'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'});tr=o.development_publish(root,'TERMINAL.json',{'status':'COMPLETE' if error is None else 'FAILED_CONSUMED','exit_code':0 if error is None else 1,'plan':binding,'claim':claim,'result':rr,'remaining_invocations':0,'relaunch_allowed':False})
        for sig,handler in handlers.items():signal.signal(sig,handler)
    if error:raise RuntimeError(error)
    return {'result':rr,'terminal':tr}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--v3-actual-step-observer-plan-sha256',required=True);parser.add_argument('--preflight',action='store_true');a=parser.parse_args();sha=a.v3_actual_step_observer_plan_sha256
    need(sys.argv[1:]==['--v3-actual-step-observer-plan-sha256',sha]+(['--preflight'] if a.preflight else []),'OBSERVER_EXACT_CLI')
    if a.preflight:
        guard.require_native_v3_actual_step_observer_execution(plan_sha256=sha);plan,binding,pref,live=inspect(sha,False);import torch;need(not torch.cuda.is_initialized(),'OBSERVER_PREFLIGHT_NO_CUDA');print(json.dumps({'status':'PREFLIGHT_V3_ACTUAL_STEP_OBSERVER_METADATA_ONLY','plan':binding,'lane':plan['lane'],'policy':pref,'physical_input_identity':plan['physical_input_identity'],'models':0,'payloads_decoded':0,'CUDA_initialized':torch.cuda.is_initialized(),'CLAIM_created':ROOTS[plan['lane']].exists()}))
    else:print(json.dumps(run(sha),sort_keys=True))

if __name__=='__main__':main()
