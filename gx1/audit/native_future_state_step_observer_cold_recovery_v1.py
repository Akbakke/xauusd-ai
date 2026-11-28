"""One new cold replay of original REFERENCE C433; never relaunch old COLD."""
from pathlib import Path
import argparse,copy,hashlib,json,signal,subprocess,sys,time
REPO=Path(__file__).resolve().parents[2]
if str(REPO)not in sys.path:sys.path.insert(0,str(REPO))
from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
_FIREWALL=install_native_m5_direction_io_firewall()
from gx1.audit import native_future_state_step_observer_qualify_v1 as q
from gx1.contracts import gx1_capped_execution_v1 as guard
o=q.o;core=q.core;train=q.train;need=q.need;read=q.read;ref=q.ref
NAME='NATIVE_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_20261128_001'
STAGE=NAME.lower();PLAN=REPO/'configs/research'/(NAME+'.json');ROOT=Path('/home/andre2/GX1_RUNS')/NAME
SNAPSHOT=o.E/'V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_ADMITTED_POLICY_20261128_001.json'
ANCHOR='23b166541d86a02b91205c8ce3de7b63392c6be6'
DESIGN={'path':str(o.E/'V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_DESIGN_20261128_001.json'),'sha256':'a4e157293c9339b2afbc4ef2bbf34718c832d932bead198398fc0a636b32b173'}
SELF='gx1/audit/native_future_state_step_observer_cold_recovery_v1.py'
TEST='tests/test_native_future_state_step_observer_cold_recovery_v1.py'
PROFILE='gx1_native_actual_step_observer_cold_recovery_trainer20g_v1'


def gate(binding,admitted=False):
    return {'status':'ADMITTED_EXACT_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY'if admitted else'PREPARED_CLOSED_NOT_ADMITTED','plan':binding,'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':admitted,'test_authorized':False,'spending_authorized':False}


def require_gate(value,binding,admitted=False):
    expected=gate(binding,admitted);need(type(value)is dict and type(value.get('remaining_invocations'))is int and all(type(value.get(k))is bool for k in('execution_authorized','relaunch_allowed','training_authorized','test_authorized','spending_authorized'))and all(value.get(k)==v for k,v in expected.items()),'OBSERVER_COLD_RECOVERY_TYPED_GATE');return value


def snapshot(binding,pref):
    sr=ref(SNAPSHOT);policy=read(sr);need(pref=={'path':str(o.POLICY),'sha256':sr['sha256']}and all(policy.get(k)is(k in o.DEV_FLAGS[:2])for k in o.DEV_FLAGS),'OBSERVER_COLD_RECOVERY_EXACT_SNAPSHOT_FLAGS');require_gate(policy.get(STAGE),binding,True);need(policy[STAGE].get('design')==DESIGN,'OBSERVER_COLD_RECOVERY_SNAPSHOT_DESIGN');return sr


def original_metadata():
    design=read(DESIGN);old=read(design['original_failed_COLD']['plan']);reference=q.closed_reference();need({k:reference[k]for k in('plan','result','terminal','claim')}==design['reference'],'OBSERVER_COLD_RECOVERY_ORIGINAL_REFERENCE')
    reference_plan=read(reference['plan']);need(old['lane']=='cold'and old['observer_enabled']is True and old['starting_cursor']==433 and old['ending_cursor']==434 and old['checkpoint_cursors']==[434]and old['parents']['reference']['plan']==reference['plan']and old['metadata']['trajectory']==reference_plan['metadata']['trajectory']and old['protocol']==reference_plan['protocol'],'OBSERVER_COLD_RECOVERY_EXACT_OLD_TRAJECTORY')
    refs={'recovery_design':DESIGN,'original_cold_plan':design['original_failed_COLD']['plan'],**{'original_failed_cold_'+k:v for k,v in design['original_failed_COLD'].items()if k!='plan'},**{'reused_reference_'+k:v for k,v in design['reference'].items()},'original_reference_backup':design['reference_backup'],'original_source_qualification':design['source_qualification'],'post_failure_host_observation':design['post_failure_host'],'original_actual_review':design['basis']}
    failed=read(refs['original_failed_cold_result']);terminal=read(refs['original_failed_cold_terminal']);acceptance=read(refs['original_failed_cold_acceptance']);need(failed['plan']==terminal['plan']==design['original_failed_COLD']['plan']and terminal['result']==refs['original_failed_cold_result']and failed['status']==terminal['status']=='FAILED_CONSUMED'and terminal['exit_code']==1 and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False and acceptance['status']=='FAILED_CONSUMED'and acceptance['result']==refs['original_failed_cold_result']and acceptance['terminal']==refs['original_failed_cold_terminal'],'OBSERVER_COLD_RECOVERY_PRESERVED_FAILED_COLD')
    core.future_cpu_parent_stopped(read(failed['claim'])['process_identity'])
    for cursor,key in((433,'reference_C433_receipt'),(434,'reference_C434_receipt')):
        binding=design[key];receipt=read(binding);need(binding==old['inputs']['reference_cursor'+str(cursor-432)+'_receipt']==reference['value']['report']['checkpoint_receipts'][str(cursor)]and receipt['original_plan']==receipt['execution_plan']==reference['plan']and receipt['producer_lane']=='reference'and type(receipt['logical_cursor'])is int and receipt['logical_cursor']==receipt['optimizer_updates']==cursor and receipt['trajectory_sha256']==old['metadata']['trajectory']['contract_sha256'],'OBSERVER_COLD_RECOVERY_AUTHENTIC_REFERENCE_CURSOR')
        proof=train.require_recovery_checkpoint_proof(receipt['proof']);need(proof['trajectory']==core.state_sha(old['metadata']['trajectory'])and proof['position']==core.state_sha(train.position(old['metadata']['trajectory']['schedule'],cursor))and proof['training']==core.state_sha(True)and proof['scheduler_state']==core.state_sha(None),'OBSERVER_COLD_RECOVERY_REFERENCE_FULL10_METADATA')
    need(read(design['reference_C433_receipt'])['checkpoint']==old['inputs']['reference_checkpoint'],'OBSERVER_COLD_RECOVERY_ONLY_REFERENCE_C433')
    closed,producer,original,fr,receipt,identity=q.original_metadata();need(original==old['metadata']['original']and producer==old['metadata']['producer']and receipt==old['metadata']['receipt']and fr==old['authentic_final_receipt']and closed['plan']==old['producer_plan']and identity==old['physical_input_identity'],'OBSERVER_COLD_RECOVERY_FRESH_ORIGINAL_METADATA')
    for name,digest in design['source_math_owners'].items():need(ref(REPO/name)['sha256']==digest,'OBSERVER_COLD_RECOVERY_FROZEN_MATH:'+name)
    backup=read(design['reference_backup']);manifest=read(backup['manifest']);need(backup['separate_mac_disk']is backup['F_FULLFSYNC']is backup['independent_member_readback']is True and backup['manifest']['sha256']==backup['manifest_sha256'],'OBSERVER_COLD_RECOVERY_AUTHENTIC_MAC_BACKUP');members={(v['path'],v['sha256'])for v in manifest['files']}
    for binding in(design['reference_C433_receipt'],design['reference_C434_receipt'],old['inputs']['reference_checkpoint'],read(design['reference_C434_receipt'])['checkpoint']):need((binding['path'],binding['sha256'])in members,'OBSERVER_COLD_RECOVERY_REFERENCE_BACKUP_COVERAGE')
    refs['original_reference_backup_manifest']=backup['manifest']
    for binding in refs.values():read(binding)
    return old,reference_plan,refs,identity


def execution_plan(old,runtime,head,owners,inputs,bridge):
    """Change execution generation only; preserve the genuine technical trajectory."""
    plan=copy.deepcopy(old);plan.update(schema_version='gx1_native_v3_actual_step_observer_cold_recovery_plan_v1',stage=STAGE,source_commit=head,output_root=str(ROOT),source_owners=owners,guard_source_bridge=bridge,original_cold_plan=inputs['original_cold_plan'],reference_plan=old['parents']['reference']['plan'],recovery_design=DESIGN,inputs=inputs,execution_profile=PROFILE)
    plan['metadata']['original_cold']=copy.deepcopy(old);plan['metadata']['runtime']=copy.deepcopy(runtime);roles=sorted(k for k,v in inputs.items()if Path(v['path']).suffix not in('.npz','.npy','.pt','.pth','.parquet','.safetensors','.pkl','.pickle','.joblib'));plan['metadata_input_roles']=roles
    plan['payload_stat']={k:{'size':Path(v['path']).stat().st_size,'mtime_ns':Path(v['path']).stat().st_mtime_ns}for k,v in inputs.items()if k not in roles};return plan


def build_plan(runtime,source_commit=None):
    old,reference,extra,identity=original_metadata();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();need(source_commit in(None,head)and runtime['source_commit']==head,'OBSERVER_COLD_RECOVERY_CURRENT_SOURCE_RUNTIME');core.future_cuda_require_runtime(runtime);train.runtime_compatible(old['metadata']['trajectory']['runtime_compatibility'],runtime)
    allowed={*o.GUARDS,o.DEV_GUARD_TEST};owners={}
    for name,digest in old['source_owners'].items():
        current=ref(REPO/name)['sha256'];need(name in allowed or current==digest,'OBSERVER_COLD_RECOVERY_ONLY_GUARD_GENERATION:'+name);owners[name]=current
    owners.update({name:ref(REPO/name)['sha256']for name in(SELF,TEST)});old_guards={name:hashlib.sha256(subprocess.check_output(['git','show',ANCHOR+':'+name],cwd=REPO)).hexdigest()for name in o.GUARDS};bridge=guard.native_v3_actual_step_observer_cold_recovery_guard_source_bridge(old_guards,producer_commit=ANCHOR)
    inputs={**copy.deepcopy(old['inputs']),**extra};plan=execution_plan(old,runtime,head,owners,inputs,bridge);need(plan['physical_input_identity']==identity and plan['execution']==guard._native_v3_development_execution('warm'),'OBSERVER_COLD_RECOVERY_UNCHANGED_IDENTITY_CAPS');return plan


def inspect(sha,admitted=False):
    bound=guard._native_v3_actual_step_observer_cold_recovery_bound_plan(sha);binding=bound['plan'];plan=read(binding);need(plan==build_plan(plan['metadata']['runtime'],plan['source_commit']),'OBSERVER_COLD_RECOVERY_EXACT_PLAN');o.recovery_check_inputs(plan);pref=ref(o.POLICY);require_gate(read(pref).get(STAGE),binding,admitted)
    live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['metadata']['runtime'],live)
    if admitted:snapshot(binding,pref)
    return plan,binding,pref,live


def observe_boots():
    observed={};errors={};queries={'physical_windows_boot_id':core.physical_windows_boot_id,'wsl_boot_id':lambda:Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    for name,query in queries.items():
        try:observed[name]=query()
        except BaseException as exc:errors[name]={'type':type(exc).__name__,'message':str(exc)[:2400]}
    return {'observed':observed,'query_errors':errors}


def boot_matches(claimed,observation):
    return not observation['query_errors']and observation['observed']==claimed


def bookends(plan,binding,pref,boots):
    observed=observe_boots();checks={'plan':lambda:ref(PLAN)==binding,'sources':lambda:all(ref(REPO/n)['sha256']==v for n,v in plan['source_owners'].items()),'inputs':lambda:all(ref(v['path'])==v for v in plan['inputs'].values())and ref(SNAPSHOT)['sha256']==pref['sha256'],'policy':lambda:ref(o.POLICY)==pref,'boots':lambda:boot_matches(boots,observed),'HEAD':lambda:subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_commit']};result={};errors={}
    for name,check in checks.items():
        try:result[name]=bool(check())
        except BaseException as exc:result[name]=False;errors[name]={'type':type(exc).__name__,'message':str(exc)[:2400]}
    return result,{'claimed_boots':boots,'boot_observation':observed,'check_errors':errors}


def run(sha):
    execution=guard.require_native_v3_actual_step_observer_cold_recovery_execution(plan_sha256=sha);plan,binding,pref,live=inspect(sha,True);root=ROOT;root.mkdir(exist_ok=False);boots={k:live[k]for k in('physical_windows_boot_id','wsl_boot_id')};started=time.monotonic();claim=o.development_publish(root,'CLAIM.json',{'plan':binding,'lane':'cold','policy_at_claim':pref,'admitted_policy_snapshot':snapshot(binding,pref),'execution':execution,'boots':boots,'process_identity':core.future_cpu_process_identity(),'remaining_invocations':0,'relaunch_allowed':False});report=None;error=None
    def stop(*_):raise TimeoutError('OBSERVER_COLD_RECOVERY_INTERNAL_WALL_OR_SIGNAL')
    handlers={sig:signal.getsignal(sig)for sig in(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    for sig in handlers:signal.signal(sig,stop)
    signal.setitimer(signal.ITIMER_REAL,plan['execution']['internal_seconds'])
    try:
        allowed={v['path']for v in plan['inputs'].values()if Path(v['path']).suffix in('.npz','.npy','.pt','.pth','.parquet','.safetensors','.pkl','.pickle','.joblib')};allowed.add(str(root/'CURSOR434.pt'));sys.addaudithook(lambda e,a:o.development_io_audit(allowed,e,a));o.recovery_check_inputs(plan,physical=True);report=q.physical(plan,binding,root,live);need(core.common.firewall_state()['denied_attempts']==0,'OBSERVER_COLD_RECOVERY_ZERO_FIREWALL_DENIALS')
    except BaseException as exc:error=type(exc).__name__+':'+str(exc)[:2400]
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);book,evidence=bookends(plan,binding,pref,boots);error=error or(None if all(book.values())else'OBSERVER_COLD_RECOVERY_BOOKEND_FAILED');rr=o.development_publish(root,'RESULT.json',{'status':'COMPLETE_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY'if error is None else'FAILED_CONSUMED','plan':binding,'claim':claim,'lane':'cold','report':report,'error':error,'bookends':book,'bookend_evidence':evidence,'elapsed_seconds':time.monotonic()-started,'original_failed_cold_plan':plan['original_cold_plan'],'new_candidate':False,'test_data_used':False,'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'});tr=o.development_publish(root,'TERMINAL.json',{'status':'COMPLETE'if error is None else'FAILED_CONSUMED','exit_code':0 if error is None else 1,'plan':binding,'claim':claim,'result':rr,'remaining_invocations':0,'relaunch_allowed':False})
        for sig,handler in handlers.items():signal.signal(sig,handler)
    if error:raise RuntimeError(error)
    return {'result':rr,'terminal':tr}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--v3-actual-step-observer-cold-recovery-plan-sha256',required=True);parser.add_argument('--preflight',action='store_true');args=parser.parse_args();sha=args.v3_actual_step_observer_cold_recovery_plan_sha256;need(sys.argv[1:]==['--v3-actual-step-observer-cold-recovery-plan-sha256',sha]+(['--preflight']if args.preflight else[]),'OBSERVER_COLD_RECOVERY_EXACT_CLI')
    if args.preflight:
        guard.require_native_v3_actual_step_observer_cold_recovery_execution(plan_sha256=sha);plan,binding,pref,live=inspect(sha,False);import torch;need(not torch.cuda.is_initialized(),'OBSERVER_COLD_RECOVERY_PREFLIGHT_NO_CUDA');print(json.dumps({'status':'PREFLIGHT_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_METADATA_ONLY','plan':binding,'policy':pref,'physical_input_identity':plan['physical_input_identity'],'models':0,'payloads_decoded':0,'CUDA_initialized':False,'CLAIM_created':ROOT.exists()}))
    else:print(json.dumps(run(sha),sort_keys=True))

if __name__=='__main__':main()
