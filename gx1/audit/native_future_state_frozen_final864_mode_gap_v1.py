"""Zero-update total-mode diagnostic at one authentic learned FINAL864."""
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
NAME='NATIVE_V3_FROZEN_FINAL864_MODE_GAP_20261128_001';STAGE=NAME.lower();PLAN=REPO/'configs/research'/(NAME+'.json');ROOT=Path('/home/andre2/GX1_RUNS')/NAME
SNAPSHOT=E/'V3_FROZEN_FINAL864_MODE_GAP_ADMITTED_POLICY_20261128_001.json'
DESIGN={'path':str(E/'V3_FROZEN_FINAL864_MODE_GAP_DESIGN_20261128_001.json'),'sha256':'3b49a1de3bb2bc257e157d9fd8de137b57121d94fe30b76c75edac0b7519c097'}
ANCHOR='6ab45cfd46e0391b4f48d746214f0ab4518e7f9b';SELF='gx1/audit/native_future_state_frozen_final864_mode_gap_v1.py';TEST='tests/test_native_future_state_frozen_final864_mode_gap_v1.py';GUARD_TEST='tests/test_native_frozen_final864_mode_gap_guard_v1.py'
PROFILE='gx1_native_frozen_final864_mode_gap_trainer20g_v1';STATUS_PREFLIGHT='PREFLIGHT_V3_FROZEN_FINAL864_MODE_GAP_METADATA_ONLY'


def gate(binding,admitted=False):
    return {'status':'ADMITTED_EXACT_V3_FROZEN_FINAL864_MODE_GAP'if admitted else'PREPARED_CLOSED_NOT_ADMITTED','plan':binding,'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':False,'test_authorized':False,'spending_authorized':False}


def require_gate(value,binding,admitted=False):
    expected=gate(binding,admitted);need(type(value)is dict and type(value.get('remaining_invocations'))is int and all(type(value.get(k))is bool for k in('execution_authorized','relaunch_allowed','training_authorized','test_authorized','spending_authorized'))and all(value.get(k)==v for k,v in expected.items()),'FROZEN_FINAL864_TYPED_GATE');return value


def merge_inputs(original,extra):
    need(all(k not in original or original[k]==v for k,v in extra.items()),'FROZEN_FINAL864_INPUT_ROLE_COLLISION');return{**copy.deepcopy(original),**copy.deepcopy(extra)}


def snapshot(binding,pref):
    sr=ref(SNAPSHOT);policy=read(sr);need(pref=={'path':str(o.POLICY),'sha256':sr['sha256']}and all(policy.get(k)is False for k in o.DEV_FLAGS),'FROZEN_FINAL864_SNAPSHOT_ALL6_CLOSED');require_gate(policy.get(STAGE),binding,True);need(policy[STAGE].get('design')==DESIGN,'FROZEN_FINAL864_SNAPSHOT_DESIGN');return sr


def validate_final_receipt(parent,receipt,design,report):
    need(receipt['checkpoint']==design['authentic_FINAL864_checkpoint']and report['checkpoint_receipts']['864']==design['authentic_FINAL864_receipt']and receipt['original_plan']==receipt['execution_plan']==design['producer_plan']and receipt['producer_lane']=='continuation'and type(receipt['logical_cursor'])is type(receipt['optimizer_updates'])is int and receipt['logical_cursor']==receipt['optimizer_updates']==864 and receipt['trajectory_sha256']==parent['metadata']['trajectory']['contract_sha256'],'FROZEN_FINAL864_OWN_GENUINE_LINEAGE')
    proof=train.require_recovery_checkpoint_proof(receipt['proof']);need(proof==report['full_FINAL864_proof']and proof['trajectory']==core.state_sha(parent['metadata']['trajectory'])and proof['position']==core.state_sha(train.position(parent['schedule'],864))and proof['training']==core.state_sha(True)and proof['scheduler_state']==core.state_sha(None),'FROZEN_FINAL864_FULL10_METADATA');return proof


def validate_score_contracts(numeric,wrapper,design,cohort,factory,spec):
    need(wrapper['actual_phase']=='FINAL864'and wrapper['global_cursor']==864 and wrapper['numeric_owner_score']==design['saved_eval_numeric_owner_score']and wrapper['numeric_owner_logits']==numeric['logits']==design['saved_eval_logits']and numeric['phase']=='PASS2'and numeric['rows']==20727 and numeric['batches']==81,'FROZEN_FINAL864_AUTHENTIC_EVAL_OUTPUTS')
    review=numeric['review'];need(review['COHORT_score_prior_sha256']==cohort['contract_sha256']and review['factory_prior_sha256']==factory['contract_sha256']and review['spec_sha256']==spec['contract_sha256']and spec['cohort_prior']==cohort and spec['whole_TRAIN_spec']['prior_contract_sha256']==factory['contract_sha256'],'FROZEN_FINAL864_INTERNAL_PRIOR_SPEC_CONTRACTS')


def final_metadata():
    design=read(DESIGN);parent=read(design['producer_plan']);result=read(design['producer_result']);terminal=read(design['producer_terminal']);claim=read(result['claim']);acceptance=read(design['producer_acceptance']);receipt=read(design['authentic_FINAL864_receipt'])
    need(result['plan']==terminal['plan']==claim['plan']==design['producer_plan']and terminal['result']==design['producer_result']and terminal['claim']==result['claim']and result['status']=='COMPLETE_V3_OBSERVED_TRAIN_CONTINUATION_002'and terminal['status']==acceptance['status']=='COMPLETE'and terminal['exit_code']==0 and result['error']is None and result['report']['ending_cursor']==864 and set(result['bookends'])=={'plan','sources','inputs','policy','boots','HEAD'}and all(v is True for v in result['bookends'].values()),'FROZEN_FINAL864_CLOSED_PRODUCER')
    need(acceptance['plan']==design['producer_plan']and acceptance['result']==design['producer_result']and acceptance['terminal']==design['producer_terminal']and acceptance['claim']==result['claim']and acceptance['all_bookends_pass']is acceptance['hardware_guard_accepted']is acceptance['actual_process_gone']is True and terminal['remaining_invocations']==claim['remaining_invocations']==0 and terminal['relaunch_allowed']is claim['relaunch_allowed']is False,'FROZEN_FINAL864_ACCEPTED_CONSUMED_PARENT');core.future_cpu_parent_stopped(claim['process_identity']);validate_final_receipt(parent,receipt,design,result['report'])
    closed,original_producer,original,fr,original_receipt,identity=q.original_metadata();need(closed['plan']==parent['original_producer_plan']==design['original_producer_plan']and parent['model_contract']==original['model_contract']and parent['recipe']==original['recipe']and parent['physical_input_identity']==identity,'FROZEN_FINAL864_FRESH_PHYSICAL_IDENTITY')
    numeric=read(design['saved_eval_numeric_owner_score']);wrapper=read(design['saved_eval_score_wrapper']);cohort=read(design['frozen_COHORT_prior']);factory=read(design['frozen_factory_prior']);spec=read(design['frozen_numeric_owner_spec']);validate_score_contracts(numeric,wrapper,design,cohort,factory,spec)
    need(original['inputs']['cohort_prior']==design['frozen_COHORT_prior']and original['metadata']['prior']==design['frozen_factory_prior']and original['inputs']['cohort_spec']==design['frozen_numeric_owner_spec'],'FROZEN_FINAL864_ORIGINAL_PRIOR_ROLES')
    scores=copy.deepcopy(closed['value']['report']['scores']);scores['FINAL648']=parent['inputs']['observed_continuation_002_BASELINE648_score'];scores['FINAL864']=design['saved_eval_score_wrapper'];need(receipt['proof']['scores']==core.state_sha(scores),'FROZEN_FINAL864_AUTHENTIC_FULL_SCORE_HISTORY');core.future_cuda_runtime_matches(parent['metadata']['runtime'],receipt['producer_runtime'])
    need(design['actual_global_cursor']==864 and design['passes']==list(o.TRAIN_EVAL_GAP_PASSES)and design['scope']['model_forwards']==405 and design['scope']['optimizer_updates']==0,'FROZEN_FINAL864_EXACT_PASS_PROTOCOL')
    refs={'frozen_final864_mode_gap_design':DESIGN,'frozen_final864_mode_gap_parent_plan':design['producer_plan'],'frozen_final864_mode_gap_original_plan':design['original_producer_plan'],'frozen_final864_mode_gap_parent_claim':result['claim'],'frozen_final864_mode_gap_parent_policy_snapshot':claim['admitted_policy_snapshot'],**{'frozen_final864_mode_gap_'+k:design[k]for k in('authentic_FINAL864_receipt','authentic_FINAL864_checkpoint','saved_eval_logits','saved_eval_numeric_owner_score','saved_eval_score_wrapper','producer_result','producer_terminal','producer_acceptance','basis','endpoint_assessment','frozen_COHORT_prior','frozen_factory_prior','frozen_numeric_owner_spec')}}
    return original,receipt,parent['metadata']['trajectory'],parent,scores,refs,identity


def operation_limits():return copy.deepcopy(read(DESIGN)['limits'])


def build_plan(runtime,source_commit=None):
    original,receipt,trajectory,parent,scores,extra,identity=final_metadata();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();need(source_commit in(None,head)and runtime['source_commit']==head,'FROZEN_FINAL864_FROZEN_SOURCE_RUNTIME');core.future_cuda_require_runtime(runtime);train.runtime_compatible(trajectory['runtime_compatibility'],runtime);owners={}
    for name,digest in parent['source_owners'].items():
        current=ref(REPO/name)['sha256'];need(name in o.GUARDS or current==digest,'FROZEN_FINAL864_UNCHANGED_NUMERIC_SOURCE:'+name);owners[name]=current
    owners.update({name:ref(REPO/name)['sha256']for name in(SELF,TEST,GUARD_TEST)});old={name:hashlib.sha256(subprocess.check_output(['git','show',ANCHOR+':'+name],cwd=REPO)).hexdigest()for name in o.GUARDS};bridge=guard.native_v3_frozen_final864_mode_gap_guard_source_bridge(old,producer_commit=ANCHOR);design=read(DESIGN);inputs=merge_inputs(parent['inputs'],extra);roles=sorted(k for k,v in inputs.items()if Path(v['path']).suffix not in('.npz','.npy','.pt','.pth','.parquet','.safetensors','.pkl','.pickle','.joblib'))
    return {'schema_version':'gx1_native_v3_frozen_final864_mode_gap_plan_v1','stage':STAGE,'source_commit':head,'output_root':str(ROOT),'source_owners':owners,'guard_source_bridge':bridge,'design':DESIGN,'producer_plan':design['producer_plan'],'original_producer_plan':design['original_producer_plan'],'authentic_final_receipt':design['authentic_FINAL864_receipt'],'authentic_final_checkpoint':receipt['checkpoint'],'metadata':{'parent_plan':design['producer_plan'],'original_plan':design['original_producer_plan'],'receipt':receipt,'trajectory':trajectory,'scores':scores,'runtime':copy.deepcopy(runtime)},'recipe':parent['recipe'],'model_contract':parent['model_contract'],'physical_input_identity':identity,'backbone_gradient_policy':None,'actual_global_cursor':864,'numeric_owner_phase':'PASS2','passes':design['passes'],'limits':design['limits'],'inputs':inputs,'metadata_input_roles':roles,'payload_stat':{k:{'size':Path(v['path']).stat().st_size,'mtime_ns':Path(v['path']).stat().st_mtime_ns}for k,v in inputs.items()if k not in roles},'execution':guard._native_v3_development_execution('warm'),'execution_profile':PROFILE,'scope':design['scope']}


def inspect(sha,admitted=False):
    bound=guard._native_v3_frozen_final864_mode_gap_bound_plan(sha);binding=bound['plan'];plan=read(binding);need(plan==build_plan(plan['metadata']['runtime'],plan['source_commit']),'FROZEN_FINAL864_EXACT_PLAN');o.recovery_check_inputs(plan);pref=ref(o.POLICY);policy=read(pref);need(all(policy.get(k)is False for k in o.DEV_FLAGS),'FROZEN_FINAL864_ALL6_GLOBAL_FLAGS_CLOSED');require_gate(policy.get(STAGE),binding,admitted);live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['metadata']['runtime'],live)
    if admitted:snapshot(binding,pref)
    return plan,binding,pref,live


def physical(plan,binding,root,live):
    import torch
    from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
    original_producer=read(plan['metadata']['original_plan']);original=original_producer['metadata']['original'];receipt=plan['metadata']['receipt'];trajectory=plan['metadata']['trajectory'];counts=dict.fromkeys(plan['limits'],0);context=o.development_nn_context(original);stream=o.issue_mix_actual_stream(context,original_producer);need(stream==read(plan['inputs']['observed_continuation_producer_stream_binding']),'FROZEN_FINAL864_ORIGINAL_SOURCE_ISSUE_TRUTH_STREAM');counts['feature_cache_payload_loads']=counts['target_archive_loads']=1;core.future_cuda_seed_and_runtime({'recipe':original['recipe'],'metadata':{'runtime':live}})
    model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=context['input_binding'],architecture=original['metadata']['architecture'],device=torch.device('cuda'),train_prior=context['prior'],target_contract=context['target_contract'],backbone_gradient_policy=None);counts['model_constructions']=1;need(model.research_objective==original['model_contract']and core.state_sha(model.state_dict())==original['initial_reference']['model_state_sha256']and core.state_sha(core.training_rng())==original['initial_reference']['rng_sha256'],'FROZEN_FINAL864_FACTORY_FUNCTION_RNG');optimizer=torch.optim.AdamW(model.parameters(),**original['recipe']['optimizer']);counts['optimizer_constructions']=1;need(not optimizer.state,'FROZEN_FINAL864_FRESH_OPTIMIZER');cp_ref=plan['authentic_final_checkpoint'];need(cp_ref==receipt['checkpoint']and ref(cp_ref['path'])==cp_ref,'FROZEN_FINAL864_ONLY_GENUINE_CHECKPOINT_LOAD');cp=torch.load(cp_ref['path'],map_location='cpu',weights_only=True);counts['checkpoint_loads']=1
    need(cp['producer']=={'original_plan':plan['producer_plan'],'lane':'continuation','runtime':receipt['producer_runtime']}and cp['scores']==plan['metadata']['scores'],'FROZEN_FINAL864_ACTUAL_PRODUCER_SCORE_HISTORY');train.strict_restore(model,optimizer,cp,trajectory=trajectory,original_plan_ref=plan['producer_plan'],expected_cursor=864,current_runtime=live,expected_state_proof=receipt['proof']);counts['model_restores']=1;steps=copy.deepcopy(cp['steps']);scores=copy.deepcopy(cp['scores']);del cp
    def proof():return train.checkpoint_proof(train.make_checkpoint(model,optimizer,trajectory=trajectory,original_plan_ref=plan['producer_plan'],lane='diagnostic',cursor=864,steps=steps,scores=scores,producer_runtime=live))
    saved=o.train_eval_gap_saved_outputs(plan['inputs']['frozen_final864_mode_gap_saved_eval_logits'],context['truth']['issue_ns'],counts)
    def counter(key):
        def hook(*_):counts[key]+=1;need(counts[key]<=plan['limits'][key],'FROZEN_FINAL864_FORWARD_BOUND')
        return hook
    hooks=[model.register_forward_pre_hook(counter('model_forwards')),model.backbone.register_forward_pre_hook(counter('backbone_forwards')),model.readout.register_forward_pre_hook(counter('readout_forwards'))]
    try:
        report=o.train_eval_gap_passes(model,optimizer,context,root,counts,steps,scores,proof,receipt['proof'],saved);need(counts==plan['limits']and proof()==receipt['proof']and torch.cuda.max_memory_reserved()<=plan['execution']['cuda_peak_reserved_bytes_max'],'FROZEN_FINAL864_EXACT_COUNTS_FULL10_VRAM');state=ref(root/'STATE_INVARIANTS.json');return {**report,'counts':counts,'actual_global_cursor':864,'final_checkpoint_receipt':plan['authentic_final_receipt'],'state_invariants':state,'final_state_proof':receipt['proof'],'all10_semantic_proofs_unchanged':True,'optimizer_updates':0,'new_checkpoint_writes':0,'diagnostic_only':True,'new_predictor':False,'no_new_trained_candidate':True,'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN'}
    finally:
        for hook in hooks:hook.remove()


def payload_allowlist(plan,root):return o.train_eval_gap_payload_allowlist(plan,root)


def bookends(plan,binding,pref,boots):
    observed=recovery.observe_boots();checks={'plan':lambda:ref(PLAN)==binding,'sources':lambda:all(ref(REPO/n)['sha256']==v for n,v in plan['source_owners'].items()),'inputs':lambda:all(ref(v['path'])==v for v in plan['inputs'].values())and snapshot(binding,pref)['sha256']==pref['sha256'],'policy':lambda:ref(o.POLICY)==pref,'boots':lambda:recovery.boot_matches(boots,observed),'HEAD':lambda:subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_commit']};book={};errors={}
    for name,check in checks.items():
        try:book[name]=bool(check())
        except BaseException as exc:book[name]=False;errors[name]={'type':type(exc).__name__,'message':str(exc)[:2400]}
    return book,{'claimed_boots':boots,'boot_observation':observed,'check_errors':errors}


def run(sha):
    execution=guard.require_native_v3_frozen_final864_mode_gap_execution(plan_sha256=sha);plan,binding,pref,live=inspect(sha,True);root=ROOT;root.mkdir(exist_ok=False);boots={k:live[k]for k in('physical_windows_boot_id','wsl_boot_id')};started=time.monotonic();claim=o.development_publish(root,'CLAIM.json',{'plan':binding,'policy_at_claim':pref,'admitted_policy_snapshot':snapshot(binding,pref),'execution':execution,'boots':boots,'process_identity':core.future_cpu_process_identity(),'remaining_invocations':0,'relaunch_allowed':False});report=None;error=None
    def stop(*_):raise TimeoutError('FROZEN_FINAL864_INTERNAL_WALL_OR_SIGNAL')
    handlers={sig:signal.getsignal(sig)for sig in(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    for sig in handlers:signal.signal(sig,stop)
    signal.setitimer(signal.ITIMER_REAL,plan['execution']['internal_seconds'])
    try:
        allowed=payload_allowlist(plan,root);sys.addaudithook(lambda e,a:o.development_io_audit(allowed,e,a));o.recovery_check_inputs(plan,physical=True);report=physical(plan,binding,root,live);need(core.common.firewall_state()['denied_attempts']==0,'FROZEN_FINAL864_ZERO_FIREWALL_DENIALS')
    except BaseException as exc:error=type(exc).__name__+':'+str(exc)[:2400]
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);book,evidence=bookends(plan,binding,pref,boots);error=error or(None if all(book.values())else'FROZEN_FINAL864_BOOKEND_FAILED');extra={'state_invariants':ref(root/'STATE_INVARIANTS.json')}if(root/'STATE_INVARIANTS.json').is_file()else{};rr=o.development_publish(root,'RESULT.json',{'status':'COMPLETE_V3_FROZEN_FINAL864_MODE_GAP'if error is None else'FAILED_CONSUMED','plan':binding,'claim':claim,'report':report,'error':error,'bookends':book,'bookend_evidence':evidence,'elapsed_seconds':time.monotonic()-started,'actual_global_cursor':864,'diagnostic_only':True,'new_predictor':False,'optimizer_updates':0,'new_checkpoint_writes':0,'test_data_used':False,'later_generalization':'NOT_MEASURED','economic_edge':'UNKNOWN',**extra});tr=o.development_publish(root,'TERMINAL.json',{'status':'COMPLETE'if error is None else'FAILED_CONSUMED','exit_code':0 if error is None else 1,'plan':binding,'claim':claim,'result':rr,'remaining_invocations':0,'relaunch_allowed':False})
        for sig,handler in handlers.items():signal.signal(sig,handler)
    if error:raise RuntimeError(error)
    return {'result':rr,'terminal':tr}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--v3-frozen-final864-mode-gap-plan-sha256',required=True);parser.add_argument('--preflight',action='store_true');args=parser.parse_args();sha=args.v3_frozen_final864_mode_gap_plan_sha256;need(sys.argv[1:]==['--v3-frozen-final864-mode-gap-plan-sha256',sha]+(['--preflight']if args.preflight else[]),'FROZEN_FINAL864_EXACT_CLI')
    if args.preflight:
        guard.require_native_v3_frozen_final864_mode_gap_execution(plan_sha256=sha);plan,binding,pref,live=inspect(sha,False);import torch;need(not torch.cuda.is_initialized(),'FROZEN_FINAL864_PREFLIGHT_NO_CUDA');print(json.dumps({'status':STATUS_PREFLIGHT,'plan':binding,'policy':pref,'physical_input_identity':plan['physical_input_identity'],'models':0,'payloads_decoded':0,'CUDA_initialized':False,'CLAIM_created':ROOT.exists()}))
    else:print(json.dumps(run(sha),sort_keys=True))

if __name__=='__main__':main()
