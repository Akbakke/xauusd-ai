"""Small hash-bound V2 AdamW prefix and genuine new-process next-step replay."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
if __name__=='__main__':
 from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
 install_native_m5_direction_io_firewall()
from gx1.audit import native_future_state_training_v1 as train
from gx1.audit import native_future_state_provider_hlc_baseline_v1 as adapter
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
core=train.core;REPO=train.REPO;POLICY=train.POLICY;E=Path('/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001')
NAME='NATIVE_PROVIDER_HLC_COLD_REPLAY_20261123_001';STAGE=NAME.lower();PLAN=REPO/'configs/research'/(NAME+'.json');ROOTS={lane:Path('/home/andre2/GX1_RUNS')/(NAME+'_'+lane)for lane in ('warm','cold')}
SELF='gx1/audit/native_provider_hlc_replay_v1.py';TEST='tests/test_native_provider_hlc_replay_v1.py';GUARD_TEST='tests/test_native_provider_hlc_replay_guard_v1.py';ANCHOR='27fc84b9111d198ad0a6f6570d7e8efa7d0f1d49'
BASE_ACCEPTANCE={'path':str(E/'PROVIDER_HLC_INITIAL002_ACCEPTANCE001.json'),'sha256':'af144facfc45d51e52e892a6a3cd77a75081b645f312064c242e66015e468222'}
BASE_RESULT={'path':'/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_WHOLE_TRAIN_INITIAL_20261123_002/RESULT.json','sha256':'a6ac22899121e99f55cc8af0c864e89f9054691d6857f57ce6e25acf9e2cce59'}
ROWS=354550;BATCH=96;PREFIX_BATCHES=3;PREFIX_ROWS=288
FLAGS=('training_enabled','native_training_enabled','native_launch_authorized','full_epoch_training_allowed','full_val_allowed')
GUARDS=('gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh')
EXECUTION={'class':'trainer','device':'cuda:0','memory':'20G','swap':'512M','numerical_threads':8,'cpu_affinity':list(range(19)),'tasks_max':128,'outer_seconds':1800,'internal_seconds':1770,'kill_after_seconds':10,'cuda_peak_reserved_bytes_max':12884901888,'GPU_core_stop_C':75,'GPU_memory_stop_C':80,'GPU_power_limit_W':250,'GPU_draw_stop_W':250,'CPU_stop_C':80,'CPU_package_stop_W':125,'host_min_available_bytes':8589934592,'host_min_commit_headroom_bytes':8589934592,'watchdog_interval_seconds':1}
LIMITS={
 'warm':{'model_constructions':1,'optimizer_constructions':1,'feature_cache_payload_loads':1,'target_archive_loads':1,'model_forwards':9,'backbone_forwards':9,'readout_forwards':9,'optimizer_steps':3,'backward_calls':5,'autograd_calls':3,'checkpoint_writes':2,'checkpoint_readbacks':2,'checkpoint_receipt_writes':2,'checkpoint_loads':0,'model_restores':0},
 'cold':{'model_constructions':1,'optimizer_constructions':1,'feature_cache_payload_loads':1,'target_archive_loads':1,'model_forwards':2,'backbone_forwards':2,'readout_forwards':2,'optimizer_steps':1,'backward_calls':1,'autograd_calls':0,'checkpoint_writes':1,'checkpoint_readbacks':1,'checkpoint_receipt_writes':1,'checkpoint_loads':1,'model_restores':1}}
def need(ok,why):
 if not bool(ok):raise ValueError('PROVIDER_HLC_REPLAY_'+why)
def ref(path):
 p=Path(path).resolve();h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return {'path':str(p),'sha256':h.hexdigest()}
def read(binding):need(ref(binding['path'])==binding,'REFERENCE');return json.loads(Path(binding['path']).read_text())
def pub(root,name,obj):
 b=obj if isinstance(obj,bytes)else(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();p=root/name;_publish_bytes(p,b);return ref(p)
def validate_flags(policy,admitted):
 wanted={k:False for k in FLAGS}
 if admitted:wanted.update(training_enabled=True,native_training_enabled=True)
 need(all(policy.get(k)is value for k,value in wanted.items()),'EXACT_TECHNICAL_TRAINING_FLAGS')
def gate(binding,admitted):return {'status':'ADMITTED_EXACT_TECHNICAL_COLD_REPLAY'if admitted else'PREPARED_NOT_ADMITTED','plan':binding,'relaunch_allowed':False,'main_fit_authorized':False,'full_epoch_training_allowed':False,'test_authorized':False,'lanes':{k:{'execution_authorized':admitted,'remaining_invocations':int(admitted)}for k in ROOTS}}
def require_initial_metadata(acceptance,result,base,receipt):
 need(acceptance['full_initialbaseline_qualified']is True and acceptance['result']==BASE_RESULT and result['status']=='COMPLETE_NATIVE_PROVIDER_HLC_WHOLE_TRAIN_INITIAL'and result['source_policy_inputs_boot_unchanged']is True,'QUALIFIED_INITIAL')
 r=result['report'];need(r['whole_TRAIN_initialbaseline_qualified']is True and r['initial']['rows']==ROWS and r['initial']['initial_prior_exact']is True and r['counts']['optimizer_steps']==0,'WHOLE_INITIAL_SCOPE')
 need(receipt['plan']==result['plan']and receipt['model_state_sha256']==r['initial_state_sha256']and receipt['model_contract_sha256']==base['model_contract']['contract_sha256']and receipt['target_contract_sha256']==base['metadata']['target_contract']['contract_sha256']and receipt['initial_only']is True and receipt['optimizer_updates']==0,'INITIAL_CONTRACT')
 return True

def build_plan(runtime,source_commit=None):
 from gx1.contracts.gx1_capped_execution_v1 import native_provider_hlc_replay_source_bridge
 from gx1.contracts.market_state_ta_integrated_model_v1 import native_future_state_model_contract
 a=read(BASE_ACCEPTANCE);r=read(BASE_RESULT);base=read(r['plan']);initial=read(a['initial_receipt']);require_initial_metadata(a,r,base,initial)
 core.future_cuda_require_runtime(runtime);need(all(runtime[k]==base['metadata']['runtime'][k]for k in ('gpu','library_versions')),'INITIAL_RUNTIME_COMPATIBILITY')
 main=read(base['inputs']['original_MAIN_plan']);old=main['recipe'];need(old['train_batch_rows']==BATCH and old['seed']==20260911 and old['gradient_clip_norm']==1.,'EXISTING_RECIPE')
 recipe={k:copy.deepcopy(old[k])for k in ('seed','optimizer','gradient_clip_norm','scheduler')};recipe.update(train_batch_rows=BATCH,prefix_batches=PREFIX_BATCHES,prefix_rows=PREFIX_ROWS,warm_updates=3,cold_updates=1,cold_restore_cursor=2,cold_replay_logical_step=3,shuffle=False,drop_last=False)
 meta=copy.deepcopy(base['metadata']);meta['runtime']=runtime;prior=read(meta['prior']);model_contract=native_future_state_model_contract(meta['input_binding'],architecture=meta['architecture'],train_prior=prior,target_contract=meta['target_contract'],backbone_gradient_policy='M5_primary');need(model_contract==base['model_contract'],'SAME_FORWARD_FUNCTION')
 schedule={'logical_batches':3,'batches_per_pass':3,'update_flags':[True,True,True],'batch_rows':BATCH,'rows':PREFIX_ROWS,'source_clock_only_selector':'first288_positions_of_unchanged354550TRAIN_axis','whole_issue_ns_sha256':base['metadata']['TRAIN_issue_ns_sha256'],'whole_source_rows_sha256':read(meta['normalization'])['fit_population']['selected_source_rows_sha256'],'batch_source_rows_sha256':main['schedule']['batch_source_rows_sha256'][:3],'batch_issue_ns_sha256':main['schedule']['batch_issue_ns_sha256'][:3],'target_selection':False}
 inputs={**base['inputs'],'baseline_acceptance':BASE_ACCEPTANCE,'baseline_result':BASE_RESULT,'baseline_plan':r['plan'],'baseline_initial_receipt':a['initial_receipt']}
 owners=set(base['source_owners'])|{SELF,TEST,GUARD_TEST};need(all(ref(REPO/n)['sha256']==value for n,value in base['source_owners'].items()if n not in GUARDS),'UNCHANGED_GENUINE_NUMERIC_OWNERS')
 hashes={n:hashlib.sha256(subprocess.check_output(['git','show',ANCHOR+':'+n],cwd=REPO)).hexdigest()for n in GUARDS}
 return {'schema_version':'gx1_native_provider_hlc_cold_replay_plan_v1','stage':STAGE,'source_commit':source_commit or runtime['source_commit'],'source_owners':{n:ref(REPO/n)['sha256']for n in sorted(owners)},'source_bridge':native_provider_hlc_replay_source_bridge(hashes,producer_commit=ANCHOR),'metadata':meta,'model_contract':model_contract,'inputs':inputs,'initial_reference':initial,'recipe':recipe,'schedule':schedule,'backbone_gradient_policy':'M5_primary','execution':EXECUTION,'limits':LIMITS,'output_roots':{k:str(p)for k,p in ROOTS.items()},'authority':'TECHNICAL_PREFIX_ONLY_NO_MAIN_FIT_EPOCH_SELECTION_OR_PROMOTION','qualification_scope':'3logical warm updates and onlysame3rdstepcoldreplay fromCP2;populatednonzero backbone/readout AdamWmoments;noVAL/TEST/refit/priorfit/newbaseline/oldcheckpointresume.'}
def trajectory(plan):
 return train.seal({'schema_version':'gx1_native_provider_hlc_replay_trajectory_v1','inputs':plan['inputs'],'model_contract':plan['model_contract'],'schedule':plan['schedule'],'recipe':plan['recipe'],'target_contract':plan['metadata']['target_contract'],'prior_truth_sha256':plan['metadata']['truth_sha256'],'numeric_source_owners':plan['source_owners'],'runtime_compatibility':{k:plan['metadata']['runtime'][k]for k in ('gpu','library_versions')}})
def check_sources_inputs(plan):
 need(Path.cwd().resolve()==REPO and subprocess.check_output(['git','branch','--show-current'],text=True,cwd=REPO).strip()=='work/gx1-current'and subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=REPO).strip()==plan['source_commit'],'FROZEN_CWD_BRANCH_HEAD')
 for n,d in plan['source_owners'].items():need(ref(REPO/n)['sha256']==d,'SOURCE_OWNER')
 for b in plan['inputs'].values():need(ref(b['path'])==b,'INPUT_BYTES')
def warm_chain(binding,policy_ref,plan):
 root=ROOTS['warm'];t=read(ref(root/'TERMINAL.json'));need(t['status']=='COMPLETE'and t['exit_code']==0 and t['plan']==binding and t['result']['path']==str(root/'RESULT.json')and t['claim']['path']==str(root/'CLAIM.json'),'WARM_TERMINAL')
 claim=read(t['claim']);need(claim['policy_at_claim']==policy_ref and claim['plan']==binding and claim['lane']=='warm','SAME_FROZEN_POLICY');r=read(t['result']);need(r['status']=='COMPLETE_TECHNICAL_WARM_PREFIX'and r['plan']==binding and r['claim']==t['claim']and r['source_inputs_policy_boot_unchanged']is True and r['process_identity']==claim['process_identity'],'WARM_RESULT');core.future_cpu_parent_stopped(r['process_identity'])
 report=r['report'];need(report['counts']==LIMITS['warm']and report['trajectory']==trajectory(plan),'WARM_BUDGET_TRAJECTORY')
 need(report['cursor2_receipt']['path']==str(root/'CURSOR002.json')and report['cursor3_proof']['path']==str(root/'CURSOR003_PROOF.json'),'FIXED_REPLAY_RECEIPT_PATHS');receipt=read(report['cursor2_receipt']);need(receipt['original_plan']==receipt['execution_plan']==binding and receipt['producer_lane']=='warm'and receipt['logical_cursor']==receipt['optimizer_updates']==2 and receipt['trajectory_sha256']==trajectory(plan)['contract_sha256']and receipt['checkpoint']['path']==str(root/'CURSOR002.pt'),'CURSOR2_RECEIPT');need(ref(receipt['checkpoint']['path'])==receipt['checkpoint'],'CURSOR2_BYTES');reference=read(report['cursor3_proof']);
 for proof,cursor in ((receipt['proof'],2),(reference['proof'],3)):
  train.require_recovery_checkpoint_proof(proof);need(proof['trajectory']==core.state_sha(trajectory(plan))and proof['position']==core.state_sha(train.position(plan['schedule'],cursor)),'REPLAY_PROOF_TRAJECTORY_CURSOR')
 need(set(reference['nonzero_moments'])=={'backbone','readout'}and all(type(n)is int and n>0 for n in reference['nonzero_moments'].values()),'REFERENCE_NONZERO_MOMENTS');return r,receipt,reference

def inspect(sha,lane,admitted):
 need(lane in ROOTS,'LANE');binding={'path':str(PLAN),'sha256':sha};p=read(binding);need(p==build_plan(p['metadata']['runtime'],p['source_commit']),'EXACT_PLAN');check_sources_inputs(p);pref=ref(POLICY);policy=read(pref);validate_flags(policy,admitted);need(policy[STAGE]==gate(binding,admitted),'ONE_USE_GATE');need(not ROOTS[lane].exists()and not ROOTS[lane].is_symlink(),'FRESH_OUTPUT');live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(p['metadata']['runtime'],live)
 if admitted and lane=='cold':warm_chain(binding,pref,p)
 return p,binding,pref,live

def prefix_batches(context,plan):
 need(len(context['source_rows'])==ROWS and core.digest(context['truth']['issue_ns'])==plan['schedule']['whole_issue_ns_sha256']and core.digest(context['source_rows'])==plan['schedule']['whole_source_rows_sha256'],'FULL_CLOCK')
 batches=[np.arange(j*BATCH,(j+1)*BATCH,dtype=np.int64)for j in range(3)]
 for j,take in enumerate(batches):need(core.digest(context['source_rows'][take])==plan['schedule']['batch_source_rows_sha256'][j]and core.digest(context['truth']['issue_ns'][take])==plan['schedule']['batch_issue_ns_sha256'][j],'CLOCK_PREFIX_BINDING')
 return batches

def nonzero_moments(model,optimizer):
 result={}
 for prefix in ('backbone.','readout.'):
  active=[]
  for n,p in model.named_parameters():
   state=optimizer.state.get(p)
   if n.startswith(prefix)and state and bool(state['exp_avg'].count_nonzero())and bool(state['exp_avg_sq'].count_nonzero()):active.append(n)
  result[prefix[:-1]]=len(active)
 need(result['backbone']>0 and result['readout']>0,'NONZERO_BODY_AND_READOUT_MOMENTS');return result

def score(model,packet,plan):
 import torch
 from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
 before=core.state_sha(model.state_dict());rng=core.training_rng();mode=model.training;model.eval()
 try:
  with torch.no_grad():
   output,_,_=core.future_cuda_forward(model,packet['features'],equal_tf=True);cpu={k:v.detach().cpu()for k,v in output.items()};s=objective.native_future_state_score_sums(model_output=cpu,truth=packet['truth'],fold_upper_ns=train.UPPER,target_contract=plan['metadata']['target_contract'])
  return {'scores':s,'output_sha256':{k:core.digest(v.numpy())for k,v in cpu.items()},'scope':'fixedthirdprefixbatch_technical_score_notgeneralization'}
 finally:
  model.train(mode);core.restore_training_rng(rng);need(core.state_sha(model.state_dict())==before,'SCORE_MUTATION')

def physical(plan,binding,pref,live,lane):
 import torch
 from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
 root=ROOTS[lane];counts=dict.fromkeys(LIMITS[lane],0);context=adapter.load_context(plan['metadata']);counts['feature_cache_payload_loads']=counts['target_archive_loads']=1;batches=prefix_batches(context,plan)
 core.future_cuda_seed_and_runtime({'recipe':plan['recipe'],'metadata':{'runtime':live}})
 model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=context['input_binding'],architecture=plan['metadata']['architecture'],device=torch.device('cuda'),train_prior=context['prior'],target_contract=context['target_contract'],backbone_gradient_policy='M5_primary');counts['model_constructions']=1
 initial=plan['initial_reference'];need(model.research_objective==plan['model_contract']and core.state_sha(model.state_dict())==initial['model_state_sha256']and core.state_sha(core.training_rng())==initial['rng_sha256'],'FRESH_QUALIFIED_INITIAL_STATE_RNG')
 optimizer=torch.optim.AdamW(model.parameters(),**plan['recipe']['optimizer']);counts['optimizer_constructions']=1;need(not optimizer.state,'EMPTY_FRESH_OPTIMIZER');traj=trajectory(plan);steps=[];scores={};receipts={};extra={}
 def count(key):
  def hook(*args):counts[key]+=1;need(counts[key]<=LIMITS[lane][key],'FORWARD_BUDGET')
  return hook
 hooks=[model.register_forward_pre_hook(count('model_forwards')),model.backbone.register_forward_pre_hook(count('backbone_forwards')),model.readout.register_forward_pre_hook(count('readout_forwards'))]
 try:
  if lane=='warm':
   first=adapter.packet(context,batches[0]);extra['finite_check_gradient_parity']=core.future_cuda_parity_pair(model,first,plan,counts,equal_tf=False);extra['equal_TF_eval_parity']=core.future_cuda_parity_pair(model,first,plan,counts,equal_tf=True)
   for j in range(3):
    packet=first if j==0 else adapter.packet(context,batches[j]);steps.append(train.training_step(model,optimizer,packet,plan,counts,j));need(steps[-1]['skip_update']is False,'PREFIX_SUPPORT_NO_RESELECTION')
    if j==0:extra['post_first_head_data_gradient']=core.future_cuda_gradient_probe(model,packet,plan,counts)
    if j==1:extra['nonzero_moments_at_cursor2']=nonzero_moments(model,optimizer)
    if j<2:
     cp=train.make_checkpoint(model,optimizer,trajectory=traj,original_plan_ref=binding,lane=lane,cursor=j+1,steps=list(steps),scores=copy.deepcopy(scores),producer_runtime=live);receipts[j+1]=train.publish_checkpoint_receipt(root/('CURSOR%03d.pt'%(j+1)),cp,counts,execution_ref=binding)
    if j==1:need(steps[-1]['backbone_data_gradient_l2']>0,'REAL_BODY_DATA_GRADIENT')
   scores['CURSOR3']=score(model,packet,plan);cp=train.make_checkpoint(model,optimizer,trajectory=traj,original_plan_ref=binding,lane=lane,cursor=3,steps=list(steps),scores=scores,producer_runtime=live);proof=pub(root,'CURSOR003_PROOF.json',{'proof':train.checkpoint_proof(cp),'nonzero_moments':nonzero_moments(model,optimizer)});report={'cursor2_receipt':receipts[2],'cursor1_receipt':receipts[1],'cursor3_proof':proof,**extra}
  else:
   original,receipt,reference=warm_chain(binding,pref,plan);cp=torch.load(receipt['checkpoint']['path'],map_location='cpu',weights_only=True);counts['checkpoint_loads']=1
   train.strict_restore(model,optimizer,cp,trajectory=traj,original_plan_ref=binding,expected_cursor=2,current_runtime=live,expected_state_proof=receipt['proof']);counts['model_restores']=1;moments=nonzero_moments(model,optimizer);steps=copy.deepcopy(cp['steps']);scores=copy.deepcopy(cp['scores']);packet=adapter.packet(context,batches[2]);steps.append(train.training_step(model,optimizer,packet,plan,counts,2));scores['CURSOR3']=score(model,packet,plan)
   replay=train.make_checkpoint(model,optimizer,trajectory=traj,original_plan_ref=binding,lane=lane,cursor=3,steps=steps,scores=scores,producer_runtime=live);need(train.checkpoint_proof(replay)==reference['proof'],'BITEXACT_COLD_NEXT_STEP_FULL_STATE');receipt3=train.publish_checkpoint_receipt(root/'REPLAYED_CURSOR003.pt',replay,counts,execution_ref=binding);report={'cold_replay_bit_exact':True,'source_checkpoint_receipt':original['report']['cursor2_receipt'],'replayed_checkpoint_receipt':receipt3,'reference_cursor3_proof':original['report']['cursor3_proof'],'restored_nonzero_moments':moments,'full_cursor3_proof':train.checkpoint_proof(replay)}
  need(counts==LIMITS[lane],'EXACT_COUNTS');need(torch.cuda.max_memory_reserved()<=EXECUTION['cuda_peak_reserved_bytes_max'],'VRAM_CAP');return {'counts':counts,'trajectory':traj,'fresh_initial_state_sha256':initial['model_state_sha256'],'logical_updates':3,'physical_updates_this_lane':counts['optimizer_steps'],'prefix_rows':PREFIX_ROWS,'batch_rows':BATCH,'cold_replay_qualified':lane=='cold','cuda_peak_reserved_bytes':int(torch.cuda.max_memory_reserved()),'weights_authority':'TECHNICAL_ONLY_NEVER_MAIN_INITIALIZER_OR_SELECTION','learning_generalization_edge':'UNKNOWN',**report}
 finally:
  for h in hooks:h.remove()

def run(sha,lane):
 from gx1.contracts.gx1_capped_execution_v1 import require_native_provider_hlc_replay_execution
 began=time.monotonic()
 def stop(*_):raise TimeoutError('PROVIDER_HLC_REPLAY_SIGNAL_OR_WALL')
 for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
 signal.setitimer(signal.ITIMER_REAL,1770);guard=require_native_provider_hlc_replay_execution(plan_sha256=sha,lane=lane);plan,binding,pref,live=inspect(sha,lane,True);root=ROOTS[lane];need(guard['plan']==binding and guard['output_root']==str(root),'GUARD_PLAN_ROOT');root.mkdir(exist_ok=False);identity=core.future_cpu_process_identity();claim=pub(root,'CLAIM.json',{'plan':binding,'policy_at_claim':pref,'lane':lane,'runtime':live,'process_identity':identity,'remaining_invocations':0,'main_fit_authorized':False})
 try:
  allowed={plan['inputs'][k]['path']for k in ('feature_archive','target_arrays')};allowed|={str(ROOTS['warm']/('CURSOR%03d.pt'%j))for j in (1,2)};allowed.add(str(ROOTS['cold']/'REPLAYED_CURSOR003.pt'))
  def audit(event,args):
   if event in ('socket.connect','socket.getaddrinfo','urllib.Request'):raise RuntimeError('REPLAY_NETWORK_FORBIDDEN')
   if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
    p=os.fsdecode(args[0])
    if p.endswith(('.npz','.pt','.pth','.parquet','.safetensors','.pkl'))and str(Path(p).resolve())not in allowed:raise RuntimeError('REPLAY_PAYLOAD_PATH_FORBIDDEN')
  sys.addaudithook(audit);report=physical(plan,binding,pref,live,lane);check_sources_inputs(plan);need(ref(PLAN)==binding and ref(POLICY)==pref,'SAME_FROZEN_PLAN_POLICY');need(core.physical_windows_boot_id()==live['physical_windows_boot_id']and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==live['wsl_boot_id'],'BOTH_BOOT_BOOKEND');need(core.common.firewall_state()['denied_attempts']==0,'M1_FIREWALL')
  result=pub(root,'RESULT.json',{'status':'COMPLETE_TECHNICAL_WARM_PREFIX'if lane=='warm'else'COMPLETE_TECHNICAL_COLD_REPLAY','plan':binding,'claim':claim,'process_identity':identity,'report':report,'source_inputs_policy_boot_unchanged':True,'elapsed_seconds':time.monotonic()-began,'test_data_used':False});terminal=pub(root,'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False});return {'result':result,'terminal':terminal}
 except BaseException as e:pub(root,'TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'plan':binding,'claim':claim,'error':type(e).__name__+':'+str(e)[:1800],'remaining_invocations':0,'relaunch_allowed':False});raise
 finally:signal.setitimer(signal.ITIMER_REAL,0)
# Explicit None candidate cold replay; original M5 replay remains consumed.
ALL_NAME='NATIVE_PROVIDER_HLC_ALL_TF_COLD_REPLAY_20261124_001';ALL_STAGE=ALL_NAME.lower();ALL_PLAN=REPO/'configs/research'/(ALL_NAME+'.json');ALL_ROOTS={lane:Path('/home/andre2/GX1_RUNS')/(ALL_NAME+'_'+lane)for lane in ('warm','cold')}
ALL_SCOPE={'path':str(E/'PROVIDER_HLC_ALL_TF_FRESH_INITIAL_REPLAY_PREPARATION_SCOPE001.json'),'sha256':'aa4a01793dfd02344eefc491a83639caa3a00cf626f7e3827c1d8c7730816a21'}
ALL_PROFILE={'path':str(E/'PROVIDER_HLC_ALL_TF_FRESH_INITIAL_REPLAY_PROFILE_DESIGN001.json'),'sha256':'9c00fad884062c672a1dd66949eacfa3b9b86689b4ad136837d8d38780290632'}
ALL_PARENT='f33da18dd6a0e3d59e4876fa26f8e62ec5ab97b8';ALL_EXECUTION=copy.deepcopy(EXECUTION);ALL_LIMITS=copy.deepcopy(LIMITS);ALL_CONTRACT_SHA='6d6ac22819e54c7b12aebcf4e1afe27b990d8a4d90d16500e352c7353d99cca0'
ALL_INITIAL_NAME='NATIVE_PROVIDER_HLC_ALL_TF_WHOLE_TRAIN_INITIAL_20261124_001';ALL_INITIAL_PLAN=REPO/'configs/research'/(ALL_INITIAL_NAME+'.json');ALL_INITIAL_ROOT=Path('/home/andre2/GX1_RUNS')/ALL_INITIAL_NAME
all_tf_check_sources_inputs=check_sources_inputs
all_tf_validate_flags=validate_flags

def all_tf_require_initial_metadata(acceptance,result,base,receipt):
 from gx1.audit import native_future_state_provider_hlc_initial_v1 as initial_owner
 need(acceptance['full_initialbaseline_qualified']is True and acceptance['result']['path']==str(ALL_INITIAL_ROOT/'RESULT.json')and result['plan']['path']==str(ALL_INITIAL_PLAN)and result['status']=='COMPLETE_NATIVE_PROVIDER_HLC_ALL_TF_WHOLE_TRAIN_INITIAL'and result['source_policy_inputs_boot_unchanged']is True,'ALL_TF_QUALIFIED_INITIAL')
 r=result['report'];need(r['whole_TRAIN_initialbaseline_qualified']is True and r['initial']['rows']==ROWS and r['initial']['batch_count']==1385 and r['initial']['initial_prior_exact']is True and r['counts']==initial_owner.ALL_LIMITS,'ALL_TF_WHOLE_INITIAL_SCOPE')
 need(base['stage']==ALL_INITIAL_NAME.lower()and base['output_root']==str(ALL_INITIAL_ROOT)and base['recipe']['backbone_gradient_policy']is None and base['recipe']['seed']==20260911 and base['recipe']['no_optimization']is True and base['model_contract']['contract_sha256']==ALL_CONTRACT_SHA and 'backbone_gradient_policy'not in base['model_contract']and r['backbone_gradient_policy']is None,'ALL_TF_OWN_INITIAL_CONTRACT')
 need(acceptance['initial_receipt']['path']==str(ALL_INITIAL_ROOT/'INITIAL.json')and r['initial_checkpoint']==acceptance['initial_receipt']and receipt['plan']==result['plan']and receipt['checkpoint']['path']==str(ALL_INITIAL_ROOT/'INITIAL.pt')and receipt['model_state_sha256']==r['initial_state_sha256']and receipt['model_contract_sha256']==ALL_CONTRACT_SHA and receipt['target_contract_sha256']==base['metadata']['target_contract']['contract_sha256']and receipt['initial_only']is True and receipt['optimizer_updates']==0,'ALL_TF_INITIAL_RECEIPT')
 return True

def build_all_tf_plan(runtime,initial_acceptance_ref,source_commit=None):
 from gx1.contracts.gx1_capped_execution_v1 import native_provider_hlc_all_tf_replay_source_bridge
 from gx1.contracts.market_state_ta_integrated_model_v1 import native_future_state_model_contract
 scope=read(ALL_SCOPE);profile=read(ALL_PROFILE);need(scope['candidate_policy']['explicit_backbone_gradient_policy']is None and scope['candidate_policy']['model_contract_sha256_at_unchanged_input_target_architecture']==ALL_CONTRACT_SHA and profile['scope']==ALL_SCOPE and profile['candidate_backbone_gradient_policy']is None and profile['replay']['new_plan_id']==ALL_NAME and profile['replay']['execution']==ALL_EXECUTION and profile['replay']['limits']==ALL_LIMITS,'ALL_TF_SCOPE_PROFILE')
 need(Path(initial_acceptance_ref['path']).is_absolute(),'ALL_TF_ABSOLUTE_INITIAL_ACCEPTANCE');a=read(initial_acceptance_ref);r=read(a['result']);base=read(r['plan']);initial=read(a['initial_receipt']);all_tf_require_initial_metadata(a,r,base,initial)
 core.future_cuda_require_runtime(runtime);need(all(runtime[k]==base['metadata']['runtime'][k]for k in ('gpu','library_versions')),'INITIAL_RUNTIME_COMPATIBILITY')
 main=read(base['inputs']['original_MAIN_plan']);old=main['recipe'];need(old['train_batch_rows']==BATCH and old['seed']==20260911 and old['gradient_clip_norm']==1.,'EXISTING_RECIPE')
 recipe={k:copy.deepcopy(old[k])for k in ('seed','optimizer','gradient_clip_norm','scheduler')};recipe.update(train_batch_rows=BATCH,prefix_batches=PREFIX_BATCHES,prefix_rows=PREFIX_ROWS,warm_updates=3,cold_updates=1,cold_restore_cursor=2,cold_replay_logical_step=3,shuffle=False,drop_last=False)
 need(recipe==profile['replay']['recipe'],'ALL_TF_RECIPE_PROFILE');recipe['backbone_gradient_policy']=None
 meta=copy.deepcopy(base['metadata']);meta['runtime']=runtime;prior=read(meta['prior']);model_contract=native_future_state_model_contract(meta['input_binding'],architecture=meta['architecture'],train_prior=prior,target_contract=meta['target_contract'],backbone_gradient_policy=None);need(model_contract==base['model_contract']and model_contract['contract_sha256']==ALL_CONTRACT_SHA and 'backbone_gradient_policy'not in model_contract,'ALL_TF_SAME_FORWARD_FUNCTION')
 schedule={'logical_batches':3,'batches_per_pass':3,'update_flags':[True,True,True],'batch_rows':BATCH,'rows':PREFIX_ROWS,'source_clock_only_selector':'first288_positions_of_unchanged354550TRAIN_axis','whole_issue_ns_sha256':base['metadata']['TRAIN_issue_ns_sha256'],'whole_source_rows_sha256':read(meta['normalization'])['fit_population']['selected_source_rows_sha256'],'batch_source_rows_sha256':main['schedule']['batch_source_rows_sha256'][:3],'batch_issue_ns_sha256':main['schedule']['batch_issue_ns_sha256'][:3],'target_selection':False}
 inputs={**base['inputs'],'baseline_acceptance':initial_acceptance_ref,'baseline_result':a['result'],'baseline_plan':r['plan'],'baseline_initial_receipt':a['initial_receipt'],'scope':ALL_SCOPE,'profile_design':ALL_PROFILE}
 owners=set(base['source_owners'])|{SELF,TEST,GUARD_TEST};need(all(ref(REPO/n)['sha256']==value for n,value in base['source_owners'].items()if n not in GUARDS),'UNCHANGED_GENUINE_NUMERIC_OWNERS')
 hashes={n:hashlib.sha256(subprocess.check_output(['git','show',ALL_PARENT+':'+n],cwd=REPO)).hexdigest()for n in GUARDS}
 return {'schema_version':'gx1_native_provider_hlc_all_tf_cold_replay_plan_v1','stage':ALL_STAGE,'source_commit':source_commit or runtime['source_commit'],'source_owners':{n:ref(REPO/n)['sha256']for n in sorted(owners)},'source_bridge':native_provider_hlc_all_tf_replay_source_bridge(hashes,producer_commit=ALL_PARENT),'metadata':meta,'model_contract':model_contract,'inputs':inputs,'initial_reference':initial,'recipe':recipe,'schedule':schedule,'backbone_gradient_policy':None,'execution':ALL_EXECUTION,'limits':ALL_LIMITS,'output_roots':{k:str(p)for k,p in ALL_ROOTS.items()},'authority':'TECHNICAL_PREFIX_ONLY_NO_MAIN_FIT_EPOCH_SELECTION_OR_PROMOTION','qualification_scope':'FreshsamearmNone3warmupdatesandcoldstep3fromownC2;all10proofs/nonzeroAdamWmoments;neverfitinitializeroroldM5checkpointload.'}

def all_tf_gate(binding,admitted):return {'status':'ADMITTED_EXACT_ALL_TF_TECHNICAL_COLD_REPLAY'if admitted else'PREPARED_NOT_ADMITTED','plan':binding,'relaunch_allowed':False,'main_fit_authorized':False,'full_epoch_training_allowed':False,'test_authorized':False,'lanes':{k:{'execution_authorized':admitted,'remaining_invocations':int(admitted)}for k in ALL_ROOTS}}
def all_tf_warm_chain(binding,policy_ref,plan):
 root=ALL_ROOTS['warm'];t=read(ref(root/'TERMINAL.json'));need(t['status']=='COMPLETE'and t['exit_code']==0 and t['plan']==binding and t['result']['path']==str(root/'RESULT.json')and t['claim']['path']==str(root/'CLAIM.json'),'WARM_TERMINAL')
 claim=read(t['claim']);need(claim['policy_at_claim']==policy_ref and claim['plan']==binding and claim['lane']=='warm','SAME_FROZEN_POLICY');r=read(t['result']);need(r['status']=='COMPLETE_ALL_TF_TECHNICAL_WARM_PREFIX'and r['plan']==binding and r['claim']==t['claim']and r['source_inputs_policy_boot_unchanged']is True and r['process_identity']==claim['process_identity'],'WARM_RESULT');core.future_cpu_parent_stopped(r['process_identity'])
 report=r['report'];need(report['counts']==ALL_LIMITS['warm']and report['trajectory']==trajectory(plan),'WARM_BUDGET_TRAJECTORY')
 need(report['cursor2_receipt']['path']==str(root/'CURSOR002.json')and report['cursor3_proof']['path']==str(root/'CURSOR003_PROOF.json'),'FIXED_REPLAY_RECEIPT_PATHS');receipt=read(report['cursor2_receipt']);need(receipt['original_plan']==receipt['execution_plan']==binding and receipt['producer_lane']=='warm'and receipt['logical_cursor']==receipt['optimizer_updates']==2 and receipt['trajectory_sha256']==trajectory(plan)['contract_sha256']and receipt['checkpoint']['path']==str(root/'CURSOR002.pt'),'CURSOR2_RECEIPT');need(ref(receipt['checkpoint']['path'])==receipt['checkpoint'],'CURSOR2_BYTES');reference=read(report['cursor3_proof']);
 for proof,cursor in ((receipt['proof'],2),(reference['proof'],3)):
  train.require_recovery_checkpoint_proof(proof);need(proof['trajectory']==core.state_sha(trajectory(plan))and proof['position']==core.state_sha(train.position(plan['schedule'],cursor)),'REPLAY_PROOF_TRAJECTORY_CURSOR')
 need(set(reference['nonzero_moments'])=={'backbone','readout'}and all(type(n)is int and n>0 for n in reference['nonzero_moments'].values()),'REFERENCE_NONZERO_MOMENTS');return r,receipt,reference

def all_tf_inspect(sha,lane,admitted):
 need(lane in ALL_ROOTS,'LANE');binding={'path':str(ALL_PLAN),'sha256':sha};p=read(binding);need(p==build_all_tf_plan(p['metadata']['runtime'],p['inputs']['baseline_acceptance'],p['source_commit']),'EXACT_PLAN');all_tf_check_sources_inputs(p);pref=ref(POLICY);policy=read(pref);validate_flags(policy,admitted);need(policy[ALL_STAGE]==all_tf_gate(binding,admitted),'ONE_USE_GATE');need(not ALL_ROOTS[lane].exists()and not ALL_ROOTS[lane].is_symlink(),'FRESH_OUTPUT');live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(p['metadata']['runtime'],live)
 if admitted and lane=='cold':all_tf_warm_chain(binding,pref,p)
 return p,binding,pref,live

def all_tf_physical(plan,binding,pref,live,lane):
 import torch
 from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
 root=ALL_ROOTS[lane];counts=dict.fromkeys(ALL_LIMITS[lane],0);context=adapter.load_context(plan['metadata']);counts['feature_cache_payload_loads']=counts['target_archive_loads']=1;batches=prefix_batches(context,plan)
 core.future_cuda_seed_and_runtime({'recipe':plan['recipe'],'metadata':{'runtime':live}})
 model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=context['input_binding'],architecture=plan['metadata']['architecture'],device=torch.device('cuda'),train_prior=context['prior'],target_contract=context['target_contract'],backbone_gradient_policy=None);counts['model_constructions']=1
 initial=plan['initial_reference'];need(model.research_objective==plan['model_contract']and core.state_sha(model.state_dict())==initial['model_state_sha256']and core.state_sha(core.training_rng())==initial['rng_sha256'],'FRESH_QUALIFIED_INITIAL_STATE_RNG')
 optimizer=torch.optim.AdamW(model.parameters(),**plan['recipe']['optimizer']);counts['optimizer_constructions']=1;need(not optimizer.state,'EMPTY_FRESH_OPTIMIZER');traj=trajectory(plan);steps=[];scores={};receipts={};extra={}
 def count(key):
  def hook(*args):counts[key]+=1;need(counts[key]<=ALL_LIMITS[lane][key],'FORWARD_BUDGET')
  return hook
 hooks=[model.register_forward_pre_hook(count('model_forwards')),model.backbone.register_forward_pre_hook(count('backbone_forwards')),model.readout.register_forward_pre_hook(count('readout_forwards'))]
 try:
  if lane=='warm':
   first=adapter.packet(context,batches[0]);extra['finite_check_gradient_parity']=core.future_cuda_parity_pair(model,first,plan,counts,equal_tf=False);extra['equal_TF_eval_parity']=core.future_cuda_parity_pair(model,first,plan,counts,equal_tf=True)
   for j in range(3):
    packet=first if j==0 else adapter.packet(context,batches[j]);steps.append(train.training_step(model,optimizer,packet,plan,counts,j));need(steps[-1]['skip_update']is False,'PREFIX_SUPPORT_NO_RESELECTION')
    if j==0:extra['post_first_head_data_gradient']=core.future_cuda_gradient_probe(model,packet,plan,counts)
    if j==1:extra['nonzero_moments_at_cursor2']=nonzero_moments(model,optimizer)
    if j<2:
     cp=train.make_checkpoint(model,optimizer,trajectory=traj,original_plan_ref=binding,lane=lane,cursor=j+1,steps=list(steps),scores=copy.deepcopy(scores),producer_runtime=live);receipts[j+1]=train.publish_checkpoint_receipt(root/('CURSOR%03d.pt'%(j+1)),cp,counts,execution_ref=binding)
    if j==1:need(steps[-1]['backbone_data_gradient_l2']>0,'REAL_BODY_DATA_GRADIENT')
   scores['CURSOR3']=score(model,packet,plan);cp=train.make_checkpoint(model,optimizer,trajectory=traj,original_plan_ref=binding,lane=lane,cursor=3,steps=list(steps),scores=scores,producer_runtime=live);proof=pub(root,'CURSOR003_PROOF.json',{'proof':train.checkpoint_proof(cp),'nonzero_moments':nonzero_moments(model,optimizer)});report={'cursor2_receipt':receipts[2],'cursor1_receipt':receipts[1],'cursor3_proof':proof,**extra}
  else:
   original,receipt,reference=all_tf_warm_chain(binding,pref,plan);cp=torch.load(receipt['checkpoint']['path'],map_location='cpu',weights_only=True);counts['checkpoint_loads']=1
   train.strict_restore(model,optimizer,cp,trajectory=traj,original_plan_ref=binding,expected_cursor=2,current_runtime=live,expected_state_proof=receipt['proof']);counts['model_restores']=1;moments=nonzero_moments(model,optimizer);steps=copy.deepcopy(cp['steps']);scores=copy.deepcopy(cp['scores']);packet=adapter.packet(context,batches[2]);steps.append(train.training_step(model,optimizer,packet,plan,counts,2));scores['CURSOR3']=score(model,packet,plan)
   replay=train.make_checkpoint(model,optimizer,trajectory=traj,original_plan_ref=binding,lane=lane,cursor=3,steps=steps,scores=scores,producer_runtime=live);need(train.checkpoint_proof(replay)==reference['proof'],'BITEXACT_COLD_NEXT_STEP_FULL_STATE');receipt3=train.publish_checkpoint_receipt(root/'REPLAYED_CURSOR003.pt',replay,counts,execution_ref=binding);report={'cold_replay_bit_exact':True,'source_checkpoint_receipt':original['report']['cursor2_receipt'],'replayed_checkpoint_receipt':receipt3,'reference_cursor3_proof':original['report']['cursor3_proof'],'restored_nonzero_moments':moments,'full_cursor3_proof':train.checkpoint_proof(replay)}
  need(counts==ALL_LIMITS[lane],'EXACT_COUNTS');need(torch.cuda.max_memory_reserved()<=ALL_EXECUTION['cuda_peak_reserved_bytes_max'],'VRAM_CAP');return {'counts':counts,'trajectory':traj,'fresh_initial_state_sha256':initial['model_state_sha256'],'logical_updates':3,'physical_updates_this_lane':counts['optimizer_steps'],'prefix_rows':PREFIX_ROWS,'batch_rows':BATCH,'cold_replay_qualified':lane=='cold','cuda_peak_reserved_bytes':int(torch.cuda.max_memory_reserved()),'weights_authority':'TECHNICAL_ONLY_NEVER_MAIN_INITIALIZER_OR_SELECTION','learning_generalization_edge':'UNKNOWN',**report}
 finally:
  for h in hooks:h.remove()

def all_tf_run(sha,lane):
 from gx1.contracts.gx1_capped_execution_v1 import require_native_provider_hlc_all_tf_replay_execution
 began=time.monotonic()
 def stop(*_):raise TimeoutError('PROVIDER_HLC_REPLAY_SIGNAL_OR_WALL')
 for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
 signal.setitimer(signal.ITIMER_REAL,1770);guard=require_native_provider_hlc_all_tf_replay_execution(plan_sha256=sha,lane=lane);plan,binding,pref,live=all_tf_inspect(sha,lane,True);root=ALL_ROOTS[lane];need(guard['plan']==binding and guard['output_root']==str(root),'GUARD_PLAN_ROOT');root.mkdir(exist_ok=False);identity=core.future_cpu_process_identity();claim=pub(root,'CLAIM.json',{'plan':binding,'policy_at_claim':pref,'lane':lane,'runtime':live,'process_identity':identity,'remaining_invocations':0,'main_fit_authorized':False})
 try:
  allowed={plan['inputs'][k]['path']for k in ('feature_archive','target_arrays')};allowed|={str(ALL_ROOTS['warm']/('CURSOR%03d.pt'%j))for j in (1,2)};allowed.add(str(ALL_ROOTS['cold']/'REPLAYED_CURSOR003.pt'))
  def audit(event,args):
   if event in ('socket.connect','socket.getaddrinfo','urllib.Request'):raise RuntimeError('REPLAY_NETWORK_FORBIDDEN')
   if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
    p=os.fsdecode(args[0])
    if p.endswith(('.npz','.pt','.pth','.parquet','.safetensors','.pkl'))and str(Path(p).resolve())not in allowed:raise RuntimeError('REPLAY_PAYLOAD_PATH_FORBIDDEN')
  sys.addaudithook(audit);report=all_tf_physical(plan,binding,pref,live,lane);all_tf_check_sources_inputs(plan);need(ref(ALL_PLAN)==binding and ref(POLICY)==pref,'SAME_FROZEN_PLAN_POLICY');need(core.physical_windows_boot_id()==live['physical_windows_boot_id']and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==live['wsl_boot_id'],'BOTH_BOOT_BOOKEND');need(core.common.firewall_state()['denied_attempts']==0,'M1_FIREWALL')
  result=pub(root,'RESULT.json',{'status':'COMPLETE_ALL_TF_TECHNICAL_WARM_PREFIX'if lane=='warm'else'COMPLETE_ALL_TF_TECHNICAL_COLD_REPLAY','plan':binding,'claim':claim,'process_identity':identity,'report':report,'source_inputs_policy_boot_unchanged':True,'elapsed_seconds':time.monotonic()-began,'test_data_used':False});terminal=pub(root,'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False});return {'result':result,'terminal':terminal}
 except BaseException as e:pub(root,'TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'plan':binding,'claim':claim,'error':type(e).__name__+':'+str(e)[:1800],'remaining_invocations':0,'relaunch_allowed':False});raise
 finally:signal.setitimer(signal.ITIMER_REAL,0)
def main_all_tf():
 p=argparse.ArgumentParser();p.add_argument('--all-tf-plan-sha256',required=True);p.add_argument('--lane',choices=tuple(ALL_ROOTS),required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args();need(sys.argv[1:]==['--all-tf-plan-sha256',a.all_tf_plan_sha256,'--lane',a.lane]+(['--preflight']if a.preflight else[]),'EXACT_CLI')
 if a.preflight:
  # Cold dependencies are real after warm completion, under the same admitted policy.
  admitted=a.lane=='cold'and read(ref(POLICY)).get(ALL_STAGE,{}).get('status')=='ADMITTED_EXACT_ALL_TF_TECHNICAL_COLD_REPLAY';plan,binding,pref,live=all_tf_inspect(a.all_tf_plan_sha256,a.lane,admitted);print(json.dumps({'status':'PREFLIGHT_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PASS','lane':a.lane,'plan':binding,'policy':pref,'payloads_decoded':0,'models':0,'CUDA_initialized':False,'output_created':False}))
 else:print(json.dumps(all_tf_run(a.all_tf_plan_sha256,a.lane),sort_keys=True))
# End explicit None candidate cold replay.

def main():
 if sys.argv[1:2]==['--all-tf-plan-sha256']:return main_all_tf()
 p=argparse.ArgumentParser();p.add_argument('--plan-sha256',required=True);p.add_argument('--lane',choices=tuple(ROOTS),required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args();need(sys.argv[1:]==['--plan-sha256',a.plan_sha256,'--lane',a.lane]+(['--preflight']if a.preflight else[]),'EXACT_CLI')
 if a.preflight:
  # Cold dependencies are real after warm completion, under the same admitted policy.
  admitted=a.lane=='cold'and read(ref(POLICY)).get(STAGE,{}).get('status')=='ADMITTED_EXACT_TECHNICAL_COLD_REPLAY';plan,binding,pref,live=inspect(a.plan_sha256,a.lane,admitted);print(json.dumps({'status':'PREFLIGHT_NATIVE_PROVIDER_HLC_REPLAY_PASS','lane':a.lane,'plan':binding,'policy':pref,'payloads_decoded':0,'models':0,'CUDA_initialized':False,'output_created':False}))
 else:print(json.dumps(run(a.plan_sha256,a.lane),sort_keys=True))
if __name__=='__main__':main()
