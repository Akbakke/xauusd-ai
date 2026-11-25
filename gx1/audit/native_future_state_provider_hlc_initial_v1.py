"""One fresh, read-only, complete-TRAIN provider-HLC v2 NN INITIAL baseline."""
from __future__ import annotations
import argparse,hashlib,io,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
if __name__=='__main__':
 from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
 install_native_m5_direction_io_firewall()
from gx1.audit import qualify_native_m5_full_gc_direction_v1 as core
from gx1.contracts import native_future_state_provider_hlc_v1 as hlc
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
REPO=Path('/home/andre2/src/GX1_CURRENT');POLICY=REPO/'NEXT_RUN_POLICY.json'
NAME='NATIVE_PROVIDER_HLC_WHOLE_TRAIN_INITIAL_20261123_002';STAGE='native_provider_hlc_whole_train_initial_20261123_002'
PLAN=REPO/'configs/research'/(NAME+'.json');ROOT=Path('/home/andre2/GX1_RUNS')/NAME
SELF='gx1/audit/native_future_state_provider_hlc_initial_v1.py'
HELPER='gx1/audit/native_future_state_provider_hlc_baseline_v1.py';HELPER_TEST='tests/test_native_future_state_provider_hlc_baseline_v1.py';GUARD_TEST='tests/test_native_future_state_provider_hlc_baseline_guard_v1.py'
E=Path(hlc.EROOT);ANCHOR='914e29a516d6762b59f6c8c21801e40af751e238'
MAIN={'path':str(REPO/'configs/research/NATIVE_FUTURE_STATE_ONE_EPOCH_20261121_001.json'),'sha256':'c3e769fc5468739b576144f2cb2e5bf572d6b662a3c490ee56d8641fde1dd561'}
DATASET={'path':'/home/andre2/GX1_RUNS/NATIVE_D1_PROVIDER_HLC_TARGET_QUALIFICATION_20261123_001/FUTURE_STATE_HLC_DATASET001.json','sha256':'b7727cddf8bba2f92caa9b8979995478b641518ddb768670b0ec6c0c8152bfc3'}
PRIOR={'path':'/home/andre2/GX1_RUNS/NATIVE_D1_PROVIDER_HLC_TARGET_QUALIFICATION_20261123_001/MATCHED_TRAIN_PRIOR001.json','sha256':'78b3766212e480c6eac9066b7f15fc174c9da55ee4f755b3286653d858051266'}
ROWS=354550;BATCH=256;BATCHES=1385;LAST=246;SEED=20260911
FLAGS=('training_enabled','native_training_enabled','native_launch_authorized','full_epoch_training_allowed','full_val_allowed')
EXECUTION={'class':'trainer','device':'cuda:0','memory':'20G','swap':'512M','tasks_max':128,'numerical_threads':8,'cpu_affinity':list(range(19)),'outer_seconds':3600,'internal_seconds':3570,'kill_after_seconds':10,'cuda_peak_reserved_bytes_max':12884901888,'GPU_power_limit_W':250,'GPU_power_draw_stop_W':250,'GPU_core_stop_C':75,'GPU_memory_stop_C':80,'host_CPU_stop_C':80,'host_CPU_package_stop_W':125,'host_min_available_bytes':8589934592,'host_min_commit_headroom_bytes':8589934592,'watchdog_interval_seconds':1}
LIMITS={'model_constructions':1,'model_forwards':BATCHES,'backbone_forwards':BATCHES,'readout_forwards':BATCHES,'feature_cache_payload_loads':1,'v2_truth_archive_loads':1,'truth_fields_decoded':14,'new_checkpoint_writes':1,'new_checkpoint_readbacks':1,'existing_checkpoint_loads':0,'model_state_restores':0,'optimizer_constructions':0,'optimizer_steps':0,'backward_calls':0,'autograd_calls':0,'normalization_fits':0,'prior_fits':0,'CONTROL_VAL_TEST_rows':0}
def need(ok,why):
 if not bool(ok):raise ValueError('PROVIDER_HLC_INITIAL_'+why)
def ref(path):
 p=Path(path).resolve();h=hashlib.sha256()
 with p.open('rb')as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return {'path':str(p),'sha256':h.hexdigest()}
def read(binding):need(ref(binding['path'])==binding,'REFERENCE');return json.loads(Path(binding['path']).read_text())
def publish(name,obj):
 p=ROOT/name;b=obj if isinstance(obj,bytes)else(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();_publish_bytes(p,b);return ref(p)
def guard_anchor():
 names=('gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh')
 return {n:hashlib.sha256(subprocess.check_output(['git','show',ANCHOR+':'+n],cwd=REPO)).hexdigest()for n in names}
def build_plan(runtime,source_commit=None):
 from gx1.contracts.gx1_capped_execution_v1 import native_provider_hlc_initial_source_bridge
 from gx1.contracts.market_state_ta_integrated_model_v1 import native_future_state_model_contract
 from gx1.contracts.market_state_ta_integrated_objective_v1 import require_native_future_state_train_prior
 core.future_cuda_require_runtime(runtime);main=read(MAIN);old=main['metadata'];data=read(DATASET);prior=require_native_future_state_train_prior(read(PRIOR),target_contract=data['contract']);snapshot=read(old['source_snapshot']);norm=read(old['normalization'])
 need(data['contract']==hlc.provider_hlc_state_contract()and data['base_dataset']==hlc.SOURCE_BINDINGS['base_dataset']and data['cutoff_ns']==hlc.TRAIN_UPPER_NS and data['model_input']is False,'TARGET_DATASET')
 need(data['target_observation_certificate_ref']['source_binding_sha256']==hlc.provider_hlc_state_contract()['provider_hlc_source_binding_sha256'],'CERT_SOURCE')
 need(prior['fit_population']['issue_rows']==ROWS and prior['fit_population']['issue_ns_sha256']==hlc.TRAIN_ISSUE_SHA256 and prior['fit_population']['cutoff_ns']==hlc.TRAIN_UPPER_NS,'NEW_PRIOR_POPULATION')
 meta={k:old[k]for k in ('source_snapshot','normalization','input_binding_ref','input_binding','architecture','native_manifest')};meta.update(dataset=DATASET,prior=PRIOR,target_contract=data['contract'],truth_sha256=prior['fit_population']['truth_sha256'],TRAIN_rows=ROWS,TRAIN_issue_ns_sha256=hlc.TRAIN_ISSUE_SHA256,runtime=runtime)
 need(read(meta['input_binding_ref'])==meta['input_binding']and meta['input_binding']['input_normalization_sha256']==norm['contract_sha256'],'INPUT_NORMALIZATION_BINDING')
 cert=data['target_observation_certificate_ref'];inputs={'original_MAIN_plan':MAIN,'source_snapshot':meta['source_snapshot'],'normalization':meta['normalization'],'input_binding_ref':meta['input_binding_ref'],'feature_archive':snapshot['archive'],'dataset':DATASET,'target_arrays':data['arrays'],'new_prior':PRIOR,'certificate':{k:cert[k]for k in ('path','sha256')},'provider_response':hlc.SOURCE_BINDINGS['reference_raw'],'target_acceptance':{'path':str(E/'D1_PROVIDER_HLC_TARGET_ACCEPTANCE001.json'),'sha256':'f20b79e2d37f3315157cd89ca74b2a12cb0e3c520a5a9bf3347ffbee7568981c'}}
 names=set(main['source_owners'])|{SELF,HELPER,HELPER_TEST,GUARD_TEST,'gx1/contracts/native_future_state_provider_hlc_v1.py','tests/test_native_future_state_provider_hlc_v1.py','tests/test_native_future_state_provider_hlc_consumers_v1.py'}
 headers=snapshot['arrays'];feature_bytes=sum(int(np.prod(x['shape']))*np.dtype(x['dtype']).itemsize for x in headers.values());need(len(headers)==27 and feature_bytes==1433216758,'FEATURE_ARRAY_BUDGET')
 model_contract=native_future_state_model_contract(meta['input_binding'],architecture=meta['architecture'],train_prior=prior,target_contract=meta['target_contract'],backbone_gradient_policy='M5_primary')
 return {'schema_version':'gx1_native_provider_hlc_whole_train_initial_plan_v1','stage':STAGE,'source_commit':source_commit or runtime['source_commit'],'source_owners':{n:ref(REPO/n)['sha256']for n in sorted(names)},'source_bridge':native_provider_hlc_initial_source_bridge(guard_anchor(),producer_commit=ANCHOR),'inputs':inputs,'metadata':meta,'model_contract':model_contract,'execution':EXECUTION,'limits':LIMITS,'recipe':{'seed':SEED,'batch_rows':BATCH,'batches':BATCHES,'last_rows':LAST,'rows':ROWS,'shuffle':False,'drop_last':False,'backbone_gradient_policy':'M5_primary','no_optimization':True},'output_root':str(ROOT),'input_budget':{'feature_arrays':27,'feature_ndarray_bytes':feature_bytes,'v2_truth_arrays':14,'v2_truth_ndarray_bytes':50700650,'full_original_30array_target_loads':0},'scope':'Actual fresh v2 NN INITIAL across all354550TRAIN issues and all5TF/allgenuineinputs; notfit,optimizerreplay,learning,generalization,calendar/vintage or edge. Analyticprior or a smallprobe cannotreplace actual fullforward coverage.'}
def gate(binding,admitted):return {'status':'ADMITTED_ONE_USE_WHOLE_TRAIN_INITIAL_FORWARD_ONLY'if admitted else'PREPARED_NOT_ADMITTED','plan':binding,'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':False,'optimizer_authorized':False,'test_authorized':False}
def validate_flags(policy):need(all(policy.get(k)is False for k in FLAGS),'ALL_TRAINING_FLAGS_FALSE')
def check_sources_inputs(plan):
 need(subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='work/gx1-current'and subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_commit'],'SOURCE_HEAD')
 for n,digest in plan['source_owners'].items():need(ref(REPO/n)['sha256']==digest,'SOURCE_OWNER:'+n)
 for binding in plan['inputs'].values():need(ref(binding['path'])==binding,'INPUT_BOOKEND')
def inspect(plan_sha,admitted):
 need(Path.cwd().resolve()==REPO,'CWD');binding={'path':str(PLAN),'sha256':plan_sha};plan=read(binding);need(plan==build_plan(plan['metadata']['runtime'],plan['source_commit']),'EXACT_PLAN');check_sources_inputs(plan);pref=ref(POLICY);policy=read(pref);validate_flags(policy);need(policy[STAGE]==gate(binding,admitted),'ONE_USE_POLICY');need(not ROOT.exists()and not ROOT.is_symlink(),'FRESH_ROOT');live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['metadata']['runtime'],live);return plan,binding,pref,live

def write_initial_checkpoint(model,plan,binding,live,counts):
 import torch
 state={k:x.detach().cpu().clone()for k,x in model.state_dict().items()};rng=core.training_rng()
 cp={'schema_version':'gx1_native_provider_hlc_fresh_initial_checkpoint_v1','plan':binding,'model_contract':plan['model_contract'],'input_binding':plan['metadata']['input_binding'],'normalization':plan['metadata']['normalization'],'prior':PRIOR,'dataset':DATASET,'target_contract':plan['metadata']['target_contract'],'truth_sha256':plan['metadata']['truth_sha256'],'seed':SEED,'runtime':live,'model_state':state,'model_state_sha256':core.state_sha(state),'rng_state':rng,'training':model.training,'optimizer':None,'optimizer_updates':0,'authority':'INITIAL_ONLY_NO_FIT_OR_RESUME_AUTHORITY'}
 stream=io.BytesIO();torch.save(cp,stream);checkpoint=publish('INITIAL.pt',stream.getvalue());stream.close();counts['new_checkpoint_writes']+=1
 checked=torch.load(checkpoint['path'],map_location='cpu',weights_only=True);counts['new_checkpoint_readbacks']+=1;need(core.state_sha(cp)==core.state_sha(checked)and core.state_sha(checked['model_state'])==core.state_sha(model.state_dict())and core.state_sha(rng)==core.state_sha(core.training_rng()),'INITIAL_DISK_FULL_STATE_RNG')
 receipt=publish('INITIAL.json',{'schema_version':cp['schema_version'],'plan':binding,'checkpoint':checkpoint,'checkpoint_tree_sha256':core.state_sha(cp),'model_state_sha256':cp['model_state_sha256'],'rng_sha256':core.state_sha(rng),'model_contract_sha256':plan['model_contract']['contract_sha256'],'target_contract_sha256':plan['metadata']['target_contract']['contract_sha256'],'optimizer_updates':0,'initial_only':True})
 return receipt,cp['model_state_sha256']
def physical(plan,binding,live,progress):
 import torch
 from gx1.audit import native_future_state_provider_hlc_baseline_v1 as adapter
 from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
 counts=dict.fromkeys(LIMITS,0);progress.update(phase='LOAD_UNCHANGED_FEATURES_V2_TRUTH',counts=counts);context=adapter.load_context(plan['metadata']);counts['feature_cache_payload_loads']=counts['v2_truth_archive_loads']=1;counts['truth_fields_decoded']=14
 need(len(context['source_rows'])==ROWS and len(context['truth']['issue_ns'])==ROWS,'FULL_CONTEXT_ROWS');runtime=core.future_cuda_seed_and_runtime({'recipe':{'seed':SEED},'metadata':{'runtime':live}})
 model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=plan['metadata']['input_binding'],architecture=plan['metadata']['architecture'],device=torch.device('cuda'),train_prior=context['prior'],target_contract=plan['metadata']['target_contract'],backbone_gradient_policy='M5_primary');counts['model_constructions']=1
 need(model.research_objective==plan['model_contract']and all(p.grad is None for p in model.parameters()),'FRESH_MODEL_IDENTITY');initial,state_sha=write_initial_checkpoint(model,plan,binding,live,counts)
 def counter(key):
  def count(*args):counts[key]+=1;need(counts[key]<=LIMITS[key],'FORWARD_BUDGET:'+key)
  return count
 hooks=[model.register_forward_pre_hook(counter('model_forwards')),model.backbone.register_forward_pre_hook(counter('backbone_forwards')),model.readout.register_forward_pre_hook(counter('readout_forwards'))]
 last_cursor=0
 def observe(event):
  nonlocal last_cursor
  progress.update(event)
  number=event['batch_cursor'];need(number==last_cursor+1 and 1<=number<=BATCHES and counts['model_forwards']==number,'PROGRESS_CURSOR');last_cursor=number
  if number and(number%256==0 or number==BATCHES):publish('PROGRESS_%04d.json'%number,{'phase':'ACTUAL_WHOLE_TRAIN_INITIAL','progress':event,'counts':counts,'initial_checkpoint':initial,'optimizer_updates':0})
 try:
  progress['phase']='ACTUAL_WHOLE_TRAIN_INITIAL_FORWARD';score=adapter.score_initial(model,context,batch_rows=BATCH,device='cuda',progress_callback=observe)
 finally:
  for hook in hooks:hook.remove()
 need(last_cursor==BATCHES and score['rows']==ROWS and score['batch_count']==BATCHES and score['initial_prior_exact']is True and core.state_sha(model.state_dict())==state_sha and all(p.grad is None for p in model.parameters()),'WHOLE_TRAIN_INITIAL_STATE_AND_SCORE');need(counts==LIMITS,'EXACT_COUNTS');need(torch.cuda.max_memory_reserved()<=EXECUTION['cuda_peak_reserved_bytes_max'],'CUDA_MEMORY_CAP')
 progress['phase']='COMPLETE';return {'whole_TRAIN_initialbaseline_qualified':True,'initial':score,'initial_state_sha256':state_sha,'initial_checkpoint':initial,'counts':counts,'runtime':runtime,'cuda_peak_reserved_bytes':int(torch.cuda.max_memory_reserved()),'process_peak_RSS_KiB':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),'genuine_inputs':'Unchanged9featurekeys/eightfamilies/M5-M15-H1-H4-D1/D1macro;targetmetadata outsideforward.','optimizer_replay_qualified':False,'learning_qualified':False,'later_generalization':'UNKNOWN','economic_edge':'UNKNOWN'}
def run(plan_sha):
 from gx1.contracts.gx1_capped_execution_v1 import require_native_provider_hlc_initial_execution
 began=time.monotonic()
 def stopped(*_):raise TimeoutError('PROVIDER_HLC_INITIAL_WALL_OR_SIGNAL')
 for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stopped)
 signal.setitimer(signal.ITIMER_REAL,EXECUTION['internal_seconds']);guard=require_native_provider_hlc_initial_execution(plan_sha256=plan_sha);plan,binding,pref,live=inspect(plan_sha,True);need(guard['plan']==binding and guard['output_root']==str(ROOT),'EXACT_GUARD');ROOT.mkdir(exist_ok=False);claim=publish('CLAIM.json',{'plan':binding,'policy':pref,'runtime':live,'process_identity':core.future_cpu_process_identity(),'remaining_invocations':0,'relaunch_allowed':False,'training_authorized':False,'optimizer_updates':0});progress={'phase':'CLAIMED'}
 try:
  import torch
  allowed={plan['inputs'][k]['path']for k in ('feature_archive','target_arrays')};allowed.add(str(ROOT/'INITIAL.pt'))
  def audit(event,args):
   if event in ('socket.connect','socket.getaddrinfo','urllib.Request'):raise RuntimeError('PROVIDER_INITIAL_NETWORK_FORBIDDEN')
   if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
    path=os.fsdecode(args[0])
    if path.endswith(('.npz','.pt','.pth','.parquet','.safetensors','.pkl'))and str(Path(path).resolve())not in allowed:raise RuntimeError('PROVIDER_INITIAL_PAYLOAD_PATH_FORBIDDEN')
  sys.addaudithook(audit)
  def no_optimization(*args,**kwargs):raise RuntimeError('PROVIDER_INITIAL_OPTIMIZATION_FORBIDDEN')
  torch.optim.Optimizer.__init__=no_optimization;torch.Tensor.backward=no_optimization;torch.autograd.grad=no_optimization
  report=physical(plan,binding,live,progress);check_sources_inputs(plan);need(ref(PLAN)==binding and ref(POLICY)==pref,'POLICY_PLAN_BOOKEND');need(core.physical_windows_boot_id()==live['physical_windows_boot_id']and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==live['wsl_boot_id'],'BOOT_BOOKEND');need(core.common.firewall_state()['denied_attempts']==0,'M1_FIREWALL')
  result=publish('RESULT.json',{'status':'COMPLETE_NATIVE_PROVIDER_HLC_WHOLE_TRAIN_INITIAL','plan':binding,'claim':claim,'report':report,'source_policy_inputs_boot_unchanged':True,'test_data_used':False,'elapsed_seconds':time.monotonic()-began});terminal=publish('TERMINAL.json',{'status':'COMPLETE','exit_code':0,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False});return {'result':result,'terminal':terminal}
 except BaseException as error:
  publish('TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'plan':binding,'claim':claim,'error':type(error).__name__+':'+str(error)[:1800],'progress':progress,'remaining_invocations':0,'relaunch_allowed':False});raise
 finally:signal.setitimer(signal.ITIMER_REAL,0)
# Explicit None fresh INITIAL; original M5 producer remains consumed.
ALL_NAME='NATIVE_PROVIDER_HLC_ALL_TF_WHOLE_TRAIN_INITIAL_20261124_002';ALL_STAGE=ALL_NAME.lower();ALL_PLAN=REPO/'configs/research'/(ALL_NAME+'.json');ALL_ROOT=Path('/home/andre2/GX1_RUNS')/ALL_NAME
ALL_SCOPE={'path':str(E/'PROVIDER_HLC_ALL_TF_FRESH_INITIAL_REPLAY_PREPARATION_SCOPE001.json'),'sha256':'aa4a01793dfd02344eefc491a83639caa3a00cf626f7e3827c1d8c7730816a21'}
ALL_PROFILE={'path':str(E/'PROVIDER_HLC_ALL_TF_FRESH_INITIAL_REPLAY_PROFILE_DESIGN002.json'),'sha256':'0247b13570a6fb964cd7bb1420eb648547215902f361288aefefddeb7fbe9060'}
ALL_PARENT='f33da18dd6a0e3d59e4876fa26f8e62ec5ab97b8';ALL_TEST='tests/test_native_future_state_provider_hlc_initial_v1.py'
ALL_EXECUTION=dict(EXECUTION);ALL_LIMITS=dict(LIMITS);ALL_CONTRACT_SHA='6d6ac22819e54c7b12aebcf4e1afe27b990d8a4d90d16500e352c7353d99cca0'
all_tf_check_sources_inputs=check_sources_inputs

def all_tf_guard_anchor():
 names=('gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh')
 return {n:hashlib.sha256(subprocess.check_output(['git','show',ALL_PARENT+':'+n],cwd=REPO)).hexdigest()for n in names}
def build_all_tf_plan(runtime,source_commit=None):
 from gx1.contracts.gx1_capped_execution_v1 import native_provider_hlc_all_tf_initial_source_bridge
 from gx1.contracts.market_state_ta_integrated_model_v1 import native_future_state_model_contract
 from gx1.contracts.market_state_ta_integrated_objective_v1 import require_native_future_state_train_prior
 scope=read(ALL_SCOPE);profile=read(ALL_PROFILE);need(scope['candidate_policy']['explicit_backbone_gradient_policy']is None and scope['candidate_policy']['model_contract_sha256_at_unchanged_input_target_architecture']==ALL_CONTRACT_SHA and profile['scope']==ALL_SCOPE and profile['candidate_backbone_gradient_policy']is None and profile['initial']['new_plan_id']==ALL_NAME and profile['initial']['execution']==ALL_EXECUTION and profile['initial']['limits']==ALL_LIMITS,'ALL_TF_SCOPE_PROFILE')
 core.future_cuda_require_runtime(runtime);main_all_tf=read(MAIN);old=main_all_tf['metadata'];data=read(DATASET);prior=require_native_future_state_train_prior(read(PRIOR),target_contract=data['contract']);snapshot=read(old['source_snapshot']);norm=read(old['normalization'])
 need(data['contract']==hlc.provider_hlc_state_contract()and data['base_dataset']==hlc.SOURCE_BINDINGS['base_dataset']and data['cutoff_ns']==hlc.TRAIN_UPPER_NS and data['model_input']is False,'TARGET_DATASET')
 need(data['target_observation_certificate_ref']['source_binding_sha256']==hlc.provider_hlc_state_contract()['provider_hlc_source_binding_sha256'],'CERT_SOURCE')
 need(prior['fit_population']['issue_rows']==ROWS and prior['fit_population']['issue_ns_sha256']==hlc.TRAIN_ISSUE_SHA256 and prior['fit_population']['cutoff_ns']==hlc.TRAIN_UPPER_NS,'NEW_PRIOR_POPULATION')
 meta={k:old[k]for k in ('source_snapshot','normalization','input_binding_ref','input_binding','architecture','native_manifest')};meta.update(dataset=DATASET,prior=PRIOR,target_contract=data['contract'],truth_sha256=prior['fit_population']['truth_sha256'],TRAIN_rows=ROWS,TRAIN_issue_ns_sha256=hlc.TRAIN_ISSUE_SHA256,runtime=runtime)
 need(read(meta['input_binding_ref'])==meta['input_binding']and meta['input_binding']['input_normalization_sha256']==norm['contract_sha256'],'INPUT_NORMALIZATION_BINDING')
 cert=data['target_observation_certificate_ref'];inputs={'original_MAIN_plan':MAIN,'source_snapshot':meta['source_snapshot'],'normalization':meta['normalization'],'input_binding_ref':meta['input_binding_ref'],'feature_archive':snapshot['archive'],'dataset':DATASET,'target_arrays':data['arrays'],'new_prior':PRIOR,'certificate':{k:cert[k]for k in ('path','sha256')},'provider_response':hlc.SOURCE_BINDINGS['reference_raw'],'target_acceptance':{'path':str(E/'D1_PROVIDER_HLC_TARGET_ACCEPTANCE001.json'),'sha256':'f20b79e2d37f3315157cd89ca74b2a12cb0e3c520a5a9bf3347ffbee7568981c'}}
 names=set(main_all_tf['source_owners'])|{SELF,HELPER,HELPER_TEST,GUARD_TEST,ALL_TEST,'gx1/audit/native_provider_hlc_replay_v1.py','tests/test_native_provider_hlc_replay_v1.py','tests/test_native_provider_hlc_replay_guard_v1.py','gx1/contracts/native_future_state_provider_hlc_v1.py','tests/test_native_future_state_provider_hlc_v1.py','tests/test_native_future_state_provider_hlc_consumers_v1.py'}
 headers=snapshot['arrays'];feature_bytes=sum(int(np.prod(x['shape']))*np.dtype(x['dtype']).itemsize for x in headers.values());need(len(headers)==27 and feature_bytes==1433216758,'FEATURE_ARRAY_BUDGET')
 model_contract=native_future_state_model_contract(meta['input_binding'],architecture=meta['architecture'],train_prior=prior,target_contract=meta['target_contract'],backbone_gradient_policy=None)
 reference_result=read(scope['unchanged_reference']['initial_measurement']);reference_acceptance=read(scope['unchanged_reference']['initial_acceptance']);reference_plan=read(reference_result['plan']);reference_receipt=read(reference_acceptance['initial_receipt'])
 need(profile['initial']['recipe']=={'seed':SEED,'batch_rows':BATCH,'batches':BATCHES,'last_rows':LAST,'rows':ROWS,'shuffle':False,'drop_last':False,'backbone_gradient_policy':None,'no_optimization':True},'ALL_TF_RECIPE_PROFILE')
 need(reference_acceptance['result']==scope['unchanged_reference']['initial_measurement']and reference_acceptance['full_initialbaseline_qualified']is True and reference_result['report']['whole_TRAIN_initialbaseline_qualified']is True and reference_receipt['initial_only']is True and reference_receipt['optimizer_updates']==0 and reference_receipt['model_state_sha256']==reference_result['report']['initial_state_sha256']and reference_plan['model_contract']['backbone_gradient_policy']['policy']=='M5_primary','ALL_TF_REFERENCE_INITIAL')
 need(all(meta[k]==reference_plan['metadata'][k]for k in ('source_snapshot','normalization','input_binding','architecture','dataset','prior','target_contract','truth_sha256'))and model_contract['contract_sha256']==ALL_CONTRACT_SHA and 'backbone_gradient_policy'not in model_contract,'ALL_TF_SAME_INPUT_VALUE_FUNCTION')
 inputs.update(scope=ALL_SCOPE,profile_design=ALL_PROFILE,predecessor_acceptance=scope['predecessor_acceptance'],predecessor_result=scope['predecessor_result'],reference_initial_acceptance=scope['unchanged_reference']['initial_acceptance'],reference_initial_result=scope['unchanged_reference']['initial_measurement'],reference_initial_plan=reference_result['plan'],reference_initial_receipt=reference_acceptance['initial_receipt'])
 return {'schema_version':'gx1_native_provider_hlc_all_tf_whole_train_initial_plan_v1','stage':ALL_STAGE,'source_commit':source_commit or runtime['source_commit'],'source_owners':{n:ref(REPO/n)['sha256']for n in sorted(names)},'source_bridge':native_provider_hlc_all_tf_initial_source_bridge(all_tf_guard_anchor(),producer_commit=ALL_PARENT),'inputs':inputs,'metadata':meta,'model_contract':model_contract,'execution':ALL_EXECUTION,'limits':ALL_LIMITS,'recipe':{'seed':SEED,'batch_rows':BATCH,'batches':BATCHES,'last_rows':LAST,'rows':ROWS,'shuffle':False,'drop_last':False,'backbone_gradient_policy':None,'no_optimization':True},'output_root':str(ALL_ROOT),'input_budget':{'feature_arrays':27,'feature_ndarray_bytes':feature_bytes,'v2_truth_arrays':14,'v2_truth_ndarray_bytes':50700650,'full_original_30array_target_loads':0},'reference_initial':reference_receipt,'reference_model_contract':reference_plan['model_contract'],'backbone_gradient_policy':None,'scope_ref':ALL_SCOPE,'scope':'Actual fresh explicitNone v2 NN INITIAL across all354550TRAIN issues and all5TF/allgenuineinputs; notfit,optimizerreplay,learning,generalization,calendar/vintage or edge. Analyticprior or a smallprobe cannotreplace actual fullforward coverage.'}
def all_tf_publish(name,obj):
 p=ALL_ROOT/name;b=obj if isinstance(obj,bytes)else(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();_publish_bytes(p,b);return ref(p)
def all_tf_gate(binding,admitted):return {'status':'ADMITTED_ONE_USE_WHOLE_TRAIN_INITIAL_FORWARD_ONLY'if admitted else'PREPARED_NOT_ADMITTED','plan':binding,'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':False,'optimizer_authorized':False,'test_authorized':False}
def all_tf_inspect(plan_sha,admitted):
 need(Path.cwd().resolve()==REPO,'CWD');binding={'path':str(ALL_PLAN),'sha256':plan_sha};plan=read(binding);need(plan==build_all_tf_plan(plan['metadata']['runtime'],plan['source_commit']),'EXACT_PLAN');all_tf_check_sources_inputs(plan);pref=ref(POLICY);policy=read(pref);validate_flags(policy);need(policy[ALL_STAGE]==all_tf_gate(binding,admitted),'ONE_USE_POLICY');need(not ALL_ROOT.exists()and not ALL_ROOT.is_symlink(),'FRESH_ROOT');live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['metadata']['runtime'],live);return plan,binding,pref,live

def all_tf_write_initial_checkpoint(model,plan,binding,live,counts):
 import torch
 state={k:x.detach().cpu().clone()for k,x in model.state_dict().items()};rng=core.training_rng()
 cp={'schema_version':'gx1_native_provider_hlc_fresh_initial_checkpoint_v1','plan':binding,'model_contract':plan['model_contract'],'input_binding':plan['metadata']['input_binding'],'normalization':plan['metadata']['normalization'],'prior':PRIOR,'dataset':DATASET,'target_contract':plan['metadata']['target_contract'],'truth_sha256':plan['metadata']['truth_sha256'],'seed':SEED,'runtime':live,'model_state':state,'model_state_sha256':core.state_sha(state),'rng_state':rng,'training':model.training,'optimizer':None,'optimizer_updates':0,'authority':'INITIAL_ONLY_NO_FIT_OR_RESUME_AUTHORITY'}
 stream=io.BytesIO();torch.save(cp,stream);checkpoint=all_tf_publish('INITIAL.pt',stream.getvalue());stream.close();counts['new_checkpoint_writes']+=1
 checked=torch.load(checkpoint['path'],map_location='cpu',weights_only=True);counts['new_checkpoint_readbacks']+=1;need(core.state_sha(cp)==core.state_sha(checked)and core.state_sha(checked['model_state'])==core.state_sha(model.state_dict())and core.state_sha(rng)==core.state_sha(core.training_rng()),'INITIAL_DISK_FULL_STATE_RNG')
 receipt=all_tf_publish('INITIAL.json',{'schema_version':cp['schema_version'],'plan':binding,'checkpoint':checkpoint,'checkpoint_tree_sha256':core.state_sha(cp),'model_state_sha256':cp['model_state_sha256'],'rng_sha256':core.state_sha(rng),'model_contract_sha256':plan['model_contract']['contract_sha256'],'target_contract_sha256':plan['metadata']['target_contract']['contract_sha256'],'optimizer_updates':0,'initial_only':True})
 return receipt,cp['model_state_sha256']
def all_tf_compare_fresh_reference(model,plan):
 import torch
 state=dict(model.state_dict());own=core.state_sha(state);reference=plan['reference_model_contract']
 need(model.research_objective==plan['model_contract']and 'backbone_gradient_policy'not in model.research_objective,'ALL_TF_FRESH_NONE_IDENTITY')
 # Hash comparison only: this copied dictionary is never installed into a model.
 state['future_state_model_sha256']=torch.tensor(list(bytes.fromhex(reference['contract_sha256'])),dtype=torch.uint8)
 normalized=core.state_sha(state);rng=core.state_sha(core.training_rng())
 need(normalized==plan['reference_initial']['model_state_sha256']and rng==plan['reference_initial']['rng_sha256']and core.state_sha(model.state_dict())==own,'ALL_TF_FRESH_REFERENCE_PARAMETERS_RNG')
 return {'comparison_only_no_state_load':True,'policy_normalized_state_sha256':normalized,'fresh_model_state_sha256':own,'fresh_rng_sha256':rng}

def all_tf_physical(plan,binding,live,progress):
 import torch
 from gx1.audit import native_future_state_provider_hlc_baseline_v1 as adapter
 from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
 counts=dict.fromkeys(ALL_LIMITS,0);progress.update(phase='LOAD_UNCHANGED_FEATURES_V2_TRUTH',counts=counts);context=adapter.load_context(plan['metadata']);counts['feature_cache_payload_loads']=counts['v2_truth_archive_loads']=1;counts['truth_fields_decoded']=14
 need(len(context['source_rows'])==ROWS and len(context['truth']['issue_ns'])==ROWS,'FULL_CONTEXT_ROWS');runtime=core.future_cuda_seed_and_runtime({'recipe':{'seed':SEED},'metadata':{'runtime':live}})
 model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=plan['metadata']['input_binding'],architecture=plan['metadata']['architecture'],device=torch.device('cuda'),train_prior=context['prior'],target_contract=plan['metadata']['target_contract'],backbone_gradient_policy=None);counts['model_constructions']=1
 need(model.research_objective==plan['model_contract']and all(p.grad is None for p in model.parameters()),'FRESH_MODEL_IDENTITY');comparison=all_tf_compare_fresh_reference(model,plan);initial,state_sha=all_tf_write_initial_checkpoint(model,plan,binding,live,counts)
 def counter(key):
  def count(*args):counts[key]+=1;need(counts[key]<=ALL_LIMITS[key],'FORWARD_BUDGET:'+key)
  return count
 hooks=[model.register_forward_pre_hook(counter('model_forwards')),model.backbone.register_forward_pre_hook(counter('backbone_forwards')),model.readout.register_forward_pre_hook(counter('readout_forwards'))]
 last_cursor=0
 def observe(event):
  nonlocal last_cursor
  progress.update(event)
  number=event['batch_cursor'];need(number==last_cursor+1 and 1<=number<=BATCHES and counts['model_forwards']==number,'PROGRESS_CURSOR');last_cursor=number
  if number and(number%256==0 or number==BATCHES):all_tf_publish('PROGRESS_%04d.json'%number,{'phase':'ACTUAL_WHOLE_TRAIN_INITIAL','progress':event,'counts':counts,'initial_checkpoint':initial,'optimizer_updates':0})
 try:
  progress['phase']='ACTUAL_WHOLE_TRAIN_INITIAL_FORWARD';score=adapter.score_initial(model,context,batch_rows=BATCH,device='cuda',progress_callback=observe)
 finally:
  for hook in hooks:hook.remove()
 need(last_cursor==BATCHES and score['rows']==ROWS and score['batch_count']==BATCHES and score['initial_prior_exact']is True and core.state_sha(model.state_dict())==state_sha and all(p.grad is None for p in model.parameters()),'WHOLE_TRAIN_INITIAL_STATE_AND_SCORE');need(counts==ALL_LIMITS,'EXACT_COUNTS');need(torch.cuda.max_memory_reserved()<=ALL_EXECUTION['cuda_peak_reserved_bytes_max'],'CUDA_MEMORY_CAP')
 progress['phase']='COMPLETE';return {'fresh_reference_comparison':comparison,'backbone_gradient_policy':None,'whole_TRAIN_initialbaseline_qualified':True,'initial':score,'initial_state_sha256':state_sha,'initial_checkpoint':initial,'counts':counts,'runtime':runtime,'cuda_peak_reserved_bytes':int(torch.cuda.max_memory_reserved()),'process_peak_RSS_KiB':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),'genuine_inputs':'Unchanged9featurekeys/eightfamilies/M5-M15-H1-H4-D1/D1macro;targetmetadata outsideforward.','optimizer_replay_qualified':False,'learning_qualified':False,'later_generalization':'UNKNOWN','economic_edge':'UNKNOWN'}
def all_tf_run(plan_sha):
 from gx1.contracts.gx1_capped_execution_v1 import require_native_provider_hlc_all_tf_initial_execution
 began=time.monotonic()
 def stopped(*_):raise TimeoutError('PROVIDER_HLC_INITIAL_WALL_OR_SIGNAL')
 for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stopped)
 signal.setitimer(signal.ITIMER_REAL,ALL_EXECUTION['internal_seconds']);guard=require_native_provider_hlc_all_tf_initial_execution(plan_sha256=plan_sha);plan,binding,pref,live=all_tf_inspect(plan_sha,True);need(guard['plan']==binding and guard['output_root']==str(ALL_ROOT),'EXACT_GUARD');ALL_ROOT.mkdir(exist_ok=False);claim=all_tf_publish('CLAIM.json',{'plan':binding,'policy':pref,'runtime':live,'process_identity':core.future_cpu_process_identity(),'remaining_invocations':0,'relaunch_allowed':False,'training_authorized':False,'optimizer_updates':0});progress={'phase':'CLAIMED'}
 try:
  import torch
  allowed={plan['inputs'][k]['path']for k in ('feature_archive','target_arrays')};allowed.add(str(ALL_ROOT/'INITIAL.pt'))
  def audit(event,args):
   if event in ('socket.connect','socket.getaddrinfo','urllib.Request'):raise RuntimeError('PROVIDER_INITIAL_NETWORK_FORBIDDEN')
   if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
    path=os.fsdecode(args[0])
    if path.endswith(('.npz','.pt','.pth','.parquet','.safetensors','.pkl'))and str(Path(path).resolve())not in allowed:raise RuntimeError('PROVIDER_INITIAL_PAYLOAD_PATH_FORBIDDEN')
  sys.addaudithook(audit)
  def no_optimization(*args,**kwargs):raise RuntimeError('PROVIDER_INITIAL_OPTIMIZATION_FORBIDDEN')
  torch.optim.Optimizer.__init__=no_optimization;torch.Tensor.backward=no_optimization;torch.autograd.grad=no_optimization
  report=all_tf_physical(plan,binding,live,progress);all_tf_check_sources_inputs(plan);need(ref(ALL_PLAN)==binding and ref(POLICY)==pref,'POLICY_PLAN_BOOKEND');need(core.physical_windows_boot_id()==live['physical_windows_boot_id']and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==live['wsl_boot_id'],'BOOT_BOOKEND');need(core.common.firewall_state()['denied_attempts']==0,'M1_FIREWALL')
  result=all_tf_publish('RESULT.json',{'status':'COMPLETE_NATIVE_PROVIDER_HLC_ALL_TF_WHOLE_TRAIN_INITIAL','plan':binding,'claim':claim,'report':report,'source_policy_inputs_boot_unchanged':True,'test_data_used':False,'elapsed_seconds':time.monotonic()-began});terminal=all_tf_publish('TERMINAL.json',{'status':'COMPLETE','exit_code':0,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False});return {'result':result,'terminal':terminal}
 except BaseException as error:
  all_tf_publish('TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'plan':binding,'claim':claim,'error':type(error).__name__+':'+str(error)[:1800],'progress':progress,'remaining_invocations':0,'relaunch_allowed':False});raise
 finally:signal.setitimer(signal.ITIMER_REAL,0)
def main_all_tf():
 parser=argparse.ArgumentParser();parser.add_argument('--all-tf-plan-sha256',required=True);parser.add_argument('--preflight',action='store_true');a=parser.parse_args();need(sys.argv[1:]==['--all-tf-plan-sha256',a.all_tf_plan_sha256]+(['--preflight']if a.preflight else[]),'EXACT_CLI')
 if a.preflight:
  plan,binding,pref,live=all_tf_inspect(a.all_tf_plan_sha256,False);print(json.dumps({'status':'PREFLIGHT_PROVIDER_HLC_ALL_TF_WHOLE_TRAIN_INITIAL_PASS','plan':binding,'policy':pref,'cuda_initialized':False,'payloads_decoded':0,'models':0,'checkpoint_loads':0,'optimizer_updates':0}))
 else:print(json.dumps(all_tf_run(a.all_tf_plan_sha256),sort_keys=True))

# End explicit None fresh INITIAL.

# Durable all-TF INITIAL segments; previous producers remain consumed.
SEG_NAME='NATIVE_PROVIDER_HLC_ALL_TF_WHOLE_TRAIN_INITIAL_20261124_003';SEG_STAGE=SEG_NAME.lower();SEG_BASE_PLAN=REPO/'configs/research'/(SEG_NAME+'.json');SEG_ROOT=Path('/home/andre2/GX1_RUNS')/SEG_NAME
SEG_DESIGN={'path':str(E/'PROVIDER_HLC_ALL_TF_RESUMABLE_INITIAL_DESIGN001.json'),'sha256':'db02fbcd9c93ab4ddef082395ac9eb4a02d11373fa082d23dd6279ae1033fd7b'}
SEG_PARENT='9d7b39da9214f74dcf6a7b8bf1a0fd3a719da29f';SEG_EXECUTION=dict(EXECUTION)
SEG_RECIPE={'seed':SEED,'batch_rows':BATCH,'batches':BATCHES,'last_rows':LAST,'rows':ROWS,'shuffle':False,'drop_last':False,'backbone_gradient_policy':None,'no_optimization':True}
SEG_BASE_SCHEMA='gx1_native_provider_hlc_all_tf_resumable_initial_base_plan_v1';SEG_SCHEMA='gx1_native_provider_hlc_all_tf_initial_segment_plan_v1'
SEG_COUNT_KEYS=tuple(ALL_LIMITS)+('output_archive_write_calls','output_archive_readbacks','completed_batch_receipt_write_calls','committed_batches','prefix_output_archive_loads')

def segment_guard_anchor():
 return {n:hashlib.sha256(subprocess.check_output(['git','show',SEG_PARENT+':'+n],cwd=REPO)).hexdigest()for n in ('gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh')}
def segment_paths(sha,*,base_root=None):
 base_root=SEG_ROOT if base_root is None else base_root;need(base_root in (SEG_ROOT,V3_ROOT),'SEGMENT_BASE_ROOT')
 need(type(sha)is str and len(sha)==64 and all(c in '0123456789abcdef'for c in sha),'SEGMENT_SHA');return base_root/'PLANS'/(sha+'.json'),base_root/('SEGMENT_'+sha)
def segment_publish(root,name,obj):
 raw=obj if isinstance(obj,bytes)else(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();path=root/name;_publish_bytes(path,raw);return ref(path)
def segment_gate(binding,admitted):
 return {'status':'ADMITTED_EXACT_INITIAL_SCORE_SEGMENT'if admitted else'PREPARED_NOT_ADMITTED','plan':binding,'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':False,'optimizer_authorized':False,'test_authorized':False}
def segment_limits(cursor,first,*,v3=False):
 count_keys=V3_COUNT_KEYS if v3 else SEG_COUNT_KEYS
 need(type(cursor)is int and 0<=cursor<=BATCHES,'SEGMENT_CURSOR');n=BATCHES-cursor;result=dict.fromkeys(count_keys,0);result.update(model_constructions=1,feature_cache_payload_loads=1,v2_truth_archive_loads=1,truth_fields_decoded=14,model_forwards=n,backbone_forwards=n,readout_forwards=n,new_checkpoint_writes=int(first),new_checkpoint_readbacks=int(first),output_archive_write_calls=n,output_archive_readbacks=n,completed_batch_receipt_write_calls=n,committed_batches=n,prefix_output_archive_loads=cursor)
 if v3:result.update(in_memory_checkpoint_strict_loads=int(first),in_memory_output_archive_strict_loads=n)
 return result

def build_base_plan(runtime,source_commit=None):
 from gx1.contracts.gx1_capped_execution_v1 import native_provider_hlc_all_tf_initial_segment_source_bridge
 from gx1.contracts.market_state_ta_integrated_model_v1 import native_future_state_model_contract
 from gx1.contracts.market_state_ta_integrated_objective_v1 import require_native_future_state_train_prior
 scope=read(ALL_SCOPE);design=read(SEG_DESIGN);need(scope['candidate_policy']['explicit_backbone_gradient_policy']is None and design['model_contract_sha256']==ALL_CONTRACT_SHA and design['guard_parent_commit']==SEG_PARENT and design['segment_output_root_rule']=='base_root/SEGMENT_{64hex_plan_file_sha256}' and design['segment_plan_path_rule']=='base_root/PLANS/{64hex_plan_file_sha256}.json' and design['unchanged_execution']==SEG_EXECUTION and design['whole_clock_recipe']==SEG_RECIPE,'SEGMENT_DESIGN')
 core.future_cuda_require_runtime(runtime);main_all_tf=read(MAIN);old=main_all_tf['metadata'];data=read(DATASET);prior=require_native_future_state_train_prior(read(PRIOR),target_contract=data['contract']);snapshot=read(old['source_snapshot']);norm=read(old['normalization'])
 need(data['contract']==hlc.provider_hlc_state_contract()and data['base_dataset']==hlc.SOURCE_BINDINGS['base_dataset']and data['cutoff_ns']==hlc.TRAIN_UPPER_NS and data['model_input']is False,'TARGET_DATASET')
 need(data['target_observation_certificate_ref']['source_binding_sha256']==hlc.provider_hlc_state_contract()['provider_hlc_source_binding_sha256'],'CERT_SOURCE')
 need(prior['fit_population']['issue_rows']==ROWS and prior['fit_population']['issue_ns_sha256']==hlc.TRAIN_ISSUE_SHA256 and prior['fit_population']['cutoff_ns']==hlc.TRAIN_UPPER_NS,'NEW_PRIOR_POPULATION')
 meta={k:old[k]for k in ('source_snapshot','normalization','input_binding_ref','input_binding','architecture','native_manifest')};meta.update(dataset=DATASET,prior=PRIOR,target_contract=data['contract'],truth_sha256=prior['fit_population']['truth_sha256'],TRAIN_rows=ROWS,TRAIN_issue_ns_sha256=hlc.TRAIN_ISSUE_SHA256,runtime=runtime)
 need(read(meta['input_binding_ref'])==meta['input_binding']and meta['input_binding']['input_normalization_sha256']==norm['contract_sha256'],'INPUT_NORMALIZATION_BINDING')
 cert=data['target_observation_certificate_ref'];inputs={'original_MAIN_plan':MAIN,'source_snapshot':meta['source_snapshot'],'normalization':meta['normalization'],'input_binding_ref':meta['input_binding_ref'],'feature_archive':snapshot['archive'],'dataset':DATASET,'target_arrays':data['arrays'],'new_prior':PRIOR,'certificate':{k:cert[k]for k in ('path','sha256')},'provider_response':hlc.SOURCE_BINDINGS['reference_raw'],'target_acceptance':{'path':str(E/'D1_PROVIDER_HLC_TARGET_ACCEPTANCE001.json'),'sha256':'f20b79e2d37f3315157cd89ca74b2a12cb0e3c520a5a9bf3347ffbee7568981c'}}
 names=set(main_all_tf['source_owners'])|{SELF,HELPER,HELPER_TEST,GUARD_TEST,ALL_TEST,'gx1/audit/native_provider_hlc_replay_v1.py','tests/test_native_provider_hlc_replay_v1.py','tests/test_native_provider_hlc_replay_guard_v1.py','gx1/contracts/native_future_state_provider_hlc_v1.py','tests/test_native_future_state_provider_hlc_v1.py','tests/test_native_future_state_provider_hlc_consumers_v1.py'}
 headers=snapshot['arrays'];feature_bytes=sum(int(np.prod(x['shape']))*np.dtype(x['dtype']).itemsize for x in headers.values());need(len(headers)==27 and feature_bytes==1433216758,'FEATURE_ARRAY_BUDGET')
 model_contract=native_future_state_model_contract(meta['input_binding'],architecture=meta['architecture'],train_prior=prior,target_contract=meta['target_contract'],backbone_gradient_policy=None)
 reference_result=read(scope['unchanged_reference']['initial_measurement']);reference_acceptance=read(scope['unchanged_reference']['initial_acceptance']);reference_plan=read(reference_result['plan']);reference_receipt=read(reference_acceptance['initial_receipt'])
 need(design['whole_clock_recipe']=={'seed':SEED,'batch_rows':BATCH,'batches':BATCHES,'last_rows':LAST,'rows':ROWS,'shuffle':False,'drop_last':False,'backbone_gradient_policy':None,'no_optimization':True},'ALL_TF_RECIPE_PROFILE')
 need(reference_acceptance['result']==scope['unchanged_reference']['initial_measurement']and reference_acceptance['full_initialbaseline_qualified']is True and reference_result['report']['whole_TRAIN_initialbaseline_qualified']is True and reference_receipt['initial_only']is True and reference_receipt['optimizer_updates']==0 and reference_receipt['model_state_sha256']==reference_result['report']['initial_state_sha256']and reference_plan['model_contract']['backbone_gradient_policy']['policy']=='M5_primary','ALL_TF_REFERENCE_INITIAL')
 need(all(meta[k]==reference_plan['metadata'][k]for k in ('source_snapshot','normalization','input_binding','architecture','dataset','prior','target_contract','truth_sha256'))and model_contract['contract_sha256']==ALL_CONTRACT_SHA and 'backbone_gradient_policy'not in model_contract,'ALL_TF_SAME_INPUT_VALUE_FUNCTION')
 inputs.update(scope=ALL_SCOPE,resumable_design=SEG_DESIGN,predecessor_acceptance=scope['predecessor_acceptance'],predecessor_result=scope['predecessor_result'],reference_initial_acceptance=scope['unchanged_reference']['initial_acceptance'],reference_initial_result=scope['unchanged_reference']['initial_measurement'],reference_initial_plan=reference_result['plan'],reference_initial_receipt=reference_acceptance['initial_receipt'])
 return {'schema_version':'gx1_native_provider_hlc_all_tf_resumable_initial_base_plan_v1','stage':SEG_STAGE,'source_commit':source_commit or runtime['source_commit'],'source_owners':{n:ref(REPO/n)['sha256']for n in sorted(names)},'source_bridge':native_provider_hlc_all_tf_initial_segment_source_bridge(segment_guard_anchor(),producer_commit=SEG_PARENT),'inputs':inputs,'metadata':meta,'model_contract':model_contract,'execution':SEG_EXECUTION,'uninterrupted_reference_limits':ALL_LIMITS,'logical_coverage':{'rows':ROWS,'batches':BATCHES,'batch_rows':BATCH,'last_rows':LAST},'recipe':{'seed':SEED,'batch_rows':BATCH,'batches':BATCHES,'last_rows':LAST,'rows':ROWS,'shuffle':False,'drop_last':False,'backbone_gradient_policy':None,'no_optimization':True},'output_root':str(SEG_ROOT),'input_budget':{'feature_arrays':27,'feature_ndarray_bytes':feature_bytes,'v2_truth_arrays':14,'v2_truth_ndarray_bytes':50700650,'full_original_30array_target_loads':0},'reference_initial':reference_receipt,'reference_model_contract':reference_plan['model_contract'],'backbone_gradient_policy':None,'scope_ref':SEG_DESIGN,'scope':'Actual fresh explicitNone v2 NN INITIAL across all354550TRAIN issues and all5TF/allgenuineinputs; notfit,optimizerreplay,learning,generalization,calendar/vintage or edge. Analyticprior or a smallprobe cannotreplace actual fullforward coverage.'}

def segment_validate_header(plan,base,binding):
 _scope=initial_version_scope(base);SEG_BASE_PLAN,SEG_ROOT,SEG_STAGE,SEG_BASE_SCHEMA,SEG_SCHEMA,ALL_CONTRACT_SHA,DATASET,PRIOR,SEG_COUNT_KEYS=(_scope[k]for k in ('base_plan','root','stage','base_schema','schema','model_sha','dataset','prior','count_keys'))
 expected_inputs={**base['inputs'],'base_plan':binding}
 if plan['parent_terminal']is not None:expected_inputs.update(parent_terminal=plan['parent_terminal'],initial_receipt=plan['initial_receipt'])
 need(plan['inputs']==expected_inputs and {k:v for k,v in plan['metadata'].items()if k!='runtime'}=={k:v for k,v in base['metadata'].items()if k!='runtime'}and plan['source_bridge']==base['source_bridge'],'SEGMENT_SAME_INPUT_VALUE_FUNCTION')
 need(plan['schema_version']==SEG_SCHEMA and plan['stage']==SEG_STAGE and plan['base_plan']==binding and plan['base_root']==str(SEG_ROOT)and plan['recipe']==base['recipe']==SEG_RECIPE and plan['model_contract']==base['model_contract']and plan['execution']==base['execution']==SEG_EXECUTION and plan['source_owners']==base['source_owners']and 'output_root'not in plan and 'plan_path'not in plan,'SEGMENT_HEADER')
 need(plan['initial_mode']==('first_fresh'if plan['parent_terminal']is None else'fresh_compare')and (plan['parent_terminal']is not None or plan['prefix']=={'cursor':0,'rows_done':0,'head':None}and plan['initial_receipt']is None)and plan['limits']==segment_limits(plan['prefix']['cursor'],plan['initial_mode']=='first_fresh',v3=_scope['v3']),'SEGMENT_LIMITS')

def segment_closed_parent(base_ref,parent_terminal_ref):
 base=read(base_ref)
 _scope=initial_version_scope(base);SEG_BASE_PLAN,SEG_ROOT,SEG_STAGE,SEG_BASE_SCHEMA,SEG_SCHEMA,ALL_CONTRACT_SHA,DATASET,PRIOR,SEG_COUNT_KEYS=(_scope[k]for k in ('base_plan','root','stage','base_schema','schema','model_sha','dataset','prior','count_keys'))
 need(base_ref['path']==str(SEG_BASE_PLAN),'FIXED_BASE_PLAN');base=read(base_ref);need(base['schema_version']==SEG_BASE_SCHEMA and base['stage']==SEG_STAGE and base['output_root']==str(SEG_ROOT)and base['model_contract']['contract_sha256']==ALL_CONTRACT_SHA and 'backbone_gradient_policy'not in base['model_contract'],'FIXED_BASE_IDENTITY');empty={'cursor':0,'rows_done':0,'head':None}
 if parent_terminal_ref is None:return {'prefix':empty,'initial_receipt':None,'initial':None,'segments':[],'receipts':[]}
 nodes=[];seen=set();current=parent_terminal_ref
 while current is not None:
  root=Path(current['path']).parent;sha=root.name.removeprefix('SEGMENT_');pp,expected=segment_paths(sha,base_root=SEG_ROOT);need(root==expected and Path(current['path'])==root/'TERMINAL.json'and sha not in seen,'PARENT_FIXED_ROOT');seen.add(sha)
  terminal=read(current);binding={'path':str(pp),'sha256':sha};plan=read(binding);segment_validate_header(plan,base,base_ref);claim=read(terminal['claim']);closure=read(terminal['closure_proof'])
  need(terminal['status']in ('COMPLETE','FAILED_CONSUMED')and terminal['exit_code']==(0 if terminal['status']=='COMPLETE'else 1)and terminal['plan']==binding and terminal['base_plan']==base_ref and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False and terminal['claim']['path']==str(root/'CLAIM.json')and terminal['closure_proof']['path']==str(root/'CLOSURE_PROOF.json'),'CLOSED_PARENT_TERMINAL')
  need(claim['plan']==binding and claim['base_plan']==base_ref and closure['plan']==binding and closure['base_plan']==base_ref and closure['claim']==terminal['claim']and closure['policy']==claim['policy']and closure['status']=='CLOSED_SEGMENT_REUSE_PROOF_PASS'and all(closure[k]is True for k in ('source_inputs_unchanged','policy_unchanged','physical_boot_unchanged','wsl_boot_unchanged','model_RNG_grad_unchanged','M1_firewall_clear')),'CLOSED_PARENT_PROOFS')
  need(closure['prefix_before']==plan['prefix']and closure['prefix_after']==terminal['prefix']and closure['initial_receipt']==terminal['initial_receipt']and set(closure['counts'])==set(SEG_COUNT_KEYS)and all(type(n)is int and 0<=n<=plan['limits'][k]for k,n in closure['counts'].items()),'CLOSED_PARENT_COUNTS')
  need(closure['prefix_after']['cursor']>=plan['prefix']['cursor']and closure['counts']['committed_batches']==closure['prefix_after']['cursor']-plan['prefix']['cursor'],'CLOSED_PARENT_PROGRESS')
  if terminal['status']=='COMPLETE':
   completed=read(terminal['result']);need(terminal['result']['path']==str(root/'RESULT.json')and completed['status']==_scope['result_status']and completed['plan']==base_ref and completed['execution_plan']==binding and completed['claim']==terminal['claim']and completed['report']['whole_TRAIN_initialbaseline_qualified']is True and completed['report']['counts']==closure['counts']and terminal['prefix']['cursor']==BATCHES,'COMPLETE_SEGMENT_RESULT')
  declared_initial=read(closure['initial_receipt']);need(closure['model_RNG_proof']['model_state_sha256']==declared_initial['model_state_sha256']and closure['model_RNG_proof']['rng_sha256']==declared_initial['rng_sha256']and closure['model_RNG_proof']['gradients_none']is True and closure['model_RNG_proof']['fresh_constructor_no_weight_load']is True,'CLOSED_PARENT_MODEL_RNG_PROOF')
  core.future_cpu_parent_stopped(claim['process_identity']);core.future_cuda_runtime_matches(plan['metadata']['runtime'],claim['runtime'])
  nodes.append({'plan':binding,'claim':terminal['claim'],'terminal':current,'closure_proof':terminal['closure_proof'],'counts':closure['counts'],'elapsed_seconds':terminal['elapsed_seconds'],'prefix_before':plan['prefix'],'prefix_after':terminal['prefix'],'initial_receipt':terminal['initial_receipt'],'runtime':claim['runtime']});current=plan['parent_terminal']
 nodes.reverse();initial_ref=nodes[0]['initial_receipt'];initial=read(initial_ref)
 need(initial_ref['path']==str(SEG_ROOT/'INITIAL.json')and initial['plan']==base_ref and initial['checkpoint']['path']==str(SEG_ROOT/'INITIAL.pt')and initial['schema_version']=='gx1_native_provider_hlc_fresh_initial_checkpoint_v1'and initial['model_contract_sha256']==ALL_CONTRACT_SHA and initial['target_contract_sha256']==base['metadata']['target_contract']['contract_sha256']and initial['initial_only']is True and initial['optimizer_updates']==0,'PREFIX_OWN_INITIAL')
 need(ref(initial['checkpoint']['path'])==initial['checkpoint'],'PREFIX_INITIAL_CHECKPOINT_BYTES')
 previous=empty
 for node in nodes:
  need(node['prefix_before']==previous and node['initial_receipt']==initial_ref and all(node['runtime'][k]==base['metadata']['runtime'][k]for k in ('gpu','library_versions')),'PARENT_GAPFREE_VALUE_FUNCTION');previous=node['prefix_after']
 records=[];head=previous['head'];cursor=previous['cursor'];owners={n['plan']['sha256']:n for n in nodes}
 need(type(cursor)is int and 0<=cursor<=BATCHES and previous['rows_done']==min(cursor*BATCH,ROWS)and(head is None)==(cursor==0),'PREFIX_CURSOR')
 while head is not None:
  receipt=read(head);event=receipt['event'];execution=receipt['execution_plan'];pp,root=segment_paths(execution['sha256'],base_root=SEG_ROOT);node=owners.get(execution['sha256']);left=(cursor-1)*BATCH;right=min(cursor*BATCH,ROWS)
  need(node is not None and execution==node['plan']and head['path']==str(root/('BATCH_%04d.json'%cursor))and receipt['schema_version']=='gx1_native_provider_hlc_initial_score_batch_receipt_v1'and receipt['base_plan']==base_ref and receipt['claim']==node['claim']and node['prefix_before']['cursor']<cursor<=node['prefix_after']['cursor'],'PREFIX_RECEIPT_AUTHORITY')
  need(event['batch_cursor']==event['batch']['cursor']==cursor and event['left']==left and event['right']==event['rows_done']==event['batch']['rows_done']==right and event['batch']['rows']==right-left and event['rows_total']==ROWS and event['batch_count']==BATCHES and event['model_state_sha256']==initial['model_state_sha256']and event['rng_sha256']==initial['rng_sha256']and event['gradients_none']is True,'PREFIX_BATCH_PROOF')
  need(cursor!=node['prefix_after']['cursor']or head==node['prefix_after']['head'],'PREFIX_CLOSED_HEAD')
  need(receipt['archive']['path']==str(root/('BATCH_%04d.npz'%cursor))and ref(receipt['archive']['path'])==receipt['archive']and set(receipt['output_headers'])=={'activity_logits','side_logits'},'PREFIX_ARCHIVE')
  for key,header in receipt['output_headers'].items():need(header=={'dtype':'<f4','shape':[right-left,5],'sha256':event['batch']['output_sha256'][key]},'PREFIX_HEADER')
  records.append((head,receipt));head=receipt['previous_receipt'];cursor-=1
 need(cursor==0,'PREFIX_NO_HOLES');records.reverse();return {'prefix':previous,'initial_receipt':initial_ref,'initial':initial,'segments':nodes,'receipts':records}

def build_segment_plan(runtime,base_plan_ref,parent_terminal_ref=None,source_commit=None):
 base=read(base_plan_ref)
 _scope=initial_version_scope(base);SEG_BASE_PLAN,SEG_ROOT,SEG_STAGE,SEG_BASE_SCHEMA,SEG_SCHEMA,ALL_CONTRACT_SHA,DATASET,PRIOR,SEG_COUNT_KEYS=(_scope[k]for k in ('base_plan','root','stage','base_schema','schema','model_sha','dataset','prior','count_keys'))
 need(base_plan_ref['path']==str(SEG_BASE_PLAN),'FIXED_BASE_PLAN');base=read(base_plan_ref);need(base==rebuild_initial_base(base),'EXACT_BASE_PLAN');core.future_cuda_require_runtime(runtime)
 need(all(runtime[k]==base['metadata']['runtime'][k]for k in ('gpu','library_versions')),'SAME_CUDA_VALUE_FUNCTION');parent=segment_closed_parent(base_plan_ref,parent_terminal_ref);prefix=parent['prefix'];first=parent_terminal_ref is None
 inputs={**base['inputs'],'base_plan':base_plan_ref}
 if not first:inputs.update(parent_terminal=parent_terminal_ref,initial_receipt=parent['initial_receipt'])
 meta=dict(base['metadata']);meta['runtime']=runtime
 return {'schema_version':SEG_SCHEMA,'stage':SEG_STAGE,'source_commit':source_commit or runtime['source_commit'],'source_owners':base['source_owners'],'source_bridge':base['source_bridge'],'base_plan':base_plan_ref,'base_root':str(SEG_ROOT),'parent_terminal':parent_terminal_ref,'prefix':prefix,'initial_mode':'first_fresh'if first else'fresh_compare','initial_receipt':parent['initial_receipt'],'metadata':meta,'model_contract':base['model_contract'],'recipe':SEG_RECIPE,'execution':SEG_EXECUTION,'limits':segment_limits(prefix['cursor'],first,v3=_scope['v3']),'inputs':inputs,'authority':'ZERO_UPDATE_ACTUAL_INITIAL_SCORE_ONLY_NO_FIT_OR_RELAUNCH'}

def segment_check_sources_inputs(plan):
 check_sources_inputs(plan);base=read(plan['base_plan']);need(all(ref(REPO/n)['sha256']==d for n,d in base['source_owners'].items()),'BASE_SOURCES_FROZEN');segment_closed_parent(plan['base_plan'],plan['parent_terminal'])

def inspect_segment(sha,admitted,*,v3=False):
 pp,root=segment_paths(sha,base_root=V3_ROOT if v3 else SEG_ROOT);binding={'path':str(pp),'sha256':sha};plan=read(binding);scope=initial_version_scope(plan);need(scope['v3']is v3,'EXACT_SEGMENT_CLI_GENERATION');need(plan==build_segment_plan(plan['metadata']['runtime'],plan['base_plan'],plan['parent_terminal'],plan['source_commit']),'EXACT_SEGMENT_PLAN');segment_check_sources_inputs(plan);pref=ref(POLICY);policy=read(pref);validate_flags(policy)
 if v3:need(policy.get('exit_training_and_evaluation_this_wave')is False,'V3_EXIT_GATE_CLOSED')
 need(policy[scope['stage']]==segment_gate(binding,admitted),'ONE_USE_SEGMENT_GATE');need(not root.exists()and not root.is_symlink()and not(scope['root']/'RESULT.json').exists(),'FRESH_SEGMENT_ROOT');first=plan['initial_mode']=='first_fresh';need(first==(not(scope['root']/'INITIAL.json').exists())and(first or plan['initial_receipt']is not None),'EXPLICIT_FIRST_OR_RESUME');need(not first or not(scope['root']/'INITIAL.pt').exists(),'FRESH_INITIAL_PATH');live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['metadata']['runtime'],live);return plan,binding,pref,live

def segment_publish_batch(plan,binding,claim,event,outputs,previous_receipt,*,counts=None):
 _scope=initial_version_scope(plan);SEG_BASE_PLAN,SEG_ROOT,SEG_STAGE,SEG_BASE_SCHEMA,SEG_SCHEMA,ALL_CONTRACT_SHA,DATASET,PRIOR,SEG_COUNT_KEYS=(_scope[k]for k in ('base_plan','root','stage','base_schema','schema','model_sha','dataset','prior','count_keys'))
 root=segment_paths(binding['sha256'],base_root=SEG_ROOT)[1];cursor=event['batch_cursor'];need(plan['prefix']['cursor']<cursor<=BATCHES and set(outputs)=={'activity_logits','side_logits'}and event['gradients_none']is True,'COMMIT_DOMAIN');headers={}
 for key,value in outputs.items():
  need(isinstance(value,np.ndarray)and value.dtype==np.float32 and value.shape==(event['right']-event['left'],5)and np.isfinite(value).all()and core.digest(value)==event['batch']['output_sha256'][key],'COMMIT_OUTPUT');headers[key]={'dtype':value.dtype.str,'shape':list(value.shape),'sha256':core.digest(value)}
 stream=io.BytesIO();np.savez(stream,**outputs);serialized=stream.getvalue()
 if _scope['v3']:
  if counts is not None:counts['in_memory_output_archive_strict_loads']+=1
  with np.load(io.BytesIO(serialized),allow_pickle=False)as checked:
   need(set(checked.files)==set(headers),'PREPUBLICATION_OUTPUT_INVENTORY')
   for key,header in headers.items():
    value=checked[key];need(value.dtype.str==header['dtype']and list(value.shape)==header['shape']and core.digest(value)==header['sha256'],'PREPUBLICATION_OUTPUT_BYTES')
 if counts is not None:counts['output_archive_write_calls']+=1
 archive=segment_publish(root,'BATCH_%04d.npz'%cursor,serialized);stream.close()
 if counts is not None:counts['output_archive_readbacks']+=1
 with np.load(archive['path'],allow_pickle=False)as saved:
  need(set(saved.files)==set(outputs),'OUTPUT_READBACK_KEYS')
  for key,header in headers.items():
   value=saved[key];need(value.dtype.str==header['dtype']and list(value.shape)==header['shape']and core.digest(value)==header['sha256'],'OUTPUT_READBACK_PROOF')
 receipt={'schema_version':'gx1_native_provider_hlc_initial_score_batch_receipt_v1','base_plan':plan['base_plan'],'execution_plan':binding,'claim':claim,'previous_receipt':previous_receipt,'archive':archive,'output_headers':headers,'event':event}
 if counts is not None:counts['completed_batch_receipt_write_calls']+=1
 result=segment_publish(root,'BATCH_%04d.json'%cursor,receipt);need(read(result)==receipt,'COMMIT_RECEIPT_READBACK')
 if counts is not None:counts['committed_batches']+=1
 return result

def load_segment_prefix(plan,context,*,counts=None):
 parent=segment_closed_parent(plan['base_plan'],plan['parent_terminal']);rows=parent['prefix']['rows_done'];outputs={k:np.empty((rows,5),dtype=np.float32)for k in ('activity_logits','side_logits')};batches=[]
 for _,receipt in parent['receipts']:
  event=receipt['event'];left,right=event['left'],event['right'];need(core.digest(context['truth']['issue_ns'][left:right])==event['issue_ns_sha256']and core.digest(context['source_rows'][left:right])==event['source_rows_sha256'],'PREFIX_PHYSICAL_CLOCK')
  if counts is not None:counts['prefix_output_archive_loads']+=1
  with np.load(receipt['archive']['path'],allow_pickle=False)as saved:
   need(set(saved.files)==set(outputs),'PREFIX_ARRAY_KEYS')
   for key,header in receipt['output_headers'].items():
    value=saved[key];need(value.dtype.str==header['dtype']and list(value.shape)==header['shape']and core.digest(value)==header['sha256'],'PREFIX_ARRAY_PROOF');outputs[key][left:right]=value
  batches.append(event['batch'])
 return {'rows_done':rows,'batches':batches,'outputs':outputs}

def segment_compare_fresh(model,initial_receipt,*,model_contract_sha256=None):
 contract_sha=ALL_CONTRACT_SHA if model_contract_sha256 is None else model_contract_sha256
 need(model.research_objective['contract_sha256']==contract_sha and 'backbone_gradient_policy'not in model.research_objective and core.state_sha(model.state_dict())==initial_receipt['model_state_sha256']and core.state_sha(core.training_rng())==initial_receipt['rng_sha256']and all(p.grad is None for p in model.parameters())and model.training is True,'SEGMENT_FRESH_SAME_STATE_RNG');return {'model_state_sha256':initial_receipt['model_state_sha256'],'rng_sha256':initial_receipt['rng_sha256'],'gradients_none':True,'fresh_constructor_no_weight_load':True,'constructor_training_mode_restored':True}

def segment_write_initial_checkpoint(model,base,binding,live,counts):
 _scope=initial_version_scope(base);SEG_BASE_PLAN,SEG_ROOT,SEG_STAGE,SEG_BASE_SCHEMA,SEG_SCHEMA,ALL_CONTRACT_SHA,DATASET,PRIOR,SEG_COUNT_KEYS=(_scope[k]for k in ('base_plan','root','stage','base_schema','schema','model_sha','dataset','prior','count_keys'))
 import torch
 state={k:x.detach().cpu().clone()for k,x in model.state_dict().items()};rng=core.training_rng();cp={'schema_version':'gx1_native_provider_hlc_fresh_initial_checkpoint_v1','plan':binding,'model_contract':base['model_contract'],'input_binding':base['metadata']['input_binding'],'normalization':base['metadata']['normalization'],'prior':PRIOR,'dataset':DATASET,'target_contract':base['metadata']['target_contract'],'truth_sha256':base['metadata']['truth_sha256'],'seed':SEED,'runtime':live,'model_state':state,'model_state_sha256':core.state_sha(state),'rng_state':rng,'training':model.training,'optimizer':None,'optimizer_updates':0,'authority':'INITIAL_ONLY_NO_FIT_OR_RESUME_AUTHORITY'}
 stream=io.BytesIO();torch.save(cp,stream);serialized=stream.getvalue()
 if _scope['v3']:
  counts['in_memory_checkpoint_strict_loads']+=1;memory=torch.load(io.BytesIO(serialized),map_location='cpu',weights_only=True);need(set(memory)==set(cp)and core.state_sha(memory)==core.state_sha(cp),'PREPUBLICATION_INITIAL_FULL_INVENTORY_STATE_RNG')
 checkpoint=segment_publish(SEG_ROOT,'INITIAL.pt',serialized);stream.close();counts['new_checkpoint_writes']+=1;checked=torch.load(checkpoint['path'],map_location='cpu',weights_only=True);counts['new_checkpoint_readbacks']+=1;need(core.state_sha(checked)==core.state_sha(cp)and core.state_sha(model.state_dict())==cp['model_state_sha256']and core.state_sha(core.training_rng())==core.state_sha(rng),'SEGMENT_INITIAL_FULL_TREE_READBACK')
 return segment_publish(SEG_ROOT,'INITIAL.json',{'schema_version':cp['schema_version'],'plan':binding,'checkpoint':checkpoint,'checkpoint_tree_sha256':core.state_sha(cp),'model_state_sha256':cp['model_state_sha256'],'rng_sha256':core.state_sha(rng),'model_contract_sha256':base['model_contract']['contract_sha256'],'target_contract_sha256':base['metadata']['target_contract']['contract_sha256'],'optimizer_updates':0,'initial_only':True})

def segment_physical(plan,binding,claim,live,progress):
 _scope=initial_version_scope(plan);SEG_BASE_PLAN,SEG_ROOT,SEG_STAGE,SEG_BASE_SCHEMA,SEG_SCHEMA,ALL_CONTRACT_SHA,DATASET,PRIOR,SEG_COUNT_KEYS=(_scope[k]for k in ('base_plan','root','stage','base_schema','schema','model_sha','dataset','prior','count_keys'))
 import torch
 from gx1.audit import native_future_state_provider_hlc_baseline_v1 as adapter
 from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
 base=read(plan['base_plan']);counts=dict.fromkeys(SEG_COUNT_KEYS,0);progress.update(phase='LOAD_UNCHANGED_INPUTS',counts=counts,prefix=dict(plan['prefix']),initial_receipt=plan['initial_receipt']);context=adapter.load_context(plan['metadata']);counts.update(feature_cache_payload_loads=1,v2_truth_archive_loads=1,truth_fields_decoded=14);prefix=load_segment_prefix(plan,context,counts=counts);core.future_cuda_seed_and_runtime({'recipe':SEG_RECIPE,'metadata':{'runtime':live}})
 model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=plan['metadata']['input_binding'],architecture=plan['metadata']['architecture'],device=torch.device('cuda'),train_prior=context['prior'],target_contract=plan['metadata']['target_contract'],backbone_gradient_policy=None);counts['model_constructions']=1;need(model.research_objective==base['model_contract'],'SEGMENT_MODEL_IDENTITY')
 if plan['initial_mode']=='first_fresh':
  (None if _scope['v3']else all_tf_compare_fresh_reference(model,base));progress['initial_receipt']=segment_write_initial_checkpoint(model,base,plan['base_plan'],live,counts)
 initial=read(progress['initial_receipt']);fresh=segment_compare_fresh(model,initial,model_contract_sha256=ALL_CONTRACT_SHA);hooks=[]
 def counter(key):
  def tick(*args):counts[key]+=1;need(counts[key]<=plan['limits'][key],'SEGMENT_FORWARD_CAP')
  return tick
 hooks=[model.register_forward_pre_hook(counter('model_forwards')),model.backbone.register_forward_pre_hook(counter('backbone_forwards')),model.readout.register_forward_pre_hook(counter('readout_forwards'))]
 def commit(event,outputs):
  need(event['batch_cursor']==progress['prefix']['cursor']+1 and all(counts[k]==event['batch_cursor']-plan['prefix']['cursor']for k in ('model_forwards','backbone_forwards','readout_forwards')),'ACTUAL_COMMITTED_FORWARD_COUNT');head=segment_publish_batch(plan,binding,claim,event,outputs,progress['prefix']['head'],counts=counts);progress['prefix']={'cursor':event['batch_cursor'],'rows_done':event['rows_done'],'head':head}
 try:
  progress['phase']='ACTUAL_FIXED_CLOCK_INITIAL';score=adapter.score_initial_segmented(model,context,batch_rows=BATCH,device='cuda',completed_prefix=prefix,commit_batch=commit,progress_callback=lambda event:progress.update(last_batch=event))
  need(counts==plan['limits']and progress['prefix']['cursor']==BATCHES and score['actual_NN_forward_rows']==ROWS and score['initial_prior_exact']is True,'SEGMENT_COMPLETE_COUNTS_COVERAGE');need(torch.cuda.max_memory_reserved()<=SEG_EXECUTION['cuda_peak_reserved_bytes_max'],'SEGMENT_VRAM_CAP');progress['score']=score;return {'initial':score,'fresh_constructor':fresh,'cuda_peak_reserved_bytes':int(torch.cuda.max_memory_reserved()),'process_peak_RSS_KiB':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
 finally:
  for hook in hooks:hook.remove()
  progress['model_RNG_proof']={**segment_compare_fresh(model,initial,model_contract_sha256=ALL_CONTRACT_SHA),'model_RNG_grad_unchanged':True}

def segment_durable_prefix(plan,binding,claim,initial_ref):
 _scope=initial_version_scope(plan);SEG_BASE_PLAN,SEG_ROOT,SEG_STAGE,SEG_BASE_SCHEMA,SEG_SCHEMA,ALL_CONTRACT_SHA,DATASET,PRIOR,SEG_COUNT_KEYS=(_scope[k]for k in ('base_plan','root','stage','base_schema','schema','model_sha','dataset','prior','count_keys'))
 prefix=dict(plan['prefix']);root=segment_paths(binding['sha256'],base_root=SEG_ROOT)[1];initial=read(initial_ref)
 for cursor in range(prefix['cursor']+1,BATCHES+1):
  path=root/('BATCH_%04d.json'%cursor)
  if not path.exists():break
  binding_ref=ref(path);receipt=read(binding_ref);event=receipt['event'];left=(cursor-1)*BATCH;right=min(cursor*BATCH,ROWS)
  need(receipt['schema_version']=='gx1_native_provider_hlc_initial_score_batch_receipt_v1'and receipt['base_plan']==plan['base_plan']and receipt['execution_plan']==binding and receipt['claim']==claim and receipt['previous_receipt']==prefix['head']and event['batch_cursor']==event['batch']['cursor']==cursor and event['left']==left and event['right']==event['rows_done']==event['batch']['rows_done']==right and event['batch']['rows']==right-left and event['batch_count']==BATCHES and event['rows_total']==ROWS and event['model_state_sha256']==initial['model_state_sha256']and event['rng_sha256']==initial['rng_sha256']and event['gradients_none']is True,'DURABLE_RECEIPT_PROOF')
  need(receipt['archive']['path']==str(root/('BATCH_%04d.npz'%cursor))and ref(receipt['archive']['path'])==receipt['archive']and set(receipt['output_headers'])=={'activity_logits','side_logits'},'DURABLE_ARCHIVE')
  for key,header in receipt['output_headers'].items():need(header=={'dtype':'<f4','shape':[right-left,5],'sha256':event['batch']['output_sha256'][key]},'DURABLE_ARRAY_HEADER')
  prefix={'cursor':cursor,'rows_done':right,'head':binding_ref}
 return prefix

def segment_bookends(plan,binding,pref,live,progress,claim):
 _scope=initial_version_scope(plan);SEG_BASE_PLAN,SEG_ROOT,SEG_STAGE,SEG_BASE_SCHEMA,SEG_SCHEMA,ALL_CONTRACT_SHA,DATASET,PRIOR,SEG_COUNT_KEYS=(_scope[k]for k in ('base_plan','root','stage','base_schema','schema','model_sha','dataset','prior','count_keys'))
 errors=[];flags={}
 if progress.get('initial_receipt')is not None and 'counts'in progress:
  try:progress['prefix']=segment_durable_prefix(plan,binding,claim,progress['initial_receipt']);progress['counts']['committed_batches']=progress['prefix']['cursor']-plan['prefix']['cursor']
  except BaseException as error:errors.append('durable_prefix:'+type(error).__name__+':'+str(error)[:500])
 for key,check in (('source_inputs_unchanged',lambda:(need(ref(segment_paths(binding['sha256'],base_root=SEG_ROOT)[0])==binding,'SEGMENT_PLAN_BOOKEND'),segment_check_sources_inputs(plan))),('policy_unchanged',lambda:need(ref(POLICY)==pref,'SEGMENT_POLICY_BOOKEND')),('physical_boot_unchanged',lambda:need(core.physical_windows_boot_id()==live['physical_windows_boot_id'],'SEGMENT_PHYSICAL_BOOT')),('wsl_boot_unchanged',lambda:need(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==live['wsl_boot_id'],'SEGMENT_WSL_BOOT')),('M1_firewall_clear',lambda:need(core.common.firewall_state()['denied_attempts']==0,'SEGMENT_M1_FIREWALL'))):
  try:check();flags[key]=True
  except BaseException as error:flags[key]=False;errors.append(key+':'+type(error).__name__+':'+str(error)[:500])
 flags['model_RNG_grad_unchanged']=progress.get('model_RNG_proof',{}).get('model_RNG_grad_unchanged')is True
 return {'schema_version':'gx1_native_provider_hlc_initial_segment_closure_v1','status':'CLOSED_SEGMENT_REUSE_PROOF_PASS'if all(flags.values())and not errors and progress.get('initial_receipt')else'CLOSED_SEGMENT_REUSE_PROOF_FAILED','plan':binding,'base_plan':plan['base_plan'],'claim':claim,'policy':pref,'prefix_before':plan['prefix'],'prefix_after':progress.get('prefix',plan['prefix']),'initial_receipt':progress.get('initial_receipt'),'counts':progress.get('counts',dict.fromkeys(SEG_COUNT_KEYS,0)),'model_RNG_proof':progress.get('model_RNG_proof'),'elapsed_before_bookends_seconds':progress.get('elapsed_seconds'),'errors':errors,**flags}

def run_segment(sha,*,v3=False):
 if v3:
  from gx1.contracts.gx1_capped_execution_v1 import require_native_all_tf_provider_hlc_v3_initial_segment_execution as require_execution
 else:
  from gx1.contracts.gx1_capped_execution_v1 import require_native_provider_hlc_all_tf_initial_segment_execution as require_execution
 began=time.monotonic()
 def stopped(*_):raise TimeoutError('INITIAL_SEGMENT_WALL_OR_SIGNAL')
 for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stopped)
 signal.setitimer(signal.ITIMER_REAL,SEG_EXECUTION['internal_seconds']);guard=require_execution(plan_sha256=sha);plan,binding,pref,live=inspect_segment(sha,True,v3=v3);_scope=initial_version_scope(plan);SEG_ROOT=_scope['root'];SEG_COUNT_KEYS=_scope['count_keys'];root=segment_paths(sha,base_root=SEG_ROOT)[1];need(guard['plan']==binding and guard['output_root']==str(root),'SEGMENT_EXACT_GUARD');root.mkdir(exist_ok=False);claim=segment_publish(root,'CLAIM.json',{'plan':binding,'base_plan':plan['base_plan'],'policy':pref,'runtime':live,'process_identity':core.future_cpu_process_identity(),'remaining_invocations':0,'relaunch_allowed':False,'training_authorized':False});progress={'phase':'CLAIMED','prefix':plan['prefix'],'initial_receipt':plan['initial_receipt']};closure_ref=None;execution_result_ref=None;terminal_ref=None
 try:
  import torch
  parent=segment_closed_parent(plan['base_plan'],plan['parent_terminal']);allowed={plan['inputs'][k]['path']for k in ('feature_archive','target_arrays')}|{r['archive']['path']for _,r in parent['receipts']}|{str(root/('BATCH_%04d.npz'%j))for j in range(plan['prefix']['cursor']+1,BATCHES+1)}|{str(SEG_ROOT/'INITIAL.pt')}
  def audit(event,args):
   if event in ('socket.connect','socket.getaddrinfo','urllib.Request'):raise RuntimeError('INITIAL_SEGMENT_NETWORK_FORBIDDEN')
   if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
    path=os.fsdecode(args[0])
    if path.endswith(('.npz','.pt','.pth','.parquet','.safetensors','.pkl'))and str(Path(path).resolve())not in allowed:raise RuntimeError('INITIAL_SEGMENT_PAYLOAD_PATH_FORBIDDEN')
  sys.addaudithook(audit)
  def no_optimization(*args,**kwargs):raise RuntimeError('INITIAL_SEGMENT_OPTIMIZATION_FORBIDDEN')
  torch.optim.Optimizer.__init__=no_optimization;torch.Tensor.backward=no_optimization;torch.autograd.grad=no_optimization
  measurement=segment_physical(plan,binding,claim,live,progress);progress['elapsed_seconds']=time.monotonic()-began;closure=segment_bookends(plan,binding,pref,live,progress,claim);closure_ref=segment_publish(root,'CLOSURE_PROOF.json',closure);need(closure['status']=='CLOSED_SEGMENT_REUSE_PROOF_PASS','SEGMENT_COMPLETION_BOOKENDS')
  parent=segment_closed_parent(plan['base_plan'],plan['parent_terminal']);history=parent['segments'];total={k:sum(node['counts'][k]for node in history)+progress['counts'][k]for k in SEG_COUNT_KEYS};initial=read(progress['initial_receipt']);segments=[{k:node[k]for k in ('plan','claim','terminal','closure_proof','counts')}for node in history]+[{'plan':binding,'claim':claim,'closure_proof':closure_ref,'counts':progress['counts']}]
  report={**measurement,'whole_TRAIN_initialbaseline_qualified':True,'initial_checkpoint':progress['initial_receipt'],'initial_state_sha256':initial['model_state_sha256'],'initial_rng_sha256':initial['rng_sha256'],'backbone_gradient_policy':None,'unique_complete_batches':BATCHES,'actual_NN_forward_rows':ROWS,'counts':progress['counts'],'physical_execution_counts':total,'segments':segments,'physical_elapsed_seconds_across_segments':sum(node['elapsed_seconds']for node in history)+time.monotonic()-began,'source_inputs_unchanged_across_segments':True,'each_segment_policy_boot_unchanged':True,'optimizer_replay_qualified':False,'learning_qualified':False,'later_generalization':'UNKNOWN','economic_edge':'UNKNOWN'}
  result_obj={'status':_scope['result_status'],'plan':plan['base_plan'],'execution_plan':binding,'claim':claim,'report':report,'elapsed_seconds':time.monotonic()-began,'source_inputs_unchanged_across_segments':True,'each_segment_policy_boot_unchanged':True,'test_data_used':False}
  execution_result_ref=segment_publish(root,'RESULT.json',result_obj)
  terminal_ref=segment_publish(root,'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'plan':binding,'base_plan':plan['base_plan'],'claim':claim,'result':execution_result_ref,'closure_proof':closure_ref,'elapsed_seconds':time.monotonic()-began,'prefix':progress['prefix'],'initial_receipt':progress['initial_receipt'],'remaining_invocations':0,'relaunch_allowed':False})
  report={**report,'segments':segments[:-1]+[{**segments[-1],'terminal':terminal_ref,'result':execution_result_ref}]}
  result=segment_publish(SEG_ROOT,'RESULT.json',{**result_obj,'report':report,'completed_execution_result':execution_result_ref,'completed_execution_terminal':terminal_ref})
  return {'result':result,'terminal':terminal_ref}
 except BaseException as error:
  if closure_ref is None:
   progress['elapsed_seconds']=time.monotonic()-began;closure=segment_bookends(plan,binding,pref,live,progress,claim);closure_ref=segment_publish(root,'CLOSURE_PROOF.json',closure)
  # A COMPLETE terminal is immutable even if later aggregate publication fails.
  if (root/'TERMINAL.json').exists():
   terminal_ref=ref(root/'TERMINAL.json');segment_publish(root,'PUBLICATION_FAILURE.json',{'plan':binding,'base_plan':plan['base_plan'],'terminal':terminal_ref,'execution_result':execution_result_ref,'aggregate_RESULT_present':(SEG_ROOT/'RESULT.json').exists(),'error':type(error).__name__+':'+str(error)[:1800],'remaining_invocations':0,'relaunch_allowed':False})
  else:
   segment_publish(root,'TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'plan':binding,'base_plan':plan['base_plan'],'claim':claim,'closure_proof':closure_ref,'elapsed_seconds':time.monotonic()-began,'prefix':progress.get('prefix',plan['prefix']),'initial_receipt':progress.get('initial_receipt'),'error':type(error).__name__+':'+str(error)[:1800],'progress':progress,'remaining_invocations':0,'relaunch_allowed':False})
  raise
 finally:signal.setitimer(signal.ITIMER_REAL,0)

def main_segment():
 parser=argparse.ArgumentParser();parser.add_argument('--all-tf-score-segment-plan-sha256',required=True);parser.add_argument('--preflight',action='store_true');a=parser.parse_args();need(sys.argv[1:]==['--all-tf-score-segment-plan-sha256',a.all_tf_score_segment_plan_sha256]+(['--preflight']if a.preflight else[]),'EXACT_SEGMENT_CLI')
 if a.preflight:
  plan,binding,pref,live=inspect_segment(a.all_tf_score_segment_plan_sha256,False);print(json.dumps({'status':'PREFLIGHT_PROVIDER_HLC_ALL_TF_INITIAL_SEGMENT_PASS','plan':binding,'base_plan':plan['base_plan'],'policy':pref,'prefix':plan['prefix'],'claim_created':False,'cuda_initialized':False,'payloads_decoded':0,'models':0,'checkpoint_loads':0,'optimizer_updates':0}))
 else:print(json.dumps(run_segment(a.all_tf_score_segment_plan_sha256),sort_keys=True))

# End durable all-TF INITIAL segments.

# Exact V3 scope; the consumed V2 durable worker keeps its original defaults.
V3_COUNT_KEYS=SEG_COUNT_KEYS+('in_memory_checkpoint_strict_loads','in_memory_output_archive_strict_loads')
V3_NAME='NATIVE_ALL_TF_PROVIDER_HLC_V3_WHOLE_TRAIN_INITIAL_20261125_001';V3_STAGE=V3_NAME.lower();V3_BASE_PLAN=REPO/'configs/research'/(V3_NAME+'.json');V3_ROOT=Path('/home/andre2/GX1_RUNS')/V3_NAME
V3_PARENT='8fd74a76cc45ac7306da4b905c82e9cd9b12976d';V3_BASE_SCHEMA='gx1_native_all_tf_provider_hlc_v3_resumable_initial_base_plan_v1';V3_SEG_SCHEMA='gx1_native_all_tf_provider_hlc_v3_initial_segment_plan_v1';V3_RESULT_STATUS='COMPLETE_NATIVE_ALL_TF_PROVIDER_HLC_V3_SEGMENTED_WHOLE_TRAIN_INITIAL'
def initial_version_scope(value=None):
 from types import MappingProxyType
 schema=None if value is None else value.get('schema_version');v3=schema in (V3_BASE_SCHEMA,V3_SEG_SCHEMA)
 need(value is None or v3 or schema in (SEG_BASE_SCHEMA,SEG_SCHEMA),'INITIAL_EXACT_GENERATION')
 if v3:
  hlc.require_all_tf_provider_hlc_state_contract(value['metadata']['target_contract']);need(value['stage']==V3_STAGE and value.get('output_root',value.get('base_root'))==str(V3_ROOT),'V3_FIXED_SCOPE')
 return MappingProxyType({'v3':v3,'base_plan':V3_BASE_PLAN if v3 else SEG_BASE_PLAN,'root':V3_ROOT if v3 else SEG_ROOT,'stage':V3_STAGE if v3 else SEG_STAGE,'base_schema':V3_BASE_SCHEMA if v3 else SEG_BASE_SCHEMA,'schema':V3_SEG_SCHEMA if v3 else SEG_SCHEMA,'model_sha':value['model_contract']['contract_sha256']if v3 else ALL_CONTRACT_SHA,'count_keys':V3_COUNT_KEYS if v3 else SEG_COUNT_KEYS,'dataset':value['metadata']['dataset']if v3 else DATASET,'prior':value['metadata']['prior']if v3 else PRIOR,'result_status':V3_RESULT_STATUS if v3 else'COMPLETE_NATIVE_PROVIDER_HLC_ALL_TF_SEGMENTED_WHOLE_TRAIN_INITIAL'})

def v3_guard_anchor():
 return {n:hashlib.sha256(subprocess.check_output(['git','show',V3_PARENT+':'+n],cwd=REPO)).hexdigest()for n in ('gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh')}

def v3_truth_input_budget(data,target):
 from gx1.audit.native_future_state_training_v1 import provider_hlc_truth_fields
 fields=provider_hlc_truth_fields(target);headers=data['array_headers'];base=read(hlc.ALL_TF_SOURCE_BINDINGS['base_dataset'])
 expected=set(base['array_headers'])|{'provider_hlc_observed','state_hlc_observed'}
 need(len(fields)==14 and set(fields)<=set(headers) and set(headers)==expected,'V3_TRUTH_ARCHIVE_AND_CONSUMED_FIELDS')
 size=sum(int(np.prod(headers[k]['shape']))*np.dtype(headers[k]['dtype']).itemsize for k in fields)
 return {'V3_truth_arrays':len(fields),'V3_truth_ndarray_bytes':size,'V3_target_archive_fields':len(headers)}

def build_v3_base_plan(runtime,*,dataset_ref,prior_ref,target_acceptance_ref,source_commit=None):
 from gx1.contracts.gx1_capped_execution_v1 import native_all_tf_provider_hlc_v3_initial_source_bridge
 from gx1.contracts.market_state_ta_integrated_model_v1 import native_future_state_model_contract
 from gx1.contracts.market_state_ta_integrated_objective_v1 import require_native_future_state_train_prior
 core.future_cuda_require_runtime(runtime);main=read(MAIN);old=main['metadata'];data=read(dataset_ref);target=hlc.require_all_tf_provider_hlc_state_contract(data['contract']);prior=require_native_future_state_train_prior(read(prior_ref),target_contract=target);acceptance=read(target_acceptance_ref)
 need(acceptance['status']=='PASS_ALL_TF_PROVIDER_HLC_TARGET_AND_PRIOR_QUALIFICATION'and acceptance['dataset']==dataset_ref and acceptance['prior']==prior_ref,'V3_TARGET_PRIOR_ACCEPTANCE')
 need(data['schema_version']==hlc.ALL_TF_SCHEMA and data['base_dataset']==hlc.ALL_TF_SOURCE_BINDINGS['base_dataset']and data['base_targets']==hlc.ALL_TF_SOURCE_BINDINGS['base_targets']and data['native_manifest']==hlc.ALL_TF_SOURCE_BINDINGS['native_manifest']and data['cutoff_ns']==hlc.TRAIN_UPPER_NS and data['model_input']is False,'V3_TARGET_DATASET')
 cert=hlc.require_all_tf_provider_hlc_certificate_ref(data['target_observation_certificate_ref']);pop=prior['fit_population'];need(pop['issue_rows']==ROWS and pop['issue_ns_sha256']==hlc.TRAIN_ISSUE_SHA256 and pop['cutoff_ns']==hlc.TRAIN_UPPER_NS,'V3_NEW_PRIOR_POPULATION')
 snapshot=read(old['source_snapshot']);norm=read(old['normalization']);meta={k:old[k]for k in ('source_snapshot','normalization','input_binding_ref','input_binding','architecture','native_manifest')};meta.update(dataset=dataset_ref,prior=prior_ref,target_contract=target,truth_sha256=pop['truth_sha256'],TRAIN_rows=ROWS,TRAIN_issue_ns_sha256=hlc.TRAIN_ISSUE_SHA256,runtime=runtime)
 need(read(meta['input_binding_ref'])==meta['input_binding']and meta['input_binding']['input_normalization_sha256']==norm['contract_sha256'],'V3_UNCHANGED_INPUT_NORMALIZATION_BINDING')
 inputs={'original_MAIN_plan':MAIN,'source_snapshot':meta['source_snapshot'],'normalization':meta['normalization'],'input_binding_ref':meta['input_binding_ref'],'feature_archive':snapshot['archive'],'dataset':dataset_ref,'target_arrays':data['arrays'],'new_prior':prior_ref,'certificate':{k:cert[k]for k in ('path','sha256')},'target_acceptance':target_acceptance_ref,**{k:hlc.ALL_TF_SOURCE_BINDINGS[k]for k in ('provider_manifest','restoration_result','m5_raw_integrity_acceptance')}}
 provider_manifest=read(hlc.ALL_TF_SOURCE_BINDINGS['provider_manifest']);provider_raw=[item['response']for entries in provider_manifest['provider_responses'].values()for item in entries];need(len(provider_raw)==len({x['path']for x in provider_raw})==160,'V3_CERTIFICATE_RESPONSE_INVENTORY');inputs.update({'provider_raw_response_%03d'%j:value for j,value in enumerate(provider_raw)})
 names=set(main['source_owners'])|{SELF,HELPER,HELPER_TEST,GUARD_TEST,ALL_TEST,'tests/test_native_future_state_provider_hlc_v3_adapters.py','tests/test_native_all_tf_provider_hlc_v3_guard_v1.py','gx1/audit/native_provider_hlc_replay_v1.py','tests/test_native_provider_hlc_replay_v1.py','tests/test_native_provider_hlc_replay_guard_v1.py','gx1/contracts/native_future_state_provider_hlc_v1.py','tests/test_native_future_state_provider_hlc_all_tf_v3.py'}
 headers=snapshot['arrays'];feature_bytes=sum(int(np.prod(x['shape']))*np.dtype(x['dtype']).itemsize for x in headers.values());need(len(headers)==27 and feature_bytes==1433216758,'V3_UNCHANGED_FEATURE_ARRAY_BUDGET');truth_budget=v3_truth_input_budget(data,target)
 model_contract=native_future_state_model_contract(meta['input_binding'],architecture=meta['architecture'],train_prior=prior,target_contract=target,backbone_gradient_policy=None)
 return {'schema_version':V3_BASE_SCHEMA,'stage':V3_STAGE,'source_commit':source_commit or runtime['source_commit'],'source_owners':{n:ref(REPO/n)['sha256']for n in sorted(names)},'source_bridge':native_all_tf_provider_hlc_v3_initial_source_bridge(v3_guard_anchor(),producer_commit=V3_PARENT),'inputs':inputs,'metadata':meta,'model_contract':model_contract,'execution':SEG_EXECUTION,'uninterrupted_reference_limits':ALL_LIMITS,'logical_coverage':{'rows':ROWS,'batches':BATCHES,'batch_rows':BATCH,'last_rows':LAST},'recipe':dict(SEG_RECIPE),'output_root':str(V3_ROOT),'input_budget':{'feature_arrays':27,'feature_ndarray_bytes':feature_bytes,**truth_budget,'certificate_response_files':len(provider_raw),'certificate_response_bytes':sum(Path(x['path']).stat().st_size for x in provider_raw)},'backbone_gradient_policy':None,'scope':'OwnfreshnewpriorV3INITIAL;unchangednativefeatures/norm;all354550TRAINactualNNforwards/zerooptimization;noV2tensorreferenceoroldCPinitializer.'}

def rebuild_initial_base(base):
 if base.get('schema_version')==V3_BASE_SCHEMA:
  return build_v3_base_plan(base['metadata']['runtime'],dataset_ref=base['metadata']['dataset'],prior_ref=base['metadata']['prior'],target_acceptance_ref=base['inputs']['target_acceptance'],source_commit=base['source_commit'])
 return build_base_plan(base['metadata']['runtime'],base['source_commit'])

def main_v3_segment():
 p=argparse.ArgumentParser();p.add_argument('--all-tf-v3-score-segment-plan-sha256',required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args();sha=a.all_tf_v3_score_segment_plan_sha256;need(sys.argv[1:]==['--all-tf-v3-score-segment-plan-sha256',sha]+(['--preflight']if a.preflight else[]),'EXACT_V3_SEGMENT_CLI')
 if a.preflight:
  plan,binding,pref,live=inspect_segment(sha,False,v3=True);print(json.dumps({'status':'PREFLIGHT_ALL_TF_PROVIDER_HLC_V3_INITIAL_METADATA_ONLY','plan':binding,'policy':pref,'models':0,'payloads_decoded':0,'checkpoint_loads':0,'CUDA_initialized':False,'CLAIM_created':False}))
 else:print(json.dumps(run_segment(sha,v3=True),sort_keys=True))
# End exact V3 scope.


def main():
 if sys.argv[1:2]==['--all-tf-v3-score-segment-plan-sha256']:return main_v3_segment()
 if sys.argv[1:2]==['--all-tf-score-segment-plan-sha256']:return main_segment()
 if sys.argv[1:2]==['--all-tf-plan-sha256']:return main_all_tf()
 parser=argparse.ArgumentParser();parser.add_argument('--plan-sha256',required=True);parser.add_argument('--preflight',action='store_true');a=parser.parse_args();need(sys.argv[1:]==['--plan-sha256',a.plan_sha256]+(['--preflight']if a.preflight else[]),'EXACT_CLI')
 if a.preflight:
  plan,binding,pref,live=inspect(a.plan_sha256,False);print(json.dumps({'status':'PREFLIGHT_PROVIDER_HLC_WHOLE_TRAIN_INITIAL_PASS','plan':binding,'policy':pref,'cuda_initialized':False,'payloads_decoded':0,'models':0,'checkpoint_loads':0,'optimizer_updates':0}))
 else:print(json.dumps(run(a.plan_sha256),sort_keys=True))
if __name__=='__main__':main()
