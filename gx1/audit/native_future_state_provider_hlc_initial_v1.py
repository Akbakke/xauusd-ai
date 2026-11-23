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
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--plan-sha256',required=True);parser.add_argument('--preflight',action='store_true');a=parser.parse_args();need(sys.argv[1:]==['--plan-sha256',a.plan_sha256]+(['--preflight']if a.preflight else[]),'EXACT_CLI')
 if a.preflight:
  plan,binding,pref,live=inspect(a.plan_sha256,False);print(json.dumps({'status':'PREFLIGHT_PROVIDER_HLC_WHOLE_TRAIN_INITIAL_PASS','plan':binding,'policy':pref,'cuda_initialized':False,'payloads_decoded':0,'models':0,'checkpoint_loads':0,'optimizer_updates':0}))
 else:print(json.dumps(run(a.plan_sha256),sort_keys=True))
if __name__=='__main__':main()
