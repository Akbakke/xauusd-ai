"""Bounded two-arm suffix-order intervention, with genuine technical updates.

MAIN remains consumed. This new study restores authentic CP3328 twice, proves
its chronological suffix bitexact, then changes only whole-batch order.
"""
from __future__ import annotations
import argparse,copy,io,json,signal,subprocess,time
from pathlib import Path
import numpy as np
if __name__=='__main__':
 from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
 install_native_m5_direction_io_firewall()
from gx1.audit import diagnose_native_future_state_bias_v1 as frozen
from gx1.audit import native_future_state_training_v1 as train
from gx1.audit import qualify_native_m5_full_gc_direction_v1 as core
REPO=train.REPO;POLICY=train.POLICY
STAGE='native_future_state_bias_suffix_order_20261122_001';BASENAME='NATIVE_FUTURE_STATE_BIAS_SUFFIX_ORDER_20261122_001'
PLAN=REPO/'configs/research'/f'{BASENAME}.json';ROOT=Path('/home/andre2/GX1_RUNS')/BASENAME
SELF='gx1/audit/diagnose_native_future_state_bias_suffix_v1.py';TEST='tests/test_diagnose_native_future_state_bias_suffix_v1.py'
GUARD_TEST='tests/test_native_future_state_bias_suffix_guard_v1.py'
START_CURSOR=3328;END_CURSOR=3694;SUFFIX_UPDATES=366;SEED=20260911
REFERENCE_KEYS=('model_state','optimizer_state','rng_state','scheduler_state','position','training','optimizer_parameter_names','steps')
LIMITS={'model_constructions':1,'optimizer_constructions':2,'technical_parameter_fits':2,
 'model_forwards':813,'backbone_forwards':813,'readout_forwards':813,'backward_calls':732,'optimizer_steps':732,
 'checkpoint_loads':1,'checkpoint_restores':2,'checkpoint_writes':1,'checkpoint_readbacks':1,'capture_writes':1,'capture_readbacks':1,
 'feature_cache_payload_loads':1,'whole_target_archive_loads':1,'cohort_archive_loads':1,
 'autograd_calls':0,'normalization_fits':0,'prior_fits':0,'CONTROL_VAL_TEST_rows':0}
read,ref,publish,sha=train.read,train.ref,train.publish,train.sha

def need(ok,why):
 if not bool(ok):raise ValueError('NATIVE_BIAS_SUFFIX_'+why)

def make_orders(main):
 r=main['recipe'];s=main['schedule']
 need(main['limits']['main']['optimizer_steps']==END_CURSOR and r['checkpoint_every_optimizer_updates']==256
  and r['train_batch_rows']==96 and r['seed']==SEED and r['passes']==1
  and s['logical_batches']==s['batches_per_pass']==END_CURSOR and s['rows']==354550
  and s['batch_rows']==96 and s['drop_last']is False and s['shuffle']is False,'ORIGINAL_SCHEDULE')
 chronological=list(range(START_CURSOR,END_CURSOR))
 reordered=np.random.default_rng(r['seed']).permutation(np.arange(START_CURSOR,END_CURSOR-1,dtype=np.int64)).tolist()+[END_CURSOR-1]
 need(len(chronological)==len(reordered)==SUFFIX_UPDATES and sorted(reordered)==chronological
  and reordered[-1]==chronological[-1]and reordered!=chronological,'EXACT_PERMUTATION')
 return train.seal({'chronological':chronological,'reordered':reordered,'seed':r['seed']})

def validate_flags(policy,*,admitted):
 expected={k:False for k in ('training_enabled','native_training_enabled','native_launch_authorized','full_epoch_training_allowed','full_val_allowed')}
 if admitted:expected['training_enabled']=expected['native_training_enabled']=True
 need(all(policy.get(k)is v for k,v in expected.items()),'EXACT_TECHNICAL_TRAINING_FLAGS')

def gate_expected(binding,admitted):
 return {'status':'ADMITTED_EXACT_TWO_ARM_SUFFIX_ORDER_STUDY'if admitted else'PREPARED_NOT_ADMITTED',
  'plan':binding,'execution_authorized':admitted,'remaining_invocations':1 if admitted else 0,'relaunch_allowed':False,
  'training_authorized':admitted,'full_epoch_training_allowed':False,'test_authorized':False}

def validate_reference_steps(actual,expected):
 need(type(actual)is list and type(expected)is list and len(actual)==len(expected)==SUFFIX_UPDATES
  and actual==expected,'REFERENCE_STEPS_BITEXACT');return True

def validate_reference_state(actual_proof,expected_final_proof):
 need(type(actual_proof)is dict and set(actual_proof)==set(REFERENCE_KEYS)
  and type(expected_final_proof)is dict and set(REFERENCE_KEYS)<=set(expected_final_proof)
  and all(actual_proof[k]==expected_final_proof[k]for k in REFERENCE_KEYS),'REFERENCE_STATE_BITEXACT')
 return {'bitexact':True,'components':list(REFERENCE_KEYS),'proof':actual_proof}

def _receipt(binding,main_ref,cursor):
 r=read(binding)
 need(r['original_plan']==r['execution_plan']==main_ref and r['producer_lane']=='main'
  and r['optimizer_updates']==r['logical_cursor']==cursor,'ORIGINAL_CHECKPOINT_RECEIPT');return r

def build_plan(*,runtime,source_base_commit=None):
 main_ref=ref(train.MAIN_PLAN);pair_ref=ref(train.PAIR_PLAN);main=read(main_ref)
 train.require_source_inventory(main['source_owners']);pair=read(pair_ref)
 core.future_cuda_require_runtime(runtime);frozen.validate_function_metadata(main,pair)
 need(main['mode']=='whole_train_one_epoch'and main['backbone_gradient_policy']=='M5_primary'
  and main['metadata']['cohort_report']==pair['metadata']['cohort_report'],'ORIGINAL_FUNCTION_COHORT')
 need(all(runtime[k]==main['metadata']['runtime'][k]for k in ('library_versions','gpu')),'ORIGINAL_RUNTIME_FUNCTION')
 terminal_ref=ref(train.MAIN_ROOT/'TERMINAL.json');terminal=read(terminal_ref);result=read(terminal['result'])
 need(terminal['status']=='COMPLETE'and terminal['exit_code']==0 and terminal['plan']==result['plan']==main_ref
  and result['source_unchanged']is True and result['report']['counts']==main['limits']['main'],'ORIGINAL_COMPLETE')
 ft_ref=ref(frozen.ROOT/'TERMINAL.json');ft=read(ft_ref);fr=read(ft['result']);fp=read(fr['plan'])
 need(ft['status']=='COMPLETE'and ft['exit_code']==0 and fr['status']=='COMPLETE_NATIVE_BIAS_FROZEN_DIAGNOSIS'
  and fr['source_unchanged']is True and fp['inputs']['main_plan']==main_ref and fp['inputs']['paired_plan']==pair_ref
  and fr['report']['counts']==frozen.LIMITS and fr['report']['rows']==frozen.COHORT_ROWS,'FROZEN_REFERENCE_COMPLETE')
 need(sum(x['cursor']==END_CURSOR for x in fr['report']['checkpoint_metrics'])==1
  and all(v['bitexact']for values in fr['report']['diagonal_parity'].values()for v in values.values()),'FROZEN_FINAL_EXACT_REFERENCE')
 cp_ref=ref(train.MAIN_ROOT/f'UPDATE{START_CURSOR:06d}.json');_receipt(cp_ref,main_ref,START_CURSOR)
 final_ref=ref(train.MAIN_ROOT/'FINAL.json');_receipt(final_ref,main_ref,END_CURSOR)
 names=set(main['source_owners'])|{frozen.SELF,frozen.TEST,frozen.GUARD_TEST,SELF,TEST,GUARD_TEST};orders=make_orders(main)
 return {'schema_version':'gx1_native_bias_suffix_order_plan_v1','stage':STAGE,
  'inputs':{'main_plan':main_ref,'paired_plan':pair_ref,'main_result':terminal['result'],'main_terminal':terminal_ref,
   'frozen_result':ft['result'],'frozen_terminal':ft_ref,'frozen_plan':fr['plan'],'checkpoint_receipt':cp_ref,'original_final_receipt':final_ref},
  'runtime':runtime,'source_base_commit':source_base_commit or runtime['source_commit'],
  'source_owners':{k:sha(REPO/k)for k in sorted(names)},'source_bridge':frozen.source_bridge(main),'output_root':str(ROOT),
  'start_cursor':START_CURSOR,'end_cursor':END_CURSOR,'orders':orders,
  'order_source_row_hashes':{arm:[main['schedule']['batch_source_rows_sha256'][i]for i in orders[arm]]for arm in ('chronological','reordered')},
  'limits':LIMITS,'execution':{'class':'trainer','memory':'20G','swap':'512M','outer_seconds':3600,'internal_seconds':3570,
   'kill_after_seconds':10,'numerical_threads':8,'cpu_affinity':list(range(19)),'tasks_max':128},
  'authority':{'technical_training':True,'technical_parameter_fits':2,'optimizer_updates':732,'normalization_fits':0,'prior_fits':0,
   'full_epoch':False,'model_selection':False,'CONTROL_VAL_TEST':False,'generalization_or_economics':False},
  'scope':'Conditional CP3328 suffix-order causality only; two366update continuations, no extra full epoch or MAIN relaunch.'}

def inspect(plan_sha,*,admitted):
 need(Path.cwd().resolve()==REPO and subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='work/gx1-current','CWD_BRANCH')
 binding={'path':str(PLAN),'sha256':plan_sha};plan=read(binding)
 need(plan==build_plan(runtime=plan['runtime'],source_base_commit=plan['source_base_commit']),'EXACT_PLAN')
 need(subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_base_commit'],'FROZEN_HEAD')
 pref=ref(POLICY);policy=read(pref);need(policy[STAGE]==gate_expected(binding,admitted),'ONE_USE_GATE');validate_flags(policy,admitted=admitted)
 need(not ROOT.exists()and not ROOT.is_symlink(),'FRESH_OUTPUT')
 live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['runtime'],live)
 return plan,binding,pref,live

def cohort_context(full,pair):
 """Same qualified fifteen-array cohort decoder, sharing the full provider."""
 from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
 meta=pair['metadata'];report=meta['cohort_report'];data=read(report['cohort_snapshot']);binding=data['arrays']
 need(ref(binding['path'])==binding,'COHORT_ARCHIVE');expected={'selected_positions','source_rows','issue_ns'}|{'truth__'+k for k in train.TRUTH};arrays={}
 with np.load(binding['path'],allow_pickle=False)as z:
  need(set(z.files)==set(data['array_headers'])==expected,'COHORT15_INVENTORY')
  for k in sorted(expected):
   a=z[k];h=data['array_headers'][k];need(a.dtype.str==h['dtype']and list(a.shape)==h['shape']and train.array_sha(a)==h['sha256'],'COHORT_ARRAY_BYTES');arrays[k]=a
 positions=arrays['selected_positions'];source=arrays['source_rows'];truth={k:arrays['truth__'+k]for k in train.TRUTH};provider=full['provider'];at=np.searchsorted(full['positions'],positions)
 need(len(source)==frozen.COHORT_ROWS and np.all(at<len(full['positions']))and np.array_equal(full['positions'][at],positions)
  and np.array_equal(source,provider.source_row[positions])and np.array_equal(arrays['issue_ns'],truth['issue_ns'])
  and np.array_equal(truth['issue_ns'],provider.issue_ns[positions])and provider.input_ready[positions].all()
  and all(np.array_equal(v,full['truth'][k][at])for k,v in truth.items()),'COHORT_EXACT_MAIN_SUBSET')
 scoring=objective.require_native_future_state_train_prior(read(meta['matched_scoring_prior']),target_contract=meta['target_contract'])
 need(objective._native_prior_array_sha(truth)==report['truth_sha256']==scoring['fit_population']['truth_sha256']
  and train.array_sha(source)==pair['schedule']['source_rows_sha256']and train.array_sha(truth['issue_ns'])==pair['schedule']['issue_ns_sha256'],'COHORT_ORIGINAL_PRIOR_AXIS')
 return {'provider':provider,'normalization':full['normalization'],'prior':full['prior'],'scoring_prior':scoring,'positions':positions,'source_rows':source,'truth':truth}

def _actual_reference_proof(model,optimizer,cp,steps,main):
 return {'model_state':core.state_sha(model.state_dict()),'optimizer_state':core.state_sha(optimizer.state_dict()),'rng_state':core.state_sha(core.training_rng()),
  'scheduler_state':core.state_sha(None),'position':core.state_sha(train.position(main['schedule'],END_CURSOR)),'training':core.state_sha(model.training),
  'optimizer_parameter_names':core.state_sha([n for n,_ in model.named_parameters()]),'steps':core.state_sha(cp['steps']+steps)}

def _optimizer(model,main):
 import torch
 options={**main['recipe']['optimizer'],'betas':tuple(main['recipe']['optimizer']['betas'])};return torch.optim.AdamW(model.parameters(),**options)

def _score_capture(model,context,main):
 outputs={}
 def capture(module,args,output):
  for k,v in output.items():outputs.setdefault(k,[]).append(v.detach().cpu().numpy().copy())
 batches=core.all_score_batches(np.arange(frozen.COHORT_ROWS,dtype=np.int64),frozen.BATCH);need(len(batches)==81,'EXACT_SCORE_BATCHES')
 hook=model.register_forward_hook(capture)
 try:score=core.future_cuda_score(model,(train.packet(context,b)for b in batches),main,context['prior'],initial=False,equal_tf=True)
 finally:hook.remove()
 arrays={k:np.concatenate(v)for k,v in outputs.items()};need(all(len(v)==frozen.COHORT_ROWS for v in arrays.values()),'CAPTURE_ROWS');return score,arrays

def physical(plan,binding,live):
 import torch
 from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
 main=read(plan['inputs']['main_plan']);pair=read(plan['inputs']['paired_plan']);original=read(plan['inputs']['main_result']);fr=read(plan['inputs']['frozen_result'])
 cp_receipt=read(plan['inputs']['checkpoint_receipt']);final_receipt=read(plan['inputs']['original_final_receipt'])
 full=train.load_context(main);context=cohort_context(full,pair);counts=dict.fromkeys(LIMITS,0)
 counts['feature_cache_payload_loads']=counts['whole_target_archive_loads']=counts['cohort_archive_loads']=1
 need(len(full['batches'])==END_CURSOR and all(len(full['batches'][i])==96 for i in range(START_CURSOR,END_CURSOR-1))
  and len(full['batches'][-1])==354550-(END_CURSOR-1)*96,'FULL_SUFFIX_GEOMETRY')
 core.future_cuda_seed_and_runtime({'recipe':main['recipe'],'metadata':{'runtime':live}});m=main['metadata']
 model=build_native_future_state_model(input_normalization=full['normalization'],input_binding=m['input_binding'],architecture=m['architecture'],device=torch.device('cuda'),
  train_prior=full['prior'],target_contract=m['target_contract'],backbone_gradient_policy='M5_primary');counts['model_constructions']=1
 need(ref(cp_receipt['checkpoint']['path'])==cp_receipt['checkpoint'],'BASE_CHECKPOINT_BYTES')
 cp=torch.load(cp_receipt['checkpoint']['path'],map_location='cpu',weights_only=True);counts['checkpoint_loads']=1
 frozen.validate_checkpoint(cp,cp_receipt,main,original,START_CURSOR);original_cp_proof=train.checkpoint_proof(cp)
 def counter(name):
  def hook(*a):counts[name]+=1;need(counts[name]<=LIMITS[name],'FORWARD_BUDGET')
  return hook
 hooks=[model.register_forward_pre_hook(counter('model_forwards')),model.backbone.register_forward_pre_hook(counter('backbone_forwards')),model.readout.register_forward_pre_hook(counter('readout_forwards'))]
 reports={};allocator=[];reference_end_rng=None;optimizer=None
 try:
  for arm in ('chronological','reordered'):
   optimizer=_optimizer(model,main);counts['optimizer_constructions']+=1
   # AdamW may share CPU step tensors with its load input. Clone this one
   # authenticated payload per restore; keep the retained original immutable.
   restored=copy.deepcopy(cp)
   train.strict_restore(model,optimizer,restored,trajectory=cp['trajectory'],original_plan_ref=plan['inputs']['main_plan'],
    expected_cursor=START_CURSOR,current_runtime=live,expected_state_proof=original_cp_proof);counts['checkpoint_restores']+=1;del restored
   need(train.checkpoint_proof(cp)==original_cp_proof,'BASE_CP_IMMUTABLE_AFTER_RESTORE');counts['technical_parameter_fits']+=1;steps=[]
   for index in plan['orders'][arm]:steps.append(train.training_step(model,optimizer,train.packet(full,full['batches'][index]),main,counts,index))
   need(len(steps)==SUFFIX_UPDATES and all(not x['skip_update']for x in steps),'EXACT_SUFFIX_UPDATES')
   need(train.checkpoint_proof(cp)==original_cp_proof,'BASE_CP_IMMUTABLE_AFTER_UPDATES')
   end_rng=core.state_sha(core.training_rng());proof=_actual_reference_proof(model,optimizer,cp,steps,main)
   if arm=='chronological':
    validate_reference_steps(steps,original['report']['steps'][START_CURSOR:]);verified=validate_reference_state(proof,final_receipt['proof']);reference_end_rng=end_rng
    verified['suffix_steps_sha256']=core.state_sha(steps);verified['source_batch_order_sha256']=core.state_sha(plan['orders'][arm])
    publish(ROOT/'REFERENCE_PROOF.json',verified);reports[arm]={'reference_bitexact':verified,'scores_reused_after_state_proof':{'source':plan['inputs']['frozen_result'],
     'cursor':END_CURSOR,'checkpoint_report':next(x for x in fr['report']['checkpoint_metrics']if x['cursor']==END_CURSOR)},'optimizer_updates':SUFFIX_UPDATES}
    allocator.append(train.release_phase_allocator_cache(model,optimizer,label='REFERENCE_TO_SECOND_RESTORE'));del optimizer;optimizer=None
   else:
    need(end_rng==reference_end_rng,'SAME_END_RNG_WITH_SAME_BATCH_SHAPES')
    checkpoint={'schema_version':'gx1_native_future_state_bias_suffix_checkpoint_v1','study_plan':binding,'study_arm':'reordered',
     'source_checkpoint_receipt':plan['inputs']['checkpoint_receipt'],'original_main_plan':plan['inputs']['main_plan'],
     'source_batch_order':plan['orders'][arm],'source_batch_order_sha256':core.state_sha(plan['orders'][arm]),
     'restored_original_optimizer_step':START_CURSOR,'optimizer_updates_in_study':SUFFIX_UPDATES,'model_state':model.state_dict(),
     'optimizer_state':optimizer.state_dict(),'scheduler_state':None,'rng_state':core.training_rng(),'training':model.training,
     'optimizer_parameter_names':[n for n,_ in model.named_parameters()],'steps':steps,'authority':'diagnostic_only_no_promotion_or_MAIN_trajectory_claim'}
    candidate_checkpoint=core.future_cuda_publish_checkpoint(ROOT/'REORDERED_FINAL.pt',checkpoint,counts);candidate_proof=core.state_sha(checkpoint)
    allocator.append(train.release_phase_allocator_cache(model,optimizer,label='REORDERED_TRAIN_TO_SCORE'));score,outputs=_score_capture(model,context,main)
    need(core.state_sha(checkpoint)==candidate_proof,'SCORE_DID_NOT_MUTATE_CANDIDATE_STATE');metrics=frozen.metric_rows(outputs,context['truth'],m['target_contract'])
    arrays={'issue_ns':context['truth']['issue_ns'],**outputs};stream=io.BytesIO();np.savez_compressed(stream,**arrays)
    capture=publish(ROOT/'REORDERED_CAPTURE.npz',stream.getvalue());stream.close();counts['capture_writes']+=1
    with np.load(capture['path'],allow_pickle=False)as z:need(set(z.files)==set(arrays)and all(np.array_equal(z[k],v)for k,v in arrays.items()),'CAPTURE_DISK_READBACK')
    counts['capture_readbacks']+=1
    reports[arm]={'generated_score':score,'metrics':metrics,'optimizer_updates':SUFFIX_UPDATES,'steps':steps,'checkpoint':candidate_checkpoint,
     'checkpoint_tree_sha256':candidate_proof,'capture':capture,'capture_array_headers':{k:{'shape':list(v.shape),'dtype':v.dtype.str,'sha256':train.array_sha(v)}for k,v in arrays.items()},
     'end_rng_matches_chronological':True,'final_state_components':proof}
  need(counts==LIMITS,'EXACT_PHYSICAL_COUNTS');need(torch.cuda.max_memory_reserved()<=core.LARGER_VRAM_MAX,'CUDA_VRAM_CAP')
  constants={'whole_TRAIN_factory':train.constant_prior_scores(context['prior'],context['truth'],m['target_contract']),
   'matched_SCORING_ONLY':train.constant_prior_scores(context['scoring_prior'],context['truth'],m['target_contract'])}
  need(constants==fr['report']['constant_references'],'SAME_FROZEN_SCORING_REFERENCES');need(train.checkpoint_proof(cp)==original_cp_proof,'BASE_CP_IMMUTABLE_BOOKEND')
  return {'counts':counts,'arms':reports,'constant_references':constants,'base_checkpoint_proof':original_cp_proof,'allocator_phase_boundaries':allocator,
   'cuda_peak_reserved_bytes':int(torch.cuda.max_memory_reserved()),'causal_scope':'Order intervention conditional on authentic CP3328 and this one fixed permutation; no full-epoch/origin/generalization/edge claim.'}
 finally:
  for h in hooks:h.remove()

def _input_bookend(plan):
 for binding in plan['inputs'].values():read(binding)
 main=read(plan['inputs']['main_plan']);pair=read(plan['inputs']['paired_plan'])
 for k in ('source_snapshot','normalization','input_binding_ref','prior','dataset','native_manifest'):read(main['metadata'][k])
 snapshot=read(main['metadata']['source_snapshot']);data=read(main['metadata']['dataset']);cohort=read(pair['metadata']['cohort_report']['cohort_snapshot'])
 need(ref(snapshot['archive']['path'])==snapshot['archive']and ref(data['arrays']['path'])==data['arrays']and ref(cohort['arrays']['path'])==cohort['arrays'],'FROZEN_INPUT_PAYLOAD_BOOKEND')
 cp=read(plan['inputs']['checkpoint_receipt']);need(ref(cp['checkpoint']['path'])==cp['checkpoint'],'BASE_CHECKPOINT_BOOKEND')

def run(plan_sha):
 from gx1.contracts.gx1_capped_execution_v1 import require_native_future_state_bias_suffix_execution
 began=time.monotonic();plan,binding,pref,live=inspect(plan_sha,admitted=True);guard=require_native_future_state_bias_suffix_execution(plan_sha256=plan_sha)
 need(guard['plan']==binding and guard['output_root']==str(ROOT)and guard['outer_seconds']==3600,'EXACT_GUARD')
 ROOT.mkdir(exist_ok=False);claim=publish(ROOT/'CLAIM.json',{'plan':binding,'policy':pref,'runtime':live,'process_identity':core.future_cpu_process_identity(),
  'remaining_invocations':0,'relaunch_allowed':False,'technical_parameter_fits':2,'optimizer_updates':732,'full_epoch':False})
 def expired(*a):raise TimeoutError('BIAS_SUFFIX_WALL')
 old=signal.signal(signal.SIGALRM,expired)
 try:
  remaining=3570-(time.monotonic()-began);need(remaining>0,'WALL_BEFORE_INPUTS');signal.setitimer(signal.ITIMER_REAL,remaining)
  report=physical(plan,binding,live)
  need(ref(POLICY)==pref and ref(PLAN)==binding and all(sha(REPO/k)==v for k,v in plan['source_owners'].items()),'SOURCE_POLICY_BOOKEND');_input_bookend(plan)
  need(subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==live['source_commit']and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==live['wsl_boot_id']
   and core.physical_windows_boot_id()==live['physical_windows_boot_id'],'BOOT_HEAD_BOOKEND');need(core.common.firewall_state()['denied_attempts']==0,'M1_FIREWALL')
  result=publish(ROOT/'RESULT.json',{'status':'COMPLETE_NATIVE_BIAS_SUFFIX_ORDER_STUDY','plan':binding,'claim':claim,'report':report,
   'elapsed_seconds':time.monotonic()-began,'source_unchanged':True,'test_data_used':False})
  terminal=publish(ROOT/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,'source_unchanged':True,'test_data_used':False})
  return {'result':result,'terminal':terminal}
 except BaseException as error:
  publish(ROOT/'TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'plan':binding,'claim':claim,'error':type(error).__name__+':'+str(error)[:2000],'remaining_invocations':0,'relaunch_allowed':False});raise
 finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)

def main():
 import sys
 parser=argparse.ArgumentParser();parser.add_argument('--plan-sha256',required=True);parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
 need(sys.argv[1:]==['--plan-sha256',args.plan_sha256]+(['--preflight']if args.preflight else[]),'EXACT_CLI')
 if args.preflight:
  p,binding,pref,live=inspect(args.plan_sha256,admitted=False);print(json.dumps({'status':'PREFLIGHT_BIAS_SUFFIX_ORDER','plan':binding,'policy':pref,'CUDA_initialized':False,'models':0,'checkpoint_payloads':0,'optimizer_updates':0}))
 else:print(json.dumps(run(args.plan_sha256),sort_keys=True))

if __name__=='__main__':main()
