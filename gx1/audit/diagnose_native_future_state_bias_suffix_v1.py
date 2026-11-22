"""One bounded suffix-order candidate, reusing the authentic bitexact control.

Consumed001 is never relaunched. Study002 restores original CP3328 once and
changes whole-batch order, retaining native candidate checkpoints every64updates.
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
STAGE='native_future_state_bias_suffix_order_20261122_002';BASENAME='NATIVE_FUTURE_STATE_BIAS_SUFFIX_ORDER_20261122_002'
PLAN=REPO/'configs/research'/f'{BASENAME}.json';ROOT=Path('/home/andre2/GX1_RUNS')/BASENAME
SELF='gx1/audit/diagnose_native_future_state_bias_suffix_v1.py';TEST='tests/test_diagnose_native_future_state_bias_suffix_v1.py'
GUARD_TEST='tests/test_native_future_state_bias_suffix_guard_v1.py'
START_CURSOR=3328;END_CURSOR=3694;SUFFIX_UPDATES=366;SEED=20260911
REFERENCE_KEYS=('model_state','optimizer_state','rng_state','scheduler_state','position','training','optimizer_parameter_names','steps')
LIMITS={'model_constructions':1,'optimizer_constructions':1,'technical_parameter_fits':1,
 'model_forwards':447,'backbone_forwards':447,'readout_forwards':447,'backward_calls':366,'optimizer_steps':366,
 'checkpoint_loads':1,'checkpoint_restores':1,'checkpoint_writes':6,'checkpoint_readbacks':6,'capture_writes':1,'capture_readbacks':1,
 'feature_cache_payload_loads':1,'whole_target_archive_loads':1,'cohort_archive_loads':1,
 'autograd_calls':0,'normalization_fits':0,'prior_fits':0,'CONTROL_VAL_TEST_rows':0}
CONTROL_BASENAME='NATIVE_FUTURE_STATE_BIAS_SUFFIX_ORDER_20261122_001'
CONTROL_ROOT=ROOT.parent/CONTROL_BASENAME
CONTROL_PLAN={'path':str(REPO/'configs/research'/f'{CONTROL_BASENAME}.json'),'sha256':'d2c2f9e5b2b8b8519bbe27d8e2280c6dfa97f0c3fb3f7b6f21f2deee329f8cf5'}
CONTROL_CLAIM={'path':str(CONTROL_ROOT/'CLAIM.json'),'sha256':'5caec0630116c626c68bd7661c72de76caf8529c7d7a820271041e7b9de942b0'}
CONTROL_PROOF={'path':str(CONTROL_ROOT/'REFERENCE_PROOF.json'),'sha256':'7e8cce88f6d94a800d8f8f95b99f436b07940d1b935f968b433960c0797dd0e0'}
CONTROL_SOURCE='63a1ff0ac8889046691dafe1acd1a63b55974cc0'
RESOURCE_SOURCE='6749062726c8562db35194857f56ee8aa4859d23'
RESOURCE_REVIEW={'path':'/home/andre2/GX1_RUNS/GX1_NATIVE_FUTURE_STATE_BIAS_CAUSE_20261122_001/HOST_RESOURCE_QUALIFICATION_REVIEW001.json','sha256':'439fbebc9c822b3eaae80d1c1f525499a04d418e34eeb06f41ad72f8df20eaae'}
RESOURCE_MAC_BACKUP={'path':'/Users/andrekildalbakke/GX1_BACKUP/CURRENT_HOST_RESOURCE_SAFETY_20261122_001/MANIFEST.json','sha256':'43ff822f0161ea1447e5fc659feb92de95c45efd400c30625a1061e9e393b12c'}
DECISION_PREREGISTRATION={'path':'/home/andre2/GX1_RUNS/GX1_NATIVE_FUTURE_STATE_BIAS_CAUSE_20261122_001/ORDER_CANDIDATE002_DECISION001.json','sha256':'f97d7c2bf08c5ea4f0dadc85f87b9070819abc5df8aeea4f03c73f349d973c56'}
RESOURCE_SOURCES={'scripts/gx1_guarded_trainer_exec.sh','scripts/gx1_host_telemetry_bridge_query.sh',
 'scripts/windows/Install-GX1-HostTelemetryBridge.ps1','tests/test_gx1_capped_run_contract.py',
 'tests/test_host_telemetry_bridge_query.py','tests/test_host_telemetry_bridge_installer_contract.py'}
CHECKPOINT_UPDATES=tuple(range(64,SUFFIX_UPDATES,64))+(SUFFIX_UPDATES,)
CHECKPOINT_SCHEMA='gx1_native_future_state_bias_suffix_candidate_checkpoint_v2'
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
 return {'status':'ADMITTED_EXACT_SINGLE_ARM_SUFFIX_ORDER_STUDY'if admitted else'PREPARED_NOT_ADMITTED',
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

def validate_reference_reuse(control_plan,claim,saved,*,control_plan_ref,main,inputs,original,final_receipt):
 """Authenticate completed001control evidence; do not claim a new control run."""
 need(control_plan_ref==CONTROL_PLAN and control_plan['stage']=='native_future_state_bias_suffix_order_20261122_001'
  and control_plan['output_root']==str(CONTROL_ROOT)and control_plan['source_base_commit']==CONTROL_SOURCE
  and claim['plan']==control_plan_ref
  and {k:v for k,v in claim['runtime'].items()if k!='observed_utc'}=={k:v for k,v in control_plan['runtime'].items()if k!='observed_utc'}
  and claim['runtime']['source_commit']==control_plan['source_base_commit']
  and claim['remaining_invocations']==0 and claim['relaunch_allowed']is False
  and claim['technical_parameter_fits']==2 and claim['optimizer_updates']==732 and claim['full_epoch']is False,'AUTHENTIC_CONTROL_CLAIM')
 need(set(control_plan['inputs'])=={'main_plan','paired_plan','main_result','main_terminal','frozen_result','frozen_terminal','frozen_plan','checkpoint_receipt','original_final_receipt'}
  and all(control_plan['inputs'][k]==inputs[k]for k in control_plan['inputs'])
  and control_plan['orders']==make_orders(main)
  and control_plan['order_source_row_hashes']=={arm:[main['schedule']['batch_source_rows_sha256'][i]for i in control_plan['orders'][arm]]for arm in ('chronological','reordered')},'CONTROL_ORIGINAL_INPUTS_ORDER')
 expected_steps=original['report']['steps'][START_CURSOR:]
 need(len(expected_steps)==SUFFIX_UPDATES and type(saved)is dict
  and set(saved)=={'bitexact','components','proof','source_batch_order_sha256','suffix_steps_sha256'}
  and saved['bitexact']is True and saved['components']==list(REFERENCE_KEYS)
  and saved['source_batch_order_sha256']==core.state_sha(control_plan['orders']['chronological'])
  and saved['suffix_steps_sha256']==core.state_sha(expected_steps),'CONTROL_STEP_ORDER_PROOF')
 validate_reference_state(saved['proof'],final_receipt['proof'])
 return {'generated_in_this_study':False,'control_plan':control_plan_ref,'claim':CONTROL_CLAIM,'proof_artifact':CONTROL_PROOF,
  'bitexact':True,'components':list(REFERENCE_KEYS),'proof':saved['proof'],
  'source_batch_order_sha256':saved['source_batch_order_sha256'],'suffix_steps_sha256':saved['suffix_steps_sha256']}


def checkpoint_state(model,optimizer,*,binding,plan,steps):
 completed=len(steps);order=plan['orders']['reordered'];prefix=order[:completed]
 need(completed in CHECKPOINT_UPDATES and completed<=len(order)==SUFFIX_UPDATES
  and all(not x['skip_update']for x in steps)
  and [x['logical_step']for x in steps]==[i+1 for i in prefix],'CANDIDATE_CHECKPOINT_PROGRESS')
 return {'schema_version':CHECKPOINT_SCHEMA,'study_plan':binding,'study_arm':'reordered',
  'source_checkpoint_receipt':plan['inputs']['checkpoint_receipt'],'original_main_plan':plan['inputs']['main_plan'],
  'reused_control_proof':plan['inputs']['original_control_proof'],'source_batch_order':order,
  'source_batch_order_sha256':core.state_sha(order),'visited_source_batch_order':prefix,'visited_source_batch_order_sha256':core.state_sha(prefix),
  'progress':{'optimizer_updates_completed':completed,'optimizer_updates_remaining':SUFFIX_UPDATES-completed,
   'next_order_position':completed,'last_original_source_batch_index':prefix[-1],'restored_original_optimizer_step':START_CURSOR,
   'optimizer_step_count_after_restore':START_CURSOR+completed},
  'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'scheduler_state':None,'rng_state':core.training_rng(),
  'training':model.training,'optimizer_parameter_names':[n for n,_ in model.named_parameters()],'steps':list(steps),
  'authority':'diagnostic_only_no_promotion_or_MAIN_trajectory_claim'}


def publish_candidate_checkpoint(model,optimizer,*,binding,plan,steps,counts):
 checkpoint=checkpoint_state(model,optimizer,binding=binding,plan=plan,steps=steps);completed=len(steps)
 stem='REORDERED_FINAL'if completed==SUFFIX_UPDATES else f'UPDATE{completed:06d}'
 tree=core.state_sha(checkpoint);saved=core.future_cuda_publish_checkpoint(ROOT/(stem+'.pt'),checkpoint,counts)
 receipt={'schema_version':'gx1_native_future_state_bias_suffix_candidate_receipt_v2','study_plan':binding,'checkpoint':saved,
  'checkpoint_tree_sha256':tree,'progress':checkpoint['progress'],'source_batch_order_sha256':checkpoint['source_batch_order_sha256'],
  'visited_source_batch_order_sha256':checkpoint['visited_source_batch_order_sha256'],'main_trajectory_claim':False}
 receipt_ref=publish(ROOT/(stem+'.json'),receipt)
 return {'receipt':receipt_ref,**receipt},checkpoint


def retained_checkpoint_receipts(binding):
 retained=[]
 for completed in CHECKPOINT_UPDATES:
  stem='REORDERED_FINAL'if completed==SUFFIX_UPDATES else f'UPDATE{completed:06d}'
  path=ROOT/(stem+'.json')
  if path.exists():
   item=read(ref(path));need(item['study_plan']==binding and item['progress']['optimizer_updates_completed']==completed
    and item['main_trajectory_claim']is False,'RETAINED_CHECKPOINT_BINDING');retained.append({'receipt':ref(path),**item})
 return retained


def metric_comparison(candidate,reference):
 need(set(candidate)==set(reference),'SAME_METRIC_BLOCKS');report={}
 for block,cells in candidate.items():
  need(set(cells)==set(reference[block])==set(train.TF),'SAME_FIVE_TFS');report[block]={}
  for tf in train.TF:
   report[block][tf]={}
   for task in ('activity','side'):
    c,r=cells[tf][task],reference[block][tf][task];need(c['count']==r['count'],'SAME_METRIC_SUPPORT')
    if not c['count']:report[block][tf][task]={'count':0};continue
    need(c['unique_target_bars']==r['unique_target_bars']and c['actual_positive_fraction']==r['actual_positive_fraction'],'SAME_METRIC_TRUTH')
    report[block][tf][task]={'count':c['count'],'probability_minus_actual_delta':c['probability_minus_actual']-r['probability_minus_actual'],
     'absolute_probability_residual_delta':abs(c['probability_minus_actual'])-abs(r['probability_minus_actual']),
     **{k+'_delta':None if c[k]is None or r[k]is None else c[k]-r[k]for k in ('Brier','logloss','AUC')}}
 return report


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
 final_ref=ref(train.MAIN_ROOT/'FINAL.json');final_receipt=_receipt(final_ref,main_ref,END_CURSOR)
 inputs={'main_plan':main_ref,'paired_plan':pair_ref,'main_result':terminal['result'],'main_terminal':terminal_ref,
  'frozen_result':ft['result'],'frozen_terminal':ft_ref,'frozen_plan':fr['plan'],'checkpoint_receipt':cp_ref,'original_final_receipt':final_ref,
  'original_control_plan':CONTROL_PLAN,'original_control_claim':CONTROL_CLAIM,'original_control_proof':CONTROL_PROOF,'resource_qualification_review':RESOURCE_REVIEW,'decision_preregistration':DECISION_PREREGISTRATION}
 read(DECISION_PREREGISTRATION)
 reference=validate_reference_reuse(read(CONTROL_PLAN),read(CONTROL_CLAIM),read(CONTROL_PROOF),control_plan_ref=CONTROL_PLAN,
  main=main,inputs=inputs,original=result,final_receipt=final_receipt)
 capture=fr['report']['capture'];need(ref(capture['path'])==capture,'FROZEN_CAPTURE_BYTES')
 reference.update(frozen_result=ft['result'],frozen_capture=capture,frozen_capture_array_headers=fr['report']['capture_array_headers'],
  checkpoint_report=next(x for x in fr['report']['checkpoint_metrics']if x['cursor']==END_CURSOR))
 review=read(RESOURCE_REVIEW);need(review['status']=='PASS_CURRENT_BOOT_IDLE_AND_STOP_MECHANISMS_NOT_LOAD_PERFORMANCE'
  and review['all_model_gates_closed']is True and review['model_or_stress_started']is False,'RESOURCE_QUALIFICATION')
 read({'path':review['live_guard']['path'],'sha256':review['live_guard']['sha256']})
 names=set(main['source_owners'])|{frozen.SELF,frozen.TEST,frozen.GUARD_TEST,SELF,TEST,GUARD_TEST}|RESOURCE_SOURCES;orders=make_orders(main)
 return {'schema_version':'gx1_native_bias_suffix_order_plan_v2','stage':STAGE,
  'inputs':inputs,'reference_control_reuse':reference,
  'candidate_checkpoints':{'schema_version':CHECKPOINT_SCHEMA,'every_optimizer_updates':64,'update_ordinals':list(CHECKPOINT_UPDATES),'main_trajectory_claim':False},
  'resource_qualification':{'review':RESOURCE_REVIEW,'source_commit':RESOURCE_SOURCE,'independent_mac_backup':RESOURCE_MAC_BACKUP},
  'runtime':runtime,'source_base_commit':source_base_commit or runtime['source_commit'],
  'source_owners':{k:sha(REPO/k)for k in sorted(names)},'source_bridge':frozen.source_bridge(main),'output_root':str(ROOT),
  'start_cursor':START_CURSOR,'end_cursor':END_CURSOR,'orders':orders,
  'order_source_row_hashes':{arm:[main['schedule']['batch_source_rows_sha256'][i]for i in orders[arm]]for arm in ('chronological','reordered')},
  'limits':LIMITS,'execution':{'class':'trainer','memory':'20G','swap':'512M','outer_seconds':3600,'internal_seconds':3570,
   'kill_after_seconds':10,'numerical_threads':8,'cpu_affinity':list(range(19)),'tasks_max':128},
  'authority':{'technical_training':True,'technical_parameter_fits':1,'optimizer_updates':366,'normalization_fits':0,'prior_fits':0,
   'full_epoch':False,'model_selection':False,'CONTROL_VAL_TEST':False,'generalization_or_economics':False},
  'scope':'One366update reordered candidate, reusing completed001bitexact chronological control; fixed RNG-stream order effect only, no origin/generalization/economics claim.'}

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
 reference=validate_reference_reuse(read(plan['inputs']['original_control_plan']),read(plan['inputs']['original_control_claim']),
  read(plan['inputs']['original_control_proof']),control_plan_ref=plan['inputs']['original_control_plan'],main=main,inputs=plan['inputs'],original=original,final_receipt=final_receipt)
 need(reference['proof']==plan['reference_control_reuse']['proof'],'REFERENCE_REUSE_BOOKEND_BEFORE_MODEL')
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
 try:
  optimizer=_optimizer(model,main);counts['optimizer_constructions']=1
  # Clone authenticated CP so AdamW's CPU step tensors cannot mutate its proof.
  restored=copy.deepcopy(cp)
  train.strict_restore(model,optimizer,restored,trajectory=cp['trajectory'],original_plan_ref=plan['inputs']['main_plan'],
   expected_cursor=START_CURSOR,current_runtime=live,expected_state_proof=original_cp_proof);counts['checkpoint_restores']=1;del restored
  need(train.checkpoint_proof(cp)==original_cp_proof,'BASE_CP_IMMUTABLE_AFTER_RESTORE');counts['technical_parameter_fits']=1;steps=[];checkpoints=[]
  for index in plan['orders']['reordered']:
   steps.append(train.training_step(model,optimizer,train.packet(full,full['batches'][index]),main,counts,index))
   need(not steps[-1]['skip_update'],'EXACT_SUFFIX_UPDATE')
   if len(steps)in CHECKPOINT_UPDATES:
    saved,checkpoint=publish_candidate_checkpoint(model,optimizer,binding=binding,plan=plan,steps=steps,counts=counts);checkpoints.append(saved)
    if len(steps)<SUFFIX_UPDATES:del checkpoint
  need(len(steps)==SUFFIX_UPDATES and train.checkpoint_proof(cp)==original_cp_proof,'EXACT_SUFFIX_AND_BASE_CP_IMMUTABLE')
  end_rng=core.state_sha(core.training_rng());need(end_rng==reference['proof']['rng_state'],'SAME_END_RNG_WITH_SAME_BATCH_SHAPES')
  proof=_actual_reference_proof(model,optimizer,cp,steps,main);candidate_proof=core.state_sha(checkpoint)
  allocator=[train.release_phase_allocator_cache(model,optimizer,label='REORDERED_TRAIN_TO_SCORE')];score,outputs=_score_capture(model,context,main)
  need(core.state_sha(checkpoint)==candidate_proof,'SCORE_DID_NOT_MUTATE_CANDIDATE_STATE');metrics=frozen.metric_rows(outputs,context['truth'],m['target_contract'])
  arrays={'issue_ns':context['truth']['issue_ns'],**outputs};stream=io.BytesIO();np.savez_compressed(stream,**arrays)
  capture=publish(ROOT/'REORDERED_CAPTURE.npz',stream.getvalue());stream.close();counts['capture_writes']+=1
  with np.load(capture['path'],allow_pickle=False)as z:need(set(z.files)==set(arrays)and all(np.array_equal(z[k],v)for k,v in arrays.items()),'CAPTURE_DISK_READBACK')
  counts['capture_readbacks']+=1
  reports={'chronological':{'generated_in_this_study':False,'reference_bitexact':reference,
   'scores_and_capture_reused_after_proof':plan['reference_control_reuse'],'optimizer_updates_in_this_study':0},
   'reordered':{'generated_score':score,'metrics':metrics,'comparison_to_reused_chronological':metric_comparison(metrics,plan['reference_control_reuse']['checkpoint_report']['metrics']),
    'optimizer_updates':SUFFIX_UPDATES,'steps':steps,'checkpoints':checkpoints,'checkpoint':checkpoints[-1],
    'checkpoint_tree_sha256':candidate_proof,'capture':capture,'capture_array_headers':{k:{'shape':list(v.shape),'dtype':v.dtype.str,'sha256':train.array_sha(v)}for k,v in arrays.items()},
    'end_rng_matches_reused_chronological':True,'final_state_components':proof}}
  need(counts==LIMITS,'EXACT_PHYSICAL_COUNTS');need(torch.cuda.max_memory_reserved()<=core.LARGER_VRAM_MAX,'CUDA_VRAM_CAP')
  constants={'whole_TRAIN_factory':train.constant_prior_scores(context['prior'],context['truth'],m['target_contract']),
   'matched_SCORING_ONLY':train.constant_prior_scores(context['scoring_prior'],context['truth'],m['target_contract'])}
  need(constants==fr['report']['constant_references'],'SAME_FROZEN_SCORING_REFERENCES');need(train.checkpoint_proof(cp)==original_cp_proof,'BASE_CP_IMMUTABLE_BOOKEND')
  return {'counts':counts,'arms':reports,'constant_references':constants,'base_checkpoint_proof':original_cp_proof,'allocator_phase_boundaries':allocator,
   'cuda_peak_reserved_bytes':int(torch.cuda.max_memory_reserved()),'causal_scope':'Order effect conditional on authentic CP3328, fixed RNG stream and one permutation including input-to-dropout-mask assignment; no origin/generalization/edge claim.'}
 finally:
  for h in hooks:h.remove()


def _input_bookend(plan):
 for binding in plan['inputs'].values():read(binding)
 main=read(plan['inputs']['main_plan']);pair=read(plan['inputs']['paired_plan'])
 for k in ('source_snapshot','normalization','input_binding_ref','prior','dataset','native_manifest'):read(main['metadata'][k])
 snapshot=read(main['metadata']['source_snapshot']);data=read(main['metadata']['dataset']);cohort=read(pair['metadata']['cohort_report']['cohort_snapshot'])
 need(ref(snapshot['archive']['path'])==snapshot['archive']and ref(data['arrays']['path'])==data['arrays']and ref(cohort['arrays']['path'])==cohort['arrays'],'FROZEN_INPUT_PAYLOAD_BOOKEND')
 cp=read(plan['inputs']['checkpoint_receipt']);need(ref(cp['checkpoint']['path'])==cp['checkpoint'],'BASE_CHECKPOINT_BOOKEND')
 capture=plan['reference_control_reuse']['frozen_capture'];need(ref(capture['path'])==capture,'FROZEN_REFERENCE_CAPTURE_BOOKEND')

def run(plan_sha):
 from gx1.contracts.gx1_capped_execution_v1 import require_native_future_state_bias_suffix_execution
 began=time.monotonic();plan,binding,pref,live=inspect(plan_sha,admitted=True);guard=require_native_future_state_bias_suffix_execution(plan_sha256=plan_sha)
 need(guard['plan']==binding and guard['output_root']==str(ROOT)and guard['outer_seconds']==3600,'EXACT_GUARD')
 ROOT.mkdir(exist_ok=False);claim=publish(ROOT/'CLAIM.json',{'plan':binding,'policy':pref,'runtime':live,'process_identity':core.future_cpu_process_identity(),
  'remaining_invocations':0,'relaunch_allowed':False,'technical_parameter_fits':1,'optimizer_updates':366,'full_epoch':False})
 def expired(*a):raise TimeoutError('BIAS_SUFFIX_WALL')
 def stopped(signum,frame):raise InterruptedError('BIAS_SUFFIX_EXTERNAL_SIGNAL_'+str(signum))
 old={sig:signal.signal(sig,expired if sig==signal.SIGALRM else stopped)for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
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
  publish(ROOT/'TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'plan':binding,'claim':claim,'error':type(error).__name__+':'+str(error)[:2000],'remaining_invocations':0,'relaunch_allowed':False,
   'retained_candidate_checkpoints':retained_checkpoint_receipts(binding),'checkpoint_retention':'Native durable receipts survive; SIGKILL/hostloss may prevent terminal publication.'});raise
 finally:
  signal.setitimer(signal.ITIMER_REAL,0)
  for sig,handler in old.items():signal.signal(sig,handler)

def main():
 import sys
 parser=argparse.ArgumentParser();parser.add_argument('--plan-sha256',required=True);parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
 need(sys.argv[1:]==['--plan-sha256',args.plan_sha256]+(['--preflight']if args.preflight else[]),'EXACT_CLI')
 if args.preflight:
  p,binding,pref,live=inspect(args.plan_sha256,admitted=False);print(json.dumps({'status':'PREFLIGHT_BIAS_SUFFIX_ORDER','plan':binding,'policy':pref,'CUDA_initialized':False,'models':0,'checkpoint_payloads':0,'optimizer_updates':0}))
 else:print(json.dumps(run(args.plan_sha256),sort_keys=True))

if __name__=='__main__':main()
