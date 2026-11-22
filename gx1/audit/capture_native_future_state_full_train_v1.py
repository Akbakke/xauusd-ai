"""Exact continuous native TRAIN capture from the authenticated original FINAL.

Forward-only: every E256 batch is checked against the already recorded PASS1.
No optimizer, fitting, event selection, or publication into an earlier run.
"""
from __future__ import annotations
import argparse,io,json,signal,subprocess,time
from pathlib import Path
import numpy as np
if __name__=='__main__':
 from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
 install_native_m5_direction_io_firewall()
from gx1.audit import continue_native_future_state_bias_suffix_v1 as continuation
from gx1.audit import diagnose_native_future_state_bias_suffix_v1 as previous
from gx1.audit import diagnose_native_future_state_bias_v1 as frozen
from gx1.audit import native_future_state_training_v1 as train
from gx1.audit import qualify_native_m5_full_gc_direction_v1 as core
REPO=train.REPO;POLICY=train.POLICY
STAGE='native_future_state_full_train_capture_20261122_001';BASENAME='NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_20261122_001'
PLAN=REPO/'configs/research'/f'{BASENAME}.json';ROOT=Path('/home/andre2/GX1_RUNS')/BASENAME
SELF='gx1/audit/capture_native_future_state_full_train_v1.py';TEST='tests/test_capture_native_future_state_full_train_v1.py'
GUARD_TEST='tests/test_native_future_state_full_train_capture_guard_v1.py'
ANCHOR='0e8fe2ebe117a7eed5f774db4ec09ffd299c8df0'
PARENT={'path':str(continuation.PLAN),'sha256':'6c061afd7963b787f714740813f2499e7e763e4a48ba0f13123b78222e194ba2'}
PARENT_TERMINAL={'path':str(continuation.ROOT/'TERMINAL.json'),'sha256':'22638ed8505952cc9c182c507adfb04b24696074a3d0db6769a5c5e3ac3ce763'}
CHECKPOINT_RECEIPT={'path':str(previous.ROOT/'chronological/FINAL.json'),'sha256':'290de3259981d0940fb9da23dddabbd5c314301a351410fd96dbd2a636a885a1'}
PREREGISTRATION={'path':'/home/andre2/GX1_RUNS/GX1_NATIVE_SELECTIVE_MOVE_PREPARATION_20261122_001/SELECTIVE_MOVE_PREREGISTRATION001.json','sha256':'94e93090a49b62c3a0b17fc0403db6a69993bd140e65f4fa378dd43ed52825c9'}
ROWS=354550;BATCH=256;BATCHES=1385;HEADS=('activity_logits','side_logits')
SCORE_KEYS=('rows','batch_count','batches','scores','known_activity_cells','proper_partial_NLL','initial_prior_exact','output_sha256','score_scope')
LIMITS={'model_constructions':1,'checkpoint_loads':1,'model_state_restores':1,'model_forwards':1385,'backbone_forwards':1385,'readout_forwards':1385,
 'feature_cache_payload_loads':1,'whole_target_archive_loads':1,'capture_writes':1,'capture_readbacks':1,
 'optimizer_constructions':0,'technical_parameter_fits':0,'backward_calls':0,'optimizer_steps':0,'checkpoint_writes':0,'checkpoint_readbacks':0,
 'cohort_archive_loads':0,'autograd_calls':0,'normalization_fits':0,'prior_fits':0,'CONTROL_VAL_TEST_rows':0}
read,ref,publish,sha=train.read,train.ref,train.publish,train.sha

def need(ok,why):
 if not bool(ok):raise ValueError('NATIVE_FULL_TRAIN_CAPTURE_'+why)

def validate_flags(policy,*,admitted):
 need(all(policy.get(k)is False for k in ('training_enabled','native_training_enabled','native_launch_authorized','full_epoch_training_allowed','full_val_allowed')),'ALL_TRAINING_FLAGS_FALSE')

def gate_expected(binding,admitted):
 return {'status':'ADMITTED_EXACT_READ_ONLY_FULL_TRAIN_CAPTURE'if admitted else'PREPARED_NOT_ADMITTED','plan':binding,
  'execution_authorized':admitted,'remaining_invocations':1 if admitted else 0,'relaunch_allowed':False,
  'training_authorized':False,'full_epoch_training_allowed':False,'test_authorized':False}

def source_bridge(parent,main):
 from gx1.contracts.gx1_capped_execution_v1 import native_future_state_full_train_capture_source_bridge
 names=continuation.parent_names(main)|{continuation.SELF,continuation.TEST,continuation.GUARD_TEST}
 need(len(names)==77 and set(parent['source_owners'])==names,'EXACT_PARENT77_INVENTORY')
 bridge=native_future_state_full_train_capture_source_bridge(parent['source_owners'],producer_commit=ANCHOR)
 need(all(sha(REPO/k)==v for k,v in parent['source_owners'].items()if k not in frozen.GUARDS),'UNCHANGED75_PARENT_SOURCES')
 return {'anchor_commit':ANCHOR,'guard_delta':bridge,'unchanged_parent_owners':75}

def score_batches():
 batches=core.all_score_batches(np.arange(ROWS,dtype=np.int64),BATCH)
 need(len(batches)==BATCHES and all(len(x)==BATCH for x in batches[:-1])and len(batches[-1])==246,'EXACT_FULL_E256_GEOMETRY')
 return batches

def validate_pass1(reference):
 need(reference['rows']==ROWS and reference['batch_count']==len(reference['batches'])==BATCHES and reference['initial_prior_exact']is False
  and set(reference['output_sha256'])==set(HEADS)and [x['rows']for x in reference['batches']]==[256]*(BATCHES-1)+[246]
  and all(set(x['output_sha256'])==set(HEADS)for x in reference['batches']),'ORIGINAL_PASS1_GEOMETRY_HASHES')
 return reference

def validate_checkpoint_receipt(receipt,parent003):
 order=parent003['orders']['chronological']
 need(order==list(range(3328,3694))and receipt['schema_version']=='gx1_native_future_state_bias_suffix_probe_receipt_v3'
  and receipt['study_plan']==continuation.PARENT and receipt['study_arm']=='chronological'and receipt['main_trajectory_claim']is False
  and receipt['progress']==continuation.study_progress(order,366)
  and receipt['checkpoint']['path']==str(previous.ROOT/'chronological/FINAL.pt')
  and receipt['source_batch_order_sha256']==receipt['visited_source_batch_order_sha256']==core.state_sha(order),'CHRONOLOGICAL_FINAL_RECEIPT')
 return receipt

def validate_checkpoint(cp,receipt,parent003,main,original,final_receipt):
 validate_checkpoint_receipt(receipt,parent003);order=parent003['orders']['chronological']
 need(type(cp)is dict and set(cp)==continuation.V3_KEYS and cp['schema_version']==previous.CHECKPOINT_SCHEMA
  and core.state_sha(cp)==receipt['checkpoint_tree_sha256'],'AUTHENTIC_CHRONOLOGICAL_FULL_TREE')
 need(cp['study_plan']==continuation.PARENT and cp['study_arm']=='chronological'
  and cp['source_checkpoint_receipt']==parent003['inputs']['checkpoint_receipt']and cp['original_main_plan']==original['plan']==parent003['inputs']['main_plan']
  and cp['producer_runtime']==parent003['runtime']and cp['driver_requalification']==parent003['driver_requalification']
  and cp['source_batch_order']==cp['visited_source_batch_order']==order
  and cp['source_batch_order_sha256']==cp['visited_source_batch_order_sha256']==receipt['source_batch_order_sha256']
  and cp['progress']==receipt['progress']and cp['training']is True and cp['scheduler_state']is None
  and cp['steps']==original['report']['steps'][3328:]and len(cp['steps'])==366
  and cp['authority']=='diagnostic_only_no_promotion_or_MAIN_trajectory_claim','CHRONOLOGICAL_FINAL_LINEAGE')
 proof={k:core.state_sha(cp[k])for k in ('model_state','optimizer_state','rng_state','scheduler_state','training','optimizer_parameter_names')}
 proof['position']=core.state_sha(train.position(main['schedule'],cp['progress']['optimizer_step_count_after_restore']))
 proof['steps']=core.state_sha(original['report']['steps'][:3328]+cp['steps'])
 previous.validate_reference_state(proof,final_receipt['proof'])
 return proof

def restore_model_only(model,cp,expected_model_sha):
 import torch
 need(cp['optimizer_parameter_names']==[n for n,_ in model.named_parameters()],'MODEL_PARAMETER_NAMES')
 expected=model.state_dict();state=cp['model_state'];need(set(state)==set(expected),'MODEL_STATE_KEYS')
 for name,value in state.items():
  need(isinstance(value,torch.Tensor)and value.device.type=='cpu'and value.shape==expected[name].shape
   and value.dtype==expected[name].dtype and bool(torch.isfinite(value).all()),'MODEL_TENSOR_HEADER')
 for name,value in model.named_buffers():
  if name in expected:need(torch.equal(state[name],value.detach().cpu()),'ACTUAL_NORM_INPUT_PRIOR_BUFFER_IDENTITY')
 need(core.state_sha(state)==expected_model_sha,'ORIGINAL_FINAL_MODEL_BEFORE_COPY')
 model.load_state_dict(state,strict=True);model.train(cp['training'])
 need(core.state_sha(model.state_dict())==expected_model_sha and all(p.grad is None for p in model.parameters()),'MODEL_COPY_BITEXACT')
 return {'model_state_sha256':expected_model_sha,'parameter_names_sha256':core.state_sha(cp['optimizer_parameter_names']),'normalization_input_prior_buffers_identical':True}

def validate_preregistration(registration,main_ref,result_ref,model_sha):
 capture=registration['capture'];identity=registration['model_identity']
 expected={'rows':ROWS,'batch_rows':BATCH,'forwards':BATCHES,'models':1,'checkpoint_payload_loads':1,'checkpoint_restores_model_only':1,
  'backward_calls':0,'checkpoint_writes':0,'fits':0,'normalization_fits':0,'optimizer_constructions':0,'optimizer_updates':0,'prior_fits':0,'CONTROL_VAL_TEST_rows':0,
  'fail_at_first_mismatch_no_tolerance_expansion':True,'require_each_original_PASS1_batch_output_hash_bitexact':True,'require_full_output_hash_and_proper_score_tree_bitexact':True,
  'keep_all_genuine_inputs_eight_families_D1macro_five_TF':True}
 need(registration['schema_version']=='gx1_native_selective_move_preregistration_v1'and registration['status']=='PREREGISTERED_NOT_EXECUTION_AUTHORITY'
  and registration['execution_authorized']is False and all(type(capture[k])is type(v)and capture[k]==v for k,v in expected.items())
  and identity['checkpoint_receipt']==CHECKPOINT_RECEIPT and identity['original_MAIN_plan']==main_ref
  and identity['original_MAIN_result']==result_ref and identity['model_state_sha256']==model_sha,'EXACT_PREREGISTERED_CAPTURE_SCOPE_MODEL')
 return True


def build_plan(*,runtime,source_base_commit=None):
 main_ref=ref(train.MAIN_PLAN);main=read(main_ref);train.require_source_inventory(main['source_owners'])
 parent=read(PARENT);expected=continuation.parent_names(main)|{continuation.SELF,continuation.TEST,continuation.GUARD_TEST}
 need(len(expected)==77 and set(parent['source_owners'])==expected,'PARENT_INVENTORY_BEFORE_HASHING')
 parent003=read(continuation.PARENT);pair_ref=parent['inputs']['paired_plan'];pair=read(pair_ref)
 core.future_cuda_require_runtime(runtime);train.runtime_compatible({k:parent003['runtime'][k]for k in ('library_versions','gpu')},runtime)
 frozen.validate_function_metadata(main,pair)
 mt_ref=ref(train.MAIN_ROOT/'TERMINAL.json');mt=read(mt_ref);original=read(mt['result']);reference=validate_pass1(original['report']['scores']['PASS1'])
 need(mt['status']=='COMPLETE'and mt['exit_code']==0 and mt['plan']==original['plan']==main_ref and original['source_unchanged']is True,'ORIGINAL_MAIN_COMPLETE')
 final_ref=parent003['inputs']['original_final_receipt'];final=read(final_ref)
 need(final['original_plan']==final['execution_plan']==main_ref and final['producer_lane']=='main'and final['logical_cursor']==final['optimizer_updates']==3694,'ORIGINAL_FINAL_RECEIPT')
 validate_preregistration(read(PREREGISTRATION),main_ref,mt['result'],final['proof']['model_state'])
 pt=read(PARENT_TERMINAL);pr=read(pt['result'])
 need(pt['status']=='COMPLETE'and pt['exit_code']==0 and pt['plan']==pr['plan']==PARENT and pr['source_unchanged']is True
  and pr['report']['counts']==continuation.LIMITS,'SOURCE_PARENT_COMPLETE')
 receipt=validate_checkpoint_receipt(read(CHECKPOINT_RECEIPT),parent003)
 driver=read(parent['inputs']['driver_requalification']);state=read(parent['inputs']['control_state_proof']);logits=read(parent['inputs']['control_logit_proof'])
 need(driver['bitexact']is True and driver['tolerance_expanded']is False and driver['original_reference_unchanged']is True
  and driver['study_plan']==state['study_plan']==continuation.PARENT and state['bitexact']is True
  and state['proof']=={k:final['proof'][k]for k in previous.REFERENCE_KEYS}
  and logits['bitexact']is True and logits['tolerance']==0. and set(logits['heads'])==set(HEADS)
  and all(v['bitexact']is True and v['actual_sha256']==v['original_sha256']for v in logits['heads'].values()),'QUALIFIED_ORIGINAL_FINAL_DRIVER')
 inputs={'preregistration':PREREGISTRATION,'main_plan':main_ref,'paired_plan':pair_ref,'main_terminal':mt_ref,'main_result':mt['result'],'original_final_receipt':final_ref,
  'parent_plan':PARENT,'parent_terminal':PARENT_TERMINAL,'parent_result':pt['result'],'scientific_parent_plan':continuation.PARENT,'checkpoint_receipt':CHECKPOINT_RECEIPT,
  **{k:parent['inputs'][k]for k in ('driver_requalification','control_state_proof','control_logit_proof')}}
 names=set(parent['source_owners'])|{SELF,TEST,GUARD_TEST}
 return {'schema_version':'gx1_native_future_state_full_train_capture_plan_v1','stage':STAGE,'inputs':inputs,'runtime':runtime,
  'source_base_commit':source_base_commit or runtime['source_commit'],'source_bridge':source_bridge(parent,main),'source_owners':{k:sha(REPO/k)for k in sorted(names)},
  'output_root':str(ROOT),'preclaim_input_identity':continuation.preclaim_input_identity(main,pair),
  'capture':{'rows':ROWS,'batch_rows':BATCH,'batches':BATCHES,'final_batch_rows':246,'whole_unfiltered_TRAIN_issue_ns_sha256':main['schedule']['issue_ns_sha256'],
   'whole_TRAIN_source_rows_sha256':main['schedule']['source_rows_sha256'],'arrays':['issue_ns','source_rows','selected_positions',*HEADS],
   'checkpoint_model_sha256':final['proof']['model_state'],'original_PASS1_output_sha256':reference['output_sha256'],'first_batch_mismatch_stops':True,
   'all_raw_logits_and_original_proper_scores_bitexact':True,'probability_factorization':'DOWN=a*(1-s),FLAT=1-a,UP=a*s; a=sigmoid(activity),s=sigmoid(side)'},
  'limits':LIMITS,'execution':{'class':'trainer','memory':'20G','swap':'512M','outer_seconds':3600,'internal_seconds':3570,'kill_after_seconds':10,
   'numerical_threads':8,'cpu_affinity':list(range(19)),'tasks_max':128},
  'authority':{'training':False,'optimizer':False,'technical_parameter_fits':0,'optimizer_updates':0,'normalization_fits':0,'prior_fits':0,
   'checkpoint_writes':0,'model_selection':False,'CONTROL_VAL_TEST':False,'generalization_or_economics':False},
  'input_coverage_reader_calls':{'readiness_report':1,'support_report':1},
  'scope':'Exact full continuous354550issue TRAIN capture of originalFINAL; exactly two loaded-provider readers (readiness_report once, support_report once) before checkpoint/forward; no extra archive/model/forward, eventdefinition, thresholdselection, fitting, trading or economics.'}

def inspect(plan_sha,*,admitted):
 need(Path.cwd().resolve()==REPO and subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='work/gx1-current','CWD_BRANCH')
 binding={'path':str(PLAN),'sha256':plan_sha};plan=read(binding)
 need(plan==build_plan(runtime=plan['runtime'],source_base_commit=plan['source_base_commit']),'EXACT_PLAN')
 need(subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_base_commit'],'FROZEN_HEAD')
 pref=ref(POLICY);policy=read(pref);need(policy[STAGE]==gate_expected(binding,admitted),'READ_ONLY_ONE_USE_GATE');validate_flags(policy,admitted=admitted)
 need(not ROOT.exists()and not ROOT.is_symlink(),'FRESH_OUTPUT');live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['runtime'],live)
 return plan,binding,pref,live

def validate_score(score,reference):
 need(set(score)==set(SCORE_KEYS)and all(score[k]==reference[k]for k in SCORE_KEYS),'FULL_PASS1_SCORES_BITEXACT')
 return True

def capture_score(model,context,main,reference,progress=None):
 progress={}if progress is None else progress;validate_pass1(reference);batches=score_batches();arrays={k:np.empty((ROWS,5),np.float32)for k in HEADS};seen=0
 def capture(module,args,output):
  nonlocal seen
  need(seen<BATCHES and type(output)is dict and set(output)==set(HEADS),'CAPTURE_BATCH_HEAD_INVENTORY')
  expected=reference['batches'][seen];take=batches[seen];actual={k:v.detach().cpu().numpy().copy()for k,v in output.items()}
  need(all(v.dtype==np.float32 and v.shape==(len(take),5)and np.isfinite(v).all()for v in actual.values()),'CAPTURE_BATCH_DTYPE_SHAPE_FINITE')
  hashes={k:train.array_sha(v)for k,v in actual.items()};progress.update(attempted_score_batch=seen+1,attempted_issue_rows=seen*BATCH+len(take))
  if hashes!=expected['output_sha256']:
   progress['batch_parity_failure']={'batch_ordinal':seen+1,'first_issue_index':int(take[0]),'rows':len(take),'actual':hashes,'original_PASS1':expected['output_sha256']}
   need(False,'BATCH_OUTPUT_BITEXACT_'+str(seen+1))
  for k,value in actual.items():arrays[k][take]=value
  seen+=1;progress.update(validated_score_batches=seen,validated_issue_rows=min(seen*BATCH,ROWS))
 hook=model.register_forward_hook(capture)
 try:score=core.future_cuda_score(model,(train.packet(context,b)for b in batches),main,context['prior'],initial=False,equal_tf=True)
 finally:hook.remove()
 need(seen==BATCHES and all(train.array_sha(v)==reference['output_sha256'][k]for k,v in arrays.items()),'FULL_OUTPUT_BITEXACT')
 validate_score(score,reference);return score,arrays

def publish_capture(arrays,counts):
 need(set(arrays)=={'issue_ns','source_rows','selected_positions',*HEADS},'FULL_CAPTURE_INVENTORY')
 stream=io.BytesIO();np.savez_compressed(stream,**arrays);binding=publish(ROOT/'CAPTURE.npz',stream.getvalue());stream.close();counts['capture_writes']+=1
 with np.load(binding['path'],allow_pickle=False)as z:
  need(set(z.files)==set(arrays)and all(z[k].dtype==v.dtype and z[k].shape==v.shape and train.array_sha(z[k])==train.array_sha(v)for k,v in arrays.items()),'CAPTURE_DISK_BYTE_IDENTITY')
 counts['capture_readbacks']+=1
 return {'capture':binding,'capture_array_headers':{k:{'shape':list(v.shape),'dtype':v.dtype.str,'sha256':train.array_sha(v)}for k,v in arrays.items()}}

def provider_coverage_reports(full):
 """Two readers over loaded inputs; finite values are not observation masks."""
 rows=full['source_rows'];provider=full['provider'];normalization=full['normalization']['surfaces']
 need(isinstance(rows,np.ndarray)and rows.dtype==np.int64 and rows.shape==(ROWS,),'PROVIDER_COVERAGE_SOURCE_AXIS')
 readiness=provider.readiness_report(rows)
 support=provider.support_report(rows)
 need(readiness['rows']==readiness['input_ready']==support['source_rows']==support['input_ready_rows']==ROWS
  and readiness['input_not_ready']==readiness['unmaterialized_current_rows']==0,'PROVIDER_WHOLE_TRAIN_READY')
 need(set(support['surfaces'])==set(normalization),'PROVIDER_NORMALIZATION_SURFACE_INVENTORY')
 for name,surface in support['surfaces'].items():
  fitted=normalization[name]
  need(surface['physical_rows']==fitted['fit_row_count']and surface['field_names']==fitted['field_names']
   and len(surface['observed_counts'])==len(surface['field_names'])
   and all(type(n)is int and 0<=n<=surface['physical_rows']for n in surface['observed_counts']),'PROVIDER_PHYSICAL_ROWS_FIELDS:'+name)
 return {'schema_version':'gx1_native_full_train_provider_coverage_v1','reader_calls':{'readiness_report':1,'support_report':1},
  'readiness_report':readiness,'finite_support_report':support,
  'true_observability_scope':'readiness.current_field_missingness uses the actual signal current-row boolean observed mask; not a per-history/all-TF observation census',
  'support_observed_counts_semantics':'np.isfinite counts only; zero-encoded unknowns can be finite; these are NOT true-observation counts',
  'learned_utility':'UNKNOWN_NOT_MEASURED_BY_READER_REPORTS'}


def physical(plan,binding,live,progress=None):
 import torch
 from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
 progress={}if progress is None else progress;counts=dict.fromkeys(LIMITS,0);progress.update(phase='FULL_INPUT_LOAD',counts=counts)
 main=read(plan['inputs']['main_plan']);original=read(plan['inputs']['main_result']);reference=validate_pass1(original['report']['scores']['PASS1'])
 parent003=read(plan['inputs']['scientific_parent_plan']);receipt=read(plan['inputs']['checkpoint_receipt']);final=read(plan['inputs']['original_final_receipt'])
 full=train.load_context(main);counts['feature_cache_payload_loads']=counts['whole_target_archive_loads']=1
 need(len(full['positions'])==len(full['source_rows'])==len(full['truth']['issue_ns'])==ROWS
  and train.array_sha(full['truth']['issue_ns'])==main['schedule']['issue_ns_sha256']and train.array_sha(full['source_rows'])==main['schedule']['source_rows_sha256'],'WHOLE_CONTINUOUS_TRAIN_AXIS')
 progress['phase']='PROVIDER_COVERAGE_READERS';coverage=provider_coverage_reports(full);progress['provider_coverage']=coverage
 need(ref(receipt['checkpoint']['path'])==receipt['checkpoint'],'CHECKPOINT_BYTES');cp=torch.load(receipt['checkpoint']['path'],map_location='cpu',weights_only=True);counts['checkpoint_loads']=1
 proof=validate_checkpoint(cp,receipt,parent003,main,original,final);original_tree=receipt['checkpoint_tree_sha256']
 core.future_cuda_seed_and_runtime({'recipe':main['recipe'],'metadata':{'runtime':live}});meta=main['metadata']
 model=build_native_future_state_model(input_normalization=full['normalization'],input_binding=meta['input_binding'],architecture=meta['architecture'],device=torch.device('cuda'),
  train_prior=full['prior'],target_contract=meta['target_contract'],backbone_gradient_policy='M5_primary');counts['model_constructions']=1
 restored=restore_model_only(model,cp,proof['model_state']);counts['model_state_restores']=1
 def counter(name):
  def hook(*args):counts[name]+=1;need(counts[name]<=LIMITS[name],'FORWARD_BUDGET')
  return hook
 hooks=[model.register_forward_pre_hook(counter('model_forwards')),model.backbone.register_forward_pre_hook(counter('backbone_forwards')),model.readout.register_forward_pre_hook(counter('readout_forwards'))]
 try:
  progress['phase']='FULL_E256_CAPTURE';score,outputs=capture_score(model,full,main,reference,progress)
  need(core.state_sha(cp)==original_tree and core.state_sha(model.state_dict())==proof['model_state'],'CHECKPOINT_MODEL_IMMUTABLE_BOOKEND')
  arrays={'issue_ns':full['truth']['issue_ns'],'source_rows':full['source_rows'],'selected_positions':full['positions'],**outputs}
  progress['phase']='CAPTURE_PUBLISH';capture=publish_capture(arrays,counts)
  need(counts==LIMITS,'EXACT_PHYSICAL_COUNTS');need(torch.cuda.max_memory_reserved()<=core.LARGER_VRAM_MAX,'CUDA_VRAM_CAP');progress['phase']='COMPLETE'
  return {'counts':counts,'rows':ROWS,'batch_rows':BATCH,'batch_count':BATCHES,'original_PASS1_batch_hashes_all_bitexact':True,
   'original_PASS1_full_output_and_proper_scores_bitexact':True,'score':score,'reference_producer':plan['inputs']['main_result'],
   'model_state_restored':restored,'original_FINAL_components':proof,'input_identity':plan['preclaim_input_identity'],'provider_coverage':coverage,**capture,
   'cuda_peak_reserved_bytes':int(torch.cuda.max_memory_reserved()),'scope':'Complete TRAIN rawlogit capture only; selectivity/timing/eventanalysis not performed; generalization and economics unproven.'}
 finally:
  for hook in hooks:hook.remove()

def input_bookend(plan):
 for binding in plan['inputs'].values():read(binding)
 receipt=read(plan['inputs']['checkpoint_receipt']);need(ref(receipt['checkpoint']['path'])==receipt['checkpoint'],'CHECKPOINT_PAYLOAD_BOOKEND')
 parent003=read(plan['inputs']['scientific_parent_plan']);previous._input_bookend(parent003)

def run(plan_sha):
 from gx1.contracts.gx1_capped_execution_v1 import require_native_future_state_full_train_capture_execution
 began=time.monotonic();plan,binding,pref,live=inspect(plan_sha,admitted=True);guard=require_native_future_state_full_train_capture_execution(plan_sha256=plan_sha)
 need(guard['plan']==binding and guard['output_root']==str(ROOT)and guard['outer_seconds']==3600,'EXACT_GUARD')
 ROOT.mkdir(exist_ok=False);claim=publish(ROOT/'CLAIM.json',{'plan':binding,'policy':pref,'runtime':live,'process_identity':core.future_cpu_process_identity(),
  'remaining_invocations':0,'relaunch_allowed':False,'technical_parameter_fits':0,'optimizer_updates':0,'full_epoch':False})
 progress={'phase':'CLAIMED','validated_score_batches':0,'validated_issue_rows':0}
 def expired(*args):raise TimeoutError('FULL_TRAIN_CAPTURE_WALL')
 def stopped(signum,frame):raise InterruptedError('FULL_TRAIN_CAPTURE_EXTERNAL_SIGNAL_'+str(signum))
 old={sig:signal.signal(sig,expired if sig==signal.SIGALRM else stopped)for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
 try:
  remaining=3570-(time.monotonic()-began);need(remaining>0,'WALL_BEFORE_INPUTS');signal.setitimer(signal.ITIMER_REAL,remaining)
  report=physical(plan,binding,live,progress)
  need(ref(POLICY)==pref and ref(PLAN)==binding and all(sha(REPO/k)==v for k,v in plan['source_owners'].items()),'SOURCE_POLICY_PLAN_BOOKEND');input_bookend(plan)
  need(subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==live['source_commit']and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==live['wsl_boot_id']
   and core.physical_windows_boot_id()==live['physical_windows_boot_id'],'BOOT_HEAD_BOOKEND');need(core.common.firewall_state()['denied_attempts']==0,'M1_FIREWALL')
  result=publish(ROOT/'RESULT.json',{'status':'COMPLETE_NATIVE_FULL_TRAIN_CAPTURE','plan':binding,'claim':claim,'report':report,'source_unchanged':True,'test_data_used':False,'elapsed_seconds':time.monotonic()-began})
  terminal=publish(ROOT/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,'source_unchanged':True,'test_data_used':False})
  return {'result':result,'terminal':terminal}
 except BaseException as error:
  publish(ROOT/'TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'plan':binding,'claim':claim,'error':type(error).__name__+':'+str(error)[:2000],
   'remaining_invocations':0,'relaunch_allowed':False,'progress':progress});raise
 finally:
  signal.setitimer(signal.ITIMER_REAL,0)
  for sig,handler in old.items():signal.signal(sig,handler)

def main():
 import sys
 parser=argparse.ArgumentParser();parser.add_argument('--plan-sha256',required=True);parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
 need(sys.argv[1:]==['--plan-sha256',args.plan_sha256]+(['--preflight']if args.preflight else[]),'EXACT_CLI')
 if args.preflight:
  plan,binding,pref,live=inspect(args.plan_sha256,admitted=False);print(json.dumps({'status':'PREFLIGHT_FULL_TRAIN_CAPTURE','plan':binding,'policy':pref,'CUDA_initialized':False,'models':0,'checkpoint_payloads':0,'optimizer_updates':0}))
 else:print(json.dumps(run(args.plan_sha256),sort_keys=True))

if __name__=='__main__':main()
