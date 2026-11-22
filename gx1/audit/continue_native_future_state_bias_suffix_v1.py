"""Cold-qualified continuation of immutable003; no MAIN trajectory substitution."""
from __future__ import annotations
import argparse,copy,io,json,signal,subprocess,time
from pathlib import Path
import numpy as np
if __name__=='__main__':
 from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
 install_native_m5_direction_io_firewall()
from gx1.audit import diagnose_native_future_state_bias_suffix_v1 as previous
from gx1.audit import diagnose_native_future_state_bias_v1 as frozen
from gx1.audit import native_future_state_training_v1 as train
from gx1.audit import qualify_native_m5_full_gc_direction_v1 as core
REPO=train.REPO;POLICY=train.POLICY
STAGE='native_future_state_bias_suffix_continue_20261122_004';BASENAME='NATIVE_FUTURE_STATE_BIAS_SUFFIX_CONTINUE_20261122_004'
PLAN=REPO/'configs/research'/f'{BASENAME}.json';ROOT=Path('/home/andre2/GX1_RUNS')/BASENAME
SELF='gx1/audit/continue_native_future_state_bias_suffix_v1.py';TEST='tests/test_continue_native_future_state_bias_suffix_v1.py'
GUARD_TEST='tests/test_native_future_state_bias_suffix_continue_guard_v1.py'
ANCHOR='8ead4ba4e9c8fb4fcd35e6cd40818d6920fe16ed';GUARDS=frozen.GUARDS
PARENT={'path':str(previous.PLAN),'sha256':'7ee2dc2c3d2d4c99bc149dc66904e01fef2a874f112592eb0f2042ff38754215'}
DECISION={'path':'/home/andre2/GX1_RUNS/GX1_NATIVE_FUTURE_STATE_BIAS_CAUSE_20261122_001/ORDER_CONTINUATION004_DECISION001.json','sha256':'359e248e4e387ecc58fc156042481736e63b932a3b16ca5a0d5758cbd70eb60f'}
QUAL_START=128;RESTORE_CURSOR=192;END_CURSOR=366;NEW_CHECKPOINTS=(256,320,366)
STATE_KEYS=('model_state','optimizer_state','rng_state','scheduler_state','training','optimizer_parameter_names','progress','steps')
V3_KEYS={'schema_version','study_plan','study_arm','source_checkpoint_receipt','original_main_plan','producer_runtime','driver_requalification',
 'source_batch_order','source_batch_order_sha256','visited_source_batch_order','visited_source_batch_order_sha256','progress',
 'model_state','optimizer_state','scheduler_state','rng_state','training','optimizer_parameter_names','steps','authority'}
LIMITS={'model_constructions':1,'optimizer_constructions':2,'technical_parameter_fits':2,'model_forwards':319,'backbone_forwards':319,'readout_forwards':319,
 'backward_calls':238,'optimizer_steps':238,'checkpoint_loads':2,'checkpoint_restores':2,'checkpoint_writes':3,'checkpoint_readbacks':3,
 'capture_writes':1,'capture_readbacks':1,'feature_cache_payload_loads':1,'whole_target_archive_loads':1,'cohort_archive_loads':1,
 'autograd_calls':0,'normalization_fits':0,'prior_fits':0,'CONTROL_VAL_TEST_rows':0}
read,ref,publish,sha=train.read,train.ref,train.publish,train.sha

def need(ok,why):
 if not bool(ok):raise ValueError('NATIVE_BIAS_CONTINUE_'+why)

def validate_flags(policy,*,admitted):return previous.validate_flags(policy,admitted=admitted)

def gate_expected(binding,admitted):
 return {'status':'ADMITTED_EXACT_COLD_QUALIFIED_SUFFIX_CONTINUATION'if admitted else'PREPARED_NOT_ADMITTED','plan':binding,
  'execution_authorized':admitted,'remaining_invocations':1 if admitted else 0,'relaunch_allowed':False,
  'training_authorized':admitted,'full_epoch_training_allowed':False,'test_authorized':False}

def parent_names(main):
 train.require_source_inventory(main['source_owners'])
 return set(main['source_owners'])|{frozen.SELF,frozen.TEST,frozen.GUARD_TEST,previous.SELF,previous.TEST,previous.GUARD_TEST}|previous.RESOURCE_SOURCES

def source_bridge(parent,main):
 from gx1.contracts.gx1_capped_execution_v1 import native_future_state_bias_suffix_continuation_source_bridge
 names=parent_names(main);need(len(names)==74 and set(parent['source_owners'])==names and parent['source_base_commit']==ANCHOR,'PARENT_SOURCE_INVENTORY')
 bridge=native_future_state_bias_suffix_continuation_source_bridge(parent['source_owners'],producer_commit=ANCHOR)
 need(all(sha(REPO/k)==v for k,v in parent['source_owners'].items()if k not in GUARDS),'UNCHANGED_72_PARENT_OWNERS')
 return {'parent_commit':ANCHOR,'guard_delta':bridge,'unchanged_parent_owners':72}

def study_progress(order,cursor):
 need(type(cursor)is int and 0<cursor<=END_CURSOR and len(order)==END_CURSOR,'STUDY_CURSOR')
 return {'optimizer_updates_completed':cursor,'optimizer_updates_remaining':END_CURSOR-cursor,'next_order_position':cursor,
  'last_original_source_batch_index':order[cursor-1],'restored_original_optimizer_step':previous.START_CURSOR,
  'optimizer_step_count_after_restore':previous.START_CURSOR+cursor}

def validate_receipt(receipt,parent,cursor):
 order=parent['orders']['reordered'];need(cursor in (QUAL_START,RESTORE_CURSOR),'SOURCE_CHECKPOINT_CHOICE')
 expected=str(previous.ROOT/'reordered'/f'UPDATE{cursor:06d}.pt')
 need(receipt['schema_version']=='gx1_native_future_state_bias_suffix_probe_receipt_v3'and receipt['study_plan']==PARENT
  and receipt['study_arm']=='reordered'and receipt['main_trajectory_claim']is False
  and receipt['progress']==study_progress(order,cursor)and receipt['checkpoint']['path']==expected
  and type(receipt['checkpoint']['sha256'])is str and len(receipt['checkpoint']['sha256'])==64
  and receipt['source_batch_order_sha256']==core.state_sha(order)
  and receipt['visited_source_batch_order_sha256']==core.state_sha(order[:cursor]),'SOURCE_RECEIPT_PROVENANCE')
 return receipt

def component_proof(state):return {k:core.state_sha(state[k])for k in STATE_KEYS}

def validate_v3_checkpoint(cp,receipt,parent,cursor):
 validate_receipt(receipt,parent,cursor);order=parent['orders']['reordered']
 need(type(cp)is dict and set(cp)==V3_KEYS and cp['schema_version']==previous.CHECKPOINT_SCHEMA
  and core.state_sha(cp)==receipt['checkpoint_tree_sha256'],'AUTHENTIC_V3_FULL_TREE')
 need(cp['study_plan']==PARENT and cp['study_arm']=='reordered'and cp['source_checkpoint_receipt']==parent['inputs']['checkpoint_receipt']
  and cp['original_main_plan']==parent['inputs']['main_plan']and cp['producer_runtime']==parent['runtime']
  and cp['driver_requalification']==parent['driver_requalification']and cp['authority']=='diagnostic_only_no_promotion_or_MAIN_trajectory_claim'
  and cp['source_batch_order']==order and cp['source_batch_order_sha256']==receipt['source_batch_order_sha256']
  and cp['visited_source_batch_order']==order[:cursor]and cp['visited_source_batch_order_sha256']==receipt['visited_source_batch_order_sha256']
  and cp['progress']==receipt['progress']and cp['scheduler_state']is None and cp['training']is True
  and type(cp['steps'])is list and len(cp['steps'])==cursor
  and all(x['logical_step']==index+1 and x['skip_update']is False for x,index in zip(cp['steps'],order[:cursor])),'V3_STUDY_LINEAGE_ORDER_STEPS')
 return component_proof(cp)

def strict_restore_v3(model,optimizer,cp,receipt,parent,cursor,live):
 """Validate actual V3 lineage and state; never fabricate a MAIN checkpoint view."""
 import torch
 original=validate_v3_checkpoint(cp,receipt,parent,cursor)
 train.runtime_compatible({k:cp['producer_runtime'][k]for k in ('library_versions','gpu')},live)
 need(cp['optimizer_parameter_names']==[n for n,_ in model.named_parameters()],'RESTORE_PARAMETER_NAMES')
 expected=model.state_dict();state=cp['model_state'];need(set(state)==set(expected),'RESTORE_MODEL_KEYS')
 for name,value in state.items():
  need(isinstance(value,torch.Tensor)and value.device.type=='cpu'and value.shape==expected[name].shape
   and value.dtype==expected[name].dtype and bool(torch.isfinite(value).all()),'RESTORE_MODEL_HEADER')
 for name,value in model.named_buffers():
  if name in expected:need(torch.equal(state[name],value.detach().cpu()),'ACTUAL_NORMALIZATION_INPUT_PRIOR_IDENTITY')
 opt=cp['optimizer_state'];fresh=optimizer.state_dict();parameters=list(model.parameters())
 need(type(opt)is dict and set(opt)=={'state','param_groups'}and len(opt['param_groups'])==1
  and opt['param_groups']==fresh['param_groups'],'RESTORE_ADAMW_RECIPE')
 ids=opt['param_groups'][0]['params'];updates=cp['progress']['optimizer_step_count_after_restore']
 need(len(ids)==len(parameters)and len(set(ids))==len(ids)and all(type(i)is int for i in ids)
  and all(type(i)is int for i in opt['state'])and set(opt['state'])<=set(ids),'RESTORE_ADAMW_IDS')
 for index,item in opt['state'].items():
  parameter=parameters[ids.index(index)]
  need(type(item)is dict and set(item)=={'step','exp_avg','exp_avg_sq'}
   and all(isinstance(v,torch.Tensor)and v.device.type=='cpu'and bool(torch.isfinite(v).all())for v in item.values())
   and item['step'].numel()==1 and item['step'].dtype==torch.float32 and float(item['step']).is_integer()and 1<=float(item['step'])<=updates
   and all(item[k].shape==parameter.shape and item[k].dtype==parameter.dtype for k in ('exp_avg','exp_avg_sq'))
   and bool((item['exp_avg_sq']>=0).all()),'RESTORE_ADAMW_SLOTS')
 heads={ids[i]for i,(name,_)in enumerate(model.named_parameters())if name.startswith('readout.')}
 need(heads and heads<=set(opt['state'])and all(float(opt['state'][i]['step'])==updates for i in heads),'RESTORE_READOUT_STEP3520_OR3456')
 rng=cp['rng_state'];before=core.training_rng()
 need(type(rng)is dict and set(rng)==set(before)and type(rng['torch_cuda'])is list and len(rng['torch_cuda'])==len(before['torch_cuda'])
  and all(isinstance(v,torch.Tensor)and v.dtype==torch.uint8 and v.ndim==1 and v.device.type=='cpu'for v in rng['torch_cuda']),'RESTORE_FULL_RNG_HEADER')
 try:
  core.restore_training_rng(rng);need(core.state_sha(core.training_rng())==original['rng_state'],'RESTORE_RNG_ROUNDTRIP')
 finally:core.restore_training_rng(before)
 model.load_state_dict(state,strict=True);optimizer.load_state_dict(copy.deepcopy(opt));model.train(cp['training']);core.restore_training_rng(rng)
 for parameter,item in optimizer.state.items():
  need(item['step'].device.type=='cpu'and item['exp_avg'].device==item['exp_avg_sq'].device==parameter.device,'RESTORE_SLOT_DEVICES')
 actual=live_state(model,optimizer,cp['source_batch_order'],cursor,cp['steps'])
 need(component_proof(actual)==original and core.state_sha(cp)==receipt['checkpoint_tree_sha256'],'RESTORE_FULL_COMPONENTS_AND_CP_IMMUTABLE')
 return original

def live_state(model,optimizer,order,cursor,steps):
 return {'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'rng_state':core.training_rng(),'scheduler_state':None,
  'training':model.training,'optimizer_parameter_names':[n for n,_ in model.named_parameters()],'progress':study_progress(order,cursor),'steps':list(steps)}

def validate_qualification(actual,expected):
 a=component_proof(actual);e=component_proof(expected);need(a==e,'COLD_QUALIFICATION_ALL_COMPONENTS_BITEXACT')
 return {'bitexact':True,'components':list(STATE_KEYS),'actual':a,'expected':e,'tolerance':0.}

def archive_header_identity(binding,expected):
 import zipfile
 headers={}
 with zipfile.ZipFile(binding['path'])as archive:
  need(len(archive.namelist())==len(expected)and set(archive.namelist())=={k+'.npy'for k in expected},'ACTUAL_ARCHIVE_HEADER_INVENTORY')
  for name,original in expected.items():
   with archive.open(name+'.npy')as entry:
    version=np.lib.format.read_magic(entry)
    need(version in ((1,0),(2,0)),'ARRAY_HEADER_FORMAT')
    shape,fortran,dtype=(np.lib.format.read_array_header_1_0(entry)if version==(1,0)else np.lib.format.read_array_header_2_0(entry))
   need(not dtype.hasobject and dtype.str==original['dtype']and list(shape)==original['shape'],'ACTUAL_ARRAY_HEADER_IDENTITY')
   headers[name]={'dtype':dtype.str,'shape':list(shape),'fortran_order':bool(fortran)}
 return {'archive':binding,'header_sha256':core.state_sha(headers),'arrays':headers,'payload_bytes_verified_here':False}


def preclaim_input_identity(main,pair):
 meta=main['metadata'];norm=read(meta['normalization']);binding=read(meta['input_binding_ref']);snapshot=read(meta['source_snapshot']);dataset=read(meta['dataset'])
 cohort=read(pair['metadata']['cohort_report']['cohort_snapshot'])
 need(binding==meta['input_binding']and binding['input_normalization_sha256']==norm['contract_sha256']
  and all(binding[k]==norm[k]for k in ('feature_provider_contract_sha256','source_axes_sha256','source_clock_contract')),'ACTUAL_INPUT_NORM_PROVIDER_CLOCK_IDENTITY')
 return {'normalization':meta['normalization'],'input_binding':meta['input_binding_ref'],'source_snapshot':meta['source_snapshot'],
  'normalization_contract_sha256':norm['contract_sha256'],'input_contract_sha256':binding['contract_sha256'],
  'feature_headers':archive_header_identity(snapshot['archive'],snapshot['arrays']),
  'target_headers':archive_header_identity(dataset['arrays'],dataset['array_headers']),
  'cohort_headers':archive_header_identity(cohort['arrays'],cohort['array_headers']),
  'tensor_identity_required_not_yet_verified_before_first_forward':True,'checkpoint_payloads_deserialized_in_preflight':0}


def build_plan(*,runtime,source_base_commit=None):
 main=read(ref(train.MAIN_PLAN));train.require_source_inventory(main['source_owners'])
 parent=read(PARENT);need(set(parent['source_owners'])==parent_names(main)and len(parent['source_owners'])==74,'EXACT_PARENT_INVENTORY')
 decision=read(DECISION);need(decision['stage']==STAGE and decision['scientific_lineage']==PARENT and decision['source_anchor_commit']==ANCHOR
  and decision['execution_authorized']is False and decision['budget']['optimizer_updates']==238 and decision['budget']['model_forwards']==319,'EXACT_PREREGISTRATION')
 pair=read(parent['inputs']['paired_plan']);core.future_cuda_require_runtime(runtime);frozen.validate_function_metadata(main,pair)
 need(ref(train.MAIN_PLAN)==parent['inputs']['main_plan']and previous.make_orders(main)==parent['orders'],'ORIGINAL_MAIN_ORDER_IDENTITY')
 train.runtime_compatible({k:parent['runtime'][k]for k in ('library_versions','gpu')},runtime)
 terminal=read(decision['original_terminal']);claim=read(terminal['claim']);driver=read(decision['driver_requalification'])
 state=read(decision['control_state_proof']);logits=read(decision['control_logit_proof'])
 need(terminal['status']=='FAILED_CONSUMED'and terminal['plan']==claim['plan']==driver['study_plan']==state['study_plan']==PARENT
  and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False and claim['runtime']==driver['producer_runtime']==state['producer_runtime']
  and driver['bitexact']is True and driver['tolerance_expanded']is False and state['bitexact']is True
  and state['proof']==parent['original_reference']['expected_final_state_proof']and logits['bitexact']is True and logits['tolerance']==0.
  and all(v['bitexact']is True and v['actual_sha256']==v['original_sha256']==parent['original_reference']['frozen_capture_array_headers']['cursor3694__'+k]['sha256']for k,v in logits['heads'].items())
  and set(logits['heads'])=={'activity_logits','side_logits'},'QUALIFIED_IMMUTABLE_PARENT_CONTROL')
 refs={k:decision[k]for k in ('qualification_checkpoint_receipt','continuation_checkpoint_receipt')}
 for key,cursor in (('qualification_checkpoint_receipt',QUAL_START),('continuation_checkpoint_receipt',RESTORE_CURSOR)):
  validate_receipt(read(refs[key]),parent,cursor)
  need(any(x['receipt']==refs[key]for x in terminal['retained_candidate_checkpoints']),'DURABLE_TERMINAL_RECEIPT')
 order=parent['orders']['reordered'];need(len(order[128:192])==64 and len(order[192:])==174 and order[192]==3518 and order[-1]==3693,'EXACT_REMAINING_ORDER')
 inputs={'parent_plan':PARENT,'parent_claim':terminal['claim'],'parent_terminal':decision['original_terminal'],'decision_preregistration':DECISION,
  **refs,'driver_requalification':decision['driver_requalification'],'control_state_proof':decision['control_state_proof'],'control_logit_proof':decision['control_logit_proof'],
  'main_plan':parent['inputs']['main_plan'],'paired_plan':parent['inputs']['paired_plan'],'frozen_result':parent['inputs']['frozen_result']}
 names=set(parent['source_owners'])|{SELF,TEST,GUARD_TEST}
 return {'schema_version':'gx1_native_bias_suffix_continue_plan_v1','stage':STAGE,'inputs':inputs,'runtime':runtime,
  'source_base_commit':source_base_commit or runtime['source_commit'],'source_bridge':source_bridge(parent,main),'source_owners':{k:sha(REPO/k)for k in sorted(names)},
  'output_root':str(ROOT),'preclaim_input_identity':preclaim_input_identity(main,pair),'checkpoint_input_lineage':{'parent_plan':PARENT,'actual_receipts':refs,'main_plan':parent['inputs']['main_plan']},'independent_mac_backup':decision['independent_Mac_backup'],'original_reference':parent['original_reference'],
  'order':order,'order_sha256':core.state_sha(order),'qualification_order':order[128:192],'continuation_order':order[192:],
  'qualification':{'restore_ordinal':128,'target_ordinal':192,'updates':64,'all_step_dicts_bitexact':True,'all_state_components_bitexact':True},
  'continuation':{'restore_ordinal':192,'updates':174,'new_checkpoint_ordinals':list(NEW_CHECKPOINTS),'local_checkpoint_ordinals':[64,128,174],'eval_batches':81,'eval_rows':20727},
  'limits':LIMITS,'execution':{'class':'trainer','memory':'20G','swap':'512M','outer_seconds':3600,'internal_seconds':3570,'kill_after_seconds':10,
   'numerical_threads':8,'cpu_affinity':list(range(19)),'tasks_max':128},
  'authority':{'technical_training':True,'technical_parameter_fits':2,'optimizer_updates':238,'normalization_fits':0,'prior_fits':0,'full_epoch':False,
   'model_selection':False,'CONTROL_VAL_TEST':False,'generalization_or_economics':False},
  'scope':'Complete same003fixedpermutation after exact128to192coldqualification; no003relaunch, MAINepoch, newseed/order, CPselection, origin/generalization/edge claim.'}

def inspect(plan_sha,*,admitted):
 need(Path.cwd().resolve()==REPO and subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='work/gx1-current','CWD_BRANCH')
 binding={'path':str(PLAN),'sha256':plan_sha};plan=read(binding)
 need(plan==build_plan(runtime=plan['runtime'],source_base_commit=plan['source_base_commit']),'EXACT_PLAN')
 need(subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_base_commit'],'FROZEN_HEAD')
 pref=ref(POLICY);policy=read(pref);need(policy[STAGE]==gate_expected(binding,admitted),'ONE_USE_GATE');validate_flags(policy,admitted=admitted)
 need(not ROOT.exists()and not ROOT.is_symlink(),'FRESH_OUTPUT');live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['runtime'],live)
 return plan,binding,pref,live

def save_candidate(model,optimizer,plan,binding,steps,counts):
 cursor=len(steps);need(cursor in NEW_CHECKPOINTS,'NEW_CHECKPOINT_ORDINAL')
 payload={'schema_version':'gx1_native_bias_suffix_continuation_checkpoint_v1','execution_plan':binding,'scientific_parent_plan':PARENT,
  'source_checkpoint_receipt':plan['inputs']['continuation_checkpoint_receipt'],'producer_runtime':plan['runtime'],'source_batch_order':plan['order'],
  'source_batch_order_sha256':plan['order_sha256'],'visited_source_batch_order':plan['order'][:cursor],
  'visited_source_batch_order_sha256':core.state_sha(plan['order'][:cursor]),'continuation_updates_completed':cursor-RESTORE_CURSOR,
  **live_state(model,optimizer,plan['order'],cursor,steps),'authority':'diagnostic_only_no_MAIN_trajectory_or_promotion'}
 stem='FINAL'if cursor==END_CURSOR else f'UPDATE{cursor:06d}';tree=core.state_sha(payload)
 cpref=core.future_cuda_publish_checkpoint(ROOT/(stem+'.pt'),payload,counts)
 receipt={'schema_version':'gx1_native_bias_suffix_continuation_receipt_v1','execution_plan':binding,'scientific_parent_plan':PARENT,
  'checkpoint':cpref,'checkpoint_tree_sha256':tree,'progress':payload['progress'],'continuation_updates_completed':cursor-RESTORE_CURSOR,
  'visited_source_batch_order_sha256':payload['visited_source_batch_order_sha256'],'main_trajectory_claim':False}
 return {'receipt':publish(ROOT/(stem+'.json'),receipt),**receipt},payload

def retained_receipts(binding):
 found=[]
 for cursor in NEW_CHECKPOINTS:
  path=ROOT/('FINAL.json'if cursor==END_CURSOR else f'UPDATE{cursor:06d}.json')
  if path.exists():
   value=read(ref(path));need(value['execution_plan']==binding and value['scientific_parent_plan']==PARENT,'RETAINED_LINEAGE');found.append({'receipt':ref(path),**value})
 return found

def physical(plan,binding,live,progress=None):
 import torch
 from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
 progress={}if progress is None else progress;parent=read(PARENT);main=read(plan['inputs']['main_plan']);pair=read(plan['inputs']['paired_plan']);fr=read(plan['inputs']['frozen_result'])
 counts=dict.fromkeys(LIMITS,0);progress.update(phase='INPUT_LOAD',counts=counts,candidate_started=False)
 full=train.load_context(main);context=previous.cohort_context(full,pair);counts['feature_cache_payload_loads']=counts['whole_target_archive_loads']=counts['cohort_archive_loads']=1
 need(len(full['batches'])==previous.END_CURSOR and len(full['batches'][-1])==22,'ORIGINAL_BATCH_GEOMETRY')
 payloads={};receipts={}
 for cursor,key in ((128,'qualification_checkpoint_receipt'),(192,'continuation_checkpoint_receipt')):
  receipt=read(plan['inputs'][key]);need(ref(receipt['checkpoint']['path'])==receipt['checkpoint'],'SOURCE_PAYLOAD_BYTES')
  cp=torch.load(receipt['checkpoint']['path'],map_location='cpu',weights_only=True);counts['checkpoint_loads']+=1
  validate_v3_checkpoint(cp,receipt,parent,cursor);payloads[cursor]=cp;receipts[cursor]=receipt
 need(payloads[128]['steps']==payloads[192]['steps'][:128],'AUTHENTIC_SHARED_STEP_PREFIX')
 core.future_cuda_seed_and_runtime({'recipe':main['recipe'],'metadata':{'runtime':live}});m=main['metadata']
 model=build_native_future_state_model(input_normalization=full['normalization'],input_binding=m['input_binding'],architecture=m['architecture'],device=torch.device('cuda'),
  train_prior=full['prior'],target_contract=m['target_contract'],backbone_gradient_policy='M5_primary');counts['model_constructions']=1
 def counter(name):
  def hook(*args):counts[name]+=1;need(counts[name]<=LIMITS[name],'FORWARD_BUDGET')
  return hook
 hooks=[model.register_forward_pre_hook(counter('model_forwards')),model.backbone.register_forward_pre_hook(counter('backbone_forwards')),model.readout.register_forward_pre_hook(counter('readout_forwards'))]
 allocator=[]
 try:
  optimizer=previous._optimizer(model,main);counts['optimizer_constructions']+=1;progress['phase']='COLD_RESTORE128'
  strict_restore_v3(model,optimizer,payloads[128],receipts[128],parent,128,live);counts['checkpoint_restores']+=1;counts['technical_parameter_fits']+=1
  steps=list(payloads[128]['steps']);progress['phase']='COLD_REPLAY128_TO192'
  for ordinal,index in enumerate(plan['qualification_order'],start=129):
   progress['attempting_original_source_batch_index']=index
   actual=train.training_step(model,optimizer,train.packet(full,full['batches'][index]),main,counts,index)
   progress['qualification_updates_completed']=ordinal-128;expected=payloads[192]['steps'][ordinal-1]
   if actual!=expected:
    progress['failure_evidence']=publish(ROOT/'QUALIFICATION_STEP_MISMATCH.json',{'ordinal':ordinal,'source_batch_index':index,'actual':actual,'expected':expected,'candidate_started':False})
    need(False,'COLD_QUALIFICATION_STEP_BITEXACT')
   steps.append(actual)
  actual=live_state(model,optimizer,plan['order'],192,steps);expected=payloads[192]
  comparison={'actual':component_proof(actual),'expected':component_proof(expected)}
  progress['state_comparison']=publish(ROOT/'QUALIFICATION_STATE_COMPARISON.json',comparison)
  qualified=validate_qualification(actual,expected);qualified.update(execution_plan=binding,source_receipt=plan['inputs']['qualification_checkpoint_receipt'],target_receipt=plan['inputs']['continuation_checkpoint_receipt'])
  progress['qualification_proof']=publish(ROOT/'COLD_QUALIFICATION.json',qualified)
  del actual  # Release GPU AdamW references before second optimizer construction.
  allocator.append(train.release_phase_allocator_cache(model,optimizer,label='QUALIFIED_REPLAY_TO_SECOND_RESTORE'));del optimizer
  optimizer=previous._optimizer(model,main);counts['optimizer_constructions']+=1;progress.update(phase='CONTINUATION_RESTORE192',candidate_started=True)
  strict_restore_v3(model,optimizer,payloads[192],receipts[192],parent,192,live);counts['checkpoint_restores']+=1;counts['technical_parameter_fits']+=1
  steps=list(payloads[192]['steps']);saved=[];final_payload=None
  for ordinal,index in enumerate(plan['continuation_order'],start=193):
   progress.update(phase='CONTINUATION_TRAIN',attempting_original_source_batch_index=index)
   actual_step=train.training_step(model,optimizer,train.packet(full,full['batches'][index]),main,counts,index)
   need(actual_step['skip_update']is False and actual_step['logical_step']==index+1,'EXACT_CONTINUATION_UPDATE');steps.append(actual_step)
   progress.update(continuation_updates_completed=ordinal-192,total_scientific_suffix_updates_completed=ordinal)
   if ordinal in NEW_CHECKPOINTS:
    receipt,final_payload=save_candidate(model,optimizer,plan,binding,steps,counts);saved.append(receipt)
    if ordinal<END_CURSOR:final_payload=None
  need(len(steps)==366 and len(saved)==3,'EXACT_CONTINUATION_COMPLETION');final_tree=core.state_sha(final_payload)
  end_rng=core.state_sha(core.training_rng());reference_rng=read(plan['inputs']['control_state_proof'])['proof']['rng_state']
  need(end_rng==reference_rng,'SAME_END_RNG_WITH_SAME_BATCH_SHAPES')
  allocator.append(train.release_phase_allocator_cache(model,optimizer,label='CONTINUATION_TRAIN_TO_SCORE'));progress['phase']='FINAL_SCORE'
  score,outputs=previous._score_capture(model,context,main);need(core.state_sha(final_payload)==final_tree,'SCORE_PRESERVES_FINAL_STATE')
  metrics=frozen.metric_rows(outputs,context['truth'],m['target_contract']);arrays={'issue_ns':context['truth']['issue_ns'],**outputs};stream=io.BytesIO();np.savez_compressed(stream,**arrays)
  capture=publish(ROOT/'CAPTURE.npz',stream.getvalue());stream.close();counts['capture_writes']+=1
  with np.load(capture['path'],allow_pickle=False)as z:need(set(z.files)==set(arrays)and all(train.array_sha(z[k])==train.array_sha(v)for k,v in arrays.items()),'CAPTURE_DISK_BITEXACT')
  counts['capture_readbacks']+=1;need(counts==LIMITS,'EXACT_PHYSICAL_COUNTS');need(torch.cuda.max_memory_reserved()<=core.LARGER_VRAM_MAX,'CUDA_VRAM_CAP')
  for cursor in (128,192):need(core.state_sha(payloads[cursor])==receipts[cursor]['checkpoint_tree_sha256'],'AUTHENTIC_SOURCE_CP_IMMUTABLE')
  constants={'whole_TRAIN_factory':train.constant_prior_scores(context['prior'],context['truth'],m['target_contract']),
   'matched_SCORING_ONLY':train.constant_prior_scores(context['scoring_prior'],context['truth'],m['target_contract'])}
  need(constants==fr['report']['constant_references'],'SAME_SCORING_REFERENCES');progress['phase']='COMPLETE'
  return {'counts':counts,'end_rng_matches_qualified_control':True,'end_rng_sha256':end_rng,'qualified_control_rng_sha256':reference_rng,'cold_qualification':qualified,'new_optimizer_updates':174,'scientific_suffix_updates_completed':366,'discarded_unsaved003updates':9,
   'score':score,'metrics':metrics,'comparison_to_qualified_original_control':previous.metric_comparison(metrics,plan['original_reference']['checkpoint_report']['metrics']),
   'reused_reference':{'producer':plan['inputs']['frozen_result'],'qualified_by':plan['inputs']['driver_requalification'],'original_reference':plan['original_reference']},
   'constant_references':constants,'checkpoints':saved,'capture':capture,'capture_array_headers':{k:{'shape':list(v.shape),'dtype':v.dtype.str,'sha256':train.array_sha(v)}for k,v in arrays.items()},
   'allocator_phase_boundaries':allocator,'cuda_peak_reserved_bytes':int(torch.cuda.max_memory_reserved()),
   'causal_scope':'Same fixed003order effect and RNG stream including input-to-dropout assignment, conditional on CP3328; no origin/generalization/edge claim.'}
 finally:
  for hook in hooks:hook.remove()

def input_bookend(plan):
 for binding in plan['inputs'].values():read(binding)
 for key in ('qualification_checkpoint_receipt','continuation_checkpoint_receipt'):
  cp=read(plan['inputs'][key]);need(ref(cp['checkpoint']['path'])==cp['checkpoint'],'SOURCE_PAYLOAD_BOOKEND')
 parent=read(PARENT);previous._input_bookend(parent)

def run(plan_sha):
 from gx1.contracts.gx1_capped_execution_v1 import require_native_future_state_bias_suffix_continuation_execution
 began=time.monotonic();plan,binding,pref,live=inspect(plan_sha,admitted=True);guard=require_native_future_state_bias_suffix_continuation_execution(plan_sha256=plan_sha)
 need(guard['plan']==binding and guard['output_root']==str(ROOT)and guard['outer_seconds']==3600,'EXACT_GUARD')
 ROOT.mkdir(exist_ok=False);claim=publish(ROOT/'CLAIM.json',{'plan':binding,'policy':pref,'runtime':live,'process_identity':core.future_cpu_process_identity(),
  'remaining_invocations':0,'relaunch_allowed':False,'technical_parameter_fits':2,'optimizer_updates':238,'full_epoch':False})
 progress={'phase':'CLAIMED','candidate_started':False}
 def expired(*args):raise TimeoutError('BIAS_CONTINUE_WALL')
 def stopped(signum,frame):raise InterruptedError('BIAS_CONTINUE_EXTERNAL_SIGNAL_'+str(signum))
 old={sig:signal.signal(sig,expired if sig==signal.SIGALRM else stopped)for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
 try:
  remaining=3570-(time.monotonic()-began);need(remaining>0,'WALL_BEFORE_INPUTS');signal.setitimer(signal.ITIMER_REAL,remaining)
  report=physical(plan,binding,live,progress)
  need(ref(POLICY)==pref and ref(PLAN)==binding and all(sha(REPO/k)==v for k,v in plan['source_owners'].items()),'SOURCE_POLICY_PLAN_BOOKEND');input_bookend(plan)
  need(subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==live['source_commit']and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==live['wsl_boot_id']
   and core.physical_windows_boot_id()==live['physical_windows_boot_id'],'BOOT_HEAD_BOOKEND');need(core.common.firewall_state()['denied_attempts']==0,'M1_FIREWALL')
  result=publish(ROOT/'RESULT.json',{'status':'COMPLETE_NATIVE_BIAS_SUFFIX_CONTINUATION','plan':binding,'claim':claim,'report':report,'source_unchanged':True,'test_data_used':False,'elapsed_seconds':time.monotonic()-began})
  terminal=publish(ROOT/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,'source_unchanged':True,'test_data_used':False})
  return {'result':result,'terminal':terminal}
 except BaseException as error:
  publish(ROOT/'TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'plan':binding,'claim':claim,'error':type(error).__name__+':'+str(error)[:2000],
   'remaining_invocations':0,'relaunch_allowed':False,'progress':progress,'retained_checkpoints':retained_receipts(binding)});raise
 finally:
  signal.setitimer(signal.ITIMER_REAL,0)
  for sig,handler in old.items():signal.signal(sig,handler)

def main():
 import sys
 parser=argparse.ArgumentParser();parser.add_argument('--plan-sha256',required=True);parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
 need(sys.argv[1:]==['--plan-sha256',args.plan_sha256]+(['--preflight']if args.preflight else[]),'EXACT_CLI')
 if args.preflight:
  plan,binding,pref,live=inspect(args.plan_sha256,admitted=False);print(json.dumps({'status':'PREFLIGHT_BIAS_SUFFIX_CONTINUE','plan':binding,'policy':pref,'CUDA_initialized':False,'models':0,'checkpoint_payloads':0,'optimizer_updates':0}))
 else:print(json.dumps(run(args.plan_sha256),sort_keys=True))

if __name__=='__main__':main()
