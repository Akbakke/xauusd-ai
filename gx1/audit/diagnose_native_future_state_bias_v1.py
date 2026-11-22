"""Frozen checkpoint diagnosis; no optimizer, fitting or forecast authority."""
from __future__ import annotations
import argparse,io,json,signal,subprocess,time
from pathlib import Path
import numpy as np
if __name__=='__main__':
 from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
 install_native_m5_direction_io_firewall()
from gx1.audit import native_future_state_training_v1 as train
from gx1.audit import qualify_native_m5_full_gc_direction_v1 as core
REPO=train.REPO;POLICY=train.POLICY
STAGE='native_future_state_bias_frozen_20261122_001';BASENAME='NATIVE_FUTURE_STATE_BIAS_FROZEN_20261122_001'
PLAN=REPO/'configs/research'/f'{BASENAME}.json';ROOT=Path('/home/andre2/GX1_RUNS')/BASENAME
SELF='gx1/audit/diagnose_native_future_state_bias_v1.py';TEST='tests/test_diagnose_native_future_state_bias_v1.py'
GUARD_TEST='tests/test_native_future_state_bias_guard_v1.py'
GUARDS={'scripts/gx1_capped_run.sh','gx1/contracts/gx1_capped_execution_v1.py'}
COHORT_ROWS=20727;BATCH=256;SWAP_CURSORS=(256,3694)
LIMITS={'model_constructions':1,'model_forwards':1296,'backbone_forwards':1296,'readout_forwards':1620,
 'checkpoint_loads':16,'readout_loads':4,'optimizer_constructions':0,'optimizer_updates':0,'autograd_calls':0,
 'normalization_fits':0,'prior_fits':0,'CONTROL_VAL_TEST_rows':0}
read,ref,publish,sha=train.read,train.ref,train.publish,train.sha

def need(ok,why):
 if not bool(ok):raise ValueError('NATIVE_BIAS_FROZEN_'+why)

def expected_cursors(main):
 n=main['limits']['main']['optimizer_steps'];every=main['recipe']['checkpoint_every_optimizer_updates']
 need(n==3694 and every==256,'ORIGINAL_SCHEDULE');return [0,*range(every,n,every),n]

def source_bridge(main):
 from gx1.contracts.gx1_capped_execution_v1 import native_future_state_bias_guard_source_bridge
 train.require_source_inventory(main['source_owners'])
 need(all(sha(REPO/k)==v for k,v in main['source_owners'].items()if k not in GUARDS),'ORIGINAL_NUMERIC_SOURCES')
 return native_future_state_bias_guard_source_bridge(main['source_owners'],producer_commit=main['source_base_commit'])

def validate_function_metadata(main,pair):
 from gx1.contracts.market_state_ta_integrated_model_v1 import native_future_state_model_contract
 m=main['metadata'];p=pair['metadata']
 need(all(m[k]==p[k]for k in ('source_snapshot','normalization','input_binding_ref','input_binding','prior','target_contract','architecture')),'COHORT_FUNCTION_IDENTITY')
 norm=read(m['normalization']);binding=read(m['input_binding_ref']);prior=read(m['prior'])
 need(binding==m['input_binding']and binding['input_normalization_sha256']==norm['contract_sha256'],'PHYSICAL_NORM_INPUT_METADATA')
 expected=native_future_state_model_contract(binding,architecture=m['architecture'],train_prior=prior,target_contract=m['target_contract'],backbone_gradient_policy='M5_primary')
 need(expected==m['arm_model_contracts']['m5_primary'],'CHECKPOINT_FUNCTION_CONTRACT')

def build_plan(*,runtime,source_base_commit=None):
 main=read(ref(train.MAIN_PLAN));train.require_source_inventory(main['source_owners']);pair=read(ref(train.PAIR_PLAN));core.future_cuda_require_runtime(runtime);validate_function_metadata(main,pair)
 need(main['mode']=='whole_train_one_epoch'and main['backbone_gradient_policy']=='M5_primary'
  and main['metadata']['cohort_report']==pair['metadata']['cohort_report'],'ORIGINAL_COHORT')
 terminal=read(ref(train.MAIN_ROOT/'TERMINAL.json'));result=read(terminal['result'])
 need(terminal['status']=='COMPLETE'and terminal['exit_code']==0 and result['source_unchanged']is True,'ORIGINAL_COMPLETE')
 need(all(runtime[k]==main['metadata']['runtime'][k]for k in ('library_versions','gpu')),'ORIGINAL_RUNTIME_FUNCTION')
 inputs={'main_plan':ref(train.MAIN_PLAN),'paired_plan':ref(train.PAIR_PLAN),'main_result':terminal['result'],
  'main_terminal':ref(train.MAIN_ROOT/'TERMINAL.json')};receipts=[]
 for n in expected_cursors(main):
  path=train.MAIN_ROOT/('INITIAL.json'if n==0 else'FINAL.json'if n==3694 else f'UPDATE{n:06d}.json');b=ref(path);c=read(b)
  need(c['original_plan']==inputs['main_plan']and c['execution_plan']==inputs['main_plan']and c['producer_lane']=='main'
   and c['optimizer_updates']==c['logical_cursor']==n,'RECEIPT_IDENTITY');receipts.append(b)
 names=set(main['source_owners'])|{SELF,TEST,GUARD_TEST}
 return {'schema_version':'gx1_native_bias_frozen_plan_v1','stage':STAGE,'inputs':inputs,'checkpoint_receipts':receipts,
  'runtime':runtime,'source_base_commit':source_base_commit or runtime['source_commit'],
  'source_owners':{k:sha(REPO/k)for k in sorted(names)},'source_bridge':source_bridge(main),'output_root':str(ROOT),
  'cohort_rows':COHORT_ROWS,'eval_batch_rows':BATCH,'checkpoint_cursors':expected_cursors(main),'swap_cursors':list(SWAP_CURSORS),
  'limits':LIMITS,'execution':{'class':'trainer','memory':'20G','swap':'512M','outer_seconds':3600,'internal_seconds':3570,
   'kill_after_seconds':10,'numerical_threads':8,'cpu_affinity':list(range(19))},
  'authority':{'training':False,'model_selection':False,'optimizer':False,'CONTROL_VAL_TEST':False,'generalization_or_economics':False},
  'scope':'Same existing source-clock TRAIN cohort; all16CP and predetermined CP256/FINAL2x2 swaps. No bestcheckpoint selection.'}

def gate_expected(binding,admitted):
 return {'status':'ADMITTED_EXACT_READ_ONLY_FROZEN_DIAGNOSIS'if admitted else'PREPARED_NOT_ADMITTED','plan':binding,
  'execution_authorized':admitted,'remaining_invocations':1 if admitted else 0,'relaunch_allowed':False,'training_authorized':False,'test_authorized':False}

def inspect(plan_sha,*,admitted):
 need(Path.cwd().resolve()==REPO and subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='work/gx1-current','CWD_BRANCH')
 binding={'path':str(PLAN),'sha256':plan_sha};plan=read(binding)
 need(plan==build_plan(runtime=plan['runtime'],source_base_commit=plan['source_base_commit']),'EXACT_PLAN')
 need(subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_base_commit'],'FROZEN_HEAD')
 pref=ref(POLICY);policy=read(pref);need(policy[STAGE]==gate_expected(binding,admitted),'ONE_USE_GATE')
 need(all(policy.get(k)is False for k in ('training_enabled','native_training_enabled','native_launch_authorized','full_epoch_training_allowed','full_val_allowed')),'ALL_TRAINING_CLOSED')
 need(not ROOT.exists()and not ROOT.is_symlink(),'FRESH_OUTPUT')
 live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['runtime'],live)
 return plan,binding,pref,live

def validate_checkpoint(cp,receipt,main,result,cursor):
 keys={'schema_version','trajectory','producer','position','model_state','optimizer_state','scheduler_state','rng_state','training','optimizer_parameter_names','steps','scores'}
 need(type(cp)is dict and set(cp)==keys and cp['schema_version']=='gx1_native_future_state_training_checkpoint_v1','CHECKPOINT_SCHEMA')
 need(cp['trajectory']==result['report']['trajectory']==train.trajectory_bindings(main,'m5_primary')
  and cp['producer']=={'original_plan':result['plan'],'lane':'main','runtime':receipt['producer_runtime']}
  and cp['position']==train.position(main['schedule'],cursor)and receipt['logical_cursor']==receipt['optimizer_updates']==cursor
  and cp['scheduler_state']is None and cp['training']is True and len(cp['steps'])==cursor,'CHECKPOINT_PROVENANCE')
 need(train.checkpoint_proof(cp)==receipt['proof'],'AUTHENTIC_FULL_CHECKPOINT')

def metric_rows(outputs,truth,target,*,side_bias=None):
 from gx1.contracts.market_state_ta_integrated_objective_v1 import _native_future_state_truth
 masks=_native_future_state_truth(truth,rows=len(truth['issue_ns']),fold_upper_ns=train.UPPER,target_contract=target)
 years=truth['issue_ns'].astype('datetime64[ns]').astype('datetime64[Y]').astype(str)
 groups={'ALL':np.ones(len(years),bool),**{str(y):years==y for y in np.unique(years)}};report={}
 for group,rows in groups.items():
  cells={}
  for j,tf in enumerate(train.TF):
   cells[tf]={}
   for task,label in (('activity','activity'),('side','side_up')):
    take=rows&masks[task][:,j];n=int(take.sum())
    if not n:cells[tf][task]={'count':0,'AUC':None,'undefined_reason':'NO_OBSERVED_SUPPORT'};continue
    z=outputs[task+'_logits'][take,j].astype(np.float64);y=truth[label][take,j].astype(np.float64)
    p=np.exp(-np.logaddexp(0.,-z));residual=p-y;rank=train.binary_rank_metrics(z.astype(np.float32),y.astype(np.bool_))
    cell={'count':n,'unique_target_bars':int(np.unique(truth['target_start_ns'][take,j]).size),'actual_positive_fraction':float(y.mean()),
     'mean_probability':float(p.mean()),'probability_minus_actual':float(residual.mean()),'Brier':float(np.mean(residual**2)),
     'logloss':float(np.mean(np.logaddexp(0.,z)-y*z)),'AUC':rank['AUC'],'AUC_undefined_reason':rank['AUC_undefined_reason'],
     'mean_logit':float(z.mean()),'conditional_BCE_intercept_derivative':float(residual.mean()),'native_partial_NLL_intercept_derivative':float(residual.sum()/int(masks['activity'][rows].sum())),'Brier_intercept_derivative':float(2*np.mean(residual*p*(1-p)))}
    if task=='side'and side_bias is not None:cell.update(head_bias=float(side_bias[j]),mean_weight_dot_hidden=float(z.mean()-side_bias[j]))
    cells[tf][task]=cell
  report[group]=cells
 return report

def swap_effects(outputs,truth,target):
 from gx1.contracts.market_state_ta_integrated_objective_v1 import _native_future_state_truth
 masks=_native_future_state_truth(truth,rows=len(truth['issue_ns']),fold_upper_ns=train.UPPER,target_contract=target);report={}
 for j,tf in enumerate(train.TF):
  take=masks['side'][:,j]
  if not take.any():report[tf]={'count':0};continue
  z00,z01,z10,z11=[outputs[k]['side_logits'][take,j].astype(np.float64)for k in ('H256_R256','H256_R3694','H3694_R256','H3694_R3694')]
  head=z01-z00;hidden=z10-z00;interaction=z11-z10-z01+z00
  need(np.allclose(head+hidden+interaction,z11-z00,atol=1e-6,rtol=1e-5),'SWAP_DECOMPOSITION')
  report[tf]={'count':int(take.sum()),'head_effect_mean_logit':float(head.mean()),'hidden_effect_mean_logit':float(hidden.mean()),
   'interaction_mean_logit':float(interaction.mean()),'actual_mean_logit_delta':float((z11-z00).mean()),
   'scope':'fixedcohort functionalintervention, nottraining-order causation'}
 return report

def physical(plan,live):
 import torch
 from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
 from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
 main=read(plan['inputs']['main_plan']);pair=read(plan['inputs']['paired_plan']);result=read(plan['inputs']['main_result'])
 context=train.load_context(pair);need(len(context['source_rows'])==COHORT_ROWS,'EXACT_EXISTING_COHORT');target=main['metadata']['target_contract']
 need(context['normalization']==read(main['metadata']['normalization'])and context['prior']==read(main['metadata']['prior']),'MAIN_FUNCTION_INPUTS')
 core.future_cuda_seed_and_runtime({'recipe':main['recipe'],'metadata':{'runtime':live}});m=main['metadata']
 model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=m['input_binding'],architecture=m['architecture'],
  device=torch.device('cuda'),train_prior=context['prior'],target_contract=target,backbone_gradient_policy='M5_primary')
 counts=dict.fromkeys(LIMITS,0);counts['model_constructions']=1
 def counter(name):
  def hook(*a):counts[name]+=1;need(counts[name]<=LIMITS[name],'FORWARD_BUDGET')
  return hook
 hooks=[model.register_forward_pre_hook(counter('model_forwards')),model.backbone.register_forward_pre_hook(counter('backbone_forwards')),
  model.readout.register_forward_pre_hook(counter('readout_forwards'))]
 batches=core.all_score_batches(np.arange(COHORT_ROWS,dtype=np.int64),BATCH);need(len(batches)==81,'EXACT_EVAL_BATCHES')
 all_outputs={};selected_hidden={};selected_heads={};reports=[];diagonal={}
 try:
  for cursor,binding in zip(plan['checkpoint_cursors'],plan['checkpoint_receipts']):
   receipt=read(binding);need(ref(receipt['checkpoint']['path'])==receipt['checkpoint'],'CHECKPOINT_BYTES')
   cp=torch.load(receipt['checkpoint']['path'],map_location='cpu',weights_only=True);counts['checkpoint_loads']+=1
   validate_checkpoint(cp,receipt,main,result,cursor);model.load_state_dict(cp['model_state'],strict=True);model.eval()
   state_before=core.state_sha(model.state_dict());rng=core.training_rng();outputs={};hidden=[]
   bias=np.array([float(cp['model_state'][f'readout.side_heads.{tf}.bias'].item())for tf in train.TF])
   if cursor in SWAP_CURSORS:selected_heads[cursor]={k.removeprefix('readout.'):v.clone()for k,v in cp['model_state'].items()if k.startswith('readout.')}
   del cp
   with torch.no_grad():
    for b in batches:
     out,h,activation=core.future_cuda_forward(model,train.packet(context,b)['features'],equal_tf=True,capture=True)
     need(not torch.is_grad_enabled()and activation['batch_eval_active'],'NO_GRAD_EQUAL_TF')
     for k,v in out.items():outputs.setdefault(k,[]).append(v.detach().cpu().numpy().copy())
     if cursor in SWAP_CURSORS:hidden.append(h.numpy())
   outputs={k:np.concatenate(v)for k,v in outputs.items()};all_outputs[cursor]=outputs
   if cursor in SWAP_CURSORS:selected_hidden[cursor]=np.concatenate(hidden)
   need(core.state_sha(model.state_dict())==state_before and core.state_sha(core.training_rng())==core.state_sha(rng)
    and all(p.grad is None for p in model.parameters()),'FROZEN_MODEL_RNG_GRADS')
   if cursor==0:
    prior=objective.native_future_state_train_prior_logits(context['prior'],rows=COHORT_ROWS,target_contract=target)
    need(all(np.array_equal(outputs[k],v.numpy())for k,v in prior.items()),'INITIAL_EXACT_PRIOR')
   with torch.no_grad():proper=objective.native_future_state_score_sums(model_output={k:torch.from_numpy(v)for k,v in outputs.items()},truth=context['truth'],fold_upper_ns=train.UPPER,target_contract=target)
   reports.append({'cursor':cursor,'metrics':metric_rows(outputs,context['truth'],target,side_bias=bias),'proper_scores':proper,'output_sha256':{k:train.array_sha(v)for k,v in outputs.items()}})
   need(torch.cuda.max_memory_reserved()<=core.LARGER_VRAM_MAX,'CUDA_VRAM_CAP')
  final_state=core.state_sha(model.state_dict());swap={}
  for hc in SWAP_CURSORS:
   for rc in SWAP_CURSORS:
    label=f'H{hc}_R{rc}';model.readout.load_state_dict(selected_heads[rc],strict=True);counts['readout_loads']+=1;outs={}
    with torch.no_grad(),torch.autocast(device_type='cuda',enabled=False):
     for b in batches:
      out=model.readout(torch.from_numpy(selected_hidden[hc][b]).to('cuda'))
      for k,v in out.items():outs.setdefault(k,[]).append(v.detach().cpu().numpy().copy())
    swap[label]={k:np.concatenate(v)for k,v in outs.items()}
    if hc==rc:
     diagonal[label]={k:core.future_cuda_array_parity(all_outputs[hc][k],v)for k,v in swap[label].items()}
     need(all(x['qualified']for x in diagonal[label].values()),'SWAP_DIAGONAL_PARITY')
  need(core.state_sha(model.state_dict())==final_state and all(p.grad is None for p in model.parameters()),'FINAL_STATE_RESTORED')
  need(counts==LIMITS,'EXACT_COUNTS');arrays={'issue_ns':context['truth']['issue_ns']}
  arrays.update({f'cursor{c}__{k}':v for c,o in all_outputs.items()for k,v in o.items()})
  arrays.update({f'hidden{c}':v for c,v in selected_hidden.items()});arrays.update({f'{label}__{k}':v for label,o in swap.items()for k,v in o.items()})
  stream=io.BytesIO();np.savez_compressed(stream,**arrays);capture=publish(ROOT/'CAPTURE.npz',stream.getvalue());stream.close()
  with np.load(capture['path'],allow_pickle=False)as z:need(set(z.files)==set(arrays)and all(np.array_equal(z[k],v)for k,v in arrays.items()),'CAPTURE_DISK_READBACK')
  refs={'whole_TRAIN_factory':train.constant_prior_scores(context['prior'],context['truth'],target),'matched_SCORING_ONLY':train.constant_prior_scores(context['scoring_prior'],context['truth'],target)}
  return {'counts':counts,'rows':COHORT_ROWS,'checkpoint_metrics':reports,'constant_references':refs,
   'functional_swap_metrics':{k:metric_rows(v,context['truth'],target)for k,v in swap.items()},'functional_swap_effects':swap_effects(swap,context['truth'],target),
   'diagonal_parity':diagonal,'capture':capture,'capture_array_headers':{k:{'shape':list(v.shape),'dtype':v.dtype.str,'sha256':train.array_sha(v)}for k,v in arrays.items()},
   'cuda_peak_reserved_bytes':int(torch.cuda.max_memory_reserved()),'causal_scope':'Functional head/representation decomposition on fixedTRAINcohort; no trainingordercause or independentedge claim.'}
 finally:
  for h in hooks:h.remove()

def run(plan_sha):
 from gx1.contracts.gx1_capped_execution_v1 import require_native_future_state_bias_frozen_execution
 began=time.monotonic();plan,binding,pref,live=inspect(plan_sha,admitted=True);guard=require_native_future_state_bias_frozen_execution(plan_sha256=plan_sha)
 need(guard['plan']==binding and guard['output_root']==str(ROOT)and guard['outer_seconds']==3600,'EXACT_GUARD')
 ROOT.mkdir(exist_ok=False);claim=publish(ROOT/'CLAIM.json',{'plan':binding,'policy':pref,'runtime':live,'process_identity':core.future_cpu_process_identity(),'remaining_invocations':0,'relaunch_allowed':False})
 def expired(*a):raise TimeoutError('FROZEN_DIAGNOSIS_WALL')
 old=signal.signal(signal.SIGALRM,expired)
 try:
  remaining=3570-(time.monotonic()-began);need(remaining>0,'WALL_BEFORE_INPUTS');signal.setitimer(signal.ITIMER_REAL,remaining)
  report=physical(plan,live)
  need(ref(POLICY)==pref and ref(PLAN)==binding and all(sha(REPO/k)==v for k,v in plan['source_owners'].items()),'SOURCE_POLICY_BOOKEND')
  need(subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==live['source_commit']and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==live['wsl_boot_id']and core.physical_windows_boot_id()==live['physical_windows_boot_id'],'BOOT_HEAD_BOOKEND')
  need(core.common.firewall_state()['denied_attempts']==0,'M1_FIREWALL')
  result=publish(ROOT/'RESULT.json',{'status':'COMPLETE_NATIVE_BIAS_FROZEN_DIAGNOSIS','plan':binding,'claim':claim,'report':report,'elapsed_seconds':time.monotonic()-began,'source_unchanged':True,'test_data_used':False})
  terminal=publish(ROOT/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,'source_unchanged':True,'test_data_used':False})
  return {'result':result,'terminal':terminal}
 except BaseException as error:
  publish(ROOT/'TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'plan':binding,'claim':claim,'error':type(error).__name__+':'+str(error)[:2000],'remaining_invocations':0,'relaunch_allowed':False});raise
 finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--plan-sha256',required=True);parser.add_argument('--preflight',action='store_true');args=parser.parse_args();import sys
 need(sys.argv[1:]==['--plan-sha256',args.plan_sha256]+(['--preflight']if args.preflight else[]),'EXACT_CLI')
 if args.preflight:
  p,binding,pref,live=inspect(args.plan_sha256,admitted=False);print(json.dumps({'status':'PREFLIGHT_FROZEN_DIAGNOSIS','plan':binding,'policy':pref,'CUDA_initialized':False,'models':0,'checkpoint_payloads':0}))
 else:print(json.dumps(run(args.plan_sha256),sort_keys=True))

if __name__=='__main__':
 from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
 install_native_m5_direction_io_firewall();main()
