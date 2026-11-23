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

def source_bridge(main,*,driver_probe=False):
 from gx1.contracts.gx1_capped_execution_v1 import native_future_state_bias_guard_source_bridge
 train.require_source_inventory(main['source_owners'])
 if driver_probe:
  proof=native_future_state_bias_guard_source_bridge(main['source_owners'],producer_commit=main['source_base_commit'],driver_requalification=True)
  exempt=GUARDS|{'gx1/audit/native_future_state_training_v1.py'}
  need(all(sha(REPO/k)==v for k,v in main['source_owners'].items()if k not in exempt),'ORIGINAL_59_NUMERIC_SOURCES')
  return proof
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

# Separate CPU-only projection-moment probe; the original frozen study stays consumed.
MOMENT_STAGE='native_future_state_projection_moment_20261123_001'
MOMENT_BASENAME='NATIVE_FUTURE_STATE_PROJECTION_MOMENT_20261123_001'
MOMENT_PLAN=REPO/'configs/research'/(MOMENT_BASENAME+'.json');MOMENT_ROOT=Path('/home/andre2/GX1_RUNS')/MOMENT_BASENAME
MOMENT_EROOT=Path('/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001')
MOMENT_CAPTURE=('issue_ns','hidden3694','cursor3694__side_logits')
MOMENT_TRUTH=('issue_ns','truth__activity_known','truth__side_known','truth__side_up')
MOMENT_EXECUTION={'class':'audit','memory':'4G','swap':'512M','numerical_threads':1,'outer_seconds':300,'internal_seconds':270,'kill_after_seconds':10}
MOMENT_BUDGET={'checkpoint_loads':1,'checkpoint_bytes':97898135,'capture_archive_loads':1,'capture_fields':list(MOMENT_CAPTURE),
 'cohort_archive_loads':1,'cohort_fields':list(MOMENT_TRUTH),'ndarray_bytes_decoded':54118197,'rows':20727,'timeframes':5,'hidden_width':128,
 'technical_zero_side_gradient_arms':2,'analytical_delta_logit_projections':10,'model_constructions':0,'model_forwards':0,'readout_forwards':0,
 'optimizer_constructions':0,'optimizer_steps':0,'applied_parameter_updates':0,'autograd_calls':0,'fits':0,'calibrator_fits':0,'features':0,'source_prices':0,'new_labels':0,'HTTP_calls':0,'VAL_TEST_rows':0}


def moment_zero_gradient_delta(p,m,v,step,options,*,reset_first_moment=False):
 p,m,v=(np.asarray(x,dtype=np.float64)for x in (p,m,v));need(p.shape==m.shape==v.shape and all(np.isfinite(x).all()for x in (p,m,v))and np.all(v>=0),'MOMENT_ARRAYS')
 need(type(step)is int and step>=0 and type(reset_first_moment)is bool,'MOMENT_STEP')
 lr,eps,decay=options['lr'],options['eps'],options['weight_decay'];beta1,beta2=options['betas']
 need(all(np.isfinite(x)for x in (lr,eps,decay,beta1,beta2))and lr>=0 and eps>0 and decay>=0 and 0<=beta1<1 and 0<=beta2<1
  and options.get('maximize',False)is False and options.get('amsgrad',False)is False,'MOMENT_OPTIONS')
 t=step+1;m_next=np.zeros_like(m)if reset_first_moment else beta1*m;v_next=beta2*v
 decay_delta=-lr*decay*p;moment_delta=-lr*(m_next/(1-beta1**t))/(np.sqrt(v_next/(1-beta2**t))+eps)
 return {'delta':decay_delta+moment_delta,'decay_delta':decay_delta,'moment_delta':moment_delta}


def moment_extract_heads(cp,recipe,*,width=128):
 import torch
 opt=cp['optimizer_state'];need(type(opt)is dict and set(opt)=={'state','param_groups'}and len(opt['param_groups'])==1,'MOMENT_OPTIMIZER_SCHEMA')
 group=opt['param_groups'][0];names=cp['optimizer_parameter_names'];ids=group['params'];need(type(names)is list and all(type(n)is str for n in names)and len(set(names))==len(names)
  and type(ids)is list and len(ids)==len(names)and len(set(ids))==len(ids)and all(type(i)is int for i in ids)
  and set(group)==set(recipe)|{'params'},'MOMENT_PARAMETER_ID_ORDER')
 for k,value in recipe.items():need((tuple(group[k])==tuple(value)if k=='betas'else type(group[k])is type(value)and group[k]==value),'MOMENT_ACTUAL_RECIPE_'+k)
 need(group['amsgrad']is False and group['maximize']is False and cp['scheduler_state']is None,'MOMENT_SUPPORTED_ADAMW')
 options={k:group[k]for k in recipe};heads={};updates=cp['position']['optimizer_updates']
 for tf in train.TF:
  head={'step':None,'options':options.copy(),'parameter_names':{},'parameter_ids':{},'tensor_state_sha256':{}}
  for kind,shape in (('weight',(1,width)),('bias',(1,))):
   name=f'readout.side_heads.{tf}.{kind}';need(name in names and name in cp['model_state'],'MOMENT_ACTUAL_PARAMETER_NAME')
   index=ids[names.index(name)];slot=opt['state'][index];parameter=cp['model_state'][name]
   need(type(slot)is dict and set(slot)=={'step','exp_avg','exp_avg_sq'}and all(isinstance(x,torch.Tensor)and x.device.type=='cpu'and not x.requires_grad and bool(torch.isfinite(x).all())for x in (parameter,*slot.values())),'MOMENT_CPU_TENSORS')
   step=slot['step'];need(step.dtype==torch.float32 and step.numel()==1 and float(step).is_integer()and float(step)==updates and type(updates)is int and updates>0,'MOMENT_ACTUAL_STEP')
   need(parameter.dtype==torch.float32 and tuple(parameter.shape)==shape and all(slot[k].dtype==torch.float32 and slot[k].shape==parameter.shape for k in ('exp_avg','exp_avg_sq'))
    and bool((slot['exp_avg_sq']>=0).all()),'MOMENT_PARAMETER_SLOTS')
   head['step']=int(float(step));head['parameter_names'][kind]=name;head['parameter_ids'][kind]=index
   for label,tensor in ((kind,parameter),(kind+'_exp_avg',slot['exp_avg']),(kind+'_exp_avg_sq',slot['exp_avg_sq'])):
    head[label]=tensor.detach().numpy().astype(np.float64,copy=True).reshape(-1);head['tensor_state_sha256'][label]=core.state_sha(tensor)
  heads[tf]=head
 return heads


def moment_projection_report(hidden,logits,activity_known,side_known,side_up,heads):
 n=len(logits);need(hidden.dtype==np.float32 and hidden.ndim==3 and hidden.shape[:2]==(n,5)and logits.dtype==np.float32 and logits.shape==(n,5)
  and np.isfinite(hidden).all()and np.isfinite(logits).all()and all(a.dtype==np.bool_ and a.shape==(n,5)for a in (activity_known,side_known,side_up))
  and not np.any(side_known&~activity_known)and not np.any(side_up&~side_known),'MOMENT_FIXED_COHORT')
 D=int(activity_known.sum());report={}
 def metrics(z,take,y):
  if not take.any():return {'count':0,'mean_q':None,'actual_UP_fraction':None,'probability_minus_actual':None,'Brier':None,'NLL':None}
  z=z[take];y=y[take].astype(np.float64);q=np.exp(-np.logaddexp(0.,-z))
  return {'count':len(z),'mean_q':float(q.mean()),'actual_UP_fraction':float(y.mean()),'probability_minus_actual':float((q-y).mean()),
   'Brier':float(np.mean((q-y)**2)),'NLL':float(np.mean(np.logaddexp(0.,z)-y*z))}
 for j,tf in enumerate(train.TF):
  h=hidden[:,j].astype(np.float64);z=logits[:,j].astype(np.float64);sk=side_known[:,j];y=side_up[:,j];head=heads[tf]
  need(head['weight'].shape==head['weight_exp_avg'].shape==head['weight_exp_avg_sq'].shape==(h.shape[1],)
   and head['bias'].shape==head['bias_exp_avg'].shape==head['bias_exp_avg_sq'].shape==(1,),'MOMENT_HEAD_WIDTH')
  baseline=metrics(z,sk,y);residual=np.exp(-np.logaddexp(0.,-z[sk]))-y[sk]
  gb=float(residual.sum()/D)if D else 0.;gw=(h[sk].T@residual)/D if D else np.zeros(h.shape[1],np.float64);arms={}
  for label,reset in (('original_m',False),('reset_m',True)):
   dw=moment_zero_gradient_delta(head['weight'],head['weight_exp_avg'],head['weight_exp_avg_sq'],head['step'],head['options'],reset_first_moment=reset)
   db=moment_zero_gradient_delta(head['bias'],head['bias_exp_avg'],head['bias_exp_avg_sq'],head['step'],head['options'],reset_first_moment=reset)
   dz=h@dw['delta']+db['delta'][0];after=metrics(z+dz,sk,y)
   arms[label]={**after,'mean_delta_side_logit_known':float(dz[sk].mean())if sk.any()else None,'mean_delta_side_logit_all_rows':float(dz.mean()),
    'projection_weight_delta_l2':float(np.linalg.norm(dw['delta'])),'bias_delta':float(db['delta'][0]),'bias_decay_delta':float(db['decay_delta'][0]),
    'bias_moment_delta':float(db['moment_delta'][0]),'projection_moment_delta_l2':float(np.linalg.norm(dw['moment_delta'])),
    'native_partial_directional_derivative':float(gw@dw['delta']+gb*db['delta'][0])if D and sk.any()else None,
    'native_partial_moment_directional_derivative':float(gw@dw['moment_delta']+gb*db['moment_delta'][0])if D and sk.any()else None,
    'mean_q_delta':after['mean_q']-baseline['mean_q']if sk.any()else None,'Brier_delta':after['Brier']-baseline['Brier']if sk.any()else None,
    'NLL_delta':after['NLL']-baseline['NLL']if sk.any()else None}
  report[tf]={'baseline':baseline,'arms':arms,'step_before':head['step'],'hypothetical_step':head['step']+1,
   'fixed_cohort_native_partial_bias_derivative':gb if D and sk.any()else None,
   'fixed_cohort_gradient_dot_first_moment':float(gw@head['weight_exp_avg']+gb*head['bias_exp_avg'][0])if D and sk.any()else None,
   'first_moment_effect_original_minus_reset':{k:arms['original_m'][k]-arms['reset_m'][k]if arms['original_m'][k]is not None else None for k in
    ('mean_delta_side_logit_known','mean_delta_side_logit_all_rows','mean_q','Brier','NLL','native_partial_directional_derivative')},
   'support_status':'OBSERVED_FIXED_REFERENCE_COHORT'if sk.any()else'NO_SUPPORT_NOT_A_LOSS_RESULT'}
 return {'per_tf':report,'activity_denominator_all_TF':D,'probability_semantics':'Conditionalq=P(UP|activity) only; no jointa*q attribution',
  'numerical_scope':'Float64 analytic AdamW equations and fixed-h delta-logit projections, not bitexact CUDA/foreach updates or a model forward.'}


def moment_build_plan(*,design,source_base_commit=None):
 import torch,sys
 from gx1.audit import audit_native_selective_moves_v1 as saved
 d=read(design);need(Path(design['path'])==MOMENT_EROOT/'PROJECTION_MOMENT_DESIGN001.json'and d['schema_version']=='gx1_native_projection_moment_design_v1'
  and d['execution_authorized']is False and d['execution']==MOMENT_EXECUTION and d['budget']==MOMENT_BUDGET,'MOMENT_LOCKED_DESIGN')
 frozen_ref=ref(ROOT/'RESULT.json');frozen=read(frozen_ref);ft_ref=ref(ROOT/'TERMINAL.json');ft=read(ft_ref);fp=read(frozen['plan'])
 need(ft['status']=='COMPLETE'and ft['exit_code']==0 and ft['result']==frozen_ref and frozen['status']=='COMPLETE_NATIVE_BIAS_FROZEN_DIAGNOSIS'
  and frozen['source_unchanged']is True and frozen['test_data_used']is False and frozen['report']['rows']==COHORT_ROWS,'MOMENT_AUTHENTIC_FROZEN_COHORT')
 main=read(fp['inputs']['main_plan']);train.require_source_inventory(main['source_owners']);pair=read(fp['inputs']['paired_plan']);validate_function_metadata(main,pair)
 mt=read(fp['inputs']['main_terminal']);original=read(fp['inputs']['main_result']);receipt_ref=ref(train.MAIN_ROOT/'FINAL.json');receipt=read(receipt_ref)
 need(mt['status']=='COMPLETE'and mt['exit_code']==0 and mt['result']==fp['inputs']['main_result']and original['plan']==fp['inputs']['main_plan']
  and original['source_unchanged']is True and receipt['checkpoint']==original['report']['final_checkpoint']
  and receipt['proof']==original['report']['final_proof']and receipt['original_plan']==receipt['execution_plan']==fp['inputs']['main_plan']
  and fp['checkpoint_receipts'][-1]==receipt_ref and receipt['producer_lane']=='main'and receipt['logical_cursor']==receipt['optimizer_updates']==3694,'MOMENT_AUTHENTIC_MAIN_FINAL')
 need(str(torch.__version__)==receipt['producer_runtime']['library_versions']['torch']and np.__version__==receipt['producer_runtime']['library_versions']['numpy'],'MOMENT_CPU_READER_LIBRARY_IDENTITY')
 cohort_ref=pair['metadata']['cohort_report']['cohort_snapshot'];cohort=read(cohort_ref);ch=frozen['report']['capture_array_headers'];th=cohort['array_headers']
 need(ch['hidden3694']['shape']==[COHORT_ROWS,5,128]and ch['hidden3694']['dtype']=='<f4'and ch['cursor3694__side_logits']['shape']==[COHORT_ROWS,5]
  and ch['issue_ns']==th['issue_ns']and main['metadata']['cohort_report']==pair['metadata']['cohort_report'],'MOMENT_SAME_CAPTURE_TRUTH')
 npbytes=sum(int(np.prod(h['shape']))*np.dtype(h['dtype']).itemsize for h in [*(ch[k]for k in MOMENT_CAPTURE),*(th[k]for k in MOMENT_TRUTH)])
 need(npbytes==54118197 and Path(receipt['checkpoint']['path']).stat().st_size==97898135,'MOMENT_PAYLOAD_BUDGET')
 for key in ('observed_parameter_attribution','batch_risk_result','qualification_requirements'):read(d[key])
 batch=read(d['batch_risk_result']);need(batch['status']=='COMPLETE_NATIVE_BATCH_RISK_ACCOUNTING'and batch['source_unchanged']is True and batch['test_data_used']is False,'MOMENT_COMPLETED_BATCH_ACCOUNTING')
 names=set(fp['source_owners'])|{SELF,TEST,saved.SELF,saved.TEST};need(all(not Path(x).is_absolute()and '..'not in Path(x).parts and Path(x).suffix in ('.py','.sh')for x in names),'MOMENT_SOURCE_NAMES')
 inputs={'design':design,**{k:fp['inputs'][k]for k in ('main_plan','paired_plan','main_result','main_terminal')},'final_receipt':receipt_ref,'checkpoint':receipt['checkpoint'],
  'frozen_result':frozen_ref,'frozen_terminal':ft_ref,'frozen_plan':frozen['plan'],'capture':frozen['report']['capture'],'cohort_snapshot':cohort_ref,'cohort_arrays':cohort['arrays'],
  **{k:d[k]for k in ('observed_parameter_attribution','batch_risk_result','qualification_requirements')}}
 return {'schema_version':'gx1_native_projection_moment_plan_v1','stage':MOMENT_STAGE,'status':'PREPARED_NOT_ADMITTED','plan_path':str(MOMENT_PLAN),'output_root':str(MOMENT_ROOT),
  'source_base_commit':source_base_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'source_owners':{n:sha(REPO/n)for n in sorted(names)},
  'inputs':inputs,'execution':MOMENT_EXECUTION,'budget':MOMENT_BUDGET,'runtime_versions':{'numpy':np.__version__,'torch':str(torch.__version__),'python':sys.version.split()[0]},'GPU_runtime_used':False,'capture_array_headers':ch,'cohort_array_headers':th,'question':d['question'],
  'lr_scope':'Use actual stored terminal optimizer-group lr for a hypotheticalnextg0 step only; terminal training schedule has no admitted next update or defined next training LR.',
  'numeric_scope':'Float64 analytic head deltas on savedFINAL hidden; no actual parameter assignment, optimizer construction, model/readout forward, or CUDA bitexact-update claim.'}


def moment_source_clean(plan):
 return (subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='work/gx1-current'
  and subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==plan['source_base_commit']
  and not subprocess.check_output(['git','status','--porcelain','--','gx1','scripts','tests'],cwd=REPO,text=True)
  and all(sha(REPO/n)==v for n,v in plan['source_owners'].items()))


def moment_gate_expected(binding,admitted):
 return {'status':'ADMITTED_EXACT_CPU_ONLY_PROJECTION_MOMENT'if admitted else'PREPARED_NOT_ADMITTED','plan':binding,
  'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':False,'test_authorized':False}


def moment_inspect(plan_sha,*,admitted):
 need(Path.cwd().resolve()==REPO and type(plan_sha)is str and len(plan_sha)==64 and all(c in '0123456789abcdef'for c in plan_sha),'MOMENT_CWD_SHA')
 binding={'path':str(MOMENT_PLAN),'sha256':plan_sha};plan=read(binding)
 need(plan==moment_build_plan(design=plan['inputs']['design'],source_base_commit=plan['source_base_commit'])and moment_source_clean(plan),'MOMENT_EXACT_PLAN_SOURCE')
 pref=ref(POLICY);policy=read(pref);need(all(policy.get(k)is False for k in ('training_enabled','native_training_enabled','native_launch_authorized','full_epoch_training_allowed','full_val_allowed'))
  and policy[MOMENT_STAGE]==moment_gate_expected(binding,admitted),'MOMENT_EXACT_GATE_ALL_TRAINING_CLOSED')
 need(not MOMENT_ROOT.exists()and not MOMENT_ROOT.is_symlink(),'MOMENT_FRESH_ROOT');return plan,binding,pref


def moment_checked_inputs(plan):
 from gx1.audit import audit_native_selective_moves_v1 as saved
 for b in plan['inputs'].values():need(ref(b['path'])==b,'MOMENT_INPUT_SHA')
 saved.archive_headers(plan['inputs']['capture'],{k:plan['capture_array_headers'][k]for k in MOMENT_CAPTURE},exact_inventory=plan['capture_array_headers'])
 saved.archive_headers(plan['inputs']['cohort_arrays'],{k:plan['cohort_array_headers'][k]for k in MOMENT_TRUTH},exact_inventory=plan['cohort_array_headers'])


def moment_calculate(plan,*,progress=None):
 import torch
 from gx1.audit import audit_native_selective_moves_v1 as saved
 torch.set_num_threads(1);need(torch.get_num_threads()==1 and not torch.cuda.is_initialized(),'MOMENT_CPU_ONLY');progress={}if progress is None else progress
 main=read(plan['inputs']['main_plan']);original=read(plan['inputs']['main_result']);receipt=read(plan['inputs']['final_receipt']);frozen=read(plan['inputs']['frozen_result'])['report']
 progress['phase']='SAFE_CPU_CHECKPOINT';cp=torch.load(plan['inputs']['checkpoint']['path'],map_location='cpu',weights_only=True);progress['checkpoint_loads']=1
 validate_checkpoint(cp,receipt,main,original,3694);heads=moment_extract_heads(cp,main['recipe']['optimizer']);need(train.checkpoint_proof(cp)==receipt['proof'],'MOMENT_CHECKPOINT_UNMUTATED');del cp
 progress['phase']='SAVED_HIDDEN_TRUTH';capture=saved.load_fields(plan['inputs']['capture'],MOMENT_CAPTURE,plan['capture_array_headers']);truth=saved.load_fields(plan['inputs']['cohort_arrays'],MOMENT_TRUTH,plan['cohort_array_headers'])
 need(np.array_equal(capture['issue_ns'],truth['issue_ns'])and len(capture['issue_ns'])==COHORT_ROWS,'MOMENT_SAME_ORIGINAL_COHORT')
 progress['capture_fields_decoded']=list(MOMENT_CAPTURE);progress['cohort_fields_decoded']=list(MOMENT_TRUTH)
 with torch.no_grad():report=moment_projection_report(capture['hidden3694'],capture['cursor3694__side_logits'],truth['truth__activity_known'],truth['truth__side_known'],truth['truth__side_up'],heads)
 reference=frozen['checkpoint_metrics'][-1];need(reference['cursor']==3694,'MOMENT_FINAL_REFERENCE')
 for j,tf in enumerate(train.TF):
  cell=reference['metrics']['ALL'][tf]['side'];actual=report['per_tf'][tf]['baseline'];need(float(heads[tf]['bias'][0])==cell['head_bias']and actual['count']==cell['count']
   and actual['mean_q']==cell['mean_probability']and actual['Brier']==cell['Brier']and actual['NLL']==cell['logloss']
   and report['per_tf'][tf]['fixed_cohort_native_partial_bias_derivative']==cell['native_partial_NLL_intercept_derivative'],'MOMENT_EXISTING_FINAL_METRICS_'+tf)
 need(not torch.cuda.is_initialized(),'MOMENT_NO_CUDA_INITIALIZATION');progress['phase']='COMPLETE_ANALYTIC_PROBE';support=original['report']['steps'][-1]['support']['side']
 return {**report,'parameter_binding':{tf:{k:heads[tf][k]for k in ('step','options','parameter_names','parameter_ids','tensor_state_sha256')}for tf in train.TF},
  'cohort_scope':{'original_issue_rows':COHORT_ROWS,'whole_TRAIN_issue_rows':354550,'sampling':'Original immutable20727subset, not fullTRAIN; no new labels/calendar masks',
   'unique_known_side_targets_from_original_frozen_report':{tf:reference['metrics']['ALL'][tf]['side']['unique_target_bars']for tf in train.TF}},
  'authentic_full_checkpoint_proof_verified_before_and_after_extraction':True,'stored_terminal_optimizer_lr_used':heads[train.TF[0]]['options']['lr'],
  'next_actual_training_lr':'UNDEFINED_NO_ADMITTED_NEXT_TRAINING_STEP','last_original_batch_side_support':dict(zip(train.TF,support)),
  'last_original_batch_zero_side_support_TFs':[tf for tf,n in zip(train.TF,support)if n==0],
  'gradient_semantics':'Technical zero gradient tensor, not grad=None; grad=None would skip the AdamW parameter update.',
  'all5_probes':'Hypothetical next zero-side-gradient head updates; no actual next batch, parameter update or training trajectory is reconstructed.',
  'payload_counts':MOMENT_BUDGET,'limits':['The large observed term is projection weight dot representation. This probe holds h fixed and isolates only current direct-head first-moment mechanics.',
   'All5g0 probes are reported; H4/D1 also had zero side support in the original final batch. This does not turn hypothetical step3695 into an actual historical step.',
   'Conditionalq only. Activity heads and jointa*q, shared-backbone moments/Jacobians, dropout/clip gradients, full causal origin, generalization and economics are not measured.',
   'Float64 diagnostic equations are not claimed bitexact to actualCUDA/foreach kernels. Original state, targets, capture and prior are never modified.']}


def moment_publish(name,value):
 return publish(MOMENT_ROOT/name,value)


def moment_run(plan_sha):
 from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution,require_capped_lock_ancestry
 began=time.monotonic();cap=require_capped_cpu_audit_execution();lock=require_capped_lock_ancestry()
 need(cap['memory_max']==cap['memory_high']==4*1024**3 and cap['swap']==512*1024**2 and cap['numerical_threads']==1
  and lock['scope']==cap['cgroup_path'][len('/sys/fs/cgroup'):],'MOMENT_CAPS')
 plan,binding,pref=moment_inspect(plan_sha,admitted=True);moment_checked_inputs(plan);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();MOMENT_ROOT.mkdir(parents=True,exist_ok=False)
 handlers={s:signal.getsignal(s)for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)};claim=result=None;progress={'phase':'CLAIM'}
 try:
  def expired(*_):raise TimeoutError('PROJECTION_MOMENT_WALL')
  def stopped(s,*_):raise InterruptedError('PROJECTION_MOMENT_EXTERNAL_SIGNAL_'+str(s))
  signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
  claim=moment_publish('CLAIM.json',{'stage':MOMENT_STAGE,'plan':binding,'policy_at_claim':pref,'boot_id':boot,'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False})
  remaining=MOMENT_EXECUTION['internal_seconds']-(time.monotonic()-began);need(remaining>0,'MOMENT_TIME');signal.setitimer(signal.ITIMER_REAL,remaining)
  report=moment_calculate(plan,progress=progress);moment_checked_inputs(plan)
  need(moment_source_clean(plan)and ref(POLICY)==pref and ref(MOMENT_PLAN)==binding and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot,'MOMENT_SOURCE_INPUT_PLAN_POLICY_BOOT_BOOKEND')
  result=moment_publish('RESULT.json',{'status':'COMPLETE_NATIVE_PROJECTION_MOMENT','stage':MOMENT_STAGE,'plan':binding,'claim':claim,'policy_at_claim':pref,'boot_id':boot,
   'report':report,'source_unchanged':True,'test_data_used':False,'economic_edge_proven':False,'elapsed_seconds':time.monotonic()-began})
  terminal=moment_publish('TERMINAL.json',{'status':'COMPLETE','exit_code':0,'stage':MOMENT_STAGE,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False,
   'source_unchanged':True,'test_data_used':False});return {'result':result,'terminal':terminal}
 except BaseException as error:
  signal.setitimer(signal.ITIMER_REAL,0);moment_publish('TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'stage':MOMENT_STAGE,'plan':binding,'claim':claim,'result':result,
   'remaining_invocations':0,'relaunch_allowed':False,'progress':progress,'error':type(error).__name__+':'+str(error)[:2000],'test_data_used':False});raise
 finally:
  signal.setitimer(signal.ITIMER_REAL,0)
  for s,h in handlers.items():signal.signal(s,h)


def main():
 parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--plan-sha256');mode.add_argument('--projection-moment-plan-sha256');parser.add_argument('--preflight',action='store_true');args=parser.parse_args();import sys
 if args.projection_moment_plan_sha256 is not None:
  need(sys.argv[1:]==['--projection-moment-plan-sha256',args.projection_moment_plan_sha256]+(['--preflight']if args.preflight else[]),'MOMENT_EXACT_CLI')
  if args.preflight:
   p,binding,pref=moment_inspect(args.projection_moment_plan_sha256,admitted=False);moment_checked_inputs(p);print(json.dumps({'status':'PREFLIGHT_PROJECTION_MOMENT_PASS','plan':binding,'policy':pref,'claim_created':False,'checkpoint_payloads':0,'array_payloads_decoded':0,'models':0,'forwards':0,'optimizer_updates':0}))
  else:print(json.dumps(moment_run(args.projection_moment_plan_sha256),sort_keys=True))
 else:
  need(sys.argv[1:]==['--plan-sha256',args.plan_sha256]+(['--preflight']if args.preflight else[]),'EXACT_CLI')
  if args.preflight:
   p,binding,pref,live=inspect(args.plan_sha256,admitted=False);print(json.dumps({'status':'PREFLIGHT_FROZEN_DIAGNOSIS','plan':binding,'policy':pref,'CUDA_initialized':False,'models':0,'checkpoint_payloads':0}))
  else:print(json.dumps(run(args.plan_sha256),sort_keys=True))

if __name__=='__main__':
 from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
 install_native_m5_direction_io_firewall();main()
