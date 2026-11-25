"""One separately admitted native CUDA B96 route/profile, with no updates."""
from __future__ import annotations
import argparse,contextlib,copy,hashlib,json,math,os,resource,signal,subprocess,sys,time
from pathlib import Path
if __name__=='__main__':
 from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
 install_native_m5_direction_io_firewall()
import numpy as np
import torch
from gx1.audit import diagnose_native_provider_hlc_endpoint_cross_v1 as endpoint
from gx1.audit import native_provider_hlc_learning_v1 as learning
from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
readout,native,model_owner,objective=endpoint.readout,endpoint.native,endpoint.model_owner,endpoint.objective
saved,core,train=endpoint.saved,endpoint.core,endpoint.train
ref,read,reference=endpoint.ref,endpoint.read,endpoint.reference
REPO,POLICY=endpoint.REPO,endpoint.POLICY
NAME='NATIVE_PROVIDER_HLC_BLOCK_ORDER_GRADIENT_PROFILE_20261125_001';MODULE='gx1.audit.diagnose_native_provider_hlc_gradient_profile_v1'
STAGE=NAME.lower();PLAN=REPO/'configs/research'/(NAME+'.json');ROOT=Path('/home/andre2/GX1_RUNS')/NAME
SELF='gx1/audit/diagnose_native_provider_hlc_gradient_profile_v1.py';TEST='tests/test_diagnose_native_provider_hlc_gradient_profile_v1.py';GUARD_TEST='tests/test_native_provider_hlc_gradient_profile_guard_v1.py'
E=learning.E;ANCHOR='ce74532bb671af34110e357b5fdf57e19ba3ecd4'
DESIGN={'path':str(E/'V2_NONE_BLOCK_ORDER_NATIVE_GRADIENT_PROFILE_DESIGN001.json'),'sha256':'90a14be7284f8068a3f1e078abd1d65df4c145ad16b46abdcdc014eb8978f023'}
TF=train.TF;OUTPUTS=endpoint.OUTPUTS;EXECUTION=copy.deepcopy(endpoint.EXECUTION)
ENV={**endpoint.ENV,'GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS':'false','GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_PLAN_SHA256':'','GX1_NATIVE_PROVIDER_HLC_GRADIENT_PROFILE':'true','GX1_NATIVE_PROVIDER_HLC_GRADIENT_PROFILE_PLAN_SHA256':'SHA'}
SCHEMA='gx1_native_provider_hlc_gradient_profile_plan_v1'
FORWARD_PHASES=({'rows':256,'grouped':True,'equal_tf':True,'grad_enabled':False},{'rows':96,'grouped':True,'equal_tf':False,'grad_enabled':False},{'rows':96,'grouped':True,'equal_tf':False,'grad_enabled':True})
LOSS_TERMS=('canonical','activity_all',*('side_'+tf for tf in TF));Q_TERMS=tuple('mean_q_known_'+tf for tf in TF)


def need(ok,why):
 if not bool(ok):raise ValueError('PROVIDER_HLC_GRADIENT_PROFILE_'+why)


def profile_plan_budget(side_counts=None):
 b={k:0 for k in ('HTTP_calls','RNG_restores','VAL_TEST_rows','backward_calls','new_labels','normalization_fits','optimizer_constructions','optimizer_restores','optimizer_updates','prior_fits')}
 b.update(NPZ_archive_loads=3,NPZ_arrays_decoded=44,NPZ_decoded_bytes=1500935808,checkpoint_deserializations=1,model_constructions=1,model_state_installs=1,native_model_forwards=3,native_backbone_forwards=3,native_readout_forwards=3,native_forward_issue_rows=448,autograd_calls_max=12)
 if side_counts is not None:
  need(isinstance(side_counts,(list,tuple))and len(side_counts)==5 and all(type(n)is int and 0<=n<=96 for n in side_counts),'SIDE_COUNTS')
  b['actual_autograd_calls']=7+sum(n>0 for n in side_counts)
 return b


def profile_parity(reference_value,candidate):
 need(set(reference_value)==set(candidate)==set(OUTPUTS),'PARITY_HEADS')
 checks={k:core.future_cuda_array_parity(reference_value[k],candidate[k])for k in OUTPUTS}
 need(all(v['qualified']for v in checks.values()),'PARITY_BEFORE_DERIVATIVES:'+json.dumps(checks,sort_keys=True))
 return checks


def profile_component_terms(output,truth,target_contract):
 canonical=objective.native_future_state_loss(model_output=output,truth=truth,fold_upper_ns=train.UPPER,target_contract=target_contract)
 masks=objective._native_future_state_truth(truth,rows=len(output['side_logits']),fold_upper_ns=train.UPPER,target_contract=target_contract)
 D=canonical['count'];need(D>0 and not canonical['skip_update'],'FULL_ACTIVITY_DENOMINATOR')
 values={};known={}
 for task,key,label in (('activity','activity_logits','activity'),('side','side_logits','side_up')):
  z=output[key];mask=torch.as_tensor(masks[task],dtype=torch.bool,device=z.device);y=torch.as_tensor(truth[label].copy(),dtype=z.dtype,device=z.device)
  values[task]=torch.where(mask,torch.nn.functional.binary_cross_entropy_with_logits(z,y,reduction='none'),torch.zeros_like(z));known[task]=mask
 terms={'canonical':canonical['loss'],'activity_all':values['activity'].sum()/D}
 terms.update({'side_'+tf:values['side'][:,j].sum()/D for j,tf in enumerate(TF)})
 for j,tf in enumerate(TF):
  if bool(known['side'][:,j].any()):terms['mean_q_known_'+tf]=torch.sigmoid(output['side_logits'][:,j].to(torch.float64))[known['side'][:,j]].mean()
 need(all(v.ndim==0 and bool(torch.isfinite(v))for v in terms.values()),'FINITE_SCALAR_TERMS')
 return terms,{'activity':known['activity'].sum(0).cpu().tolist(),'side':known['side'].sum(0).cpu().tolist(),'activity_denominator':D}


def profile_parameter_axes(named):
 names=[n for n,_ in named];need(len(names)==len(set(names))>0,'PARAMETER_NAMES')
 axes={'full':list(range(len(names))),'backbone':[j for j,n in enumerate(names)if n.startswith('backbone.')],'readout':[j for j,n in enumerate(names)if n.startswith('readout.')]}
 need(sorted(axes['backbone']+axes['readout'])==axes['full'] and not set(axes['backbone'])&set(axes['readout']),'PARAMETER_PARTITIONS')
 for task in ('activity','side'):
  for tf in TF:
   ix=[j for j,n in enumerate(names)if n.startswith('readout.'+task+'_heads.'+tf+'.')]
   if ix:axes['head:'+task+'.'+tf]=ix
 need(sorted(i for k,v in axes.items()if k.startswith('head:')for i in v)==axes['readout'],'READOUT_HEAD_PARTITIONS')
 return axes


def profile_gradient_report(loss_gradients,q_gradients,axes,*,atol=1e-6,rtol=1e-5):
 need(atol==1e-6 and rtol==1e-5 and set(loss_gradients)==set(LOSS_TERMS),'GRADIENT_COMPONENTS_TOLERANCE')
 total=loss_gradients['canonical'];n=len(total);need(axes['full']==list(range(n)) and all(len(g)==n for g in loss_gradients.values()),'GRADIENT_AXIS')
 components=[loss_gradients[k]for k in LOSS_TERMS[1:]];squared=0.
 for j,t in enumerate(total):
  used=[g[j]for g in components if g[j]is not None]
  base=t if t is not None else used[0]if used else None
  if base is None:continue
  need(all(v is None or(v.device.type=='cpu'and v.dtype==torch.float64 and v.shape==base.shape and bool(torch.isfinite(v).all()))for v in [t,*[g[j]for g in components]]),'FINITE_CPU64_GRADIENT')
  actual=torch.zeros_like(base)if t is None else t;combined=sum(used,torch.zeros_like(base));need(torch.allclose(actual,combined,atol=atol,rtol=rtol),'CANONICAL_COMPONENT_IDENTITY')
  squared+=float((actual-combined).square().sum())
 def measure(a,b,ix):return core.shared_gradient_measures([a[i]for i in ix],[b[i]for i in ix])
 report={'component_identity':True,'canonical_L2':measure(total,total,axes['full'])['g5_l2'],'component_error_L2':math.sqrt(squared),'per_TF':{}}
 for tf in TF:
  q=q_gradients.get(tf)
  if q is None:report['per_TF'][tf]={'status':'UNKNOWN_NO_SIDE_SUPPORT','axes':{}};continue
  need(len(q)==n,'MEAN_Q_GRADIENT_AXIS');cells={}
  for axis,ix in axes.items():
   m=measure(q,total,ix);dots={k:measure(q,loss_gradients[k],ix)['cross_dot']for k in LOSS_TERMS[1:]}
   need(math.isclose(m['cross_dot'],sum(dots.values()),rel_tol=rtol,abs_tol=atol),'DOT_COMPONENT_IDENTITY')
   cells[axis]={'gQ_dot_gLoss':m['cross_dot'],'minus_gLoss_directional_derivative':-m['cross_dot'],'gQ_dot_components':dots,'gQ_L2':m['g5_l2'],'gLoss_L2':m['gaux_l2'],'cosine':m['cosine']}
  need(math.isclose(cells['full']['gQ_dot_gLoss'],cells['backbone']['gQ_dot_gLoss']+cells['readout']['gQ_dot_gLoss'],rel_tol=rtol,abs_tol=atol),'DOT_PARTITION_IDENTITY')
  report['per_TF'][tf]={'status':'MEASURED','axes':cells}
 return report


@contextlib.contextmanager
def profile_runtime_blockers():
 changes=[];attempts=[]
 def forbid(owner,name):
  if not hasattr(owner,name):return
  old=getattr(owner,name)
  def blocked(*a,**k):attempts.append(name);raise RuntimeError('GRADIENT_PROFILE_FORBIDDEN_'+name)
  changes.append((owner,name,old));setattr(owner,name,blocked)
 try:
  for owner,names in ((torch.autograd,('backward',)),(torch.Tensor,('backward',)),(torch.optim.Optimizer,('__init__','load_state_dict','step')),(train,('strict_restore',)),(core,('restore_training_rng',)),(torch,('set_rng_state',)),(torch.cuda,('set_rng_state','set_rng_state_all')),(torch.serialization,('add_safe_globals',))):
   for name in names:forbid(owner,name)
  for name in ('AdamW','Adam','SGD','RMSprop','Adagrad','Adadelta','Adamax','NAdam','RAdam','ASGD','LBFGS','SparseAdam','Rprop'):
   if hasattr(torch.optim,name):
    for method in ('__init__','step'):forbid(getattr(torch.optim,name),method)
  yield attempts
 finally:
  for owner,name,old in reversed(changes):setattr(owner,name,old)


def checked_inputs(plan):
 need('provider_response'in plan['inputs'],'EXPLICIT_PROVIDER_RESPONSE')
 for binding in plan['inputs'].values():reference(binding)
 for item in plan['input_inventory'].values():
  need(reference(item['binding']).stat().st_size==item['archive_bytes'],'ARCHIVE_BYTES');saved.archive_headers(item['binding'],item['array_headers'],exact_inventory=item['archive_inventory'])
 need(readout.checkpoint_zip_metadata(plan['inputs']['final_checkpoint'])==plan['checkpoint_zip_metadata'],'CHECKPOINT_ZIP_METADATA')


def build_plan(*,runtime,source_base_commit=None,support_ref=None):
 d=read(DESIGN);need(d['schema_version']=='gx1_block_order_native_gradient_profile_design_v1'and d['name']==NAME and d['profile_panel_index']==0 and d['resource_profile']['execution']==EXECUTION and d['parity']['atol']==1e-6 and d['parity']['rtol']==1e-5 and d['derivatives']['terms']==list(LOSS_TERMS+Q_TERMS)and d['derivatives']['max_autograd_calls']==12 and all(d[k]is False for k in ('execution_authorized','training_authorized','test_authorized','six_panel_gradient_admitted')),'LOCKED_DESIGN')
 if support_ref is None:support_ref=read(ref(POLICY))['native_twenty_step_qualification_20261123_001']['v2_None_batch_order_gradient_panel_support_result']
 need(support_ref['path']==str(E/'V2_NONE_BLOCK_ORDER_GRADIENT_PANEL_SUPPORT_RESULT001.json'),'EXACT_CENSUS_PATH')
 census=read(support_ref)
 inputs={'support_census':copy.deepcopy(support_ref),'design':DESIGN,'panel_manifest':d['panel'],'original_epoch_plan':d['actual_completed_plan'],'completed_result':d['actual_completed_result'],'completed_terminal':d['actual_completed_terminal'],'final_receipt':d['actual_checkpoint_receipt'],'final_checkpoint':d['actual_checkpoint'],'capture':d['actual_saved_FINAL_logits']}
 original=read(inputs['original_epoch_plan']);result=read(inputs['completed_result']);terminal=read(inputs['completed_terminal']);receipt=read(inputs['final_receipt']);manifest=read(inputs['panel_manifest']);report=result['report'];m=original['metadata'];execution=read(result['plan'])
 need(result['status']=='COMPLETE_NONE_BLOCK_ORDER_TRAIN_COMPARISON'and all(result[k]is True for k in ('source_inputs_policy_boot_unchanged','source_unchanged'))and result['test_data_used']is False and terminal['status']=='COMPLETE'and terminal['exit_code']==0 and terminal['result']==inputs['completed_result']and terminal['plan']==result['plan']and terminal['claim']==result['claim'],'AUTHENTIC_COMPLETED_FINAL')
 need(report['original_plan']==inputs['original_epoch_plan']and execution['metadata']['original']==original and report['final_checkpoint_receipt']==inputs['final_receipt']and report['final_logits']==inputs['capture']and report['actual_optimizer_updates']==report['logical_batches']==3694 and report['full_TRAIN_rows']==354550 and original['backbone_gradient_policy']is None and original['model_contract']['contract_sha256']==d['model_contract_sha256'],'SAME_CURRENT_NONE_FINAL')
 trajectory=learning.trajectory(original);train.require_recovery_checkpoint_proof(receipt['proof'])
 need(report['trajectory']==trajectory and receipt['original_plan']==inputs['original_epoch_plan']and receipt['execution_plan']==result['plan']and receipt['checkpoint']==inputs['final_checkpoint']and receipt['logical_cursor']==receipt['optimizer_updates']==3694 and receipt['trajectory_sha256']==trajectory['contract_sha256']and receipt['proof']['trajectory']==core.state_sha(trajectory)and receipt['proof']['model_state']==report['whole_FINAL_score']['state_sha256_before_and_after']and receipt['proof']['rng_state']==report['whole_FINAL_score']['rng_sha256_before_and_restored'],'AUTHENTIC_FINAL_PROOFS')
 need(manifest['schema_version']=='gx1_block_order_gradient_source_clock_panel_v1'and manifest['canonical_population_main_plan']==inputs['original_epoch_plan']and manifest['selection_uses_labels_predictions_support_or_model_state']is False and manifest['canonical_schedule_contract_sha256']==original['schedule']['canonical_schedule']['contract_sha256'],'BLIND_CANONICAL_PANEL')
 need(census['schema_version']=='gx1_fixed_clock_native_gradient_panel_support_v1'and census['status']=='COMPLETE_FIXED_CLOCK_SUPPORT_CENSUS_NO_NN'and census['design']==DESIGN and census['panel']==inputs['panel_manifest']and census['first_profile_panel_index']==0 and census['all_bound_file_stat_SHA_bookends_passed']is True,'AUTHENTIC_FIXED_SUPPORT_CENSUS')
 panel=manifest['panel'][0];need(panel['rows']==96 and panel['original_B96_id_zero_based']==369 and panel['start_issue_index']==35424 and panel['stop_issue_index_exclusive']==35520 and panel['anchor_start_issue_index']==35328 and panel['anchor_stop_issue_index_exclusive']==35584,'EXACT_FIRST_CLOCK_POINT')
 inputs.update(provider_response=original['inputs']['provider_response'],completed_plan=result['plan'],quality=report['final_quality'],source_snapshot=m['source_snapshot'],feature_archive=original['inputs']['feature_archive'],dataset=m['dataset'],target_arrays=original['inputs']['target_arrays'],normalization=m['normalization'],prior=m['prior'],input_binding=m['input_binding_ref'])
 expected_support=census['per_panel'][0]
 need(expected_support['original_B96_id_zero_based']==panel['original_B96_id_zero_based']and expected_support['rows']==96 and expected_support['issue_ns_sha256']==panel['issue_ns_sha256']and expected_support['source_rows_sha256_from_sealed_metadata']==panel['source_rows_sha256'],'CENSUS_FIRST_CLOCK_IDENTITY')
 need(census['first_profile_exact_autograd_calls_if_native_parity_passes']==profile_plan_budget([expected_support['per_TF'][tf]['side_known_rows']for tf in TF])['actual_autograd_calls'],'CENSUS_EXACT_AUTOGRAD_BUDGET')
 meta={k:read(inputs[k])for k in ('source_snapshot','dataset','normalization','prior','input_binding','quality')};meta.update(original_epoch_plan=original,final_receipt=receipt,trajectory=trajectory,completed_result=result,completed_plan=execution,full_quality=meta['quality'],support_census=census)
 need(meta['input_binding']==m['input_binding']and meta['normalization']['contract_sha256']==m['input_binding']['input_normalization_sha256']and meta['dataset']['contract']==m['target_contract']and meta['quality']['truth_sha256']==m['truth_sha256']==report['whole_FINAL_score']['truth_sha256']and meta['quality']['output_sha256']==report['whole_FINAL_score']['output_sha256']and meta['prior']['contract_sha256']==original['model_contract']['train_prior_sha256'],'PHYSICAL_SAME_INPUT_TARGET_PRIOR')
 core.future_cuda_require_runtime(runtime);need(all(runtime[k]==m['runtime'][k]for k in ('gpu','library_versions')),'NATIVE_RUNTIME_COMPATIBILITY')
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();need(source_base_commit in(None,head)and runtime['source_commit']==head,'CURRENT_SOURCE_COMMIT')
 inventory=endpoint.input_inventory(inputs,meta)
 from gx1.contracts.native_future_state_provider_hlc_v1 import require_provider_hlc_certificate_ref
 certificate=require_provider_hlc_certificate_ref(meta['dataset']['target_observation_certificate_ref']);inputs['certificate']={k:certificate[k]for k in ('path','sha256')};reference(inputs['certificate'])
 qualified_endpoint={'path':str(REPO/'configs/research/NATIVE_PROVIDER_HLC_NONE_ENDPOINT_CROSS_20261124_002.json'),'sha256':'c8faa4be8a0ee18491f4b43c9e06ee3c0a5e3fb9b3e6eb265efa5c1e00087220'};endpoint_plan=read(qualified_endpoint);inputs['qualified_endpoint_plan']=qualified_endpoint
 names=set(original['source_owners'])|set(endpoint_plan['source_owners'])|{endpoint.SELF,endpoint.TEST,endpoint.GUARD_TEST,readout.SELF,readout.TEST,'gx1/audit/diagnose_native_provider_hlc_gradient_v1.py','tests/test_diagnose_native_provider_hlc_gradient_v1.py',SELF,TEST,GUARD_TEST}
 for name in names-{SELF,TEST,GUARD_TEST,'scripts/gx1_capped_run.sh'}:
  need((REPO/name).read_bytes()==subprocess.check_output(['git','show',ANCHOR+':'+name],cwd=REPO),'UNCHANGED_EXISTING_NUMERIC_OWNER')
 bridge=shell_bridge()
 return {'schema_version':SCHEMA,'stage':STAGE,'status':'PREPARED_NOT_ADMITTED','plan_path':str(PLAN),'output_root':str(ROOT),'source_base_commit':head,'runtime':runtime,'source_owners':{n:ref(REPO/n)['sha256']for n in sorted(names)},'source_bridge':{'anchor':ANCHOR,'existing_numerical_owners_unchanged':True,'shell':bridge},'inputs':inputs,'input_roles_not_JSON':['final_checkpoint','feature_archive','target_arrays','capture'],'metadata':meta,'input_inventory':inventory,'input_budget':{'archives':3,'fields':44,'decoded_bytes':1500935808},'checkpoint_zip_metadata':readout.checkpoint_zip_metadata(inputs['final_checkpoint']),'budget':profile_plan_budget(),'execution':EXECUTION,'model_contract':original['model_contract'],'target_contract':m['target_contract'],'panel':panel,'parity_tolerance':{'atol':1e-6,'rtol':1e-5},'forward_phases':copy.deepcopy(d['forward_phases']),'scientific_scope':{'question':d['question'],'limits':d['limits'],'eval_point_not_historical_AdamW':True,'full20_approved':False,'later_generalization':'UNKNOWN','economic_edge':'UNKNOWN'}}


def shell_bridge():
 # The shared lock owner is unchanged; strip only this authority's five marked shell insertions.
 path='scripts/gx1_capped_run.sh';before=subprocess.check_output(['git','show',ANCHOR+':'+path],cwd=REPO);current=(REPO/path).read_text();normalized=current
 for section in ('STATE','MATCHERS','DISPATCH','NESTED'):
  start='# BEGIN GX1 PROVIDER HLC GRADIENT PROFILE ROUTE 20261125 001 '+section+'\n';end='# END GX1 PROVIDER HLC GRADIENT PROFILE ROUTE 20261125 001 '+section+'\n'
  need(normalized.count(start)==normalized.count(end)==1,'SHELL_SINGLE_ADDITIVE_SCOPE');a=normalized.index(start);z=normalized.index(end,a)+len(end);normalized=normalized[:a]+normalized[z:]
 exports='  --setenv=GX1_NATIVE_PROVIDER_HLC_GRADIENT_PROFILE="$NATIVE_PROVIDER_HLC_GRADIENT_PROFILE" \\\n  --setenv=GX1_NATIVE_PROVIDER_HLC_GRADIENT_PROFILE_PLAN_SHA256="$NATIVE_PROVIDER_HLC_GRADIENT_PROFILE_PLAN_SHA256" \\\n'
 need(normalized.count(exports)==1,'EXACT_PROFILE_EXPORTS');normalized=normalized.replace(exports,'',1)
 need(normalized.encode()==before,'UNCHANGED_OLD_SHELL_BYTES')
 return {'anchor':ANCHOR,'original_sha256':hashlib.sha256(before).hexdigest(),'current_sha256':hashlib.sha256(current.encode()).hexdigest(),'old_bytes_preserved':True}


def gate_expected(binding,admitted):return {'status':'ADMITTED_EXACT_NATIVE_GRADIENT_PROFILE'if admitted else'PREPARED_NOT_ADMITTED','plan':binding,'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,'training_authorized':False,'test_authorized':False}


def require_scope(sha,*,preflight=False,environ=None,argv=None):
 env=os.environ if environ is None else environ;wanted={k:sha if v=='SHA'else v for k,v in ENV.items()}
 need(type(sha)is str and len(sha)==64 and all(c in'0123456789abcdef'for c in sha),'CLI_SHA')
 need(all(env.get(k)==v for k,v in wanted.items())and not any(k.startswith('GX1_NATIVE_')and k not in wanted for k in env),'EXACT_EXECUTION_ENV')
 command=[str(REPO/'.venv/bin/python'),'-m',MODULE,'--plan-sha256',sha]+(['--preflight']if preflight else[])
 need((sys.orig_argv if argv is None else argv)==command and Path.cwd().resolve()==REPO and PLAN.parent.resolve()==PLAN.parent and ROOT.parent.resolve()==ROOT.parent and ROOT.parent.is_dir()and not ROOT.exists()and not ROOT.is_symlink(),'EXACT_CANONICAL_CLI_FRESH_ROOT')
 return require_guarded_cuda_trainer_execution()


def inspect(sha,*,admitted):
 binding={'path':str(PLAN),'sha256':sha};reference(binding);plan=read(binding)
 need(plan==build_plan(runtime=plan['runtime'],source_base_commit=plan['source_base_commit'],support_ref=plan['inputs']['support_census'])and readout.source_clean(plan),'EXACT_PLAN_SOURCE')
 pref=ref(POLICY);policy=read(pref);need(all(policy.get(k)is False for k in saved.FLAGS)and policy[STAGE]==gate_expected(binding,admitted),'ONE_USE_TRAINING_CLOSED')
 need(not ROOT.exists()and not ROOT.is_symlink(),'FRESH_OUTPUT');checked_inputs(plan)
 live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['runtime'],live)
 return plan,binding,pref,live


def publish(name,value):return learning.pub(ROOT,name,value)


def model_proof(model):return {**endpoint.model_proof(model),'all_buffers_sha256':core.state_sha(dict(model.named_buffers()))}


def physical(plan,live,*,progress=None):
 began=time.monotonic();progress={}if progress is None else progress;counts={k:0 for k,v in profile_plan_budget().items()if type(v)is int and k!='autograd_calls_max'};progress['counts']=counts;progress['phase']='CONTEXT'
 hooks=[];original=plan['metadata']['original_epoch_plan'];m=original['metadata'];panel=plan['panel']
 try:
  with endpoint.InputLedger(plan['input_inventory'])as ledger:
   context=native.load_context(m);capture=saved.load_fields(plan['inputs']['capture'],endpoint.CAPTURE_FIELDS,plan['input_inventory']['capture']['array_headers']);decoded=ledger.verify()
  counts.update(NPZ_archive_loads=3,NPZ_arrays_decoded=44,NPZ_decoded_bytes=sum(v['decoded_bytes']for v in decoded.values()))
  need(np.array_equal(context['truth']['issue_ns'],capture['issue_ns'])and context['truth_sha256']==m['truth_sha256'],'FULL_SAVED_CLOCK')
  a=np.arange(panel['anchor_start_issue_index'],panel['anchor_stop_issue_index_exclusive'],dtype=np.int64);take=np.arange(panel['start_issue_index'],panel['stop_issue_index_exclusive'],dtype=np.int64);offset=int(take[0]-a[0])
  need(core.digest(context['source_rows'][take])==panel['source_rows_sha256']and core.digest(context['truth']['issue_ns'][take])==panel['issue_ns_sha256'],'ACTUAL_BLIND_B96_HASHES')
  anchor_packet=native.packet(context,a);packet=native.packet(context,take);packet_sha=core.state_sha(packet);anchor_sha=core.state_sha(anchor_packet)
  masks=objective._native_future_state_truth(packet['truth'],rows=96,fold_upper_ns=train.UPPER,target_contract=m['target_contract']);expected_support=plan['metadata']['support_census']['per_panel'][0]['per_TF'];actual_support={}
  for j,tf in enumerate(TF):
   side=masks['side'][:,j];up=side&packet['truth']['side_up'][:,j];down=side&~packet['truth']['side_up'][:,j];starts=packet['truth']['target_start_ns'][:,j]
   actual_support[tf]={'activity_known_rows':int(masks['activity'][:,j].sum()),'side_known_rows':int(side.sum()),'UP_rows':int(up.sum()),'DOWN_rows':int(down.sum()),'known_side_distinct_target_starts':int(np.unique(starts[side]).size),'UP_distinct_target_starts':int(np.unique(starts[up]).size),'DOWN_distinct_target_starts':int(np.unique(starts[down]).size)}
   need(all(expected_support[tf][k]==v for k,v in actual_support[tf].items()),'ACTUAL_SUPPORT_MATCHES_IMMUTABLE_CENSUS')
  progress['actual_support_census']=actual_support
  expected_calls=plan['metadata']['support_census']['first_profile_exact_autograd_calls_if_native_parity_passes']
  expected={k:capture[k][a].copy()for k in OUTPUTS};del capture,context
  core.future_cuda_seed_and_runtime({'recipe':original['recipe'],'metadata':{'runtime':live}})
  model=model_owner.build_native_future_state_model(input_normalization=read(m['normalization']),input_binding=m['input_binding'],architecture=m['architecture'],device=torch.device('cuda'),train_prior=read(m['prior']),target_contract=m['target_contract'],backbone_gradient_policy=None);counts['model_constructions']=1;model.eval()
  cp=torch.load(plan['inputs']['final_checkpoint']['path'],map_location='cpu',weights_only=True);counts['checkpoint_deserializations']=1
  readout.validate_final(cp,plan['metadata']['final_receipt'],original,plan['inputs']['original_epoch_plan'],plan['metadata']['trajectory'],plan['inputs']['completed_plan']);install=endpoint.strict_model_install(model,cp['model_state'],counts);del cp
  before=model_proof(model);ambient=torch.is_grad_enabled()
  def counter(name):
   def hook(*_):counts[name]+=1;need(counts[name]<=3,'FORWARD_BUDGET')
   return hook
  hooks=[o.register_forward_pre_hook(counter(n))for o,n in ((model,'native_model_forwards'),(model.backbone,'native_backbone_forwards'),(model.readout,'native_readout_forwards'))]
  def forward(p,*,equal_tf,grad):
   counts['native_forward_issue_rows']+=len(p['truth']['issue_ns'])
   with torch.enable_grad()if grad else torch.no_grad():out,hidden,activation=core.future_cuda_forward(model,p['features'],grouped=True,equal_tf=equal_tf,capture=False)
   need(hidden is None and activation['grad_enabled']is grad and activation['scope_enabled']is equal_tf and activation['batch_eval_active']is equal_tf and activation['model_training']is False,'ACTUAL_FORWARD_MODE')
   need(all(v.dtype==torch.float32 and v.device.type=='cuda'and tuple(v.shape)==(len(p['truth']['issue_ns']),5)for v in out.values()),'NATIVE_CUDA_F32_HEADS')
   return out
  def arrays(out):return {k:v.detach().cpu().numpy().copy()for k,v in out.items()}
  def retain_phase(name,values):
   progress.setdefault('phase_logits',{})[name]={k:{'shape':list(v.shape),'dtype':v.dtype.str,'sha256':core.digest(v),'values':v.tolist()}for k,v in values.items()}
  with profile_runtime_blockers()as attempts:
   progress['phase']='ANCHOR_B256_NO_GRAD';anchor=forward(anchor_packet,equal_tf=True,grad=False);anchor_arrays=arrays(anchor);retain_phase('anchor_B256_no_grad',anchor_arrays);parity={'anchor_saved':profile_parity(expected,anchor_arrays)};del anchor
   progress['phase']='B96_NO_GRAD';baseline=forward(packet,equal_tf=False,grad=False);baseline_arrays=arrays(baseline);retain_phase('B96_no_grad',baseline_arrays);parity['B96_anchor']=profile_parity({k:v[offset:offset+96]for k,v in anchor_arrays.items()},baseline_arrays);parity['B96_saved']=profile_parity({k:v[offset:offset+96]for k,v in expected.items()},baseline_arrays);del baseline
   progress['phase']='B96_GRAD_ENABLED';out=forward(packet,equal_tf=False,grad=True);actual=arrays(out);retain_phase('B96_grad_enabled',actual);parity['grad_baseline']=profile_parity(baseline_arrays,actual);parity['grad_saved']=profile_parity({k:v[offset:offset+96]for k,v in expected.items()},actual);progress['parity']=parity
   need(counts['native_model_forwards']==3 and counts.get('autograd_calls',0)==0,'PARITY_BEFORE_ANY_AUTOGRAD')
   with torch.enable_grad():
    terms,support=profile_component_terms(out,packet['truth'],m['target_contract']);budget=profile_plan_budget(support['side']);named=tuple(model.named_parameters());need(all(p.requires_grad for _,p in named),'FULL_PARAMETER_AXIS_REQUIRES_GRAD');params=tuple(p for _,p in named);axes=profile_parameter_axes(named);loss_grads={};q_grads={tf:None for tf in TF};counts['autograd_calls']=0
    for j,(name,scalar)in enumerate(terms.items()):
     gradients=torch.autograd.grad(scalar,params,create_graph=False,retain_graph=j<len(terms)-1,allow_unused=True);counts['autograd_calls']+=1;need(counts['autograd_calls']<=12,'AUTOGRAD_BUDGET')
     cpu=tuple(None if g is None else g.detach().to(device='cpu',dtype=torch.float64)for g in gradients);del gradients
     need(all(g is None or bool(torch.isfinite(g).all())for g in cpu),'FINITE_PARAMETER_DERIVATIVES')
     if name in LOSS_TERMS:loss_grads[name]=cpu
     else:q_grads[name.removeprefix('mean_q_known_')]=cpu
     del cpu
    scalars={n:float(v.detach())for n,v in terms.items()};del terms,out
   gradient=profile_gradient_report(loss_grads,q_grads,axes);del loss_grads,q_grads
   need(counts['autograd_calls']==budget['actual_autograd_calls']==expected_calls and not attempts,'EXACT_DERIVATIVE_COUNTS_NO_FORBIDDEN_CALLS')
  need(before==model_proof(model)and core.state_sha(packet)==packet_sha and core.state_sha(anchor_packet)==anchor_sha and torch.is_grad_enabled()==ambient,'MODEL_RNG_BUFFER_PACKET_GRAD_BOOKEND')
  from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as body
  need(body._BATCH_EQUAL_LENGTH_MTF_EVAL.get()is False and body._MODEL_FINITE_CHECK_SCOPE.get()is None,'CONTEXT_BOOKEND')
  for key,value in profile_plan_budget().items():
   if key!='autograd_calls_max':need(counts[key]==value,'EXACT_ACTUAL_BUDGET_'+key)
  peak=int(torch.cuda.max_memory_reserved());need(peak<=EXECUTION['VRAM_max_bytes'],'VRAM_CAP')
  progress['phase']='COMPLETE'
  return {'counts':counts,'actual_autograd_calls':counts['autograd_calls'],'budget':budget,'panel':panel,'support':support,'parity':parity,'native_outputs':progress['phase_logits'],'scalar_components':scalars,'gradient_report':gradient,'parameter_axis':{'names':[n for n,_ in named],'partitions':axes,'None_is_disconnected_zero_only':True,'aggregation_dtype':'float64_cpu'},'strict_install':install,'model_proofs':{'before':before,'after':model_proof(model)},'cuda_peak_reserved_bytes':peak,'process_peak_RSS_KiB':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),'elapsed_seconds':time.monotonic()-began,'eval_point_not_historical_AdamW':True,'chosen_fix':'UNKNOWN','later_generalization':'UNKNOWN','economic_edge':'UNKNOWN','full20_approved':False}
 finally:
  for hook in hooks:hook.remove()


@contextlib.contextmanager
def payload_boundary(plan):
 allowed={v['path']for k,v in plan['inputs'].items()if k in plan['input_roles_not_JSON']};active=[True]
 def audit(event,args):
  if not active[0]:return
  if event in('socket.connect','socket.getaddrinfo','urllib.Request'):raise RuntimeError('GRADIENT_PROFILE_NETWORK_FORBIDDEN')
  if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
   p=Path(os.fsdecode(args[0]))
   if p.suffix in('.npz','.pt','.pth','.parquet','.safetensors','.pkl'):need(str(p)in allowed,'PAYLOAD_ALLOWLIST')
 sys.addaudithook(audit)
 try:yield
 finally:active[0]=False


def run(sha):
    began = time.monotonic(); guard = require_scope(sha)
    plan,binding,pref,live = inspect(sha,admitted=True); checked_inputs(plan)
    handover = ref(REPO/'CURRENT_HANDOVER.md'); ROOT.mkdir(exist_ok=False)
    handlers = {sig:signal.getsignal(sig)for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    claim = result = None; progress = {'phase':'CLAIM'}
    try:
        def expired(*_): raise TimeoutError('PROVIDER_HLC_GRADIENT_PROFILE_WALL')
        def stopped(sig,*_): raise InterruptedError('PROVIDER_HLC_GRADIENT_PROFILE_SIGNAL_'+str(sig))
        signal.signal(signal.SIGALRM,expired); signal.signal(signal.SIGTERM,stopped); signal.signal(signal.SIGINT,stopped)
        claim = publish('CLAIM.json',{'stage':STAGE,'plan':binding,'policy_at_claim':pref,'handover_at_claim':handover,'runtime':live,
                                    'guard':guard,'process_identity':core.future_cpu_process_identity(),'remaining_invocations':0,
                                    'relaunch_allowed':False,'training_authorized':False,'test_data_used':False})
        remaining = EXECUTION['internal_seconds']-(time.monotonic()-began); need(remaining>0,'TIME_BEFORE_PAYLOAD')
        signal.setitimer(signal.ITIMER_REAL,remaining)
        with payload_boundary(plan): report = physical(plan,live,progress=progress)
        checked_inputs(plan)
        need(readout.source_clean(plan)and ref(POLICY)==pref and ref(PLAN)==binding and ref(REPO/'CURRENT_HANDOVER.md')==handover,
             'SOURCE_INPUT_PLAN_POLICY_HANDOVER_BOOKEND')
        need(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==live['wsl_boot_id']
             and core.physical_windows_boot_id()==live['physical_windows_boot_id'],'BOTH_BOOT_BOOKEND')
        need(core.common.firewall_state()['denied_attempts']==0,'M1_FIREWALL')
        result = publish('RESULT.json',{'status':'COMPLETE_NATIVE_GRADIENT_PROFILE','stage':STAGE,'plan':binding,'claim':claim,
                                       'policy_at_claim':pref,'handover_at_claim':handover,'report':report,'source_inputs_policy_boot_unchanged':True,
                                       'source_unchanged':True,'test_data_used':False,'economic_edge_proven':False,'elapsed_seconds':time.monotonic()-began})
        terminal = publish('TERMINAL.json',{'status':'COMPLETE','exit_code':0,'stage':STAGE,'plan':binding,'claim':claim,'result':result,
                                           'remaining_invocations':0,'relaunch_allowed':False,'source_unchanged':True,'test_data_used':False})
        return {'result':result,'terminal':terminal}
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL,0)
        publish('TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'stage':STAGE,'plan':binding,'claim':claim,'result':result,
                               'progress':progress,'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False,
                               'error':type(error).__name__+':'+str(error)[:2000]})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig,handler in handlers.items(): signal.signal(sig,handler)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan-sha256',required=True); parser.add_argument('--preflight',action='store_true')
    args = parser.parse_args()
    need(sys.argv[1:]==['--plan-sha256',args.plan_sha256]+(['--preflight']if args.preflight else[]),'EXACT_CLI')
    if args.preflight:
        guard = require_scope(args.plan_sha256,preflight=True)
        plan,binding,pref,live = inspect(args.plan_sha256,admitted=False)
        print(json.dumps({'status':'EXACT_NATIVE_GRADIENT_PROFILE_PREFLIGHT_PASS','plan':binding,'policy':pref,'runtime':live,'guard':guard,
                          'claim_created':False,'checkpoint_deserializations':0,'NPZ_arrays_decoded':0,'models':0,'NN_forwards':0,'CUDA_initialized':False}))
    else: print(json.dumps(run(args.plan_sha256),sort_keys=True))


if __name__=='__main__':main()
