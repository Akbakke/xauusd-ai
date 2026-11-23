"""Focused failure and numerical attribution checks, no CUDA initialization."""
import copy
import numpy as np
import pytest
from gx1.audit import diagnose_native_future_state_bias_v1 as owner
from tests.test_native_future_state_training_v1 import context_fixture


def data(monkeypatch):
 c,t=context_fixture(monkeypatch);n=len(c['truth']['issue_ns'])
 out={'activity_logits':np.full((n,5),.3,np.float32),'side_logits':np.full((n,5),.2,np.float32)}
 return c['truth'],t,out


def test_known_support_probabilities_scores_and_derivatives(monkeypatch):
 truth,target,out=data(monkeypatch);r=owner.metric_rows(out,truth,target,side_bias=np.full(5,.1))['ALL']['M5']['side']
 y=truth['side_up'][truth['side_known'][:,0],0].astype(float);p=1/(1+np.exp(-.2));res=p-y
 assert r['count']==len(y)and r['mean_probability']==pytest.approx(p,abs=1e-8)
 assert r['Brier']==pytest.approx(np.mean(res**2),abs=1e-8)
 assert r['conditional_BCE_intercept_derivative']==pytest.approx(res.mean(),abs=1e-8)
 assert r['Brier_intercept_derivative']==pytest.approx(2*np.mean(res*p*(1-p)),abs=1e-8)
 assert r['mean_weight_dot_hidden']==pytest.approx(.1,abs=1e-8)
 assert r['AUC']==.5


def test_derivative_matches_direct_small_perturbation(monkeypatch):
 truth,target,out=data(monkeypatch);r=owner.metric_rows(out,truth,target)['ALL']['M5']['side'];eps=1e-3
 plus={k:v.copy()for k,v in out.items()};minus={k:v.copy()for k,v in out.items()}
 plus['side_logits']+=eps;minus['side_logits']-=eps
 a=owner.metric_rows(plus,truth,target)['ALL']['M5']['side'];b=owner.metric_rows(minus,truth,target)['ALL']['M5']['side']
 assert (a['logloss']-b['logloss'])/(2*eps)==pytest.approx(r['conditional_BCE_intercept_derivative'],abs=2e-6)
 assert (a['Brier']-b['Brier'])/(2*eps)==pytest.approx(r['Brier_intercept_derivative'],abs=2e-6)


def test_one_class_auc_is_unknown(monkeypatch):
 truth,target,out=data(monkeypatch);truth['side_up']=truth['side_known'].copy()
 r=owner.metric_rows(out,truth,target)['ALL']['D1']['side']
 assert r['AUC']is None and r['AUC_undefined_reason']


def test_no_support_remains_unknown(monkeypatch):
 truth,target,out=data(monkeypatch);truth['side_known'][:,-1]=False;truth['side_up'][:,-1]=False
 r=owner.metric_rows(out,truth,target)['ALL']['D1']['side']
 assert r=={'count':0,'AUC':None,'undefined_reason':'NO_OBSERVED_SUPPORT'}


def test_known_target_bar_clusters_not_independent_cells(monkeypatch):
 truth,target,out=data(monkeypatch);r=owner.metric_rows(out,truth,target)['ALL']
 assert r['D1']['side']['unique_target_bars']<r['D1']['side']['count']


def test_swap_decomposition_retains_interaction(monkeypatch):
 truth,target,out=data(monkeypatch);mapping={}
 for name,z in [('H256_R256',0.),('H256_R3694',.1),('H3694_R256',.2),('H3694_R3694',.5)]:
  mapping[name]={'side_logits':np.full_like(out['side_logits'],z)}
 r=owner.swap_effects(mapping,truth,target)['M5']
 assert r['head_effect_mean_logit']==pytest.approx(.1)
 assert r['hidden_effect_mean_logit']==pytest.approx(.2)
 assert r['interaction_mean_logit']==pytest.approx(.2)
 assert r['actual_mean_logit_delta']==pytest.approx(.5)


def test_gate_grants_no_training():
 p={'path':'/scope/plan.json','sha256':'a'*64};g=owner.gate_expected(p,True)
 assert g['execution_authorized']is True and g['remaining_invocations']==1
 assert g['training_authorized']is False and g['test_authorized']is False and g['relaunch_allowed']is False
 assert owner.gate_expected(p,False)['remaining_invocations']==0


@pytest.mark.parametrize('n,every',[(3693,256),(3694,128),(3695,256)])
def test_wrong_original_schedule_rejected(n,every):
 with pytest.raises(ValueError,match='ORIGINAL_SCHEDULE'):
  owner.expected_cursors({'limits':{'main':{'optimizer_steps':n}},'recipe':{'checkpoint_every_optimizer_updates':every}})


def test_all_exact_sixteen_checkpoints():
 r=owner.expected_cursors({'limits':{'main':{'optimizer_steps':3694}},'recipe':{'checkpoint_every_optimizer_updates':256}})
 assert len(r)==16 and r[0]==0 and r[1]==256 and r[-2:]==[3584,3694]


@pytest.mark.parametrize('mutation',['schema','producer','proof','position'])
def test_checkpoint_identity_fail_closed(monkeypatch,mutation):
 trajectory={'bound':'trajectory'};monkeypatch.setattr(owner.train,'trajectory_bindings',lambda *a:trajectory)
 monkeypatch.setattr(owner.train,'position',lambda *a:{'logical_cursor':0});monkeypatch.setattr(owner.train,'checkpoint_proof',lambda cp:{'auth':'original'})
 ref={'path':'/original/plan.json','sha256':'a'*64};receipt={'producer_runtime':{'boot':1},'logical_cursor':0,'optimizer_updates':0,'proof':{'auth':'original'}}
 cp={'schema_version':'gx1_native_future_state_training_checkpoint_v1','trajectory':trajectory,'producer':{'original_plan':ref,'lane':'main','runtime':{'boot':1}},'position':{'logical_cursor':0},'scheduler_state':None,'training':True,'steps':[],'model_state':{},'optimizer_state':{},'rng_state':{},'optimizer_parameter_names':[],'scores':{}}
 main={'schedule':{}};result={'plan':ref,'report':{'trajectory':trajectory}}
 owner.validate_checkpoint(cp,receipt,main,result,0)
 if mutation=='schema':cp['extra']='forbidden'
 elif mutation=='producer':cp['producer']['lane']='foreign'
 elif mutation=='proof':receipt['proof']={'auth':'wrong'}
 else:cp['position']={'logical_cursor':1}
 with pytest.raises(ValueError,match='CHECKPOINT'):
  owner.validate_checkpoint(cp,receipt,main,result,0)


def test_expired_startup_emits_consumed_failure(monkeypatch,tmp_path):
 from gx1.contracts import gx1_capped_execution_v1 as guard
 root=tmp_path/'run';monkeypatch.setattr(owner,'ROOT',root);binding={'path':'/plan.json','sha256':'a'*64};records=[]
 monkeypatch.setattr(owner,'inspect',lambda *a,**kw:({},binding,{},{}))
 monkeypatch.setattr(guard,'require_native_future_state_bias_frozen_execution',lambda **kw:{'plan':binding,'output_root':str(root),'outer_seconds':3600})
 ticks=iter([0.,4000.]);monkeypatch.setattr(owner.time,'monotonic',lambda:next(ticks))
 monkeypatch.setattr(owner.core,'future_cpu_process_identity',lambda:{})
 monkeypatch.setattr(owner,'publish',lambda p,d:(records.append((p.name,d))or{'path':str(p),'sha256':'b'*64}))
 monkeypatch.setattr(owner.signal,'signal',lambda *a:None);monkeypatch.setattr(owner.signal,'setitimer',lambda *a:None)
 with pytest.raises(ValueError,match='WALL_BEFORE_INPUTS'):owner.run('a'*64)
 assert [n for n,d in records]==['CLAIM.json','TERMINAL.json']
 assert records[-1][1]['status']=='FAILED_CONSUMED'and records[-1][1]['remaining_invocations']==0


def test_source_inventory_is_checked_before_hashing(monkeypatch):
 def reject(*args):raise ValueError('bad inventory')
 monkeypatch.setattr(owner.train,'require_source_inventory',reject)
 monkeypatch.setattr(owner,'sha',lambda *args:pytest.fail('payload hash before inventory'))
 with pytest.raises(ValueError,match='bad inventory'):owner.source_bridge({'source_owners':{'/payload.npz':'a'*64}})


def test_plan_builder_rejects_inventory_before_source_hashes(monkeypatch):
 bad={'source_owners':{'/payload.npz':'a'*64}}
 monkeypatch.setattr(owner,'ref',lambda p:{'path':str(p),'sha256':'a'*64})
 monkeypatch.setattr(owner,'read',lambda r:bad)
 def reject(*a):raise ValueError('bad inventory')
 monkeypatch.setattr(owner.train,'require_source_inventory',reject)
 monkeypatch.setattr(owner,'sha',lambda *a:pytest.fail('source hashed before inventory'))
 with pytest.raises(ValueError,match='bad inventory'):owner.build_plan(runtime={})



def driver_source_fixture(monkeypatch):
 from gx1.contracts import gx1_capped_execution_v1 as guard
 loader='gx1/audit/native_future_state_training_v1.py';numeric='gx1/model_numeric.py';calls=[]
 main={'source_owners':{**{k:'old-'+k for k in owner.GUARDS},loader:'old-loader',numeric:'numeric'},'source_base_commit':'original'}
 monkeypatch.setattr(owner.train,'require_source_inventory',lambda owners:calls.append('inventory'))
 def source_hash(path):calls.append('hash:'+str(path));return 'numeric'if str(path).endswith(numeric)else'changed-loader'
 monkeypatch.setattr(owner,'sha',source_hash)
 def bridge(owners,**kwargs):calls.append(('bridge',kwargs));return {'checked_guard_owners':3}
 monkeypatch.setattr(guard,'native_future_state_bias_guard_source_bridge',bridge)
 return main,calls,guard,numeric


def test_explicit_driver_probe_validates_three_owner_bridge_before_other_numeric_sources(monkeypatch):
 main,calls,guard,numeric=driver_source_fixture(monkeypatch)
 assert owner.source_bridge(main,driver_probe=True)=={'checked_guard_owners':3}
 assert calls[0]=='inventory'and calls[1]==('bridge',{'producer_commit':'original','driver_requalification':True})
 assert len(calls)==3 and calls[-1].endswith(numeric)


def test_driver_probe_bridge_failure_precedes_all_other_source_hashing(monkeypatch):
 main,calls,guard,numeric=driver_source_fixture(monkeypatch)
 def reject(*args,**kwargs):calls.append('rejected-bridge');raise ValueError('unreviewed loader')
 monkeypatch.setattr(guard,'native_future_state_bias_guard_source_bridge',reject)
 with pytest.raises(ValueError,match='unreviewed loader'):owner.source_bridge(main,driver_probe=True)
 assert calls==['inventory','rejected-bridge']


def test_driver_probe_still_rejects_changed_numeric_owner(monkeypatch):
 main,calls,guard,numeric=driver_source_fixture(monkeypatch);main['source_owners'][numeric]='changed-math'
 with pytest.raises(ValueError,match='ORIGINAL_59_NUMERIC_SOURCES'):owner.source_bridge(main,driver_probe=True)


def test_default_source_bridge_does_not_exempt_changed_metadata_loader(monkeypatch):
 main,calls,guard,numeric=driver_source_fixture(monkeypatch)
 with pytest.raises(ValueError,match='ORIGINAL_NUMERIC_SOURCES'):owner.source_bridge(main)
 assert not any(isinstance(x,tuple)and x[0]=='bridge'for x in calls)


# Separate saved-state zero-gradient AdamW projection; no model or optimizer construction.
def moment_toy_options():
 return {'lr':.01,'betas':(.8,.9),'eps':1e-8,'weight_decay':.2,'amsgrad':False,'maximize':False,
  'foreach':False,'capturable':False,'differentiable':False,'fused':False}


def test_moment_projection_positive_first_moment_lowers_parameter_and_negative_pushes_up():
 p=np.zeros(2,np.float64);m=np.array([1.,-1.]);v=np.ones(2,np.float64)
 result=owner.moment_zero_gradient_delta(p,m,v,7,moment_toy_options())
 assert result['delta'][0]<0 and result['delta'][1]>0
 assert np.array_equal(result['decay_delta'],np.zeros(2))


def test_moment_projection_uses_actual_step_bias_corrections_and_resets_only_first_moment():
 options=moment_toy_options();p=np.array([2.,-3.]);m=np.array([1.,-2.]);v=np.array([4.,9.]);step=7
 original=owner.moment_zero_gradient_delta(p,m,v,step,options)
 reset=owner.moment_zero_gradient_delta(p,m,v,step,options,reset_first_moment=True)
 b1,b2=options['betas'];t=step+1
 expected_decay=-options['lr']*options['weight_decay']*p
 expected_moment=-options['lr']*(b1*m)/(1-b1**t)/(np.sqrt(b2*v/(1-b2**t))+options['eps'])
 np.testing.assert_allclose(original['moment_delta'],expected_moment,rtol=1e-14,atol=0)
 np.testing.assert_array_equal(original['decay_delta'],expected_decay)
 np.testing.assert_allclose(original['delta'],expected_decay+expected_moment,rtol=1e-14,atol=0)
 np.testing.assert_array_equal(reset['decay_delta'],original['decay_delta'])
 np.testing.assert_array_equal(reset['moment_delta'],np.zeros_like(p))
 np.testing.assert_array_equal(reset['delta'],expected_decay)
 wrong_step=owner.moment_zero_gradient_delta(p,m,v,0,options)
 assert not np.allclose(wrong_step['moment_delta'],original['moment_delta'],rtol=1e-6,atol=0)


def test_moment_projection_preserves_inputs_and_rejects_negative_v_or_invalid_options():
 p=np.array([2.,-3.]);m=np.array([1.,-2.]);v=np.array([4.,9.]);options=moment_toy_options()
 before=(p.copy(),m.copy(),v.copy(),copy.deepcopy(options))
 owner.moment_zero_gradient_delta(p,m,v,7,options,reset_first_moment=True)
 for actual,expected in zip((p,m,v),before[:3]):np.testing.assert_array_equal(actual,expected)
 assert options==before[3]
 with pytest.raises(ValueError):owner.moment_zero_gradient_delta(p,m,-v,7,options)
 for key,value in (('betas',(1.,.9)),('eps',-1e-8),('lr',-.01),('weight_decay',-.1)):
  invalid={**options,key:value}
  with pytest.raises(ValueError):owner.moment_zero_gradient_delta(p,m,v,7,invalid)


def moment_toy_projection():
 hidden=np.zeros((2,5,2),np.float32);hidden[:,:,0]=1.;logits=np.zeros((2,5),np.float32)
 ak=np.ones((2,5),np.bool_);sk=np.zeros((2,5),np.bool_);sk[:,0]=True;up=np.zeros_like(sk);heads={}
 for tf in owner.train.TF:
  heads[tf]={'weight':np.zeros(2,np.float64),'bias':np.zeros(1,np.float64),
   'weight_exp_avg':np.zeros(2,np.float64),'bias_exp_avg':np.zeros(1,np.float64),
   'weight_exp_avg_sq':np.ones(2,np.float64),'bias_exp_avg_sq':np.ones(1,np.float64),'step':7,'options':moment_toy_options()}
 return hidden,logits,ak,sk,up,heads


def test_moment_projection_weight_effect_uses_hidden_while_bias_effect_does_not():
 values=moment_toy_projection();values[-1]['M5']['weight_exp_avg'][0]=1.
 cell=owner.moment_projection_report(*values)['per_tf']['M5'];arm=cell['arms']['original_m']
 expected=owner.moment_zero_gradient_delta(np.zeros(2),np.array([1.,0.]),np.ones(2),7,moment_toy_options())['delta'][0]
 assert arm['bias_delta']==0 and arm['mean_delta_side_logit_known']==pytest.approx(expected,abs=1e-15)
 assert cell['arms']['reset_m']['mean_delta_side_logit_known']==0
 values[0][:]=0;values[-1]['M5']['weight_exp_avg'][:]=0;values[-1]['M5']['bias_exp_avg'][0]=-1.
 arm=owner.moment_projection_report(*values)['per_tf']['M5']['arms']['original_m']
 assert arm['projection_weight_delta_l2']==0 and arm['bias_delta']>0
 assert arm['mean_delta_side_logit_known']==arm['bias_delta']


def test_moment_projection_conditional_probability_is_not_scaled_by_activity_support():
 values=moment_toy_projection();values[1][:,0]=np.float32(.8);values[-1]['M5']['bias'][0]=float(np.float32(.8))
 dense=owner.moment_projection_report(*values);values[2][:]=False;values[2][:,0]=True
 sparse=owner.moment_projection_report(*values)
 a,b=dense['per_tf']['M5'],sparse['per_tf']['M5']
 assert dense['activity_denominator_all_TF']==10 and sparse['activity_denominator_all_TF']==2
 assert a['baseline']==b['baseline'] and a['baseline']['mean_q']==pytest.approx(1/(1+np.exp(-float(np.float32(.8)))),abs=1e-15)
 assert a['arms']['original_m']['mean_q']==b['arms']['original_m']['mean_q']
 assert b['arms']['original_m']['native_partial_directional_derivative']==pytest.approx(5*a['arms']['original_m']['native_partial_directional_derivative'],abs=1e-15)
 assert 'no joint' in dense['probability_semantics']


def test_moment_projection_native_partial_directional_derivative_and_alignment_match_fixed_hidden():
 values=moment_toy_projection();head=values[-1]['M5'];head['options']['weight_decay']=0.
 head['weight_exp_avg'][0]=1.;head['bias_exp_avg'][0]=.5
 cell=owner.moment_projection_report(*values)['per_tf']['M5'];arm=cell['arms']['original_m']
 dz=arm['mean_delta_side_logit_known'];assert dz<0
 assert arm['native_partial_directional_derivative']==pytest.approx(dz/10,abs=1e-15)
 assert arm['native_partial_moment_directional_derivative']==pytest.approx(dz/10,abs=1e-15)
 assert cell['fixed_cohort_gradient_dot_first_moment']==pytest.approx(.15,abs=1e-15)
 assert arm['NLL_delta']<0 and cell['arms']['reset_m']['native_partial_directional_derivative']==0
 effect=cell['first_moment_effect_original_minus_reset']
 assert effect['mean_delta_side_logit_known']==pytest.approx(dz,abs=1e-15)


def test_moment_projection_no_side_support_retains_unknown_metrics():
 values=moment_toy_projection();cell=owner.moment_projection_report(*values)['per_tf']['D1']
 assert cell['baseline']['count']==0 and cell['baseline']['mean_q']is None
 assert cell['baseline']['Brier']is cell['baseline']['NLL']is None
 for arm in cell['arms'].values():
  assert arm['count']==0 and arm['mean_q']is None and arm['mean_delta_side_logit_known']is None
  assert arm['native_partial_directional_derivative']is None
 assert cell['fixed_cohort_gradient_dot_first_moment']is None and cell['support_status'].startswith('NO_SUPPORT')


def moment_toy_checkpoint():
 import torch
 recipe=moment_toy_options();names=[];ids=[];model={};state={}
 for j,tf in enumerate(owner.train.TF):
  for k,shape in (('weight',(1,2)),('bias',(1,))):
   name=f'readout.side_heads.{tf}.{k}';index=101+2*j if k=='weight'else 5+3*j
   names.append(name);ids.append(index);model[name]=torch.full(shape,float(j+1),dtype=torch.float32)
   state[index]={'step':torch.tensor(7.,dtype=torch.float32),'exp_avg':torch.full(shape,float(j+.25),dtype=torch.float32),'exp_avg_sq':torch.ones(shape,dtype=torch.float32)}
 cp={'optimizer_state':{'state':state,'param_groups':[{**copy.deepcopy(recipe),'params':ids}]},'optimizer_parameter_names':names,
  'scheduler_state':None,'position':{'optimizer_updates':7},'model_state':model}
 return cp,recipe


def test_moment_projection_extracts_actual_nonordinal_parameter_ids_without_construction(monkeypatch):
 import torch
 cp,recipe=moment_toy_checkpoint()
 monkeypatch.setattr(torch.optim,'AdamW',lambda *args,**kwargs:pytest.fail('optimizer construction'))
 heads=owner.moment_extract_heads(cp,recipe,width=2)
 assert heads['M5']['parameter_ids']=={'weight':101,'bias':5}
 assert heads['D1']['parameter_ids']=={'weight':109,'bias':17}
 assert heads['M5']['parameter_names']['weight']=='readout.side_heads.M5.weight'
 assert heads['M5']['step']==7 and heads['M5']['weight'].dtype==np.float64 and heads['M5']['weight'].shape==(2,)
 np.testing.assert_array_equal(heads['M5']['weight_exp_avg'],np.full(2,.25))
 heads['M5']['weight'][0]=99
 assert float(cp['model_state']['readout.side_heads.M5.weight'][0,0])==1.


def test_moment_projection_extract_rejects_duplicate_ids_wrong_name_recipe_and_actual_step():
 import torch
 for mode in ('ids','name','recipe','step'):
  cp,recipe=moment_toy_checkpoint();group=cp['optimizer_state']['param_groups'][0]
  if mode=='ids':group['params'][1]=group['params'][0]
  elif mode=='name':cp['optimizer_parameter_names'][0]='readout.side_heads.FOREIGN.weight'
  elif mode=='recipe':group['amsgrad']=True
  else:cp['optimizer_state']['state'][group['params'][0]]['step']=torch.tensor(6.,dtype=torch.float32)
  with pytest.raises(ValueError,match='MOMENT_'):owner.moment_extract_heads(cp,recipe,width=2)


def moment_projection_lifecycle(monkeypatch,tmp_path):
 from gx1.contracts import gx1_capped_execution_v1 as guard
 monkeypatch.setattr(owner,'MOMENT_ROOT',tmp_path/'OUT');binding={'path':str(tmp_path/'PLAN.json'),'sha256':'1'*64};pref={'path':str(tmp_path/'POLICY.json'),'sha256':'2'*64}
 monkeypatch.setattr(guard,'require_capped_cpu_audit_execution',lambda:{'memory_max':4*1024**3,'memory_high':4*1024**3,'swap':512*1024**2,'numerical_threads':1,'cgroup_path':'/sys/fs/cgroup/synthetic'})
 monkeypatch.setattr(guard,'require_capped_lock_ancestry',lambda:{'scope':'/synthetic'})
 monkeypatch.setattr(owner,'moment_inspect',lambda *args,**kwargs:({},binding,pref));monkeypatch.setattr(owner,'moment_checked_inputs',lambda plan:None)
 monkeypatch.setattr(owner,'moment_source_clean',lambda plan:True);original=owner.ref
 monkeypatch.setattr(owner,'ref',lambda path:pref if path==owner.POLICY else binding if path==owner.MOMENT_PLAN else original(path))
 return binding,pref


def test_moment_projection_preflight_has_no_checkpoint_arrays_or_claim(monkeypatch,tmp_path,capsys):
 import json,sys,torch
 moment_projection_lifecycle(monkeypatch,tmp_path);seen=[]
 monkeypatch.setattr(owner,'moment_checked_inputs',lambda plan:seen.append('metadata'))
 def forbidden(*args,**kwargs):raise AssertionError('preflight CP/array/claim')
 monkeypatch.setattr(torch,'load',forbidden);monkeypatch.setattr(owner.np,'load',forbidden)
 monkeypatch.setattr(owner,'moment_calculate',forbidden);monkeypatch.setattr(owner,'moment_publish',forbidden)
 monkeypatch.setattr(sys,'argv',['audit','--projection-moment-plan-sha256','1'*64,'--preflight']);owner.main()
 result=json.loads(capsys.readouterr().out)
 assert result['status']=='PREFLIGHT_PROJECTION_MOMENT_PASS'and result['claim_created']is False
 assert result['checkpoint_payloads']==result['array_payloads_decoded']==result['models']==result['forwards']==result['optimizer_updates']==0
 assert seen==['metadata']and not owner.MOMENT_ROOT.exists()


def test_moment_projection_global_and_both_scope_training_flags_remain_closed(monkeypatch,tmp_path):
 import json
 monkeypatch.setattr(owner,'REPO',tmp_path);monkeypatch.chdir(tmp_path);monkeypatch.setattr(owner,'MOMENT_PLAN',tmp_path/'PLAN.json')
 monkeypatch.setattr(owner,'POLICY',tmp_path/'POLICY.json');monkeypatch.setattr(owner,'MOMENT_ROOT',tmp_path/'OUT')
 plan={'source_base_commit':'synthetic','inputs':{'design':{}}};owner.MOMENT_PLAN.write_text(json.dumps(plan));binding=owner.ref(owner.MOMENT_PLAN)
 flags=('training_enabled','native_training_enabled','native_launch_authorized','full_epoch_training_allowed','full_val_allowed')
 policy={key:False for key in flags};policy[owner.MOMENT_STAGE]=owner.moment_gate_expected(binding,False)
 monkeypatch.setattr(owner,'moment_build_plan',lambda **kwargs:plan);monkeypatch.setattr(owner,'moment_source_clean',lambda value:True)
 def checked():
  owner.POLICY.write_text(json.dumps(policy));return owner.moment_inspect(binding['sha256'],admitted=False)
 assert policy[owner.MOMENT_STAGE]['training_authorized']is policy[owner.MOMENT_STAGE]['test_authorized']is False
 assert checked()[0]==plan
 for flag in flags:
  policy[flag]=True
  with pytest.raises(ValueError,match='MOMENT_EXACT_GATE_ALL_TRAINING_CLOSED'):checked()
  policy[flag]=False
 for flag in ('training_authorized','test_authorized'):
  policy[owner.MOMENT_STAGE][flag]=True
  with pytest.raises(ValueError,match='MOMENT_EXACT_GATE_ALL_TRAINING_CLOSED'):checked()
  policy[owner.MOMENT_STAGE][flag]=False


def test_moment_projection_exact_cli_rejects_duplicate_preflight_before_authority(monkeypatch):
 import sys
 monkeypatch.setattr(owner,'moment_inspect',lambda *args,**kwargs:pytest.fail('bad CLI reached authority'))
 monkeypatch.setattr(sys,'argv',['audit','--projection-moment-plan-sha256','1'*64,'--preflight','--preflight'])
 with pytest.raises(ValueError,match='MOMENT_EXACT_CLI'):owner.main()


def test_moment_projection_failure_consumes_claim_and_retains_progress(monkeypatch,tmp_path):
 import json
 moment_projection_lifecycle(monkeypatch,tmp_path)
 def fail(plan,*,progress):
  progress['phase']='synthetic stopped';raise ValueError('synthetic moment failure')
 monkeypatch.setattr(owner,'moment_calculate',fail)
 with pytest.raises(ValueError,match='synthetic moment failure'):owner.moment_run('1'*64)
 terminal=json.loads((owner.MOMENT_ROOT/'TERMINAL.json').read_text())
 assert terminal['status']=='FAILED_CONSUMED'and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False
 assert terminal['result']is None and terminal['progress']['phase']=='synthetic stopped'
 assert (owner.MOMENT_ROOT/'CLAIM.json').exists()and not (owner.MOMENT_ROOT/'RESULT.json').exists()


def test_moment_projection_success_is_durable_and_source_bookend_blocks_later_result(monkeypatch,tmp_path):
 import json
 moment_projection_lifecycle(monkeypatch,tmp_path/'good');monkeypatch.setattr(owner,'moment_calculate',lambda plan,**kwargs:{'synthetic':True})
 result=owner.moment_run('1'*64);terminal=json.loads((owner.MOMENT_ROOT/'TERMINAL.json').read_text())
 assert terminal['status']=='COMPLETE'and terminal['result']==result['result']and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False
 assert json.loads((owner.MOMENT_ROOT/'RESULT.json').read_text())['economic_edge_proven']is False
 moment_projection_lifecycle(monkeypatch,tmp_path/'drift');monkeypatch.setattr(owner,'moment_source_clean',lambda plan:False)
 with pytest.raises(ValueError,match='MOMENT_SOURCE_INPUT_PLAN_POLICY_BOOT_BOOKEND'):owner.moment_run('1'*64)
 terminal=json.loads((owner.MOMENT_ROOT/'TERMINAL.json').read_text())
 assert terminal['status']=='FAILED_CONSUMED'and not (owner.MOMENT_ROOT/'RESULT.json').exists()
