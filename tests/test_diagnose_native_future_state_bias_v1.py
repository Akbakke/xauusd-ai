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
