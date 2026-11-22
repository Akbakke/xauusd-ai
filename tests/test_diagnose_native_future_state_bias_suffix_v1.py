"""Bounded suffix intervention contracts; no model or CUDA construction."""
import copy
import numpy as np
import pytest
from gx1.audit import diagnose_native_future_state_bias_suffix_v1 as owner

FLAGS=('training_enabled','native_training_enabled','native_launch_authorized','full_epoch_training_allowed','full_val_allowed')


def main_fixture():
 return {'limits':{'main':{'optimizer_steps':3694}},
  'recipe':{'checkpoint_every_optimizer_updates':256,'train_batch_rows':96,'seed':20260911,'passes':1},
  'schedule':{'logical_batches':3694,'batches_per_pass':3694,'drop_last':False,'shuffle':False,'batch_rows':96,'rows':354550}}


def test_exact_batch_set_and_short_batch_position_are_preserved():
 main=main_fixture();before=copy.deepcopy(main);orders=owner.make_orders(main)
 chronological=orders['chronological'];reordered=orders['reordered']
 assert chronological==list(range(3328,3694))
 assert len(reordered)==366 and len(set(reordered))==366
 assert sorted(reordered)==chronological and reordered[-1]==3693
 assert reordered[:-1]!=chronological[:-1] and 3693 not in reordered[:-1]
 assert orders['seed']==20260911 and main==before
 assert owner.make_orders(main)==orders


@pytest.mark.parametrize('section,key,value',[
 ('limits','optimizer_steps',3693),('recipe','seed',20260912),
 ('recipe','passes',2),('recipe','train_batch_rows',64),('recipe','checkpoint_every_optimizer_updates',128),
 ('schedule','logical_batches',3695),('schedule','batches_per_pass',3693),
 ('schedule','batch_rows',64),('schedule','rows',354549),
 ('schedule','drop_last',True),('schedule','shuffle',True)])
def test_changed_original_schedule_or_seed_fails_closed(section,key,value):
 main=main_fixture();scope=main['limits']['main']if section=='limits'else main[section];scope[key]=value
 with pytest.raises(ValueError):owner.make_orders(main)


def test_preflight_has_all_training_flags_closed():
 policy=dict.fromkeys(FLAGS,False);owner.validate_flags(policy,admitted=False)


def test_only_two_private_training_flags_are_open_when_admitted():
 policy=dict.fromkeys(FLAGS,False);policy['training_enabled']=policy['native_training_enabled']=True
 owner.validate_flags(policy,admitted=True)
 assert policy['native_launch_authorized']is False and policy['full_epoch_training_allowed']is False


@pytest.mark.parametrize('admitted',[False,True])
@pytest.mark.parametrize('key',FLAGS)
def test_wrong_training_authority_is_rejected(admitted,key):
 policy=dict.fromkeys(FLAGS,False)
 if admitted:policy['training_enabled']=policy['native_training_enabled']=True
 policy[key]=not policy[key]
 with pytest.raises(ValueError):owner.validate_flags(policy,admitted=admitted)


def test_gate_is_one_private_technical_study_and_cannot_relaunch():
 binding={'path':'/scope/study.json','sha256':'a'*64};a=owner.gate_expected(binding,True);p=owner.gate_expected(binding,False)
 assert a['plan']==binding and a['execution_authorized']is True and a['remaining_invocations']==1
 assert a['relaunch_allowed']is False and a['test_authorized']is False
 assert p['execution_authorized']is False and p['remaining_invocations']==0


def reference_steps():
 return [{'logical_step':i+1,'loss':float(i),'output_sha256':{'side_logits':str(i)}}for i in range(3328,3694)]


def test_reference_steps_must_match_every_original_report():
 expected=reference_steps();owner.validate_reference_steps(copy.deepcopy(expected),expected)


@pytest.mark.parametrize('change',['short','reordered','loss','output'])
def test_reference_step_mismatch_fails_before_candidate(change):
 expected=reference_steps();actual=copy.deepcopy(expected)
 if change=='short':actual.pop()
 elif change=='reordered':actual[0],actual[1]=actual[1],actual[0]
 elif change=='loss':actual[0]['loss']+=1
 else:actual[0]['output_sha256']['side_logits']='foreign'
 with pytest.raises(ValueError):owner.validate_reference_steps(actual,expected)


def reference_proof():
 return {name:name+'-authentic'for name in ('model_state','optimizer_state','rng_state','scheduler_state','position','training','optimizer_parameter_names','steps')}


def test_reference_state_uses_all_required_components_and_leaves_provenance_untouched():
 actual=reference_proof();expected={**actual,'trajectory':'original-trajectory','scores':'qualified-scores'};before=copy.deepcopy(expected)
 result=owner.validate_reference_state(actual,expected)
 assert result['bitexact']is True and result['proof']==actual
 assert set(result['components'])==set(actual)and expected==before


@pytest.mark.parametrize('key',tuple(reference_proof()))
@pytest.mark.parametrize('change',['missing','different'])
def test_any_reference_component_failure_blocks_reference_reuse(key,change):
 actual=reference_proof();expected={**actual,'trajectory':'original','scores':'original'}
 if change=='missing':actual.pop(key)
 else:actual[key]='changed'
 with pytest.raises(ValueError,match='REFERENCE_STATE'):owner.validate_reference_state(actual,expected)


def test_reference_proof_rejects_extra_claimed_components():
 actual={**reference_proof(),'extra':'unbound'}
 with pytest.raises(ValueError,match='REFERENCE_STATE'):owner.validate_reference_state(actual,reference_proof())


def test_exact_budget_counts_both_genuine_technical_parameter_fits():
 limits=owner.LIMITS
 assert limits['model_constructions']==1 and limits['optimizer_constructions']==2
 assert limits['checkpoint_loads']==1 and limits['checkpoint_restores']==2
 assert limits['model_forwards']==limits['backbone_forwards']==limits['readout_forwards']==813
 assert limits['backward_calls']==limits['optimizer_steps']==732
 assert limits['technical_parameter_fits']==2
 assert limits['checkpoint_writes']==limits['checkpoint_readbacks']==1
 assert limits['normalization_fits']==limits['prior_fits']==limits['autograd_calls']==limits['CONTROL_VAL_TEST_rows']==0


def mock_claimed_run(monkeypatch,tmp_path):
 from gx1.contracts import gx1_capped_execution_v1 as guard
 root=tmp_path/'run';binding={'path':'/study/plan.json','sha256':'a'*64};records=[]
 monkeypatch.setattr(owner,'ROOT',root)
 monkeypatch.setattr(owner,'inspect',lambda *args,**kw:({},binding,{},{}))
 monkeypatch.setattr(guard,'require_native_future_state_bias_suffix_execution',lambda **kw:{'plan':binding,'output_root':str(root),'outer_seconds':3600,'internal_seconds':3570})
 monkeypatch.setattr(owner.core,'future_cpu_process_identity',lambda:{'pid':7})
 monkeypatch.setattr(owner,'publish',lambda path,value:(records.append((path.name,value))or{'path':str(path),'sha256':'b'*64}))
 monkeypatch.setattr(owner.signal,'signal',lambda *args:None)
 monkeypatch.setattr(owner.signal,'setitimer',lambda *args:None)
 return root,records


def test_claimed_reference_failure_is_consumed_and_has_terminal(monkeypatch,tmp_path):
 root,records=mock_claimed_run(monkeypatch,tmp_path)
 def fail(*args):raise ValueError('REFERENCE_STATE_MISMATCH')
 monkeypatch.setattr(owner,'physical',fail);monkeypatch.setattr(owner.time,'monotonic',lambda:0.)
 with pytest.raises(ValueError,match='REFERENCE_STATE'):owner.run('a'*64)
 assert root.is_dir()and [name for name,_ in records]==['CLAIM.json','TERMINAL.json']
 terminal=records[-1][1]
 assert terminal['status']=='FAILED_CONSUMED'and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False
 assert 'REFERENCE_STATE' in terminal['error']


def test_expired_startup_cannot_leave_a_claim_without_terminal(monkeypatch,tmp_path):
 _,records=mock_claimed_run(monkeypatch,tmp_path)
 def forbidden(*args):pytest.fail('suffix physical execution after expired startup')
 monkeypatch.setattr(owner,'physical',forbidden)
 ticks=iter([0.,4000.]);monkeypatch.setattr(owner.time,'monotonic',lambda:next(ticks))
 with pytest.raises(ValueError,match='WALL_BEFORE_INPUTS'):owner.run('a'*64)
 assert [name for name,_ in records]==['CLAIM.json','TERMINAL.json']
 assert records[-1][1]['status']=='FAILED_CONSUMED'and records[-1][1]['remaining_invocations']==0


class CaptureArray:
 def __init__(self,value):self.value=value
 def detach(self):return self
 def cpu(self):return self
 def numpy(self):return self.value


class CaptureModel:
 def __init__(self):self.callbacks=[];self.removals=0
 def register_forward_hook(self,callback):
  model=self;model.callbacks.append(callback)
  class Handle:
   def remove(self):model.callbacks.remove(callback);model.removals+=1
  return Handle()
 def emit(self,output):
  for callback in list(self.callbacks):callback(self,(),output)


def capture_context(monkeypatch):
 context={'source_rows':np.arange(20727,dtype=np.int64),'prior':object()};model=CaptureModel();main={'role':'capture-test'};seen=[]
 def packet(actual_context,take):
  assert actual_context is context
  return {'rows':take.copy()}
 monkeypatch.setattr(owner.train,'packet',packet)
 return context,model,main,seen


def emit_capture(model,take):
 values=np.repeat(take[:,None],5,axis=1).astype(np.float32)
 model.emit({'activity_logits':CaptureArray(values),'side_logits':CaptureArray(-values)})


def test_score_capture_preserves_all_81_batches_and_short_tail(monkeypatch):
 context,model,main,seen=capture_context(monkeypatch);expected_score={'rows':20727,'batch_count':81}
 def score(actual_model,packets,actual_main,prior,*,initial,equal_tf):
  assert actual_model is model and actual_main is main and prior is context['prior']
  assert initial is False and equal_tf is True
  for packet in packets:
   take=packet['rows'];seen.append(take.copy());emit_capture(model,take)
  return expected_score
 monkeypatch.setattr(owner.core,'future_cuda_score',score)
 report,arrays=owner._score_capture(model,context,main)
 assert report is expected_score and len(seen)==81
 assert all(len(take)==256 for take in seen[:-1])and len(seen[-1])==247
 assert np.array_equal(np.concatenate(seen),np.arange(20727))
 expected=np.repeat(np.arange(20727,dtype=np.float32)[:,None],5,axis=1)
 assert set(arrays)=={'activity_logits','side_logits'}
 assert np.array_equal(arrays['activity_logits'],expected)and np.array_equal(arrays['side_logits'],-expected)
 assert model.removals==1 and not model.callbacks


def test_score_capture_rejects_missing_row_and_removes_hook(monkeypatch):
 context,model,main,seen=capture_context(monkeypatch)
 def score(actual_model,packets,*args,**kwargs):
  for index,packet in enumerate(packets):
   take=packet['rows'];seen.append(take)
   emit_capture(model,take[:-1]if index==80 else take)
  return {'rows':20726}
 monkeypatch.setattr(owner.core,'future_cuda_score',score)
 with pytest.raises(ValueError,match='CAPTURE_ROWS'):owner._score_capture(model,context,main)
 assert len(seen)==81 and model.removals==1 and not model.callbacks


def test_score_capture_removes_hook_after_scoring_failure(monkeypatch):
 context,model,main,seen=capture_context(monkeypatch)
 def score(actual_model,packets,*args,**kwargs):
  for index,packet in enumerate(packets):
   seen.append(packet['rows']);emit_capture(model,packet['rows'])
   if index==1:raise RuntimeError('scoring failure')
 monkeypatch.setattr(owner.core,'future_cuda_score',score)
 with pytest.raises(RuntimeError,match='scoring failure'):owner._score_capture(model,context,main)
 assert len(seen)==2 and model.removals==1 and not model.callbacks


def test_main_source_inventory_rejected_before_source_hashing(monkeypatch):
 allowed={str(owner.train.MAIN_PLAN),str(owner.train.PAIR_PLAN)}
 def binding(path):
  assert str(path)in allowed
  return {'path':str(path),'sha256':'a'*64}
 def read(ref):
  if ref['path']==str(owner.train.MAIN_PLAN):return {'source_owners':{'/untrusted/raw.bin':'a'*64}}
  pytest.fail('read beyond rejected MAIN inventory')
 monkeypatch.setattr(owner,'ref',binding);monkeypatch.setattr(owner,'read',read)
 monkeypatch.setattr(owner.train,'expected_source_names',lambda:{'safe.py'})
 monkeypatch.setattr(owner,'sha',lambda *args:pytest.fail('source hash before inventory validation'))
 with pytest.raises(ValueError,match='EXACT_SOURCE_INVENTORY'):owner.build_plan(runtime={})
