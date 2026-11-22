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


def test_exact_budget_counts_both_driver_qualified_technical_parameter_fits():
 limits=owner.LIMITS
 assert limits['model_constructions']==1 and limits['optimizer_constructions']==2
 assert limits['checkpoint_loads']==1 and limits['checkpoint_restores']==2
 assert limits['model_forwards']==limits['backbone_forwards']==limits['readout_forwards']==894
 assert limits['backward_calls']==limits['optimizer_steps']==732
 assert limits['technical_parameter_fits']==2
 assert limits['checkpoint_writes']==limits['checkpoint_readbacks']==12
 assert limits['capture_writes']==limits['capture_readbacks']==2 and limits['reference_capture_archive_loads']==1
 assert owner.STAGE.endswith('_003')and '003'in str(owner.ROOT)
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


class CheckpointModel:
 training=True
 def state_dict(self):return {'model':'actual-state'}
 def named_parameters(self):return [('backbone.weight',object()),('readout.weight',object())]


class CheckpointOptimizer:
 def state_dict(self):return {'AdamW':'actual-slots'}


def candidate_fixture(monkeypatch,completed):
 main=main_fixture();order=owner.make_orders(main)
 plan={'orders':order,'runtime':{'gpu':{'driver_version':'610.88'}},'driver_requalification':{'equivalence_proven':False},'inputs':{'checkpoint_receipt':{'path':'/original/CP3328','sha256':'a'*64},'main_plan':{'path':'/original/MAIN','sha256':'b'*64},'original_final_receipt':{'path':'/original/FINAL','sha256':'c'*64}}}
 steps=[{'logical_step':i+1,'skip_update':False}for i in order['reordered'][:completed]]
 monkeypatch.setattr(owner.core,'training_rng',lambda:{'actual':'RNG'})
 return CheckpointModel(),CheckpointOptimizer(),plan,steps


@pytest.mark.parametrize('completed',[64,128,192,256,320,366])
def test_six_durable_checkpoints_bind_actual_progress_and_permuted_prefix(monkeypatch,completed):
 model,optimizer,plan,steps=candidate_fixture(monkeypatch,completed);binding={'path':'/002plan','sha256':'c'*64}
 cp=owner.checkpoint_state(model,optimizer,binding=binding,plan=plan,steps=steps)
 assert cp['schema_version']==owner.CHECKPOINT_SCHEMA and cp['study_plan']==binding
 assert cp['progress']['optimizer_updates_completed']==completed and cp['progress']['optimizer_updates_remaining']==366-completed
 assert cp['progress']['optimizer_step_count_after_restore']==3328+completed
 assert cp['visited_source_batch_order']==plan['orders']['reordered'][:completed]
 assert cp['visited_source_batch_order_sha256']==owner.core.state_sha(cp['visited_source_batch_order'])
 assert cp['model_state']==model.state_dict()and cp['optimizer_state']==optimizer.state_dict()and cp['rng_state']=={'actual':'RNG'}
 assert cp['steps']==steps and cp['steps']is not steps and 'trajectory'not in cp and 'execution_plan'not in cp


@pytest.mark.parametrize('change',['unscheduled','renumbered','skip'])
def test_checkpoint_cannot_claim_unfinished_or_chronologically_renumbered_progress(monkeypatch,change):
 completed=63 if change=='unscheduled'else 64;model,optimizer,plan,steps=candidate_fixture(monkeypatch,completed)
 if change=='renumbered':steps[0]['logical_step']=3329
 if change=='skip':steps[-1]['skip_update']=True
 with pytest.raises(ValueError,match='CHECKPOINT_PROGRESS'):owner.checkpoint_state(model,optimizer,binding={},plan=plan,steps=steps)


def test_checkpoint_publication_uses_native_full_readback_and_retains_previous_file(monkeypatch,tmp_path):
 model,optimizer,plan,steps=candidate_fixture(monkeypatch,64);binding={'path':'/plan','sha256':'a'*64};writes=[];counts={}
 monkeypatch.setattr(owner,'ROOT',tmp_path)
 def native(path,checkpoint,actual_counts):
  assert actual_counts is counts;writes.append((path.name,checkpoint));return {'path':str(path),'sha256':'b'*64}
 monkeypatch.setattr(owner.core,'future_cuda_publish_checkpoint',native)
 monkeypatch.setattr(owner,'publish',lambda path,data:{'path':str(path),'sha256':'c'*64})
 actual,cp=owner.publish_candidate_checkpoint(model,optimizer,binding=binding,plan=plan,steps=steps,counts=counts)
 assert writes==[('UPDATE000064.pt',cp)]and actual['receipt']['path'].endswith('UPDATE000064.json')
 assert actual['checkpoint_tree_sha256']==owner.core.state_sha(cp)and actual['main_trajectory_claim']is False


def test_external_stop_is_consumed_and_reports_completed_durable_checkpoints(monkeypatch,tmp_path):
 _,records=mock_claimed_run(monkeypatch,tmp_path);handlers={};checkpoint={'receipt':{'path':'/002/UPDATE000064.json','sha256':'d'*64}}
 monkeypatch.setattr(owner.signal,'signal',lambda sig,handler:handlers.setdefault(sig,handler))
 monkeypatch.setattr(owner,'retained_checkpoint_receipts',lambda binding:[checkpoint])
 monkeypatch.setattr(owner.time,'monotonic',lambda:0.)
 def stopped(*args):handlers[owner.signal.SIGTERM](owner.signal.SIGTERM,None)
 monkeypatch.setattr(owner,'physical',stopped)
 with pytest.raises(InterruptedError,match='EXTERNAL_SIGNAL'):owner.run('a'*64)
 assert records[-1][1]['status']=='FAILED_CONSUMED'and records[-1][1]['retained_candidate_checkpoints']==[checkpoint]
 assert records[0][1]['technical_parameter_fits']==2 and records[0][1]['optimizer_updates']==732
 assert records[-1][1]['progress']['candidate_started']is False


def test_metric_comparison_retains_five_tf_support_and_distinguishes_absolute_bias():
 reference={};candidate={}
 for tf in owner.train.TF:
  r={'count':162,'unique_target_bars':162,'actual_positive_fraction':.5,'probability_minus_actual':.10,'Brier':.25,'logloss':.7,'AUC':None}
  c={**r,'probability_minus_actual':-.04,'Brier':.26,'logloss':.72}
  reference[tf]={task:copy.deepcopy(r)for task in ('activity','side')};candidate[tf]={task:copy.deepcopy(c)for task in ('activity','side')}
 actual=owner.metric_comparison({'ALL':candidate},{'ALL':reference})
 assert set(actual['ALL'])==set(owner.train.TF)
 for tf in owner.train.TF:
  cell=actual['ALL'][tf]['side'];assert cell['count']==162
  assert cell['absolute_probability_residual_delta']==pytest.approx(-.06)
  assert cell['Brier_delta']==pytest.approx(.01)and cell['AUC_delta']is None
 candidate[owner.train.TF[0]]['side']['count']=161
 with pytest.raises(ValueError,match='SAME_METRIC_SUPPORT'):owner.metric_comparison({'ALL':candidate},{'ALL':reference})


@pytest.mark.parametrize('difference',[0,63,64])
def test_chronological_training_stops_at_first_changed_step_and_preserves_prior_checkpoints(monkeypatch,tmp_path,difference):
 main=main_fixture();plan={'orders':owner.make_orders(main)};original=[None]*3328+reference_steps();calls=[];saved=[];artifacts=[];counts={'optimizer_steps':0}
 monkeypatch.setattr(owner,'ROOT',tmp_path);monkeypatch.setattr(owner.train,'packet',lambda full,batch:batch)
 def step(model,optimizer,packet,actual_main,actual_counts,index):
  calls.append(index);actual_counts['optimizer_steps']+=1;actual=copy.deepcopy(original[index])
  actual['skip_update']=False
  if index==3328+difference:actual['loss']+=1
  return actual
 # Original real step dictionaries include skip_update=False.
 for item in original[3328:]:item['skip_update']=False
 monkeypatch.setattr(owner.train,'training_step',step)
 monkeypatch.setattr(owner,'publish',lambda path,data:(artifacts.append((path.name,data))or{'path':str(path),'sha256':'e'*64}))
 def checkpoint(*args,**kwargs):saved.append((kwargs['arm'],len(kwargs['steps'])));return {},{}
 monkeypatch.setattr(owner,'publish_candidate_checkpoint',checkpoint);progress={}
 with pytest.raises(ValueError,match='REFERENCE_STEP_BITEXACT'):
  owner.train_arm(object(),object(),{'batches':list(range(3694))},main,plan,{},counts,'chronological',progress,original)
 assert calls==list(range(3328,3329+difference))and counts['optimizer_steps']==difference+1
 assert saved==[('chronological',64)]if difference==64 else saved==[]
 assert artifacts[-1][0]=='REFERENCE_STEP_MISMATCH.json'and artifacts[-1][1]['source_batch_index']==3328+difference
 assert artifacts[-1][1]['candidate_started']is False and progress['attempting_original_source_batch_index']==3328+difference


def reference_capture_fixture(monkeypatch,tmp_path):
 issue=np.array([1,2,3],np.int64);outputs={name:np.arange(15,dtype=np.float32).reshape(3,5)for name in ('activity_logits','side_logits')}
 arrays={'issue_ns':issue,**{f'cursor3694__{name}':a for name,a in outputs.items()}};path=tmp_path/'original.npz';np.savez_compressed(path,**arrays)
 reference={'frozen_capture':owner.ref(path),'frozen_capture_array_headers':{k:{'shape':list(v.shape),'dtype':v.dtype.str,'sha256':owner.train.array_sha(v)}for k,v in arrays.items()}}
 return issue,outputs,reference,dict.fromkeys(owner.LIMITS,0)


def test_new_driver_control_requires_both_original_final_heads_bitexact(monkeypatch,tmp_path):
 issue,outputs,reference,counts=reference_capture_fixture(monkeypatch,tmp_path)
 report=owner.reference_logits_parity(outputs,issue,reference,counts)
 assert report['bitexact']is True and report['tolerance']==0. and set(report['heads'])=={'activity_logits','side_logits'}
 assert all(x['mismatched_elements']==0 for x in report['heads'].values())and counts['reference_capture_archive_loads']==1


@pytest.mark.parametrize('head',['activity_logits','side_logits'])
@pytest.mark.parametrize('difference',['one_ulp','dtype'])
def test_any_control_head_change_fails_without_tolerance_expansion(monkeypatch,tmp_path,head,difference):
 issue,outputs,reference,counts=reference_capture_fixture(monkeypatch,tmp_path)
 if difference=='one_ulp':outputs[head][1,1]=np.nextafter(outputs[head][1,1],np.float32(np.inf))
 else:outputs[head]=outputs[head].astype(np.float64)
 report=owner.reference_logits_parity(outputs,issue,reference,counts)
 assert report['bitexact']is False and report['heads'][head]['bitexact']is False and report['tolerance']==0.
 if difference=='one_ulp':assert report['heads'][head]['mismatched_elements']==1


def test_original_capture_issue_axis_mismatch_stops_qualification(monkeypatch,tmp_path):
 issue,outputs,reference,counts=reference_capture_fixture(monkeypatch,tmp_path)
 with pytest.raises(ValueError,match='ISSUE_AXIS'):owner.reference_logits_parity(outputs,issue+1,reference,counts)
