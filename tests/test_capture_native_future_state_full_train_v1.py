"""Full continuous TRAIN capture contracts; fake forwards and tiny CPU copy only."""
import copy
import numpy as np
import pytest
import torch
from gx1.audit import capture_native_future_state_full_train_v1 as owner
from gx1.contracts.immutable_event_authority_v1 import ImmutableEventAuthorityError


FLAGS=('training_enabled','native_training_enabled','native_launch_authorized','full_epoch_training_allowed','full_val_allowed')


def test_exact_forward_only_budget_and_private_one_use_gate():
 l=owner.LIMITS
 assert l['model_constructions']==l['checkpoint_loads']==l['model_state_restores']==1
 assert l['model_forwards']==l['backbone_forwards']==l['readout_forwards']==1385
 assert l['capture_writes']==l['capture_readbacks']==1
 assert all(l[k]==0 for k in ('optimizer_constructions','technical_parameter_fits','backward_calls','optimizer_steps','checkpoint_writes','checkpoint_readbacks','cohort_archive_loads','autograd_calls','normalization_fits','prior_fits','CONTROL_VAL_TEST_rows'))
 binding={'path':'/capture/plan','sha256':'a'*64};gate=owner.gate_expected(binding,True)
 assert gate['plan']==binding and gate['remaining_invocations']==1 and gate['execution_authorized']is True
 assert gate['training_authorized']is False and gate['full_epoch_training_allowed']is False and gate['test_authorized']is False and gate['relaunch_allowed']is False
 assert owner.gate_expected(binding,False)['remaining_invocations']==0
 for admitted in (False,True):owner.validate_flags(dict.fromkeys(FLAGS,False),admitted=admitted)


@pytest.mark.parametrize('admitted',[False,True])
@pytest.mark.parametrize('flag',FLAGS)
def test_every_training_flag_stays_false_even_when_capture_is_admitted(admitted,flag):
 flags=dict.fromkeys(FLAGS,False);flags[flag]=True
 with pytest.raises(ValueError,match='ALL_TRAINING_FLAGS_FALSE'):owner.validate_flags(flags,admitted=admitted)


def test_full_e256_geometry_has_every_issue_once_and_short_final246():
 batches=owner.score_batches()
 assert len(batches)==1385 and all(len(x)==256 for x in batches[:-1])and len(batches[-1])==246
 assert np.array_equal(np.concatenate(batches),np.arange(354550,dtype=np.int64))


def outputs_for(take):
 value=np.repeat(take[:,None],5,axis=1).astype(np.float32)
 return {'activity_logits':value,'side_logits':-value}


def pass1_fixture():
 whole=outputs_for(np.arange(owner.ROWS,dtype=np.int64))
 batches=[{'rows':len(take),'output_sha256':{k:owner.train.array_sha(v)for k,v in outputs_for(take).items()}}for take in owner.score_batches()]
 ref={'rows':owner.ROWS,'batch_count':owner.BATCHES,'batches':batches,'scores':{'per_tf':'native proper sums'},'known_activity_cells':1772750,
  'proper_partial_NLL':.625,'initial_prior_exact':False,'output_sha256':{k:owner.train.array_sha(v)for k,v in whole.items()},'score_scope':'whole continuous TRAIN E256',
  'extra_original_rank_diagnostics':{'retained':True}}
 return ref,whole


@pytest.mark.parametrize('change',['rows','short_batch','head_inventory','initial_prior'])
def test_original_pass1_geometry_cannot_be_reinterpreted(change):
 ref,_=pass1_fixture()
 if change=='rows':ref['rows']-=1
 elif change=='short_batch':ref['batches'][-1]['rows']=245
 elif change=='head_inventory':ref['batches'][0]['output_sha256'].pop('activity_logits')
 else:ref['initial_prior_exact']=True
 with pytest.raises(ValueError,match='ORIGINAL_PASS1_GEOMETRY_HASHES'):owner.validate_pass1(ref)


class CaptureArray:
 def __init__(self,value):self.value=value
 def detach(self):return self
 def cpu(self):return self
 def numpy(self):return self.value


class CaptureModel:
 def __init__(self):self.callbacks=[];self.removals=0
 def register_forward_hook(self,callback):
  self.callbacks.append(callback);model=self
  class Handle:
   def remove(self):model.callbacks.remove(callback);model.removals+=1
  return Handle()
 def emit(self,arrays):
  for callback in list(self.callbacks):callback(self,(),{k:CaptureArray(v)for k,v in arrays.items()})


def fake_scoring(monkeypatch,reference,change=None):
 model=CaptureModel();context={'prior':object()};main={'recipe':'unchanged'};seen=[]
 monkeypatch.setattr(owner.train,'packet',lambda actual,take:{'take':take.copy()})
 def score(actual_model,packets,actual_main,prior,*,initial,equal_tf):
  assert actual_model is model and actual_main is main and prior is context['prior']and initial is False and equal_tf is True
  for ordinal,packet in enumerate(packets):
   take=packet['take'];seen.append(take.copy());arrays=outputs_for(take)
   if ordinal==0 and change:
    if change=='hash':arrays['side_logits'][0,0]=np.float32(0.)
    elif change=='dtype':arrays['side_logits']=arrays['side_logits'].astype(np.float64)
    elif change=='shape':arrays['side_logits']=arrays['side_logits'][:-1]
    elif change=='heads':arrays['extra_logits']=arrays['side_logits'].copy()
   model.emit(arrays)
  return {k:reference[k]for k in owner.SCORE_KEYS}
 monkeypatch.setattr(owner.core,'future_cuda_score',score)
 return model,context,main,seen


def test_full_capture_replays_all1385_batch_hashes_and_entire354550_issue_axis(monkeypatch):
 reference,expected=pass1_fixture();model,context,main,seen=fake_scoring(monkeypatch,reference);progress={}
 score,arrays=owner.capture_score(model,context,main,reference,progress)
 assert len(seen)==1385 and len(seen[-1])==246 and np.array_equal(np.concatenate(seen),np.arange(354550))
 assert set(arrays)==set(owner.HEADS)and all(np.array_equal(arrays[k],v)for k,v in expected.items())
 assert all(owner.train.array_sha(arrays[k])==reference['output_sha256'][k]for k in owner.HEADS)
 assert score=={k:reference[k]for k in owner.SCORE_KEYS}and 'extra_original_rank_diagnostics'not in score
 assert progress['validated_score_batches']==1385 and progress['validated_issue_rows']==354550
 assert model.removals==1 and not model.callbacks


def test_first_batch_signed_zero_hash_mismatch_aborts_immediately(monkeypatch):
 reference,_=pass1_fixture();model,context,main,seen=fake_scoring(monkeypatch,reference,'hash');progress={}
 with pytest.raises(ValueError,match='BATCH_OUTPUT_BITEXACT_1'):owner.capture_score(model,context,main,reference,progress)
 assert len(seen)==1 and model.removals==1 and not model.callbacks
 assert progress.get('validated_score_batches',0)==0
 failure=progress['batch_parity_failure'];assert failure['batch_ordinal']==1 and failure['first_issue_index']==0 and failure['rows']==256
 assert failure['actual']!=failure['original_PASS1']


@pytest.mark.parametrize('change',['dtype','shape','heads'])
def test_bad_batch_domain_is_rejected_before_another_forward_and_hook_removed(monkeypatch,change):
 reference,_=pass1_fixture();model,context,main,seen=fake_scoring(monkeypatch,reference,change)
 with pytest.raises(ValueError,match='CAPTURE_BATCH_'):owner.capture_score(model,context,main,reference,{})
 assert len(seen)==1 and model.removals==1 and not model.callbacks


def test_matching_batches_still_require_original_full_output_hash(monkeypatch):
 reference,_=pass1_fixture();reference['output_sha256']['side_logits']='f'*64
 model,context,main,seen=fake_scoring(monkeypatch,reference)
 with pytest.raises(ValueError,match='FULL_OUTPUT_BITEXACT'):owner.capture_score(model,context,main,reference,{})
 assert len(seen)==1385 and model.removals==1 and not model.callbacks


@pytest.mark.parametrize('change',['proper_partial_NLL','scores'])
def test_original_proper_scores_cannot_be_replaced_by_new_values(change):
 reference,_=pass1_fixture();score={k:copy.deepcopy(reference[k])for k in owner.SCORE_KEYS};score[change]={'foreign':True}
 with pytest.raises(ValueError,match='FULL_PASS1_SCORES_BITEXACT'):owner.validate_score(score,reference)


class TinyModel(torch.nn.Module):
 def __init__(self):
  super().__init__();self.backbone=torch.nn.Linear(1,1);self.readout=torch.nn.Linear(1,1);self.register_buffer('normalization_identity',torch.tensor([1.]))


def checkpoint_fixture():
 model=TinyModel();order=list(range(3328,3694));steps=[{'logical_step':i+1,'skip_update':False,'loss':float(i)}for i in order]
 main={'schedule':{'logical_batches':3694,'batches_per_pass':3694,'update_flags':[True]*3694}}
 parent={'orders':{'chronological':order},'inputs':{'checkpoint_receipt':{'path':'/original/CP3328.json','sha256':'b'*64},'main_plan':{'path':'/original/MAIN.json','sha256':'c'*64}},
  'runtime':{'library_versions':{'torch':'fixed'},'gpu':{'driver_version':'610.88'}},'driver_requalification':{'equivalence_proven':False}}
 state=copy.deepcopy(model.state_dict())
 for name,value in state.items():
  if name!='normalization_identity':value.fill_(.25)
 cp={'schema_version':owner.previous.CHECKPOINT_SCHEMA,'study_plan':copy.deepcopy(owner.continuation.PARENT),'study_arm':'chronological',
  'source_checkpoint_receipt':copy.deepcopy(parent['inputs']['checkpoint_receipt']),'original_main_plan':copy.deepcopy(parent['inputs']['main_plan']),
  'producer_runtime':copy.deepcopy(parent['runtime']),'driver_requalification':copy.deepcopy(parent['driver_requalification']),
  'source_batch_order':order.copy(),'source_batch_order_sha256':owner.core.state_sha(order),'visited_source_batch_order':order.copy(),
  'visited_source_batch_order_sha256':owner.core.state_sha(order),'progress':owner.continuation.study_progress(order,366),
  'model_state':state,'optimizer_state':{'authentic':'full AdamW source state'},'scheduler_state':None,'rng_state':{'authentic':'all CPU CUDA NumPy Python states'},
  'training':True,'optimizer_parameter_names':[n for n,_ in model.named_parameters()],'steps':steps,'authority':'diagnostic_only_no_promotion_or_MAIN_trajectory_claim'}
 receipt={'schema_version':'gx1_native_future_state_bias_suffix_probe_receipt_v3','study_plan':copy.deepcopy(owner.continuation.PARENT),'study_arm':'chronological',
  'main_trajectory_claim':False,'progress':copy.deepcopy(cp['progress']),'checkpoint':{'path':str(owner.previous.ROOT/'chronological/FINAL.pt'),'sha256':'a'*64},
  'checkpoint_tree_sha256':owner.core.state_sha(cp),'source_batch_order_sha256':cp['source_batch_order_sha256'],'visited_source_batch_order_sha256':cp['visited_source_batch_order_sha256']}
 original={'plan':copy.deepcopy(parent['inputs']['main_plan']),'report':{'steps':[None]*3328+copy.deepcopy(steps)}}
 proof={k:owner.core.state_sha(cp[k])for k in ('model_state','optimizer_state','rng_state','scheduler_state','training','optimizer_parameter_names')}
 proof['position']=owner.core.state_sha(owner.train.position(main['schedule'],3694));proof['steps']=owner.core.state_sha(original['report']['steps'])
 return model,cp,receipt,parent,main,original,{'proof':proof}


def test_authentic_chronological_final_validates_all8_original_components_without_optimizer():
 _,cp,receipt,parent,main,original,final=checkpoint_fixture();before=owner.core.state_sha(cp)
 assert owner.validate_checkpoint(cp,receipt,parent,main,original,final)==final['proof']
 assert owner.core.state_sha(cp)==before and 'trajectory'not in cp


@pytest.mark.parametrize('change',['tree','schema','arm','plan','progress','original_step'])
def test_invalid_v3_provenance_is_rejected_before_model_copy(change):
 _,cp,receipt,parent,main,original,final=checkpoint_fixture()
 if change=='tree':receipt['checkpoint_tree_sha256']='f'*64
 elif change=='schema':cp['schema_version']='fake_MAIN_schema'
 elif change=='arm':cp['study_arm']='reordered'
 elif change=='plan':cp['study_plan']['sha256']='f'*64
 elif change=='progress':cp['progress']['optimizer_updates_completed']=365
 else:cp['steps'][0]['loss']+=1
 if change!='tree':receipt['checkpoint_tree_sha256']=owner.core.state_sha(cp)
 with pytest.raises(ValueError):owner.validate_checkpoint(cp,receipt,parent,main,original,final)


@pytest.mark.parametrize('key',owner.previous.REFERENCE_KEYS)
def test_every_original_final_component_is_required_before_first_forward(key):
 _,cp,receipt,parent,main,original,final=checkpoint_fixture();final['proof'][key]='f'*64
 with pytest.raises(ValueError,match='REFERENCE_STATE'):owner.validate_checkpoint(cp,receipt,parent,main,original,final)


def test_model_only_restore_is_bitexact_and_does_not_construct_optimizer_or_mutate_source(monkeypatch):
 model,cp,_,_,_,_,final=checkpoint_fixture();before=owner.core.state_sha(cp)
 monkeypatch.setattr(torch.optim,'AdamW',lambda *args,**kwargs:pytest.fail('capture constructed an optimizer'))
 proof=owner.restore_model_only(model,cp,final['proof']['model_state'])
 assert proof['model_state_sha256']==final['proof']['model_state']and proof['normalization_input_prior_buffers_identical']is True
 assert model.training is True and all(p.grad is None for p in model.parameters())
 with torch.no_grad():next(model.parameters()).add_(1)
 assert owner.core.state_sha(cp)==before


@pytest.mark.parametrize('change',['names','shape','dtype','nonfinite','buffer','expected_hash'])
def test_model_copy_headers_and_actual_normalization_identity_fail_before_copy(change):
 model,cp,_,_,_,_,final=checkpoint_fixture();first=next(iter(cp['model_state']));expected=final['proof']['model_state'];before=owner.core.state_sha(model.state_dict())
 if change=='names':cp['optimizer_parameter_names'].reverse()
 elif change=='shape':cp['model_state'][first]=torch.zeros(2)
 elif change=='dtype':cp['model_state'][first]=cp['model_state'][first].double()
 elif change=='nonfinite':cp['model_state'][first].fill_(float('nan'))
 elif change=='buffer':cp['model_state']['normalization_identity'].fill_(2.)
 else:expected='f'*64
 with pytest.raises(ValueError):owner.restore_model_only(model,cp,expected)
 assert owner.core.state_sha(model.state_dict())==before


def test_model_copy_postcopy_corruption_cannot_pass(monkeypatch):
 model,cp,_,_,_,_,final=checkpoint_fixture();load=model.load_state_dict
 def corrupt(state,strict):
  result=load(state,strict=strict)
  with torch.no_grad():next(model.parameters()).add_(1)
  return result
 monkeypatch.setattr(model,'load_state_dict',corrupt)
 with pytest.raises(ValueError,match='MODEL_COPY_BITEXACT'):owner.restore_model_only(model,cp,final['proof']['model_state'])


def small_capture():
 take=np.arange(7,dtype=np.int64)
 return {'issue_ns':take*300_000_000_000,'source_rows':take+100,'selected_positions':take+200,**outputs_for(take)}


def test_atomic_capture_publication_has_exact5_array_byte_readback_and_no_replace(monkeypatch,tmp_path):
 monkeypatch.setattr(owner,'ROOT',tmp_path);arrays=small_capture();counts={'capture_writes':0,'capture_readbacks':0}
 report=owner.publish_capture(arrays,counts);binding=report['capture'];before=owner.sha(binding['path'])
 assert counts=={'capture_writes':1,'capture_readbacks':1}
 with np.load(binding['path'],allow_pickle=False)as z:
  assert set(z.files)==set(arrays)and all(owner.train.array_sha(z[k])==owner.train.array_sha(a)for k,a in arrays.items())
 with pytest.raises(ImmutableEventAuthorityError,match=r'^immutable event already exists: ')as error:owner.publish_capture(arrays,counts)
 assert str(error.value)=='immutable event already exists: '+binding['path']
 assert owner.sha(binding['path'])==before


def test_disk_readback_rejects_signed_zero_byte_change(monkeypatch,tmp_path):
 monkeypatch.setattr(owner,'ROOT',tmp_path);arrays=small_capture();counts={'capture_writes':0,'capture_readbacks':0}
 def wrong_publish(path,data):
  changed={k:v.copy()for k,v in arrays.items()};changed['side_logits'][0,0]=np.float32(0.)
  np.savez_compressed(path,**changed);return owner.ref(path)
 monkeypatch.setattr(owner,'publish',wrong_publish)
 with pytest.raises(ValueError,match='CAPTURE_DISK_BYTE_IDENTITY'):owner.publish_capture(arrays,counts)
 assert counts=={'capture_writes':1,'capture_readbacks':0}


def test_incomplete_capture_inventory_is_not_published(monkeypatch):
 arrays=small_capture();arrays.pop('selected_positions')
 monkeypatch.setattr(owner,'publish',lambda *args:pytest.fail('incomplete capture was published'))
 with pytest.raises(ValueError,match='FULL_CAPTURE_INVENTORY'):owner.publish_capture(arrays,{'capture_writes':0,'capture_readbacks':0})


def test_main_inventory_rejected_before_other_reads_or_source_hashing(monkeypatch):
 main_path=str(owner.train.MAIN_PLAN);monkeypatch.setattr(owner,'ref',lambda path:{'path':str(path),'sha256':'a'*64})
 def read(binding):
  assert binding['path']==main_path
  return {'source_owners':{'/untrusted.bin':'a'*64}}
 monkeypatch.setattr(owner,'read',read);monkeypatch.setattr(owner.train,'expected_source_names',lambda:{'safe.py'})
 monkeypatch.setattr(owner,'sha',lambda *args:pytest.fail('source hashed before exact inventory'))
 with pytest.raises(ValueError,match='EXACT_SOURCE_INVENTORY'):owner.build_plan(runtime={})


def mocked_run(monkeypatch,tmp_path):
 from gx1.contracts import gx1_capped_execution_v1 as guard
 root=tmp_path/'capture';binding={'path':'/capture/plan.json','sha256':'a'*64};records=[];handlers={}
 monkeypatch.setattr(owner,'ROOT',root);monkeypatch.setattr(owner,'inspect',lambda *args,**kwargs:({},binding,{},{}))
 monkeypatch.setattr(guard,'require_native_future_state_full_train_capture_execution',lambda **kwargs:{'plan':binding,'output_root':str(root),'outer_seconds':3600})
 monkeypatch.setattr(owner.core,'future_cpu_process_identity',lambda:{'pid':1})
 monkeypatch.setattr(owner,'publish',lambda path,value:(records.append((path.name,value))or{'path':str(path),'sha256':'b'*64}))
 monkeypatch.setattr(owner.signal,'signal',lambda sig,handler:handlers.setdefault(sig,handler));monkeypatch.setattr(owner.signal,'setitimer',lambda *args:None)
 return records,handlers


def test_claimed_first_batch_failure_is_consumed_with_exact_progress(monkeypatch,tmp_path):
 records,_=mocked_run(monkeypatch,tmp_path);monkeypatch.setattr(owner.time,'monotonic',lambda:0.)
 def fail(plan,binding,live,progress):
  progress['batch_parity_failure']={'batch_ordinal':1,'rows':256};raise ValueError('BATCH_OUTPUT_BITEXACT_1')
 monkeypatch.setattr(owner,'physical',fail)
 with pytest.raises(ValueError,match='BATCH_OUTPUT_BITEXACT_1'):owner.run('a'*64)
 assert [name for name,_ in records]==['CLAIM.json','TERMINAL.json']
 terminal=records[-1][1];assert terminal['status']=='FAILED_CONSUMED'and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False
 assert terminal['progress']['validated_score_batches']==0 and terminal['progress']['batch_parity_failure']['batch_ordinal']==1
 assert records[0][1]['technical_parameter_fits']==records[0][1]['optimizer_updates']==0


def test_startup_expiry_is_consumed_with_terminal_before_any_forward(monkeypatch,tmp_path):
 records,_=mocked_run(monkeypatch,tmp_path);ticks=iter([0.,4000.]);monkeypatch.setattr(owner.time,'monotonic',lambda:next(ticks))
 monkeypatch.setattr(owner,'physical',lambda *args:pytest.fail('expired startup entered model execution'))
 with pytest.raises(ValueError,match='WALL_BEFORE_INPUTS'):owner.run('a'*64)
 assert [name for name,_ in records]==['CLAIM.json','TERMINAL.json']and records[-1][1]['remaining_invocations']==0


def test_external_stop_retains_progress_and_never_relaunches(monkeypatch,tmp_path):
 records,handlers=mocked_run(monkeypatch,tmp_path);monkeypatch.setattr(owner.time,'monotonic',lambda:0.)
 def stop(plan,binding,live,progress):progress.update(validated_score_batches=64,validated_issue_rows=16384);handlers[owner.signal.SIGTERM](owner.signal.SIGTERM,None)
 monkeypatch.setattr(owner,'physical',stop)
 with pytest.raises(InterruptedError,match='EXTERNAL_SIGNAL'):owner.run('a'*64)
 assert records[-1][1]['status']=='FAILED_CONSUMED'and records[-1][1]['progress']['validated_score_batches']==64


def test_success_publishes_generated_capture_report_and_complete_terminal(monkeypatch,tmp_path):
 records,_=mocked_run(monkeypatch,tmp_path);binding={'path':'/capture/plan.json','sha256':'a'*64};pref={'path':'/policy','sha256':'c'*64}
 live={'source_commit':'fixedHEAD','wsl_boot_id':'fixedBOOT','physical_windows_boot_id':4};native={'counts':copy.deepcopy(owner.LIMITS),'capture':{'path':'/capture/CAPTURE.npz'}}
 monkeypatch.setattr(owner,'inspect',lambda *args,**kwargs:({'source_owners':{}},binding,pref,live))
 monkeypatch.setattr(owner,'physical',lambda *args:native);monkeypatch.setattr(owner,'input_bookend',lambda plan:None)
 monkeypatch.setattr(owner,'ref',lambda path:pref if path==owner.POLICY else binding)
 monkeypatch.setattr(owner.subprocess,'check_output',lambda *args,**kwargs:'fixedHEAD\n')
 class BootPath:
  def read_text(self):return'fixedBOOT\n'
 monkeypatch.setattr(owner,'Path',lambda path:BootPath());monkeypatch.setattr(owner.core,'physical_windows_boot_id',lambda:4)
 monkeypatch.setattr(owner.core.common,'firewall_state',lambda:{'denied_attempts':0});monkeypatch.setattr(owner.time,'monotonic',lambda:0.)
 result=owner.run('a'*64)
 assert [name for name,_ in records]==['CLAIM.json','RESULT.json','TERMINAL.json']and records[1][1]['report']is native
 assert records[1][1]['status']=='COMPLETE_NATIVE_FULL_TRAIN_CAPTURE'and records[-1][1]['result']==result['result']
 assert records[-1][1]['status']=='COMPLETE'and records[-1][1]['remaining_invocations']==0 and records[-1][1]['test_data_used']is False


def test_exact_preflight_cli_never_creates_claim_output_or_model(monkeypatch,tmp_path,capsys):
 import sys,json
 root=tmp_path/'uncreated';seen=[];monkeypatch.setattr(owner,'ROOT',root);monkeypatch.setattr(sys,'argv',['module','--plan-sha256','a'*64,'--preflight'])
 def inspect(sha,*,admitted):seen.append((sha,admitted));return {},{'path':'/plan','sha256':sha},{},{}
 monkeypatch.setattr(owner,'inspect',inspect);monkeypatch.setattr(owner,'run',lambda *args:pytest.fail('preflight started capture'))
 monkeypatch.setattr(owner,'publish',lambda *args:pytest.fail('preflight published output'))
 owner.main();report=json.loads(capsys.readouterr().out)
 assert seen==[('a'*64,False)]and not root.exists()
 assert report['models']==report['checkpoint_payloads']==report['optimizer_updates']==0 and report['CUDA_initialized']is False


def test_noncanonical_cli_order_rejected_before_inspect(monkeypatch):
 import sys
 monkeypatch.setattr(sys,'argv',['module','--preflight','--plan-sha256','a'*64])
 monkeypatch.setattr(owner,'inspect',lambda *args,**kwargs:pytest.fail('noncanonical CLI entered inspect'))
 with pytest.raises(ValueError,match='EXACT_CLI'):owner.main()


def preregistration_fixture():
 capture={'rows':owner.ROWS,'batch_rows':owner.BATCH,'forwards':owner.BATCHES,'models':1,'checkpoint_payload_loads':1,'checkpoint_restores_model_only':1,
  'backward_calls':0,'checkpoint_writes':0,'fits':0,'normalization_fits':0,'optimizer_constructions':0,'optimizer_updates':0,'prior_fits':0,'CONTROL_VAL_TEST_rows':0,
  'fail_at_first_mismatch_no_tolerance_expansion':True,'require_each_original_PASS1_batch_output_hash_bitexact':True,'require_full_output_hash_and_proper_score_tree_bitexact':True,
  'keep_all_genuine_inputs_eight_families_D1macro_five_TF':True}
 main_ref={'path':'/MAIN.json','sha256':'a'*64};result_ref={'path':'/RESULT.json','sha256':'b'*64};model_sha='c'*64
 registration={'schema_version':'gx1_native_selective_move_preregistration_v1','status':'PREREGISTERED_NOT_EXECUTION_AUTHORITY','execution_authorized':False,
  'capture':capture,'model_identity':{'checkpoint_receipt':copy.deepcopy(owner.CHECKPOINT_RECEIPT),'original_MAIN_plan':main_ref,'original_MAIN_result':result_ref,'model_state_sha256':model_sha}}
 return registration,main_ref,result_ref,model_sha


def test_preregistration_binds_read_only_capture_budget_and_exact_model():
 registration,main_ref,result_ref,model_sha=preregistration_fixture()
 assert owner.validate_preregistration(registration,main_ref,result_ref,model_sha)is True


@pytest.mark.parametrize('change',['boolean_model_count','extra_optimizer_update','foreign_checkpoint','foreign_model'])
def test_preregistered_budget_types_and_model_identity_cannot_be_widened(change):
 registration,main_ref,result_ref,model_sha=preregistration_fixture()
 if change=='boolean_model_count':registration['capture']['models']=True
 elif change=='extra_optimizer_update':registration['capture']['optimizer_updates']=1
 elif change=='foreign_checkpoint':registration['model_identity']['checkpoint_receipt']['sha256']='f'*64
 else:registration['model_identity']['model_state_sha256']='f'*64
 with pytest.raises(ValueError,match='EXACT_PREREGISTERED_CAPTURE_SCOPE_MODEL'):owner.validate_preregistration(registration,main_ref,result_ref,model_sha)



def provider_coverage_fixture(monkeypatch):
 monkeypatch.setattr(owner,'ROWS',3);rows=np.array([100,101,102],dtype=np.int64)
 normalization={name:{'fit_row_count':3,'field_names':[name+'.optional']}for name in ('signal','ctx_cont','mtf_m5','mtf_m15','mtf_h1','mtf_h4','mtf_d1')}
 class Provider:
  def __init__(self):
   self.calls=[];self.readiness={'rows':3,'input_ready':3,'input_not_ready':0,'unmaterialized_current_rows':0,
    'current_field_missingness':[{'field':'optional_unobserved','unobserved_current_rows':3}],'failed_conditions':{},'reason_counts_overlap':True}
   self.support={'source_rows':3,'input_ready_rows':3,'surfaces':{name:{'physical_rows':3,'field_names':v['field_names'],
    'observed_counts':[3]}for name,v in normalization.items()},'families':['fixture_family'],'macro_fields':['fixture_macro'],'field_lineage':{'optional_unobserved':'fixture'}}
  def readiness_report(self,actual_rows):self.calls.append(('readiness_report',actual_rows));return self.readiness
  def support_report(self,actual_rows):self.calls.append(('support_report',actual_rows));return self.support
 provider=Provider();return {'provider':provider,'source_rows':rows,'normalization':{'surfaces':normalization}},provider


def test_provider_coverage_reads_each_existing_report_once_and_keeps_optional_missingness(monkeypatch):
 full,provider=provider_coverage_fixture(monkeypatch);before=copy.deepcopy((provider.readiness,provider.support));result=owner.provider_coverage_reports(full)
 assert [(name,rows is full['source_rows'])for name,rows in provider.calls]==[('readiness_report',True),('support_report',True)]
 assert result['readiness_report']is provider.readiness and result['finite_support_report']is provider.support
 assert (provider.readiness,provider.support)==before
 assert result['readiness_report']['current_field_missingness'][0]['unobserved_current_rows']==3
 assert result['finite_support_report']['surfaces']['signal']['observed_counts']==[3]
 assert 'NOT true-observation counts'in result['support_observed_counts_semantics']and result['learned_utility'].startswith('UNKNOWN')
 assert result['reader_calls']=={'readiness_report':1,'support_report':1}


@pytest.mark.parametrize('change',['not_ready','unmaterialized','physical_rows','field_names','count_outside_rows'])
def test_provider_coverage_rejects_wrong_whole_population_or_normalizer_binding(monkeypatch,change):
 full,provider=provider_coverage_fixture(monkeypatch)
 if change=='not_ready':provider.support['input_ready_rows']=2
 elif change=='unmaterialized':provider.readiness['unmaterialized_current_rows']=1
 elif change=='physical_rows':provider.support['surfaces']['mtf_d1']['physical_rows']=2
 elif change=='field_names':provider.support['surfaces']['signal']['field_names']=['wrong']
 else:provider.support['surfaces']['ctx_cont']['observed_counts']=[4]
 with pytest.raises(ValueError):owner.provider_coverage_reports(full)
 assert len(provider.calls)==2
