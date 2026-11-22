"""Focused004 continuation, authentic V3 restore and cold-qualification contracts."""
import copy
import random
import numpy as np
import pytest
import torch
from gx1.audit import continue_native_future_state_bias_suffix_v1 as owner


def main_fixture():
 return {'limits':{'main':{'optimizer_steps':3694}},
  'recipe':{'checkpoint_every_optimizer_updates':256,'train_batch_rows':96,'seed':20260911,'passes':1},
  'schedule':{'logical_batches':3694,'batches_per_pass':3694,'drop_last':False,'shuffle':False,'batch_rows':96,'rows':354550}}


def order_fixture():return owner.previous.make_orders(main_fixture())['reordered']


class TinyModel(torch.nn.Module):
 def __init__(self,buffer=False):
  super().__init__();self.backbone=torch.nn.Linear(1,1);self.readout=torch.nn.Linear(1,1)
  if buffer:self.register_buffer('normalization_identity',torch.tensor([1.]))


def make_rng(cursor):
 _,keys,_,gauss,cached=np.random.get_state()
 return {'python':random.getstate(),'numpy_name':'MT19937','numpy_keys':torch.from_numpy(keys.copy()),'numpy_position':cursor,
  'numpy_has_gauss':gauss,'numpy_cached':cached,'torch_cpu':torch.get_rng_state().clone(),'torch_cuda':[torch.arange(8,dtype=torch.uint8)]}


def checkpoint_fixture(cursor=192,buffer=False):
 model=TinyModel(buffer);optimizer=torch.optim.AdamW(model.parameters(),lr=.001);order=order_fixture()
 parent={'orders':{'reordered':order},'inputs':{'checkpoint_receipt':{'path':'/original/CP3328.json','sha256':'b'*64},
  'main_plan':{'path':'/original/MAIN.json','sha256':'c'*64}},'runtime':{'library_versions':{'torch':'fixed'},'gpu':{'driver_version':'610.88'}},
  'driver_requalification':{'equivalence_proven':False}}
 state=copy.deepcopy(model.state_dict())
 for k,v in state.items():
  if k!='normalization_identity':v.fill_(.25)
 slots=optimizer.state_dict();parameters=list(model.parameters());ids=slots['param_groups'][0]['params']
 slots['state']={i:{'step':torch.tensor(float(3328+cursor),dtype=torch.float32),'exp_avg':torch.zeros_like(p),'exp_avg_sq':torch.ones_like(p)}for i,p in zip(ids,parameters)}
 steps=[{'logical_step':i+1,'skip_update':False,'loss':float(n),'output_sha256':{'side_logits':str(n)}}for n,i in enumerate(order[:cursor])]
 cp={'schema_version':owner.previous.CHECKPOINT_SCHEMA,'study_plan':copy.deepcopy(owner.PARENT),'study_arm':'reordered',
  'source_checkpoint_receipt':copy.deepcopy(parent['inputs']['checkpoint_receipt']),'original_main_plan':copy.deepcopy(parent['inputs']['main_plan']),
  'producer_runtime':copy.deepcopy(parent['runtime']),'driver_requalification':copy.deepcopy(parent['driver_requalification']),
  'source_batch_order':order.copy(),'source_batch_order_sha256':owner.core.state_sha(order),'visited_source_batch_order':order[:cursor],
  'visited_source_batch_order_sha256':owner.core.state_sha(order[:cursor]),'progress':owner.study_progress(order,cursor),
  'model_state':state,'optimizer_state':slots,'scheduler_state':None,'rng_state':make_rng(cursor),'training':True,
  'optimizer_parameter_names':[n for n,_ in model.named_parameters()],'steps':steps,
  'authority':'diagnostic_only_no_promotion_or_MAIN_trajectory_claim'}
 receipt={'schema_version':'gx1_native_future_state_bias_suffix_probe_receipt_v3','study_plan':copy.deepcopy(owner.PARENT),'study_arm':'reordered',
  'checkpoint':{'path':str(owner.previous.ROOT/'reordered'/f'UPDATE{cursor:06d}.pt'),'sha256':'a'*64},'checkpoint_tree_sha256':owner.core.state_sha(cp),
  'progress':copy.deepcopy(cp['progress']),'source_batch_order_sha256':cp['source_batch_order_sha256'],
  'visited_source_batch_order_sha256':cp['visited_source_batch_order_sha256'],'main_trajectory_claim':False}
 return model,optimizer,parent,cp,receipt


def fake_runtime(monkeypatch,initial):
 active={'value':copy.deepcopy(initial)}
 monkeypatch.setattr(owner.core,'training_rng',lambda:copy.deepcopy(active['value']))
 monkeypatch.setattr(owner.core,'restore_training_rng',lambda rng:active.update(value=copy.deepcopy(rng)))
 monkeypatch.setattr(owner.train,'runtime_compatible',lambda original,current:True)
 return active


def test_exact_budget_has_two_real_fits_and_only_174_new_scientific_updates():
 l=owner.LIMITS
 assert l['model_constructions']==1 and l['optimizer_constructions']==l['technical_parameter_fits']==2
 assert l['checkpoint_loads']==l['checkpoint_restores']==2
 assert l['model_forwards']==l['backbone_forwards']==l['readout_forwards']==319
 assert l['backward_calls']==l['optimizer_steps']==238
 assert l['checkpoint_writes']==l['checkpoint_readbacks']==3 and l['capture_writes']==l['capture_readbacks']==1
 assert l['normalization_fits']==l['prior_fits']==l['autograd_calls']==l['CONTROL_VAL_TEST_rows']==0
 assert owner.STAGE.endswith('_004')and owner.ROOT!=owner.previous.ROOT


def test_fixed_order_uses_64_replay_and_174_remaining_batches_without_moving_short_last():
 order=order_fixture();assert len(order[128:192])==64 and len(order[192:])==174
 assert order[192]==3518 and order[-1]==3693 and len(set(order))==366
 assert owner.study_progress(order,192)=={'optimizer_updates_completed':192,'optimizer_updates_remaining':174,'next_order_position':192,
  'last_original_source_batch_index':order[191],'restored_original_optimizer_step':3328,'optimizer_step_count_after_restore':3520}
 assert list(owner.NEW_CHECKPOINTS)==[256,320,366]
 assert [n-192 for n in owner.NEW_CHECKPOINTS]==[64,128,174]


def test_private_gate_and_flags_never_relaunch_or_open_full_epoch():
 binding={'path':'/004plan','sha256':'a'*64};a=owner.gate_expected(binding,True);p=owner.gate_expected(binding,False)
 assert a['status']=='ADMITTED_EXACT_COLD_QUALIFIED_SUFFIX_CONTINUATION'and a['remaining_invocations']==1
 assert a['plan']==binding and a['relaunch_allowed']is False and a['test_authorized']is False and a['full_epoch_training_allowed']is False
 assert p['remaining_invocations']==0 and p['execution_authorized']is False
 flags=dict.fromkeys(('training_enabled','native_training_enabled','native_launch_authorized','full_epoch_training_allowed','full_val_allowed'),False)
 owner.validate_flags(flags,admitted=False);flags['training_enabled']=flags['native_training_enabled']=True;owner.validate_flags(flags,admitted=True)
 flags['full_epoch_training_allowed']=True
 with pytest.raises(ValueError):owner.validate_flags(flags,admitted=True)


@pytest.mark.parametrize('cursor',[128,192])
def test_authentic_v3_has_exact_state_inventory_and_preserves_original_lineage(cursor):
 _,_,parent,cp,receipt=checkpoint_fixture(cursor);before=owner.core.state_sha(cp)
 proof=owner.validate_v3_checkpoint(cp,receipt,parent,cursor)
 assert set(proof)==set(owner.STATE_KEYS)and len(proof)==8
 assert cp['study_plan']==owner.PARENT and 'trajectory'not in cp and 'execution_plan'not in cp
 assert owner.core.state_sha(cp)==before


@pytest.mark.parametrize('change',['tree','schema','study_plan','arm','runtime','main','driver','order_prefix','progress','step','scheduler','training'])
def test_bad_v3_provenance_order_or_state_cannot_reach_parameter_copy(change):
 model,_,parent,cp,receipt=checkpoint_fixture();initial=owner.core.state_sha(model.state_dict())
 if change=='tree':receipt['checkpoint_tree_sha256']='f'*64
 elif change=='schema':cp['schema_version']='MAIN-schema'
 elif change=='study_plan':cp['study_plan']['sha256']='f'*64
 elif change=='arm':cp['study_arm']='chronological'
 elif change=='runtime':cp['producer_runtime']['gpu']['driver_version']='other'
 elif change=='main':cp['original_main_plan']['sha256']='f'*64
 elif change=='driver':cp['driver_requalification']['equivalence_proven']=True
 elif change=='order_prefix':cp['visited_source_batch_order'][0],cp['visited_source_batch_order'][1]=cp['visited_source_batch_order'][1],cp['visited_source_batch_order'][0]
 elif change=='progress':cp['progress']['next_order_position']=201
 elif change=='step':cp['steps'][-1]['logical_step']+=1
 elif change=='scheduler':cp['scheduler_state']={}
 else:cp['training']=False
 if change!='tree':receipt['checkpoint_tree_sha256']=owner.core.state_sha(cp)
 with pytest.raises(ValueError):owner.validate_v3_checkpoint(cp,receipt,parent,192)
 assert owner.core.state_sha(model.state_dict())==initial


def test_cold_restore_copies_full_components_and_retains_immutable_adamw_cpu_steps(monkeypatch):
 model,optimizer,parent,cp,receipt=checkpoint_fixture();active=fake_runtime(monkeypatch,make_rng(0));before=owner.core.state_sha(cp)
 expected=owner.strict_restore_v3(model,optimizer,cp,receipt,parent,192,parent['runtime'])
 assert owner.component_proof(owner.live_state(model,optimizer,cp['source_batch_order'],192,cp['steps']))==expected
 assert owner.core.state_sha(active['value'])==expected['rng_state']and model.training is True
 next(iter(optimizer.state.values()))['step'].add_(1)
 assert owner.core.state_sha(cp)==before


@pytest.mark.parametrize('change',['names','shape','dtype','nonfinite','optimizer_recipe','head_step','negative_variance','cuda_rng_header'])
def test_restore_headers_fail_before_any_model_copy(monkeypatch,change):
 model,optimizer,parent,cp,receipt=checkpoint_fixture();fake_runtime(monkeypatch,make_rng(0));first=next(iter(cp['model_state']));head=next(i for i,n in enumerate(cp['optimizer_parameter_names'])if n.startswith('readout.'))
 if change=='names':cp['optimizer_parameter_names'].reverse()
 elif change=='shape':cp['model_state'][first]=torch.zeros(2)
 elif change=='dtype':cp['model_state'][first]=cp['model_state'][first].double()
 elif change=='nonfinite':cp['model_state'][first].fill_(float('nan'))
 elif change=='optimizer_recipe':cp['optimizer_state']['param_groups'][0]['lr']=.002
 elif change=='head_step':cp['optimizer_state']['state'][head]['step'].sub_(1)
 elif change=='negative_variance':cp['optimizer_state']['state'][head]['exp_avg_sq'].fill_(-1)
 else:cp['rng_state']['torch_cuda'][0]=torch.ones(8,dtype=torch.float32)
 receipt['checkpoint_tree_sha256']=owner.core.state_sha(cp);before=owner.core.state_sha(model.state_dict())
 with pytest.raises(ValueError):owner.strict_restore_v3(model,optimizer,cp,receipt,parent,192,parent['runtime'])
 assert owner.core.state_sha(model.state_dict())==before and not optimizer.state


def test_normalization_buffer_mismatch_is_rejected_before_copy(monkeypatch):
 model,optimizer,parent,cp,receipt=checkpoint_fixture(buffer=True);fake_runtime(monkeypatch,make_rng(0))
 cp['model_state']['normalization_identity'].fill_(2);receipt['checkpoint_tree_sha256']=owner.core.state_sha(cp)
 with pytest.raises(ValueError,match='ACTUAL_NORMALIZATION_INPUT_PRIOR_IDENTITY'):owner.strict_restore_v3(model,optimizer,cp,receipt,parent,192,parent['runtime'])
 assert float(model.normalization_identity)==1.


def test_postcopy_component_mismatch_cannot_pass_cold_restore(monkeypatch):
 model,optimizer,parent,cp,receipt=checkpoint_fixture();fake_runtime(monkeypatch,make_rng(0));load=model.load_state_dict
 def corrupt(state,strict):
  result=load(state,strict=strict)
  with torch.no_grad():next(model.parameters()).add_(1)
  return result
 monkeypatch.setattr(model,'load_state_dict',corrupt)
 with pytest.raises(ValueError,match='RESTORE_FULL_COMPONENTS'):owner.strict_restore_v3(model,optimizer,cp,receipt,parent,192,parent['runtime'])


@pytest.mark.parametrize('key',owner.STATE_KEYS)
def test_any_cold_qualification_component_mismatch_blocks_continuation(key):
 _,_,_,expected,_=checkpoint_fixture();actual={k:copy.deepcopy(expected[k])for k in owner.STATE_KEYS};actual[key]='different'
 with pytest.raises(ValueError,match='COLD_QUALIFICATION_ALL_COMPONENTS'):owner.validate_qualification(actual,expected)


def test_qualification_compares_native_state_without_rewriting_parent_provenance():
 _,_,_,expected,_=checkpoint_fixture();actual={k:copy.deepcopy(expected[k])for k in owner.STATE_KEYS};before=owner.core.state_sha(expected)
 proof=owner.validate_qualification(actual,expected)
 assert proof['bitexact']is True and proof['tolerance']==0. and proof['actual']==proof['expected']
 assert owner.core.state_sha(expected)==before and expected['study_plan']==owner.PARENT


@pytest.mark.parametrize('cursor',[256,320,366])
def test_new_checkpoint_binds_004_execution_and_original_003_scientific_prefix(monkeypatch,tmp_path,cursor):
 model,optimizer,parent,cp,_=checkpoint_fixture();fake_runtime(monkeypatch,cp['rng_state']);writes=[];receipts=[];order=parent['orders']['reordered']
 plan={'runtime':parent['runtime'],'order':order,'order_sha256':owner.core.state_sha(order),'inputs':{'continuation_checkpoint_receipt':{'path':'/003/192.json','sha256':'d'*64}}};binding={'path':'/004/plan.json','sha256':'e'*64}
 steps=[{'logical_step':i+1,'skip_update':False}for i in order[:cursor]];counts={};monkeypatch.setattr(owner,'ROOT',tmp_path)
 def native(path,payload,actual_counts):
  assert actual_counts is counts;writes.append((path.name,payload));return {'path':str(path),'sha256':'f'*64}
 monkeypatch.setattr(owner.core,'future_cuda_publish_checkpoint',native)
 monkeypatch.setattr(owner,'publish',lambda path,value:(receipts.append((path.name,value))or{'path':str(path),'sha256':'a'*64}))
 saved,payload=owner.save_candidate(model,optimizer,plan,binding,steps,counts)
 stem='FINAL'if cursor==366 else f'UPDATE{cursor:06d}'
 assert writes[0][0]==stem+'.pt'and receipts[0][0]==stem+'.json'
 assert payload['execution_plan']==binding and payload['scientific_parent_plan']==owner.PARENT
 assert payload['progress']==owner.study_progress(order,cursor)and payload['continuation_updates_completed']==cursor-192
 assert payload['visited_source_batch_order']==order[:cursor]and payload['steps']==steps
 assert 'trajectory'not in payload and 'study_plan'not in payload and saved['main_trajectory_claim']is False
 assert saved['checkpoint_tree_sha256']==owner.core.state_sha(payload)


def test_unregistered_checkpoint_cannot_be_emitted(monkeypatch):
 model,optimizer,parent,cp,_=checkpoint_fixture();fake_runtime(monkeypatch,cp['rng_state'])
 with pytest.raises(ValueError,match='NEW_CHECKPOINT_ORDINAL'):owner.save_candidate(model,optimizer,{}, {},cp['steps'],{})


def mocked_run(monkeypatch,tmp_path):
 from gx1.contracts import gx1_capped_execution_v1 as guard
 root=tmp_path/'004';binding={'path':'/004/plan.json','sha256':'a'*64};records=[];handlers={}
 monkeypatch.setattr(owner,'ROOT',root);monkeypatch.setattr(owner,'inspect',lambda *args,**kwargs:({},binding,{},{}))
 monkeypatch.setattr(guard,'require_native_future_state_bias_suffix_continuation_execution',lambda **kwargs:{'plan':binding,'output_root':str(root),'outer_seconds':3600})
 monkeypatch.setattr(owner.core,'future_cpu_process_identity',lambda:{'pid':1})
 monkeypatch.setattr(owner,'publish',lambda path,value:(records.append((path.name,value))or{'path':str(path),'sha256':'b'*64}))
 monkeypatch.setattr(owner.signal,'signal',lambda sig,handler:handlers.setdefault(sig,handler));monkeypatch.setattr(owner.signal,'setitimer',lambda *args:None)
 return records,handlers


def test_failed_cold_qualification_consumes_new_plan_without_starting_candidate(monkeypatch,tmp_path):
 records,_=mocked_run(monkeypatch,tmp_path);monkeypatch.setattr(owner.time,'monotonic',lambda:0.)
 def fail(plan,binding,live,progress):progress.update(phase='COLD_REPLAY128_TO192',qualification_updates_completed=1);raise ValueError('COLD_QUALIFICATION_STEP_BITEXACT')
 monkeypatch.setattr(owner,'physical',fail)
 with pytest.raises(ValueError,match='COLD_QUALIFICATION_STEP'):owner.run('a'*64)
 assert [name for name,_ in records]==['CLAIM.json','TERMINAL.json']
 terminal=records[-1][1];assert terminal['status']=='FAILED_CONSUMED'and terminal['relaunch_allowed']is False and terminal['remaining_invocations']==0
 assert terminal['progress']['candidate_started']is False and terminal['retained_checkpoints']==[]
 assert records[0][1]['technical_parameter_fits']==2 and records[0][1]['optimizer_updates']==238


def test_external_stop_reports_retained_new_checkpoint_and_never_relaunches(monkeypatch,tmp_path):
 records,handlers=mocked_run(monkeypatch,tmp_path);kept={'receipt':{'path':'/004/256.json','sha256':'c'*64}}
 monkeypatch.setattr(owner,'retained_receipts',lambda binding:[kept]);monkeypatch.setattr(owner.time,'monotonic',lambda:0.)
 monkeypatch.setattr(owner,'physical',lambda *args:handlers[owner.signal.SIGTERM](owner.signal.SIGTERM,None))
 with pytest.raises(InterruptedError,match='EXTERNAL_SIGNAL'):owner.run('a'*64)
 assert records[-1][1]['status']=='FAILED_CONSUMED'and records[-1][1]['retained_checkpoints']==[kept]


def test_startup_expiry_after_claim_still_publishes_consumed_terminal(monkeypatch,tmp_path):
 records,_=mocked_run(monkeypatch,tmp_path);ticks=iter([0.,4000.]);monkeypatch.setattr(owner.time,'monotonic',lambda:next(ticks))
 monkeypatch.setattr(owner,'physical',lambda *args:pytest.fail('expired startup entered physical execution'))
 with pytest.raises(ValueError,match='WALL_BEFORE_INPUTS'):owner.run('a'*64)
 assert [name for name,_ in records]==['CLAIM.json','TERMINAL.json']and records[-1][1]['remaining_invocations']==0


def test_main_inventory_is_validated_before_source_hash_or_other_artifact_read(monkeypatch):
 main_path=str(owner.train.MAIN_PLAN);monkeypatch.setattr(owner,'ref',lambda path:{'path':str(path),'sha256':'a'*64})
 def read(binding):
  assert binding['path']==main_path
  return {'source_owners':{'/untrusted.bin':'a'*64}}
 monkeypatch.setattr(owner,'read',read);monkeypatch.setattr(owner.train,'expected_source_names',lambda:{'safe.py'})
 monkeypatch.setattr(owner,'sha',lambda *args:pytest.fail('source hash before exact inventory'))
 with pytest.raises(ValueError,match='EXACT_SOURCE_INVENTORY'):owner.build_plan(runtime={})


def test_archive_identity_reads_headers_without_loading_array_payloads(monkeypatch,tmp_path):
 arrays={'feature':np.zeros((3,5),np.float32),'issue_ns':np.arange(3,dtype=np.int64)};path=tmp_path/'headers.npz';np.savez_compressed(path,**arrays)
 expected={k:{'dtype':v.dtype.str,'shape':list(v.shape)}for k,v in arrays.items()}
 monkeypatch.setattr(owner.np,'load',lambda *args,**kwargs:pytest.fail('header-only identity materialized array payload'))
 report=owner.archive_header_identity({'path':str(path),'sha256':'a'*64},expected)
 assert report['payload_bytes_verified_here']is False and set(report['arrays'])==set(arrays)
 assert report['arrays']['feature']=={'dtype':'<f4','shape':[3,5],'fortran_order':False}


@pytest.mark.parametrize('change',['dtype','shape','inventory','object'])
def test_actual_archive_header_mismatch_fails_closed(tmp_path,change):
 a=np.zeros((3,5),np.float32);arrays={'feature':a};expected={'feature':{'dtype':'<f4','shape':[3,5]}}
 if change=='dtype':arrays['feature']=a.astype(np.float64)
 elif change=='shape':arrays['feature']=np.zeros((5,3),np.float32)
 elif change=='inventory':arrays['extra']=np.zeros(1,np.float32)
 else:arrays['feature']=np.full((3,5),'object',object)
 path=tmp_path/'bad.npz';np.savez_compressed(path,**arrays)
 with pytest.raises(ValueError,match='ACTUAL_.*HEADER'):owner.archive_header_identity({'path':str(path),'sha256':'a'*64},expected)


def test_preclaim_input_identity_rejects_norm_provider_clock_mismatch_before_archives(monkeypatch):
 fields={'feature_provider_contract_sha256':'provider','source_axes_sha256':'axes','source_clock_contract':{'clock':'M5'}}
 norm={**fields,'contract_sha256':'norm'};binding={**fields,'contract_sha256':'input','input_normalization_sha256':'norm'}
 binding['source_clock_contract']={'clock':'foreign'}
 meta={'normalization':{'path':'/norm'},'input_binding_ref':{'path':'/binding'},'input_binding':copy.deepcopy(binding),
  'source_snapshot':{'path':'/snapshot'},'dataset':{'path':'/dataset'}}
 values={'/norm':norm,'/binding':binding,'/snapshot':{},'/dataset':{},'/cohort':{}}
 monkeypatch.setattr(owner,'read',lambda ref:values[ref['path']])
 monkeypatch.setattr(owner,'archive_header_identity',lambda *args:pytest.fail('archive read after mismatched normalization identity'))
 pair={'metadata':{'cohort_report':{'cohort_snapshot':{'path':'/cohort'}}}}
 with pytest.raises(ValueError,match='ACTUAL_INPUT_NORM_PROVIDER_CLOCK_IDENTITY'):owner.preclaim_input_identity({'metadata':meta},pair)


def test_success_publishes_generated_report_and_complete_terminal(monkeypatch,tmp_path):
 records,_=mocked_run(monkeypatch,tmp_path);binding={'path':'/004/plan.json','sha256':'a'*64};pref={'path':'/policy','sha256':'c'*64}
 live={'source_commit':'fixedHEAD','wsl_boot_id':'fixedBOOT','physical_windows_boot_id':4};native={'counts':copy.deepcopy(owner.LIMITS),'measured':'generated'}
 monkeypatch.setattr(owner,'inspect',lambda *args,**kwargs:({'source_owners':{}},binding,pref,live))
 monkeypatch.setattr(owner,'physical',lambda *args:native);monkeypatch.setattr(owner,'input_bookend',lambda plan:None)
 monkeypatch.setattr(owner,'ref',lambda path:pref if path==owner.POLICY else binding)
 monkeypatch.setattr(owner.subprocess,'check_output',lambda *args,**kwargs:'fixedHEAD\n')
 class BootPath:
  def read_text(self):return'fixedBOOT\n'
 monkeypatch.setattr(owner,'Path',lambda path:BootPath())
 monkeypatch.setattr(owner.core,'physical_windows_boot_id',lambda:4)
 monkeypatch.setattr(owner.core.common,'firewall_state',lambda:{'denied_attempts':0})
 monkeypatch.setattr(owner.time,'monotonic',lambda:0.)
 result=owner.run('a'*64)
 assert [name for name,_ in records]==['CLAIM.json','RESULT.json','TERMINAL.json']
 assert records[1][1]['report']is native and records[1][1]['status']=='COMPLETE_NATIVE_BIAS_SUFFIX_CONTINUATION'
 terminal=records[-1][1];assert terminal['status']=='COMPLETE'and terminal['result']==result['result']
 assert terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False and terminal['test_data_used']is False


def test_exact_preflight_cli_has_no_claim_output_model_or_checkpoint(monkeypatch,tmp_path,capsys):
 import sys,json
 root=tmp_path/'uncreated004';seen=[];binding={'path':'/004/plan.json','sha256':'a'*64}
 monkeypatch.setattr(owner,'ROOT',root);monkeypatch.setattr(sys,'argv',['module','--plan-sha256','a'*64,'--preflight'])
 def inspect(sha,*,admitted):seen.append((sha,admitted));return {},binding,{},{}
 monkeypatch.setattr(owner,'inspect',inspect)
 monkeypatch.setattr(owner,'run',lambda *args:pytest.fail('preflight started run'))
 monkeypatch.setattr(owner,'publish',lambda *args:pytest.fail('preflight published output'))
 owner.main();report=json.loads(capsys.readouterr().out)
 assert seen==[('a'*64,False)]and not root.exists()
 assert report['models']==report['checkpoint_payloads']==report['optimizer_updates']==0 and report['CUDA_initialized']is False


def test_noncanonical_cli_order_rejected_before_preflight(monkeypatch):
 import sys
 monkeypatch.setattr(sys,'argv',['module','--preflight','--plan-sha256','a'*64])
 monkeypatch.setattr(owner,'inspect',lambda *args,**kwargs:pytest.fail('noncanonical CLI entered inspect'))
 with pytest.raises(ValueError,match='EXACT_CLI'):owner.main()
