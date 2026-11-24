"""Whole-batch order/cursor contracts and real engine dispatch with synthetic state."""
import ast
import copy
import inspect as pyinspect
from pathlib import Path
import json

import numpy as np
import pytest
from gx1.audit import native_provider_hlc_block_order_v1 as m


def canonical():
 return {'contract_sha256':'ffe58ecfb95bb3e5958b31fa017db8277a56ab05c880d37c8ac383b962809631',
  'rows':354550,'batch_rows':96,'logical_batches':3694,'batches_per_pass':3694,'optimizer_updates':3694,'passes':1,
  'skipped_batches':0,'shuffle':False,'drop_last':False,'update_flags':[True]*3694,
  'source_rows_sha256':'9fa3aabd86cc726c0dca3e1ac9b6629bc549f3f2524626d69075fa255f382cae',
  'issue_ns_sha256':'c0805f39affb9bd80e679e0b41f69a558ea975d0e479ddeee815c7dd370997ea',
  'batch_source_rows_sha256':[m.core.state_sha(('source',i))for i in range(3694)],
  'batch_issue_ns_sha256':[m.core.state_sha(('issue',i))for i in range(3694)]}


def recipe():
 return {'passes':1,'train_batch_rows':96,'eval_batch_rows':256,'checkpoint_every_optimizer_updates':256,
  'seed':20260911,'scheduler':None,'gradient_clip_norm':1.}


@pytest.fixture
def schedule(monkeypatch):
 r=recipe();monkeypatch.setattr(m,'read',lambda binding:{'recipe':r}if binding==m.DESIGN else pytest.fail('unexpected metadata read'))
 return m.build_schedule(canonical(),r)


def test_schedule_is_one_local_fixed_permutation_of_whole_blocks_and_B22_last(schedule):
 before=m.core.state_sha(np.random.get_state());base=canonical();original=copy.deepcopy(base)
 out=m.build_schedule(base,recipe())
 assert out==schedule and base==original and m.core.state_sha(np.random.get_state())==before
 expected=np.r_[np.random.Generator(np.random.PCG64(20260911)).permutation(3693),3693].astype(np.int64)
 assert out['original_batch_ids']==expected.tolist()and sorted(out['original_batch_ids'])==list(range(3694))
 assert out['original_batch_ids'][-1]==3693 and out['last_batch_rows']==22
 assert out['batch_source_rows_sha256']==[base['batch_source_rows_sha256'][i]for i in expected]
 assert out['batch_issue_ns_sha256']==[base['batch_issue_ns_sha256'][i]for i in expected]
 assert out['shuffle']is True and out['drop_last']is False
 assert m.validate_schedule(out,recipe())is out


@pytest.mark.parametrize('bad',['seed','passes','batch'])
def test_wrong_seed_or_old_B16_fourpass_cannot_construct_schedule(bad):
 r=recipe();r[{'seed':'seed','passes':'passes','batch':'train_batch_rows'}[bad]]={'seed':7,'passes':4,'batch':16}[bad]
 with pytest.raises(ValueError,match='^PROVIDER_HLC_LEARNING_ALL_TF_EXACT_RECIPE$'):
  m.build_schedule(canonical(),r)


@pytest.mark.parametrize('bad',['duplicate','tail','future_order','source','issue','seal'])
def test_complete_future_schedule_is_bound_even_with_same_consumed_prefix(schedule,bad):
 other=copy.deepcopy(schedule)
 if bad=='duplicate':other['original_batch_ids'][100]=other['original_batch_ids'][101]
 elif bad=='tail':other['original_batch_ids'][-1],other['original_batch_ids'][-2]=other['original_batch_ids'][-2:]
 elif bad=='future_order':other['original_batch_ids'][200:202]=other['original_batch_ids'][200:202][::-1]
 elif bad=='source':other['batch_source_rows_sha256'][200]='a'*64
 elif bad=='issue':other['batch_issue_ns_sha256'][200]='a'*64
 else:other['contract_sha256']='0'*64
 assert other['original_batch_ids'][:20]==schedule['original_batch_ids'][:20]
 with pytest.raises(ValueError,match='^PROVIDER_HLC_BLOCK_ORDER_EXACT_SEALED_SCHEDULE$'):
  m.cursor_binding(other,20)


@pytest.mark.parametrize('cursor',[0,1,256,3584,3693,3694])
def test_cursor_is_execution_count_and_EOF_never_restarts_first_batch(schedule,cursor):
 witness=m.cursor_binding(schedule,cursor)
 assert witness['position']==m.train.position(schedule,cursor)
 if cursor==3694:
  assert witness['phase']=='FINAL_SCORE'and witness['next_original_batch_id']is None
  assert witness['next_source_rows_sha256']is witness['next_issue_ns_sha256']is None
 else:
  assert witness['phase']=='TRAIN'and witness['next_original_batch_id']==schedule['original_batch_ids'][cursor]
  assert witness['next_source_rows_sha256']==schedule['batch_source_rows_sha256'][cursor]
 if cursor==3693:assert witness['next_original_batch_id']==3693


@pytest.mark.parametrize('cursor',[-1,3695,True,1.5])
def test_cursor_rejects_out_of_scope_or_noninteger(schedule,cursor):
 with pytest.raises(ValueError,match='^NATIVE_FUTURE_TRAIN_LOGICAL_CURSOR$'):m.cursor_binding(schedule,cursor)


def journal(schedule,cursor):
 return [{'logical_step':j+1,'skip_update':False,'order_binding':m.expected_step_binding(schedule,j)}for j in range(cursor)]


def test_whole_consumed_prefix_has_logical_indices_and_original_batch_ids(schedule):
 steps=journal(schedule,512);assert steps[0]['order_binding']['original_batch_id']!=0
 witness=m.validate_prefix(schedule,steps,512)
 assert witness==m.cursor_binding(schedule,512)


@pytest.mark.parametrize('bad',['length','logical_ID','duplicate','source','issue','skipped'])
def test_prefix_forgery_reaches_exact_gate_before_any_restore(schedule,bad):
 steps=journal(schedule,256);gate='PREFIX_EXECUTION_BINDING'
 if bad=='length':steps.pop();gate='PREFIX_LENGTH'
 elif bad=='logical_ID':steps[2]['logical_step']=schedule['original_batch_ids'][2]+1
 elif bad=='duplicate':steps[2]['order_binding']=copy.deepcopy(steps[1]['order_binding'])
 elif bad in('source','issue'):steps[2]['order_binding'][bad+('_rows_sha256'if bad=='source'else'_ns_sha256')]='0'*64
 else:steps[2]['skip_update']=True
 with pytest.raises(ValueError,match='^PROVIDER_HLC_BLOCK_ORDER_'+gate+'$'):m.validate_prefix(schedule,steps,256)


def context_schedule(monkeypatch):
 source=np.arange(354550,dtype=np.int64)+100;issues=np.arange(354550,dtype=np.int64)*300000000000
 base=canonical();base['source_rows_sha256']=m.core.digest(source);base['issue_ns_sha256']=m.core.digest(issues)
 batches=m.core.all_score_batches(np.arange(354550,dtype=np.int64),96)
 base['batch_source_rows_sha256']=[m.core.digest(source[t])for t in batches]
 base['batch_issue_ns_sha256']=[m.core.digest(issues[t])for t in batches]
 # Synthetic physical axes have separate hashes. Keep the real geometry/recipe gate.
 original=m.learning.all_tf_validate_clock
 def validate(r,s):
  proof=copy.deepcopy(s);proof['source_rows_sha256']=canonical()['source_rows_sha256'];proof['issue_ns_sha256']=canonical()['issue_ns_sha256']
  original(r,proof)
 monkeypatch.setattr(m.learning,'all_tf_validate_clock',validate)
 monkeypatch.setattr(m,'read',lambda binding:{'recipe':recipe()}if binding==m.DESIGN else pytest.fail('unexpected input'))
 return {'source_rows':source,'truth':{'issue_ns':issues}},m.build_schedule(base,recipe())


def test_resolved_batches_preserve_all354550_rows_and_each_window_interior(monkeypatch):
 context,s=context_schedule(monkeypatch);batches=m.resolve_batches(context,s)
 assert len(batches)==3694 and sum(map(len,batches))==354550 and len(batches[-1])==22
 assert np.array_equal(np.sort(np.concatenate(batches)),np.arange(354550))
 for j,take in enumerate(batches):
  original=s['original_batch_ids'][j]
  assert np.array_equal(take,np.arange(original*96,min(original*96+96,354550),dtype=np.int64))
  packet={'truth':{'issue_ns':context['truth']['issue_ns'][take].copy()}}
  assert m.step_binding(context,s,j,take,packet)==m.expected_step_binding(s,j)


@pytest.mark.parametrize('bad',['interior','packet','source'])
def test_actual_packet_cannot_use_a_different_batch_clock_or_row_order(monkeypatch,bad):
 context,s=context_schedule(monkeypatch);take=m.resolve_batches(context,s)[0].copy()
 packet={'truth':{'issue_ns':context['truth']['issue_ns'][take].copy()}}
 gate='ACTUAL_PACKET_CLOCK'
 if bad=='interior':take=take[::-1];gate='ACTUAL_ORIGINAL_BATCH_ROWS'
 elif bad=='packet':packet['truth']['issue_ns'][0]+=1
 else:context['source_rows'][take[0]]+=1
 with pytest.raises(ValueError,match='^PROVIDER_HLC_BLOCK_ORDER_'+gate+'$'):m.step_binding(context,s,0,take,packet)


def test_engine_seams_preserve_entire_old_owner_and_numeric_primitives():
 assert m.engine_source_bridge()['all_other_functions_and_normalized_default_branches_byte_identical']is True
 for name in ('training_step','position','make_checkpoint','checkpoint_proof','strict_restore'):
  assert getattr(m.train,name)is getattr(m.learning.train,name)
 assert m.guard_source_bridge()['all_other_guard_shell_bytes_and_execution_ENV_proof_unchanged']is True


def physical_fixture(monkeypatch,start):
 from tests.test_native_provider_hlc_all_tf_train_end_recovery_v2 import _physical
 import torch
 from gx1.contracts import market_state_ta_integrated_model_v1 as model_owner
 oldplan,binding,f,events,saved,disk=_physical(monkeypatch,start)
 original=oldplan['metadata']['original'];base=original['schedule'];base['contract_sha256']=canonical()['contract_sha256']
 base['source_rows_sha256']=m.core.digest(np.arange(354550,dtype=np.int64));base['issue_ns_sha256']=base['source_rows_sha256']
 # Existing fixture covers numerical-free checkpoint/write/scoring lifecycle. Add real order adapter.
 validate=m.learning.all_tf_validate_clock
 def clock(r,s):
  proof=copy.deepcopy(s);proof['source_rows_sha256']=canonical()['source_rows_sha256'];proof['issue_ns_sha256']=canonical()['issue_ns_sha256'];validate(r,proof)
 monkeypatch.setattr(m.learning,'all_tf_validate_clock',clock)
 monkeypatch.setattr(m,'read',lambda value:{'recipe':original['recipe']}if value==m.DESIGN else pytest.fail('foreign read'))
 original['schedule']=m.build_schedule(base,original['recipe']);original['schema_version']=m.MAIN_SCHEMA;original['stage']=m.STAGE;original['output_root']=str(m.ROOT)
 original['train_end_checkpoint_before_score']=True;original['initial_reference']['model_contract_sha256']=original['model_contract']['contract_sha256']
 oldplan.update(schema_version=m.RESUME_SCHEMA,stage=m.RECOVERY_STAGE,output_root=str(m.ROOT),schedule=original['schedule'])
 binding={'path':str(m.ROOT/'RECOVERY_PLANS'/('b'*64+'.json')),'sha256':'b'*64}
 oldplan['metadata']['trajectory']=m.learning.trajectory(original)
 oldplan['metadata']['original_plan']={'path':str(m.PLAN),'sha256':'a'*64}
 model=model_owner.build_native_future_state_model();model.named_buffers=lambda:[]
 context=m.learning.adapter.load_context(original['metadata'])
 monkeypatch.setattr(m.learning.adapter,'packet',lambda actual,take:{'truth':{'rows':len(take),'issue_ns':actual['truth']['issue_ns'][take]}})
 step=m.train.training_step
 def update(model,optimizer,packet,plan,counts,index):
  result=step(model,optimizer,packet,plan,counts,index)
  return {**result,'logical_step':index+1,'skip_update':False}
 monkeypatch.setattr(m.train,'training_step',update)
 loader=torch.load
 def load(*a,**kw):
  cp=loader(*a,**kw);cp['trajectory']=m.learning.trajectory(original);cp['steps']=journal(original['schedule'],start)
  cp['producer']['original_plan']=oldplan['metadata']['original_plan']
  cp['position']=m.train.position(original['schedule'],start)
  oldplan['metadata']['trajectory']=cp['trajectory'];oldplan['metadata']['receipt']['proof']=m.train.checkpoint_proof(cp)
  return cp
 monkeypatch.setattr(torch,'load',load)
 return oldplan,binding,f,events,saved,disk,context


@pytest.mark.parametrize('start',[3584,3694])
def test_order_resume_reuses_real_engine_and_TRAIN_END_precedes_canonical_score(monkeypatch,start):
 plan,binding,f,events,saved,disk,context=physical_fixture(monkeypatch,start)
 report=m.learning.physical_all_tf_resume(plan,binding,{},f.runtime,order_scope={'module':m.MODULE,'mode':'resume','plan':binding})
 assert report['counts']==plan['limits']and report['new_optimizer_updates']==3694-start
 assert report['order_cursor']['phase']=='FINAL_SCORE'and report['actual_order_journal_verified']is True
 assert report['whole_FINAL_score']['rows']==354550 and report['whole_FINAL_score']['batch_count']==1385
 assert list(saved)==(['TRAIN_END.pt','FINAL.pt']if start<3694 else['FINAL.pt'])
 if start<3694:
  assert events.index('TRAIN_END.pt')<events.index('first_score')
  assert m.validate_prefix(plan['schedule'],saved['TRAIN_END.pt']['steps'],3694)['phase']=='FINAL_SCORE'
 else:assert 'TRAIN_END.pt'not in saved and events.count('update')==0
 assert context['source_rows'][0]==0 and context['source_rows'][-1]==354549


def test_newmain_uses_same_engine_writes17CP_and_TRAIN_END_before_score(monkeypatch):
 plan,binding,f,events,saved,disk,context=physical_fixture(monkeypatch,3584)
 original=plan['metadata']['original'];original['limits']=copy.deepcopy(plan['limits'])
 original['limits'].update(model_forwards=5079,backbone_forwards=5079,readout_forwards=5079,optimizer_steps=3694,backward_calls=3694,
  checkpoint_writes=17,checkpoint_readbacks=17,checkpoint_receipt_writes=17,checkpoint_loads=0,model_restores=0)
 binding={'path':str(m.PLAN),'sha256':'b'*64}
 report=m.learning.physical_all_tf(original,binding,{},f.runtime,order_scope={'module':m.MODULE,'mode':'main','plan':binding})
 assert report['counts']==original['limits']and len(saved)==17
 assert events.index('TRAIN_END.pt')<events.index('first_score')
 assert m.validate_prefix(original['schedule'],saved['TRAIN_END.pt']['steps'],3694)['phase']=='FINAL_SCORE'
 assert saved['TRAIN_END.pt']['scores']==f.scores and 'PASS1'in saved['FINAL.pt']['scores']
 assert report['order_cursor']['next_original_batch_id']is None


def recovery_fixture(monkeypatch,cursor=3694):
 from tests.test_native_provider_hlc_all_tf_train_end_recovery_v2 import _authenticated
 f=_authenticated(monkeypatch,cursor);original=f.original;canonical_plan=copy.deepcopy(original)
 canonical_plan['schedule']['batch_source_rows_sha256']=canonical()['batch_source_rows_sha256']
 canonical_plan['schedule']['batch_issue_ns_sha256']=canonical()['batch_issue_ns_sha256']
 original.update(schema_version=m.MAIN_SCHEMA,stage=m.STAGE,output_root=str(m.ROOT),train_end_checkpoint_before_score=True)
 original['inputs'].update(chronological_None_plan=m.CHRONOLOGY,block_order_design=m.DESIGN)
 original['schedule']=m.build_schedule(canonical_plan['schedule'],original['recipe'])
 original_ref={'path':str(m.PLAN),'sha256':'a'*64};producer_ref={'path':str(m.ROOT/'RECOVERY_PLANS'/('a'*64+'.json')),'sha256':'a'*64}
 root=m.recovery_root('a'*64);old_claim=f.values[f.scope['producer_claim']['path']]
 scope=f.scope;scope.update(schema_version=m.SCOPE_SCHEMA,original_plan=original_ref,producer_plan=producer_ref,
  producer_claim={'path':str(root/'CLAIM.json'),'sha256':'a'*64},producer_terminal={'path':str(root/'TERMINAL.json'),'sha256':'a'*64},
  checkpoint_receipt={'path':str(root/('TRAIN_END.json'if cursor==3694 else'UPDATE%06d.json'%cursor)),'sha256':'a'*64},
  checkpoint={'path':str(root/('TRAIN_END.pt'if cursor==3694 else'UPDATE%06d.pt'%cursor)),'sha256':'a'*64})
 trajectory=m.learning.trajectory(original)
 f.producer.update(schema_version=m.RESUME_SCHEMA,stage=m.RECOVERY_STAGE,output_root=str(m.ROOT),schedule=original['schedule'])
 f.producer['metadata'].update(original=original,original_plan=original_ref,trajectory=trajectory)
 f.receipt.update(checkpoint=scope['checkpoint'],original_plan=original_ref,execution_plan=producer_ref,trajectory_sha256=trajectory['contract_sha256'])
 f.receipt['proof'].update(trajectory=m.core.state_sha(trajectory),position=m.core.state_sha(m.train.position(original['schedule'],cursor)))
 old_claim['plan']=producer_ref;f.terminal.update(plan=producer_ref,claim=scope['producer_claim'])
 f.review.update(execution_plan=producer_ref,original_plan=original_ref,last_durable_checkpoint=scope['checkpoint'],last_durable_receipt=scope['checkpoint_receipt'])
 f.backup.update(checkpoint=scope['checkpoint'],checkpoint_receipt=scope['checkpoint_receipt'])
 values={m.CHRONOLOGY['path']:canonical_plan,m.DESIGN['path']:{'recipe':original['recipe']},original_ref['path']:original,
  producer_ref['path']:f.producer,scope['checkpoint_receipt']['path']:f.receipt,scope['producer_claim']['path']:old_claim,
  scope['producer_terminal']['path']:f.terminal,scope['stop_review']['path']:f.review,scope['backup']['path']:f.backup,
  scope['producer_admitted_policy']['path']:{m.RECOVERY_STAGE:m.gate(producer_ref,True,f.producer['starting_cursor'])}}
 previous=m.learning.read
 monkeypatch.setattr(m,'read',lambda value:values[value['path']]if value['path']in values else previous(value))
 return f


@pytest.mark.parametrize('cursor',[3584,3694])
def test_recovery_metadata_accepts_only_the_new_order_trajectory_and_phase(monkeypatch,cursor):
 f=recovery_fixture(monkeypatch,cursor)
 receipt,trajectory,producer=m.validate_recovery(f.original,f.scope)
 assert receipt is f.receipt and trajectory==m.learning.trajectory(f.original)and producer is f.producer
 limits,cursors=m.learning.all_tf_resume_limits(f.original,cursor,phase_recovery=True)
 assert limits['optimizer_steps']==3694-cursor and limits['model_forwards']==3694-cursor+1385
 assert limits['checkpoint_writes']==(1 if cursor==3694 else 2)


@pytest.mark.parametrize('bad,gate',[('FINAL','RECOVERY_REFERENCE_PATH'),('scores','ALL_TF_RECOVERY_AUTHENTIC_STATE_PROOFS'),
 ('wrong_producer','ALL_TF_RECOVERY_AUTHENTIC_TRAIN_END_PRODUCER'),('old_scope','ALL_TF_RECOVERY_SCOPE_SCHEMA'),
 ('cursor','ALL_TF_RECOVERY_EXACT_DURABLE_CURSOR'),('stopped','ALL_TF_RECOVERY_FAILED_CONSUMED_PRODUCER')])
def test_order_recovery_forgery_hits_its_intended_gate(monkeypatch,bad,gate):
 f=recovery_fixture(monkeypatch)
 if bad=='FINAL':
  f.scope['checkpoint']['path']=f.scope['checkpoint']['path'].replace('TRAIN_END.pt','FINAL.pt')
 elif bad=='scores':f.receipt['proof']['scores']=m.core.state_sha({'PASS1':{}})
 elif bad=='wrong_producer':f.producer['train_end_checkpoint_before_score']=False
 elif bad=='old_scope':f.scope['schema_version']='gx1_native_provider_hlc_all_tf_recovery_scope_v2'
 elif bad=='cursor':f.receipt['logical_cursor']=True
 else:f.terminal['status']='COMPLETE'
 prefix='NATIVE_FUTURE_TRAIN_'if bad=='FINAL'else'PROVIDER_HLC_BLOCK_ORDER_'
 with pytest.raises(ValueError,match='^'+prefix+gate+'$'):m.validate_recovery(f.original,f.scope)


@pytest.mark.parametrize('mutation',[None,'before','after'])
def test_historical_guard_test_bridge_is_exact_test_only_bytes(mutation):
 before=m.subprocess.check_output(['git','show',m.ANCHOR+':'+m.HISTORICAL_GUARD_TEST],cwd=m.REPO)
 after=(m.REPO/m.HISTORICAL_GUARD_TEST).read_bytes()
 if mutation=='before':before+=b'\n# forbidden\n'
 if mutation=='after':after+=b'\n# forbidden\n'
 if mutation is None:
  proof=m.historical_guard_test_bridge(before,after)
  assert proof['all_other_test_bytes_unchanged']and proof['test_only_historical_fixture_and_two_signatures']
 else:
  with pytest.raises(ValueError,match='^PROVIDER_HLC_BLOCK_ORDER_HISTORICAL_GUARD_TEST_EXACT_SHA$'):
   m.historical_guard_test_bridge(before,after)
