"""Scoped metadata/cache/recovery tests; numerical primitives are reused unchanged."""
import copy
from pathlib import Path

import numpy as np
import pytest

from gx1.audit import native_provider_hlc_row_membership_fit_v1 as m


def reference(path):
 return {'path':str(path),'sha256':'a'*64}


def test_single_order_and_native_activity_are_cached_for_all_packets(monkeypatch):
 schedule={'contract_sha256':'schedule','descriptor':{'order':reference('/synthetic/order.npy')}}
 activity=np.ones((354550,5),np.bool_);context={'source_rows':np.arange(354550,dtype=np.int64),'truth':{},'target_contract':{}}
 counts={'order_array_loads':0};events=[];takes=[np.arange(96,dtype=np.int64)]
 monkeypatch.setattr(m,'validate_schedule',lambda *a:None)
 monkeypatch.setattr(m.learning,'resume_reference',lambda r:events.append(('ref',r)))
 monkeypatch.setattr(m.np,'load',lambda p,**kw:events.append(('load',p,kw))or np.arange(354550,dtype=np.int64))
 from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
 monkeypatch.setattr(objective,'_native_future_state_truth',lambda *a,**kw:events.append(('native_mask',))or{'activity':activity})
 monkeypatch.setattr(m.schedule_owner,'resolve_batches',lambda c,s,o,a:events.append(('resolve',a))or takes)
 assert m.resolve_batches(context,schedule,counts)is takes
 assert counts['order_array_loads']==1 and not activity.flags.writeable
 assert sum(e[0]=='load'for e in events)==sum(e[0]=='native_mask'for e in events)==1
 monkeypatch.setattr(m.schedule_owner,'step_binding',lambda c,s,i,t,p,a:events.append(('step_mask',a))or{'index':i})
 assert m.step_binding(context,schedule,0,takes[0],{})=={'index':0}
 assert events[-1][1]is activity
 with pytest.raises(ValueError,match='ONE_ORDER_LOAD_AND_ACTIVITY_COMPUTATION$'):
  m.resolve_batches(context,schedule,counts)
 assert sum(e[0]=='load'for e in events)==1


def test_packet_cannot_use_missing_foreign_or_mutable_activity_cache(monkeypatch):
 activity=np.ones((354550,5),np.bool_);schedule={'contract_sha256':'right'}
 for cache in(None,{'schedule_sha256':'wrong','activity':activity},{'schedule_sha256':'right','activity':activity}):
  context={}if cache is None else{m._CACHE:cache}
  with pytest.raises(ValueError,match='BOUND_READONLY_ACTIVITY_CACHE$'):
   m.step_binding(context,schedule,0,np.arange(96),{})
 activity.flags.writeable=False
 monkeypatch.setattr(m.schedule_owner,'step_binding',lambda *a:pytest.fail('bad cache reached delegate'))
 with pytest.raises(ValueError,match='BOUND_READONLY_ACTIVITY_CACHE$'):
  m.step_binding({m._CACHE:{'schedule_sha256':'wrong','activity':activity}},schedule,0,np.arange(96),{})


@pytest.mark.parametrize('mode',['main','resume'])
def test_engine_identity_is_new_exact_namespace_without_old_route_impersonation(monkeypatch,mode):
 sha='a'*64;binding=reference(m.PLAN if mode=='main'else m.ROOT/'RECOVERY_PLANS'/(sha+'.json'))
 plan={'schema_version':m.MAIN_SCHEMA if mode=='main'else m.RESUME_SCHEMA,'stage':m.STAGE if mode=='main'else m.RECOVERY_STAGE,
  'output_root':str(m.ROOT),'backbone_gradient_policy':None,'execution':m.learning.EXECUTION,'schedule':{'descriptor':{'order':reference('/synthetic/order.npy')}},
  'recipe':{},'inputs':{'clock_schedule':m.CLOCK,'membership_manifest':m.MANIFEST,'source_row_order':reference('/synthetic/order.npy')},
  'train_end_checkpoint_before_score':True}
 monkeypatch.setattr(m,'validate_schedule',lambda *a:None)
 scope={'module':m.MODULE,'mode':mode,'plan':binding}
 assert m.require_engine_scope(scope,plan,binding,mode)==(m.ROOT if mode=='main'else m.recovery_root(sha))
 for bad in('gx1.audit.native_provider_hlc_block_order_v1','gx1.audit.native_provider_hlc_row_membership_qualification_v1'):
  with pytest.raises(ValueError,match='EXACT_ENGINE_SCOPE$'):
   m.require_engine_scope({**scope,'module':bad},plan,binding,mode)
 bad=copy.deepcopy(plan);bad['schema_version']='gx1_native_provider_hlc_block_order_one_epoch_plan_v1'
 with pytest.raises(ValueError,match='ENGINE_PLAN_IDENTITY$'):
  m.require_engine_scope(scope,bad,binding,mode)


def recovery_fixture(monkeypatch,cursor=3694):
 from tests.test_native_provider_hlc_all_tf_train_end_recovery_v2 import _authenticated
 f=_authenticated(monkeypatch,cursor);original=f.original;claim=f.values[f.scope['producer_claim']['path']]
 original.update(schema_version=m.MAIN_SCHEMA,stage=m.STAGE,output_root=str(m.ROOT),train_end_checkpoint_before_score=True)
 original['limits']['order_array_loads']=1
 original['context_IO']={'archive_loads':2,'decoded_fields':41,'decoded_bytes':1483917408,
  'archives':{'feature_archive':{'binding':reference('/synthetic/features.npz')},'target_arrays':{'binding':reference('/synthetic/targets.npz')}}}
 original['inputs'].update(canonical_fullfit_plan=m.CANONICAL,clock_schedule=m.CLOCK,membership_manifest=m.MANIFEST,
  technical_acceptance=m.TECHNICAL_ACCEPTANCE,row_membership_fit_design=m.DESIGN,row_membership_review_protocol=m.REVIEW_PROTOCOL)
 original_ref=reference(m.PLAN);producer_ref=reference(m.ROOT/'RECOVERY_PLANS'/('a'*64+'.json'));root=m.recovery_root('a'*64)
 name='TRAIN_END'if cursor==3694 else'UPDATE%06d'%cursor
 f.scope.update(schema_version=m.SCOPE_SCHEMA,original_plan=original_ref,producer_plan=producer_ref,
  producer_claim=reference(root/'CLAIM.json'),producer_terminal=reference(root/'TERMINAL.json'),
  checkpoint_receipt=reference(root/(name+'.json')),checkpoint=reference(root/(name+'.pt')))
 trajectory=m.learning.trajectory(original)
 f.producer.update(schema_version=m.RESUME_SCHEMA,stage=m.RECOVERY_STAGE,output_root=str(m.ROOT),schedule=original['schedule'])
 f.producer['metadata'].update(original=original,original_plan=original_ref,trajectory=trajectory)
 f.receipt.update(checkpoint=f.scope['checkpoint'],original_plan=original_ref,execution_plan=producer_ref,
  trajectory_sha256=trajectory['contract_sha256'])
 f.receipt['proof'].update(trajectory=m.core.state_sha(trajectory),position=m.core.state_sha(m.train.position(original['schedule'],cursor)))
 claim['plan']=producer_ref;f.terminal.update(plan=producer_ref,claim=f.scope['producer_claim'])
 f.review.update(execution_plan=producer_ref,original_plan=original_ref,last_durable_checkpoint=f.scope['checkpoint'],last_durable_receipt=f.scope['checkpoint_receipt'])
 f.backup.update(checkpoint=f.scope['checkpoint'],checkpoint_receipt=f.scope['checkpoint_receipt'])
 values={m.CANONICAL['path']:{'recipe':original['recipe']},m.CLOCK['path']:original['schedule'],
  original['inputs']['baseline_initial_receipt']['path']:original['initial_reference'],
  original_ref['path']:original,producer_ref['path']:f.producer,f.scope['checkpoint_receipt']['path']:f.receipt,
  f.scope['producer_claim']['path']:claim,f.scope['producer_terminal']['path']:f.terminal,
  f.scope['stop_review']['path']:f.review,f.scope['backup']['path']:f.backup,
  f.scope['producer_admitted_policy']['path']:{m.RECOVERY_STAGE:m.gate(producer_ref,True,f.producer['starting_cursor'])},
  f.scope_ref['path']:f.scope}
 previous=m.learning.read
 monkeypatch.setattr(m,'read',lambda r:values[r['path']]if r['path']in values else previous(r))
 # The pure schedule has its own tamper/clock/support/EOF suite; this fixture
 # exercises the independent producer/receipt/policy/stop/backup authority.
 monkeypatch.setattr(m,'validate_schedule',lambda *a:None)
 monkeypatch.setattr(m,'ref',lambda path:{'path':str(path),'sha256':original['source_owners'].get(Path(path).name,'a'*64)})
 return f,values


@pytest.mark.parametrize('cursor',[256,3584,3694])
def test_own_checkpoint_lineage_and_suffix_budget_including_score_only(monkeypatch,cursor):
 f,values=recovery_fixture(monkeypatch,cursor)
 if cursor==256:
  f.producer['starting_cursor']=0
  values[f.scope['producer_admitted_policy']['path']][m.RECOVERY_STAGE]=m.gate(f.scope['producer_plan'],True,0)
 receipt,traj,producer=m.validate_recovery(f.original,f.scope)
 assert receipt is f.receipt and traj==m.learning.trajectory(f.original)and producer is f.producer
 plan=m.build_resume_plan(f.runtime,f.scope_ref)
 assert plan['schedule']==f.original['schedule']and plan['metadata']['trajectory']==traj
 assert plan['context_IO']==f.original['context_IO']and plan['context_IO']is not f.original['context_IO']
 assert plan['limits']['order_array_loads']==1 and plan['new_INITIAL_writes']==0
 assert plan['limits']['optimizer_steps']==3694-cursor and plan['limits']['model_forwards']==3694-cursor+1385
 assert plan['remaining_TRAIN_rows']==max(0,354550-cursor*96)and plan['remaining_FINAL_rows']==354550
 assert plan['score_only']is(cursor==3694)
 if cursor==3694:
  assert plan['limits']['backward_calls']==0 and plan['limits']['checkpoint_writes']==1


@pytest.mark.parametrize('bad,gate',[
 ('technical_root','RECOVERY_PREVIOUS_SUFFIX_LINEAGE'),('technical_schema','RECOVERY_PREVIOUS_SUFFIX_LINEAGE'),
 ('scope','RECOVERY_SCOPE_SCHEMA'),('cursor_bool','RECOVERY_EXACT_DURABLE_CURSOR'),
 ('scores','RECOVERY_AUTHENTIC_STATE_PROOFS'),('training','RECOVERY_AUTHENTIC_STATE_PROOFS'),
 ('producer_finished','RECOVERY_FAILED_CONSUMED_PRODUCER'),('backup','RECOVERY_DURABLE_BACKUP'),
 ('stop_review','RECOVERY_STOP_REVIEW'),('no_train_end','RECOVERY_AUTHENTIC_TRAIN_END_PRODUCER')])
def test_recovery_forgery_reaches_its_own_gate_after_baseline_accepts(monkeypatch,bad,gate):
 f,values=recovery_fixture(monkeypatch)
 m.validate_recovery(f.original,f.scope)
 if bad=='technical_root':f.producer['output_root']='/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_20261125_001_warm'
 elif bad=='technical_schema':f.producer['schema_version']='gx1_native_provider_hlc_row_membership_qualification_plan_v1'
 elif bad=='scope':f.scope['schema_version']='gx1_native_provider_hlc_block_order_recovery_scope_v1'
 elif bad=='cursor_bool':f.receipt['logical_cursor']=True
 elif bad=='scores':f.receipt['proof']['scores']=m.core.state_sha({'PASS1':{}})
 elif bad=='training':f.receipt['proof']['training']=m.core.state_sha(False)
 elif bad=='producer_finished':f.terminal['status']='COMPLETE'
 elif bad=='backup':f.backup['Mac_F_FULLFSYNC_and_disk_SHA_readback']=False
 elif bad=='stop_review':f.review['last_durable_cursor']=3584
 else:f.producer['train_end_checkpoint_before_score']=False
 with pytest.raises(ValueError,match='^PROVIDER_HLC_ROW_MEMBERSHIP_FIT_'+gate+'$'):
  m.validate_recovery(f.original,f.scope)


def review_fixture(monkeypatch):
 prior=reference('/synthetic/prior.json');initial_result=reference('/synthetic/initial_result.json');old_result=reference('/synthetic/current_result.json');quality_ref=reference('/synthetic/quality.json')
 original={'model_contract':{'contract_sha256':'contract'},'inputs':{'baseline_initial_receipt':reference('/synthetic/INITIAL.json'),
  'baseline_result':initial_result,'chronological_None_result':reference('/synthetic/chrono_result.json'),'chronological_None_quality':reference('/synthetic/chrono_quality.json')},
  'metadata':{'dataset':reference('/synthetic/dataset.json'),'prior':prior,'normalization':reference('/synthetic/norm.json'),
   'input_binding_ref':reference('/synthetic/binding.json'),'truth_sha256':'truth'}}
 meta=original['metadata'];identities={**{k:meta[k]for k in('dataset','prior','normalization')},
  'input_binding':meta['input_binding_ref'],'truth_sha256':'truth','initial_receipt':original['inputs']['baseline_initial_receipt'],
  'model_contract_sha256':'contract','original_plan':m.CANONICAL}
 groups=[tf+'|'+str(y)for tf in m.train.TF for y in range(2021,2026)]
 groups+=['D1|'+s+'|'+str(y)for s in('v1_known','newly_known')for y in('pooled',2021,2022,2023,2024,2025)]
 protocol={'schema_version':'gx1_source_row_membership_comparison_protocol_v1','name':'ROW_MEMBERSHIP_ONE_FIXED_COMPARISON_BEFORE_FIT',
  'primary_witnesses':30,'secondary_groups':37,'primary_references':{'own_INITIAL':initial_result,'prior':prior,'current_blockorder':old_result},
  'required_binding_identities':identities,'secondary_group_ids':groups,'current_blockorder_quality':quality_ref,
  'evaluation':{'rows':354550,'timeframes':list(m.train.TF),'batch_rows':256,'batches':1385,'last_batch_rows':246,
   'UTC_target_start_years':[2021,2022,2023,2024,2025],'D1_support_partitions':['v1_known','newly_known'],
   'allselectedrows_and_masks_retained':True,'unknown_side_remains_unknown':True},
  'extra_secondary_context':{'chronological_None':{'result':original['inputs']['chronological_None_result'],'quality':original['inputs']['chronological_None_quality']}},
  **{k:False for k in('execution_authorized','training_authorized','test_authorized','step15_auto_approved','all20_approved','edge_proved')}}
 quality={'rows':354550,'truth_sha256':'truth','output_sha256':{'side_logits':'output'}}
 old={'status':'COMPLETE_NONE_BLOCK_ORDER_TRAIN_COMPARISON','source_inputs_policy_boot_unchanged':True,'test_data_used':False,
  'report':{'original_plan':m.CANONICAL,'actual_optimizer_updates':3694,'final_quality':quality_ref,'whole_FINAL_score':{'output_sha256':quality['output_sha256']}}}
 values={old_result['path']:old,quality_ref['path']:quality};monkeypatch.setattr(m,'read',lambda r:values[r['path']])
 return protocol,original,{'plan':m.learning.ALL_TF_BASE}


@pytest.mark.parametrize('bad', ['count','group','prior','automatic_step15'])
def test_review_cannot_expand_refs_select_groups_or_approve_step15(monkeypatch,bad):
 protocol,original,initial=review_fixture(monkeypatch)
 assert m._review(protocol,original,initial)is protocol
 if bad=='count':protocol['primary_witnesses']=40
 elif bad=='group':protocol['secondary_group_ids']=protocol['secondary_group_ids'][:-1]
 elif bad=='prior':protocol['primary_references']['prior']=reference('/synthetic/other_prior.json')
 else:protocol['step15_auto_approved']=True
 with pytest.raises(ValueError,match='PROVIDER_HLC_ROW_MEMBERSHIP_FIT_'):
  m._review(protocol,original,initial)


def test_engine_bridge_only_normalizes_four_named_deltas(monkeypatch,tmp_path):
 main=" if order_scope is not None:batches=order.resolve_batches(context,plan['schedule'])\n"
 resume=" if order_scope is not None:batches=order.resolve_batches(context,original['schedule'])\n"
 old='header\n'+main+'middle\n'+resume+'footer\n'
 def block(tag):
  return '# BEGIN GX1 ROW MEMBERSHIP ENGINE 20261125 001 '+tag+'\nnew_route\n# END GX1 ROW MEMBERSHIP ENGINE 20261125 001 '+tag+'\n'
 current=block('MAIN_MODULE')+block('RESUME_MODULE')+old.replace(main,block('MAIN_BATCHES')).replace(resume,block('RESUME_BATCHES'))
 path=tmp_path/m.ENGINE;path.parent.mkdir(parents=True);path.write_text(current)
 monkeypatch.setattr(m,'REPO',tmp_path);monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**kw:old.encode())
 proof=m.engine_source_bridge();assert proof['original_sha256']==proof['normalized_sha256']and proof['all_other_bytes_unchanged']is True
 path.write_text(current+'undeclared_numeric_change\n')
 with pytest.raises(ValueError,match='ENGINE_ALL_OLD_BYTES_AND_DEFAULTS$'):
  m.engine_source_bridge()


@pytest.mark.parametrize('mode',['main','resume'])
def test_oneuse_gates_keep_future_epochs_and_test_closed(mode):
 binding=reference(m.PLAN);gate=m.gate(binding,True,None if mode=='main'else 3694)
 assert gate['remaining_invocations']==1 and gate['relaunch_allowed']is False and gate['test_authorized']is False
 assert gate['additional_epochs_allowed']is False
 assert 'ROW_MEMBERSHIP'in gate['status']


@pytest.mark.parametrize('flag',['--row-membership-plan-sha256','--row-membership-resume-plan-sha256'])
def test_exact_preflight_cannot_call_physical_or_create_claim(monkeypatch,flag,capsys):
 from gx1.contracts import gx1_capped_execution_v1 as guard
 sha='a'*64
 monkeypatch.setattr(m.sys,'argv',[m.MODULE,flag,sha,'--preflight'])
 monkeypatch.setattr(guard,'require_native_provider_hlc_all_tf_learning_execution',lambda **kw:{})
 monkeypatch.setattr(m,'inspect',lambda *a:({},reference(m.PLAN),{},{}))
 monkeypatch.setattr(m,'run',lambda *a:pytest.fail('preflight tried execution'))
 m.main();output=capsys.readouterr().out
 assert '"claim_created": false'in output and '"NPY_arrays_decoded": 0'in output
