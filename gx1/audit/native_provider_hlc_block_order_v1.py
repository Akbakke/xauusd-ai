"""Separately admitted whole-batch order; existing native training engine only."""
from __future__ import annotations
import argparse,ast,copy,hashlib,json,os,re,signal,subprocess,sys,time
from pathlib import Path
if __name__=='__main__':
 from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
 install_native_m5_direction_io_firewall()
import numpy as np
from gx1.audit import native_provider_hlc_learning_v1 as learning
train,core=learning.train,learning.core
REPO,POLICY=learning.REPO,learning.POLICY
read,ref,pub=learning.read,learning.ref,learning.pub
NAME='NATIVE_PROVIDER_HLC_ALL_TF_BLOCK_ORDER_20261124_001';STAGE=NAME.lower();RECOVERY_STAGE=STAGE+'_recovery'
MODULE='gx1.audit.native_provider_hlc_block_order_v1';SELF='gx1/audit/native_provider_hlc_block_order_v1.py';TEST='tests/test_native_provider_hlc_block_order_v1.py'
GUARD_TEST='tests/test_native_provider_hlc_block_order_guard_v1.py'
PLAN=REPO/'configs/research'/(NAME+'.json');ROOT=Path('/home/andre2/GX1_RUNS')/NAME
ANCHOR='d5f337d12ab14791669e5beb8e9fbf254e40edd3'
DESIGN={'path':str(learning.E/'V2_NONE_BATCH_ORDER_IMPLEMENTATION_DESIGN002.json'),'sha256':'89b0ede3c3cee404580fb1289b02c6a4ae244302dc060b65b8d1126ae4089c84'}
CHRONOLOGY={'path':str(learning.ALL_TF_PLAN),'sha256':'9a82753bcf6702a14a2eda1a22acdde203cb612105b7837a2e34d7d5c21a70f8'}
MAIN_SCHEMA='gx1_native_provider_hlc_block_order_one_epoch_plan_v1'
RESUME_SCHEMA='gx1_native_provider_hlc_block_order_remaining_epoch_plan_v1'
SCHEDULE_SCHEMA='gx1_native_provider_hlc_whole_block_order_schedule_v1'
SCOPE_SCHEMA='gx1_native_provider_hlc_block_order_recovery_scope_v1'
ENGINE='gx1/audit/native_provider_hlc_learning_v1.py'
GUARDS={'gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh'}
ENGINE_TAGS=('MAIN_SCOPE','MAIN_BATCHES','MAIN_CHECKPOINT','MAIN_PACKET','MAIN_JOURNAL','MAIN_TRAIN_END','MAIN_REPORT',
             'RESUME_SCOPE','RESUME_BATCHES','RESUME_PREFIX','RESUME_BUFFERS','RESUME_CHECKPOINT','RESUME_PACKET','RESUME_JOURNAL','RESUME_REPORT')


def need(ok,why):
 if not bool(ok):raise ValueError('PROVIDER_HLC_BLOCK_ORDER_'+why)


def build_schedule(canonical_schedule,recipe):
 """Bind order from authenticated canonical metadata, before reading payloads."""
 learning.all_tf_validate_clock(recipe,canonical_schedule)
 before=core.state_sha(np.random.get_state());generator=np.random.Generator(np.random.PCG64(recipe['seed']))
 ids=np.r_[generator.permutation(3693),3693].astype(np.int64)
 body=copy.deepcopy(canonical_schedule);body.pop('contract_sha256')
 body.update(schema_version=SCHEDULE_SCHEMA,canonical_schedule=copy.deepcopy(canonical_schedule),
   canonical_schedule_sha256=canonical_schedule['contract_sha256'],original_batch_ids=ids.tolist(),
   original_batch_ids_sha256=core.digest(ids),seed=20260911,algorithm='dedicated_local_numpy_Generator_PCG64',
   tail_rule='original_B22_id3693_last',full_batches=3693,last_batch_rows=22,
   batch_source_rows_sha256=[canonical_schedule['batch_source_rows_sha256'][i]for i in ids],
   batch_issue_ns_sha256=[canonical_schedule['batch_issue_ns_sha256'][i]for i in ids],shuffle=True,drop_last=False)
 need(core.state_sha(np.random.get_state())==before,'SCHEDULE_GLOBAL_RNG_UNCHANGED')
 return train.seal(body)


def validate_schedule(schedule,recipe=None):
 need(type(schedule)is dict and schedule.get('schema_version')==SCHEDULE_SCHEMA,'SCHEDULE_SCHEMA')
 canonical=schedule['canonical_schedule']
 recipe=read(DESIGN)['recipe']if recipe is None else recipe
 need(len(canonical['batch_source_rows_sha256'])==len(canonical['batch_issue_ns_sha256'])==3694,'CANONICAL_BATCH_INVENTORY')
 need(schedule==build_schedule(canonical,recipe),'EXACT_SEALED_SCHEDULE')
 ids=schedule['original_batch_ids']
 need(all(type(i)is int for i in ids)and len(ids)==3694 and sorted(ids)==list(range(3694))and ids[-1]==3693,'EXACT_BATCH_BIJECTION')
 return schedule


def _cursor_binding(schedule,cursor):
 pos=train.position(schedule,cursor);ids=schedule['original_batch_ids'];next_id=ids[cursor]if cursor<len(ids)else None
 return {'schedule_sha256':schedule['contract_sha256'],'position':pos,'phase':'FINAL_SCORE'if next_id is None else'TRAIN',
         'next_original_batch_id':next_id,
         'next_source_rows_sha256':schedule['batch_source_rows_sha256'][cursor]if next_id is not None else None,
         'next_issue_ns_sha256':schedule['batch_issue_ns_sha256'][cursor]if next_id is not None else None}


def cursor_binding(schedule,cursor):
 validate_schedule(schedule)
 return _cursor_binding(schedule,cursor)


def resolve_batches(context,schedule):
 validate_schedule(schedule);canonical=schedule['canonical_schedule']
 source=context['source_rows'];issues=context['truth']['issue_ns']
 need(isinstance(source,np.ndarray)and source.dtype==np.int64 and source.shape==(354550,)and np.all(np.diff(source)>0)
  and isinstance(issues,np.ndarray)and issues.dtype==np.int64 and issues.shape==(354550,)and np.all(np.diff(issues)>0)
  and core.digest(source)==canonical['source_rows_sha256']and core.digest(issues)==canonical['issue_ns_sha256'],'CANONICAL_FULL_CLOCK')
 base=core.all_score_batches(np.arange(354550,dtype=np.int64),96)
 need(len(base)==3694 and [len(b)for b in base]==[96]*3693+[22],'CANONICAL_BATCH_GEOMETRY')
 for i,take in enumerate(base):
  need(core.digest(source[take])==canonical['batch_source_rows_sha256'][i]
   and core.digest(issues[take])==canonical['batch_issue_ns_sha256'][i],'CANONICAL_BATCH_CLOCK')
 return [base[i]for i in schedule['original_batch_ids']]


def expected_step_binding(schedule,index):
 need(type(index)is int and 0<=index<3694,'EXECUTION_INDEX')
 original=schedule['original_batch_ids'][index]
 return {'execution_index':index,'original_batch_id':original,'rows':22 if original==3693 else 96,
         'source_rows_sha256':schedule['batch_source_rows_sha256'][index],'issue_ns_sha256':schedule['batch_issue_ns_sha256'][index]}


def step_binding(context,schedule,index,take,packet):
 expected=expected_step_binding(schedule,index);original=expected['original_batch_id']
 canonical=np.arange(original*96,min(original*96+96,354550),dtype=np.int64)
 need(isinstance(take,np.ndarray)and take.dtype==np.int64 and np.array_equal(take,canonical),'ACTUAL_ORIGINAL_BATCH_ROWS')
 issues=context['truth']['issue_ns'][take]
 need(core.digest(context['source_rows'][take])==expected['source_rows_sha256']and core.digest(issues)==expected['issue_ns_sha256']
  and np.array_equal(packet['truth']['issue_ns'],issues),'ACTUAL_PACKET_CLOCK')
 return expected


def validate_prefix(schedule,steps,cursor):
 validate_schedule(schedule);train.position(schedule,cursor)
 need(type(steps)is list and len(steps)==cursor,'PREFIX_LENGTH')
 for j,step in enumerate(steps):
  need(type(step)is dict and type(step.get('logical_step'))is int and step['logical_step']==j+1
   and step.get('skip_update')is False and step.get('order_binding')==expected_step_binding(schedule,j),'PREFIX_EXECUTION_BINDING')
 return _cursor_binding(schedule,cursor)


def engine_source_bridge():
 before=subprocess.check_output(['git','show',ANCHOR+':'+ENGINE],cwd=REPO);after=(REPO/ENGINE).read_text();normalized=after
 for tag in ENGINE_TAGS:
  start='# BEGIN GX1 BLOCK ORDER ENGINE 20261124 001 '+tag+'\n';end='# END GX1 BLOCK ORDER ENGINE 20261124 001 '+tag+'\n'
  need(normalized.count(start)==normalized.count(end)==1,'ENGINE_SINGLE_NAMED_SEAM')
  a=normalized.index(start);z=normalized.index(end,a)+len(end);normalized=normalized[:a]+normalized[z:]
 normalized=normalized.replace('def physical_all_tf(plan,binding,pref,live,*,order_scope=None):','def physical_all_tf(plan,binding,pref,live):',1)
 normalized=normalized.replace('def physical_all_tf_resume(plan,binding,pref,live,*,order_scope=None):','def physical_all_tf_resume(plan,binding,pref,live):',1)
 a=normalized.index('def physical_all_tf(');z=normalized.index('\ndef physical_all_tf_resume(',a)
 main=normalized[a:z].replace('order_root','ALL_TF_ROOT').replace('canonical_schedule',"plan['schedule']").replace("  report={'counts':counts","  return {'counts':counts",1)
 normalized=normalized[:a]+main+normalized[z:]
 a=normalized.index('def physical_all_tf_resume(');z=normalized.index('\ndef run_all_tf(',a)
 normalized=normalized[:a]+normalized[a:z].replace('canonical_schedule',"original['schedule']")+normalized[z:]
 need(normalized.encode()==before,'ENGINE_ALL_OLD_BYTES_AND_DEFAULTS')
 return {'anchor':ANCHOR,'old_sha256':hashlib.sha256(before).hexdigest(),'current_sha256':hashlib.sha256(after.encode()).hexdigest(),
         'named_seams':list(ENGINE_TAGS),'all_other_functions_and_normalized_default_branches_byte_identical':True}


def guard_source_bridge():
 name='gx1/contracts/gx1_capped_execution_v1.py'
 before=subprocess.check_output(['git','show',ANCHOR+':'+name],cwd=REPO).decode();after=(REPO/name).read_text()
 expected={'gx1/contracts/gx1_capped_execution_v1.py':'4ba6a3dc1e9eddabdbeee2ddd46f30a4408ffa26751671c48e18cb36bf55149c',
           'scripts/gx1_capped_run.sh':'e37234fb04a557ca8407838e0f95b52447f1d55562aa24494dee6d50c2962ea0'}
 need(hashlib.sha256(after.encode()).hexdigest()==expected[name],'GUARD_FROZEN_IMPLEMENTATION')
 def spans(text):
  out={};lines=text.splitlines(keepends=True)
  for node in ast.parse(text).body:
   if isinstance(node,(ast.FunctionDef,ast.Assign)):
    key=node.name if isinstance(node,ast.FunctionDef)else next((t.id for t in node.targets if isinstance(t,ast.Name)),'')
    if key in('_native_provider_hlc_all_tf_learning_bound_plan','_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_GUARD_SHELL_ADDITIONS'):
     out[key]=(''.join(lines[node.lineno-1:node.end_lineno]),node)
  need(len(out)==2,'GUARD_EXACT_CHANGED_SYMBOLS');return out
 old,new=spans(before),spans(after);normalized=after
 for key in old:
  need(normalized.count(new[key][0])==1,'GUARD_SINGLE_BINDING_DELTA');normalized=normalized.replace(new[key][0],old[key][0],1)
 need(normalized==before,'GUARD_ALL_OTHER_BYTES')
 symbol='_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_GUARD_SHELL_ADDITIONS'
 oldparts=ast.literal_eval(old[symbol][1].value);newparts=ast.literal_eval(new[symbol][1].value)
 need(len(oldparts)==len(newparts)==5 and all(oldparts[i]==newparts[i]for i in(0,3,4)),'SHELL_EXACT_CHANGED_FRAGMENTS')
 name='scripts/gx1_capped_run.sh';shell=(REPO/name).read_text();normalized=shell
 need(hashlib.sha256(shell.encode()).hexdigest()==expected[name],'SHELL_FROZEN_IMPLEMENTATION')
 for i in(1,2):
  need(normalized.count(newparts[i][2])==1,'SHELL_SINGLE_BINDING_DELTA')
  normalized=normalized.replace(newparts[i][2],oldparts[i][2],1)
 oldshell=subprocess.check_output(['git','show',ANCHOR+':'+name],cwd=REPO)
 need(normalized.encode()==oldshell,'SHELL_ALL_OTHER_BYTES')
 return {'anchor':ANCHOR,'current_sha256':expected,'old_sha256':{
  'gx1/contracts/gx1_capped_execution_v1.py':hashlib.sha256(before.encode()).hexdigest(),name:hashlib.sha256(oldshell).hexdigest()},
  'all_other_guard_shell_bytes_and_execution_ENV_proof_unchanged':True}


HISTORICAL_GUARD_TEST='tests/test_native_provider_hlc_all_tf_learning_guard_v1.py'
HISTORICAL_GUARD_TEST_OLD_SHA='98e23fbd9a58660709e30bbbd9d60dbc45e7f8712ec1482aea6f684bd9a2e13c'
HISTORICAL_GUARD_TEST_NEW_SHA='247ec727384e17435fd0c31fc305a14beeb732d5c088fff4fcba9d3505ad18d5'


def historical_guard_test_bridge(before,after):
 need(hashlib.sha256(before).hexdigest()==HISTORICAL_GUARD_TEST_OLD_SHA
  and hashlib.sha256(after).hexdigest()==HISTORICAL_GUARD_TEST_NEW_SHA,'HISTORICAL_GUARD_TEST_EXACT_SHA')
 start='# BEGIN GX1 HISTORICAL ALL TF GUARD SOURCE FIXTURE 20261124 001\n'
 end='# END GX1 HISTORICAL ALL TF GUARD SOURCE FIXTURE 20261124 001\n\n'
 normalized=after.decode();need(normalized.count(start)==normalized.count(end)==1,'HISTORICAL_GUARD_TEST_SINGLE_FIXTURE')
 left,tail=normalized.split(start);_,right=tail.split(end);normalized=left+right
 for old,new in (
  ('def test_source_bridge_preserves_every_old_owner_byte_and_ast():\n','def test_historical_source_bridge_preserves_every_old_owner_byte_and_ast(historical_guard_source):\n'),
  ('def test_source_bridge_rejects_unscoped_source_or_route_authority(monkeypatch, case):\n','def test_source_bridge_rejects_unscoped_source_or_route_authority(monkeypatch, historical_guard_source, case):\n')):
  need(normalized.count(new)==1,'HISTORICAL_GUARD_TEST_EXACT_SIGNATURE');normalized=normalized.replace(new,old,1)
 need(normalized.encode()==before,'HISTORICAL_GUARD_TEST_OTHER_BYTES')
 return {'path':HISTORICAL_GUARD_TEST,'anchor':ANCHOR,'before_sha256':HISTORICAL_GUARD_TEST_OLD_SHA,
  'after_sha256':HISTORICAL_GUARD_TEST_NEW_SHA,'test_only_historical_fixture_and_two_signatures':True,'all_other_test_bytes_unchanged':True}


def source_proof(*parents):
 engine=engine_source_bridge();guard=guard_source_bridge()
 old_test=subprocess.check_output(['git','show',ANCHOR+':'+HISTORICAL_GUARD_TEST],cwd=REPO)
 test_proof=historical_guard_test_bridge(old_test,(REPO/HISTORICAL_GUARD_TEST).read_bytes())
 need(all(p['source_owners'].get(HISTORICAL_GUARD_TEST,HISTORICAL_GUARD_TEST_OLD_SHA)==HISTORICAL_GUARD_TEST_OLD_SHA for p in parents),'AUTHENTIC_HISTORICAL_GUARD_TEST_PARENT')
 names=set().union(*(set(p['source_owners'])for p in parents))
 for name in names-GUARDS-{ENGINE,HISTORICAL_GUARD_TEST}:
  before=subprocess.check_output(['git','show',ANCHOR+':'+name],cwd=REPO)
  need((REPO/name).read_bytes()==before,'ANCHOR_NUMERIC_OWNER_UNCHANGED')
  need(all(p['source_owners'].get(name,hashlib.sha256(before).hexdigest())==hashlib.sha256(before).hexdigest()for p in parents),'AUTHENTIC_NUMERIC_PARENT')
 return {'engine':engine,'guard':guard,'historical_guard_test_only':test_proof,'all_other_numeric_owners_byte_identical':True}


def validate_reuse(chronology,base,replay,acceptance,initial_acceptance,initial,design,runtime):
 need(design['schema_version']=='gx1_None_block_order_implementation_design_v1'and design['basename']==NAME and design['stage']==STAGE
  and design['module']==MODULE and design['plan_path']==str(PLAN)and design['output_root']==str(ROOT)and design['source_anchor_commit']==ANCHOR
  and design['schemas']=={'main':MAIN_SCHEMA,'resume':RESUME_SCHEMA}and design['recipe']==chronology['recipe']and design['execution']==learning.EXECUTION
  and all(design[k]is False for k in ('execution_authorized','main_fit_admitted','test_authorized','training_authorized')),'LOCKED_DESIGN')
 need(chronology['backbone_gradient_policy']is None and base['backbone_gradient_policy']is None and replay['backbone_gradient_policy']is None
  and chronology['model_contract']==base['model_contract']==replay['model_contract']
  and chronology['model_contract']['contract_sha256']=='6d6ac22819e54c7b12aebcf4e1afe27b990d8a4d90d16500e352c7353d99cca0','SAME_NONE_MODEL_FUNCTION')
 meta=base['metadata']
 for other in (chronology,replay):
  need(all(meta[k]==other['metadata'][k]for k in meta if k!='runtime'),'SAME_INPUT_TARGET_NORM_PRIOR_ARCHITECTURE')
  need(all(runtime[k]==other['metadata']['runtime'][k]for k in ('library_versions','gpu')),'SAME_NATIVE_RUNTIME')
 need(all(runtime[k]==meta['runtime'][k]for k in ('library_versions','gpu')),'SAME_INITIAL_NATIVE_RUNTIME')
 need(initial==chronology['initial_reference']==replay['initial_reference']and initial['initial_only']is True and initial['optimizer_updates']==0
  and initial['model_state_sha256']=='55ce06ac8df8fdf3892003d571a3902ffa9616909edb26d8a60372bd09432fb9'
  and initial['rng_sha256']=='343bc9e441680c73e99ed7fe89595ed2d423ac91b7f2071cb19d74043494305d','SAME_QUALIFIED_INITIAL_STATE_RNG')
 need(initial_acceptance['full_initialbaseline_qualified']is True and initial_acceptance['actual_NN_forward_rows']==354550
  and all(initial_acceptance['actual_reference_comparison'].values())and initial_acceptance['source_and_each_segment_bookends_pass']is True
  and initial_acceptance['initializer']==initial and initial_acceptance['initial_receipt']==replay['inputs']['baseline_initial_receipt']
  and initial_acceptance['result']==replay['inputs']['baseline_result'],'REUSED_QUALIFIED_WHOLE_INITIAL')
 need(acceptance['plan']==learning.ALL_TF_REPLAY and acceptance['optimizer_replay_qualified']is True and acceptance['all10_proofs_bitexact']is True
  and len(acceptance['all10_proofs'])==10 and acceptance['both_lanes_complete_consumed']is True
  and acceptance['restored_nonzero_moments']['backbone']>0 and acceptance['restored_nonzero_moments']['readout']>0
  and acceptance['equal_TF_eval_parity_bitexact']is True and acceptance['finite_gradient_parameters_all_bitexact']is True,'REUSED_UNCHANGED_NONE_COLD_PRIMITIVES')
 need(all(chronology['recipe'][k]==replay['recipe'][k]for k in ('seed','train_batch_rows','optimizer','scheduler','gradient_clip_norm')),'SAME_REPLAY_RECIPE')
 learning.all_tf_validate_clock(chronology['recipe'],chronology['schedule'])


def build_plan(runtime,source_commit=None):
 core.future_cuda_require_runtime(runtime);need(source_commit in(None,runtime['source_commit']),'RUNTIME_SOURCE')
 design=read(DESIGN);chronology=read(CHRONOLOGY);base=read(learning.ALL_TF_BASE);replay=read(learning.ALL_TF_REPLAY)
 acceptance=read(learning.ALL_TF_REPLAY_ACCEPTANCE);initial_acceptance=read(replay['inputs']['baseline_acceptance']);initial=read(replay['inputs']['baseline_initial_receipt'])
 validate_reuse(chronology,base,replay,acceptance,initial_acceptance,initial,design,runtime)
 from gx1.contracts.market_state_ta_integrated_model_v1 import native_future_state_model_contract
 meta=copy.deepcopy(chronology['metadata']);meta['runtime']=copy.deepcopy(runtime)
 binding=read(meta['input_binding_ref']);normalization=read(meta['normalization']);prior=read(meta['prior'])
 need(binding==meta['input_binding']and binding['input_normalization_sha256']==normalization['contract_sha256']
  and native_future_state_model_contract(binding,architecture=meta['architecture'],train_prior=prior,target_contract=meta['target_contract'],backbone_gradient_policy=None)==chronology['model_contract'],'PHYSICAL_SAME_ONLINE_FUNCTION')
 review=read(design['observed_deviation']['full_population_paired_review']);final=read(review['None_result']);quality=read(review['None_quality'])
 need(final['status']=='COMPLETE_NONE_FULL_POPULATION_TRAIN_DIAGNOSTIC_EPOCH'and final['source_inputs_policy_boot_unchanged']is True and final['test_data_used']is False
  and final['report']['original_plan']==CHRONOLOGY and final['report']['actual_optimizer_updates']==3694
  and final['report']['final_quality']==review['None_quality']and quality['rows']==354550
  and quality['truth_sha256']==meta['truth_sha256']and quality['output_sha256']==final['report']['whole_FINAL_score']['output_sha256'],'FROZEN_CHRONOLOGICAL_NONE_REFERENCE')
 endpoint_acceptance=read(design['observed_deviation']['endpoint_acceptance']);endpoint=read(endpoint_acceptance['result']);endpoint_plan=read(endpoint['plan'])
 need(endpoint['status']=='COMPLETE_NATIVE_NONE_ENDPOINT_CROSS'and endpoint_plan['model_contract']==chronology['model_contract'],'AUTHENTIC_CONTEXT_INVENTORY_PROVENANCE')
 context_io={'inventory_provenance':endpoint['plan'],'archives':{k:copy.deepcopy(endpoint_plan['input_inventory'][k])for k in('feature_archive','target_arrays')},
  'archive_loads':2,'decoded_fields':41,'decoded_bytes':1483917408,'counts_authority':'Unchanged native.load_context readers plus existing feature_cache_payload_loads=1/target_archive_loads=1; no additional decoder/counter framework.'}
 need(all(context_io['archives'][k]['binding']==chronology['inputs'][k]for k in context_io['archives'])
  and sum(len(v['decoded_fields'])for v in context_io['archives'].values())==41
  and sum(v['decoded_bytes']for v in context_io['archives'].values())==1483917408,'EXACT_REUSED_CONTEXT_IO')
 proof=source_proof(chronology,base,replay)
 names=set(chronology['source_owners'])|set(base['source_owners'])|set(replay['source_owners'])|{SELF,TEST,GUARD_TEST}
 inputs={**copy.deepcopy(chronology['inputs']),'context_inventory_plan':endpoint['plan'],'context_inventory_result':endpoint_acceptance['result'],'chronological_None_plan':CHRONOLOGY,'chronological_None_result':review['None_result'],
         'chronological_None_quality':review['None_quality'],'block_order_design':DESIGN,**{k:v for k,v in design['observed_deviation'].items()if type(v)is dict}}
 schedule=build_schedule(chronology['schedule'],chronology['recipe'])
 return {'schema_version':MAIN_SCHEMA,'stage':STAGE,'source_commit':source_commit or runtime['source_commit'],
  'source_owners':{name:ref(REPO/name)['sha256']for name in sorted(names)},'source_bridge':proof,'metadata':meta,'model_contract':chronology['model_contract'],
  'initial_reference':copy.deepcopy(initial),'inputs':inputs,'recipe':copy.deepcopy(chronology['recipe']),'schedule':schedule,'backbone_gradient_policy':None,
  'execution':copy.deepcopy(learning.EXECUTION),'limits':copy.deepcopy(design['main_limits']),'context_IO':context_io,'output_root':str(ROOT),'train_end_checkpoint_before_score':True,
  'scientific_scope':{'question':design['question'],'preregistered_review':design['preregistered_review'],'limits':design['limits'],
   'new_physical_order_cold_replay':'NOT_MEASURED','only_changed_training_factor':'whole_original_B96_batch_order_B22_last',
   'proper_score_PASS_does_not_qualify_step15_bidirectionality_persistent_bias_cause_or_edge':True,
   'step15_requires':'OPP/NED/FLAT + ranking + calibration + proper scores + explained/qualified persistent bias and chosen correction',
   'new_epochs_or_selection':False,'TRAIN_diagnostic_only':True,'later_generalization':'UNKNOWN','economic_edge':'UNKNOWN','all20_approved':False},
  'authority':'ONE_SEPARATELY_ADMITTED_ORDER_COMPARISON_NO_VAL_TEST_OR_SELECTION'}


def gate(binding,admitted=False,cursor=None):
 value=learning.all_tf_gate(binding,admitted)if cursor is None else learning.all_tf_resume_gate(binding,admitted,cursor)
 value['status']=('ADMITTED_EXACT_BLOCK_ORDER_ONE_EPOCH'if cursor is None else'ADMITTED_EXACT_BLOCK_ORDER_ORIGINAL_SUFFIX')if admitted else'PREPARED_CLOSED_NOT_ADMITTED'
 return value


def recovery_root(sha):return ROOT/('RECOVERY_'+sha)


def require_engine_scope(scope,plan,binding,mode):
 need(scope=={'module':MODULE,'mode':mode,'plan':binding}and mode in('main','resume'),'EXACT_ENGINE_SCOPE')
 need(plan['schema_version']==(MAIN_SCHEMA if mode=='main'else RESUME_SCHEMA)and plan['stage']==(STAGE if mode=='main'else RECOVERY_STAGE)
  and plan['output_root']==str(ROOT)and plan['backbone_gradient_policy']is None and plan['execution']==learning.EXECUTION,'ENGINE_PLAN_IDENTITY')
 expected=PLAN if mode=='main'else ROOT/'RECOVERY_PLANS'/(binding['sha256']+'.json')
 need(binding['path']==str(expected),'ENGINE_PLAN_PATH')
 validate_schedule(plan['schedule'],plan['recipe'])
 return ROOT if mode=='main'else recovery_root(binding['sha256'])


def validate_recovery(original,scope):
 """Authenticate a stopped, consumed producer and its durable original trajectory."""
 keys={'schema_version','original_plan','producer_plan','producer_claim','producer_terminal','producer_admitted_policy','checkpoint_receipt','checkpoint','stop_review','backup'}
 need(type(scope)is dict and set(scope)==keys and scope['schema_version']==SCOPE_SCHEMA,'ALL_TF_RECOVERY_SCOPE_SCHEMA')
 for key in keys-{'schema_version'}:learning.resume_reference(scope[key])
 original_ref=scope['original_plan'];need(original_ref['path']==str(PLAN)and original['schema_version']==MAIN_SCHEMA and original['stage']==STAGE and original['output_root']==str(ROOT)and original['backbone_gradient_policy']is None and original['execution']==learning.EXECUTION,'ALL_TF_RECOVERY_ORIGINAL_IDENTITY')
 validate_schedule(original['schedule'],original['recipe']);control=read(CHRONOLOGY)
 need(original['inputs']['chronological_None_plan']==CHRONOLOGY and original['inputs']['block_order_design']==DESIGN and original['recipe']==control['recipe']and original['schedule']==build_schedule(control['schedule'],control['recipe']),'ORDER_RECOVERY_AUTHENTIC_ORIGINAL_SCHEDULE')
 initial=read(original['inputs']['baseline_initial_receipt']);need(original['initial_reference']==initial and initial['model_state_sha256']=='55ce06ac8df8fdf3892003d571a3902ffa9616909edb26d8a60372bd09432fb9'and initial['model_contract_sha256']==original['model_contract']['contract_sha256']=='6d6ac22819e54c7b12aebcf4e1afe27b990d8a4d90d16500e352c7353d99cca0','ALL_TF_RECOVERY_OWN_INITIAL_CONTRACT')
 traj=learning.trajectory(original);receipt=read(scope['checkpoint_receipt']);proof=train.require_recovery_checkpoint_proof(receipt['proof']);cursor=receipt['logical_cursor'];producer=read(scope['producer_plan']);producer_ref=scope['producer_plan'];end=original['schedule']['logical_batches'];train_end=type(cursor)is int and cursor==end
 need(set(receipt)=={'schema_version','checkpoint','original_plan','execution_plan','producer_lane','producer_runtime','logical_cursor','optimizer_updates','trajectory_sha256','proof'}and receipt['schema_version']=='gx1_native_future_state_training_checkpoint_receipt_v1'and receipt['checkpoint']==scope['checkpoint']and receipt['original_plan']==original_ref and receipt['execution_plan']==producer_ref and receipt['producer_lane']=='main'and type(cursor)is int and (train_end or(0<cursor<end and cursor%256==0)) and type(receipt['optimizer_updates'])is int and receipt['optimizer_updates']==cursor,'ALL_TF_RECOVERY_EXACT_DURABLE_CURSOR')
 producer_root=ROOT
 if producer_ref!=original_ref:
  producer_root=recovery_root(producer_ref['sha256']);need(producer_ref['path']==str(ROOT/'RECOVERY_PLANS'/(producer_ref['sha256']+'.json'))and producer['schema_version']==RESUME_SCHEMA and producer['stage']==RECOVERY_STAGE and producer['output_root']==str(ROOT)and producer['output_directory_prefix']=='RECOVERY_'and producer['metadata']['original']==original and producer['metadata']['original_plan']==original_ref and producer['metadata']['trajectory']==traj and producer['starting_cursor']<cursor and producer['model_contract']==original['model_contract']and producer['initial_reference']==original['initial_reference']and producer['backbone_gradient_policy']is None and producer['recipe']==original['recipe']and producer['schedule']==original['schedule']and producer['execution']==learning.EXECUTION,'ALL_TF_RECOVERY_PREVIOUS_SUFFIX_LINEAGE')
 else:need(producer==original,'ALL_TF_RECOVERY_ORIGINAL_PRODUCER')
 if train_end:need(producer['schema_version']in(MAIN_SCHEMA,RESUME_SCHEMA)and producer['train_end_checkpoint_before_score']is True and producer.get('score_only',False)is False,'ALL_TF_RECOVERY_AUTHENTIC_TRAIN_END_PRODUCER')
 checkpoint_name='TRAIN_END'if train_end else'UPDATE%06d'%cursor
 learning.resume_reference(scope['checkpoint_receipt'],producer_root/(checkpoint_name+'.json'));learning.resume_reference(scope['checkpoint'],producer_root/(checkpoint_name+'.pt'))
 need(receipt['trajectory_sha256']==traj['contract_sha256']and proof['trajectory']==core.state_sha(traj)and proof['position']==core.state_sha(train.position(original['schedule'],cursor))and proof['scheduler_state']==core.state_sha(None)and proof['training']==core.state_sha(True)and proof['scores']==core.state_sha({'INITIAL':{'reused_qualified_result':original['inputs']['baseline_result'],'no_repeated_initial_forwards':True}}),'ALL_TF_RECOVERY_AUTHENTIC_STATE_PROOFS')
 core.future_cuda_runtime_matches(producer['metadata']['runtime'],receipt['producer_runtime']);claim=read(scope['producer_claim']);terminal=read(scope['producer_terminal']);policy=read(scope['producer_admitted_policy'])
 learning.resume_reference(scope['producer_claim'],producer_root/'CLAIM.json');learning.resume_reference(scope['producer_terminal'],producer_root/'TERMINAL.json');train.require_recovery_ref(claim['policy_at_claim'],expected_path=POLICY)
 need(claim['plan']==producer_ref and claim['policy_at_claim']['sha256']==scope['producer_admitted_policy']['sha256']and claim['runtime']==receipt['producer_runtime']and claim['remaining_invocations']==0 and claim['TRAIN_diagnostic_only']is True,'ALL_TF_RECOVERY_AUTHENTIC_PRODUCER_CLAIM')
 learning.validate_flags(policy,True);expected=gate(producer_ref,True)if producer_ref==original_ref else gate(producer_ref,True,producer['starting_cursor']);need(policy[producer['stage']]==expected,'ALL_TF_RECOVERY_FROZEN_PRODUCER_ADMISSION')
 need(terminal['status']=='FAILED_CONSUMED'and terminal['exit_code']==1 and terminal['plan']==producer_ref and terminal['claim']==scope['producer_claim']and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False and 'result'not in terminal,'ALL_TF_RECOVERY_FAILED_CONSUMED_PRODUCER');core.future_cpu_parent_stopped(claim['process_identity'])
 review=read(scope['stop_review']);backup=read(scope['backup'])
 need(review['execution_plan']==producer_ref and review['original_plan']==original_ref and review['process_stopped']is True and review['completed_epoch']is False and review['last_durable_cursor']==cursor and review['last_durable_checkpoint']==scope['checkpoint']and review['last_durable_receipt']==scope['checkpoint_receipt'],'ALL_TF_RECOVERY_STOP_REVIEW')
 need(backup['schema_version']=='gx1_bound_separate_disk_backup_ack_v1'and backup['Mac_F_FULLFSYNC_and_disk_SHA_readback']is True and backup['checkpoint']==scope['checkpoint']and backup['checkpoint_receipt']==scope['checkpoint_receipt'],'ALL_TF_RECOVERY_DURABLE_BACKUP')
 return receipt,traj,producer


def build_resume_plan(runtime,scope,source_commit=None):
 core.future_cuda_require_runtime(runtime);need(source_commit in(None,runtime['source_commit']),'RECOVERY_RUNTIME_SOURCE')
 learning.resume_reference(scope);scope_value=read(scope);original=read(scope_value['original_plan'])
 receipt,traj,producer=validate_recovery(original,scope_value);train.runtime_compatible(traj['runtime_compatibility'],runtime)
 for parent in(original,producer):
  for name,digest in parent['source_owners'].items():
   need(hashlib.sha256(subprocess.check_output(['git','show',parent['source_commit']+':'+name],cwd=REPO)).hexdigest()==digest,'RECOVERY_GIT_PRODUCER_IDENTITY')
   need(ref(REPO/name)['sha256']==digest,'RECOVERY_FROZEN_SOURCE_OWNER')
 cursor=receipt['logical_cursor'];limits,cursors=learning.all_tf_resume_limits(original,cursor,phase_recovery=True)
 inputs={**original['inputs'],**{k:v for k,v in scope_value.items()if k!='schema_version'},'recovery_scope':copy.deepcopy(scope)}
 return {'schema_version':RESUME_SCHEMA,'stage':RECOVERY_STAGE,'source_commit':source_commit or runtime['source_commit'],
  'source_owners':copy.deepcopy(original['source_owners']),'source_bridge':copy.deepcopy(original['source_bridge']),
  'metadata':{'original':original,'original_plan':scope_value['original_plan'],'receipt':receipt,'trajectory':traj,'runtime':copy.deepcopy(runtime)},
  'inputs':inputs,'context_IO':copy.deepcopy(original['context_IO']),'recipe':copy.deepcopy(original['recipe']),'schedule':copy.deepcopy(original['schedule']),'model_contract':original['model_contract'],
  'initial_reference':copy.deepcopy(original['initial_reference']),'backbone_gradient_policy':None,'starting_cursor':cursor,
  'remaining_TRAIN_rows':original['schedule']['rows']-min(cursor*96,354550),'remaining_FINAL_rows':354550,
  'checkpoint_global_cursors':cursors,'execution':copy.deepcopy(learning.EXECUTION),'limits':limits,'output_root':str(ROOT),
  'output_directory_prefix':'RECOVERY_','scientific_scope':copy.deepcopy(original['scientific_scope']),
  'authority':'EXACT_ORIGINAL_BLOCK_ORDER_SUFFIX_NO_NEW_EPOCH_OR_VAL_TEST_SELECTION','original_trajectory_preserved_verbatim':True,
  'new_INITIAL_writes':0,'train_end_checkpoint_before_score':True,'score_only':cursor==3694}


def inspect(sha,mode,admitted):
 need(mode in('main','resume')and type(sha)is str and len(sha)==64 and all(c in'0123456789abcdef'for c in sha),'MODE_SHA')
 path=PLAN if mode=='main'else ROOT/'RECOVERY_PLANS'/(sha+'.json');binding={'path':str(path),'sha256':sha}
 learning.resume_reference(binding,path);plan=read(binding)
 expected=build_plan(plan['metadata']['runtime'],plan['source_commit'])if mode=='main'else build_resume_plan(plan['metadata']['runtime'],plan['inputs']['recovery_scope'],plan['source_commit'])
 need(plan==expected,'EXACT_CURRENT_PLAN');learning.check_sources_inputs(plan)
 pref=ref(POLICY);policy=read(pref);learning.validate_flags(policy,admitted)
 need(policy[STAGE if mode=='main'else RECOVERY_STAGE]==gate(binding,admitted,None if mode=='main'else plan['starting_cursor']),'ONE_USE_GATE')
 if mode=='resume':
  original=policy[STAGE];need(original['plan']==plan['metadata']['original_plan']and original['remaining_invocations']==0
   and all(original[k]is False for k in('execution_authorized','full_epoch_training_allowed','additional_epochs_allowed','test_authorized','relaunch_allowed')),'ORIGINAL_GATE_CONSUMED')
 output=ROOT if mode=='main'else recovery_root(sha);need(not output.exists()and not output.is_symlink(),'FRESH_OUTPUT')
 live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['metadata']['runtime'],live)
 return plan,binding,pref,live


def run(sha,mode='main'):
 from gx1.contracts.gx1_capped_execution_v1 import require_native_provider_hlc_all_tf_learning_execution
 need(mode in('main','resume'),'EXACT_MODE');output=ROOT if mode=='main'else recovery_root(sha);began=time.monotonic()
 handlers={sig:signal.getsignal(sig)for sig in(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
 def stop(*_):raise TimeoutError('BLOCK_ORDER_SIGNAL_OR_WALL')
 for sig in handlers:signal.signal(sig,stop)
 signal.setitimer(signal.ITIMER_REAL,learning.EXECUTION['internal_seconds']);claim=None
 try:
  guard=require_native_provider_hlc_all_tf_learning_execution(plan_sha256=sha,mode=mode)
  plan,binding,pref,live=inspect(sha,mode,True);need(guard['plan']==binding and guard['output_root']==str(output),'EXACT_GUARD_PLAN_ROOT')
  output.mkdir(exist_ok=False);claim=pub(output,'CLAIM.json',{'plan':binding,'policy_at_claim':pref,'runtime':live,
   'process_identity':core.future_cpu_process_identity(),'remaining_invocations':0,'TRAIN_diagnostic_only':True,'backbone_gradient_policy':None,
   **({'original_plan':plan['metadata']['original_plan'],'starting_cursor':plan['starting_cursor']}if mode=='resume'else{})})
  allowed={plan['inputs'][k]['path']for k in('feature_archive','target_arrays')}
  cursors=range(256,3694,256)if mode=='main'else plan['checkpoint_global_cursors']
  allowed|={str(output/name)for name in ['FINAL.pt','FINAL_LOGITS.npz']+['UPDATE%06d.pt'%j for j in cursors if j<3694]}
  if mode=='main':allowed|={str(output/'INITIAL.pt'),str(output/'TRAIN_END.pt')}
  else:
   allowed.add(plan['inputs']['checkpoint']['path'])
   if plan['starting_cursor']<3694:allowed.add(str(output/'TRAIN_END.pt'))
  active=[True]
  def audit(event,args):
   if not active[0]:return
   if event in('socket.connect','socket.getaddrinfo','urllib.Request'):raise RuntimeError('BLOCK_ORDER_NETWORK_FORBIDDEN')
   if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
    name=os.fsdecode(args[0])
    if name.endswith(('.npz','.pt','.pth','.parquet','.safetensors','.pkl'))and str(Path(name).resolve())not in allowed:raise RuntimeError('BLOCK_ORDER_PAYLOAD_PATH_FORBIDDEN')
  sys.addaudithook(audit)
  try:
   engine=learning.physical_all_tf if mode=='main'else learning.physical_all_tf_resume
   report=engine(plan,binding,pref,live,order_scope={'module':MODULE,'mode':mode,'plan':binding})
  finally:active[0]=False
  learning.check_sources_inputs(plan)
  need(ref(binding['path'])==binding and ref(POLICY)==pref,'FROZEN_PLAN_POLICY_BOOKEND')
  need(core.physical_windows_boot_id()==live['physical_windows_boot_id']and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==live['wsl_boot_id'],'BOTH_BOOT_BOOKEND')
  need(core.common.firewall_state()['denied_attempts']==0,'M1_FIREWALL')
  result=pub(output,'RESULT.json',{'status':'COMPLETE_NONE_BLOCK_ORDER_TRAIN_COMPARISON','mode':mode,'plan':binding,'claim':claim,'report':report,
   'source_inputs_policy_boot_unchanged':True,'source_unchanged':True,'test_data_used':False,'elapsed_seconds':time.monotonic()-began})
  terminal=pub(output,'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'mode':mode,'plan':binding,'claim':claim,'result':result,
   'remaining_invocations':0,'relaunch_allowed':False,'source_unchanged':True,'test_data_used':False});return {'result':result,'terminal':terminal}
 except BaseException as error:
  if claim is not None:pub(output,'TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'mode':mode,'plan':binding,'claim':claim,
   'error':type(error).__name__+':'+str(error)[:1800],'remaining_invocations':0,'relaunch_allowed':False})
  raise
 finally:
  signal.setitimer(signal.ITIMER_REAL,0)
  for sig,handler in handlers.items():signal.signal(sig,handler)


def main():
 parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
 group.add_argument('--block-order-plan-sha256');group.add_argument('--block-order-resume-plan-sha256');parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
 mode='main'if args.block_order_plan_sha256 is not None else'resume';sha=args.block_order_plan_sha256 or args.block_order_resume_plan_sha256
 flag='--block-order-plan-sha256'if mode=='main'else'--block-order-resume-plan-sha256'
 need(sys.argv[1:]==[flag,sha]+(['--preflight']if args.preflight else[]),'EXACT_CLI')
 if args.preflight:
  from gx1.contracts.gx1_capped_execution_v1 import require_native_provider_hlc_all_tf_learning_execution
  guard=require_native_provider_hlc_all_tf_learning_execution(plan_sha256=sha,mode=mode)
  plan,binding,pref,live=inspect(sha,mode,False)
  print(json.dumps({'status':'EXACT_BLOCK_ORDER_CLI_PREFLIGHT_PASS','mode':mode,'plan':binding,'policy':pref,'guard':guard,
   'models':0,'checkpoint_decodes':0,'NPZ_arrays_decoded':0,'claim_created':False,'CUDA_initialized':False}))
 else:print(json.dumps(run(sha,mode),sort_keys=True))


if __name__=='__main__':main()
