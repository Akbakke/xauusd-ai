"""One separately admitted full-TRAIN row-membership epoch; existing engine/math."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
if __name__=='__main__':
 from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
 install_native_m5_direction_io_firewall()
import numpy as np
from gx1.audit import native_provider_hlc_learning_v1 as learning
from gx1.audit import native_provider_hlc_row_membership_schedule_v1 as schedule_owner
from gx1.audit import native_provider_hlc_row_membership_qualification_v1 as qualified
train,core=learning.train,learning.core
REPO,POLICY,E=learning.REPO,learning.POLICY,learning.E
read,ref,pub=learning.read,learning.ref,learning.pub
NAME='NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_ONE_EPOCH_20261125_001'
MODULE='gx1.audit.native_provider_hlc_row_membership_fit_v1'
STAGE=NAME.lower();RECOVERY_STAGE=STAGE+'_recovery'
PLAN=REPO/'configs/research'/(NAME+'.json');ROOT=Path('/home/andre2/GX1_RUNS')/NAME
SELF='gx1/audit/native_provider_hlc_row_membership_fit_v1.py'
TEST='tests/test_native_provider_hlc_row_membership_fit_v1.py'
GUARD_TEST='tests/test_native_provider_hlc_row_membership_fit_guard_v1.py'
ENGINE_TEST='tests/test_native_provider_hlc_row_membership_engine_v1.py'
ENGINE='gx1/audit/native_provider_hlc_learning_v1.py'
GUARDS={'gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh'}
ANCHOR='8fd74a76cc45ac7306da4b905c82e9cd9b12976d'
MAIN_SCHEMA='gx1_native_provider_hlc_row_membership_one_epoch_plan_v1'
RESUME_SCHEMA='gx1_native_provider_hlc_row_membership_remaining_epoch_plan_v1'
SCOPE_SCHEMA='gx1_native_provider_hlc_row_membership_recovery_scope_v1'
DESIGN_PATH=E/'V2_NONE_SOURCE_ROW_MEMBERSHIP_FULL_EPOCH_IMPLEMENTATION_DESIGN001.json'
DESIGN={'path':str(DESIGN_PATH),'sha256':'c97b7ef56bf3dba81d1624ecb129e04909959bf6fbd1ed7e14c2ed127ecd191a'}
REVIEW_PROTOCOL={'path':str(E/'V2_NONE_SOURCE_ROW_MEMBERSHIP_COMPARISON_PROTOCOL001.json'),'sha256':'5f4244757ba900ce41ea621bb2267b966af99c507e8fe8d1b519b25ac00ec86c'}
REVIEW_HELPER=Path('/tmp/gx1_row_membership_result_review001.py')
CANONICAL={'path':str(REPO/'configs/research/NATIVE_PROVIDER_HLC_ALL_TF_BLOCK_ORDER_20261124_001.json'),'sha256':'a728776d56aed23a3e606418246f12b984dcf86316fd12b4b3509a66d28cf015'}
PREPARATION={'path':str(E/'V2_NONE_SOURCE_ROW_MEMBERSHIP_FULL_EPOCH_PREPARATION_SCOPE001.json'),'sha256':'1880c02b7c9c5e256d9dce604495efae3d6739f889ac80b5952482b9b756af7d'}
PACKET_ACCEPTANCE={'path':str(E/'V2_NONE_SOURCE_ROW_MEMBERSHIP_PACKET_PARITY_ACCEPTANCE001.json'),'sha256':'6bc104675eeb566d5d3b0cbcd6ffca2a2fd2c9d31da90edacb02c2bc9f33c860'}
TECHNICAL_ACCEPTANCE={'path':str(E/'V2_NONE_SOURCE_ROW_MEMBERSHIP_TECHNICAL_ACCEPTANCE001.json'),'sha256':'95163235b96ca817b29c6168412a7de7b837dbac92d0623c4baacbc87a925712'}
CLOCK={'path':str(E/'V2_NONE_SOURCE_ROW_MEMBERSHIP_PACKET_PARITY_CLOCK_SCHEDULE001.json'),'sha256':'0eed6dca4ffa634ed4504b0cea63518ddbbd77842173e7e3f4b0af713102de45'}
MANIFEST={'path':str(E/'V2_NONE_BLOCK_ORDER_SOURCE_ROW_MEMBERSHIP_MANIFEST001.json'),'sha256':'579b6b04c0a8bdff96bbdbe1670c8fb1835aea41daf883074ad3fc3b1a7f368a'}
INITIAL_ACCEPTANCE={'path':str(E/'RESUMABLE_INITIAL003_ACCEPTANCE001.json'),'sha256':'07d186a74ba0691f2735b28cf0f0defad39604573d5f33e2f298c4cf2cc53e45'}
_CHANGE='only_source_blind_sealed_row_membership_originalB22_last_D_b_reweighted_same_native_loss_formula'
_CACHE='_gx1_row_membership_fit_v1_activity'
_ENGINE_TAGS=('MAIN_MODULE','RESUME_MODULE','MAIN_BATCHES','RESUME_BATCHES')


def need(ok,why):
 if not bool(ok):raise ValueError('PROVIDER_HLC_ROW_MEMBERSHIP_FIT_'+why)


def engine_source_bridge():
 before=subprocess.check_output(['git','show',ANCHOR+':'+ENGINE],cwd=REPO);after=(REPO/ENGINE).read_text();normalized=after
 for tag in _ENGINE_TAGS:
  start='# BEGIN GX1 ROW MEMBERSHIP ENGINE 20261125 001 '+tag+'\n';end='# END GX1 ROW MEMBERSHIP ENGINE 20261125 001 '+tag+'\n'
  need(normalized.count(start)==normalized.count(end)==1,'ENGINE_EXACT_NAMED_DELTA')
  a=normalized.index(start);z=normalized.index(end,a)+len(end)
  replacement=''
  if tag=='MAIN_BATCHES':replacement=" if order_scope is not None:batches=order.resolve_batches(context,plan['schedule'])\n"
  if tag=='RESUME_BATCHES':replacement=" if order_scope is not None:batches=order.resolve_batches(context,original['schedule'])\n"
  normalized=normalized[:a]+replacement+normalized[z:]
 need(normalized.encode()==before,'ENGINE_ALL_OLD_BYTES_AND_DEFAULTS')
 return {'anchor':ANCHOR,'original_sha256':hashlib.sha256(before).hexdigest(),'current_sha256':hashlib.sha256(after.encode()).hexdigest(),'normalized_sha256':hashlib.sha256(normalized.encode()).hexdigest(),'exact_named_deltas':list(_ENGINE_TAGS),'all_other_bytes_unchanged':True}


def source_bridge(names):
 from gx1.contracts.gx1_capped_execution_v1 import native_provider_hlc_row_membership_fit_guard_source_bridge
 engine=engine_source_bridge();guard=native_provider_hlc_row_membership_fit_guard_source_bridge(producer_commit=ANCHOR)
 for name in set(names)-GUARDS-{ENGINE,SELF,TEST,GUARD_TEST,ENGINE_TEST}:
  before=subprocess.check_output(['git','show',ANCHOR+':'+name],cwd=REPO)
  need((REPO/name).read_bytes()==before,'ALL_OTHER_ANCHORED_OWNERS_BYTE_IDENTICAL')
 return {'engine':engine,'guard':guard,'all_other_anchored_numeric_and_existing_test_owners_byte_identical':True}


def validate_schedule(schedule,recipe=None):
 schedule_owner.validate_schedule(schedule)
 need(schedule['rows']==354550 and schedule['logical_batches']==3694 and schedule['batch_rows']==96
  and schedule['optimizer_updates']==3694 and schedule['skipped_batches']==0 and all(v is True for v in schedule['update_flags'])
  and schedule['descriptor']['manifest']==MANIFEST,'EXACT_WHOLE_SUPPORTED_MEMBERSHIP')
 if recipe is not None:
  manifest=read(MANIFEST)
  need(schedule['descriptor']==schedule_owner.schedule_descriptor(schedule['canonical_schedule'],recipe,manifest,MANIFEST),'EXACT_MEMBERSHIP_RECIPE_DESCRIPTOR')
 return schedule


def resolve_batches(context,schedule,counts):
 from gx1.contracts.market_state_ta_integrated_objective_v1 import _native_future_state_truth
 validate_schedule(schedule)
 need(_CACHE not in context and type(counts.get('order_array_loads'))is int and counts['order_array_loads']==0,'ONE_ORDER_LOAD_AND_ACTIVITY_COMPUTATION')
 binding=schedule['descriptor']['order'];learning.resume_reference(binding)
 counts['order_array_loads']+=1;values=np.load(binding['path'],allow_pickle=False)
 activity=_native_future_state_truth(context['truth'],rows=len(context['source_rows']),fold_upper_ns=train.UPPER,target_contract=context['target_contract'])['activity']
 need(isinstance(activity,np.ndarray)and activity.dtype==np.bool_ and activity.shape==(354550,5),'ACTUAL_ACTIVITY_MASK')
 activity.flags.writeable=False
 batches=schedule_owner.resolve_batches(context,schedule,values,activity)
 context[_CACHE]={'schedule_sha256':schedule['contract_sha256'],'activity':activity}
 return batches


def step_binding(context,schedule,index,take,packet):
 cache=context.get(_CACHE)
 need(type(cache)is dict and set(cache)=={'schedule_sha256','activity'}and cache['schedule_sha256']==schedule['contract_sha256']
  and not cache['activity'].flags.writeable,'BOUND_READONLY_ACTIVITY_CACHE')
 return schedule_owner.step_binding(context,schedule,index,take,packet,cache['activity'])


def validate_prefix(schedule,steps,cursor):return schedule_owner.validate_prefix(schedule,steps,cursor)
def cursor_binding(schedule,cursor):return schedule_owner.cursor_binding(schedule,cursor)


def _design(design,original):
 expected={'schema_version':'gx1_source_row_membership_full_epoch_implementation_design_v1','name':NAME,'module':MODULE,
  'stage':STAGE,'recovery_stage':RECOVERY_STAGE,'plan_path':str(PLAN),'output_root':str(ROOT),'source_anchor_commit':ANCHOR,
  'schemas':{'main':MAIN_SCHEMA,'resume':RESUME_SCHEMA},'canonical_fullfit_plan':CANONICAL,'preparation_scope':PREPARATION,
  'packet_acceptance':PACKET_ACCEPTANCE,'technical_acceptance':TECHNICAL_ACCEPTANCE,'clock_schedule':CLOCK,
  'membership_manifest':MANIFEST,'initial_acceptance':INITIAL_ACCEPTANCE,'recipe':original['recipe'],
  'execution':original['execution'],'limits':{**original['limits'],'order_array_loads':1},'fullfit_change_scope':_CHANGE,'review_protocol':REVIEW_PROTOCOL,
  'execution_authorized':False,'training_authorized':False,'test_authorized':False,'fullfit_admitted':False,'additional_epochs_allowed':False}
 need(set(design)==set(expected)and all(type(design.get(k))is type(v)and design[k]==v for k,v in expected.items()),'LOCKED_IMPLEMENTATION_DESIGN')
 learning.resume_reference(design['review_protocol'])


def _review(protocol,original,initial):
 meta=original['metadata'];refs=protocol['primary_references']
 need(protocol['schema_version']=='gx1_source_row_membership_comparison_protocol_v1'
  and protocol['name']=='ROW_MEMBERSHIP_ONE_FIXED_COMPARISON_BEFORE_FIT'
  and type(protocol['primary_witnesses'])is int and protocol['primary_witnesses']==30
  and type(protocol['secondary_groups'])is int and protocol['secondary_groups']==37
  and set(refs)=={'own_INITIAL','prior','current_blockorder'}
  and all(protocol[k]is False for k in('execution_authorized','training_authorized','test_authorized','step15_auto_approved','all20_approved','edge_proved')),'LOCKED_REVIEW_PROTOCOL')
 expected={'dataset':meta['dataset'],'prior':meta['prior'],'normalization':meta['normalization'],
  'input_binding':meta['input_binding_ref'],'truth_sha256':meta['truth_sha256'],
  'initial_receipt':original['inputs']['baseline_initial_receipt'],
  'model_contract_sha256':original['model_contract']['contract_sha256'],'original_plan':CANONICAL}
 need(protocol['required_binding_identities']==expected and refs['prior']==meta['prior']
  and refs['own_INITIAL']==original['inputs']['baseline_result']and initial['plan']==learning.ALL_TF_BASE,'REVIEW_EXACT_INITIAL_PRIOR_INPUTS')
 expected_groups=[tf+'|'+str(year)for tf in train.TF for year in range(2021,2026)]
 expected_groups += ['D1|'+support+'|'+str(year)for support in('v1_known','newly_known')for year in('pooled',2021,2022,2023,2024,2025)]
 need(protocol['secondary_group_ids']==expected_groups
  and protocol['evaluation']=={'rows':354550,'timeframes':list(train.TF),'batch_rows':256,'batches':1385,'last_batch_rows':246,
   'UTC_target_start_years':[2021,2022,2023,2024,2025],'D1_support_partitions':['v1_known','newly_known'],
   'allselectedrows_and_masks_retained':True,'unknown_side_remains_unknown':True},'REVIEW_ALL_30_37_WHOLE_POPULATION')
 old=read(refs['current_blockorder']);quality=read(protocol['current_blockorder_quality'])
 need(old['status']=='COMPLETE_NONE_BLOCK_ORDER_TRAIN_COMPARISON'and old['source_inputs_policy_boot_unchanged']is True
  and old['test_data_used']is False and old['report']['original_plan']==CANONICAL
  and old['report']['actual_optimizer_updates']==3694 and old['report']['final_quality']==protocol['current_blockorder_quality']
  and quality['rows']==354550 and quality['truth_sha256']==meta['truth_sha256']
  and quality['output_sha256']==old['report']['whole_FINAL_score']['output_sha256'],'REVIEW_AUTHENTIC_CURRENT_BLOCKORDER')
 need(protocol['extra_secondary_context']['chronological_None']=={'result':original['inputs']['chronological_None_result'],
  'quality':original['inputs']['chronological_None_quality']},'CHRONOLOGICAL_NONE_SECONDARY_ONLY')
 return protocol


def build_plan(runtime,source_commit=None):
 core.future_cuda_require_runtime(runtime);need(source_commit in(None,runtime['source_commit']),'RUNTIME_SOURCE')
 design_ref=DESIGN;design=read(design_ref);original=read(CANONICAL);_design(design,original)
 preparation=read(PREPARATION);acceptance=read(TECHNICAL_ACCEPTANCE);packet_acceptance=read(PACKET_ACCEPTANCE);schedule=read(CLOCK);manifest=read(MANIFEST)
 need(preparation['actual_3plus1_technical_acceptance']==TECHNICAL_ACCEPTANCE and preparation['actual_CPU_clock_packet_acceptance']==PACKET_ACCEPTANCE
  and preparation['locked_membership_manifest']==MANIFEST and preparation['original_full_epoch_plan']==CANONICAL
  and all(preparation[k]is False for k in('execution_authorized','fullfit_admitted','training_authorized','test_authorized','additional_epochs_allowed','all20_approved')),'CLOSED_PREPARATION_PROVENANCE')
 need(acceptance['schema_version']=='gx1_source_row_membership_technical_acceptance_v1'and acceptance['technical_mechanics_qualified']is True
  and acceptance['matched_INITIAL003_reused']is True and len(acceptance['all10_proofs'])==10 and all(v is True for v in acceptance['all10_proofs'].values())
  and acceptance['actual_nonzero_moments_at_C2']['backbone']>0 and acceptance['actual_nonzero_moments_at_C2']['readout']>0
  and acceptance['source_input_plan_policy_and_both_boot_bookends_pass']is True and acceptance['postrun_sources_inputs_bothboots_and_stopped_producers_pass']is True
  and all(acceptance[k]is False for k in('fullfit_admitted','training_authorized','test_authorized','all20_approved','new_grad_nograd_parity_qualified')),'ACTUAL_UNCHANGED_CANDIDATE_COLD_MECHANICS')
 technical=read(acceptance['plan']);pa,packet_result,clock_ref,packet_schedule=qualified.packet_metadata(PACKET_ACCEPTANCE)
 need(pa==packet_acceptance and clock_ref==CLOCK and packet_schedule==schedule and technical['schedule']==schedule
  and technical['inputs']['packet_acceptance']==PACKET_ACCEPTANCE and technical['inputs']['original_epoch_plan']==CANONICAL
  and technical['recipe']==original['recipe']and technical['model_contract']==original['model_contract']
  and original['backbone_gradient_policy']is technical['backbone_gradient_policy']is None,'AUTHENTIC_QUALIFIED_CANDIDATE')
 validate_schedule(schedule,original['recipe']);need(schedule['canonical_schedule']==original['schedule']['canonical_schedule'],'SAME_CANONICAL_FULL_CLOCK')
 initial_acceptance=read(INITIAL_ACCEPTANCE);base=read(learning.ALL_TF_BASE);initial=read(initial_acceptance['initial_receipt']);initial_result=read(initial_acceptance['result'])
 qualified.replay.all_tf_require_segmented_initial_metadata(initial_acceptance,initial_result,base,initial)
 need(initial==original['initial_reference']==technical['initial_reference']==initial_acceptance['initializer']
  and initial['initial_only']is True and initial['optimizer_updates']==0
  and initial['model_state_sha256']=='55ce06ac8df8fdf3892003d571a3902ffa9616909edb26d8a60372bd09432fb9'
  and initial['rng_sha256']=='343bc9e441680c73e99ed7fe89595ed2d423ac91b7f2071cb19d74043494305d','ONLY_QUALIFIED_INITIAL003_NOT_TECHNICAL_WEIGHTS')
 meta=copy.deepcopy(original['metadata']);meta['runtime']=copy.deepcopy(runtime)
 need(all(meta[k]==base['metadata'][k]for k in base['metadata']if k!='runtime')
  and all(runtime[k]==base['metadata']['runtime'][k]for k in('library_versions','gpu'))
  and all(runtime[k]==technical['metadata']['runtime'][k]for k in('library_versions','gpu')),'SAME_INITIAL_INPUTS_TARGET_NORM_PRIOR_RUNTIME')
 from gx1.contracts.market_state_ta_integrated_model_v1 import native_future_state_model_contract
 contract=native_future_state_model_contract(meta['input_binding'],architecture=meta['architecture'],train_prior=read(meta['prior']),target_contract=meta['target_contract'],backbone_gradient_policy=None)
 need(contract==original['model_contract']==base['model_contract']==technical['model_contract']
  and contract['contract_sha256']=='6d6ac22819e54c7b12aebcf4e1afe27b990d8a4d90d16500e352c7353d99cca0','SAME_ONLINE_MODEL_FUNCTION')
 review=read(design['review_protocol']);_review(review,original,initial)
 inputs=copy.deepcopy(original['inputs']);inputs.update(copy.deepcopy(packet_result['report']['inputs']))
 inputs.update(canonical_fullfit_plan=CANONICAL,fullfit_preparation_scope=PREPARATION,row_membership_fit_design=design_ref,
  row_membership_review_protocol=design['review_protocol'],row_membership_review_helper=ref(REVIEW_HELPER),membership_manifest=MANIFEST,clock_schedule=CLOCK,
  packet_acceptance=PACKET_ACCEPTANCE,packet_result=pa['result'],technical_acceptance=TECHNICAL_ACCEPTANCE,
  technical_plan=acceptance['plan'],technical_warm_result=acceptance['warm_result'],technical_cold_result=acceptance['cold_result'],
  baseline_plan=learning.ALL_TF_BASE,baseline_acceptance=INITIAL_ACCEPTANCE,baseline_result=initial_acceptance['result'],baseline_initial_receipt=initial_acceptance['initial_receipt'])
 need(inputs['source_row_order']==manifest['order']==schedule['descriptor']['order'],'OUTER_ORDER_INPUT')
 inputs.update(current_blockorder_result=review['primary_references']['current_blockorder'],current_blockorder_quality=review['current_blockorder_quality'])
 # The packet artifact already binds the genuine provider response physically.
 from gx1.contracts.native_future_state_provider_hlc_v1 import SOURCE_BINDINGS
 need(inputs['provider_response']==SOURCE_BINDINGS['reference_raw'],'EXPLICIT_OUTER_PROVIDER_RESPONSE')
 context_io=copy.deepcopy(original['context_IO']);need(context_io==packet_result['report']['context_IO']and context_io['archive_loads']==2 and context_io['decoded_fields']==41 and context_io['decoded_bytes']==1483917408,'UNCHANGED_CONTEXT_READERS')
 names=set(original['source_owners'])|set(base['source_owners'])|set(technical['source_owners'])|{SELF,TEST,GUARD_TEST,ENGINE_TEST}
 bridge=source_bridge(names)
 return {'schema_version':MAIN_SCHEMA,'stage':STAGE,'source_commit':source_commit or runtime['source_commit'],
  'source_owners':{name:ref(REPO/name)['sha256']for name in sorted(names)},'source_bridge':bridge,
  'metadata':meta,'model_contract':contract,'initial_reference':copy.deepcopy(initial),'inputs':inputs,
  'recipe':copy.deepcopy(original['recipe']),'schedule':copy.deepcopy(schedule),'backbone_gradient_policy':None,
  'execution':copy.deepcopy(learning.EXECUTION),'limits':copy.deepcopy(design['limits']),'context_IO':context_io,'output_root':str(ROOT),
  'train_end_checkpoint_before_score':True,'scientific_scope':{'review_protocol':design['review_protocol'],
   'only_changed_training_factor':_CHANGE,'technical_mechanics_reused':TECHNICAL_ACCEPTANCE,'technical_checkpoints_are_never_initializers_or_fit_recovery':True,
   'same_rows_can_change_D_b_weighted_mean_risk':True,'proper_scores_do_not_alone_pass_full_step15_or_edge':True,
   'all20_approved':False,'later_generalization':'UNKNOWN','economic_edge':'UNKNOWN','additional_epochs_allowed':False},
  'authority':'ONE_SEPARATELY_ADMITTED_MEMBERSHIP_COMPARISON_NO_VAL_TEST_OR_SELECTION'}


def gate(binding,admitted=False,cursor=None):
 value=learning.all_tf_gate(binding,admitted)if cursor is None else learning.all_tf_resume_gate(binding,admitted,cursor)
 value['status']=('ADMITTED_EXACT_ROW_MEMBERSHIP_ONE_EPOCH'if cursor is None else'ADMITTED_EXACT_ROW_MEMBERSHIP_ORIGINAL_SUFFIX')if admitted else'PREPARED_CLOSED_NOT_ADMITTED'
 return value


def recovery_root(sha):return ROOT/('RECOVERY_'+sha)


def require_engine_scope(scope,plan,binding,mode):
 need(scope=={'module':MODULE,'mode':mode,'plan':binding}and mode in('main','resume'),'EXACT_ENGINE_SCOPE')
 need(plan['schema_version']==(MAIN_SCHEMA if mode=='main'else RESUME_SCHEMA)and plan['stage']==(STAGE if mode=='main'else RECOVERY_STAGE)
  and plan['output_root']==str(ROOT)and plan['backbone_gradient_policy']is None and plan['execution']==learning.EXECUTION,'ENGINE_PLAN_IDENTITY')
 expected=PLAN if mode=='main'else ROOT/'RECOVERY_PLANS'/(binding['sha256']+'.json')
 need(binding['path']==str(expected),'ENGINE_PLAN_PATH');validate_schedule(plan['schedule'],plan['recipe'])
 need(plan['inputs']['clock_schedule']==CLOCK and plan['inputs']['membership_manifest']==MANIFEST
  and plan['inputs']['source_row_order']==plan['schedule']['descriptor']['order']and plan['train_end_checkpoint_before_score']is True,'ENGINE_EXACT_MEMBERSHIP_INPUTS')
 return ROOT if mode=='main'else recovery_root(binding['sha256'])


def validate_recovery(original,scope):
 keys={'schema_version','original_plan','producer_plan','producer_claim','producer_terminal','producer_admitted_policy','checkpoint_receipt','checkpoint','stop_review','backup'}
 need(type(scope)is dict and set(scope)==keys and scope['schema_version']==SCOPE_SCHEMA,'RECOVERY_SCOPE_SCHEMA')
 for key in keys-{'schema_version'}:learning.resume_reference(scope[key])
 original_ref=scope['original_plan'];need(original_ref['path']==str(PLAN)and original['schema_version']==MAIN_SCHEMA and original['stage']==STAGE and original['output_root']==str(ROOT)and original['backbone_gradient_policy']is None and original['execution']==learning.EXECUTION,'RECOVERY_ORIGINAL_IDENTITY')
 validate_schedule(original['schedule'],original['recipe']);control=read(CANONICAL)
 need(original['inputs']['canonical_fullfit_plan']==CANONICAL and original['inputs']['clock_schedule']==CLOCK
  and original['inputs']['membership_manifest']==MANIFEST and original['inputs']['technical_acceptance']==TECHNICAL_ACCEPTANCE
  and original['inputs']['row_membership_fit_design']==DESIGN and original['inputs']['row_membership_review_protocol']==REVIEW_PROTOCOL
  and original['recipe']==control['recipe']and original['schedule']==read(CLOCK),'RECOVERY_AUTHENTIC_ORIGINAL_SCHEDULE')
 initial=read(original['inputs']['baseline_initial_receipt'])
 need(original['initial_reference']==initial and initial['initial_only']is True and initial['optimizer_updates']==0
  and initial['model_state_sha256']=='55ce06ac8df8fdf3892003d571a3902ffa9616909edb26d8a60372bd09432fb9'
  and initial['model_contract_sha256']==original['model_contract']['contract_sha256']=='6d6ac22819e54c7b12aebcf4e1afe27b990d8a4d90d16500e352c7353d99cca0','RECOVERY_OWN_INITIAL_CONTRACT')
 traj=learning.trajectory(original);receipt=read(scope['checkpoint_receipt']);proof=train.require_recovery_checkpoint_proof(receipt['proof']);cursor=receipt['logical_cursor'];producer=read(scope['producer_plan']);producer_ref=scope['producer_plan'];end=original['schedule']['logical_batches'];train_end=type(cursor)is int and cursor==end
 need(set(receipt)=={'schema_version','checkpoint','original_plan','execution_plan','producer_lane','producer_runtime','logical_cursor','optimizer_updates','trajectory_sha256','proof'}and receipt['schema_version']=='gx1_native_future_state_training_checkpoint_receipt_v1'and receipt['checkpoint']==scope['checkpoint']and receipt['original_plan']==original_ref and receipt['execution_plan']==producer_ref and receipt['producer_lane']=='main'and type(cursor)is int and (train_end or(0<cursor<end and cursor%256==0))and type(receipt['optimizer_updates'])is int and receipt['optimizer_updates']==cursor,'RECOVERY_EXACT_DURABLE_CURSOR')
 producer_root=ROOT
 if producer_ref!=original_ref:
  producer_root=recovery_root(producer_ref['sha256'])
  need(producer_ref['path']==str(ROOT/'RECOVERY_PLANS'/(producer_ref['sha256']+'.json'))and producer['schema_version']==RESUME_SCHEMA and producer['stage']==RECOVERY_STAGE and producer['output_root']==str(ROOT)and producer['output_directory_prefix']=='RECOVERY_'and producer['metadata']['original']==original and producer['metadata']['original_plan']==original_ref and producer['metadata']['trajectory']==traj and producer['starting_cursor']<cursor and producer['model_contract']==original['model_contract']and producer['initial_reference']==original['initial_reference']and producer['backbone_gradient_policy']is None and producer['recipe']==original['recipe']and producer['schedule']==original['schedule']and producer['execution']==learning.EXECUTION,'RECOVERY_PREVIOUS_SUFFIX_LINEAGE')
 else:need(producer==original,'RECOVERY_ORIGINAL_PRODUCER')
 if train_end:need(producer['schema_version']in(MAIN_SCHEMA,RESUME_SCHEMA)and producer['train_end_checkpoint_before_score']is True and producer.get('score_only',False)is False,'RECOVERY_AUTHENTIC_TRAIN_END_PRODUCER')
 checkpoint_name='TRAIN_END'if train_end else'UPDATE%06d'%cursor
 learning.resume_reference(scope['checkpoint_receipt'],producer_root/(checkpoint_name+'.json'));learning.resume_reference(scope['checkpoint'],producer_root/(checkpoint_name+'.pt'))
 need(receipt['trajectory_sha256']==traj['contract_sha256']and proof['trajectory']==core.state_sha(traj)and proof['position']==core.state_sha(train.position(original['schedule'],cursor))and proof['scheduler_state']==core.state_sha(None)and proof['training']==core.state_sha(True)and proof['scores']==core.state_sha({'INITIAL':{'reused_qualified_result':original['inputs']['baseline_result'],'no_repeated_initial_forwards':True}}),'RECOVERY_AUTHENTIC_STATE_PROOFS')
 core.future_cuda_runtime_matches(producer['metadata']['runtime'],receipt['producer_runtime']);claim=read(scope['producer_claim']);terminal=read(scope['producer_terminal']);policy=read(scope['producer_admitted_policy'])
 learning.resume_reference(scope['producer_claim'],producer_root/'CLAIM.json');learning.resume_reference(scope['producer_terminal'],producer_root/'TERMINAL.json');train.require_recovery_ref(claim['policy_at_claim'],expected_path=POLICY)
 need(claim['plan']==producer_ref and claim['policy_at_claim']['sha256']==scope['producer_admitted_policy']['sha256']and claim['runtime']==receipt['producer_runtime']and claim['remaining_invocations']==0 and claim['TRAIN_diagnostic_only']is True,'RECOVERY_AUTHENTIC_PRODUCER_CLAIM')
 learning.validate_flags(policy,True);expected=gate(producer_ref,True)if producer_ref==original_ref else gate(producer_ref,True,producer['starting_cursor']);need(policy[producer['stage']]==expected,'RECOVERY_FROZEN_PRODUCER_ADMISSION')
 need(terminal['status']=='FAILED_CONSUMED'and terminal['exit_code']==1 and terminal['plan']==producer_ref and terminal['claim']==scope['producer_claim']and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False and 'result'not in terminal,'RECOVERY_FAILED_CONSUMED_PRODUCER');core.future_cpu_parent_stopped(claim['process_identity'])
 review=read(scope['stop_review']);backup=read(scope['backup'])
 need(review['execution_plan']==producer_ref and review['original_plan']==original_ref and review['process_stopped']is True and review['completed_epoch']is False and review['last_durable_cursor']==cursor and review['last_durable_checkpoint']==scope['checkpoint']and review['last_durable_receipt']==scope['checkpoint_receipt'],'RECOVERY_STOP_REVIEW')
 need(backup['schema_version']=='gx1_bound_separate_disk_backup_ack_v1'and backup['Mac_F_FULLFSYNC_and_disk_SHA_readback']is True and backup['checkpoint']==scope['checkpoint']and backup['checkpoint_receipt']==scope['checkpoint_receipt'],'RECOVERY_DURABLE_BACKUP')
 return receipt,traj,producer


def build_resume_plan(runtime,scope,source_commit=None):
 core.future_cuda_require_runtime(runtime);need(source_commit in(None,runtime['source_commit']),'RECOVERY_RUNTIME_SOURCE')
 learning.resume_reference(scope);scope_value=read(scope);original=read(scope_value['original_plan'])
 receipt,traj,producer=validate_recovery(original,scope_value);train.runtime_compatible(traj['runtime_compatibility'],runtime)
 for parent in(original,producer):
  for name,digest in parent['source_owners'].items():
   need(hashlib.sha256(subprocess.check_output(['git','show',parent['source_commit']+':'+name],cwd=REPO)).hexdigest()==digest,'RECOVERY_GIT_PRODUCER_IDENTITY')
   need(ref(REPO/name)['sha256']==digest,'RECOVERY_FROZEN_SOURCE_OWNER')
 cursor=receipt['logical_cursor'];limits,cursors=learning.all_tf_resume_limits(original,cursor,phase_recovery=True);limits['order_array_loads']=1
 inputs={**original['inputs'],**{k:v for k,v in scope_value.items()if k!='schema_version'},'recovery_scope':copy.deepcopy(scope)}
 return {'schema_version':RESUME_SCHEMA,'stage':RECOVERY_STAGE,'source_commit':source_commit or runtime['source_commit'],
  'source_owners':copy.deepcopy(original['source_owners']),'source_bridge':copy.deepcopy(original['source_bridge']),
  'metadata':{'original':original,'original_plan':scope_value['original_plan'],'receipt':receipt,'trajectory':traj,'runtime':copy.deepcopy(runtime)},
  'inputs':inputs,'context_IO':copy.deepcopy(original['context_IO']),'recipe':copy.deepcopy(original['recipe']),'schedule':copy.deepcopy(original['schedule']),'model_contract':original['model_contract'],
  'initial_reference':copy.deepcopy(original['initial_reference']),'backbone_gradient_policy':None,'starting_cursor':cursor,
  'remaining_TRAIN_rows':max(0,original['schedule']['rows']-cursor*96),'remaining_FINAL_rows':354550,
  'checkpoint_global_cursors':cursors,'execution':copy.deepcopy(learning.EXECUTION),'limits':limits,'output_root':str(ROOT),
  'output_directory_prefix':'RECOVERY_','scientific_scope':copy.deepcopy(original['scientific_scope']),
  'authority':'EXACT_ORIGINAL_MEMBERSHIP_SUFFIX_NO_NEW_EPOCH_OR_VAL_TEST_SELECTION','original_trajectory_preserved_verbatim':True,
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
 def stop(*_):raise TimeoutError('MEMBERSHIP_FIT_SIGNAL_OR_WALL')
 for sig in handlers:signal.signal(sig,stop)
 signal.setitimer(signal.ITIMER_REAL,learning.EXECUTION['internal_seconds']);claim=None
 try:
  guard=require_native_provider_hlc_all_tf_learning_execution(plan_sha256=sha,mode=mode)
  plan,binding,pref,live=inspect(sha,mode,True);need(guard['plan']==binding and guard['output_root']==str(output),'EXACT_GUARD_PLAN_ROOT')
  output.mkdir(exist_ok=False);claim=pub(output,'CLAIM.json',{'plan':binding,'policy_at_claim':pref,'runtime':live,
   'process_identity':core.future_cpu_process_identity(),'remaining_invocations':0,'TRAIN_diagnostic_only':True,'backbone_gradient_policy':None,
   **({'original_plan':plan['metadata']['original_plan'],'starting_cursor':plan['starting_cursor']}if mode=='resume'else{})})
  allowed={plan['inputs'][k]['path']for k in('feature_archive','target_arrays','source_row_order')}
  cursors=range(256,3694,256)if mode=='main'else plan['checkpoint_global_cursors']
  allowed|={str(output/name)for name in ['FINAL.pt','FINAL_LOGITS.npz']+['UPDATE%06d.pt'%j for j in cursors if j<3694]}
  if mode=='main':allowed|={str(output/'INITIAL.pt'),str(output/'TRAIN_END.pt')}
  else:
   allowed.add(plan['inputs']['checkpoint']['path'])
   if plan['starting_cursor']<3694:allowed.add(str(output/'TRAIN_END.pt'))
  active=[True]
  def audit(event,args):
   if not active[0]:return
   if event in('socket.connect','socket.getaddrinfo','urllib.Request'):raise RuntimeError('MEMBERSHIP_FIT_NETWORK_FORBIDDEN')
   if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
    name=os.fsdecode(args[0])
    if name.endswith(('.npy','.npz','.pt','.pth','.parquet','.safetensors','.pkl'))and str(Path(name).resolve())not in allowed:raise RuntimeError('MEMBERSHIP_FIT_PAYLOAD_PATH_FORBIDDEN')
  sys.addaudithook(audit)
  try:
   engine=learning.physical_all_tf if mode=='main'else learning.physical_all_tf_resume
   report=engine(plan,binding,pref,live,order_scope={'module':MODULE,'mode':mode,'plan':binding})
  finally:active[0]=False
  learning.check_sources_inputs(plan);need(ref(binding['path'])==binding and ref(POLICY)==pref,'FROZEN_PLAN_POLICY_BOOKEND')
  need(core.physical_windows_boot_id()==live['physical_windows_boot_id']and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==live['wsl_boot_id'],'BOTH_BOOT_BOOKEND')
  need(core.common.firewall_state()['denied_attempts']==0,'M1_FIREWALL')
  result=pub(output,'RESULT.json',{'status':'COMPLETE_NONE_ROW_MEMBERSHIP_TRAIN_COMPARISON','mode':mode,'plan':binding,'claim':claim,'report':report,
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
 group.add_argument('--row-membership-plan-sha256');group.add_argument('--row-membership-resume-plan-sha256');parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
 mode='main'if args.row_membership_plan_sha256 is not None else'resume';sha=args.row_membership_plan_sha256 or args.row_membership_resume_plan_sha256
 flag='--row-membership-plan-sha256'if mode=='main'else'--row-membership-resume-plan-sha256'
 need(sys.argv[1:]==[flag,sha]+(['--preflight']if args.preflight else[]),'EXACT_CLI')
 if args.preflight:
  from gx1.contracts.gx1_capped_execution_v1 import require_native_provider_hlc_all_tf_learning_execution
  guard=require_native_provider_hlc_all_tf_learning_execution(plan_sha256=sha,mode=mode);plan,binding,pref,live=inspect(sha,mode,False)
  print(json.dumps({'status':'EXACT_ROW_MEMBERSHIP_FIT_CLI_PREFLIGHT_PASS','mode':mode,'plan':binding,'policy':pref,'guard':guard,'models':0,'checkpoint_decodes':0,'NPZ_arrays_decoded':0,'NPY_arrays_decoded':0,'claim_created':False,'CUDA_initialized':False}))
 else:print(json.dumps(run(sha,mode),sort_keys=True))


if __name__=='__main__':main()
