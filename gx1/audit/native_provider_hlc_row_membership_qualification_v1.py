"""Separately admitted three-update membership mechanics and own cold step replay."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
if __name__=='__main__':
 from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
 install_native_m5_direction_io_firewall()
import numpy as np
from gx1.audit import native_provider_hlc_replay_v1 as replay
from gx1.audit import native_provider_hlc_row_membership_schedule_v1 as order
from gx1.audit import diagnose_native_provider_hlc_endpoint_cross_v1 as endpoint
from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
train,adapter,core=replay.train,replay.adapter,replay.core
REPO,POLICY,E=replay.REPO,replay.POLICY,replay.E
NAME='NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_20261125_001';MODULE='gx1.audit.native_provider_hlc_row_membership_qualification_v1'
STAGE=NAME.lower();PLAN=REPO/'configs/research'/(NAME+'.json');ROOTS={lane:Path('/home/andre2/GX1_RUNS')/(NAME+'_'+lane)for lane in('warm','cold')}
SELF='gx1/audit/native_provider_hlc_row_membership_qualification_v1.py';TEST='tests/test_native_provider_hlc_row_membership_qualification_v1.py';GUARD_TEST='tests/test_native_provider_hlc_row_membership_qualification_guard_v1.py'
ANCHOR='c357b7ade09a50c3d3855b3c01f86f4fbe24fb44';ANCHOR_SHELL_SHA='e1bbd4d6f12e2fc475a32d54c017dbfa7103cc4ed84be46253d05591804d89d7'
DESIGN_PATH=E/'V2_NONE_SOURCE_ROW_MEMBERSHIP_TECHNICAL_DESIGN001.json';SCOPE_PATH=E/'V2_NONE_SOURCE_ROW_MEMBERSHIP_NEXT_TECHNICAL_SCOPE001.json'
SCHEDULE_SELF='gx1/audit/native_provider_hlc_row_membership_schedule_v1.py';SCHEDULE_TEST='tests/test_native_provider_hlc_row_membership_schedule_v1.py'
SCHEMA='gx1_native_provider_hlc_row_membership_qualification_plan_v1'
EXECUTION=copy.deepcopy(replay.EXECUTION)
LIMITS=copy.deepcopy(replay.LIMITS)
for key in('model_forwards','backbone_forwards','readout_forwards'):LIMITS['warm'][key]=4
LIMITS['warm'].update(backward_calls=3,autograd_calls=0)
for lane in LIMITS:LIMITS[lane]['order_array_loads']=1
PREFIX={'warm_updates':3,'cold_updates':1,'cold_restore_cursor':2,'cold_replay_logical_step':3,'prefix_rows':288,'selection':'exact_source_only_first_three_sealed_candidate_B96_no_reselection'}
ENV={**endpoint.ENV,'GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS':'false','GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_PLAN_SHA256':'','GX1_NATIVE_PROVIDER_HLC_GRADIENT_PROFILE':'false','GX1_NATIVE_PROVIDER_HLC_GRADIENT_PROFILE_PLAN_SHA256':'','GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION':'true','GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256':'SHA','GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE':'LANE','GX1_TRAINER_MAX_WALL_SECONDS':'1800','GX1_TRAINER_MODEL_MAX_WALL_SECONDS':'1770'}
SHELL_ADDITIONS=(('# BEGIN GX1 PROVIDER HLC GRADIENT PROFILE ROUTE 20261125 001 STATE\n', 'before', '# BEGIN GX1 PROVIDER HLC ROW MEMBERSHIP QUALIFICATION ROUTE 20261125 001 STATE\nNATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_MODULE=gx1.audit.native_provider_hlc_row_membership_qualification_v1\nNATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION=false\nNATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256=\nNATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE=\n# END GX1 PROVIDER HLC ROW MEMBERSHIP QUALIFICATION ROUTE 20261125 001 STATE\n'), ('validate_target_command() {\n', 'before', '# BEGIN GX1 PROVIDER HLC ROW MEMBERSHIP QUALIFICATION ROUTE 20261125 001 MATCHERS\nmatches_bound_native_provider_hlc_row_membership_qualification() {\n  if [[ $# -eq 8 ]]; then\n    [[ "$8" == --preflight ]] || return 1\n    set -- "${@:1:7}"\n  fi\n  [[ $# -eq 7 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1\n  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == "$NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_MODULE" \\\n    && "$4" == --membership-qualification-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ \\\n    && "$6" == --lane && ( "$7" == warm || "$7" == cold ) ]] || return 1\n  local profile_plan="$REPO_ROOT/configs/research/NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_20261125_001.json"\n  [[ -f "$profile_plan" && ! -L "$profile_plan" \\\n    && "$(readlink -f "$profile_plan")" == "$profile_plan" ]] || return 1\n  [[ "$(/usr/bin/sha256sum "$profile_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\n\nrequire_native_provider_hlc_row_membership_qualification_environment() {\n  local profile_nested="$1" profile_env profile_state profile_expected\n  # Reuse the declared inactive native states; never accept another route\'s authority.\n  for profile_env in ${!GX1_NATIVE_@}; do\n    case "$profile_env" in\n      GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION|GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256|GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE) continue ;;\n      GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256) profile_state=NATIVE_FUTURE_STATE_CPU_MECHANICS_SHA256 ;;\n      *) profile_state="${profile_env#GX1_}" ;;\n    esac\n    [[ "$profile_state" != *_MODULE && -v "$profile_state" \\\n      && ( "${!profile_state}" == false || -z "${!profile_state}" ) \\\n      && "${!profile_env}" == "${!profile_state}" ]] || {\n      echo "FATAL: row membership qualification forbids mixed or unknown native environment" >&2; exit 75\n    }\n  done\n  if [[ "$profile_nested" == true ]]; then\n    for profile_state in ${!NATIVE_@}; do\n      case "$profile_state" in\n        *_MODULE|NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION|NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256|NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE) continue ;;\n        NATIVE_FUTURE_STATE_CPU_MECHANICS_SHA256) profile_env=GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256 ;;\n        *) profile_env="GX1_$profile_state" ;;\n      esac\n      [[ -v "$profile_env" && "${!profile_env}" == "${!profile_state}" ]] || {\n        echo "FATAL: row membership qualification requires every inactive protected native state" >&2; exit 75\n      }\n    done\n  fi\n}\n\nconfigure_native_provider_hlc_row_membership_qualification() {\n  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \\\n    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \\\n    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo "FATAL: row membership qualification requires exact canonical CUDA containment" >&2; exit 75\n  }\n  matches_bound_native_provider_hlc_row_membership_qualification "$@" || {\n    echo "FATAL: row membership qualification command/plan is not exactly bound" >&2; exit 75\n  }\n  local profile_output="/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_20261125_001_$7"\n  [[ ! -e "$profile_output" && ! -L "$profile_output" \\\n    && -d "${profile_output%/*}" && ! -L "${profile_output%/*}" \\\n    && "$(readlink -f "${profile_output%/*}")" == "${profile_output%/*}" ]] || {\n    echo "FATAL: row membership qualification requires a fresh canonical output" >&2; exit 75\n  }\n  if [[ -n "${GX1_CAPPED_CLASS:-}${GX1_CAPPED_MEMORY_BYTES:-}${GX1_CAPPED_SWAP_BYTES:-}${GX1_CAPPED_TASKS_MAX:-}" ]]; then\n    require_native_provider_hlc_row_membership_qualification_environment true\n    [[ "${GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION:-}" == true \\\n      && "${GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256:-}" == "$5" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE:-}" == "$7" ]] || {\n      echo "FATAL: row membership qualification mode/plan differs from protected parent" >&2; exit 75\n    }\n  else\n    require_native_provider_hlc_row_membership_qualification_environment false\n    [[ "${GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION:-false}" == false \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE:-}" ]] || {\n      echo "FATAL: unprotected row membership qualification environment pollution" >&2; exit 75\n    }\n  fi\n  NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION=true\n  NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256="$5"\n  NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE="$7"\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$profile_output"\n  TRAINER_MAX_WALL_SECONDS=1800\n  TRAINER_MODEL_MAX_WALL_SECONDS=1770\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n  TRAINER_CPU_MAX_TEMP_C=80\n  TRAINER_CPU_MAX_PACKAGE_POWER_W=125\n  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592\n  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592\n}\n# END GX1 PROVIDER HLC ROW MEMBERSHIP QUALIFICATION ROUTE 20261125 001 MATCHERS\n'), ('validate_target_command() {\n', 'after', '# BEGIN GX1 PROVIDER HLC ROW MEMBERSHIP QUALIFICATION ROUTE 20261125 001 DISPATCH\n  local gradient_profile_arg gradient_profile_reference=false\n  for gradient_profile_arg in "$@"; do\n    case "$gradient_profile_arg" in\n      "$NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_MODULE"|*/gx1/audit/native_provider_hlc_row_membership_qualification_v1.py|gx1/audit/native_provider_hlc_row_membership_qualification_v1.py)\n        gradient_profile_reference=true ;;\n    esac\n  done\n  if [[ "$gradient_profile_reference" == true ]]; then\n    configure_native_provider_hlc_row_membership_qualification "$@"\n    return\n  fi\n# END GX1 PROVIDER HLC ROW MEMBERSHIP QUALIFICATION ROUTE 20261125 001 DISPATCH\n'), ('# BEGIN GX1 PROVIDER HLC GRADIENT PROFILE ROUTE 20261125 001 NESTED\n', 'before', '# BEGIN GX1 PROVIDER HLC ROW MEMBERSHIP QUALIFICATION ROUTE 20261125 001 NESTED\n  [[ "${GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION:-false}" == "$NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION" \\\n    && "${GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256:-}" == "$NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256" \\\n    && "${GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE:-}" == "$NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE" ]] || {\n    echo "FATAL: nested row membership qualification mode/plan differs from protected parent" >&2; exit 75\n  }\n  if [[ "$NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION" == true ]]; then\n    require_native_provider_hlc_row_membership_qualification_environment true\n    [[ "${GX1_CAPPED_CLASS:-}" == trainer \\\n      && "${GX1_CAPPED_MEMORY_BYTES:-}" == 21474836480 \\\n      && "${GX1_CAPPED_SWAP_BYTES:-}" == 536870912 \\\n      && "${GX1_CAPPED_TASKS_MAX:-}" == 128 \\\n      && "${GX1_CUDA_PRODUCER_GUARD:-}" == false \\\n      && "${GX1_POWER_BENCHMARK_SCOPE_JSON:-}" == "" \\\n      && "${GX1_POWER_BENCHMARK_SCOPE_SHA256:-}" == "" ]] || {\n      echo "FATAL: nested row membership qualification containment differs from protected parent" >&2; exit 75\n    }\n    for profile_state in TRAINER_DEVICE TRAINER_EXECUTION_MODE \\\n      TRAINER_MAX_WALL_SECONDS TRAINER_MODEL_MAX_WALL_SECONDS TRAINER_ATTENDED_STAGE_REQUIRED \\\n      TRAINER_GPU_INDEX TRAINER_GPU_MAX_POWER_LIMIT_W TRAINER_GPU_MAX_POWER_DRAW_W \\\n      TRAINER_GPU_MAX_CORE_TEMP_C TRAINER_GPU_MAX_MEMORY_TEMP_C TRAINER_GPU_MAX_MEMORY_USED_MIB \\\n      TRAINER_GPU_MONITOR_INTERVAL_SECONDS TRAINER_CPU_MAX_TEMP_C TRAINER_CPU_MAX_PACKAGE_POWER_W \\\n      TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES; do\n      profile_env="GX1_$profile_state"\n      [[ -v "$profile_env" && "${!profile_env}" == "${!profile_state}" ]] || {\n        echo "FATAL: nested row membership qualification deadline/hardware differs from protected parent" >&2; exit 75\n      }\n    done\n    for profile_env in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS BLIS_NUM_THREADS; do\n      [[ "${!profile_env:-}" == 8 ]] || {\n        echo "FATAL: nested row membership qualification numerical threads differ from protected parent" >&2; exit 75\n      }\n    done\n    [[ "${ARROW_NUM_THREADS:-}" == 1 && "${POLARS_MAX_THREADS:-}" == 1 ]] || {\n      echo "FATAL: nested row membership qualification auxiliary threads differ from protected parent" >&2; exit 75\n    }\n  fi\n# END GX1 PROVIDER HLC ROW MEMBERSHIP QUALIFICATION ROUTE 20261125 001 NESTED\n'), ('  --setenv=GX1_NATIVE_PROVIDER_HLC_GRADIENT_PROFILE="$NATIVE_PROVIDER_HLC_GRADIENT_PROFILE" \\\n', 'before', '  --setenv=GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION="$NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION" \\\n  --setenv=GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256="$NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_PLAN_SHA256" \\\n  --setenv=GX1_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE="$NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_QUALIFICATION_LANE" \\\n'))
ref,read,pub=replay.ref,replay.read,replay.pub


def need(ok,why):
 if not bool(ok):raise ValueError('ROW_MEMBERSHIP_QUALIFICATION_'+why)


def reference(binding):
 need(type(binding)is dict and set(binding)=={'path','sha256'}and type(binding['path'])is str,'REFERENCE_SCHEMA')
 path=Path(binding['path']);need(path.is_absolute()and path.is_file()and not path.is_symlink()and path.resolve()==path and ref(path)==binding,'REFERENCE_BYTES')
 return path


def source_bridge(names):
 result={}
 for name in sorted(names):
  need(not Path(name).is_absolute()and'..'not in Path(name).parts and Path(name).suffix in('.py','.sh'),'SOURCE_NAME')
  if name in(SELF,TEST,GUARD_TEST,SCHEDULE_SELF,SCHEDULE_TEST):continue
  before=subprocess.check_output(['git','show',ANCHOR+':'+name],cwd=REPO);current=(REPO/name).read_bytes()
  if name=='scripts/gx1_capped_run.sh':
   need(hashlib.sha256(before).hexdigest()==ANCHOR_SHELL_SHA,'ANCHOR_SHELL')
   text=current.decode()
   for anchor,position,fragment in reversed(SHELL_ADDITIONS):
    combined=fragment+anchor if position=='before'else anchor+fragment
    need(text.count(fragment)==text.count(combined)==1,'EXACT_SHELL_INSERTION');text=text.replace(combined,anchor,1)
   current=text.encode()
  need(current==before,'UNCHANGED_ANCHORED_OWNER:'+name);result[name]=hashlib.sha256(before).hexdigest()
 return {'anchor':ANCHOR,'old_owner_sha256':result,'old_numeric_and_shared_guard_bytes_unchanged':True,'only_shell_delta':'five_declared_fragments_four_blocks_three_exports'}


def packet_metadata(acceptance_ref):
 reference(acceptance_ref);a=read(acceptance_ref)
 need(a['schema_version']=='gx1_source_row_membership_packet_parity_acceptance_v1'and a['packet_parity_qualified']is True and a['native_model_admitted']is False and a['training_authorized']is False and a['test_authorized']is False,'PACKET_ACCEPTANCE')
 r=read(a['result']);report=r['report'];terminal=read(a['terminal'])
 need(r['status']=='COMPLETE_SOURCE_ROW_MEMBERSHIP_PACKET_PARITY'and r['all_bookends_unchanged']is True and report['in_memory_41_arrays_bookends_unchanged']is True and report['all9_feature_keys_exact_dtype_shape_stride_bytes']is True and report['all14_truth_fields_and_certificate_metadata_exact']is True,'ACTUAL_PACKET_PARITY')
 need(a['design']==r['design']==terminal['design']and terminal['status']=='COMPLETE'and terminal['exit_code']==0 and terminal['result']==a['result']and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False,'PACKET_DESIGN_TERMINAL_CHAIN')
 schedule_ref=report['clock_schedule'];schedule=read(schedule_ref);order.validate_schedule(schedule)
 need(a['clock_schedule']==schedule_ref,'ACCEPTED_CLOCK_SCHEDULE')
 return a,r,schedule_ref,schedule


def build_plan(runtime,packet_acceptance_ref,source_commit=None):
 from gx1.contracts.market_state_ta_integrated_model_v1 import native_future_state_model_contract
 from gx1.contracts.native_future_state_provider_hlc_v1 import SOURCE_BINDINGS
 core.future_cuda_require_runtime(runtime);need(source_commit in(None,runtime['source_commit']),'RUNTIME_SOURCE')
 design_ref=ref(DESIGN_PATH);d=read(design_ref);scope_ref=ref(SCOPE_PATH);scope=read(scope_ref)
 need(d['schema_version']=='gx1_source_row_membership_technical_design_v1'and d['name']==NAME and d['module']==MODULE and d['stage']==STAGE and d['plan_path']==str(PLAN)and d['output_roots']=={k:str(v)for k,v in ROOTS.items()}and d['execution']==EXECUTION and d['limits']==LIMITS and d['technical_prefix']==PREFIX and d['packet_acceptance']==packet_acceptance_ref and d['next_technical_scope']==scope_ref and all(d[k]is False for k in('execution_authorized','training_authorized','test_authorized','fullfit_admitted')),'LOCKED_TECHNICAL_DESIGN')
 acceptance,result,schedule_ref,schedule=packet_metadata(packet_acceptance_ref);report=result['report'];inputs=copy.deepcopy(report['inputs'])
 for role in('original_epoch_plan','feature_archive','target_arrays','dataset','prior','normalization','input_binding','provider_response','certificate','membership_manifest','source_row_order'):reference(inputs[role])
 original=read(inputs['original_epoch_plan']);meta=copy.deepcopy(original['metadata']);meta['runtime']=copy.deepcopy(runtime);recipe=copy.deepcopy(original['recipe'])
 data=read(inputs['dataset']);certificate=data['target_observation_certificate_ref']
 need(all(inputs[k]==meta[k]for k in('dataset','prior','normalization'))and inputs['input_binding']==meta['input_binding_ref']and inputs['target_arrays']==data['arrays']and inputs['certificate']=={k:certificate[k]for k in('path','sha256')}and inputs['provider_response']==SOURCE_BINDINGS['reference_raw'],'PHYSICAL_CONTEXT_AND_PROVIDER_ROLE_BINDINGS')
 manifest=read(inputs['membership_manifest']);canonical=original['schedule'].get('canonical_schedule',original['schedule'])
 need(schedule['canonical_schedule']==canonical and schedule['descriptor']==order.schedule_descriptor(canonical,recipe,manifest,inputs['membership_manifest']),'AUTHENTIC_CANDIDATE_DESCRIPTOR')
 need(scope['schema_version']=='gx1_source_row_membership_next_technical_scope_v1'and scope['membership_manifest']==inputs['membership_manifest']and scope['same_frozen_model_contract']==original['model_contract']and scope['execution_authorized']is False,'EXACT_LOCKED_CANDIDATE')
 initial_acceptance=read(scope['baseline_acceptance']);initial_result=read(initial_acceptance['result']);base=read(scope['baseline_plan']);initial=read(scope['baseline_initial_receipt_to_requalify_for_exact_reuse'])
 replay.all_tf_require_segmented_initial_metadata(initial_acceptance,initial_result,base,initial)
 need(initial==initial_acceptance['initializer']and initial['model_state_sha256']=='55ce06ac8df8fdf3892003d571a3902ffa9616909edb26d8a60372bd09432fb9'and initial['rng_sha256']=='343bc9e441680c73e99ed7fe89595ed2d423ac91b7f2071cb19d74043494305d','MATCHED_INITIAL003_STATE_RNG')
 need(all(meta[k]==base['metadata'][k]for k in base['metadata']if k!='runtime')and all(runtime[k]==base['metadata']['runtime'][k]for k in('gpu','library_versions')),'UNCHANGED_INITIAL_FUNCTION_INPUTS_RUNTIME')
 prior=read(meta['prior']);contract=native_future_state_model_contract(meta['input_binding'],architecture=meta['architecture'],train_prior=prior,target_contract=meta['target_contract'],backbone_gradient_policy=None)
 need(original['backbone_gradient_policy']is None and contract==base['model_contract']==original['model_contract']and initial['model_contract_sha256']==contract['contract_sha256']=='6d6ac22819e54c7b12aebcf4e1afe27b990d8a4d90d16500e352c7353d99cca0'and recipe['seed']==base['recipe']['seed']and recipe['train_batch_rows']==96,'EXACT_SAME_NONE_NUMERICAL_FUNCTION')
 need(all(type(x)is int and x>0 for x in schedule['batch_D_activity_all_TF'][:3])and schedule['update_flags'][:3]==[True]*3,'OWN_FIRST_THREE_SUPPORTED_NO_RESELECTION')
 context_io=copy.deepcopy(report['context_IO']);need(context_io['archive_loads']==2 and context_io['decoded_fields']==41 and context_io['decoded_bytes']==1483917408,'ORIGINAL_CONTEXT_IO')
 names=set(original['source_owners'])|set(base['source_owners'])|{SELF,TEST,GUARD_TEST,SCHEDULE_SELF,SCHEDULE_TEST,endpoint.SELF,endpoint.TEST,endpoint.GUARD_TEST}
 bridge=source_bridge(names)
 inputs.update(technical_design=design_ref,next_technical_scope=scope_ref,packet_acceptance=packet_acceptance_ref,packet_result=acceptance['result'],clock_schedule=schedule_ref,baseline_acceptance=scope['baseline_acceptance'],baseline_plan=scope['baseline_plan'],baseline_result=initial_acceptance['result'],baseline_initial_receipt=scope['baseline_initial_receipt_to_requalify_for_exact_reuse'])
 need(manifest['order']==inputs['source_row_order']and Path(inputs['source_row_order']['path']).stat().st_size>manifest['order_array']['nbytes']and manifest['order_array']['nbytes']==2836400,'ONE_BOUND_ORDER_ARRAY')
 return {'schema_version':SCHEMA,'stage':STAGE,'source_commit':source_commit or runtime['source_commit'],'source_owners':{name:ref(REPO/name)['sha256']for name in sorted(names)},'source_bridge':bridge,'metadata':meta,'model_contract':contract,'backbone_gradient_policy':None,'initial_reference':initial,'inputs':inputs,'recipe':recipe,'schedule':schedule,'technical_prefix':PREFIX,'context_IO':context_io,'order_array_header':manifest['order_array'],'execution':EXECUTION,'limits':LIMITS,'output_roots':{k:str(v)for k,v in ROOTS.items()},'authority':'TECHNICAL_ONLY_3WARM_OWN_C2_COLD_STEP3_NO_MAIN_RESUME_FIT_OR_SELECTION','new_grad_nograd_parity_qualification':False,'all20_approved':False}


def trajectory(plan):
 return train.seal({'schema_version':'gx1_row_membership_qualification_trajectory_v1','inputs':plan['inputs'],'model_contract':plan['model_contract'],'schedule':plan['schedule'],'technical_prefix':plan['technical_prefix'],'recipe':plan['recipe'],'target_contract':plan['metadata']['target_contract'],'prior_truth_sha256':plan['metadata']['truth_sha256'],'numeric_source_owners':plan['source_owners'],'runtime_compatibility':{k:plan['metadata']['runtime'][k]for k in('gpu','library_versions')}})


def gate(binding,admitted):
 return {'status':'ADMITTED_EXACT_MEMBERSHIP_TECHNICAL_REPLAY'if admitted else'PREPARED_NOT_ADMITTED','plan':binding,'relaunch_allowed':False,'main_fit_authorized':False,'full_epoch_training_allowed':False,'test_authorized':False,'lanes':{k:{'execution_authorized':admitted,'remaining_invocations':int(admitted)}for k in ROOTS}}


def check_inputs(plan):
 replay.check_sources_inputs(plan)
 for item in plan['context_IO']['archives'].values():
  reference(item['binding']);endpoint.saved.archive_headers(item['binding'],{k:item['array_headers'][k]for k in item['decoded_fields']},exact_inventory=item['archive_inventory'])
 with reference(plan['inputs']['source_row_order']).open('rb')as f:
  version=np.lib.format.read_magic(f);need(version in((1,0),(2,0)),'NPY_VERSION')
  shape,fortran,dtype=(np.lib.format.read_array_header_1_0(f)if version==(1,0)else np.lib.format.read_array_header_2_0(f));h=plan['order_array_header']
  need(list(shape)==h['shape']and dtype.str==h['dtype']and not fortran and not dtype.hasobject,'ORDER_HEADER_ONLY')


def require_scope(sha,lane,*,preflight=False,environ=None,argv=None):
 env=os.environ if environ is None else environ;need(lane in ROOTS and type(sha)is str and len(sha)==64 and all(c in'0123456789abcdef'for c in sha),'LANE_SHA')
 expected={k:sha if v=='SHA'else lane if v=='LANE'else v for k,v in ENV.items()}
 need(all(env.get(k)==v for k,v in expected.items())and not any(k.startswith('GX1_NATIVE_')and k not in expected for k in env),'EXACT_SCOPE_ENV')
 command=[str(REPO/'.venv/bin/python'),'-m',MODULE,'--membership-qualification-plan-sha256',sha,'--lane',lane]+(['--preflight']if preflight else[])
 need((sys.orig_argv if argv is None else argv)==command and Path.cwd().resolve()==REPO and PLAN.resolve()==PLAN and not PLAN.is_symlink()and ROOTS[lane].parent.resolve()==ROOTS[lane].parent and ROOTS[lane].parent.is_dir()and not ROOTS[lane].exists()and not ROOTS[lane].is_symlink(),'EXACT_CLI_FRESH_PATHS')
 return require_guarded_cuda_trainer_execution()


def warm_chain(binding,pref,plan):
 root=ROOTS['warm'];terminal=read(ref(root/'TERMINAL.json'));claim=read(terminal['claim']);result=read(terminal['result'])
 need(terminal['status']=='COMPLETE'and terminal['exit_code']==0 and terminal['plan']==binding and claim['plan']==binding and claim['policy_at_claim']==pref and claim['lane']=='warm'and result['claim']==terminal['claim']and result['plan']==binding and result['status']=='COMPLETE_MEMBERSHIP_TECHNICAL_WARM'and result['source_inputs_policy_boot_unchanged']is True,'WARM_CHAIN')
 core.future_cpu_parent_stopped(result['process_identity']);report=result['report']
 need(report['counts']==LIMITS['warm']and report['trajectory']==trajectory(plan),'WARM_EXACT_COUNTS_TRAJECTORY')
 receipt=read(report['cursor2_receipt']);reference=read(report['cursor3_proof'])
 need(report['cursor2_receipt']['path']==str(root/'CURSOR002.json')and report['cursor3_proof']['path']==str(root/'CURSOR003_PROOF.json')and receipt['checkpoint']['path']==str(root/'CURSOR002.pt')and receipt['original_plan']==receipt['execution_plan']==binding and receipt['producer_lane']=='warm'and receipt['logical_cursor']==receipt['optimizer_updates']==2 and receipt['trajectory_sha256']==trajectory(plan)['contract_sha256'],'OWN_C2_RECEIPT')
 need(ref(receipt['checkpoint']['path'])==receipt['checkpoint']and reference['nonzero_moments']==report['nonzero_moments_at_cursor3'],'OWN_WARM_STATE_BYTES')
 return result,receipt,reference


def inspect(sha,lane,admitted):
 binding={'path':str(PLAN),'sha256':sha};reference(binding);plan=read(binding)
 need(plan==build_plan(plan['metadata']['runtime'],plan['inputs']['packet_acceptance'],plan['source_commit']),'EXACT_PLAN')
 check_inputs(plan);pref=ref(POLICY);policy=read(pref);replay.validate_flags(policy,admitted);need(policy[STAGE]==gate(binding,admitted),'ONE_USE_GATE')
 need(not ROOTS[lane].exists()and not ROOTS[lane].is_symlink(),'FRESH_LANE');live=core.collect_future_cuda_runtime_metadata();core.future_cuda_runtime_matches(plan['metadata']['runtime'],live)
 if admitted and lane=='cold':warm_chain(binding,pref,plan)
 return plan,binding,pref,live


def physical(plan,binding,pref,live,lane,*,progress=None):
 import torch
 from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_future_state_model
 from gx1.contracts.market_state_ta_integrated_objective_v1 import _native_future_state_truth
 root=ROOTS[lane];counts=dict.fromkeys(LIMITS[lane],0);progress={}if progress is None else progress;progress['counts']=counts
 with endpoint.InputLedger(plan['context_IO']['archives'])as ledger:
  context=adapter.load_context(plan['metadata']);io=ledger.verify()
 counts['feature_cache_payload_loads']=counts['target_archive_loads']=1
 values=np.load(plan['inputs']['source_row_order']['path'],allow_pickle=False);counts['order_array_loads']+=1
 h=plan['order_array_header'];need(isinstance(values,np.ndarray)and values.dtype.str==h['dtype']and list(values.shape)==h['shape']and values.nbytes==h['nbytes']and core.digest(values)==h['sha256'],'ACTUAL_SINGLE_ORDER_ARRAY')
 masks=_native_future_state_truth(context['truth'],rows=len(context['truth']['issue_ns']),fold_upper_ns=train.UPPER,target_contract=plan['metadata']['target_contract']);activity=masks['activity']
 batches=order.resolve_batches(context,plan['schedule'],values,activity)[:3];need(len(batches)==3 and all(len(t)==96 for t in batches),'EXACT_THREE_B96')
 core.future_cuda_seed_and_runtime({'recipe':plan['recipe'],'metadata':{'runtime':live}})
 model=build_native_future_state_model(input_normalization=context['normalization'],input_binding=context['input_binding'],architecture=plan['metadata']['architecture'],device=torch.device('cuda'),train_prior=context['prior'],target_contract=context['target_contract'],backbone_gradient_policy=None);counts['model_constructions']=1
 need(model.research_objective==plan['model_contract']and core.state_sha(model.state_dict())==plan['initial_reference']['model_state_sha256']and core.state_sha(core.training_rng())==plan['initial_reference']['rng_sha256'],'FRESH_MATCHED_INITIAL003')
 optimizer=torch.optim.AdamW(model.parameters(),**plan['recipe']['optimizer']);counts['optimizer_constructions']=1;need(not optimizer.state,'EMPTY_INITIAL_OPTIMIZER');traj=trajectory(plan);steps=[];scores={};receipts={};extra={}
 def count(key):
  def hook(*args):counts[key]+=1;need(counts[key]<=LIMITS[lane][key],'FORWARD_BUDGET')
  return hook
 hooks=[model.register_forward_pre_hook(count('model_forwards')),model.backbone.register_forward_pre_hook(count('backbone_forwards')),model.readout.register_forward_pre_hook(count('readout_forwards'))]
 def step(index):
  packet=adapter.packet(context,batches[index]);record=order.step_binding(context,plan['schedule'],index,batches[index],packet,activity)
  entry=train.training_step(model,optimizer,packet,plan,counts,index);need(entry['skip_update']is False,'NO_SKIP_OR_RESELECTION')
  entry['order_binding']=record;steps.append(entry);order.validate_prefix(plan['schedule'],steps,index+1)
  return packet
 try:
  if lane=='warm':
   for index in range(3):
    packet=step(index)
    if index==1:
     need(steps[-1]['backbone_data_gradient_l2']>0,'REAL_SECOND_BODY_GRADIENT');extra['nonzero_moments_at_cursor2']=replay.nonzero_moments(model,optimizer)
     need(all(n>0 for n in extra['nonzero_moments_at_cursor2'].values()),'NONZERO_BODY_READOUT_ADAMW')
    if index<2:
     cp=train.make_checkpoint(model,optimizer,trajectory=traj,original_plan_ref=binding,lane=lane,cursor=index+1,steps=list(steps),scores=copy.deepcopy(scores),producer_runtime=live);receipts[index+1]=train.publish_checkpoint_receipt(root/('CURSOR%03d.pt'%(index+1)),cp,counts,execution_ref=binding)
   scores['CURSOR3']=replay.score(model,packet,plan);cp=train.make_checkpoint(model,optimizer,trajectory=traj,original_plan_ref=binding,lane=lane,cursor=3,steps=steps,scores=scores,producer_runtime=live);moments=replay.nonzero_moments(model,optimizer)
   proof=pub(root,'CURSOR003_PROOF.json',{'proof':train.checkpoint_proof(cp),'nonzero_moments':moments});extra.update(cursor1_receipt=receipts[1],cursor2_receipt=receipts[2],cursor3_proof=proof,nonzero_moments_at_cursor3=moments)
  else:
   warm,receipt,reference=warm_chain(binding,pref,plan);cp=torch.load(receipt['checkpoint']['path'],map_location='cpu',weights_only=True);counts['checkpoint_loads']=1
   order.validate_prefix(plan['schedule'],cp['steps'],2)
   train.strict_restore(model,optimizer,cp,trajectory=traj,original_plan_ref=binding,expected_cursor=2,current_runtime=live,expected_state_proof=receipt['proof']);counts['model_restores']=1;steps=copy.deepcopy(cp['steps']);scores=copy.deepcopy(cp['scores'])
   packet=step(2);scores['CURSOR3']=replay.score(model,packet,plan);cp=train.make_checkpoint(model,optimizer,trajectory=traj,original_plan_ref=binding,lane=lane,cursor=3,steps=steps,scores=scores,producer_runtime=live)
   need(train.checkpoint_proof(cp)==reference['proof'],'OWN_C3_ALL10_BITEXACT');receipt3=train.publish_checkpoint_receipt(root/'REPLAYED_CURSOR003.pt',cp,counts,execution_ref=binding);extra.update(cold_replay_bit_exact=True,replayed_checkpoint_receipt=receipt3,reference_cursor3_proof=warm['report']['cursor3_proof'],full_cursor3_proof=train.checkpoint_proof(cp))
  need(counts==LIMITS[lane]and torch.cuda.max_memory_reserved()<=EXECUTION['cuda_peak_reserved_bytes_max'],'EXACT_COUNTS_VRAM')
  return {'counts':counts,'trajectory':traj,'context_IO_actual':io,'order_array_decoded_bytes':values.nbytes,'fresh_initial_state_sha256':plan['initial_reference']['model_state_sha256'],'prefix_rows':288,'physical_updates_this_lane':counts['optimizer_steps'],'cuda_peak_reserved_bytes':int(torch.cuda.max_memory_reserved()),'new_grad_nograd_parity_qualification':False,'weights_authority':'TECHNICAL_ONLY_NEVER_FIT_INITIALIZER','scope':'first3source_only_membership_batches_not_wholeTRAIN_learning_or_edge',**extra}
 finally:
  for hook in hooks:hook.remove()


def run(sha,lane):
 began=time.monotonic();progress={'phase':'SCOPE'};claim=result=None;handlers={s:signal.getsignal(s)for s in(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
 def stop(*_):raise TimeoutError('MEMBERSHIP_TECHNICAL_SIGNAL_OR_WALL')
 try:
  for s in handlers:signal.signal(s,stop)
  signal.setitimer(signal.ITIMER_REAL,1770);require_scope(sha,lane);plan,binding,pref,live=inspect(sha,lane,True);root=ROOTS[lane];root.mkdir(exist_ok=False);identity=core.future_cpu_process_identity()
  claim=pub(root,'CLAIM.json',{'plan':binding,'policy_at_claim':pref,'lane':lane,'runtime':live,'process_identity':identity,'remaining_invocations':0,'main_fit_authorized':False})
  allowed={plan['inputs'][k]['path']for k in('feature_archive','target_arrays','source_row_order')};allowed|={str(ROOTS['warm']/('CURSOR%03d.pt'%j))for j in(1,2)};allowed.add(str(ROOTS['cold']/'REPLAYED_CURSOR003.pt'))
  def audit(event,args):
   if event in('socket.connect','socket.getaddrinfo','urllib.Request'):raise RuntimeError('MEMBERSHIP_NETWORK_FORBIDDEN')
   if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
    p=os.fsdecode(args[0])
    if p.endswith(('.npz','.npy','.pt','.pth','.parquet','.safetensors','.pkl'))and str(Path(p).resolve())not in allowed:raise RuntimeError('MEMBERSHIP_PAYLOAD_FORBIDDEN')
  sys.addaudithook(audit);progress['phase']='PHYSICAL';report=physical(plan,binding,pref,live,lane,progress=progress);check_inputs(plan)
  need(ref(PLAN)==binding and ref(POLICY)==pref and core.physical_windows_boot_id()==live['physical_windows_boot_id']and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==live['wsl_boot_id']and core.common.firewall_state()['denied_attempts']==0,'SOURCE_INPUT_PLAN_POLICY_BOTH_BOOT_BOOKEND')
  result=pub(root,'RESULT.json',{'status':'COMPLETE_MEMBERSHIP_TECHNICAL_WARM'if lane=='warm'else'COMPLETE_MEMBERSHIP_TECHNICAL_COLD','plan':binding,'claim':claim,'process_identity':identity,'report':report,'source_inputs_policy_boot_unchanged':True,'test_data_used':False,'elapsed_seconds':time.monotonic()-began})
  terminal=pub(root,'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'plan':binding,'claim':claim,'result':result,'remaining_invocations':0,'relaunch_allowed':False});return {'result':result,'terminal':terminal}
 except BaseException as error:
  if claim is not None:pub(ROOTS[lane],'TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'plan':binding,'claim':claim,'progress':progress,'remaining_invocations':0,'relaunch_allowed':False,'error':type(error).__name__+':'+str(error)[:1800]})
  raise
 finally:
  signal.setitimer(signal.ITIMER_REAL,0)
  for s,h in handlers.items():signal.signal(s,h)


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--membership-qualification-plan-sha256',required=True);parser.add_argument('--lane',required=True,choices=('warm','cold'));parser.add_argument('--preflight',action='store_true');args=parser.parse_args();sha=args.membership_qualification_plan_sha256
 if args.preflight:
  require_scope(sha,args.lane,preflight=True);plan,binding,pref,live=inspect(sha,args.lane,False)
  print(json.dumps({'status':'PREFLIGHT_MEMBERSHIP_TECHNICAL_PASS','plan':binding,'policy':pref,'lane':args.lane,'array_payloads_decoded':0,'models':0,'optimizer_updates':0,'claim_created':False}));return
 print(json.dumps(run(sha,args.lane),sort_keys=True))


if __name__=='__main__':main()
