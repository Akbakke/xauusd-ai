"""One-use native TRAIN endpoint cross: fixed inputs, no fitting or selection."""
from __future__ import annotations
import argparse, contextlib, hashlib, io, json, math, os, signal, subprocess, sys, time
from pathlib import Path
if __name__ == '__main__':
    from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
    install_native_m5_direction_io_firewall()
import numpy as np
import torch
from gx1.audit import diagnose_native_provider_hlc_readout_v1 as readout
from gx1.audit import native_future_state_provider_hlc_baseline_v1 as native
from gx1.contracts import market_state_ta_integrated_model_v1 as model_owner
from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution

saved, core, train = readout.saved, readout.core, readout.train
ref, read, reference = readout.ref, readout.read, readout.reference
REPO, POLICY = readout.REPO, readout.POLICY
NAME = 'NATIVE_PROVIDER_HLC_NONE_ENDPOINT_CROSS_20261124_002'
MODULE = 'gx1.audit.diagnose_native_provider_hlc_endpoint_cross_v1'
STAGE = NAME.lower(); PLAN = REPO/'configs/research'/(NAME+'.json'); ROOT = Path('/home/andre2/GX1_RUNS')/NAME
SELF = 'gx1/audit/diagnose_native_provider_hlc_endpoint_cross_v1.py'
TEST = 'tests/test_diagnose_native_provider_hlc_endpoint_cross_v1.py'
GUARD_TEST = 'tests/test_native_provider_hlc_endpoint_cross_guard_v1.py'
ANCHOR = '804145939a03d6d4b000fb472ae08640d9d2e8f6'
OLD_SHELL_SHA = '1c06c95eaac4752395b71b4da65f7cb21f9613b389f4c59fb97fb552723c4e5d'
DESIGN = {'path':'/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001/V2_NONE_ENDPOINT_CROSS_DESIGN004.json', 'sha256':'b7cf349d9b28b9339189aed349aa13b473979045a830aa41c471e467323267e0'}
TF = train.TF; ROWS = 354550; BATCH = 256; PANEL_ROWS = 1526
BATCH_INDICES = (138,415,692,969,1246,1384)
OUTPUTS = ('activity_logits','side_logits')
CAPTURE_FIELDS = ('issue_ns',*OUTPUTS)
EXECUTION = {'class':'trainer','memory':'20G','swap':'512M','outer_seconds':900,'internal_seconds':870,'kill_after_seconds':10,
             'numerical_threads':8,'cpu_affinity':list(range(19)),'pids_max':128,'physical_GPU_power_limit_W':200,
             'GPU_draw_max_W':250,'GPU_core_max_C':75,'GPU_memory_max_C':80,'VRAM_max_bytes':12884901888,
             'CPU_max_C':80,'CPU_draw_max_W':125,'host_available_min_GiB':8}
BUDGET = {k:0 for k in ('HTTP_calls','RNG_restores','VAL_TEST_rows','autograd_calls','backward_calls','new_labels',
                       'normalization_fits','optimizer_constructions','optimizer_restores','optimizer_updates','parameter_gradient_calculations','prior_fits')}
BUDGET.update(NPZ_archive_loads=3,NPZ_arrays_decoded=44,NPZ_decoded_bytes=1500935808,analytic_side_cross_cells=30520,checkpoint_deserializations=2,
              checkpoint_load_policy={'fallback_or_safe_globals':False,'map_location':'cpu','weights_only':True},
              model_constructions=1,model_state_installs=2,native_backbone_forwards=12,native_forward_issue_rows=3052,
              native_model_forwards=12,native_readout_forwards=12)
SHELL_EXPORTS = '  --setenv=GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS="$NATIVE_PROVIDER_HLC_ENDPOINT_CROSS" \\\n  --setenv=GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_PLAN_SHA256="$NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_PLAN_SHA256" \\\n'
ENV = {'GX1_CAPPED_MEMORY_BYTES': '21474836480', 'GX1_CAPPED_SWAP_BYTES': '536870912', 'GX1_CAPPED_TASKS_MAX': '128', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8', 'NUMEXPR_NUM_THREADS': '8', 'VECLIB_MAXIMUM_THREADS': '8', 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL': 'false', 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256': '', 'GX1_NATIVE_PROVIDER_HLC_INITIAL': 'false', 'GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256': '', 'GX1_NATIVE_PROVIDER_HLC_REPLAY': 'false', 'GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256': '', 'GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE': '', 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY': 'false', 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256': '', 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE': '', 'GX1_NATIVE_PROVIDER_HLC_LEARNING': 'false', 'GX1_NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256': '', 'GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME': 'false', 'GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256': '', 'GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE': 'false', 'GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256': '', 'GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE': 'false', 'GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256': '', 'GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX': 'false', 'GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256': '', 'GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN': 'false', 'GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256': '', 'GX1_NATIVE_FUTURE_STATE_THROUGHPUT': 'false', 'GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256': '', 'GX1_NATIVE_FUTURE_STATE_TRAINING': 'false', 'GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256': '', 'GX1_NATIVE_FUTURE_STATE_TRAINING_LANE': '', 'GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION': 'false', 'GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS': 'false', 'GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256': '', 'GX1_NATIVE_FUTURE_STATE_CUDA_LANE': '', 'GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256': '', 'GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE': '', 'GX1_TRAINER_MAX_WALL_SECONDS': '900', 'GX1_TRAINER_MODEL_MAX_WALL_SECONDS': '870', 'GX1_TRAINER_ATTENDED_STAGE_REQUIRED': 'false', 'GX1_TRAINER_GPU_INDEX': '0', 'GX1_TRAINER_GPU_MAX_POWER_LIMIT_W': '250', 'GX1_TRAINER_GPU_MAX_POWER_DRAW_W': '250', 'GX1_TRAINER_GPU_MAX_CORE_TEMP_C': '75', 'GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C': '80', 'GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB': '12288', 'GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS': '1', 'GX1_TRAINER_CPU_MAX_TEMP_C': '80', 'GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W': '125', 'GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES': '8589934592', 'GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES': '8589934592', 'GX1_POWER_BENCHMARK_SCOPE_JSON': '', 'GX1_POWER_BENCHMARK_SCOPE_SHA256': '', 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING': 'false', 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256': '', 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE': '', 'GX1_CAPPED_CLASS': 'trainer', 'GX1_CUDA_PRODUCER_GUARD': 'false', 'GX1_TRAINER_DEVICE': 'cuda', 'GX1_TRAINER_EXECUTION_MODE': 'canonical', 'BLIS_NUM_THREADS': '8', 'ARROW_NUM_THREADS': '1', 'POLARS_MAX_THREADS': '1', 'GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS': 'true', 'GX1_NATIVE_PROVIDER_HLC_ENDPOINT_CROSS_PLAN_SHA256': 'SHA'}


def need(ok, why):
    if not bool(ok): raise ValueError('PROVIDER_HLC_ENDPOINT_CROSS_'+why)


def shell_bridge(text=None):
    path = 'scripts/gx1_capped_run.sh'
    current = (REPO/path).read_text() if text is None else text
    before = subprocess.check_output(['git','show',ANCHOR+':'+path],cwd=REPO)
    need(hashlib.sha256(before).hexdigest()==OLD_SHELL_SHA,'SHELL_ANCHOR_SHA')
    pieces = []
    for section,anchor,side in (
        ('STATE','# BEGIN GX1 ALL TF LEARNING ROUTE 20261124 001 STATE\n','before'),
        ('MATCHERS','validate_target_command() {\n','before'),
        ('DISPATCH','validate_target_command() {\n','after'),
        ('NESTED','# BEGIN GX1 ALL TF LEARNING ROUTE 20261124 001 NESTED\n','before')):
        marker = 'GX1 PROVIDER HLC ENDPOINT CROSS ROUTE 20261124 001 '+section+'\n'
        start,end = '# BEGIN '+marker,'# END '+marker
        need(current.count(start)==current.count(end)==1,'SHELL_SINGLE_ADDITIVE_BLOCK')
        left = current.index(start); right = current.index(end,left)+len(end)
        pieces.append((anchor,side,current[left:right]))
    pieces.append(('  --setenv=GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING="$NATIVE_PROVIDER_HLC_ALL_TF_LEARNING" \\\n','before',SHELL_EXPORTS))
    normalized = current
    for anchor,side,piece in reversed(pieces):
        composite = piece+anchor if side=='before' else anchor+piece
        need(normalized.count(piece)==normalized.count(composite)==1,'SHELL_EXACT_INSERTION_ANCHOR')
        normalized = normalized.replace(composite,anchor,1)
    need(normalized.encode()==before,'SHELL_OLD_BYTES_UNCHANGED')
    return {'anchor':ANCHOR,'normalized_sha256':OLD_SHELL_SHA,'current_sha256':hashlib.sha256(current.encode()).hexdigest(),
            'all_old_bytes_preserved':True,'additive_fragments':5}


def validate_metadata(inputs, meta):
    """Authenticate the two endpoints without substituting checkpoint identities."""
    original = meta['original_epoch_plan']; completed = meta['None_final_plan']; result = meta['None_final_result']
    report = result['report']; trajectory = report['trajectory']; ir = meta['initial_receipt']; fr = meta['None_final_receipt']
    base = meta['initial_base_plan']; first = meta['first_fresh_plan']; closure = meta['first_fresh_closure']
    initial = meta['initial_result']; accepted = meta['initial_acceptance']; quality = meta['full_quality']
    m = original['metadata']; target = m['target_contract']; contract = original['model_contract']
    need(result['status']=='COMPLETE_NONE_FULL_POPULATION_TRAIN_DIAGNOSTIC_EPOCH'
         and result['plan']==inputs['None_final_plan'] and result['source_inputs_policy_boot_unchanged']is True
         and result['source_unchanged']is True and result['test_data_used']is False
         and report['actual_optimizer_updates']==report['logical_batches']==3694 and report['full_TRAIN_rows']==ROWS
         and report['passes']==1 and report['original_plan']==inputs['original_epoch_plan']
         and report['final_checkpoint_receipt']==inputs['None_final_receipt'] and report['final_quality']==inputs['full_quality']
         and report['final_logits']==inputs['capture'] and completed['metadata']['original']==original
         and completed['metadata']['trajectory']==trajectory,'COMPLETED_NONE_EPOCH')
    terminal = meta['completed_terminal']
    need(terminal['status']=='COMPLETE' and type(terminal['exit_code'])is int and terminal['exit_code']==0
         and terminal['result']==inputs['None_final_result'] and terminal['plan']==result['plan']
         and terminal['claim']==result['claim'],'COMPLETED_TERMINAL')
    need(original['backbone_gradient_policy']is None and completed['backbone_gradient_policy']is None
         and contract==completed['model_contract']==base['model_contract']==first['model_contract']==trajectory['model_contract']
         and contract['contract_sha256']=='6d6ac22819e54c7b12aebcf4e1afe27b990d8a4d90d16500e352c7353d99cca0'
         and contract['readout']['d_model']==128,'OWN_NONE_FUNCTION')
    need({k:v for k,v in m.items()if k!='runtime'}=={k:v for k,v in base['metadata'].items()if k!='runtime'}
         and {k:v for k,v in base['metadata'].items()if k!='runtime'}=={k:v for k,v in first['metadata'].items()if k!='runtime'}
         and base['recipe']['seed']==first['recipe']['seed']==original['recipe']['seed'],
         'SAME_INITIAL_BASE_FUNCTION')
    core.future_cuda_runtime_matches(base['metadata']['runtime'],first['metadata']['runtime'])
    need(first['schema_version']=='gx1_native_provider_hlc_all_tf_initial_segment_plan_v1'
         and first['initial_mode']=='first_fresh' and first['base_plan']==ir['plan']==inputs['initial_base_plan']
         and first['parent_terminal']is None and first['initial_receipt']is None
         and first['prefix']=={'cursor':0,'head':None,'rows_done':0},'ACTUAL_INITIAL_BASE_IDENTITY')
    need(closure['schema_version']=='gx1_native_provider_hlc_initial_segment_closure_v1'
         and closure['plan']==inputs['first_fresh_plan'] and closure['base_plan']==inputs['initial_base_plan']
         and closure['initial_receipt']==inputs['initial_receipt'] and closure['status']=='CLOSED_SEGMENT_REUSE_PROOF_PASS'
         and closure['errors']==[] and all(closure[k]is True for k in ('M1_firewall_clear','model_RNG_grad_unchanged',
             'physical_boot_unchanged','policy_unchanged','source_inputs_unchanged','wsl_boot_unchanged')),'FIRST_FRESH_CLOSURE')
    proof = closure['model_RNG_proof']; counts = closure['counts']
    need(proof['model_state_sha256']==ir['model_state_sha256'] and proof['rng_sha256']==ir['rng_sha256']
         and all(proof[k]is True for k in ('constructor_training_mode_restored','fresh_constructor_no_weight_load','gradients_none','model_RNG_grad_unchanged'))
         and counts['new_checkpoint_writes']==counts['new_checkpoint_readbacks']==1
         and all(counts[k]==0 for k in ('existing_checkpoint_loads','model_state_restores','optimizer_constructions','optimizer_steps')),
         'FRESH_CONSTRUCTOR_PROVENANCE')
    need(initial['plan']==inputs['initial_base_plan'] and initial['source_inputs_unchanged_across_segments']is True
         and initial['each_segment_policy_boot_unchanged']is True and initial['test_data_used']is False
         and initial['report']['whole_TRAIN_initialbaseline_qualified']is True and initial['report']['initial']['initial_prior_exact']is True
         and accepted['full_initialbaseline_qualified']is True and accepted['base_plan']==inputs['initial_base_plan']
         and accepted['result']==inputs['initial_result'] and accepted['initial_receipt']==inputs['initial_receipt']
         and accepted['initializer']['checkpoint']==ir['checkpoint']==inputs['initial_checkpoint']
         and initial['report']['initial_state_sha256']==ir['model_state_sha256']==report['fresh_initial_state_sha256'],
         'ACCEPTED_WHOLE_INITIAL')
    prior = objective.require_native_future_state_train_prior(meta['prior'],target_contract=target)
    dataset = meta['dataset']; truth_sha = m['truth_sha256']; issue_sha = original['schedule']['issue_ns_sha256']
    need(m['TRAIN_rows']==ROWS and issue_sha==m['TRAIN_issue_ns_sha256']==dataset['TRAIN_issue_ns_sha256']
         and prior['fit_population']['issue_rows']==ROWS and prior['fit_population']['issue_ns_sha256']==issue_sha
         and prior['fit_population']['truth_sha256']==truth_sha==quality['truth_sha256']==report['whole_FINAL_score']['truth_sha256']
         and dataset['contract']==target and report['whole_FINAL_score']['output_sha256']==quality['output_sha256']
         and quality['rows']==ROWS and quality['frozen_prior_contract_sha256']==prior['contract_sha256']==contract['train_prior_sha256'],
         'MATCHED_FULL_TRUTH_PRIOR_OUTPUT')
    need(meta['input_binding']==m['input_binding'] and m['input_binding']['input_normalization_sha256']==meta['normalization']['contract_sha256']
         and model_owner.native_future_state_model_contract(m['input_binding'],architecture=m['architecture'],train_prior=prior,
               target_contract=target,backbone_gradient_policy=None)==contract,'PHYSICAL_INPUT_FUNCTION')
    train.require_recovery_checkpoint_proof(fr['proof'])
    need(fr['checkpoint']==inputs['None_final_checkpoint'] and fr['original_plan']==inputs['original_epoch_plan']
         and fr['execution_plan']==inputs['None_final_plan'] and fr['producer_lane']=='main'
         and type(fr['logical_cursor'])is int and type(fr['optimizer_updates'])is int and fr['logical_cursor']==fr['optimizer_updates']==3694
         and fr['trajectory_sha256']==trajectory['contract_sha256'] and fr['proof']['trajectory']==core.state_sha(trajectory)
         and fr['proof']['position']==core.state_sha(train.position(original['schedule'],3694))
         and fr['proof']['model_state']==report['whole_FINAL_score']['state_sha256_before_and_after']
         and fr['proof']['rng_state']==report['whole_FINAL_score']['rng_sha256_before_and_restored'],'FINAL_RECEIPT_METADATA')
    return {'original_epoch_plan':original,'initial_base_plan':base,'initial_receipt':ir,'final_receipt':fr,'trajectory':trajectory}


def array_bytes(headers):
    return sum(math.prod(h['shape'])*np.dtype(h['dtype']).itemsize for h in headers.values())


def input_inventory(inputs, meta):
    feature = meta['source_snapshot']; dataset = meta['dataset']; quality = meta['full_quality']
    capture = {'issue_ns':{'shape':[ROWS],'dtype':'<i8','sha256':meta['original_epoch_plan']['schedule']['issue_ns_sha256']},
               **{k:{'shape':[ROWS,5],'dtype':'<f4','sha256':quality['output_sha256'][k]}for k in OUTPUTS}}
    fields = list(train.provider_hlc_truth_fields(meta['original_epoch_plan']['metadata']['target_contract']))
    roles = {'feature_archive':(feature['arrays'],sorted(feature['arrays'])),
             'target_arrays':(dataset['array_headers'],fields),'capture':(capture,list(CAPTURE_FIELDS))}
    inventory = {}
    for role,(headers,decoded) in roles.items():
        need(len(headers)=={'feature_archive':27,'target_arrays':32,'capture':3}[role] and len(decoded)=={'feature_archive':27,'target_arrays':14,'capture':3}[role], 'EXACT_INPUT_INVENTORY')
        inventory[role] = {'binding':inputs[role],'archive_bytes':reference(inputs[role]).stat().st_size,
                           'archive_inventory':sorted(headers),'array_headers':headers,'decoded_fields':decoded,
                           'decoded_bytes':array_bytes({k:headers[k]for k in decoded}),'archive_loads':1}
    need([inventory[k]['decoded_bytes']for k in roles]==[1433216758,50700650,17018400], 'EXACT_INPUT_DECODED_BYTES')
    return inventory


def checked_inputs(plan):
    for binding in plan['inputs'].values(): reference(binding)
    for item in plan['input_inventory'].values():
        need(reference(item['binding']).stat().st_size==item['archive_bytes'],'ARCHIVE_FILE_SIZE')
        saved.archive_headers(item['binding'],item['array_headers'],exact_inventory=item['archive_inventory'])
    for endpoint,role in (('initial','initial_checkpoint'),('final','None_final_checkpoint')):
        need(readout.checkpoint_zip_metadata(plan['inputs'][role])==plan['checkpoint_zip_metadata'][endpoint],'CP_ZIP_BOOKEND')


def build_plan(*,runtime,source_base_commit=None):
    d = read(DESIGN)
    need(d['schema_version']=='gx1_None_endpoint_cross_design_v1' and d['basename']==NAME and d['stage']==STAGE
         and d['plan_path']==str(PLAN) and d['output_root']==str(ROOT) and d['module']==MODULE and d['source_anchor_commit']==ANCHOR
         and d['budget']==BUDGET and d['execution']==EXECUTION and all(d[k]is False for k in ('execution_authorized','training_authorized','test_authorized'))
         and d['panel']['original_score_batch_indices_zero_based']==list(BATCH_INDICES) and d['panel']['rows']==PANEL_ROWS
         and d['panel']['batch_rows']==BATCH and d['panel']['last_batch_rows']==246 and d['panel']['expected_years']==list(range(2021,2026)), 'LOCKED_DESIGN')
    inputs = {**d['evidence'],'design':DESIGN}
    meta = {k:read(v)for k,v in inputs.items()if k!='None_final_checkpoint'}
    original = meta['original_epoch_plan']; m = original['metadata']; result = meta['None_final_result']
    inputs.update(initial_checkpoint=meta['initial_receipt']['checkpoint'],capture=result['report']['final_logits'],
                  completed_terminal=ref(Path(inputs['None_final_result']['path']).with_name('TERMINAL.json')),
                  source_snapshot=m['source_snapshot'],feature_archive=original['inputs']['feature_archive'],dataset=m['dataset'],
                  target_arrays=original['inputs']['target_arrays'],prior=m['prior'],normalization=m['normalization'],input_binding=m['input_binding_ref'])
    from gx1.contracts.native_future_state_provider_hlc_v1 import SOURCE_BINDINGS
    certificate = read(inputs['dataset'])['target_observation_certificate_ref']
    inputs['certificate'] = {k:certificate[k]for k in ('path','sha256')}; inputs['reference_response'] = SOURCE_BINDINGS['reference_raw']
    binary = {'None_final_checkpoint','initial_checkpoint','capture','feature_archive','target_arrays'}
    for role,binding in inputs.items():
        path = reference(binding)
        need(path.suffix==('.pt'if role.endswith('checkpoint')else'.npz'if role in ('capture','feature_archive','target_arrays')else'.json'),'INPUT_ROLE_SUFFIX')
    meta.update({k:read(v)for k,v in inputs.items()if k not in binary and k not in meta})
    metadata = validate_metadata(inputs,meta)
    core.future_cuda_require_runtime(runtime)
    need(all(runtime[k]==m['runtime'][k]for k in ('library_versions','gpu')),'QUALIFIED_NATIVE_RUNTIME')
    inventory = input_inventory(inputs,meta)
    bridge = shell_bridge(); completed = meta['None_final_plan']; previous = meta['stored_support_plan']
    need(len(completed['source_owners'])==80,'NUMERICAL_OWNER_INVENTORY')
    for owners in (completed['source_owners'],previous['source_owners']):
        for name,digest in owners.items():
            need((bridge['normalized_sha256']if name=='scripts/gx1_capped_run.sh'else saved.native.sha(REPO/name))==digest,'QUALIFIED_OWNER_UNCHANGED')
    names = set(previous['source_owners'])|set(completed['source_owners'])|{SELF,TEST,GUARD_TEST}
    need(all(not Path(n).is_absolute()and'..'not in Path(n).parts and Path(n).suffix in('.py','.sh')for n in names),'SOURCE_NAMES')
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    need(source_base_commit in (None,head) and runtime['source_commit']==head,'CURRENT_SOURCE_COMMIT')
    zips = {label:readout.checkpoint_zip_metadata(inputs[role])for label,role in (('initial','initial_checkpoint'),('final','None_final_checkpoint'))}
    plan = {'schema_version':'gx1_provider_hlc_none_endpoint_cross_plan_v1','stage':STAGE,'status':'PREPARED_NOT_ADMITTED',
            'plan_path':str(PLAN),'output_root':str(ROOT),'source_base_commit':head,'runtime':runtime,
            'source_owners':{name:saved.native.sha(REPO/name)for name in sorted(names)},'source_bridge':bridge,
            'inputs':inputs,'input_roles_not_JSON':sorted(binary),'metadata':metadata,'input_inventory':inventory,
            'input_budget':d['input_budget'],'checkpoint_zip_metadata':zips,'budget':BUDGET,'execution':EXECUTION,
            'model_contract':original['model_contract'],'target_contract':m['target_contract'],'panel':d['panel'],
            'parity_tolerance':core.THROUGHPUT_EVAL_TOLERANCE,'question':d['question'],'method':d['method'],'limits':d['limits']}
    validate_input_budget(plan)
    checked_inputs(plan)
    return plan


def validate_input_budget(plan):
    items = plan['input_inventory']; b = plan['input_budget']
    roles = {'feature_archive':'feature_archive','target_archive':'target_arrays','saved_FINAL_capture':'capture'}
    for key,role in roles.items():
        item = items[role]; expected = b[key]
        need(expected['archive_loads']==item['archive_loads']==1 and expected['archive_bytes']==item['archive_bytes']
             and expected['archive_inventory_fields']==len(item['archive_inventory']) and expected['decoded_fields']==len(item['decoded_fields'])
             and expected['decoded_bytes']==item['decoded_bytes'],'LOCKED_INPUT_BUDGET')
        names = expected['decoded_field_names']
        need((names=='ALL_EXACT_27_FIELDS_BOUND_IN_PLAN_HEADERS'and role=='feature_archive'and sorted(item['decoded_fields'])==item['archive_inventory'])
             or names==item['decoded_fields'],'LOCKED_INPUT_FIELD_NAMES')
    need(b['total_archive_loads']==3 and b['total_decoded_fields']==44 and b['total_decoded_bytes']==1500935808
         and b['total_archive_bytes']==611984686 and sum(v['archive_bytes']for v in items.values())==b['total_archive_bytes']
         and sum(v['decoded_bytes']for v in items.values())==b['total_decoded_bytes'],'LOCKED_INPUT_TOTALS')



def gate_expected(binding,admitted):
    return {'status':'ADMITTED_EXACT_READ_ONLY_NATIVE_ENDPOINT_CROSS'if admitted else'PREPARED_NOT_ADMITTED',
            'plan':binding,'execution_authorized':admitted,'remaining_invocations':int(admitted),'relaunch_allowed':False,
            'training_authorized':False,'test_authorized':False}


def require_scope(sha,*,preflight=False,environ=None,argv=None):
    env = os.environ if environ is None else environ
    need(type(sha)is str and len(sha)==64 and all(c in '0123456789abcdef'for c in sha),'CLI_SHA')
    expected = {k:sha if v=='SHA'else v for k,v in ENV.items()}
    need(all(env.get(k)==v for k,v in expected.items()),'EXACT_EXECUTION_ENV')
    need(not any(k.startswith('GX1_NATIVE_')and k not in expected for k in env),'UNKNOWN_NATIVE_ENV')
    wanted = [str(REPO/'.venv/bin/python'),'-m',MODULE,'--plan-sha256',sha]+(['--preflight']if preflight else[])
    need((sys.orig_argv if argv is None else argv)==wanted,'EXACT_PROCESS_ARGV')
    need(Path.cwd().resolve()==REPO and PLAN.parent.resolve()==PLAN.parent and ROOT.parent.resolve()==ROOT.parent
         and ROOT.parent.is_dir() and not ROOT.exists() and not ROOT.is_symlink(),'CANONICAL_FRESH_PATHS')
    return require_guarded_cuda_trainer_execution()


def inspect(sha,*,admitted):
    need(Path.cwd().resolve()==REPO,'CWD')
    binding = {'path':str(PLAN),'sha256':sha}; reference(binding); plan = read(binding)
    need(plan==build_plan(runtime=plan['runtime'],source_base_commit=plan['source_base_commit']) and readout.source_clean(plan),'EXACT_PLAN_SOURCE')
    pref = ref(POLICY); policy = read(pref)
    need(all(policy.get(k)is False for k in saved.FLAGS) and policy[STAGE]==gate_expected(binding,admitted),'ONE_USE_ALL_TRAINING_CLOSED')
    need(not ROOT.exists()and not ROOT.is_symlink(),'FRESH_OUTPUT')
    live = core.collect_future_cuda_runtime_metadata(); core.future_cuda_runtime_matches(plan['runtime'],live)
    return plan,binding,pref,live


def select_panel(issues, source_rows):
    need(isinstance(issues,np.ndarray) and issues.dtype==np.int64 and issues.shape==(ROWS,) and np.all(np.diff(issues)>0)
         and core.digest(issues)==train.TRAIN_SHA and np.all((issues>=train.START)&(issues<train.UPPER))
         and isinstance(source_rows,np.ndarray)and source_rows.dtype==np.int64 and source_rows.shape==(ROWS,)
         and np.all(np.diff(source_rows)>0),'FULL_CLOCK')
    batches = [np.arange(index*BATCH,min((index+1)*BATCH,ROWS),dtype=np.int64)for index in BATCH_INDICES]
    need([len(x)for x in batches]==[256]*5+[246] and sum(map(len,batches))==PANEL_ROWS,'ORIGINAL_BATCH_BOUNDARIES')
    reports = []
    for index,take,year in zip(BATCH_INDICES,batches,(*range(2021,2026),2025)):
        times = issues[take]; years = times.astype('datetime64[ns]').astype('datetime64[Y]').astype(np.int64)+1970
        need(np.all(years==year),'PRESPECIFIED_CALENDAR_COVERAGE')
        reports.append({'original_score_batch_index_zero_based':index,'start_issue_index':int(take[0]),
                        'stop_issue_index':int(take[-1])+1,'rows':len(take),'UTC_year':year,
                        'first_issue_ns':int(times[0]),'last_issue_ns':int(times[-1]),
                        'issue_ns_sha256':core.digest(times),'source_rows_sha256':core.digest(source_rows[take])})
    return batches,reports


class InputLedger:
    """Count real existing-reader decompressions; never fabricate their results."""
    def __init__(self,inventory):
        self.inventory = inventory
        self.counts = {k:{'archive_loads':0,'decoded_fields':[],'decoded_bytes':0}for k in inventory}
        self.load = np.load

    def __enter__(self):
        np.load = self.open
        return self

    def __exit__(self,*_): np.load = self.load

    def open(self,path,*args,**kwargs):
        roles = [k for k,v in self.inventory.items()if str(path)==v['binding']['path']]
        need(len(roles)==1 and not args and kwargs=={'allow_pickle':False},'NPZ_ALLOWED_LOAD')
        role = roles[0]; count = self.counts[role]; item = self.inventory[role]
        need(count['archive_loads']==0,'NPZ_ONE_LOAD_PER_ARCHIVE'); count['archive_loads']+=1
        underlying = self.load(path,**kwargs); ledger = self
        class Archive:
            @property
            def files(self): return underlying.files
            def __enter__(self): underlying.__enter__(); return self
            def __exit__(self,*exc): return underlying.__exit__(*exc)
            def __getitem__(self,key):
                need(key in item['decoded_fields']and key not in count['decoded_fields'],'NPZ_FIELD_BUDGET')
                value = underlying[key]; h = item['array_headers'][key]
                need(isinstance(value,np.ndarray)and list(value.shape)==h['shape']and value.dtype.str==h['dtype'],'NPZ_DECODED_HEADER')
                count['decoded_fields'].append(key); count['decoded_bytes']+=value.nbytes
                need(count['decoded_bytes']<=item['decoded_bytes'],'NPZ_BYTE_BUDGET')
                return value
        return Archive()

    def verify(self):
        for role,item in self.inventory.items():
            c = self.counts[role]
            need(c['archive_loads']==1 and sorted(c['decoded_fields'])==sorted(item['decoded_fields'])
                 and c['decoded_bytes']==item['decoded_bytes'],'EXACT_ACTUAL_INPUT_DECODES')
        return self.counts


@contextlib.contextmanager
def runtime_blockers():
    changed = []; attempts = []
    def forbid(owner,name):
        if not hasattr(owner,name): return
        original = getattr(owner,name)
        def blocked(*a,**k):
            attempts.append(name); raise RuntimeError('PROVIDER_HLC_ENDPOINT_CROSS_FORBIDDEN_'+name)
        changed.append((owner,name,original)); setattr(owner,name,blocked)
    try:
        for owner,names in ((torch.autograd,('grad','backward')), (torch.Tensor,('backward',)),
                            (torch.optim.Optimizer,('__init__','load_state_dict','step')), (train,('strict_restore',)),
                            (core,('restore_training_rng',)),(torch,('set_rng_state',)),
                            (torch.cuda,('set_rng_state','set_rng_state_all')), (torch.serialization,('add_safe_globals',))):
            for name in names: forbid(owner,name)
        for name in ('AdamW','Adam','SGD','RMSprop','Adagrad','Adadelta','Adamax','NAdam','RAdam','ASGD','LBFGS','SparseAdam','Rprop'):
            if hasattr(torch.optim,name):
                cls = getattr(torch.optim,name); forbid(cls,'__init__'); forbid(cls,'step')
        yield attempts
    finally:
        for owner,name,original in reversed(changed): setattr(owner,name,original)


def strict_model_install(model,state,counts):
    own = model.state_dict()
    need(set(own)==set(state) and all(isinstance(v,torch.Tensor)and v.device.type=='cpu'and not v.requires_grad
         and v.dtype==own[k].dtype and v.shape==own[k].shape and bool(torch.isfinite(v).all())for k,v in state.items()),'STRICT_MODEL_STATE')
    buffers = dict(model.named_buffers()); before_buffers = core.state_sha(buffers)
    need(all(torch.equal(value.detach().cpu(),state[name])for name,value in buffers.items()if name in own),'IMMUTABLE_MODEL_BUFFERS')
    before_rng = core.state_sha(core.training_rng())
    need(counts['model_state_installs']<2,'MODEL_INSTALL_BUDGET')
    model.load_state_dict(state,strict=True); counts['model_state_installs']+=1
    need(core.state_sha(dict(model.named_buffers()))==before_buffers,'FACTORY_BUFFERS_UNCHANGED')
    need(counts['model_state_installs']<=2 and core.state_sha(model.state_dict())==core.state_sha(state)
         and core.state_sha(core.training_rng())==before_rng and all(p.grad is None for p in model.parameters()),'STRICT_INSTALL_PROOF')
    return {'installed_model_state_sha256':core.state_sha(state),'RNG_unchanged':True,'all_gradients_none':True,'strict':True,
            'all_factory_buffers_sha256_before_and_after':before_buffers,'all_factory_buffers_unchanged':True}


def side_parameters(state):
    readout.extract_heads(state)
    return {tf:{kind:state['readout.side_heads.'+tf+'.'+kind].detach().clone()for kind in ('weight','bias')}for tf in TF}


def model_proof(model):
    need(model.training is False and all(p.grad is None for p in model.parameters()),'FROZEN_EVAL_GRADIENTS')
    return {'model_state_sha256':core.state_sha(model.state_dict()),'RNG_sha256':core.state_sha(core.training_rng()),
            'training':False,'all_gradients_none':True}


def analytic_cross(h0,h1,p0,p1,*,device):
    """Use the original per-batch H shape and native float32 Linear algebra."""
    need(isinstance(h0,torch.Tensor)and isinstance(h1,torch.Tensor)and h0.dtype==h1.dtype==torch.float32
         and h0.device.type==h1.device.type=='cpu'and h0.shape==h1.shape and h0.ndim==3
         and h0.shape[1:]==(5,128)and len(h0)in (256,246)and bool(torch.isfinite(h0).all())and bool(torch.isfinite(h1).all()),'ANALYTIC_CAPTURE_SHAPES')
    for params in (p0,p1):
        need(set(params)==set(TF)and all(set(params[tf])=={'weight','bias'}and all(
            isinstance(params[tf][k],torch.Tensor)and params[tf][k].device.type=='cpu'and params[tf][k].dtype==torch.float32
            and params[tf][k].shape==shape and not params[tf][k].requires_grad and bool(torch.isfinite(params[tf][k]).all())
            for k,shape in (('weight',(1,128)),('bias',(1,))) )for tf in TF),'ANALYTIC_PARAMETER_BINDING')
    out = {}
    with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
        hidden = [h0.to(device),h1.to(device)]
        params = [{tf:{k:v.to(device)for k,v in p[tf].items()}for tf in TF}for p in (p0,p1)]
        for hi,h in enumerate(hidden):
            for wi,p in enumerate(params):
                z = torch.stack([torch.nn.functional.linear(h[:,j],p[tf]['weight'],p[tf]['bias']).squeeze(-1)for j,tf in enumerate(TF)],dim=1)
                out['H'+str(hi)+'_W'+str(wi)] = z.cpu().numpy().copy()
        p = params[1]
        out['projection_on_initial'] = torch.stack([torch.nn.functional.linear(hidden[0][:,j],p[tf]['weight']).squeeze(-1)for j,tf in enumerate(TF)],dim=1).cpu().numpy().copy()
        out['representation_change'] = torch.stack([torch.nn.functional.linear((hidden[1]-hidden[0])[:,j],p[tf]['weight']).squeeze(-1)for j,tf in enumerate(TF)],dim=1).cpu().numpy().copy()
    return out


def saved_final_parity(outputs,batches,saved_final):
    checks = [{k:core.future_cuda_array_parity(o[k],saved_final[k][take])for k in OUTPUTS}
              for take,o in zip(batches,outputs)]
    need(len(checks)==6 and all(p['qualified']for batch in checks for p in batch.values()),'SAVED_FINAL_NATIVE_PARITY')
    return checks


def cross_parities(cross,initial,final,b0,b1):
    need(set(cross)=={'H0_W0','H0_W1','H1_W0','H1_W1','projection_on_initial','representation_change'},'CROSS_INVENTORY')
    need(b0.dtype==b1.dtype==np.float32 and b0.shape==b1.shape==(5,),'PRIOR_BIAS_SHAPE')
    diagonal = {'initial':core.future_cuda_array_parity(initial,cross['H0_W0']),
                'final':core.future_cuda_array_parity(final,cross['H1_W1'])}
    identity = core.future_cuda_array_parity(final-b0,(b1-b0)+cross['projection_on_initial']+cross['representation_change'])
    need(all(p['qualified']for p in diagonal.values()),'CROSS_NATIVE_DIAGONAL_PARITY')
    need(identity['qualified'],'CROSS_ADDITIVE_IDENTITY')
    need(np.array_equal(cross['H0_W0'],cross['H1_W0']),'ZERO_INITIAL_W_BOTH_REPRESENTATIONS')
    return diagonal,identity


def numeric_summary(arrays,h0,h1,p0,p1):
    report = {}
    for j,tf in enumerate(TF):
        b0 = float(p0[tf]['bias'][0]); b1 = float(p1[tf]['bias'][0]); w = p1[tf]['weight'].numpy().astype(np.float64)[0]
        a = h0[:,j].numpy().astype(np.float64); b = h1[:,j].numpy().astype(np.float64); da = b-a
        an = np.linalg.norm(a,axis=1); bn = np.linalg.norm(b,axis=1); wn = np.linalg.norm(w)
        dot = np.sum(a*b,axis=1); denom = an*bn; valid = denom>0
        report[tf] = {'rows':len(a),'initial_bias':b0,'final_bias':b1,'direct_bias_delta':b1-b0,
                      'initial_weight_l2':float(torch.linalg.vector_norm(p0[tf]['weight'])),'final_weight_l2':float(wn),
                      'H0_mean_l2':float(an.mean()),'H1_mean_l2':float(bn.mean()),'H1_minus_H0_mean_l2':float(np.linalg.norm(da,axis=1).mean()),
                      'H0_H1_mean_cosine':float((dot[valid]/denom[valid]).mean())if valid.any()else None,
                      'H0_H1_cosine_nonzero_rows':int(valid.sum()),
                      'projection_on_initial_mean_logit':float(arrays['projection_on_initial'][:,j].astype(np.float64).mean()),
                      'representation_change_at_final_W_mean_logit':float(arrays['representation_change'][:,j].astype(np.float64).mean()),
                      'arms':{key:{'mean_logit':float(value[:,j].astype(np.float64).mean()),
                                   'mean_probability':float(np.exp(-np.logaddexp(0.,-value[:,j].astype(np.float64))).mean())}
                              for key,value in arrays.items()if key.startswith('H')}}
        for label,h,norm in (('H0',a,an),('H1',b,bn)):
            valid = (norm*wn)>0
            report[tf][label+'_final_W_mean_cosine'] = float(((h@w)[valid]/(norm[valid]*wn)).mean())if valid.any()else None
            report[tf][label+'_final_W_cosine_nonzero_rows'] = int(valid.sum())
    return report


def publish(name,value):
    path = ROOT/name
    raw = value if isinstance(value,bytes)else(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    saved._publish_bytes(path,raw)
    binding = ref(path); need(binding['sha256']==hashlib.sha256(raw).hexdigest(),'DURABLE_BYTES_READBACK')
    return binding


def preserve_capture(arrays,progress):
    need(bool(arrays),'CAPTURE_NONEMPTY')
    stream = io.BytesIO(); np.savez_compressed(stream,**arrays)
    capture = publish('CAPTURE.npz',stream.getvalue()); stream.close()
    headers = {k:{'shape':list(v.shape),'dtype':v.dtype.str,'sha256':core.digest(v)}for k,v in arrays.items()}
    saved.archive_headers(capture,headers,exact_inventory=list(arrays))
    receipt = publish('CAPTURE.json',{'capture':capture,'arrays':headers,'actual_progress':progress,
                                    'payload_readback':'Full serialized byte SHA after immutable fsync publication; NPY header verification only, no extra ndarray decode.'})
    progress.update(capture=capture,capture_receipt=receipt)
    return capture,receipt,headers


def physical(plan,live,*,progress=None):
    progress = {}if progress is None else progress; progress['phase']='AUTHENTICATED_CONTEXT'
    counts = {k:0 for k,v in BUDGET.items()if type(v)is int}; arrays = {}; proofs = {}; parities = {}
    progress['counts'] = counts
    hooks = []; hidden = {}; outputs = {}; parameters = {}; validations = {}; batch_layouts = {}
    original = plan['metadata']['original_epoch_plan']; m = original['metadata']; meta = plan['metadata']
    try:
        with InputLedger(plan['input_inventory'])as ledger:
            context = native.load_context(m)
            saved_final = saved.load_fields(plan['inputs']['capture'],CAPTURE_FIELDS,
                                          plan['input_inventory']['capture']['array_headers'])
            decoded = ledger.verify()
        counts.update(NPZ_archive_loads=sum(c['archive_loads']for c in decoded.values()),
                      NPZ_arrays_decoded=sum(len(c['decoded_fields'])for c in decoded.values()),
                      NPZ_decoded_bytes=sum(c['decoded_bytes']for c in decoded.values()))
        progress['input_decodes'] = decoded
        need(np.array_equal(context['truth']['issue_ns'],saved_final['issue_ns'])
             and context['truth_sha256']==m['truth_sha256'],'SAME_CONTEXT_SAVED_CLOCK')
        batches, panel = select_panel(saved_final['issue_ns'],context['source_rows'])
        indices = np.concatenate(batches)
        arrays.update(issue_indices=indices,issue_ns=saved_final['issue_ns'][indices],source_rows=context['source_rows'][indices])
        prior = objective.native_future_state_train_prior_logits(context['prior'],rows=1,target_contract=m['target_contract'])
        prior_bias = prior['side_logits'].numpy()[0].copy()
        progress['phase']='ONE_NATIVE_CUDA_CONSTRUCTOR'
        runtime = core.future_cuda_seed_and_runtime({'recipe':original['recipe'],'metadata':{'runtime':live}})
        model = model_owner.build_native_future_state_model(input_normalization=context['normalization'],input_binding=m['input_binding'],
            architecture=m['architecture'],device=torch.device('cuda'),train_prior=context['prior'],target_contract=m['target_contract'],
            backbone_gradient_policy=None)
        counts['model_constructions']+=1; model.eval()
        def counter(name):
            def hook(*_):
                counts[name]+=1; need(counts[name]<=BUDGET[name],'NATIVE_FORWARD_BUDGET')
            return hook
        hooks = [obj.register_forward_pre_hook(counter(name))for obj,name in (
            (model,'native_model_forwards'),(model.backbone,'native_backbone_forwards'),(model.readout,'native_readout_forwards'))]
        with runtime_blockers()as attempts:
            for endpoint,role in (('initial','initial_checkpoint'),('final','None_final_checkpoint')):
                progress['phase']=endpoint.upper()+'_CPU_WEIGHTS_ONLY_VALIDATION'
                reference(plan['inputs'][role]); need(counts['checkpoint_deserializations']<2,'CHECKPOINT_DECODE_BUDGET')
                cp = torch.load(plan['inputs'][role]['path'],map_location='cpu',weights_only=True)
                counts['checkpoint_deserializations']+=1
                if endpoint=='initial':
                    validations[endpoint] = readout.validate_initial(cp,meta['initial_receipt'],meta['initial_base_plan'],
                                                                    plan['inputs']['initial_base_plan'],prior_bias)
                else:
                    validations[endpoint] = readout.validate_final(cp,meta['final_receipt'],original,plan['inputs']['original_epoch_plan'],
                                                                  meta['trajectory'],plan['inputs']['None_final_plan'])
                parameters[endpoint] = side_parameters(cp['model_state'])
                install = strict_model_install(model,cp['model_state'],counts)
                # Only one complete CPU checkpoint exists at a time; retain just authenticated side W,b.
                del cp
                before = model_proof(model); hidden[endpoint] = []; outputs[endpoint] = []; batch_layouts[endpoint] = []
                progress['phase']=endpoint.upper()+'_SIX_NATIVE_FORWARDS'
                for take in batches:
                    seen = []
                    def readout_input(_module,args):
                        h = args[0]
                        need(h.dtype==torch.float32 and h.device.type=='cuda'and tuple(h.shape)==(len(take),5,128),
                             'ACTUAL_READOUT_INPUT')
                        seen.append({'shape':list(h.shape),'stride':list(h.stride()),'TF_view_strides':[list(h[:,j].stride())for j in range(5)],
                                     'dtype':str(h.dtype),'device':str(h.device),'state_sha256':core.state_sha(h)})
                    hook = model.readout.register_forward_pre_hook(readout_input)
                    try:
                        counts['native_forward_issue_rows']+=len(take)
                        with torch.no_grad():
                            out,h,activation = core.future_cuda_forward(model,native.packet(context,take)['features'],
                                                                        grouped=True,equal_tf=True,capture=True)
                            need(not torch.is_grad_enabled()and activation['batch_eval_active'],'NATIVE_EVAL_NO_GRAD')
                    finally: hook.remove()
                    need(len(seen)==1 and h.dtype==torch.float32 and h.device.type=='cpu'and list(h.shape)==seen[0]['shape']
                         and list(h.stride())==seen[0]['stride']and core.state_sha(h)==seen[0]['state_sha256'],
                         'CAPTURE_IS_ACTUAL_READOUT_INPUT_LAYOUT')
                    batch_layouts[endpoint].append(seen[0]); hidden[endpoint].append(h)
                    current = {k:out[k].detach().cpu().numpy().copy()for k in OUTPUTS}; outputs[endpoint].append(current)
                    arrays[endpoint+'__hidden'] = torch.cat(hidden[endpoint]).numpy()
                    for k in OUTPUTS: arrays[endpoint+'__'+k] = np.concatenate([o[k]for o in outputs[endpoint]])
                after = model_proof(model); need(before==after,'ENDPOINT_MODEL_RNG_GRAD_UNCHANGED')
                proofs[endpoint] = {'strict_install':install,'before':before,'after':after,'unchanged':True}
                if endpoint=='initial':
                    need(all(np.array_equal(o[k],np.repeat(prior[k].numpy(),len(take),axis=0))for take,o in zip(batches,outputs[endpoint])for k in OUTPUTS),
                         'INITIAL_EXACT_FROZEN_PRIOR')
                else:
                    parities['saved_FINAL'] = saved_final_parity(outputs[endpoint],batches,saved_final)
                    progress['native_parity'] = parities
            progress['phase']='FOUR_ANALYTIC_SIDE_ARMS'
            cross_batches = []; summaries = []; diagonal = []; identities = []
            final_before = model_proof(model)
            b0 = np.array([float(parameters['initial'][tf]['bias'][0])for tf in TF],np.float32)
            b1 = np.array([float(parameters['final'][tf]['bias'][0])for tf in TF],np.float32)
            for i,take in enumerate(batches):
                cross = analytic_cross(hidden['initial'][i],hidden['final'][i],parameters['initial'],parameters['final'],device='cuda')
                counts['analytic_side_cross_cells']+=4*len(take)*5
                cross_batches.append(cross)
                for k in cross: arrays['cross__'+k] = np.concatenate([x[k]for x in cross_batches])
                check,identity = cross_parities(cross,outputs['initial'][i]['side_logits'],outputs['final'][i]['side_logits'],b0,b1)
                diagonal.append(check); identities.append(identity)
                progress['cross_diagonal_parity'] = diagonal; progress['identity_parity'] = identities
                summaries.append({'batch':panel[i],'per_TF':numeric_summary(cross,hidden['initial'][i],hidden['final'][i],parameters['initial'],parameters['final'])})
            need(final_before==model_proof(model),'CROSS_MODEL_RNG_GRAD_UNCHANGED')
            need(attempts==[],'NO_FORBIDDEN_ATTEMPTS')
        need(counts=={k:v for k,v in BUDGET.items()if type(v)is int},'EXACT_ACTUAL_BUDGET')
        need(torch.cuda.max_memory_reserved()<=EXECUTION['VRAM_max_bytes'],'VRAM_CAP')
        progress['phase']='DURABLE_CAPTURE'
        capture,receipt,headers = preserve_capture(arrays,progress)
        pooled = {k:np.concatenate([x[k]for x in cross_batches])for k in cross_batches[0]}
        return {'counts':counts,'input_decodes':decoded,'input_budget':plan['input_budget'],'panel':panel,'runtime':runtime,
                'capture':capture,'capture_receipt':receipt,'capture_array_headers':headers,'endpoint_proofs':proofs,
                'parameter_binding':validations,'native_readout_input_layouts':batch_layouts,
                'analytic_arithmetic':{'cross_arms':4,'cross_arm_cells':30520,'additional_projection_terms':2,
                                       'additional_projection_term_cells':15260,'TF_linear_evaluations':180,
                                       'semantics':'Four-arms counter excludes the two additional algebraic projection terms; all six terms are computed without NN modules or updates.'},
                'saved_FINAL_native_parity':parities['saved_FINAL'],'diagonal_parity':diagonal,'identity_parity':identities,
                'per_batch':summaries,'pooled_per_TF':numeric_summary(pooled,torch.cat(hidden['initial']),torch.cat(hidden['final']),parameters['initial'],parameters['final']),
                'cuda_peak_reserved_bytes':int(torch.cuda.max_memory_reserved()),'initial_exact_frozen_prior':True,
                'initial_full_tree_and_FINAL_all_ten_proofs_verified':True,'no_optimizer_or_RNG_restored':True,
                'functional_scope':'Fixed blind TRAIN panel. W0=0; projection-on-initial and representation-change-at-final-W are path-dependent functional terms, not unique causal percentages or historical training-order causes.',
                'later_generalization':'UNKNOWN','economic_edge':'UNKNOWN','selected_causal_fix':'UNKNOWN','all20_approved':False,'limits':plan['limits']}
    except BaseException:
        if arrays and 'capture'not in progress: preserve_capture(arrays,progress)
        raise
    finally:
        for hook in hooks: hook.remove()


@contextlib.contextmanager
def payload_boundary(plan):
    allowed = {v['path']for k,v in plan['inputs'].items()if k in plan['input_roles_not_JSON']}
    active = [True]
    def audit(event,args):
        if not active[0]: return
        if event in ('socket.connect','socket.getaddrinfo','urllib.Request'):
            raise RuntimeError('PROVIDER_HLC_ENDPOINT_CROSS_NETWORK_FORBIDDEN')
        if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
            path = Path(os.fsdecode(args[0]))
            if path.suffix in ('.npz','.pt','.pth','.parquet','.safetensors','.pkl'):
                # Output CAPTURE is published with a temporary name by the immutable writer.
                output = path.parent==ROOT and (path.name=='CAPTURE.npz'or path.name.startswith('.CAPTURE.npz.'))
                need(str(path)in allowed or output,'PAYLOAD_ALLOWLIST')
    sys.addaudithook(audit)
    try: yield
    finally: active[0] = False


def run(sha):
    began = time.monotonic(); guard = require_scope(sha)
    plan,binding,pref,live = inspect(sha,admitted=True); checked_inputs(plan)
    handover = ref(REPO/'CURRENT_HANDOVER.md'); ROOT.mkdir(exist_ok=False)
    handlers = {sig:signal.getsignal(sig)for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    claim = result = None; progress = {'phase':'CLAIM'}
    try:
        def expired(*_): raise TimeoutError('PROVIDER_HLC_ENDPOINT_CROSS_WALL')
        def stopped(sig,*_): raise InterruptedError('PROVIDER_HLC_ENDPOINT_CROSS_SIGNAL_'+str(sig))
        signal.signal(signal.SIGALRM,expired); signal.signal(signal.SIGTERM,stopped); signal.signal(signal.SIGINT,stopped)
        claim = publish('CLAIM.json',{'stage':STAGE,'plan':binding,'policy_at_claim':pref,'handover_at_claim':handover,'runtime':live,
                                    'guard':guard,'process_identity':core.future_cpu_process_identity(),'remaining_invocations':0,
                                    'relaunch_allowed':False,'training_authorized':False,'test_data_used':False})
        remaining = EXECUTION['internal_seconds']-(time.monotonic()-began); need(remaining>0,'TIME_BEFORE_PAYLOAD')
        signal.setitimer(signal.ITIMER_REAL,remaining)
        with payload_boundary(plan): report = physical(plan,live,progress=progress)
        checked_inputs(plan)
        need(readout.source_clean(plan)and ref(POLICY)==pref and ref(PLAN)==binding and ref(REPO/'CURRENT_HANDOVER.md')==handover,
             'SOURCE_INPUT_PLAN_POLICY_HANDOVER_BOOKEND')
        need(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==live['wsl_boot_id']
             and core.physical_windows_boot_id()==live['physical_windows_boot_id'],'BOTH_BOOT_BOOKEND')
        need(core.common.firewall_state()['denied_attempts']==0,'M1_FIREWALL')
        result = publish('RESULT.json',{'status':'COMPLETE_NATIVE_NONE_ENDPOINT_CROSS','stage':STAGE,'plan':binding,'claim':claim,
                                       'policy_at_claim':pref,'handover_at_claim':handover,'report':report,'source_inputs_policy_boot_unchanged':True,
                                       'source_unchanged':True,'test_data_used':False,'economic_edge_proven':False,'elapsed_seconds':time.monotonic()-began})
        terminal = publish('TERMINAL.json',{'status':'COMPLETE','exit_code':0,'stage':STAGE,'plan':binding,'claim':claim,'result':result,
                                           'remaining_invocations':0,'relaunch_allowed':False,'source_unchanged':True,'test_data_used':False})
        return {'result':result,'terminal':terminal}
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL,0)
        publish('TERMINAL.json',{'status':'FAILED_CONSUMED','exit_code':1,'stage':STAGE,'plan':binding,'claim':claim,'result':result,
                               'progress':progress,'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False,
                               'error':type(error).__name__+':'+str(error)[:2000]})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig,handler in handlers.items(): signal.signal(sig,handler)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan-sha256',required=True); parser.add_argument('--preflight',action='store_true')
    args = parser.parse_args()
    need(sys.argv[1:]==['--plan-sha256',args.plan_sha256]+(['--preflight']if args.preflight else[]),'EXACT_CLI')
    if args.preflight:
        guard = require_scope(args.plan_sha256,preflight=True)
        plan,binding,pref,live = inspect(args.plan_sha256,admitted=False)
        print(json.dumps({'status':'EXACT_NATIVE_NONE_ENDPOINT_CROSS_PREFLIGHT_PASS','plan':binding,'policy':pref,'runtime':live,'guard':guard,
                          'claim_created':False,'checkpoint_deserializations':0,'NPZ_arrays_decoded':0,'models':0,'NN_forwards':0,'CUDA_initialized':False}))
    else: print(json.dumps(run(args.plan_sha256),sort_keys=True))


if __name__ == '__main__': main()
