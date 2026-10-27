"""Strict pure-direction research inside the existing guarded native campaign.

Metadata is stdlib-only. The shared campaign still owns source/physical-boot,
marker, launcher, signed hardware, receipts and the CURRENT lock. Numerical
recipe validation and model work belong after the canonical CUDA guard. This
qualification variant admits four private updates only, never forecast fit.
The recipe binds source bytes; the outer plan owns the committed Git identity,
so a policy can bind an exact recipe without a policy/recipe/commit hash cycle.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from gx1.contracts import local_random_access_campaign_v2 as campaign

PHASE = 'native_direction_research'
KIND = 'native_direction_research_window'
WINDOW_SCHEMA = 'gx1_native_direction_research_window_policy_v1'
RECIPE_SCHEMA = 'gx1_native_direction_research_launch_recipe_v1'
CURSOR_SCHEMA = 'gx1_native_direction_research_cursor_v1'
BPS_SCOPE = 'private_bps_b16_matched_initialization_resume_qualification'
BPS_CURSOR_SCHEMA = 'gx1_native_direction_bps_research_cursor_v1'
BPS_FOLD = {'fold_id':1,'prefix_rows':47602,'oof_start_row':47602,'oof_end_row':64890,
    'train_cutoff_ns':1735689600000000000,'oof_cutoff_ns':1743465600000000000}
BPS_GC_PARITY = {'path':'/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2/DIRECTION_GC_LAYOUT_REUSE_QUALIFICATION_ROOT_REVIEW_003.json',
    'sha256':'1d3347ee52f02af7afcac5d55c02532c0e2823e1ce20397633785a2512d39263'}
BPS_DESIGN = {'path':'/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2/DIRECTION_BPS_GENUINE_QUALIFICATION_SOURCE_DESIGN_001.json',
    'sha256':'a374d33275545127332b053aa3dced0430711951617288cca3419afa5b844c11'}
BPS_EXTRA_SOURCE_PATHS = ('gx1/contracts/entry_native_direction_bps_v1.py',
    'gx1/contracts/entry_native_direction_frozen_extraction_v1.py',
    'gx1/contracts/entry_native_direction_checkpoint_diagnostic_v1.py',
    'gx1/scripts/run_entry_native_direction_prefix_fit_window_v1.py',
    'gx1/scripts/run_entry_native_direction_bps_qualification_v1.py')
MODULE = 'gx1.scripts.run_unified_exit_native_candidate_window_v1'
CANONICAL_REPO = Path('/home/andre2/src/GX1_CURRENT')
REPO_SOURCE_PATHS = (
    'gx1/contracts/entry_exit_feature_surface_v1.py',
    'gx1/contracts/entry_exit_production_architecture_v1.py',
    'gx1/contracts/entry_gc_mtf_history_v1.py',
    'gx1/contracts/entry_model_native_input_normalization_v1.py',
    'gx1/contracts/entry_native_direction_prefix_normalization_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/features/basic_v1.py', 'gx1/features/entry_model_native_feature_layers_v1.py',
    'gx1/features/entry_specialist_feature_groups_v1.py', 'gx1/features/gc_native_context_v1.py',
    'gx1/features/htf_features.py', 'gx1/features/micro_structure_v1.py',
    'gx1/features/volatility_squeeze_state_v1.py',
    'gx1/models/entry_v10/entry_v10_input_normalization.py',
    'gx1/contracts/registry_hyperparameter_fit_v1.py', 'gx1/utils/artifact_primitives_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/entry_native_direction_feature_view_v1.py',
    'gx1/contracts/entry_native_direction_research_v1.py',
    'gx1/contracts/entry_native_direction_research_state_v1.py',
    'gx1/contracts/entry_model_native_train_recipe_v1.py',
    'gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py',
    'gx1/contracts/entry_model_native_signal_v1.py',
    'gx1/contracts/entry_model_native_training_objective_v1.py',
    'gx1/contracts/entry_causal_m1_outcomes_v1.py', 'gx1/contracts/entry_exit_feature_base_v1.py',
    'gx1/models/entry_v10/entry_v10_ctx_train_v3.py',
    'gx1/scripts/run_unified_exit_random_access_full_train_v1.py',
    'gx1/contracts/entry_training_precision_v1.py', 'gx1/contracts/model_state_digest_v1.py',
    'gx1/contracts/unified_exit_random_access_state_view_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
    'gx1/contracts/entry_native_direction_research_campaign_v1.py',
    'gx1/scripts/local_random_access_campaign_v2.py',
    'gx1/scripts/run_unified_exit_native_candidate_window_v1.py',
    'gx1/scripts/run_entry_native_direction_research_window_v1.py',
    'scripts/gx1_capped_run.sh', 'scripts/gx1_guarded_trainer_exec.sh',
    'scripts/collect_gx1_handover_readonly.py',
)
RUNTIME_SOURCE_PATHS = tuple('.venv/lib/python3.10/site-packages/' + name for name in (
    'torch/_tensor.py', 'torch/autograd/__init__.py', 'numpy/lib/_npyio_impl.py',
    'torch/optim/adamw.py', 'torch/optim/optimizer.py', 'torch/utils/_foreach_utils.py', 'torch/version.py',
    'torch/serialization.py'))
REQUIRED_SOURCE_PATHS = REPO_SOURCE_PATHS + RUNTIME_SOURCE_PATHS


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise campaign.RandomAccessCampaignError('NATIVE_DIRECTION_RESEARCH_' + reason)


def _seal(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    result[key] = campaign.canonical_sha256(result)
    return result


def _identity(value: Any, fields: set[str], schema: str, seal: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping) and set(value) == fields | {seal}, 'FIELDS:' + schema)
    result = dict(value)
    claimed = campaign._sha(result.pop(seal), seal)
    _require(result.get('schema_version') == schema and claimed == campaign.canonical_sha256(result), 'IDENTITY:' + schema)
    result[seal] = claimed
    return result


def _path(value: Any, label: str) -> str:
    path = campaign._absolute(value, label)
    _require(path.resolve() == path and not any(item.is_symlink() for item in (path, *path.parents)), 'PATH:' + label)
    return str(path)


def require_research_launch_recipe(binding: Mapping[str, str], *, source_repo: Path,
        source_commit: str, verify_files: bool = True) -> dict[str, Any]:
    """Verify metadata and source bytes; never import Torch or grant training."""
    ref = campaign.require_binding(binding, label='direction research recipe', verify_file=verify_files)
    _require(verify_files, 'RECIPE_REQUIRES_PHYSICAL_METADATA')
    value = campaign.read_bound_json(Path(ref['path']), ref['sha256'])
    fields = {'schema_version', 'source_repo', 'runtime_root', 'admission_stage', 'run_scope',
        'numerical_recipe', 'input_manifest', 'source_owners', 'execution_binding_path',
        'max_private_updates', 'max_model_constructions', 'max_forwards', 'max_backwards',
        'max_checkpoint_bytes', 'max_published_json_bytes', 'training_enabled',
        'full_epoch_training_allowed', 'test_data_used', 'predictive_value_claimed',
        'qualification_weights_for_main_fit'}
    bps = value.get('run_scope') == BPS_SCOPE
    if bps: fields |= {'fold','gc_layout_metadata_parity','qualification_design'}
    recipe = _identity(value, fields, RECIPE_SCHEMA, 'recipe_sha256')
    _require(Path(_path(recipe['source_repo'], 'source repo')) == source_repo == CANONICAL_REPO
        and campaign._COMMIT.fullmatch(source_commit) is not None, 'RECIPE_SOURCE')
    _path(recipe['runtime_root'], 'runtime root')
    _require(type(recipe['admission_stage']) is str and recipe['admission_stage'].startswith('dynamic_m5_direction_')
        and recipe['run_scope'] in ('private_four_update_resume_qualification', BPS_SCOPE), 'RECIPE_SCOPE')
    for key, expected in {'max_private_updates':4, 'max_model_constructions':4 if bps else 3,
            'max_forwards':6 if bps else 4, 'max_backwards':5 if bps else 4,
            'max_checkpoint_bytes':1073741824, 'max_published_json_bytes':33554432}.items():
        _require(type(recipe[key]) is int and recipe[key] == expected, 'RECIPE_BUDGET:' + key)
    for key, expected in {'training_enabled':True, 'full_epoch_training_allowed':False,
            'test_data_used':False, 'predictive_value_claimed':False, 'qualification_weights_for_main_fit':False}.items():
        _require(recipe[key] is expected, 'RECIPE_AUTHORITY:' + key)
    numerical = recipe['numerical_recipe']
    _require(isinstance(numerical, Mapping) and numerical.get('schema_version') == 'gx1_native_direction_research_recipe_v1'
        and type(numerical.get('batch_size')) is int and numerical['batch_size'] == (16 if bps else 2)
        and numerical.get('signed_bps_supervision',False) is bps
        and type(numerical.get('max_updates')) is int and numerical['max_updates'] == 2
        and numerical.get('genuine_updates_authorized') is False and numerical.get('native_launch_authorized') is False,
        'NUMERICAL_RECIPE_ROLE')
    # This verifies the declared270 seal only. Its exact source-owned numerical
    # values are checked by the270 validator inside the canonical CUDA runner.
    unsigned = {k:v for k,v in numerical.items() if k != 'contract_sha256'}
    declared = hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
    _require(numerical.get('contract_sha256') == declared, 'NUMERICAL_RECIPE_SEAL')
    campaign._sha(numerical.get('research_objective_sha256'), 'research objective')
    campaign.require_binding(recipe['input_manifest'], label='research input metadata', verify_file=True)
    _require(_path(recipe['execution_binding_path'], 'execution binding')
        == str(Path(recipe['runtime_root']) / 'EXECUTION_BINDING.json'), 'EXECUTION_BINDING_PATH')
    if bps:
        _require(recipe['fold'] == BPS_FOLD and recipe['gc_layout_metadata_parity'] == BPS_GC_PARITY
            and recipe['qualification_design'] == BPS_DESIGN, 'BPS_FIXED_INPUT_SCOPE')
        campaign.require_binding(recipe['gc_layout_metadata_parity'],label='bps GC metadata proof',verify_file=True)
        campaign.require_binding(recipe['qualification_design'],label='bps qualification design',verify_file=True)
    owners = recipe['source_owners']
    required_sources = REQUIRED_SOURCE_PATHS + (BPS_EXTRA_SOURCE_PATHS if bps else ())
    _require(isinstance(owners, Mapping) and set(owners) == set(required_sources), 'SOURCE_OWNER_SET')
    for key, item in owners.items():
        checked = campaign.require_binding(item, label='research source', verify_file=True)
        _require(checked['path'] == str(source_repo / key), 'SOURCE_OWNER_PATH:' + key)
    return recipe


def require_research_window_policy(value: Any, *, verify_files: bool = True) -> dict[str, Any]:
    fields = {'schema_version', 'recipe', 'invocation_number', 'max_invocation_seconds',
        'progress_path', 'campaign_cursor_path', 'output_directory', 'test_data_used'}
    result = _identity(value, fields, WINDOW_SCHEMA, 'policy_sha256')
    _require(type(result['invocation_number']) is int and result['invocation_number'] == 1
        and type(result['max_invocation_seconds']) is int and result['max_invocation_seconds'] == 12000
        and result['test_data_used'] is False, 'WINDOW_SCOPE')
    campaign.require_binding(result['recipe'], label='research window recipe', verify_file=verify_files)
    paths = [_path(result[k], k) for k in ('progress_path', 'campaign_cursor_path', 'output_directory')]
    _require(len(set(paths)) == 3 and Path(paths[0]).parent != Path(paths[1]).parent, 'WINDOW_PATH_COLLISION')
    return result


def require_research_invocation_execution(result: dict[str, Any], execution: Mapping[str, Any], *,
        source_repo: Path, source_commit: str) -> None:
    recipe_ref = campaign.require_binding(execution['prelaunch_manifest'], label='research recipe')
    recipe = require_research_launch_recipe(recipe_ref, source_repo=source_repo, source_commit=source_commit)
    policy_ref = campaign.require_binding(execution['train_session_manifest'], label='research window')
    window = require_research_window_policy(campaign.read_bound_json(Path(policy_ref['path']),policy_ref['sha256']))
    separator = result['launcher_argv'].index('--')
    _require(result['launcher_argv'][separator+1:] == [str(source_repo/'.venv/bin/python'),'-m',MODULE,
        '--window-policy',policy_ref['path'],'--window-policy-file-sha256',policy_ref['sha256'],
        '--progress-path',window['progress_path']], 'INVOCATION_ARGV')
    _require(result['kind'] == KIND and result['batch_size'] == recipe['numerical_recipe']['batch_size'] and result['epoch_index'] is None
        and result['invocation_number'] == 1 and result['window_index'] == 0 and result['window_count'] == 1
        and type(result['optimizer_step_budget']) is int and result['optimizer_step_budget'] == 4
        and result['expected_success_outcome'] == 'COMPLETE' and result['maximum_wall_seconds'] == 13800
        and result['checkpoint'] == {'pointer_path':window['campaign_cursor_path'],'before_mode':'GENESIS',
            'predecessor_invocation_number':None,'write_mode':'ATOMIC_UPDATE'}
        and result['progress_path'] == window['progress_path']
        and execution['prelaunch_manifest_sha256'] == recipe['recipe_sha256']
        and execution['train_session_manifest_sha256'] == window['policy_sha256']
        and window['recipe'] == recipe_ref, 'INVOCATION_BINDING')
    result['direction_research_recipe'] = recipe_ref
    result['direction_research_window'] = window


def require_research_plan(value: Any, *, verify_files: bool = True) -> dict[str, Any]:
    fields = {'schema_version','decision','campaign_id','phase','created_utc','source_repo','source_commit',
        'runtime_root','gpu_uuid','prepared_windows_boot','recipe','invocations','signed_guard_sources',
        'controller_sources','policy','authority','test_data_used'}
    result = _identity(value, fields, campaign.PLAN_SCHEMA, 'plan_sha256')
    _require(result['phase'] == PHASE and result['decision'] == 'PASS_PREPARED'
        and type(result['campaign_id']) is str and bool(result['campaign_id'])
        and result['source_repo'] == str(CANONICAL_REPO)
        and type(result['source_commit']) is str and campaign._COMMIT.fullmatch(result['source_commit']) is not None
        and type(result['gpu_uuid']) is str and result['gpu_uuid'].startswith('GPU-')
        and result['test_data_used'] is False, 'PLAN_IDENTITY')
    campaign._utc(result['created_utc'], 'research plan creation')
    runtime = Path(_path(result['runtime_root'], 'runtime root'))
    result['prepared_windows_boot'] = campaign.require_boot_identity(result['prepared_windows_boot'])
    _require(campaign.canonical_sha256(result['policy']) == campaign.canonical_sha256(campaign.campaign_hardware_policy(PHASE))
        and campaign.canonical_sha256(result['authority']) == campaign.canonical_sha256({'test':False,'promotion':False,'paper':False,'live':False,'cloud_spend':False}), 'PLAN_AUTHORITY')
    for key, keys in (('signed_guard_sources',{'runner','guard','query','certificate'}),
            ('controller_sources',{'controller','observer','campaign_cli'})):
        _require(isinstance(result[key],Mapping) and set(result[key]) == keys, 'PLAN_SOURCES:' + key)
        result[key] = {name:campaign.require_binding(item,label=key,verify_file=verify_files) for name,item in result[key].items()}
    result['recipe'] = campaign.require_binding(result['recipe'],label='research recipe',verify_file=verify_files)
    _require(isinstance(result['invocations'],list) and len(result['invocations']) == 1, 'PLAN_ONE_INVOCATION')
    result['invocations'] = [campaign.require_binding(result['invocations'][0],label='research invocation',verify_file=verify_files)]
    # Explicit absence of a selection artefact; required by the shared receipt transport.
    result['selection_artifact_sha256'] = None
    if verify_files:
        recipe = require_research_launch_recipe(result['recipe'],source_repo=CANONICAL_REPO,source_commit=result['source_commit'])
        _require(recipe['runtime_root'] == str(runtime), 'PLAN_RECIPE_RUNTIME')
        ref = result['invocations'][0]
        invocation = campaign.require_invocation(campaign.read_bound_json(Path(ref['path']),ref['sha256']),
            source_repo=CANONICAL_REPO,source_commit=result['source_commit'],verify_files=True)
        window = invocation.get('direction_research_window',{})
        _require(invocation['kind'] == KIND and invocation.get('direction_research_recipe') == result['recipe']
            and window.get('output_directory') == str(runtime/'private-research')
            and window.get('campaign_cursor_path') == str(runtime/'research-cursor'/'RESUME_CURSOR.json')
            and window.get('progress_path') == str(runtime/'progress'/'invocation-0001.json')
            and invocation['guard_log_path'] == str(runtime/'logs'/'invocation-0001.signed-guard.log'), 'PLAN_LAYOUT')
        result['checked_invocations'] = [invocation]
    return result


def require_native_direction_research_readiness(repo: Path, *, window_path: Path,
        window_file_sha256: str) -> dict[str, Any]:
    """An exact bound qualification exception; unbound readiness remains closed."""
    window = require_research_window_policy(campaign.read_bound_json(window_path,window_file_sha256))
    policy_path = repo/'NEXT_RUN_POLICY.json'
    policy = campaign.read_bound_json(policy_path,campaign.file_sha256(policy_path))
    import subprocess
    head = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    branch = subprocess.check_output(['git','-C',str(repo),'branch','--show-current'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(repo),'status','--porcelain=v1','--untracked-files=all'],text=True)
    recipe = require_research_launch_recipe(window['recipe'],source_repo=repo,source_commit=head)
    gate = policy.get(recipe['admission_stage'])
    _require(repo == CANONICAL_REPO and branch == 'work/gx1-current' and not dirty
        and policy.get('schema_version') == 'gx1_next_native_run_policy_v1'
        and policy.get('canonical_source_repo') == str(repo) and policy.get('canonical_branch') == branch
        and policy.get('training_module') == MODULE and policy.get('training_enabled') is True
        and policy.get('native_training_enabled') is True and policy.get('full_epoch_training_allowed') is False,
        'READINESS_GLOBAL')
    _require(isinstance(gate,Mapping) and gate.get('recipe') == window['recipe']
        and gate.get('run_root') == recipe['runtime_root'] and type(gate.get('max_invocations')) is int and gate['max_invocations'] == 1
        and type(gate.get('claimed_invocations')) is int and gate['claimed_invocations'] == 0
        and type(gate.get('completed_invocations')) is int and gate['completed_invocations'] == 0
        and gate.get('relaunch_allowed') is False, 'READINESS_ONCE')
    for key in ('execution_authorized','data_reads_authorized','genuine_constructor_authorized',
            'genuine_forward_authorized','genuine_backward_authorized','genuine_updates_authorized',
            'training_enabled','native_training_enabled'):
        _require(gate.get(key) is True, 'READINESS_GATE:' + key)
    for key in ('genuine_predictor_fit_authorized','native_launch_authorized','normalization_fit_authorized',
            'synthetic_test_execution_authorized','test_data_used','exit_evaluation','broker_access_allowed'):
        _require(gate.get(key) is False, 'READINESS_CLOSED_GATE:' + key)
    if recipe['run_scope'] == BPS_SCOPE:
        _require(gate.get('signed_bps_scale_fit_authorized') is True,'READINESS_BPS_SCALE')
    clock = campaign.require_binding(policy.get('gpu_clock_launcher'),label='existing GPU clock launcher')
    _require(clock == policy['gpu_clock_launcher'], 'READINESS_CLOCK')
    return {'decision':'READY_FOR_EXISTING_BOUND_CAMPAIGN_GATES','blocked_reasons':[],
        'canonical_source_repo':str(repo),'source_commit':head,'policy_sha256':campaign.file_sha256(policy_path),
        'training_started':False,'native_invocation_seconds':12000,'outer_guard_seconds':13800,
        'native_window_scope':{'research_only':True,'invocation_number':1,'optimizer_step_ceiling':4,
            'full_epoch_training_allowed':False,'window_policy_file_sha256':window_file_sha256}}


def require_native_direction_research_context(policy_path: Path, policy_file_sha256: str,
        progress_path: Path) -> tuple[dict[str,Any],dict[str,Any],dict[str,Any],dict[str,Any],dict[str,Any]]:
    with campaign.file_verification_scope():
        window = require_research_window_policy(campaign.read_bound_json(policy_path,policy_file_sha256))
        names = ('GX1_CAMPAIGN_PLAN_PATH','GX1_CAMPAIGN_PLAN_FILE_SHA256','GX1_CAMPAIGN_PLAN_SHA256','GX1_CAMPAIGN_INVOCATION_SHA256')
        _require(all(os.environ.get(name) for name in names), 'CAMPAIGN_ENV')
        plan_ref = {'path':os.environ[names[0]],'sha256':os.environ[names[1]]}
        plan = campaign.require_plan(campaign.read_bound_json(Path(plan_ref['path']),plan_ref['sha256']))
        invocation = plan['checked_invocations'][0]
        recipe = require_research_launch_recipe(window['recipe'],source_repo=CANONICAL_REPO,source_commit=plan['source_commit'])
        marker_path = Path(plan['runtime_root'])/'ACTIVE_INVOCATION.json'
        marker = campaign.read_bound_json(marker_path,campaign.file_sha256(marker_path))
        _require(plan['phase'] == PHASE and plan['plan_sha256'] == os.environ[names[2]]
            and invocation['invocation_sha256'] == os.environ[names[3]]
            and invocation['direction_research_window'] == window and plan['recipe'] == window['recipe']
            and str(progress_path) == window['progress_path'], 'WINDOW_CONTEXT')
        _require(marker.get('schema_version') == campaign.ACTIVE_SCHEMA
            and marker.get('marker_sha256') == campaign.canonical_sha256({k:v for k,v in marker.items() if k != 'marker_sha256'})
            and all(marker.get(key) == expected for key,expected in {
                'plan_sha256':plan['plan_sha256'],'invocation_sha256':invocation['invocation_sha256'],
                'invocation_number':1,'invocation_id':'invocation-0001','kind':KIND,
                'launcher_argv_sha256':invocation['launcher_argv_sha256'],'pointer_before_sha256':'GENESIS'}.items()), 'ACTIVE_MARKER')
        binding_path = Path(recipe['execution_binding_path'])
        binding = campaign.read_bound_json(binding_path,campaign.file_sha256(binding_path))
        _require(set(binding) == {'plan','source_head','policy'} and binding['plan'] == plan_ref
            and binding['source_head'] == plan['source_commit']
            and binding['policy'] == {'path':str(CANONICAL_REPO/'NEXT_RUN_POLICY.json'),
                'sha256':campaign.file_sha256(CANONICAL_REPO/'NEXT_RUN_POLICY.json')}, 'FROZEN_EXECUTION_BINDING')
        campaign.require_clean_source(plan)
        require_native_direction_research_readiness(CANONICAL_REPO,window_path=policy_path,window_file_sha256=policy_file_sha256)
        _require(not Path(window['output_directory']).exists() and not Path(window['campaign_cursor_path']).parent.exists()
            and not progress_path.exists(), 'GENESIS_NOT_FRESH')
    return window,plan,invocation,marker,recipe


def build_research_cursor(*, recipe_ref, outcome, optimizer_updates, parent_cursor,
        parent_order_sha256, full_state, result, batch_size=2) -> dict[str, Any]:
    value = _seal({'schema_version':BPS_CURSOR_SCHEMA if batch_size == 16 else CURSOR_SCHEMA,'recipe':dict(recipe_ref),'outcome':outcome,
        'optimizer_updates':optimizer_updates,'parent_cursor':parent_cursor,'parent_order_sha256':parent_order_sha256,
        'full_state':full_state,'result':result,'test_data_used':False}, 'cursor_sha256')
    return require_research_cursor(value,expected_recipe=recipe_ref,verify_files=False,batch_size=batch_size)


def require_research_cursor(value, *, expected_recipe, verify_files=True, batch_size=2) -> dict[str, Any]:
    fields = {'schema_version','recipe','outcome','optimizer_updates','parent_cursor',
        'parent_order_sha256','full_state','result','test_data_used'}
    _require(type(batch_size) is int and batch_size in (2,16),'CURSOR_BATCH_SIZE')
    result = _identity(value,fields,BPS_CURSOR_SCHEMA if batch_size == 16 else CURSOR_SCHEMA,'cursor_sha256')
    _require(result['recipe'] == expected_recipe and result['test_data_used'] is False
        and result['outcome'] in {'COMPLETE','FAILED'}, 'CURSOR_ROLE')
    for key in ('optimizer_updates','parent_cursor'):
        _require(type(result[key]) is int and 0 <= result[key] <= (2*batch_size if key=='parent_cursor' else 4), 'CURSOR_COUNTER:' + key)
    campaign._sha(result['parent_order_sha256'],'parent order')
    campaign.require_binding(result['result'],label='research terminal result',verify_file=verify_files)
    if result['full_state'] is not None:
        campaign.require_binding(result['full_state'],label='research full state',verify_file=verify_files)
    _require(result['outcome'] != 'COMPLETE' or (result['optimizer_updates'] == 4 and result['parent_cursor'] == 2*batch_size
        and result['full_state'] is not None), 'CURSOR_COMPLETE')
    return result


def build_research_progress(*, window, plan, invocation, checkpoint_pointer,
        optimizer_updates, parent_cursor, parent_order_sha256, outcome) -> dict[str,Any]:
    from datetime import datetime, timezone
    value = _seal({'schema_version':campaign.PROGRESS_SCHEMA,'plan_sha256':plan['plan_sha256'],
        'invocation_sha256':invocation['invocation_sha256'],'phase':KIND,'recipe':window['recipe'],
        'completed_units':optimizer_updates,'total_units':4,'optimizer_updates':optimizer_updates,
        'parent_cursor':parent_cursor,'parent_order_sha256':parent_order_sha256,
        'checkpoint_pointer':checkpoint_pointer,'terminal':True,'outcome':outcome,
        'observed_utc':datetime.now(timezone.utc).isoformat()},'progress_sha256')
    return require_research_progress(value,plan_sha256=plan['plan_sha256'],invocation=invocation,
        expected_selection_receipt_sha256=None,verify_file=False)


def require_research_progress(value, *, plan_sha256, invocation,
        expected_selection_receipt_sha256, verify_file=True) -> dict[str,Any]:
    fields = {'schema_version','plan_sha256','invocation_sha256','phase','recipe','completed_units','total_units',
        'optimizer_updates','parent_cursor','parent_order_sha256','checkpoint_pointer','terminal','outcome','observed_utc'}
    result = _identity(value,fields,campaign.PROGRESS_SCHEMA,'progress_sha256')
    _require(expected_selection_receipt_sha256 is None and invocation['kind'] == KIND
        and result['plan_sha256'] == plan_sha256 and result['invocation_sha256'] == invocation['invocation_sha256']
        and result['phase'] == KIND and result['recipe'] == invocation['direction_research_recipe']
        and result['terminal'] is True and result['outcome'] in {'COMPLETE','FAILED'}, 'PROGRESS_IDENTITY')
    campaign._utc(result['observed_utc'],'research progress')
    batch_size=invocation['batch_size']
    _require(type(batch_size) is int and batch_size in (2,16),'PROGRESS_BATCH_SIZE')
    for key in ('completed_units','total_units','optimizer_updates','parent_cursor'):
        _require(type(result[key]) is int and 0 <= result[key] <= (2*batch_size if key=='parent_cursor' else 4),'PROGRESS_COUNTER:' + key)
    _require(result['total_units'] == 4 and result['completed_units'] == result['optimizer_updates']
        and (result['outcome'] != 'COMPLETE' or result['optimizer_updates'] == 4 and result['parent_cursor'] == 2*batch_size), 'PROGRESS_COMPLETE')
    campaign._sha(result['parent_order_sha256'],'parent order')
    ref = campaign.require_binding(result['checkpoint_pointer'],label='research cursor',verify_file=verify_file)
    _require(ref['path'] == invocation['checkpoint']['pointer_path'],'PROGRESS_CURSOR_PATH')
    if verify_file:
        cursor = require_research_cursor(campaign.read_bound_json(Path(ref['path']),ref['sha256']),
            expected_recipe=result['recipe'],verify_files=True,batch_size=batch_size)
        _require(all(cursor[k] == result[k] for k in ('outcome','optimizer_updates','parent_cursor','parent_order_sha256')),
            'PROGRESS_DURABLE_CURSOR')
    return result
