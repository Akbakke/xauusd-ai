"""Strict metadata authority for one fixed pure-direction prefix trajectory.

This stdlib-only variant shares canonical source, fresh-boot, ACTIVE, receipt,
CUDA/capped and hardware authorities. It changes neither the private four-update
qualification nor the numerical270 recipe/state owners. No metadata PASS is a
prediction-quality or trading admission.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts import entry_native_direction_research_campaign_v1 as private

PHASE = 'native_direction_prefix_fit'
KIND = 'native_direction_prefix_fit_window'
WINDOW_SCHEMA = 'gx1_native_direction_prefix_fit_window_policy_v1'
RECIPE_SCHEMA = 'gx1_native_direction_prefix_fit_launch_recipe_v1'
CURSOR_SCHEMA = 'gx1_native_direction_prefix_fit_cursor_v1'
MULTIPASS_CURSOR_SCHEMA = 'gx1_native_direction_prefix_fit_multipass_cursor_v1'
INPUT_SCHEMA = 'gx1_native_direction_prefix_fit_input_manifest_v1'
ARCHIVE_MEMBERS = frozenset('parent_rows bar_start_ns dynamic_entry_m1_row decision_ns terminal_time_ns elapsed_wall_seconds terminal_source_row observed mid_move mid_return_bps sign'.split())
VIEW_SOURCE_KEYS = frozenset({'feature_view_design','input_qualification','raw_input_provider_plan','raw_gc_layout',
    'raw_gc_transition','direction_component_source','native_constructor_source','mtf_window_source','gc_gather_source','owner_profile_source'})
MODULE = private.MODULE
CANONICAL_REPO = private.CANONICAL_REPO
MAX_WINDOWS = 4
BATCH_SIZE = 16
BPS_SCOPE = 'single_prefix_one_pass_direction_signed_bps_fit'
CONTINUATION_SCOPE = 'single_prefix_three_pass_direction_continuation'
_CONTINUATION_ROOT = '/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_PREFIX_FIT_FOLD1_20261025_276'
CONTINUATION_ORIGIN = {
    'launch_recipe': {'path':_CONTINUATION_ROOT+'/RECIPE.json','sha256':'8ba298a148af3b923c7b4a567f98bf8ddd2eaef1e4eeb5144e2a785e9f9e2d93'},
    'terminal_state': {'path':_CONTINUATION_ROOT+'/RUNTIME/prefix-fit/states/w0002_TERMINAL_u002976_p047602.pt','sha256':'be9e9a54851085c6719f03a29d958f7c83b470236a0fc5ae5801c1bdd543af29'},
    'initial_measurement': {'path':_CONTINUATION_ROOT+'/RUNTIME/prefix-fit/window-0001/INITIAL_TRAIN.json','sha256':'0504cb783b1054a4a19e1da84a0013abee11100038cdbcd78b6514913064c09c'},
    'initial_state': {'path':_CONTINUATION_ROOT+'/RUNTIME/prefix-fit/states/INITIAL_0U.pt','sha256':'40a9d5ad744a06512619c3072170657f6ed430a077c3187e05e817fd5b46e9f3'},
    'terminal_train_measurement': {'path':_CONTINUATION_ROOT+'/RUNTIME/prefix-fit/window-0001/FINAL_TRAIN.json','sha256':'b25e7ef3539f19604dfbde58bfb6ff379dd7e0d813a464ab680cea4977578301'},
}
BPS_NATIVE_QUALIFICATION = {
    'path':'/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2/DIRECTION_BPS_NATIVE_B16_QUALIFICATION_20261027_002/ACTUAL_ROOT_REVIEW.json',
    'sha256':'07b26101961ce4218b51f5d87487239bf8b0a913c5ec93db872417e35e1293f9'}
BPS_EXTRA_SOURCE_PATHS = (
    'gx1/contracts/entry_native_direction_bps_v1.py',
    'gx1/contracts/entry_native_direction_frozen_extraction_v1.py',
    'gx1/contracts/entry_native_direction_checkpoint_diagnostic_v1.py',
)
BPS_NUMERICAL_SOURCE_PATHS = (
    'gx1/contracts/entry_native_direction_bps_v1.py',
    'gx1/contracts/entry_native_direction_feature_view_v1.py',
    'gx1/contracts/entry_native_direction_research_v1.py',
    'gx1/contracts/entry_native_direction_research_state_v1.py',
    'gx1/contracts/entry_model_native_train_recipe_v1.py',
    'gx1/contracts/entry_model_native_input_normalization_v1.py',
    'gx1/models/entry_v10/entry_v10_input_normalization.py',
    'gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py',
    'gx1/models/entry_v10/entry_v10_ctx_train_v3.py',
)
PARENT_ORDER_ALGORITHM = 'numpy.random.RandomState(MT19937).permutation'
REQUIRED_SOURCE_PATHS = private.REQUIRED_SOURCE_PATHS + (
    'gx1/contracts/entry_native_direction_prefix_fit_campaign_v1.py',
    'gx1/contracts/entry_native_direction_context_report_v1.py',
    'gx1/scripts/run_entry_native_direction_prefix_fit_window_v1.py',
)
FOLD_SPECS = {
    1: (47602, 64890, 1735689600000000000, 1743465600000000000),
    2: (64890, 82427, 1743465600000000000, 1751328000000000000),
    3: (82427, 100499, 1751328000000000000, 1759276800000000000),
    4: (100499, 118249, 1759276800000000000, 1767225600000000000),
}


def _require(ok, reason):
    if not ok:
        raise campaign.RandomAccessCampaignError('NATIVE_DIRECTION_PREFIX_FIT_' + reason)


def _seal(value, key):
    result = dict(value)
    result[key] = campaign.canonical_sha256(result)
    return result


def _identity(value, fields, schema, seal):
    _require(isinstance(value, Mapping) and set(value) == fields | {seal}, 'FIELDS:' + schema)
    result = dict(value)
    claimed = campaign._sha(result.pop(seal), seal)
    _require(result.get('schema_version') == schema and claimed == campaign.canonical_sha256(result), 'IDENTITY:' + schema)
    result[seal] = claimed
    return result


def _path(value, label):
    return private._path(value, label)


def _fold(value):
    keys = {'fold_id', 'prefix_rows', 'oof_start_row', 'oof_end_row', 'train_cutoff_ns', 'oof_cutoff_ns'}
    _require(isinstance(value, Mapping) and set(value) == keys and all(type(x) is int for x in value.values())
        and value['fold_id'] in FOLD_SPECS, 'FOLD_FIELDS')
    n, end, cut, oof_end = FOLD_SPECS[value['fold_id']]
    _require(dict(value) == {'fold_id': value['fold_id'], 'prefix_rows': n, 'oof_start_row': n,
        'oof_end_row': end, 'train_cutoff_ns': cut, 'oof_cutoff_ns': oof_end}, 'FOLD_EXACT')
    return dict(value)


def _view_identity(value, *, verify_files):
    _require(isinstance(value, Mapping) and set(value) == {'contract_sha256', 'prefix_normalization',
        'research_objective_sha256'}, 'VIEW_IDENTITY')
    prefix = value['prefix_normalization']
    _require(isinstance(prefix, Mapping) and set(prefix) == {'binding', 'contract_sha256', 'base_sha256',
        'gc_layout_sha256', 'gc_normalization_sha256', 'qualification_sha256'}, 'PREFIX_IDENTITY')
    for key in ('contract_sha256', 'research_objective_sha256'):
        campaign._sha(value[key], key)
    for key in set(prefix) - {'binding'}:
        campaign._sha(prefix[key], key)
    campaign.require_binding(prefix['binding'], label='prefix normalization', verify_file=verify_files)
    return json.loads(json.dumps(value, allow_nan=False))



def _input_manifest(recipe):
    ref = campaign.require_binding(recipe['input_manifest'], label='prefix input metadata', verify_file=True)
    value = campaign.read_bound_json(Path(ref['path']), ref['sha256'])
    keys = {'schema_version','source_design','physical_provider_plan','physical_provider_worker','prefix_normalization',
        'constructor_metadata','view_source_bindings','targets','archive_members','state_qualification',
        'native_private_qualification','input_qualification'}
    _require(isinstance(value, Mapping) and set(value) == keys and value['schema_version'] == INPUT_SCHEMA, 'INPUT_METADATA_KEYS')
    fold = _fold(recipe['fold']); prefix = value['prefix_normalization']
    _require(isinstance(prefix, Mapping) and set(prefix) == {'fold_id','prefix_parent_rows','cutoff_ns','output',
        'base_normalization_sha256','combined_contract_sha256','gc_layout_sha256','gc_normalization_sha256',
        'fresh_support_report','original_support_report'} and type(prefix['fold_id']) is int
        and type(prefix['prefix_parent_rows']) is int and type(prefix['cutoff_ns']) is int
        and (prefix['fold_id'],prefix['prefix_parent_rows'],prefix['cutoff_ns']) ==
            (fold['fold_id'],fold['prefix_rows'],fold['train_cutoff_ns']), 'INPUT_PREFIX_FOLD')
    for key in ('base_normalization_sha256','combined_contract_sha256','gc_layout_sha256','gc_normalization_sha256'):
        campaign._sha(prefix[key], key)
    for key in ('output','fresh_support_report','original_support_report'):
        campaign.require_binding(prefix[key], label=key, verify_file=False)
    for key in keys - {'schema_version','prefix_normalization','view_source_bindings','archive_members'}:
        campaign.require_binding(value[key], label=key, verify_file=False)
    _require(isinstance(value['archive_members'], list) and len(value['archive_members']) == 11
        and set(value['archive_members']) == ARCHIVE_MEMBERS, 'INPUT_TARGET_INVENTORY')
    _require(isinstance(value['view_source_bindings'], Mapping) and set(value['view_source_bindings']) == VIEW_SOURCE_KEYS, 'INPUT_VIEW_BINDINGS')
    for item in value['view_source_bindings'].values():
        campaign.require_binding(item, label='view source', verify_file=False)
    return value


def require_prefix_fit_launch_recipe(binding, *, source_repo, source_commit, verify_files=True):
    """Check exact metadata/source bytes; numerical270 validation stays in CUDA."""
    ref = campaign.require_binding(binding, label='prefix fit recipe', verify_file=verify_files)
    _require(verify_files, 'RECIPE_REQUIRES_PHYSICAL_METADATA')
    value = campaign.read_bound_json(Path(ref['path']), ref['sha256'])
    fields = {'schema_version', 'source_repo', 'runtime_root', 'admission_stage', 'run_scope',
        'numerical_recipe', 'input_manifest', 'source_owners', 'execution_binding_path', 'fold',
        'train_passes', 'max_windows', 'parent_order_algorithm', 'initial_measurement_rows',
        'eval_batch_size', 'checkpoint_every_updates', 'max_checkpoint_bytes',
        'max_checkpoint_total_bytes', 'max_published_json_bytes', 'training_enabled',
        'full_epoch_training_allowed', 'test_data_used', 'predictive_value_claimed',
        'qualification_weights_for_main_fit'}
    bps = value.get('run_scope') == BPS_SCOPE
    continuation = value.get('run_scope') == CONTINUATION_SCOPE
    if bps:
        fields |= {'gc_layout_metadata_parity', 'native_bps_qualification'}
    if continuation:
        fields |= {'gc_layout_metadata_parity', 'continuation_origin', 'native_continuation_qualification'}
    recipe = _identity(value, fields, RECIPE_SCHEMA, 'recipe_sha256')
    _require(Path(_path(recipe['source_repo'], 'source repo')) == source_repo == CANONICAL_REPO
        and type(source_commit) is str and campaign._COMMIT.fullmatch(source_commit) is not None, 'RECIPE_SOURCE')
    runtime = Path(_path(recipe['runtime_root'], 'runtime root'))
    _require(type(recipe['admission_stage']) is str and recipe['admission_stage'].startswith('dynamic_m5_direction_')
        and recipe['run_scope'] in (BPS_SCOPE, CONTINUATION_SCOPE, 'single_prefix_one_pass_direction_fit'), 'RECIPE_SCOPE')
    fold = _fold(recipe['fold'])
    if bps:
        _require(fold == private.BPS_FOLD
            and recipe['gc_layout_metadata_parity'] == private.BPS_GC_PARITY
            and recipe['native_bps_qualification'] == BPS_NATIVE_QUALIFICATION, 'BPS_QUALIFIED_SCOPE')
        campaign.require_binding(recipe['gc_layout_metadata_parity'], label='bps GC metadata proof', verify_file=True)
    for key, expected in {'train_passes':3 if continuation else 1, 'max_windows':4, 'initial_measurement_rows':256,
            'eval_batch_size':16, 'checkpoint_every_updates':256, 'max_checkpoint_bytes':1073741824,
            'max_checkpoint_total_bytes':4294967296, 'max_published_json_bytes':67108864}.items():
        _require(type(recipe[key]) is int and recipe[key] == expected, 'RECIPE_BUDGET:' + key)
    _require(recipe['parent_order_algorithm'] == PARENT_ORDER_ALGORITHM, 'ORDER_ALGORITHM')
    for key, expected in {'training_enabled':True, 'full_epoch_training_allowed':True,
            'test_data_used':False, 'predictive_value_claimed':False, 'qualification_weights_for_main_fit':False}.items():
        _require(recipe[key] is expected, 'RECIPE_AUTHORITY:' + key)
    numerical = recipe['numerical_recipe']
    _require(isinstance(numerical, Mapping)
        and numerical.get('signed_bps_supervision', False) is bps, 'NUMERICAL_OBJECTIVE_SCOPE')
    _require(isinstance(numerical, Mapping) and numerical.get('schema_version') == 'gx1_native_direction_research_recipe_v1'
        and type(numerical.get('batch_size')) is int and numerical['batch_size'] == BATCH_SIZE
        and type(numerical.get('max_updates')) is int and numerical['max_updates'] == (3 if continuation else 1) * ((fold['prefix_rows'] + 15) // 16)
        and numerical.get('genuine_updates_authorized') is False and numerical.get('native_launch_authorized') is False,
        'NUMERICAL_RECIPE_ROLE')
    unsigned = {k:v for k,v in numerical.items() if k != 'contract_sha256'}
    declared = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    _require(numerical.get('contract_sha256') == declared, 'NUMERICAL_RECIPE_SEAL')
    campaign._sha(numerical.get('research_objective_sha256'), 'research objective')
    _input_manifest(recipe)
    _require(_path(recipe['execution_binding_path'], 'execution binding') == str(runtime/'EXECUTION_BINDING.json'),
        'EXECUTION_BINDING_PATH')
    owners = recipe['source_owners']
    required_sources = set(REQUIRED_SOURCE_PATHS) | (set(BPS_EXTRA_SOURCE_PATHS) if bps else set())
    if continuation:
        required_sources.add('gx1/contracts/entry_native_direction_frozen_extraction_v1.py')
    _require(isinstance(owners, Mapping) and set(owners) == required_sources, 'SOURCE_OWNER_SET')
    for key, item in owners.items():
        checked = campaign.require_binding(item, label='prefix fit source', verify_file=True)
        _require(checked['path'] == str(source_repo/key), 'SOURCE_OWNER_PATH:' + key)
    if continuation:
        origin = recipe['continuation_origin']
        _require(fold['fold_id'] == 1 and isinstance(origin, Mapping)
            and origin == CONTINUATION_ORIGIN, 'CONTINUATION_ORIGIN')
        refs = {key:campaign.require_binding(ref, label='continuation '+key, verify_file=True) for key,ref in origin.items()}
        old = campaign.read_bound_json(Path(refs['launch_recipe']['path']), refs['launch_recipe']['sha256'])
        old_numerical = old.get('numerical_recipe', {})
        _require(old.get('fold') == fold and old.get('train_passes') == 1
            and old.get('run_scope') == 'single_prefix_one_pass_direction_fit'
            and old.get('input_manifest') == recipe['input_manifest']
            and old_numerical.get('max_updates') == (fold['prefix_rows'] + 15) // 16
            and {k:v for k,v in old_numerical.items() if k not in ('max_updates','contract_sha256')}
                == {k:v for k,v in numerical.items() if k not in ('max_updates','contract_sha256')}, 'CONTINUATION_ORIGINAL_RECIPE')
        initial = campaign.read_bound_json(Path(refs['initial_measurement']['path']), refs['initial_measurement']['sha256'])
        _require(initial.get('role') == 'INITIAL_TRAIN' and initial.get('fold') == fold
            and initial.get('own_initial_state') == refs['initial_state'], 'CONTINUATION_INITIAL_IDENTITY')
        _require(recipe['gc_layout_metadata_parity'] == private.BPS_GC_PARITY, 'CONTINUATION_GC_PARITY')
        campaign.require_binding(recipe['gc_layout_metadata_parity'], label='continuation GC metadata proof', verify_file=True)
        terminal_measurement = campaign.read_bound_json(Path(refs['terminal_train_measurement']['path']), refs['terminal_train_measurement']['sha256'])
        _require(terminal_measurement.get('role') == 'FINAL_TRAIN' and terminal_measurement.get('fold') == fold
            and terminal_measurement.get('own_initial_state') == refs['initial_state'], 'CONTINUATION_TERMINAL_MEASUREMENT')
        proof_ref = campaign.require_binding(recipe['native_continuation_qualification'], label='continuation zero-update proof', verify_file=True)
        proof = campaign.read_bound_json(Path(proof_ref['path']), proof_ref['sha256'])
        _require(proof.get('status') == 'PASS_ORIGINAL_NATIVE_CONTINUATION_ZERO_UPDATE'
            and proof.get('origin') == refs and proof.get('source_owners') == owners
            and proof.get('numerical_recipe_sha256') == numerical['contract_sha256']
            and proof.get('checkpoint_state_parity') is True and proof.get('terminal_train_prediction_bit_parity') is True
            and type(proof.get('optimizer_steps')) is int and proof['optimizer_steps'] == 0
            and proof.get('test_data_used') is False,
            'CONTINUATION_ZERO_UPDATE_QUALIFICATION')
    if bps:
        proof_ref = campaign.require_binding(recipe['native_bps_qualification'], label='native bps proof', verify_file=True)
        proof = campaign.read_bound_json(Path(proof_ref['path']), proof_ref['sha256'])
        _require(proof.get('status') == 'PASS_PRIVATE_NATIVE_MECHANICS_CONSUMED'
            and proof.get('matched_shared_parameters_buffers_rng_initial_direction_predictions') is True
            and proof.get('bit_exact_native_resume') is True
            and proof.get('recipe_source_pins_verified') is True
            and proof.get('private_weights_for_main_fit') is False
            and proof.get('prefix_rows') == fold['prefix_rows']
            and proof.get('prefix_cutoff_ns') == fold['train_cutoff_ns'], 'BPS_NATIVE_MECHANICS')
        qualified_ref = campaign.require_binding(proof['recipe'], label='qualified bps recipe', verify_file=True)
        qualified = campaign.read_bound_json(Path(qualified_ref['path']), qualified_ref['sha256'])
        _require(all(owners[key] == qualified['source_owners'][key] for key in BPS_NUMERICAL_SOURCE_PATHS),
            'BPS_QUALIFIED_NUMERICAL_SOURCE')
        campaign.require_binding(proof['own_prefix_bps_scale'], label='qualified own-prefix bps scale', verify_file=True)
    return recipe


def require_prefix_fit_window_policy(value, *, verify_files=True):
    fields = {'schema_version', 'recipe', 'invocation_number', 'max_invocation_seconds',
        'progress_path', 'campaign_cursor_path', 'output_directory', 'test_data_used'}
    result = _identity(value, fields, WINDOW_SCHEMA, 'policy_sha256')
    _require(type(result['invocation_number']) is int and 1 <= result['invocation_number'] <= MAX_WINDOWS
        and type(result['max_invocation_seconds']) is int and result['max_invocation_seconds'] == 12000
        and result['test_data_used'] is False, 'WINDOW_SCOPE')
    campaign.require_binding(result['recipe'], label='prefix window recipe', verify_file=verify_files)
    paths = [_path(result[key], key) for key in ('progress_path', 'campaign_cursor_path', 'output_directory')]
    _require(len(set(paths)) == 3 and Path(paths[0]).parent != Path(paths[1]).parent, 'WINDOW_PATH_COLLISION')
    return result


def require_prefix_fit_invocation_execution(result, execution, *, source_repo, source_commit):
    recipe_ref = campaign.require_binding(execution['prelaunch_manifest'], label='prefix fit recipe')
    recipe = require_prefix_fit_launch_recipe(recipe_ref, source_repo=source_repo, source_commit=source_commit)
    policy_ref = campaign.require_binding(execution['train_session_manifest'], label='prefix fit window')
    window = require_prefix_fit_window_policy(campaign.read_bound_json(Path(policy_ref['path']), policy_ref['sha256']))
    number = result['invocation_number']
    separator = result['launcher_argv'].index('--')
    _require(result['launcher_argv'][separator+1:] == [str(source_repo/'.venv/bin/python'), '-m', MODULE,
        '--window-policy', policy_ref['path'], '--window-policy-file-sha256', policy_ref['sha256'],
        '--progress-path', window['progress_path']], 'INVOCATION_ARGV')
    _require(result['kind'] == KIND and type(result['batch_size']) is int and result['batch_size'] == BATCH_SIZE
        and result['epoch_index'] is None and type(number) is int and 1 <= number <= MAX_WINDOWS
        and result['window_index'] == number-1 and result['window_count'] == MAX_WINDOWS
        and result['optimizer_step_budget'] == recipe['numerical_recipe']['max_updates']
        and type(result['optimizer_step_budget']) is int and result['expected_success_outcome'] == 'RESUMABLE_OR_COMPLETE'
        and result['maximum_wall_seconds'] == 13800 and window['invocation_number'] == number
        and result['checkpoint'] == {'pointer_path':window['campaign_cursor_path'],
            'before_mode':'GENESIS' if number == 1 else 'PREVIOUS_RECEIPT_AFTER',
            'predecessor_invocation_number':None if number == 1 else number-1, 'write_mode':'ATOMIC_UPDATE'}
        and result['progress_path'] == window['progress_path'] and window['recipe'] == recipe_ref
        and execution['prelaunch_manifest_sha256'] == recipe['recipe_sha256']
        and execution['train_session_manifest_sha256'] == window['policy_sha256'], 'INVOCATION_BINDING')
    result['direction_prefix_fit_recipe'] = recipe_ref
    result['direction_prefix_fit_window'] = window


def require_prefix_fit_plan(value, *, verify_files=True):
    fields = {'schema_version', 'decision', 'campaign_id', 'phase', 'created_utc', 'source_repo', 'source_commit',
        'runtime_root', 'gpu_uuid', 'prepared_windows_boot', 'recipe', 'invocations', 'signed_guard_sources',
        'controller_sources', 'policy', 'authority', 'test_data_used'}
    result = _identity(value, fields, campaign.PLAN_SCHEMA, 'plan_sha256')
    _require(result['phase'] == PHASE and result['decision'] == 'PASS_PREPARED'
        and type(result['campaign_id']) is str and bool(result['campaign_id']) and result['source_repo'] == str(CANONICAL_REPO)
        and type(result['source_commit']) is str and campaign._COMMIT.fullmatch(result['source_commit']) is not None
        and type(result['gpu_uuid']) is str and result['gpu_uuid'].startswith('GPU-') and result['test_data_used'] is False, 'PLAN_IDENTITY')
    campaign._utc(result['created_utc'], 'prefix fit creation')
    runtime = Path(_path(result['runtime_root'], 'runtime root'))
    result['prepared_windows_boot'] = campaign.require_boot_identity(result['prepared_windows_boot'])
    _require(campaign.canonical_sha256(result['policy']) == campaign.canonical_sha256(campaign.campaign_hardware_policy(PHASE))
        and campaign.canonical_sha256(result['authority']) == campaign.canonical_sha256(
            {'test':False, 'promotion':False, 'paper':False, 'live':False, 'cloud_spend':False}), 'PLAN_AUTHORITY')
    for key, names in (('signed_guard_sources', {'runner','guard','query','certificate'}),
            ('controller_sources', {'controller','observer','campaign_cli'})):
        _require(isinstance(result[key], Mapping) and set(result[key]) == names, 'PLAN_SOURCES:' + key)
        result[key] = {name:campaign.require_binding(ref, label=key, verify_file=verify_files) for name,ref in result[key].items()}
    result['recipe'] = campaign.require_binding(result['recipe'], label='prefix fit recipe', verify_file=verify_files)
    _require(isinstance(result['invocations'], list) and len(result['invocations']) == MAX_WINDOWS, 'PLAN_FOUR_WINDOWS')
    result['invocations'] = [campaign.require_binding(ref, label='prefix invocation', verify_file=verify_files) for ref in result['invocations']]
    _require(len({ref['path'] for ref in result['invocations']}) == MAX_WINDOWS, 'PLAN_DISTINCT_INVOCATIONS')
    result['selection_artifact_sha256'] = None
    if verify_files:
        recipe = require_prefix_fit_launch_recipe(result['recipe'], source_repo=CANONICAL_REPO, source_commit=result['source_commit'])
        _require(recipe['runtime_root'] == str(runtime), 'PLAN_RECIPE_RUNTIME')
        checked = []
        for number, ref in enumerate(result['invocations'], 1):
            item = campaign.require_invocation(campaign.read_bound_json(Path(ref['path']), ref['sha256']),
                source_repo=CANONICAL_REPO, source_commit=result['source_commit'], verify_files=True)
            window = item.get('direction_prefix_fit_window', {})
            _require(item['kind'] == KIND and item['invocation_number'] == number
                and item.get('direction_prefix_fit_recipe') == result['recipe']
                and window.get('output_directory') == str(runtime/'prefix-fit')
                and window.get('campaign_cursor_path') == str(runtime/'prefix-cursor'/'RESUME_CURSOR.json')
                and window.get('progress_path') == str(runtime/'progress'/f'invocation-{number:04d}.json')
                and item['guard_log_path'] == str(runtime/'logs'/f'invocation-{number:04d}.signed-guard.log'), 'PLAN_LAYOUT')
            checked.append(item)
        result['checked_invocations'] = checked
    return result


def require_native_direction_prefix_fit_readiness(repo, *, window_path, window_file_sha256):
    window = require_prefix_fit_window_policy(campaign.read_bound_json(window_path, window_file_sha256))
    policy_path = repo/'NEXT_RUN_POLICY.json'
    policy = campaign.read_bound_json(policy_path, campaign.file_sha256(policy_path))
    import subprocess
    head = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip()
    branch = subprocess.check_output(['git','-C',str(repo),'branch','--show-current'], text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(repo),'status','--porcelain=v1','--untracked-files=all'], text=True)
    recipe = require_prefix_fit_launch_recipe(window['recipe'], source_repo=repo, source_commit=head)
    gate = policy.get(recipe['admission_stage'])
    _require(repo == CANONICAL_REPO and branch == 'work/gx1-current' and not dirty
        and policy.get('schema_version') == 'gx1_next_native_run_policy_v1'
        and policy.get('canonical_source_repo') == str(repo) and policy.get('canonical_branch') == branch
        and policy.get('training_module') == MODULE and all(policy.get(key) is True for key in
            ('training_enabled','native_training_enabled','full_epoch_training_allowed'))
        and policy.get('full_val_allowed') is False, 'READINESS_GLOBAL')
    _require(isinstance(gate, Mapping) and gate.get('recipe') == window['recipe']
        and gate.get('run_root') == recipe['runtime_root'] and type(gate.get('max_invocations')) is int and gate['max_invocations'] == MAX_WINDOWS
        and type(gate.get('claimed_invocations')) is int and gate['claimed_invocations'] == 0
        and type(gate.get('completed_invocations')) is int and gate['completed_invocations'] == 0
        and gate.get('relaunch_allowed') is False, 'READINESS_ONCE')
    for key in ('execution_authorized','data_reads_authorized','genuine_constructor_authorized','genuine_forward_authorized',
            'genuine_backward_authorized','genuine_updates_authorized','genuine_predictor_fit_authorized',
            'training_enabled','native_training_enabled','full_epoch_training_allowed','native_launch_authorized'):
        _require(gate.get(key) is True, 'READINESS_GATE:' + key)
    for key in ('normalization_fit_authorized','synthetic_test_execution_authorized','test_data_used',
            'exit_evaluation','broker_access_allowed','full_val_allowed'):
        _require(gate.get(key) is False, 'READINESS_CLOSED_GATE:' + key)
    if recipe['run_scope'] == BPS_SCOPE:
        _require(gate.get('signed_bps_scale_fit_authorized') is True, 'READINESS_BPS_SCALE')
    campaign.require_binding(policy.get('gpu_clock_launcher'), label='existing GPU clock launcher')
    return {'decision':'READY_FOR_EXISTING_BOUND_CAMPAIGN_GATES', 'blocked_reasons':[],
        'canonical_source_repo':str(repo), 'source_commit':head, 'policy_sha256':campaign.file_sha256(policy_path),
        'training_started':False, 'native_invocation_seconds':12000, 'outer_guard_seconds':13800,
        'native_window_scope':{'prefix_fit_only':True, 'invocation_number':window['invocation_number'],
            'fold':recipe['fold'], 'max_windows':MAX_WINDOWS, 'full_epoch_training_allowed':True,
            'optimizer_step_ceiling':recipe['numerical_recipe']['max_updates'], 'window_policy_file_sha256':window_file_sha256}}


def require_native_direction_prefix_fit_context(policy_path, policy_file_sha256, progress_path):
    with campaign.file_verification_scope():
        window = require_prefix_fit_window_policy(campaign.read_bound_json(policy_path, policy_file_sha256))
        names = ('GX1_CAMPAIGN_PLAN_PATH','GX1_CAMPAIGN_PLAN_FILE_SHA256','GX1_CAMPAIGN_PLAN_SHA256','GX1_CAMPAIGN_INVOCATION_SHA256')
        _require(all(os.environ.get(name) for name in names), 'CAMPAIGN_ENV')
        plan_ref = {'path':os.environ[names[0]], 'sha256':os.environ[names[1]]}
        plan = campaign.require_plan(campaign.read_bound_json(Path(plan_ref['path']), plan_ref['sha256']))
        number = window['invocation_number']; invocation = plan['checked_invocations'][number-1]
        launch = require_prefix_fit_launch_recipe(window['recipe'], source_repo=CANONICAL_REPO, source_commit=plan['source_commit'])
        marker_path = Path(plan['runtime_root'])/'ACTIVE_INVOCATION.json'
        marker = campaign.read_bound_json(marker_path, campaign.file_sha256(marker_path))
        _require(plan['phase'] == PHASE and plan['plan_sha256'] == os.environ[names[2]]
            and invocation['invocation_sha256'] == os.environ[names[3]]
            and invocation['direction_prefix_fit_window'] == window and plan['recipe'] == window['recipe']
            and str(progress_path) == window['progress_path'], 'WINDOW_CONTEXT')
        receipt_dir = Path(plan['runtime_root'])/'receipts'
        expected = {f'invocation-{n:04d}.json' for n in range(1, number)}
        actual = {p.name for p in receipt_dir.glob('invocation-*.json')}
        _require(actual == expected, 'PRIOR_RECEIPT_SET')
        receipts = [campaign.read_bound_json(receipt_dir/name, campaign.file_sha256(receipt_dir/name)) for name in sorted(expected)]
        previous = campaign.require_receipt_chain(plan, receipts, verify_files=True)
        _require(not previous or previous[-1]['outcome'] == 'RESUMABLE', 'PRIOR_TERMINAL')
        before = 'GENESIS' if not previous else previous[-1]['checkpoint_pointer_after']['sha256']
        _require(marker.get('schema_version') == campaign.ACTIVE_SCHEMA
            and marker.get('marker_sha256') == campaign.canonical_sha256({k:v for k,v in marker.items() if k != 'marker_sha256'})
            and all(marker.get(key) == value for key,value in {'plan_sha256':plan['plan_sha256'],
                'invocation_sha256':invocation['invocation_sha256'], 'invocation_number':number,
                'invocation_id':f'invocation-{number:04d}', 'kind':KIND,
                'launcher_argv_sha256':invocation['launcher_argv_sha256'], 'pointer_before_sha256':before}.items()), 'ACTIVE_MARKER')
        binding_path = Path(launch['execution_binding_path'])
        binding = campaign.read_bound_json(binding_path, campaign.file_sha256(binding_path))
        _require(set(binding) == {'plan','source_head','policy'} and binding['plan'] == plan_ref
            and binding['source_head'] == plan['source_commit']
            and binding['policy'] == {'path':str(CANONICAL_REPO/'NEXT_RUN_POLICY.json'),
                'sha256':campaign.file_sha256(CANONICAL_REPO/'NEXT_RUN_POLICY.json')}, 'FROZEN_EXECUTION_BINDING')
        campaign.require_clean_source(plan)
        require_native_direction_prefix_fit_readiness(CANONICAL_REPO, window_path=policy_path, window_file_sha256=policy_file_sha256)
        _require(not progress_path.exists(), 'PROGRESS_ALREADY_EXISTS')
        if number == 1:
            _require(not Path(window['output_directory']).exists() and not Path(window['campaign_cursor_path']).parent.exists(), 'GENESIS_NOT_FRESH')
        else:
            cursor = require_prefix_fit_cursor(campaign.read_bound_json(Path(window['campaign_cursor_path']), before),
                expected_recipe=window['recipe'], expected_fold=launch['fold'], verify_files=True)
            _require(cursor['outcome'] == 'RESUMABLE', 'CURSOR_NOT_RESUMABLE')
    return window, plan, invocation, marker, launch


def build_prefix_fit_cursor(*, recipe_ref, fold, feature_view_identity, phase_state, outcome,
        optimizer_updates, parent_cursor, parent_order_sha256, full_state, own_initial_state,
        initial_measurement, final_train_measurement, oof_cursor, result, train_passes=1):
    _require(type(train_passes) is int and train_passes in (1, 3), 'CURSOR_PASSES')
    value = _seal({'schema_version':CURSOR_SCHEMA if train_passes == 1 else MULTIPASS_CURSOR_SCHEMA,
        **({'train_passes':train_passes} if train_passes != 1 else {}), 'recipe':dict(recipe_ref), 'fold':dict(fold),
        'feature_view_identity':feature_view_identity, 'phase_state':phase_state, 'outcome':outcome,
        'optimizer_updates':optimizer_updates, 'parent_cursor':parent_cursor, 'parent_order_sha256':parent_order_sha256,
        'full_state':full_state, 'own_initial_state':own_initial_state, 'initial_measurement':initial_measurement,
        'final_train_measurement':final_train_measurement, 'oof_cursor':oof_cursor, 'result':result,
        'test_data_used':False}, 'cursor_sha256')
    return require_prefix_fit_cursor(value, expected_recipe=recipe_ref, expected_fold=fold, verify_files=False)


def require_prefix_fit_cursor(value, *, expected_recipe, expected_fold=None, verify_files=True):
    fields = {'schema_version','recipe','fold','feature_view_identity','phase_state','outcome','optimizer_updates',
        'parent_cursor','parent_order_sha256','full_state','own_initial_state','initial_measurement',
        'final_train_measurement','oof_cursor','result','test_data_used'}
    multipass = isinstance(value, Mapping) and value.get('schema_version') == MULTIPASS_CURSOR_SCHEMA
    if multipass: fields = fields | {'train_passes'}
    result = _identity(value, fields, MULTIPASS_CURSOR_SCHEMA if multipass else CURSOR_SCHEMA, 'cursor_sha256')
    passes = result['train_passes'] if multipass else 1
    _require(type(passes) is int and passes == (3 if multipass else 1), 'CURSOR_PASSES')
    fold = _fold(result['fold']); n = fold['prefix_rows']; quarter = fold['oof_end_row']-n
    _require(result['recipe'] == expected_recipe and (expected_fold is None or fold == _fold(expected_fold))
        and result['test_data_used'] is False and result['outcome'] in {'RESUMABLE','COMPLETE','FAILED'}
        and result['phase_state'] in {'TRAINING','SCORING','COMPLETE'}, 'CURSOR_ROLE')
    campaign.require_binding(result['recipe'], label='cursor recipe', verify_file=verify_files)
    launch = None
    if verify_files:
        launch = campaign.read_bound_json(Path(result['recipe']['path']), result['recipe']['sha256'])
        _require(launch.get('fold') == fold and launch.get('train_passes') == passes and launch.get('schema_version') == RECIPE_SCHEMA
            and launch.get('recipe_sha256') == campaign.canonical_sha256({k:v for k,v in launch.items() if k != 'recipe_sha256'}),
            'CURSOR_RECIPE_FOLD')
    for key, ceiling in (('parent_cursor',passes*n), ('optimizer_updates',passes*((n+15)//16)), ('oof_cursor',quarter)):
        _require(type(result[key]) is int and 0 <= result[key] <= ceiling, 'CURSOR_COUNTER:' + key)
    completed, within = divmod(result['parent_cursor'], n)
    _require(within % BATCH_SIZE == 0, 'CURSOR_BATCH_BOUNDARY')
    _require(result['optimizer_updates'] <= completed*((n+15)//16)+(within+15)//16, 'CURSOR_UPDATES_VS_INPUTS')
    campaign._sha(result['parent_order_sha256'], 'parent order')
    campaign.require_binding(result['result'], label='prefix result', verify_file=verify_files)
    premodel_failure = (result['outcome'] == 'FAILED' and result['phase_state'] == 'TRAINING'
        and result['parent_cursor'] == result['optimizer_updates'] == result['oof_cursor'] == 0)
    if result['feature_view_identity'] is not None:
        result['feature_view_identity'] = _view_identity(result['feature_view_identity'], verify_files=verify_files)
        if launch is not None:
            prefix = _input_manifest(launch)['prefix_normalization']; bound = result['feature_view_identity']['prefix_normalization']
            _require(bound['binding'] == prefix['output'] and all(bound[target] == prefix[source] for target,source in (
                ('contract_sha256','combined_contract_sha256'),('base_sha256','base_normalization_sha256'),
                ('gc_layout_sha256','gc_layout_sha256'),('gc_normalization_sha256','gc_normalization_sha256')))
                and result['feature_view_identity']['research_objective_sha256'] == launch['numerical_recipe']['research_objective_sha256'],
                'CURSOR_INPUT_VIEW_IDENTITY')
    else:
        _require(premodel_failure, 'CURSOR_MISSING_VIEW')
    for key in ('full_state','own_initial_state','initial_measurement'):
        if result[key] is None:
            _require(premodel_failure, 'CURSOR_MISSING:' + key)
        else:
            campaign.require_binding(result[key], label=key, verify_file=verify_files)
    if result['final_train_measurement'] is not None:
        campaign.require_binding(result['final_train_measurement'], label='final TRAIN measurement', verify_file=verify_files)
    if result['phase_state'] == 'TRAINING':
        _require(result['oof_cursor'] == 0 and result['final_train_measurement'] is None and result['outcome'] != 'COMPLETE', 'CURSOR_TRAINING')
    else:
        _require(result['parent_cursor'] == passes*n, 'CURSOR_SCORING_BEFORE_TRAIN_END')
    _require(result['oof_cursor'] == 0 or result['final_train_measurement'] is not None, 'CURSOR_OOF_WITHOUT_FINAL_TRAIN')
    if result['phase_state'] == 'COMPLETE' or result['outcome'] == 'COMPLETE':
        _require(result['phase_state'] == result['outcome'] == 'COMPLETE' and result['oof_cursor'] == quarter
            and result['final_train_measurement'] is not None, 'CURSOR_COMPLETE')
    return result


def build_prefix_fit_progress(*, window, plan, invocation, checkpoint_pointer, cursor):
    from datetime import datetime, timezone
    checked = require_prefix_fit_cursor(cursor, expected_recipe=window['recipe'], verify_files=False)
    passes = checked.get('train_passes', 1)
    value = _seal({'schema_version':campaign.PROGRESS_SCHEMA,
        **({'train_passes':passes} if passes != 1 else {}), 'plan_sha256':plan['plan_sha256'],
        'invocation_sha256':invocation['invocation_sha256'], 'phase':KIND, 'recipe':window['recipe'],
        'fold':checked['fold'], 'phase_state':checked['phase_state'], 'completed_units':checked['parent_cursor'],
        'total_units':passes*checked['fold']['prefix_rows'], 'optimizer_updates':checked['optimizer_updates'],
        'parent_cursor':checked['parent_cursor'], 'parent_order_sha256':checked['parent_order_sha256'],
        'oof_cursor':checked['oof_cursor'], 'checkpoint_pointer':checkpoint_pointer, 'terminal':True,
        'outcome':checked['outcome'], 'observed_utc':datetime.now(timezone.utc).isoformat()}, 'progress_sha256')
    return require_prefix_fit_progress(value, plan_sha256=plan['plan_sha256'], invocation=invocation,
        expected_selection_receipt_sha256=None, verify_file=False)


def require_prefix_fit_progress(value, *, plan_sha256, invocation, expected_selection_receipt_sha256, verify_file=True):
    fields = {'schema_version','plan_sha256','invocation_sha256','phase','recipe','fold','phase_state','completed_units',
        'total_units','optimizer_updates','parent_cursor','parent_order_sha256','oof_cursor','checkpoint_pointer',
        'terminal','outcome','observed_utc'}
    multipass = isinstance(value, Mapping) and 'train_passes' in value
    if multipass: fields = fields | {'train_passes'}
    result = _identity(value, fields, campaign.PROGRESS_SCHEMA, 'progress_sha256')
    passes = result.get('train_passes', 1)
    _require(type(passes) is int and passes == (3 if multipass else 1), 'PROGRESS_PASSES')
    fold = _fold(result['fold']); n = fold['prefix_rows']; quarter = fold['oof_end_row']-n
    _require(invocation['optimizer_step_budget'] == passes*((n+15)//16), 'PROGRESS_PASS_BUDGET')
    _require(expected_selection_receipt_sha256 is None and invocation['kind'] == KIND
        and result['plan_sha256'] == plan_sha256 and result['invocation_sha256'] == invocation['invocation_sha256']
        and result['phase'] == KIND and result['recipe'] == invocation['direction_prefix_fit_recipe']
        and result['terminal'] is True and result['outcome'] in {'COMPLETE','RESUMABLE','FAILED'}
        and result['phase_state'] in {'TRAINING','SCORING','COMPLETE'}, 'PROGRESS_IDENTITY')
    campaign._utc(result['observed_utc'], 'prefix fit progress')
    for key, ceiling in (('completed_units',passes*n),('total_units',passes*n),('parent_cursor',passes*n),
            ('optimizer_updates',passes*((n+15)//16)),('oof_cursor',quarter)):
        _require(type(result[key]) is int and 0 <= result[key] <= ceiling, 'PROGRESS_COUNTER:' + key)
    completed, within = divmod(result['parent_cursor'], n)
    _require(result['total_units'] == passes*n and result['completed_units'] == result['parent_cursor']
        and within % 16 == 0
        and result['optimizer_updates'] <= completed*((n+15)//16)+(within+15)//16, 'PROGRESS_UNITS')
    _require(result['phase_state'] != 'TRAINING' or result['oof_cursor'] == 0, 'PROGRESS_TRAINING')
    _require(result['phase_state'] == 'TRAINING' or result['parent_cursor'] == passes*n, 'PROGRESS_SCORING')
    _require((result['outcome'] == 'COMPLETE') == (result['phase_state'] == 'COMPLETE')
        and (result['outcome'] != 'COMPLETE' or result['oof_cursor'] == quarter), 'PROGRESS_COMPLETE')
    campaign._sha(result['parent_order_sha256'], 'parent order')
    ref = campaign.require_binding(result['checkpoint_pointer'], label='prefix fit cursor', verify_file=verify_file)
    _require(ref['path'] == invocation['checkpoint']['pointer_path'], 'PROGRESS_CURSOR_PATH')
    if verify_file:
        cursor = require_prefix_fit_cursor(campaign.read_bound_json(Path(ref['path']), ref['sha256']),
            expected_recipe=result['recipe'], expected_fold=fold, verify_files=True)
        _require(cursor.get('train_passes',1) == passes and all(cursor[k] == result[k] for k in ('outcome','phase_state','optimizer_updates','parent_cursor',
            'parent_order_sha256','oof_cursor')), 'PROGRESS_DURABLE_CURSOR')
    return result
