"""Strict metadata authority for one fixed pure-direction prefix trajectory.

This stdlib-only variant shares canonical source, fresh-boot, ACTIVE, receipt,
CUDA/capped and hardware authorities. It changes neither the private four-update
qualification nor the numerical270 recipe/state owners. No metadata PASS is a
prediction-quality or trading admission.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
import subprocess
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
ORDER_SCOPE = 'single_prefix_one_pass_direction_chronological_order'
DENSE_SCOPE = 'single_prefix_dense_minute_direction_fit'
DENSE_DECIMAL_FIT_SCOPE = 'single_prefix_dense_decimal_direction_fit'
# Internal adapter for a separately admitted private decimal qualification only.
# The prefix-fit launch validator does not accept this as a training scope.
DENSE_DECIMAL_PRIVATE_SCOPE = 'private_dense_decimal_input_adapter'
# Private research qualification only; the prefix-fit validator rejects it.
M1_PRICE_PRIVATE_ADAPTER_SCOPE = 'private_dense_decimal_m1_price_input_adapter'
M5_DENSE_BPS_PRIVATE_SCOPE = 'private_dense_decimal_m5_signed_bps_60_qualification'
M1_PRICE_FIT_SCOPE = 'single_prefix_dense_decimal_m1_price_direction_fit'
M5_PRICE_FIT_SCOPE = 'single_prefix_dense_decimal_m5_price_direction_fit'
M5_FIVE_PRIORITY_FIT_SCOPE = 'single_prefix_dense_decimal_m5_five_priority_direction_fit'
FAMILY_SIGN_SCOPE = 'single_prefix_m5_family_value_sign_one_pass'
DYNAMIC_STATE_SCOPE = 'single_prefix_dynamic_mtf_state_one_pass'
DYNAMIC_STATE_PRIVATE_ADAPTER_SCOPE = 'private_dynamic_mtf_state_input_adapter'
DYNAMIC_STATE_EXTRA_SOURCE_PATHS = (
    'gx1/contracts/entry_dynamic_mtf_state_targets_v1.py',
    'gx1/features/technical_indicators_v1.py', 'gx1/time/session_detector.py',
    'gx1/contracts/entry_dynamic_mtf_state_report_v1.py',
    'gx1/scripts/run_entry_native_direction_dense_qualification_v1.py',
)
# Input adapter only; not independently admitted for fit or private execution.
FAMILY_PRIVATE_ADAPTER_SCOPE = 'private_dense_decimal_m5_family_value_input_adapter'
FAMILY_INTERVENTION_SOURCE = 'gx1/contracts/entry_direction_family_intervention_v1.py'
M5_FACTOR_SIGN_SCOPE = 'single_prefix_m5_factor_sign_one_pass'
M5_FACTOR_BPS_SCOPE = 'single_prefix_m5_factor_bps_one_pass'
M5_FACTOR_CONTINUATION_SCOPE = 'single_prefix_m5_factor_sign_two_pass'
M1_PRICE_COARSENED_MODE = 'price_and_m1_coarsened'
DENSE_NATIVE_QUALIFICATION = {'path':'/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DENSE_NATIVE_QUALIFICATION_20261028_001/ACTUAL_ROOT_REVIEW_001.json',
    'sha256':'7b5768f2310446716f9954ab2e5887831e5dc8e6bde143c7bc4fa72156b17b24'}
DENSE_DECIMAL_NATIVE_QUALIFICATION = {'path':'/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DENSE_DECIMAL_PRIVATE_QUALIFICATION_20261028_001/ACTUAL_ROOT_REVIEW_001.json',
    'sha256':'f994b56dc5b5c76cb4edb1e4797a5b3e6556239a99c65c35fd8157fac5e5b81c'}
DENSE_DECIMAL_REFERENCE_PREDICTIONS = {'path':'/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DENSE_ONEPASS_FIT_20261028_001/RUNTIME/prefix-fit/window-0001/OOF.npz',
    'sha256':'0d06be58c0e2748ecc980080c5eb5cec9a456b6326870e510744efb8bae7abd7'}
DECIMAL_EVALUATION_SOURCES = (
    'gx1/scripts/compare_entry_direction_decimal_dense_v1.py',
    'gx1/scripts/score_entry_direction_dense_onepass_v1.py',
    'gx1/scripts/score_entry_direction_dense_minutes_v1.py',
    'gx1/scripts/compare_entry_direction_saved_bps_v1.py',
    'gx1/scripts/compare_entry_direction_matched_exposure_v1.py',
    'gx1/contracts/entry_direction_block_rank_v1.py',
    'gx1/contracts/entry_direction_probability_reference_v1.py',
    'gx1/contracts/entry_native_direction_context_report_v1.py',
)
DECIMAL_TRANSITION_FUNCTIONS = {
    'gx1/contracts/entry_native_direction_prefix_fit_campaign_v1.py': ('_path','_fold','_input_manifest'),
    'gx1/scripts/run_entry_native_direction_prefix_fit_window_v1.py': ('_metadata','_launch_metadata',
        '_qualify_axes_targets','_fixed_order','_prepare','_batch','_device_batch','_prefix_prior','_evaluate'),
}
DECIMAL_TRANSITION_ALLOWED = {
    'gx1/contracts/entry_native_direction_prefix_fit_campaign_v1.py': {
        'changed_functions': ('require_prefix_fit_launch_recipe',),
        'new_functions': ('_require_decimal_source_equivalence',),
        'new_assignments': ('DENSE_DECIMAL_FIT_SCOPE','DENSE_DECIMAL_NATIVE_QUALIFICATION',
            'DENSE_DECIMAL_REFERENCE_PREDICTIONS','DECIMAL_EVALUATION_SOURCES','DECIMAL_TRANSITION_FUNCTIONS','DECIMAL_TRANSITION_ALLOWED'),
    },
    'gx1/scripts/run_entry_native_direction_prefix_fit_window_v1.py': {
        'changed_functions': ('run_native_direction_prefix_fit_window',),
        'new_functions': ('_fit_input_launch',), 'new_assignments': (),
    },
}
ORDER_ALGORITHM = 'strictly_increasing_prefix_parent_rows'
_CONTINUATION_ROOT = '/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_PREFIX_FIT_FOLD1_20261025_276'
CONTINUATION_ORIGIN = {
    'launch_recipe': {'path':_CONTINUATION_ROOT+'/RECIPE.json','sha256':'8ba298a148af3b923c7b4a567f98bf8ddd2eaef1e4eeb5144e2a785e9f9e2d93'},
    'terminal_state': {'path':_CONTINUATION_ROOT+'/RUNTIME/prefix-fit/states/w0002_TERMINAL_u002976_p047602.pt','sha256':'be9e9a54851085c6719f03a29d958f7c83b470236a0fc5ae5801c1bdd543af29'},
    'initial_measurement': {'path':_CONTINUATION_ROOT+'/RUNTIME/prefix-fit/window-0001/INITIAL_TRAIN.json','sha256':'0504cb783b1054a4a19e1da84a0013abee11100038cdbcd78b6514913064c09c'},
    'initial_state': {'path':_CONTINUATION_ROOT+'/RUNTIME/prefix-fit/states/INITIAL_0U.pt','sha256':'40a9d5ad744a06512619c3072170657f6ed430a077c3187e05e817fd5b46e9f3'},
    'terminal_train_measurement': {'path':_CONTINUATION_ROOT+'/RUNTIME/prefix-fit/window-0001/FINAL_TRAIN.json','sha256':'b25e7ef3539f19604dfbde58bfb6ff379dd7e0d813a464ab680cea4977578301'},
}
ORDER_ORIGIN = {'launch_recipe': CONTINUATION_ORIGIN['launch_recipe'],
    'initial_measurement': CONTINUATION_ORIGIN['initial_measurement'],
    'initial_predictions': private.WEIGHT_INITIAL_PREDICTIONS}
ORDER_COMPONENT_SOURCE = 'gx1/contracts/entry_native_direction_research_v1.py'

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


def require_dynamic_state_input_manifest(binding, *, fold, source_owners, verify_files=True):
    """Current full inputs over the unchanged physical provider; no refit or labels."""
    ref = campaign.require_binding(binding, label='dynamic state input manifest', verify_file=verify_files)
    value = campaign.read_bound_json(Path(ref['path']), ref['sha256'])
    fields = {'schema_version', 'fold', 'physical_provider_plan', 'physical_provider_worker',
        'prefix_normalization', 'prefix_qualification', 'normalization_verification',
        'constructor_metadata', 'view_source_bindings', 'axes'}
    manifest = _identity(value, fields, 'gx1_dynamic_mtf_state_input_manifest_v1', 'contract_sha256')
    _require(_fold(manifest['fold']) == _fold(fold) and fold['fold_id'] == 1, 'DYNAMIC_INPUT_FOLD')
    for key in fields - {'schema_version', 'fold', 'view_source_bindings'}:
        campaign.require_binding(manifest[key], label='dynamic input ' + key, verify_file=verify_files and key != 'axes')
    views = manifest['view_source_bindings']
    _require(isinstance(views, Mapping) and set(views) == VIEW_SOURCE_KEYS, 'DYNAMIC_VIEW_SOURCE_KEYS')
    for key, item in views.items():
        campaign.require_binding(item, label='dynamic view ' + key, verify_file=verify_files)
    owner_paths = {'direction_component_source':'gx1/contracts/entry_native_direction_research_v1.py',
        'native_constructor_source':'gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py',
        'mtf_window_source':'gx1/features/htf_features.py',
        'gc_gather_source':'gx1/contracts/entry_gc_mtf_history_v1.py',
        'owner_profile_source':'gx1/features/entry_specialist_feature_groups_v1.py'}
    _require(all(views[key] == source_owners[path] for key, path in owner_paths.items()), 'DYNAMIC_CURRENT_SOURCE_BINDINGS')
    _require(views['input_qualification'] == manifest['prefix_qualification'], 'DYNAMIC_CURRENT_INPUT_QUALIFICATION')
    if verify_files:
        def read(item):
            return campaign.read_bound_json(Path(item['path']), item['sha256'])
        verified = read(manifest['normalization_verification'])
        fitted = read(manifest['prefix_normalization'])
        qualification = read(manifest['prefix_qualification'])
        _require(verified.get('normalization') == manifest['prefix_normalization']
            and verified.get('all12statistic_surfaces_and_support_exactly_equal') is True
            and verified.get('source_and_payload_bookends_unchanged') is True
            and verified.get('genuine_native_model_calls') == 0
            and verified.get('test_data_used') is False, 'DYNAMIC_NORMALIZATION_VERIFICATION')
        _require(fitted.get('prefix_qualification') == qualification
            and qualification['proof']['prefix']['rows'] == fold['prefix_rows']
            and qualification['proof']['prefix']['cutoff_ns'] == fold['train_cutoff_ns'],
            'DYNAMIC_NORMALIZATION_POPULATION')
    return manifest




_WSL_STORAGE_QUERY = r"""
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding $false
$x=@(Get-ChildItem 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss' | ForEach-Object {Get-ItemProperty $_.PSPath} | Where-Object {$_.DistributionName -eq 'Ubuntu-22.04'})
if($x.Count -ne 1){throw 'distro identity'}
$base=$x[0].BasePath
$vhd=Get-Item -LiteralPath (Join-Path $base 'ext4.vhdx') -Force
if($vhd.PSIsContainer){throw 'vhd not file'}
$cur=$vhd; $noReparse=$true
while($null -ne $cur){
 if(($cur.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0){$noReparse=$false}
 if($cur.PSIsContainer){$cur=$cur.Parent}else{$cur=$cur.Directory}
}
$root=[IO.Path]::GetPathRoot($base)
$drive=New-Object IO.DriveInfo($root)
[pscustomobject]@{distribution=$x[0].DistributionName;base_path=$base;drive_root=$root;no_reparse=$noReparse;host_available_bytes=$drive.AvailableFreeSpace} | ConvertTo-Json -Compress
"""


def _extended_storage_plan(recipe, *, private_scope, recovery=False, stage=None, stage_rows=None):
    """Known exact output files plus an explicit bound reserve, not a reservation."""
    _require(type(private_scope) is bool and type(recovery) is bool and not (private_scope and recovery), 'STATE_STORAGE_SCOPE')
    numerical = recipe['numerical_recipe']
    horizon = numerical.get('extended_forecast_minutes')
    if horizon is None:
        _require(numerical.get('extended_storage_reserve_bytes') is None, 'STATE_STORAGE_UNBOUND_RESERVE')
        return None
    from gx1.contracts.entry_dynamic_mtf_state_targets_v1 import extended_dense_lead_minutes
    from gx1.contracts.entry_dynamic_mtf_state_report_v1 import dynamic_state_prediction_file_bytes
    extended_dense_lead_minutes(horizon)
    weight = numerical.get('current_state_first_loss_weight')
    _require(type(weight) is float and math.isfinite(weight) and weight > 0, 'STATE_STORAGE_CURRENT_WEIGHT')
    reserve = numerical.get('extended_storage_reserve_bytes')
    _require(type(reserve) is int and reserve > 0, 'STATE_STORAGE_RESERVE')
    max_checkpoint = recipe['max_checkpoint_bytes'] if private_scope else recipe['max_checkpoint_total_bytes']
    max_json = recipe['max_published_json_bytes']
    _require(type(max_checkpoint) is int and max_checkpoint > 0 and type(max_json) is int and max_json > 0
        and reserve >= max_checkpoint + max_json, 'STATE_STORAGE_RESERVE_FLOOR')
    if private_scope:
        stages = [('INITIAL_TRAIN', 256)]
    else:
        fold = _fold(recipe['fold'])
        stages = ([] if recovery else [('INITIAL_TRAIN', recipe['initial_measurement_rows'])])
        stages += [('FINAL_TRAIN', recipe['final_train_measurement_rows']),
                   ('OOF', fold['oof_end_row'] - fold['oof_start_row'])]
    if stage is not None:
        names = [name for name, _ in stages]
        _require(stage in names and type(stage_rows) is int and stage_rows == stages[names.index(stage)][1],
            'STATE_STORAGE_STAGE')
        stages = stages[names.index(stage):]
    else:
        _require(stage_rows is None, 'STATE_STORAGE_STAGE')
    files = {name: dynamic_state_prediction_file_bytes(rows, horizon) for name, rows in stages}
    payload = sum(sum(item.values()) for item in files.values())
    return {'horizon_minutes': horizon, 'remaining_stage_rows': dict(stages), 'logical_npy_bytes': payload,
            'reserve_bytes': reserve, 'required_free_bytes': payload + reserve,
            'scope': 'known_NPY_bytes_plus_bound_reserve_not_a_reservation'}


def _observe_wsl_host_storage(directory):
    """Resolve current distro's non-reparse VHD and measure guest and host free space."""
    _require(os.environ.get('WSL_DISTRO_NAME') == 'Ubuntu-22.04', 'STATE_STORAGE_DISTRO')
    path = Path(directory).resolve(strict=True)
    _require(path.is_dir() and os.stat(path).st_dev == os.stat('/').st_dev, 'STATE_STORAGE_GUEST_DEVICE')
    try:
        process = subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
            '-NoProfile', '-NonInteractive', '-Command', _WSL_STORAGE_QUERY], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=True)
        host = json.loads(process.stdout.decode('utf-8-sig').replace('\r', ''))
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError) as exc:
        raise RuntimeError('NATIVE_DIRECTION_PREFIX_FIT_STATE_STORAGE_HOST_UNAVAILABLE') from exc
    _require(isinstance(host, Mapping) and set(host) == {'distribution','base_path','drive_root','no_reparse','host_available_bytes'}
        and host['distribution'] == 'Ubuntu-22.04' and host['no_reparse'] is True
        and type(host['base_path']) is str and type(host['drive_root']) is str
        and re.fullmatch(r'[A-Za-z]:\\', host['drive_root']) is not None
        and host['base_path'].lower().startswith(host['drive_root'].lower())
        and type(host['host_available_bytes']) is int and host['host_available_bytes'] > 0, 'STATE_STORAGE_HOST_IDENTITY')
    mount = '/mnt/' + host['drive_root'][0].lower()
    try:
        guest = os.statvfs(path)
        mounted_host = os.statvfs(mount)
    except OSError as exc:
        raise RuntimeError('NATIVE_DIRECTION_PREFIX_FIT_STATE_STORAGE_VOLUME_UNAVAILABLE') from exc
    guest_free, mount_free = guest.f_bavail * guest.f_frsize, mounted_host.f_bavail * mounted_host.f_frsize
    _require(guest_free > 0 and mount_free > 0, 'STATE_STORAGE_AVAILABLE')
    return {'guest_available_bytes': guest_free, 'host_available_bytes': host['host_available_bytes'],
            'mounted_host_available_bytes': mount_free, 'host_base_path': host['base_path'],
            'host_drive_root': host['drive_root']}


def require_dynamic_state_storage_headroom(recipe, directory, *, private_scope, recovery=False,
                                           stage=None, stage_rows=None, expected_host=None):
    plan = _extended_storage_plan(recipe, private_scope=private_scope, recovery=recovery, stage=stage, stage_rows=stage_rows)
    if plan is None:
        return None
    observation = _observe_wsl_host_storage(directory)
    identity = (observation['host_base_path'], observation['host_drive_root'])
    _require(expected_host is None or identity == expected_host, 'STATE_STORAGE_HOST_CHANGED')
    available = min(observation['guest_available_bytes'], observation['host_available_bytes'],
                    observation['mounted_host_available_bytes'])
    _require(available >= plan['required_free_bytes'], 'STATE_STORAGE_INSUFFICIENT')
    return {'plan': plan, 'observation': observation, 'host_identity': identity,
            'observed_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'timestamped_capacity_check_not_reserved_space'}



def build_extended_train_label_audit_draft(*, numerical_recipe, target_manifest, target_sources, fold, prior_rule_review):
    """Data-only draft from a bound tape; never an admission or an independent review."""
    from gx1.contracts.entry_model_native_train_recipe_v1 import require_native_direction_research_recipe
    from gx1.contracts.entry_dynamic_mtf_state_targets_v1 import load_dynamic_mtf_state_series, extended_dense_lead_minutes
    from gx1.contracts.entry_dynamic_mtf_state_report_v1 import dense_train_query_counts, dense_train_prior_from_counts
    numerical = require_native_direction_research_recipe(numerical_recipe)
    _require(numerical.get('dynamic_state_supervision') is True
        and 'extended_forecast_minutes' in numerical, 'STATE_TRAIN_DRAFT_EXTENDED_RECIPE')
    fold = _fold(fold)
    _require(fold == private.BPS_FOLD, 'STATE_TRAIN_DRAFT_FOLD')
    _require(isinstance(target_sources, Mapping) and set(target_sources) == {'m1','m5_atr'}, 'STATE_TRAIN_DRAFT_SOURCES')
    sources = {key: campaign.require_binding(ref, label='state target source', verify_file=False)
               for key,ref in target_sources.items()}
    target_ref = campaign.require_binding(target_manifest, label='state target manifest', verify_file=True)
    rule_ref = campaign.require_binding(prior_rule_review, label='prior target rule review', verify_file=True)
    rule = campaign.read_bound_json(Path(rule_ref['path']), rule_ref['sha256'])
    _require(rule.get('schema_version') == 'gx1_dynamic_mtf_state_label_audit_v1'
        and rule.get('support_status') == 'PASS' and rule.get('blockers') == []
        and rule.get('target_rule_locked') is True and rule.get('test_data_used') is False
        and rule.get('later_metrics_inspected') is False and rule.get('model_forwards') == rule.get('optimizer_steps') == 0
        and rule.get('target_manifest') == target_ref and rule.get('target_sources') == sources
        and rule.get('fold') == fold and rule.get('independent_rule_review', {}).get('decision') == 'PASS_SAVED_TRAIN_RULE_WITNESSES'
        and all(rule.get('integrator_verification', {}).get(key) is True for key in
            ('ATR_thresholds_and_axes_bitidentical','all_TRAIN_classes_and_transition_rows_present',
             'all_output_refs_and_array_sha256_checked','terminal_and_launcher_exit_zero')), 'STATE_TRAIN_DRAFT_RULE_REVIEW')
    series = load_dynamic_mtf_state_series(target_ref, expected_source_refs=sources, expected_cutoff_ns=fold['oof_cutoff_ns'])
    _require(series.metadata['parents'] == fold['oof_end_row']
        and series.metadata['definition']['contract_sha256'] == rule['target_contract_sha256'], 'STATE_TRAIN_DRAFT_TARGET_IDENTITY')
    leads = extended_dense_lead_minutes(numerical['extended_forecast_minutes'])
    counts = dense_train_query_counts(series, prefix_rows=fold['prefix_rows'], cutoff_ns=fold['train_cutoff_ns'], lead_minutes=leads)
    prior = dense_train_prior_from_counts(counts, lead_minutes=leads)
    return {'schema_version':'gx1_dynamic_mtf_state_label_audit_draft_v1', 'support_status':'DRAFT_NOT_ADMITTED',
        'blockers':['independent_count_review_and_source_bound_plan_required'], 'target_rule_locked':True,
        'target_manifest':target_ref, 'target_sources':sources, 'fold':fold,
        'target_contract_sha256':series.metadata['definition']['contract_sha256'],
        'objective_sha256':numerical['research_objective_sha256'], 'numerical_recipe_sha256':numerical['contract_sha256'],
        'lead_minutes':list(leads), 'prior':prior, 'train_query_counts':counts, 'prior_rule_review':rule_ref,
        'model_forwards':0, 'optimizer_steps':0, 'test_data_used':False, 'later_metrics_inspected':False,
        'scope':'All pre-TEST cache bytes are SHA/structurally validated; only TRAIN prefix labels and reasons are aggregated.',
        'native_launch_authorized':False, 'predictive_value_claimed':False}



def _require_first_flat_activity_support(audit, recipe, *, target_ref, sources, fold, target_metadata):
    """Bind independent raw TRAIN evidence; this is not launch authority."""
    from gx1.contracts.entry_dynamic_mtf_state_targets_v1 import TIMEFRAMES
    ref = audit.get('first_flat_activity_source_support')
    _require(isinstance(ref, Mapping), 'STATE_FIRST_ACTIVITY_AUDIT_REQUIRED')
    ref = campaign.require_binding(ref, label='first activity TRAIN support', verify_file=True)
    proof = campaign.read_bound_json(Path(ref['path']), ref['sha256'])
    owner_key = 'gx1/contracts/entry_dynamic_mtf_state_targets_v1.py'
    owners = recipe.get('source_owners')
    _require(isinstance(owners, Mapping) and owner_key in owners, 'STATE_FIRST_ACTIVITY_SOURCE_OWNER')
    owner = campaign.require_binding(owners[owner_key], label='first activity target owner', verify_file=True)
    source_repo = Path(str(recipe.get('source_repo', '')))
    _require(source_repo.is_absolute() and str(source_repo / owner_key) == owner['path'], 'STATE_FIRST_ACTIVITY_REPO')
    dependencies = proof.get('target_dependencies')
    required = ('gx1/features/htf_features.py', 'gx1/features/technical_indicators_v1.py', 'gx1/time/session_detector.py')
    _require(isinstance(dependencies, Mapping) and set(dependencies) == set(required), 'STATE_FIRST_ACTIVITY_DEPENDENCIES')
    for name in required:
        dependency = campaign.require_binding(dependencies[name], label='first activity dependency ' + name, verify_file=True)
        _require(dependency['path'] == str(source_repo / name) and owners.get(name) == dependency,
            'STATE_FIRST_ACTIVITY_DEPENDENCY_PATH:' + name)
    _require(proof.get('schema_version') == 'gx1_first_flat_activity_train_source_support_v1'
        and proof.get('status') == 'PASS_INDEPENDENT_RAW_REVIEW_NO_LAUNCH_AUTHORITY'
        and proof.get('target_manifest') == target_ref and proof.get('target_sources') == sources
        and proof.get('target_contract_sha256') == target_metadata['definition']['contract_sha256'] and proof.get('fold') == fold
        and proof.get('target_owner') == owner
        and re.fullmatch(r'[0-9a-f]{40}', str(proof.get('reviewed_source_head', ''))) is not None
        and proof.get('model_calls') == proof.get('optimizer_steps') == 0
        and proof.get('test_data_used') is proof.get('native_launch_authorized') is False,
        'STATE_FIRST_ACTIVITY_SUPPORT_IDENTITY')
    raw_ref = campaign.require_binding(proof.get('raw_review'), label='first activity raw review', verify_file=True)
    raw = campaign.read_bound_json(Path(raw_ref['path']), raw_ref['sha256'])
    worker_ref = campaign.require_binding(proof.get('raw_worker'), label='first activity raw worker', verify_file=True)
    _require(raw.get('schema_version') == 'gx1_first_flat_activity_raw_train_review_v1'
        and raw.get('status') == 'PASS_RAW_SOURCE_TRAIN_FIRST_LABEL_PARITY_NO_LAUNCH_AUTHORITY'
        and raw.get('worker') == worker_ref and raw.get('source_head') == proof['reviewed_source_head']
        and raw.get('target_manifest') == target_ref and raw.get('target_sources') == sources and raw.get('fold') == fold
        and raw.get('model_calls') == raw.get('optimizer_steps') == raw.get('backward_calls') == raw.get('new_scoring') == 0
        and raw.get('test_data_used') is raw.get('native_launch_authorized') is False,
        'STATE_FIRST_ACTIVITY_RAW_IDENTITY')
    target_result_ref = campaign.require_binding(raw.get('target_result'), label='first activity target result', verify_file=True)
    target_result = campaign.read_bound_json(Path(target_result_ref['path']), target_result_ref['sha256'])
    calendar = target_result.get('calendar_binding')
    _require(target_result.get('target_manifest') == target_ref and target_result.get('target_sources') == sources
        and target_result.get('test_data_used') is False and raw.get('calendar_refs') == calendar
        and isinstance(calendar, Mapping)
        and set(calendar) == {'source_reference', 'exact_schedule', 'closure_authority', 'published_calendar'},
        'STATE_FIRST_ACTIVITY_CALENDAR_BINDING')
    for name, binding in calendar.items():
        campaign.require_binding(binding, label='first activity calendar ' + name, verify_file=True)
    schedule = campaign.read_bound_json(Path(calendar['exact_schedule']['path']), calendar['exact_schedule']['sha256'])
    authority = campaign.read_bound_json(Path(calendar['closure_authority']['path']), calendar['closure_authority']['sha256'])
    source = campaign.read_bound_json(Path(calendar['source_reference']['path']), calendar['source_reference']['sha256'])
    _require(schedule.get('source_method') == 'externally_sourced_exact_xau_utc_closure_intervals_v1'
        and schedule.get('instrument') == 'XAU_USD' and schedule.get('timeframe') == 'M1'
        and schedule.get('test_data_used') is False
        and schedule.get('source_reference_sha256') == calendar['source_reference']['sha256']
        and schedule.get('schedule_sha256') == target_metadata.get('closure_binding', {}).get('schedule_sha256')
        and authority.get('artifact_sha256') == target_metadata.get('closure_binding', {}).get('authority_sha256')
        and source.get('m1_source') == sources['m1'] and source.get('calendar') == calendar['published_calendar']
        and source.get('test_data_used') is False, 'STATE_FIRST_ACTIVITY_CALENDAR_IDENTITY')
    previous_ref = campaign.require_binding(raw.get('original_train_facts'), label='first activity TRAIN facts', verify_file=True)
    previous = campaign.read_bound_json(Path(previous_ref['path']), previous_ref['sha256'])
    _require(previous.get('schema_version') == 'gx1_first_flat_activity_train_audit_v1'
        and previous.get('status') == 'TRAIN_LABEL_FACTS_NOT_AN_ADMITTED_OBJECTIVE'
        and previous.get('new_scoring_or_test_used') is False
        and previous.get('references', {}).get('target_manifest') == target_ref
        and previous.get('per_timeframe') == proof.get('per_timeframe'), 'STATE_FIRST_ACTIVITY_PRIOR_FACTS')
    per_tf, reviewed = proof.get('per_timeframe'), raw.get('tf_results')
    _require(isinstance(per_tf, Mapping) and isinstance(reviewed, Mapping)
        and set(per_tf) == set(reviewed) == set(TIMEFRAMES), 'STATE_FIRST_ACTIVITY_TIMEFRAMES')
    fields = ('parents', 'activity_label_eligible', 'activity_label_ineligible', 'flat_known',
              'nonflat_direction_known', 'nonflat_direction_unknown')
    for tf in TIMEFRAMES:
        counts, item = per_tf[tf], reviewed[tf]
        _require(isinstance(counts, Mapping) and set(counts) == set(fields)
            and all(type(counts[k]) is int and counts[k] >= 0 for k in fields)
            and counts['parents'] == fold['prefix_rows']
            and counts['activity_label_eligible'] + counts['activity_label_ineligible'] == counts['parents']
            and counts['flat_known'] + counts['nonflat_direction_known'] + counts['nonflat_direction_unknown']
                == counts['activity_label_eligible']
            and counts['flat_known'] > 0 and counts['nonflat_direction_known'] + counts['nonflat_direction_unknown'] > 0
            and isinstance(item, Mapping) and item.get('counts') == counts
            and item.get('mature_first_parent_rows_compared') == fold['prefix_rows']
            and item.get('binary_mask_or_label_disagreements') == 0
            and item.get('atr_threshold_bitwise_differences') == 0
            and item.get('atr_threshold_max_abs_difference') == 0.0
            and item.get('local_range_max_abs_difference_on_eligible') == 0.0,
            'STATE_FIRST_ACTIVITY_SUPPORT_COUNTS:' + tf)
    return proof



def build_first_flat_activity_train_audit_draft(*, numerical_recipe, target_manifest, target_sources,
        fold, prior_rule_review, first_flat_activity_source_support, source_repo, source_owners):
    """Bind TRAIN-only evidence to an explicit candidate; never grant admission.

    The default grid reuses the separately reviewed frozen TRAIN prior. A
    longer dense grid uses the existing TRAIN-only counter. Both outputs remain
    drafts and must pass an independent objective-specific review and the usual
    source, policy, hardware and one-use launch authorities.
    """
    from gx1.contracts.entry_model_native_train_recipe_v1 import require_native_direction_research_recipe
    from gx1.contracts.entry_dynamic_mtf_state_targets_v1 import (
        default_trial_lead_minutes, load_dynamic_mtf_state_series, read_dynamic_mtf_state_series_metadata)
    from gx1.contracts.entry_dynamic_mtf_state_report_v1 import StateScoreAccumulator
    numerical = require_native_direction_research_recipe(numerical_recipe)
    _require(numerical.get('dynamic_state_supervision') is True
        and numerical.get('first_flat_activity_weight') is not None, 'STATE_ACTIVITY_DRAFT_RECIPE')
    _require('native_endpoint_loss' not in numerical, 'STATE_ENDPOINT_NOT_ADMITTED_IN_HISTORICAL_DRAFT')
    fold = _fold(fold)
    _require(fold == private.BPS_FOLD, 'STATE_ACTIVITY_DRAFT_FOLD')
    _require(isinstance(target_sources, Mapping) and set(target_sources) == {'m1', 'm5_atr'},
        'STATE_ACTIVITY_DRAFT_SOURCES')
    sources = {key:campaign.require_binding(value, label='activity target source', verify_file=False)
               for key,value in target_sources.items()}
    target_ref = campaign.require_binding(target_manifest, label='activity target manifest', verify_file=True)
    manifest = read_dynamic_mtf_state_series_metadata(target_ref, expected_source_refs=sources,
        expected_cutoff_ns=fold['oof_cutoff_ns'])
    meta = manifest['metadata']
    _require(meta['parents'] == fold['oof_end_row'], 'STATE_ACTIVITY_DRAFT_TARGET_POPULATION')
    support_ref = campaign.require_binding(first_flat_activity_source_support,
        label='activity TRAIN source support', verify_file=True)
    _require_first_flat_activity_support({'first_flat_activity_source_support':support_ref},
        {'source_repo':source_repo, 'source_owners':source_owners}, target_ref=target_ref,
        sources=sources, fold=fold, target_metadata=meta)
    rule_ref = campaign.require_binding(prior_rule_review, label='prior target rule review', verify_file=True)
    rule = campaign.read_bound_json(Path(rule_ref['path']), rule_ref['sha256'])
    _require(rule.get('schema_version') == 'gx1_dynamic_mtf_state_label_audit_v1'
        and rule.get('support_status') == 'PASS' and rule.get('blockers') == []
        and rule.get('target_rule_locked') is True and rule.get('test_data_used') is False
        and rule.get('later_metrics_inspected') is False and rule.get('model_forwards') == rule.get('optimizer_steps') == 0
        and rule.get('target_manifest') == target_ref and rule.get('target_sources') == sources and rule.get('fold') == fold
        and rule.get('target_contract_sha256') == meta['definition']['contract_sha256']
        and rule.get('independent_rule_review', {}).get('decision') == 'PASS_SAVED_TRAIN_RULE_WITNESSES'
        and all(rule.get('integrator_verification', {}).get(key) is True for key in
            ('ATR_thresholds_and_axes_bitidentical','all_TRAIN_classes_and_transition_rows_present',
             'all_output_refs_and_array_sha256_checked','terminal_and_launcher_exit_zero')),
        'STATE_ACTIVITY_DRAFT_RULE_REVIEW')
    if numerical.get('extended_forecast_minutes') is not None:
        draft = build_extended_train_label_audit_draft(numerical_recipe=numerical, target_manifest=target_ref,
            target_sources=sources, fold=fold, prior_rule_review=rule_ref)
        draft['blockers'].append('independent_first_flat_activity_objective_review_required')
    else:
        leads = list(default_trial_lead_minutes())
        train_ref = campaign.require_binding(rule.get('train_audit'), label='saved TRAIN target audit', verify_file=True)
        train = campaign.read_bound_json(Path(train_ref['path']), train_ref['sha256'])
        prior_ref = campaign.require_binding(train.get('train_prior'), label='saved TRAIN prior', verify_file=True)
        prior = campaign.read_bound_json(Path(prior_ref['path']), prior_ref['sha256'])
        _require(rule.get('lead_minutes') == leads and rule.get('prior') == prior
            and train.get('series') == target_ref and train.get('train_cutoff_ns') == fold['train_cutoff_ns']
            and train.get('train_parents') == fold['prefix_rows'] and train.get('later_aggregates_computed') is False
            and train.get('target_contract', {}).get('contract_sha256') == meta['definition']['contract_sha256']
            and train.get('class_counts') == prior.get('counts')
            and train.get('transition_counts') == prior.get('transition_counts'), 'STATE_ACTIVITY_DRAFT_PRIOR_BINDING')
        StateScoreAccumulator(prior, lead_minutes=leads)
        _require(prior.get('parents') == fold['prefix_rows'], 'STATE_ACTIVITY_DRAFT_PRIOR_POPULATION')
        # Recheck immutable target array hashes; no later labels are aggregated.
        series = load_dynamic_mtf_state_series(target_ref, expected_source_refs=sources,
            expected_cutoff_ns=fold['oof_cutoff_ns'])
        _require(series.metadata['parents'] == fold['oof_end_row']
            and series.metadata['definition']['contract_sha256'] == meta['definition']['contract_sha256'],
            'STATE_ACTIVITY_DRAFT_TARGET_IDENTITY')
        draft = {'schema_version':'gx1_dynamic_mtf_state_label_audit_draft_v1',
            'support_status':'DRAFT_NOT_ADMITTED',
            'blockers':['independent_objective_specific_train_review_required',
                        'fresh_source_binding_native_qualification_and_one_use_authority_required'],
            'target_rule_locked':True, 'target_manifest':target_ref, 'target_sources':sources, 'fold':fold,
            'target_contract_sha256':meta['definition']['contract_sha256'],
            'objective_sha256':numerical['research_objective_sha256'], 'numerical_recipe_sha256':numerical['contract_sha256'],
            'lead_minutes':leads, 'prior':prior, 'prior_rule_review':rule_ref, 'reused_train_audit':train_ref,
            'reused_train_prior':prior_ref, 'model_forwards':0, 'optimizer_steps':0, 'test_data_used':False,
            'later_metrics_inspected':False,
            'scope':'Verify the entire pre-TEST target cache; reuse already independently reviewed TRAIN-only counts.',
            'native_launch_authorized':False, 'predictive_value_claimed':False}
    _require(draft.get('schema_version') == 'gx1_dynamic_mtf_state_label_audit_draft_v1'
        and draft.get('support_status') == 'DRAFT_NOT_ADMITTED' and draft.get('native_launch_authorized') is False,
        'STATE_ACTIVITY_DRAFT_NOT_ADMITTED')
    draft['first_flat_activity_source_support'] = support_ref
    return draft


def require_dynamic_state_target_binding(recipe):
    """TRAIN audit is the rule lock; later cache bytes give no selection authority."""
    from gx1.contracts.entry_dynamic_mtf_state_targets_v1 import (
        read_dynamic_mtf_state_series_metadata, default_trial_lead_minutes, extended_dense_lead_minutes)
    fold = _fold(recipe['fold'])
    sources = recipe['target_sources']
    _require(isinstance(sources, Mapping) and set(sources) == {'m1', 'm5_atr'}, 'STATE_TARGET_SOURCES')
    for ref in sources.values():
        campaign.require_binding(ref, label='state raw target source', verify_file=False)
    target_ref = campaign.require_binding(recipe['target_manifest'], label='state target manifest', verify_file=True)
    manifest = read_dynamic_mtf_state_series_metadata(target_ref, expected_source_refs=sources,
        expected_cutoff_ns=fold['oof_cutoff_ns'])
    meta = manifest['metadata']
    _require(meta['parents'] == fold['oof_end_row'], 'STATE_TARGET_PARENT_POPULATION')
    horizon = recipe['numerical_recipe'].get('extended_forecast_minutes')
    leads = default_trial_lead_minutes() if horizon is None else extended_dense_lead_minutes(horizon)
    audit_ref = campaign.require_binding(recipe['label_audit'], label='TRAIN state audit', verify_file=True)
    audit = campaign.read_bound_json(Path(audit_ref['path']), audit_ref['sha256'])
    _require(audit.get('schema_version') == 'gx1_dynamic_mtf_state_label_audit_v1'
        and audit.get('support_status') == 'PASS' and audit.get('blockers') == []
        and audit.get('target_manifest') == target_ref and audit.get('target_sources') == sources
        and audit.get('fold') == fold
        and audit.get('target_contract_sha256') == meta['definition']['contract_sha256']
        and audit.get('objective_sha256') == recipe['numerical_recipe']['research_objective_sha256']
        and audit.get('lead_minutes') == list(leads)
        and audit.get('model_forwards') == audit.get('optimizer_steps') == 0
        and audit.get('test_data_used') is audit.get('later_metrics_inspected') is False,
        'STATE_TRAIN_LABEL_AUDIT')
    from gx1.contracts.entry_dynamic_mtf_state_report_v1 import StateScoreAccumulator
    # This validates exact empirical counts/probabilities; it never reads VAL.
    StateScoreAccumulator(audit['prior'], lead_minutes=audit['lead_minutes'])
    _require(audit['prior']['parents'] == fold['prefix_rows'], 'STATE_TRAIN_PRIOR_POPULATION')
    if recipe['numerical_recipe'].get('first_flat_activity_weight') is not None:
        _require_first_flat_activity_support(audit, recipe, target_ref=target_ref, sources=sources,
            fold=fold, target_metadata=meta)
    return manifest, audit


def require_dynamic_state_launch_recipe(value, *, source_repo, source_commit, private_scope):
    """Explicit new objective over existing guarded recipe/state/cursor owners."""
    fields = {'schema_version', 'source_repo', 'runtime_root', 'admission_stage', 'run_scope',
        'numerical_recipe', 'input_manifest', 'source_owners', 'execution_binding_path', 'fold',
        'max_checkpoint_bytes', 'max_published_json_bytes', 'training_enabled',
        'full_epoch_training_allowed', 'test_data_used', 'predictive_value_claimed',
        'qualification_weights_for_main_fit', 'target_manifest', 'target_sources', 'label_audit',
        'full_feature_profile'}
    recovery = is_scoring_recovery(value)
    _require(not (private_scope and recovery), 'STATE_PRIVATE_RECOVERY')
    if recovery:
        fields |= {'scoring_recovery_origin'}
    if private_scope:
        fields |= {'max_private_updates', 'max_model_constructions', 'max_forwards', 'max_backwards',
            'qualification_parent_rows'}
    else:
        fields |= {'train_passes', 'max_windows', 'parent_order_algorithm', 'initial_measurement_rows',
            'eval_batch_size', 'checkpoint_every_updates', 'max_checkpoint_total_bytes',
            'native_dynamic_state_qualification', 'final_train_measurement_rows'}
    recipe = _identity(value, fields, private.RECIPE_SCHEMA if private_scope else RECIPE_SCHEMA, 'recipe_sha256')
    _require(Path(_path(recipe['source_repo'], 'source repo')) == source_repo == CANONICAL_REPO
        and isinstance(source_commit, str) and campaign._COMMIT.fullmatch(source_commit) is not None, 'STATE_SOURCE')
    runtime = Path(_path(recipe['runtime_root'], 'runtime root'))
    _require(recipe['run_scope'] == (private.DYNAMIC_STATE_PRIVATE_SCOPE if private_scope else DYNAMIC_STATE_SCOPE)
        and type(recipe['admission_stage']) is str and recipe['admission_stage'].startswith('dynamic_mtf_state_')
        and recipe['execution_binding_path'] == str(runtime/'EXECUTION_BINDING.json'), 'STATE_SCOPE')
    fold = _fold(recipe['fold'])
    _require(fold == private.BPS_FOLD and recipe['full_feature_profile'] is True, 'STATE_FULL_PROFILE_FOLD1')
    expected = {'max_checkpoint_bytes':1073741824,
        'max_published_json_bytes':33554432 if private_scope else 67108864}
    expected.update({'max_private_updates':4, 'max_model_constructions':3, 'max_forwards':21,
        'max_backwards':4} if private_scope else {'train_passes':1, 'max_windows':1 if recovery else 4,
        'initial_measurement_rows':256, 'final_train_measurement_rows':47602, 'eval_batch_size':16,
        'checkpoint_every_updates':256, 'max_checkpoint_total_bytes':4294967296})
    _require(all(type(recipe[k]) is int and recipe[k] == v for k, v in expected.items()), 'STATE_BUDGET')
    for key, expected_value in {'training_enabled':not recovery, 'full_epoch_training_allowed':not (private_scope or recovery),
            'test_data_used':False, 'predictive_value_claimed':False,
            'qualification_weights_for_main_fit':False}.items():
        _require(recipe[key] is expected_value, 'STATE_AUTHORITY:' + key)
    if private_scope:
        rows = recipe['qualification_parent_rows']
        _require(isinstance(rows, list) and len(rows) == len(set(rows)) == 32
            and all(type(v) is int and v >= 0 for v in rows), 'STATE_QUALIFICATION_PARENTS')
    else:
        _require(recipe['parent_order_algorithm'] == PARENT_ORDER_ALGORITHM, 'STATE_PARENT_ORDER')
    numerical = recipe['numerical_recipe']
    _require(isinstance(numerical, Mapping)
        and numerical.get('schema_version') == 'gx1_native_direction_research_recipe_v1'
        and numerical.get('dynamic_state_supervision') is True
        and all(numerical.get(k, False) is False for k in ('signed_bps_supervision', 'magnitude_weighted_direction',
            'dense_minute_supervision', 'dense_signed_bps_supervision', 'five_minute_priority'))
        and type(numerical.get('batch_size')) is int and numerical['batch_size'] == 16
        and type(numerical.get('max_updates')) is int
        and numerical['max_updates'] == (2 if private_scope else (fold['prefix_rows'] + 15)//16)
        and numerical.get('genuine_updates_authorized') is numerical.get('native_launch_authorized') is False,
        'STATE_NUMERICAL_ROLE')
    # These historical campaigns have no admission for the December endpoint objective.
    _require('native_endpoint_loss' not in numerical, 'STATE_ENDPOINT_NOT_ADMITTED_IN_HISTORICAL_CAMPAIGN')
    if 'current_state_first_loss_weight' in numerical:
        weight = numerical['current_state_first_loss_weight']
        _require(type(weight) is float and math.isfinite(weight) and weight > 0, 'STATE_CURRENT_FIRST_LOSS_WEIGHT')
    if 'first_flat_activity_weight' in numerical:
        weight = numerical['first_flat_activity_weight']
        _require('current_state_first_loss_weight' in numerical and type(weight) is float
            and math.isfinite(weight) and weight > 0, 'STATE_FIRST_ACTIVITY_WEIGHT')
    if 'extended_forecast_minutes' in numerical:
        _require('current_state_first_loss_weight' in numerical, 'STATE_EXTENDED_REQUIRES_CURRENT_INPUT')
        from gx1.contracts.entry_dynamic_mtf_state_targets_v1 import extended_dense_lead_minutes
        extended_dense_lead_minutes(numerical['extended_forecast_minutes'])
        _extended_storage_plan(recipe, private_scope=private_scope, recovery=recovery)
    else:
        _require('extended_storage_reserve_bytes' not in numerical, 'STATE_STORAGE_UNBOUND_RESERVE')
    unsigned = {k:v for k,v in numerical.items() if k != 'contract_sha256'}
    # Numerical recipe seals use compact JSON without the campaign trailing newline.
    declared = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    _require(declared == numerical.get('contract_sha256'), 'STATE_NUMERICAL_SEAL')
    campaign._sha(numerical.get('research_objective_sha256'), 'state objective')
    owners = recipe['source_owners']
    _require(isinstance(owners, Mapping) and set(owners) == set(REQUIRED_SOURCE_PATHS) | set(DYNAMIC_STATE_EXTRA_SOURCE_PATHS),
        'STATE_SOURCE_OWNER_SET')
    for key, item in owners.items():
        ref = campaign.require_binding(item, label='state source owner', verify_file=True)
        _require(ref['path'] == str(source_repo/key), 'STATE_SOURCE_OWNER_PATH:' + key)
    require_dynamic_state_input_manifest(recipe['input_manifest'], fold=fold, source_owners=owners)
    require_dynamic_state_target_binding(recipe)
    if not private_scope:
        require_dynamic_state_qualification(recipe, source_repo=source_repo, source_commit=source_commit)
    return recipe


def require_dynamic_state_qualification(recipe, *, source_repo, source_commit):
    ref = campaign.require_binding(recipe['native_dynamic_state_qualification'],
        label='state native qualification', verify_file=True)
    review = campaign.read_bound_json(Path(ref['path']), ref['sha256'])
    _require(review.get('schema_version') == 'gx1_dynamic_mtf_state_private_root_review_v1'
        and review.get('status') == 'PASS_CONSUMED_DYNAMIC_STATE_NATIVE_MECHANICS'
        and review.get('test_data_used') is False and review.get('qualification_weights_for_main_fit') is False,
        'STATE_NATIVE_REVIEW')
    refs = {key:campaign.require_binding(review[key], label='state qualification ' + key, verify_file=True)
        for key in ('plan', 'result', 'receipt', 'initial_report', 'power_after')}
    plan = campaign.read_bound_json(Path(refs['plan']['path']), refs['plan']['sha256'])
    qualified_recipe = recipe
    if is_scoring_recovery(recipe):
        recovered = require_scoring_recovery_origin(recipe, source_repo=source_repo, source_commit=source_commit)
        qualified_recipe = recovered['original_recipe']
        # The exact closed fit already admitted these private receipts. Recheck their
        # bytes and evidence; only the proved control changes may differ now.
        prior = campaign.read_bound_json(Path(plan['recipe']['path']), plan['recipe']['sha256'])
        _require(prior.get('recipe_sha256') == campaign.canonical_sha256(
            {k:v for k,v in prior.items() if k != 'recipe_sha256'}), 'STATE_NATIVE_HISTORICAL_RECIPE_SEAL')
    else:
        prior = private.require_research_launch_recipe(plan['recipe'], source_repo=source_repo, source_commit=plan['source_commit'])
    result = campaign.read_bound_json(Path(refs['result']['path']), refs['result']['sha256'])
    receipt = campaign.read_bound_json(Path(refs['receipt']['path']), refs['receipt']['sha256'])
    initial = campaign.read_bound_json(Path(refs['initial_report']['path']), refs['initial_report']['sha256'])
    power = campaign.read_bound_json(Path(refs['power_after']['path']), refs['power_after']['sha256'])
    audit = campaign.read_bound_json(Path(recipe['label_audit']['path']), recipe['label_audit']['sha256'])
    _require(prior['run_scope'] == private.DYNAMIC_STATE_PRIVATE_SCOPE
        and all(prior[key] == qualified_recipe[key] for key in ('input_manifest', 'target_manifest', 'target_sources',
            'label_audit', 'fold', 'source_owners', 'full_feature_profile'))
        and {k:v for k,v in prior['numerical_recipe'].items() if k not in ('max_updates', 'contract_sha256')}
            == {k:v for k,v in recipe['numerical_recipe'].items() if k not in ('max_updates', 'contract_sha256')},
        'STATE_NATIVE_SAME_SOURCES_AND_OBJECTIVE')
    _require(result.get('schema_version') == 'gx1_dynamic_mtf_state_private_qualification_result_v1'
        and result.get('decision') == 'PASS_DYNAMIC_STATE_NATIVE_MECHANICS'
        and result.get('recipe') == plan['recipe'] and result.get('initial_report') == refs['initial_report']
        and result.get('initial_checked') is result.get('bit_exact_resume') is result.get('source_unchanged') is True
        and result.get('all_declared_gradient_groups_contacted') is True
        and result.get('test_data_used') is result.get('private_weights_for_main_fit') is result.get('predictive_value_qualified') is False
        and result.get('target_manifest') == recipe['target_manifest'] and result.get('target_sources') == recipe['target_sources']
        and result.get('label_audit') == recipe['label_audit']
        and result.get('planned_unique_parents') == 32 and result.get('planned_logical_optimizer_steps') == 2
        and result.get('physical_optimizer_updates') == 4 and result.get('parent_cursor') == 32
        and result.get('forbidden_open_paths') == [] and result.get('error') is None
        and result.get('own_prefix_prior') == initial.get('own_prefix_prior') == audit['prior']
        and receipt.get('outcome') == 'COMPLETE' and receipt.get('plan_sha256') == plan['plan_sha256']
        and receipt.get('guard_decision') == 'PASS'
        and receipt.get('trainer_guard_exit_code') == receipt.get('progress_observer_exit_code') == 0
        and receipt.get('test_data_used') is False
        and initial.get('schema_version') == 'gx1_dynamic_mtf_state_measurement_v1'
        and initial.get('role') == 'INITIAL_TRAIN' and initial.get('fold') == recipe['fold']
        and initial.get('target_manifest') == recipe['target_manifest']
        and initial.get('research_objective_sha256') == recipe['numerical_recipe']['research_objective_sha256']
        and initial.get('initial_grad_nograd_parity') is True and initial.get('test_data_used') is False,
        'STATE_NATIVE_ACTUAL_EVIDENCE')
    private.require_dynamic_state_private_counts(result.get('counts'))
    from gx1.scripts.run_entry_native_direction_dense_qualification_v1 import require_dynamic_state_gradient_contact
    require_dynamic_state_gradient_contact(result.get('steps'))
    support = result.get('batch_input_support')
    _require(isinstance(support,list) and len(support) == 2
        and [row for item in support for row in item.get('parent_rows',[])] == prior['qualification_parent_rows'],
        'STATE_QUALIFIED_PARENT_IDENTITY')
    gpu = power.get('gpu')
    _require(isinstance(gpu,list) and len(gpu) == 1 and isinstance(gpu[0],str), 'STATE_PRIVATE_POWER')
    fields = [value.strip() for value in gpu[0].split(',')]
    try:
        actual_power = float(fields[1])
    except (IndexError,TypeError,ValueError):
        actual_power = None
    keepers = power.get('keeper')
    _require(power.get('task_state') == 'Disabled' and power.get('configured_power_limit_w') == actual_power == 250
        and len(fields) == 6 and fields[0] == plan.get('gpu_uuid')
        and isinstance(keepers,list) and len(keepers) == 2
        and all(isinstance(item,Mapping) and type(item.get('State')) is int and item['State'] == 4 for item in keepers)
        and {item.get('TaskName') for item in keepers} == {'GX1GpuPowerLimit','GX1HostTelemetryBridge'}
        and power.get('test_data_used') is False, 'STATE_PRIVATE_POWER')
    return review

def _require_decimal_source_equivalence(source_repo, qualified_commit, qualified_owners, owners):
    """Prove qualified data/feature functions survived the full-fit scope addition."""
    import ast
    import subprocess
    _require(type(qualified_commit) is str and campaign._COMMIT.fullmatch(qualified_commit) is not None,
        'DECIMAL_QUALIFIED_COMMIT')
    for path, names in DECIMAL_TRANSITION_FUNCTIONS.items():
        before_ref=qualified_owners.get(path);after_ref=owners.get(path)
        _require(isinstance(before_ref,Mapping) and isinstance(after_ref,Mapping)
            and before_ref.get('path')==after_ref.get('path')==str(source_repo/path), 'DECIMAL_TRANSITION_PATH:'+path)
        before=subprocess.check_output(['git','-C',str(source_repo),'show',qualified_commit+':'+path])
        after=(source_repo/path).read_bytes()
        _require(hashlib.sha256(before).hexdigest()==before_ref.get('sha256')
            and hashlib.sha256(after).hexdigest()==after_ref.get('sha256'), 'DECIMAL_TRANSITION_SHA:'+path)
        def functions(raw):
            module=ast.parse(raw,filename=path)
            return {n.name:ast.dump(n,include_attributes=False) for n in module.body
                if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names}
        previous=functions(before);current=functions(after)
        _require(set(previous)==set(current)==set(names) and previous==current, 'DECIMAL_TRANSITION_FUNCTION:'+path)
        allowed=DECIMAL_TRANSITION_ALLOWED[path]
        changed=set(allowed['changed_functions']);new_functions=set(allowed['new_functions'])
        new_assignments=set(allowed['new_assignments'])
        def unchanged_module(raw, *, original):
            module=ast.parse(raw,filename=path);kept=[];found_functions=[];found_assignments=[]
            for node in module.body:
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in changed|new_functions:
                    found_functions.append(node.name);continue
                if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) \
                        and node.targets[0].id in new_assignments:
                    found_assignments.append(node.targets[0].id);continue
                kept.append(ast.dump(node,include_attributes=False))
            expected_functions=changed if original else changed|new_functions
            expected_assignments=set() if original else new_assignments
            _require(len(found_functions)==len(expected_functions) and set(found_functions)==expected_functions
                and len(found_assignments)==len(expected_assignments) and set(found_assignments)==expected_assignments,
                'DECIMAL_TRANSITION_NODE_SET:'+path)
            return kept
        _require(unchanged_module(before,original=True)==unchanged_module(after,original=False),
            'DECIMAL_TRANSITION_MODULE:'+path)


SCORING_RECOVERY_HEAD = '78436e71296b8ef815e15fe9998562b956b19430'
SCORING_RECOVERY_STATE_SEAL = 'acd03406a41c968b2346b49edf9995b29ea679b3f710d3637ad889ac0e75097f'
SCORING_RECOVERY_CLOSURE = {
    'path':'/home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19/FIT_SESSION_REGIME_001/INTERRUPTION_CLOSURE_REVIEW_001.json',
    'sha256':'fe507751a727533ee1d86ac210d2a79cdcc4fab9b29ea80a14bb9b11c0e473d8'}
CHART_SCORING_RECOVERY_IDENTITY = {
    'origin_schema':'gx1_family_chart_scoring_recovery_origin_v1',
    'source_head':'3ac1e688a54d0df7384d6aa6212513b264a63466',
    'state_seal':'0319a0522905bbbe4ca3dde04e51bc247df2c5ff6ed75743c736768450740bbf',
    'closure_review':{
        'path':'/home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19/FIT_CHART_GEOMETRY_001/INTERRUPTION_CLOSURE_REVIEW_001.json',
        'sha256':'2a7cebe5e101b019427aafbbecbf4594ff23e51b2900062fb0d9778c85d3d48f'},
    'integrity_schema':'gx1_family_interrupted_scoring_integrity_v1',
    'peer_schema':'gx1_family_interruption_independent_metadata_peer_review_v1',
    'closed_status':'INTERRUPTED_CONSUMED_W2_WINDOWS_HYPERVISOR_BUGCHECK_W1_TRAINING_PRESERVED',
    'closed_policy_git':{'commit':'d58040e50f05b8b6455e92526774b5465c37d859',
        'path':'NEXT_RUN_POLICY.json','sha256':'ea7c4a023dae0afa987b6008c760d8ee04772643ba7a7998c02c5e8d5f50a975'},
}
SCORING_RECOVERY_CONTROL_CHANGES = {
    'gx1/contracts/entry_native_direction_prefix_fit_campaign_v1.py': (
        ('require_prefix_fit_launch_recipe', 'require_prefix_fit_invocation_execution', 'require_prefix_fit_plan',
         'require_native_direction_prefix_fit_readiness', 'require_prefix_fit_progress'),
        ('is_scoring_recovery', 'require_scoring_recovery_origin', '_require_scoring_recovery_source_equivalence',
          'scoring_recovery_identity'),
        ('SCORING_RECOVERY_HEAD', 'SCORING_RECOVERY_STATE_SEAL', 'SCORING_RECOVERY_CLOSURE', 'SCORING_RECOVERY_CONTROL_CHANGES',
         'CHART_SCORING_RECOVERY_IDENTITY')),
    'gx1/contracts/local_random_access_campaign_v2.py': (('require_invocation',), (), ()),
    'gx1/scripts/run_entry_native_direction_prefix_fit_window_v1.py': (('run_native_direction_prefix_fit_window',), (), ()),
    'gx1/scripts/run_entry_native_direction_research_window_v1.py': (('_install_read_guard',), (), ()),
}


DYNAMIC_SCORING_RECOVERY_IDENTITY = {
    'origin_schema':'gx1_dynamic_state_terminal_scoring_recovery_origin_v1',
    'source_head':'6b7993ca4659f3f369b84bddec79d6ed987f0631',
    'state_seal':'1d4aa534ded06d08c35c8f4e027c18751f6a365552c69a33b9ad22f8ee729643',
    'closure_review':{
        'path':'/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_PREPARATION_20261102_001/FIT_ONEPASS_001/TERMINAL_FAILURE_CLOSURE.json',
        'sha256':'6bbda612b6eb65b5b15197fea1bba9158e5ff4e767c2e39abd9fc91c3e714108'},
}
DYNAMIC_SCORING_RECOVERY_CONTROL_CHANGES = {
    'gx1/contracts/entry_native_direction_prefix_fit_campaign_v1.py': (
        ('require_dynamic_state_launch_recipe', 'require_dynamic_state_qualification', 'scoring_recovery_identity',
         '_require_scoring_recovery_source_equivalence', 'require_scoring_recovery_origin', 'require_prefix_fit_cursor'),
        ('_require_dynamic_state_scoring_recovery_origin',),
        ('DYNAMIC_SCORING_RECOVERY_IDENTITY', 'DYNAMIC_SCORING_RECOVERY_CONTROL_CHANGES')),
    'gx1/scripts/run_entry_native_direction_research_window_v1.py': (('_install_read_guard',), (), ()),
}


def is_scoring_recovery(recipe):
    """Presence selects the strict recovery validator, never grants authority."""
    return 'scoring_recovery_origin' in recipe


def scoring_recovery_identity(family_arm):
    """Only exact observed, closed interruptions have recovery identities."""
    if family_arm == DYNAMIC_STATE_SCOPE:
        return dict(DYNAMIC_SCORING_RECOVERY_IDENTITY)
    if family_arm == 'chart_geometry_encoder':
        return dict(CHART_SCORING_RECOVERY_IDENTITY)
    _require(family_arm == 'session_regime_encoder', 'RECOVERY_ARM')
    return {
        'origin_schema':'gx1_family_session_scoring_recovery_origin_v1',
        'source_head':SCORING_RECOVERY_HEAD, 'state_seal':SCORING_RECOVERY_STATE_SEAL,
        'closure_review':SCORING_RECOVERY_CLOSURE,
        'integrity_schema':'gx1_session_interrupted_scoring_integrity_v1',
        'peer_schema':'gx1_session_interruption_independent_metadata_peer_review_v1',
        'closed_status':'INTERRUPTED_CONSUMED_W2_WINDOWS_BUGCHECK_W1_TRAINING_PRESERVED',
    }


def _require_scoring_recovery_source_equivalence(source_repo, original, current):
    """Only named launch/control definitions may change; numerical source is byte-identical."""
    import ast
    import subprocess
    dynamic = original.get('run_scope') == DYNAMIC_STATE_SCOPE
    identity = scoring_recovery_identity(DYNAMIC_STATE_SCOPE if dynamic else original.get('family_arm'))
    chart = original.get('family_arm') == 'chart_geometry_encoder'
    control_changes = DYNAMIC_SCORING_RECOVERY_CONTROL_CHANGES if dynamic else ({
        'gx1/contracts/entry_native_direction_prefix_fit_campaign_v1.py': (
            ('require_prefix_fit_launch_recipe', 'require_scoring_recovery_origin',
             '_require_scoring_recovery_source_equivalence'),
            ('scoring_recovery_identity',),
            ('SCORING_RECOVERY_CONTROL_CHANGES', 'CHART_SCORING_RECOVERY_IDENTITY')),
        'gx1/scripts/run_entry_native_direction_prefix_fit_window_v1.py':
            (('run_native_direction_prefix_fit_window',), (), ()),
        'gx1/scripts/run_entry_native_direction_research_window_v1.py':
            (('_install_read_guard',), (), ()),
    } if chart else SCORING_RECOVERY_CONTROL_CHANGES)
    before_owners, after_owners = original['source_owners'], current['source_owners']
    _require(set(before_owners) == set(after_owners), 'RECOVERY_SOURCE_SET')
    for path, before_ref in before_owners.items():
        after_ref = after_owners[path]
        _require(before_ref['path'] == after_ref['path'] == str(source_repo/path), 'RECOVERY_SOURCE_PATH')
        after = (source_repo/path).read_bytes()
        _require(hashlib.sha256(after).hexdigest() == after_ref['sha256'], 'RECOVERY_SOURCE_SHA:'+path)
        if path not in control_changes:
            _require(before_ref == after_ref, 'RECOVERY_NUMERICAL_SOURCE_CHANGED:'+path)
            continue
        before = subprocess.check_output(['git','-C',str(source_repo),'show',identity['source_head']+':'+path])
        _require(hashlib.sha256(before).hexdigest() == before_ref['sha256'], 'RECOVERY_ORIGINAL_SOURCE_SHA:'+path)
        changed, added, assignments = map(set, control_changes[path])
        # Chart's original source already includes the qualified session recovery.
        old_added = added - {'scoring_recovery_identity'} if chart else set()
        old_assignments = assignments - {'CHART_SCORING_RECOVERY_IDENTITY'} if chart else set()
        def unchanged(raw, old):
            text = raw.decode('utf-8'); nodes = ast.parse(text).body
            kept=[]; functions=set(); constants=set()
            for node in nodes:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in changed|added:
                    _require(node.name not in functions, 'RECOVERY_DUPLICATE_FUNCTION')
                    functions.add(node.name)
                elif isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in assignments:
                    constants.add(node.targets[0].id)
                else:
                    kept.append(ast.get_source_segment(text,node))
            _require(functions == (changed|old_added if old else changed|added)
                and constants == (old_assignments if old else assignments), 'RECOVERY_CONTROL_NODE_SET:'+path)
            return kept
        _require(unchanged(before,True) == unchanged(after,False), 'RECOVERY_NONCONTROL_SOURCE_CHANGED:'+path)
    if chart or dynamic:
        _require({path for path in before_owners if before_owners[path] != after_owners[path]} == set(control_changes),
            'RECOVERY_CHANGED_OWNER_SET')


def _require_dynamic_state_scoring_recovery_origin(recipe, *, source_repo, source_commit):
    """One closed terminal-validator failure; no receipt or new training is invented."""
    identity = scoring_recovery_identity(DYNAMIC_STATE_SCOPE)
    origin = recipe.get('scoring_recovery_origin')
    fields = {'schema_version', 'original_source_head', 'original_recipe', 'original_plan',
        'training_cursor', 'state_integrity', 'closure_review'}
    _require(isinstance(origin, Mapping) and set(origin) == fields
        and origin['schema_version'] == identity['origin_schema']
        and origin['original_source_head'] == identity['source_head']
        and origin['closure_review'] == identity['closure_review']
        and recipe['run_scope'] == DYNAMIC_STATE_SCOPE and recipe['max_windows'] == 1
        and recipe['training_enabled'] is recipe['full_epoch_training_allowed'] is False,
        'STATE_RECOVERY_ORIGIN_ROLE')
    def read(ref, label):
        bound = campaign.require_binding(ref, label=label, verify_file=True)
        return campaign.read_bound_json(Path(bound['path']), bound['sha256'])
    closure = read(origin['closure_review'], 'state recovery closure')
    _require(closure['schema_version'] == 'gx1_dynamic_state_terminal_failure_closure_v1'
        and closure['status'] == 'CLOSED_COMPLETED_TRAINING_TERMINAL_VALIDATOR_FAILED'
        and closure['original_source_head'] == identity['source_head']
        and closure['state_seal'] == identity['state_seal']
        and closure['native_exit_code_known'] is True and closure['native_exit_code'] == 1
        and closure['ordinary_receipt_present'] is closure['original_relaunch_allowed'] is False
        and closure['originals_modified'] is closure['recovery_execution_authorized'] is closure['test_data_used'] is False
        and closure['training_state_preserved'] is True and closure['active_native_pids'] == []
        and closure['revoked_invocations'] == [2, 3, 4], 'STATE_RECOVERY_CLOSED_ORIGINAL')
    refs = closure['references']
    _require(origin['state_integrity'] == closure['training_integrity']
        and all(origin[key] == refs[key] for key in ('original_recipe', 'original_plan', 'training_cursor')),
        'STATE_RECOVERY_CLOSURE_REFS')
    integrity = read(origin['state_integrity'], 'state training integrity')
    _require(integrity['schema_version'] == 'gx1_dynamic_state_training_integrity_v1'
        and integrity['status'] == 'TRAINING_STATE_BYTES_PRESERVED_TERMINAL_VALIDATOR_FAILED'
        and integrity['references'] == refs and integrity['original_source_head'] == identity['source_head']
        and integrity['state_seal'] == identity['state_seal']
        and integrity['completed_training_updates'] == 2976 and integrity['training_parents'] == 47602
        and integrity['training_passes'] == 1 and integrity['training_guard_exit_code'] == 1
        and integrity['new_model_forwards'] == integrity['new_optimizer_updates'] == 0
        and integrity['full_train_scoring_complete'] is integrity['later_scoring_complete'] is False
        and integrity['ordinary_receipt_present'] is integrity['observer_exit_code_known'] is False
        and integrity['observer_exit_code'] is None and integrity['test_data_used'] is False,
        'STATE_RECOVERY_INTEGRITY')
    policy = read(closure['closed_policy'], 'state recovery closed policy')
    gate = policy[closure['closed_stage']]
    _require(gate['status'] == 'CONSUMED_TRAINING_COMPLETE_TERMINAL_VALIDATOR_FAILED'
        and gate['recipe'] == origin['original_recipe'] and gate['claimed_invocations'] == gate['failed_invocations'] == 1
        and gate['completed_invocations'] == 0 and gate['unclaimed_invocations_revoked'] == [2, 3, 4]
        and all(gate[key] is False for key in ('execution_authorized', 'training_enabled', 'native_training_enabled',
            'full_epoch_training_allowed', 'native_launch_authorized', 'genuine_updates_authorized', 'relaunch_allowed'))
        and all(policy[key] is False for key in ('training_enabled', 'native_training_enabled', 'full_epoch_training_allowed')),
        'STATE_RECOVERY_CLOSED_POLICY')
    original = read(origin['original_recipe'], 'state recovery original recipe')
    _require(original['schema_version'] == RECIPE_SCHEMA and original['run_scope'] == DYNAMIC_STATE_SCOPE
        and original['recipe_sha256'] == campaign.canonical_sha256({k:v for k,v in original.items() if k != 'recipe_sha256'})
        and original['train_passes'] == 1 and original['max_windows'] == 4
        and original['training_enabled'] is original['full_epoch_training_allowed'] is True,
        'STATE_RECOVERY_ORIGINAL_RECIPE')
    allowed = {'recipe_sha256', 'source_owners', 'runtime_root', 'admission_stage', 'execution_binding_path',
        'max_windows', 'training_enabled', 'full_epoch_training_allowed', 'scoring_recovery_origin'}
    _require({k:v for k,v in original.items() if k not in allowed} ==
        {k:v for k,v in recipe.items() if k not in allowed}, 'RECOVERY_NUMERICAL_RECIPE_CHANGED')
    old_runtime, new_runtime = Path(original['runtime_root']), Path(recipe['runtime_root'])
    _require(new_runtime != old_runtime and old_runtime not in new_runtime.parents and new_runtime not in old_runtime.parents
        and recipe['admission_stage'] != original['admission_stage'], 'RECOVERY_NEW_RUNTIME')
    plan = read(origin['original_plan'], 'state recovery original plan')
    execution = read(refs['execution_binding'], 'state recovery original execution')
    _require(plan['source_commit'] == execution['source_head'] == identity['source_head']
        and plan['recipe'] == origin['original_recipe'] and plan['runtime_root'] == str(old_runtime)
        and plan['plan_sha256'] == campaign.canonical_sha256({k:v for k,v in plan.items() if k != 'plan_sha256'})
        and execution['plan'] == origin['original_plan'], 'STATE_RECOVERY_ORIGINAL_PLAN')
    cursor = require_prefix_fit_cursor(read(origin['training_cursor'], 'state recovery terminal cursor'),
        expected_recipe=origin['original_recipe'], expected_fold=recipe['fold'], verify_files=True)
    result = read(cursor['result'], 'state recovery terminal result')
    _require(cursor['parent_cursor'] == result['parent_cursor'] == 47602
        and cursor['optimizer_updates'] == result['optimizer_updates'] == 2976
        and cursor['phase_state'] == result['phase_state'] == 'SCORING'
        and cursor['outcome'] == result['outcome'] == 'RESUMABLE'
        and cursor['oof_cursor'] == 0 and cursor['final_train_measurement'] is None
        and cursor['full_state'] == refs['full_state'] == result['durable_checkpoint']['state']
        and cursor['own_initial_state'] == refs['own_initial_state']
        and cursor['initial_measurement'] == refs['initial_measurement']
        and cursor['result'] == refs['training_result'] and result['recipe'] == origin['original_recipe']
        and result['durable_checkpoint']['contract_sha256'] == identity['state_seal']
        and result['counts'] == integrity['counts'] and result['source_unchanged'] is True
        and result['test_data_used'] is False and result['error'] is None, 'STATE_RECOVERY_TRAINING_STATE')
    _require_scoring_recovery_source_equivalence(source_repo, original, recipe)
    return {'origin':dict(origin), 'original_recipe':original, 'cursor':cursor, 'identity':identity}


def require_scoring_recovery_origin(recipe, *, source_repo, source_commit):
    """Validate either exact observed interruption; no generic checkpoint import."""
    import subprocess
    if recipe.get('run_scope') == DYNAMIC_STATE_SCOPE:
        return _require_dynamic_state_scoring_recovery_origin(recipe, source_repo=source_repo, source_commit=source_commit)
    identity = scoring_recovery_identity(recipe.get('family_arm'))
    origin = recipe.get('scoring_recovery_origin')
    fields = {'schema_version','family_arm','original_source_head','original_recipe','original_plan',
        'training_receipt','training_cursor','training_review','state_integrity','interruption_review','closure_review'}
    _require(isinstance(origin, Mapping) and set(origin)==fields
        and origin['schema_version']==identity['origin_schema']
        and origin['family_arm']==recipe.get('family_arm')
        and recipe.get('run_scope')==FAMILY_SIGN_SCOPE
        and origin['original_source_head']==identity['source_head']
        and source_commit != identity['source_head']
        and origin['closure_review']==identity['closure_review'], 'RECOVERY_ORIGIN_ROLE')
    def read(ref, label):
        bound=campaign.require_binding(ref,label=label,verify_file=True)
        return campaign.read_bound_json(Path(bound['path']),bound['sha256'])
    closure=read(origin['closure_review'],'recovery closure')
    _require(closure['schema_version']=='gx1_family_interrupted_scoring_closure_v1'
        and closure['status']=='CLOSED_INTERRUPTED_SCORING_ORIGINALS_PRESERVED'
        and closure['family_arm']==origin['family_arm'] and closure['original_source_head']==identity['source_head']
        and closure['original_relaunch_allowed'] is False and closure['training_state_preserved'] is True
        and closure['recovery_execution_authorized'] is False and closure['test_data_used'] is False
        and closure['interrupted_native_exit_code_known'] is False and closure['interrupted_native_exit_code'] is None
        and closure['active_native_pids']==[] and closure['revoked_invocations']==[3,4], 'RECOVERY_CLOSED_ORIGINAL')
    for key, closure_key in (('state_integrity','state_integrity'),('training_review','training_window_review'),
            ('interruption_review','interruption_review')):
        _require(origin[key]==closure[closure_key], 'RECOVERY_CLOSURE_REF:'+key)
    integrity=read(origin['state_integrity'],'recovery integrity')
    peer=read(origin['interruption_review'],'recovery peer')
    review=read(origin['training_review'],'recovery training review')
    _require(integrity['schema_version']==identity['integrity_schema']
        and integrity['status']=='TRAINING_STATE_BYTES_PRESERVED_SCORING_INTERRUPTED'
        and integrity['source_head']==identity['source_head'] and integrity['native_process_absent'] is True
        and integrity['originals_modified'] is False and integrity['model_or_checkpoint_deserialized'] is False
        and integrity['completed_training_updates']==2976 and integrity['training_parents']==47602
        and integrity['full_train_scoring_complete'] is False and integrity['later_scoring_complete'] is False
        and integrity['new_model_forwards']==integrity['new_optimizer_updates']==0
        and integrity['test_data_used'] is False, 'RECOVERY_INTEGRITY')
    _require(peer['schema_version']==identity['peer_schema']
        and peer['status']=='PASS_SCOPED_METADATA_WORKER_REVIEW'
        and peer['references']['integrity']==origin['state_integrity'], 'RECOVERY_PEER')
    if origin['family_arm']=='chart_geometry_encoder':
        _require(integrity.get('family_arm')==peer.get('family_arm')==origin['family_arm']
            and peer.get('test_data_used') is False
            and peer.get('original_native_exit_code_known') is False
            and peer.get('original_native_exit_code') is None, 'RECOVERY_CHART_PEER')
    _require(review['schema_version']=='gx1_family_training_window_review_v1'
        and review['status']=='PASS_TRAINING_COMPLETE_SCORING_PENDING'
        and review['source_head']==identity['source_head'] and review['family_arm']==origin['family_arm']
        and review['optimizer_updates']==2976 and review['train_parents']==47602 and review['training_passes']==1
        and review['full_train_scored'] is False and review['later_scored'] is False
        and review['test_data_used'] is False, 'RECOVERY_TRAINING_REVIEW')
    closed=closure['closed_policy_git']
    if origin['family_arm']=='chart_geometry_encoder':
        _require(closed==identity['closed_policy_git'], 'RECOVERY_CHART_CLOSED_POLICY')
    raw=subprocess.check_output(['git','-C',str(source_repo),'show',closed['commit']+':'+closed['path']])
    _require(closed['commit']==closure['closure_source_head'] and closed['path']=='NEXT_RUN_POLICY.json'
        and hashlib.sha256(raw).hexdigest()==closed['sha256'], 'RECOVERY_CLOSED_POLICY_BYTES')
    policy=json.loads(raw); gate=policy[closure['closed_stage']]
    _require(gate['status']==identity['closed_status']
        and gate['claimed_invocations']==2 and gate['completed_invocations']==1
        and gate['interrupted_invocations']==1 and gate['unclaimed_invocations_revoked']==[3,4]
        and all(gate[key] is False for key in ('execution_authorized','training_enabled','native_training_enabled',
            'full_epoch_training_allowed','native_launch_authorized','genuine_updates_authorized','relaunch_allowed')),
        'RECOVERY_CLOSED_POLICY')
    _require(origin['original_recipe']==gate['recipe'], 'RECOVERY_ORIGINAL_RECIPE_REF')
    original=read(origin['original_recipe'],'original recipe')
    _require(original['recipe_sha256']==campaign.canonical_sha256({k:v for k,v in original.items() if k!='recipe_sha256'})
        and original['run_scope']==FAMILY_SIGN_SCOPE and original['family_arm']==origin['family_arm']
        and original['train_passes']==1 and original['max_windows']==4
        and original['training_enabled'] is True and original['full_epoch_training_allowed'] is True, 'RECOVERY_ORIGINAL_RECIPE')
    allowed={'recipe_sha256','source_owners','runtime_root','admission_stage','execution_binding_path',
        'max_windows','training_enabled','full_epoch_training_allowed','scoring_recovery_origin'}
    _require({k:v for k,v in original.items() if k not in allowed} ==
        {k:v for k,v in recipe.items() if k not in allowed}, 'RECOVERY_NUMERICAL_RECIPE_CHANGED')
    old_runtime=Path(original['runtime_root']); new_runtime=Path(recipe['runtime_root'])
    _require(new_runtime != old_runtime and old_runtime not in new_runtime.parents and new_runtime not in old_runtime.parents
        and recipe['admission_stage'] != original['admission_stage'], 'RECOVERY_NEW_RUNTIME')
    execution=read(review['execution_binding'],'original execution binding')
    _require(execution['plan']==origin['original_plan'] and execution['source_head']==identity['source_head'],
        'RECOVERY_ORIGINAL_PLAN_REF')
    plan=read(origin['original_plan'],'original plan')
    _require(plan['source_commit']==identity['source_head'] and plan['recipe']==origin['original_recipe']
        and plan['runtime_root']==str(old_runtime)
        and plan['plan_sha256']==campaign.canonical_sha256({k:v for k,v in plan.items() if k!='plan_sha256'}),
        'RECOVERY_ORIGINAL_PLAN')
    _require(origin['training_receipt']==integrity['training_receipt']==review['receipt'], 'RECOVERY_TRAINING_RECEIPT_REF')
    receipt=read(origin['training_receipt'],'training receipt')
    _require(receipt['schema_version']==campaign.RECEIPT_SCHEMA
        and receipt['receipt_sha256']==campaign.canonical_sha256({k:v for k,v in receipt.items() if k!='receipt_sha256'})
        and receipt['plan_sha256']==plan['plan_sha256'] and receipt['invocation_number']==1
        and receipt['outcome']=='RESUMABLE' and receipt['guard_decision']=='PASS'
        and receipt['trainer_guard_exit_code']==receipt['progress_observer_exit_code']==0
        and receipt['checkpoint_pointer_snapshot']==origin['training_cursor']
        and origin['training_cursor']['path']==str(old_runtime/'invocation-evidence/invocation-0001/CHECKPOINT_POINTER.json')
        and origin['training_cursor']['sha256']==integrity['cursor']['sha256']==review['cursor']['sha256'],
        'RECOVERY_TRAINING_RECEIPT')
    cursor=require_prefix_fit_cursor(read(origin['training_cursor'],'immutable training cursor'),
        expected_recipe=origin['original_recipe'],expected_fold=recipe['fold'],verify_files=True)
    result=read(cursor['result'],'training result')
    _require(cursor['parent_cursor']==47602 and cursor['optimizer_updates']==2976
        and cursor['phase_state']=='SCORING' and cursor['outcome']=='RESUMABLE'
        and cursor['oof_cursor']==0 and cursor['final_train_measurement'] is None
        and cursor['full_state']==integrity['full_state']==review['full_state']
        and cursor['own_initial_state']==integrity['own_initial_state']
        and cursor['result']==integrity['training_result']==review['result']
        and result['durable_checkpoint']['contract_sha256']==identity['state_seal']
        and result['durable_checkpoint']['state']==cursor['full_state']
        and result['source_unchanged'] is True and result['error'] is None, 'RECOVERY_TRAINING_STATE')
    _require_scoring_recovery_source_equivalence(source_repo,original,recipe)
    return {'origin':dict(origin),'original_recipe':original,'cursor':cursor,'identity':identity}


def _require_family_recipe_fields(recipe):
    # Metadata only. Actual canonical arm closure is checked against the bound
    # TRAIN normalization during input preparation and private qualification.
    from gx1.features.entry_specialist_feature_groups_v1 import MODEL_NATIVE_TRAINING_SPECIALISTS
    _require(type(recipe.get('family_arm')) is str
        and recipe['family_arm'] in ('baseline', 'macro_context', *MODEL_NATIVE_TRAINING_SPECIALISTS), 'FAMILY_ARM')
    campaign._sha(recipe.get('family_intervention_sha256'), 'family intervention')
    _require(type(recipe.get('final_train_measurement_rows')) is int
        and recipe['final_train_measurement_rows']==recipe['fold']['prefix_rows']==47602, 'FAMILY_FULL_TRAIN_ROWS')
    campaign.require_binding(recipe.get('family_initial_report'), label='family initial report', verify_file=True)


def _require_family_protocol(protocol, recipe):
    _require(isinstance(protocol, Mapping)
        and protocol.get('schema_version')=='gx1_current_family_added_value_study_v1'
        and protocol.get('fixed_fold')==recipe['fold']
        and protocol.get('targets')==recipe['dense_target']
        and protocol.get('horizons_minutes')==list(range(1,61))
        and protocol.get('primary_horizon_minutes')==5
        and protocol.get('test_data_used') is False and protocol.get('exit_evaluation') is False
        and protocol.get('broker_access_allowed') is False, 'FAMILY_PROTOCOL_SCOPE')
    comparison=protocol.get('comparison',{}); fit=comparison.get('fixed_training',{})
    _require(fit.get('same_initial_parameters_buffers_rng_order') is True
        and fit.get('passes')==recipe['train_passes']==1 and fit.get('parents')==recipe['fold']['prefix_rows']
        and fit.get('max_updates_per_arm')==2976 and fit.get('numerical_recipe')==recipe['numerical_recipe']
        and comparison.get('maximum_refit_arms')==10, 'FAMILY_PROTOCOL_FIT')
    execution=protocol.get('execution',{})
    _require(execution.get('one_heavy_job') is True and execution.get('capped_runner') is True
        and execution.get('signed_gpu_guards') is True and execution.get('source_frozen') is True
        and execution.get('gpu_limit_w')==250 and execution.get('no_hpo') is True
        and execution.get('no_extra_passes') is True, 'FAMILY_PROTOCOL_GUARDS')


def require_prefix_fit_launch_recipe(binding, *, source_repo, source_commit, verify_files=True):
    """Check exact metadata/source bytes; numerical270 validation stays in CUDA."""
    ref = campaign.require_binding(binding, label='prefix fit recipe', verify_file=verify_files)
    _require(verify_files, 'RECIPE_REQUIRES_PHYSICAL_METADATA')
    value = campaign.read_bound_json(Path(ref['path']), ref['sha256'])
    if value.get('run_scope') == DYNAMIC_STATE_SCOPE:
        return require_dynamic_state_launch_recipe(value, source_repo=source_repo, source_commit=source_commit, private_scope=False)
    fields = {'schema_version', 'source_repo', 'runtime_root', 'admission_stage', 'run_scope',
        'numerical_recipe', 'input_manifest', 'source_owners', 'execution_binding_path', 'fold',
        'train_passes', 'max_windows', 'parent_order_algorithm', 'initial_measurement_rows',
        'eval_batch_size', 'checkpoint_every_updates', 'max_checkpoint_bytes',
        'max_checkpoint_total_bytes', 'max_published_json_bytes', 'training_enabled',
        'full_epoch_training_allowed', 'test_data_used', 'predictive_value_claimed',
        'qualification_weights_for_main_fit'}
    bps = value.get('run_scope') == BPS_SCOPE
    legacy_continuation = value.get('run_scope') == CONTINUATION_SCOPE
    factor_continuation = value.get('run_scope') == M5_FACTOR_CONTINUATION_SCOPE
    factor_bps = value.get('run_scope') == M5_FACTOR_BPS_SCOPE
    family = value.get('run_scope') == FAMILY_SIGN_SCOPE
    recovery = is_scoring_recovery(value)
    if recovery:
        fields |= {'scoring_recovery_origin'}
    factor = value.get('run_scope') in (M5_FACTOR_SIGN_SCOPE, M5_FACTOR_BPS_SCOPE, M5_FACTOR_CONTINUATION_SCOPE)
    continuation = legacy_continuation or factor_continuation
    chronological = value.get('run_scope') == ORDER_SCOPE
    m1_fit = value.get('run_scope') == M1_PRICE_FIT_SCOPE
    priority_fit = value.get('run_scope') == M5_FIVE_PRIORITY_FIT_SCOPE
    m5_fit = value.get('run_scope') == M5_PRICE_FIT_SCOPE or priority_fit or factor or family
    decimal = value.get('run_scope') == DENSE_DECIMAL_FIT_SCOPE or m1_fit or m5_fit
    dense = value.get('run_scope') in (DENSE_SCOPE, DENSE_DECIMAL_FIT_SCOPE) or m1_fit or m5_fit
    if dense:
        fields |= {'gc_layout_metadata_parity','native_dense_qualification','dense_source_transition_proof','dense_target_result','dense_target','dense_evaluation_protocol'}
    if decimal:
        fields |= {'decimal_input_audit','decimal_target_refs','decimal_initial_predictions'}
    if m1_fit:
        fields |= {'candidate_mode','candidate_m1_refs','m1_pair_initial_parity'}
    if m5_fit:
        fields |= {'candidate_mode'}
    if priority_fit:
        fields |= {'priority_baseline_initial_report'}
    if family:
        fields |= {'family_arm', 'family_intervention_sha256', 'family_initial_report', 'final_train_measurement_rows'}
    if factor:
        fields |= {'factor_initial_report'}
        if factor_bps: fields |= {'dense_bps_scale'}
        if factor_continuation: fields |= {'continuation_origin'}
    if chronological:
        fields |= {'gc_layout_metadata_parity', 'order_origin', 'native_order_qualification'}
    if bps:
        fields |= {'gc_layout_metadata_parity', 'native_bps_qualification'}
    if legacy_continuation:
        fields |= {'gc_layout_metadata_parity', 'continuation_origin', 'native_continuation_qualification'}
    recipe = _identity(value, fields, RECIPE_SCHEMA, 'recipe_sha256')
    _require(not recovery or (family and recipe.get('family_arm') in ('session_regime_encoder','chart_geometry_encoder')), 'RECOVERY_SCOPE')
    _require(Path(_path(recipe['source_repo'], 'source repo')) == source_repo == CANONICAL_REPO
        and type(source_commit) is str and campaign._COMMIT.fullmatch(source_commit) is not None, 'RECIPE_SOURCE')
    runtime = Path(_path(recipe['runtime_root'], 'runtime root'))
    _require(type(recipe['admission_stage']) is str and recipe['admission_stage'].startswith('dynamic_m5_direction_')
        and recipe['run_scope'] in (BPS_SCOPE, CONTINUATION_SCOPE, ORDER_SCOPE, DENSE_SCOPE, DENSE_DECIMAL_FIT_SCOPE, M1_PRICE_FIT_SCOPE, M5_PRICE_FIT_SCOPE, M5_FIVE_PRIORITY_FIT_SCOPE, M5_FACTOR_SIGN_SCOPE, M5_FACTOR_BPS_SCOPE, M5_FACTOR_CONTINUATION_SCOPE, FAMILY_SIGN_SCOPE, 'single_prefix_one_pass_direction_fit'), 'RECIPE_SCOPE')
    fold = _fold(recipe['fold'])
    if bps:
        _require(fold == private.BPS_FOLD
            and recipe['gc_layout_metadata_parity'] == private.BPS_GC_PARITY
            and recipe['native_bps_qualification'] == BPS_NATIVE_QUALIFICATION, 'BPS_QUALIFIED_SCOPE')
        campaign.require_binding(recipe['gc_layout_metadata_parity'], label='bps GC metadata proof', verify_file=True)
    for key, expected in {'train_passes':2 if factor_continuation else 3 if legacy_continuation else 1, 'max_windows':1 if recovery else 4, 'initial_measurement_rows':256,
            'eval_batch_size':16, 'checkpoint_every_updates':256, 'max_checkpoint_bytes':1073741824,
            'max_checkpoint_total_bytes':4294967296, 'max_published_json_bytes':67108864}.items():
        _require(type(recipe[key]) is int and recipe[key] == expected, 'RECIPE_BUDGET:' + key)
    _require(recipe['parent_order_algorithm'] == (ORDER_ALGORITHM if chronological else PARENT_ORDER_ALGORITHM), 'ORDER_ALGORITHM')
    for key, expected in {'training_enabled':not recovery, 'full_epoch_training_allowed':not recovery,
            'test_data_used':False, 'predictive_value_claimed':False, 'qualification_weights_for_main_fit':False}.items():
        _require(recipe[key] is expected, 'RECIPE_AUTHORITY:' + key)
    numerical = recipe['numerical_recipe']
    _require(isinstance(numerical, Mapping)
        and numerical.get('signed_bps_supervision', False) is bps
        and numerical.get('dense_minute_supervision', False) is dense
        and numerical.get('dense_signed_bps_supervision', False) is factor_bps
        and numerical.get('magnitude_weighted_direction', False) is False
        and numerical.get('five_minute_priority', False) is priority_fit, 'NUMERICAL_OBJECTIVE_SCOPE')
    _require(isinstance(numerical, Mapping) and numerical.get('schema_version') == 'gx1_native_direction_research_recipe_v1'
        and type(numerical.get('batch_size')) is int and numerical['batch_size'] == BATCH_SIZE
        and type(numerical.get('max_updates')) is int and numerical['max_updates'] == (2 if factor_continuation else 3 if legacy_continuation else 1) * ((fold['prefix_rows'] + 15) // 16)
        and numerical.get('genuine_updates_authorized') is False and numerical.get('native_launch_authorized') is False,
        'NUMERICAL_RECIPE_ROLE')
    unsigned = {k:v for k,v in numerical.items() if k != 'contract_sha256'}
    declared = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    _require(numerical.get('contract_sha256') == declared, 'NUMERICAL_RECIPE_SEAL')
    campaign._sha(numerical.get('research_objective_sha256'), 'research objective')
    input_manifest = _input_manifest(recipe)
    _require(_path(recipe['execution_binding_path'], 'execution binding') == str(runtime/'EXECUTION_BINDING.json'),
        'EXECUTION_BINDING_PATH')
    owners = recipe['source_owners']
    required_sources = set(REQUIRED_SOURCE_PATHS) | (set(BPS_EXTRA_SOURCE_PATHS) if bps else set())
    if dense: required_sources |= set(private.DENSE_EXTRA_SOURCE_PATHS)
    if decimal: required_sources |= set(private.DECIMAL_EXTRA_SOURCE_PATHS)
    if m1_fit: required_sources |= set(private.M1_PRICE_EXTRA_SOURCE_PATHS) | {'gx1/contracts/entry_native_direction_m1_fullfit_v1.py'}
    if m5_fit: required_sources |= set(private.M1_PRICE_EXTRA_SOURCE_PATHS) | {'gx1/contracts/entry_native_direction_m5_fullfit_v1.py'}
    if factor: required_sources.add('gx1/contracts/entry_native_direction_bps_v1.py')
    if family: required_sources |= {FAMILY_INTERVENTION_SOURCE, 'gx1/scripts/compare_entry_direction_matched_exposure_v1.py'}
    if priority_fit:
        from gx1.contracts import entry_native_direction_five_priority_private_v1 as priority_private
        from gx1.contracts import entry_native_direction_five_priority_fullfit_v1 as priority_authority
        required_sources |= set(priority_private.EXTRA_SOURCE_PATHS) | {priority_authority.OWN_SOURCE}
    if continuation or chronological:
        required_sources.add('gx1/contracts/entry_native_direction_frozen_extraction_v1.py')
    _require(isinstance(owners, Mapping) and set(owners) == required_sources, 'SOURCE_OWNER_SET')
    for key, item in owners.items():
        checked = campaign.require_binding(item, label='prefix fit source', verify_file=True)
        _require(checked['path'] == str(source_repo/key), 'SOURCE_OWNER_PATH:' + key)
    if dense:
        _require(fold==private.BPS_FOLD and recipe['input_manifest']==private.DENSE_INPUT_MANIFEST
            and recipe['gc_layout_metadata_parity']==private.BPS_GC_PARITY, 'DENSE_FIXED_INPUT_SCOPE')
        if decimal:
            if m1_fit:
                from gx1.contracts import entry_native_direction_m1_fullfit_v1 as m1_authority
                _require(recipe['candidate_mode'] in m1_authority.QUALIFICATIONS
                    and recipe['native_dense_qualification']==m1_authority.QUALIFICATIONS[recipe['candidate_mode']]
                    and recipe['decimal_initial_predictions']==m1_authority.INITIALS[recipe['candidate_mode']]
                    and recipe['m1_pair_initial_parity']==m1_authority.PAIR_PARITY
                    and recipe['dense_source_transition_proof']==private.M1_PRICE_SOURCE_TRANSITION, 'M1_QUALIFIED_INPUT_SCOPE')
            elif family:
                _require(recipe['candidate_mode']=='price_only'
                    and recipe['dense_source_transition_proof']==private.M1_PRICE_SOURCE_TRANSITION, 'FAMILY_QUALIFIED_INPUT_SCOPE')
                _require_family_recipe_fields(recipe)
            elif factor:
                from gx1.contracts import entry_native_direction_m5_fullfit_v1 as m5_authority
                _require(recipe['candidate_mode']=='price_only'
                    and recipe['native_dense_qualification']==m5_authority.FACTOR_QUALIFICATION
                    and recipe['decimal_initial_predictions']==m5_authority.INITIAL
                    and recipe['dense_source_transition_proof']==private.M1_PRICE_SOURCE_TRANSITION, 'FACTOR_QUALIFIED_INPUT_SCOPE')
            elif priority_fit:
                _require(recipe['candidate_mode']=='price_only'
                    and recipe['native_dense_qualification']==priority_authority.QUALIFICATION
                    and recipe['decimal_initial_predictions']==priority_authority.m5.INITIAL
                    and recipe['dense_source_transition_proof']==private.M1_PRICE_SOURCE_TRANSITION, 'PRIORITY_QUALIFIED_INPUT_SCOPE')
            elif m5_fit:
                from gx1.contracts import entry_native_direction_m5_fullfit_v1 as m5_authority
                _require(recipe['candidate_mode']=='price_only'
                    and recipe['native_dense_qualification']==m5_authority.QUALIFICATION
                    and recipe['decimal_initial_predictions']==m5_authority.INITIAL
                    and recipe['dense_source_transition_proof']==private.M1_PRICE_SOURCE_TRANSITION, 'M5_QUALIFIED_INPUT_SCOPE')
            else:
                _require(recipe['native_dense_qualification']==DENSE_DECIMAL_NATIVE_QUALIFICATION
                    and recipe['decimal_initial_predictions']==private.DECIMAL_INITIAL_PREDICTIONS, 'DECIMAL_QUALIFIED_INPUT_SCOPE')
            _require(recipe['decimal_input_audit']==private.DECIMAL_INPUT_AUDIT, 'DECIMAL_INPUT_AUDIT_SCOPE')
            audit_ref=campaign.require_binding(recipe['decimal_input_audit'],label='decimal input audit',verify_file=True)
            audit=campaign.read_bound_json(Path(audit_ref['path']),audit_ref['sha256'])
            refs=recipe['decimal_target_refs']
            _require(audit.get('schema_version')=='gx1_decimal_bound_input_actual_audit_v1'
                and audit.get('status')=='COMPLETE' and audit.get('parents')==118249
                and audit.get('horizons_minutes')==list(range(1,61)) and audit.get('test_data_used') is False
                and audit.get('new_model_forwards')==audit.get('optimizer_steps')==0
                and audit.get('physical_parent_and_bar_start_hashes_verified') is True
                and audit.get('corrected_archive_and_receipt_links_verified') is True
                and audit.get('original_legacy_anchors_verified') is True and audit.get('immutable_outputs') is True
                and isinstance(refs,Mapping) and refs==audit.get('inputs')
                and refs.get('original')==private.DENSE_TARGET and refs.get('original_result')==private.DENSE_TARGET_RESULT
                and recipe['dense_target']==refs.get('corrected')
                and recipe['dense_target_result']==refs.get('corrected_result')
                and audit.get('source')==owners.get('gx1/contracts/entry_native_direction_decimal_input_v1.py'), 'DECIMAL_INPUT_EVIDENCE')
            for key,item in refs.items():campaign.require_binding(item,label='decimal '+key,verify_file=False)
            campaign.require_binding(recipe['decimal_initial_predictions'],label='decimal initial predictions',verify_file=True)
        else:
            _require(recipe['native_dense_qualification']==DENSE_NATIVE_QUALIFICATION
                and recipe['dense_target_result']==private.DENSE_TARGET_RESULT and recipe['dense_target']==private.DENSE_TARGET,
                'DENSE_ORIGINAL_INPUT_SCOPE')
        for key in ('gc_layout_metadata_parity','native_dense_qualification','dense_source_transition_proof','dense_target_result','dense_evaluation_protocol'):
            campaign.require_binding(recipe[key],label='dense '+key,verify_file=True)
        evaluation_name=('DIRECTION_FAMILY_ADDED_VALUE_PROTOCOL_20261030.json' if family else
            'DIRECTION_M5_SIGN_BPS_EXPOSURE_PROTOCOL_20261030.json' if factor else
            'DIRECTION_M5_FIVE_PRIORITY_DESIGN_20261029.json' if priority_fit else
            'DIRECTION_M5_PRICE_ONLY_PROTOCOL_20261029.json' if m5_fit else
            'DIRECTION_MATCHED_M1_M5_FULLFIT_PROTOCOL_20261029.json' if m1_fit else
            'DIRECTION_DENSE_DECIMAL_EVALUATION_20261028.json' if decimal else 'DIRECTION_DENSE_ONEPASS_EVALUATION_20261028.json')
        _require(recipe['dense_evaluation_protocol']['path']==str(source_repo/'docs'/evaluation_name), 'DENSE_EVALUATION_PATH')
        protocol=campaign.read_bound_json(Path(recipe['dense_evaluation_protocol']['path']),recipe['dense_evaluation_protocol']['sha256'])
        if m1_fit:
            m1_authority.require_protocol(protocol,recipe)
        elif family:
            _require_family_protocol(protocol,recipe)
        elif factor:
            m5_authority.require_factor_protocol(protocol,recipe)
        elif priority_fit:
            priority_authority.require_protocol(protocol,recipe)
        elif m5_fit:
            m5_authority.require_protocol(protocol,recipe)
        else:
            protocol_schema=('gx1_dense_decimal_direction_evaluation_protocol_v1' if decimal else 'gx1_dense_onepass_direction_evaluation_protocol_v1')
            _require(protocol.get('schema_version')==protocol_schema
                and protocol.get('status')=='PREDECLARED_BEFORE_DENSE_FIT_NO_EXECUTION_AUTHORITY'
                and protocol.get('fold')==fold and protocol.get('test_data_used') is False
                and protocol.get('native_launch_authorized') is False, 'DENSE_EVALUATION_PROTOCOL')
            if decimal:
                _require(protocol.get('corrected_target')==recipe['dense_target']
                    and protocol.get('reference_predictions')==DENSE_DECIMAL_REFERENCE_PREDICTIONS
                    and protocol.get('primary_horizon_minutes')==5, 'DECIMAL_EVALUATION_IDENTITY')
                campaign.require_binding(protocol['reference_predictions'],label='decimal reference predictions',verify_file=True)
                sources=protocol.get('comparison_sources')
                _require(isinstance(sources,Mapping) and set(sources)==set(DECIMAL_EVALUATION_SOURCES), 'DECIMAL_EVALUATION_SOURCE_SET')
                for path,item in sources.items():
                    checked=campaign.require_binding(item,label='decimal evaluation source',verify_file=True)
                    _require(checked['path']==str(source_repo/path), 'DECIMAL_EVALUATION_SOURCE_PATH:'+path)
        campaign.require_binding(recipe['dense_target'],label='dense target',verify_file=False)
        if m1_fit:
            m1_authority.require_qualification(recipe,owners,source_repo=source_repo,private_scope=M1_PRICE_PRIVATE_ADAPTER_SCOPE)
        elif family:
            if recovery:
                recovered = require_scoring_recovery_origin(recipe,source_repo=source_repo,source_commit=source_commit)
                original = recovered['original_recipe']
                private.require_family_qualification(original,original['source_owners'],source_repo=source_repo,
                    source_commit=recovered['identity']['source_head'])
            else:
                private.require_family_qualification(recipe,owners,source_repo=source_repo,source_commit=source_commit)
        elif factor:
            m5_authority.require_factor_qualification(recipe,owners,source_repo=source_repo,source_commit=source_commit)
        elif priority_fit:
            priority_authority.require_qualification(recipe,owners,source_repo=source_repo,source_commit=source_commit,private_scope=M1_PRICE_PRIVATE_ADAPTER_SCOPE)
        elif m5_fit:
            m5_authority.require_qualification(recipe,owners,source_repo=source_repo,private_scope=M1_PRICE_PRIVATE_ADAPTER_SCOPE)
        else:
            review=campaign.read_bound_json(Path(recipe['native_dense_qualification']['path']),recipe['native_dense_qualification']['sha256'])
            if decimal:
                _require(review.get('schema_version')=='gx1_decimal_dense_private_root_review_v1'
                    and review.get('status')=='PASS_CONSUMED_DECIMAL_PRIVATE_NATIVE_MECHANICS'
                    and review.get('initial_256_logits_parent_and_clock_byte_parity_independently_checked') is True
                    and review.get('corrected_all_60_train_priors_independently_checked') is True
                    and review.get('decimal_target')==recipe['dense_target']
                    and review.get('gc_projection_and_gate_nonzero_for_all_five_timeframes') is True,
                    'DECIMAL_PRIVATE_QUALIFICATION')
            else:
                _require(review.get('status')=='PASS_CONSUMED_PRIVATE_NATIVE_MECHANICS'
                    and review.get('initial_256_parity') is True and review.get('anchor_prefix_priors_equal') is True,
                    'DENSE_ORIGINAL_PRIVATE_QUALIFICATION')
            _require(review.get('sixty_horizons_have_eligible_labels_and_gradients') is True
                and review.get('chart_mtf_gradient_first_step_nonzero') is True
                and review.get('chart_local_gradient_first_step_zero_and_second_nonzero') is True
                and review.get('bit_exact_resume') is True and review.get('private_optimizer_updates')==4
                and review.get('qualification_weights_for_main_fit') is False and review.get('test_data_used') is False,
                'DENSE_PRIVATE_QUALIFICATION')
            plan_ref=campaign.require_binding(review['plan'],label='dense private plan',verify_file=True)
            private_plan=campaign.read_bound_json(Path(plan_ref['path']),plan_ref['sha256'])
            prior_ref=campaign.require_binding(private_plan['recipe'],label='dense private recipe',verify_file=True)
            prior=campaign.read_bound_json(Path(prior_ref['path']),prior_ref['sha256'])
            _require(prior.get('run_scope')==(private.DECIMAL_PRIVATE_SCOPE if decimal else private.DENSE_PRIVATE_SCOPE)
                and prior.get('input_manifest')==recipe['input_manifest']
                and prior.get('fold')==recipe['fold'] and prior.get('dense_target')==recipe['dense_target']
                and prior.get('dense_target_result')==recipe['dense_target_result']
                and prior.get('dense_source_transition_proof')==recipe['dense_source_transition_proof']
                and prior.get('qualification_weights_for_main_fit') is False
                and prior.get('numerical_recipe',{}).get('dense_minute_supervision') is True
                and {k:v for k,v in prior.get('numerical_recipe',{}).items() if k not in ('max_updates','contract_sha256')}
                    == {k:v for k,v in numerical.items() if k not in ('max_updates','contract_sha256')}, 'DENSE_PRIVATE_SOURCE_SCOPE')
            stable_sources=set(private.REQUIRED_SOURCE_PATHS)|set(private.DENSE_EXTRA_SOURCE_PATHS)
            if decimal:
                _require(prior.get('decimal_input_audit')==recipe['decimal_input_audit']
                    and prior.get('decimal_target_refs')==recipe['decimal_target_refs']
                    and prior.get('decimal_initial_predictions')==recipe['decimal_initial_predictions']
                    and private_plan.get('source_commit')==review.get('source_commit'), 'DECIMAL_PRIVATE_RECIPE_BINDING')
                result_ref=campaign.require_binding(review['result'],label='decimal private result',verify_file=True)
                receipt_ref=campaign.require_binding(review['receipt'],label='decimal private receipt',verify_file=True)
                campaign.require_binding(review['idle_restore'],label='decimal idle restoration',verify_file=True)
                qualified_result=campaign.read_bound_json(Path(result_ref['path']),result_ref['sha256'])
                qualified_receipt=campaign.read_bound_json(Path(receipt_ref['path']),receipt_ref['sha256'])
                _require(qualified_result.get('schema_version')=='gx1_native_decimal_dense_private_qualification_result_v1'
                    and qualified_result.get('decision')=='PASS_DECIMAL_DENSE_NATIVE_MECHANICS'
                    and qualified_result.get('recipe')==prior_ref
                    and qualified_result.get('decimal_target_refs')==recipe['decimal_target_refs']
                    and qualified_result.get('decimal_initial_prediction_byte_parity') is True
                    and qualified_result.get('decimal_prior_against_sign_audit') is True
                    and qualified_result.get('bit_exact_resume') is True and qualified_result.get('physical_optimizer_updates')==4
                    and qualified_result.get('source_unchanged') is True and qualified_result.get('test_data_used') is False
                    and qualified_result.get('private_weights_for_main_fit') is False
                    and qualified_receipt.get('outcome')=='COMPLETE' and qualified_receipt.get('guard_decision')=='PASS'
                    and qualified_receipt.get('plan_sha256')==private_plan.get('plan_sha256')
                    and qualified_receipt.get('test_data_used') is False, 'DECIMAL_PRIVATE_TERMINAL_EVIDENCE')
                stable_sources |= set(private.DECIMAL_EXTRA_SOURCE_PATHS)
                stable_sources -= set(DECIMAL_TRANSITION_FUNCTIONS)
                _require_decimal_source_equivalence(source_repo,review.get('source_commit'),prior.get('source_owners',{}),owners)
            else:
                stable_sources.remove('gx1/contracts/entry_native_direction_prefix_fit_campaign_v1.py')
            _require(all(owners.get(key)==prior.get('source_owners',{}).get(key) for key in stable_sources), 'DENSE_QUALIFIED_NUMERICAL_SOURCE')
    if chronological:
        origin = recipe['order_origin']
        _require(fold == private.BPS_FOLD and origin == ORDER_ORIGIN
            and recipe['gc_layout_metadata_parity'] == private.BPS_GC_PARITY, 'ORDER_ORIGIN')
        refs = {key:campaign.require_binding(ref, label='order '+key, verify_file=True) for key,ref in origin.items()}
        campaign.require_binding(recipe['gc_layout_metadata_parity'], label='order GC metadata proof', verify_file=True)
        old = campaign.read_bound_json(Path(refs['launch_recipe']['path']), refs['launch_recipe']['sha256'])
        _require(old.get('fold') == fold and old.get('train_passes') == 1
            and old.get('run_scope') == 'single_prefix_one_pass_direction_fit'
            and old.get('input_manifest') == recipe['input_manifest']
            and old.get('numerical_recipe') == numerical
            and numerical.get('signed_bps_supervision', False) is False
            and numerical.get('magnitude_weighted_direction', False) is False, 'ORDER_BASELINE_NUMERICS')
        initial = campaign.read_bound_json(Path(refs['initial_measurement']['path']), refs['initial_measurement']['sha256'])
        _require(initial.get('role') == 'INITIAL_TRAIN' and initial.get('fold') == fold
            and initial.get('predictions') == refs['initial_predictions'], 'ORDER_BASELINE_INITIAL')
        proof_ref = campaign.require_binding(recipe['native_order_qualification'], label='order native proof', verify_file=True)
        proof = campaign.read_bound_json(Path(proof_ref['path']), proof_ref['sha256'])
        transition = proof.get('historical_component_transition')
        _require(proof.get('schema_version') == 'gx1_direction_order_initial_resume_qualification_v1'
            and proof.get('status') == 'PASS_SIGN_ONLY_ORDER_INITIAL_RESUME_QUALIFIED'
            and proof.get('origin') == refs and proof.get('fold') == fold
            and proof.get('source_owners') == owners
            and proof.get('numerical_recipe_sha256') == numerical['contract_sha256']
            and proof.get('gc_layout_metadata_parity') == recipe['gc_layout_metadata_parity']
            and proof.get('initial_model_buffers_optimizer_rng_equal') is True
            and proof.get('initial_predictions_256_bit_equal') is True
            and proof.get('resume_bit_exact') is True
            and type(proof.get('private_optimizer_updates')) is int and proof['private_optimizer_updates'] == 4
            and proof.get('qualification_weights_for_main_fit') is False and proof.get('test_data_used') is False,
            'ORDER_NATIVE_QUALIFICATION')
        _require(isinstance(transition, Mapping) and set(transition) == {'original', 'current'}
            and transition['original'] == input_manifest['view_source_bindings']['direction_component_source']
            and transition['current'] == owners[ORDER_COMPONENT_SOURCE]
            and transition['original']['path'] == transition['current']['path'], 'ORDER_COMPONENT_TRANSITION')
    if legacy_continuation:
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


def guarded_prefix_power_limit(recipe):
    """Keep the user-selected 250 W limit for the matched factor routes."""
    _require(isinstance(recipe, Mapping), 'FACTOR_POWER_RECIPE')
    scope = recipe.get('run_scope')
    if scope == DYNAMIC_STATE_SCOPE:
        _require(recipe.get('full_feature_profile') is True
            and recipe.get('numerical_recipe', {}).get('dynamic_state_supervision') is True, 'DYNAMIC_STATE_POWER_SCOPE')
        return 250
    if scope in (M5_FACTOR_SIGN_SCOPE, M5_FACTOR_BPS_SCOPE, M5_FACTOR_CONTINUATION_SCOPE, FAMILY_SIGN_SCOPE):
        _require(recipe.get('candidate_mode') == 'price_only'
            and recipe.get('numerical_recipe', {}).get('dense_signed_bps_supervision', False) is (scope == M5_FACTOR_BPS_SCOPE), 'FACTOR_POWER_SCOPE')
        return 250
    return 300

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
    recovery = is_scoring_recovery(recipe)
    _require(result['requires_fresh_windows_boot'] is True or recipe['run_scope'] == FAMILY_SIGN_SCOPE, 'FAMILY_ONLY_BOOT_MODE')
    separator = result['launcher_argv'].index('--')
    _require(result['launcher_argv'][separator+1:] == [str(source_repo/'.venv/bin/python'), '-m', MODULE,
        '--window-policy', policy_ref['path'], '--window-policy-file-sha256', policy_ref['sha256'],
        '--progress-path', window['progress_path']], 'INVOCATION_ARGV')
    _require(result['kind'] == KIND and type(result['batch_size']) is int and result['batch_size'] == BATCH_SIZE
        and result['epoch_index'] is None and type(number) is int and 1 <= number <= MAX_WINDOWS
        and result['window_index'] == number-1 and result['window_count'] == (1 if recovery else MAX_WINDOWS)
        and (not recovery or number == 1)
        and result['optimizer_step_budget'] == (0 if recovery else recipe['numerical_recipe']['max_updates'])
        and type(result['optimizer_step_budget']) is int
        and result['expected_success_outcome'] == ('COMPLETE' if recovery else 'RESUMABLE_OR_COMPLETE')
        and result['maximum_wall_seconds'] == 13800 and window['invocation_number'] == number
        and result['checkpoint'] == {'pointer_path':window['campaign_cursor_path'],
            'before_mode':'GENESIS' if number == 1 else 'PREVIOUS_RECEIPT_AFTER',
            'predecessor_invocation_number':None if number == 1 else number-1, 'write_mode':'ATOMIC_UPDATE'}
        and result['progress_path'] == window['progress_path'] and window['recipe'] == recipe_ref
        and execution['prelaunch_manifest_sha256'] == recipe['recipe_sha256']
        and execution['train_session_manifest_sha256'] == window['policy_sha256'], 'INVOCATION_BINDING')
    result['direction_prefix_fit_recipe'] = recipe_ref
    result['direction_prefix_fit_window'] = window
    if recovery:
        result['direction_prefix_fit_scoring_recovery'] = True


def require_prefix_fit_plan(value, *, verify_files=True):
    fields = {'schema_version', 'decision', 'campaign_id', 'phase', 'created_utc', 'source_repo', 'source_commit',
        'runtime_root', 'gpu_uuid', 'prepared_windows_boot', 'recipe', 'invocations', 'signed_guard_sources',
        'controller_sources', 'policy', 'authority', 'test_data_used'}
    if isinstance(value, Mapping) and 'family_same_boot' in value:
        fields = fields | {'family_same_boot'}
    result = _identity(value, fields, campaign.PLAN_SCHEMA, 'plan_sha256')
    _require(result['phase'] == PHASE and result['decision'] == 'PASS_PREPARED'
        and type(result['campaign_id']) is str and bool(result['campaign_id']) and result['source_repo'] == str(CANONICAL_REPO)
        and type(result['source_commit']) is str and campaign._COMMIT.fullmatch(result['source_commit']) is not None
        and type(result['gpu_uuid']) is str and result['gpu_uuid'].startswith('GPU-') and result['test_data_used'] is False, 'PLAN_IDENTITY')
    campaign._utc(result['created_utc'], 'prefix fit creation')
    runtime = Path(_path(result['runtime_root'], 'runtime root'))
    result['prepared_windows_boot'] = campaign.require_boot_identity(result['prepared_windows_boot'])
    expected_policy = campaign.family_campaign_policy(result)
    if verify_files:
        policy_recipe = require_prefix_fit_launch_recipe(result['recipe'],
            source_repo=CANONICAL_REPO, source_commit=result['source_commit'])
        if policy_recipe['run_scope'] == DYNAMIC_STATE_SCOPE:
            _require('family_same_boot' not in result, 'STATE_FRESH_BOOT_REQUIRED')
            expected_policy['physical_power_limit_w'] = 250
    _require(campaign.canonical_sha256(result['policy']) == campaign.canonical_sha256(expected_policy)
        and campaign.canonical_sha256(result['authority']) == campaign.canonical_sha256(
            {'test':False, 'promotion':False, 'paper':False, 'live':False, 'cloud_spend':False}), 'PLAN_AUTHORITY')
    for key, names in (('signed_guard_sources', {'runner','guard','query','certificate'}),
            ('controller_sources', {'controller','observer','campaign_cli'})):
        _require(isinstance(result[key], Mapping) and set(result[key]) == names, 'PLAN_SOURCES:' + key)
        result[key] = {name:campaign.require_binding(ref, label=key, verify_file=verify_files) for name,ref in result[key].items()}
    result['recipe'] = campaign.require_binding(result['recipe'], label='prefix fit recipe', verify_file=verify_files)
    _require(isinstance(result['invocations'], list) and len(result['invocations']) in (1, MAX_WINDOWS), 'PLAN_WINDOW_COUNT')
    result['invocations'] = [campaign.require_binding(ref, label='prefix invocation', verify_file=verify_files) for ref in result['invocations']]
    _require(len({ref['path'] for ref in result['invocations']}) == len(result['invocations']), 'PLAN_DISTINCT_INVOCATIONS')
    result['selection_artifact_sha256'] = None
    if verify_files:
        recipe = require_prefix_fit_launch_recipe(result['recipe'], source_repo=CANONICAL_REPO, source_commit=result['source_commit'])
        _require(recipe['runtime_root'] == str(runtime), 'PLAN_RECIPE_RUNTIME')
        _require(len(result['invocations']) == (1 if is_scoring_recovery(recipe) else MAX_WINDOWS), 'PLAN_RECIPE_WINDOW_COUNT')
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
        campaign.require_family_plan_mode(result, recipe, checked)
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
    recovery = is_scoring_recovery(recipe)
    _require(repo == CANONICAL_REPO and branch == 'work/gx1-current' and not dirty
        and policy.get('schema_version') == 'gx1_next_native_run_policy_v1'
        and policy.get('canonical_source_repo') == str(repo) and policy.get('canonical_branch') == branch
        and policy.get('training_module') == MODULE and all(policy.get(key) is (not recovery) for key in
            ('training_enabled','native_training_enabled','full_epoch_training_allowed'))
        and policy.get('full_val_allowed') is False, 'READINESS_GLOBAL')
    _require(isinstance(gate, Mapping) and gate.get('recipe') == window['recipe']
        and gate.get('run_root') == recipe['runtime_root'] and type(gate.get('max_invocations')) is int and gate['max_invocations'] == (1 if recovery else MAX_WINDOWS)
        and type(gate.get('claimed_invocations')) is int and gate['claimed_invocations'] == 0
        and type(gate.get('completed_invocations')) is int and gate['completed_invocations'] == 0
        and gate.get('relaunch_allowed') is False, 'READINESS_ONCE')
    for key in ('execution_authorized','data_reads_authorized','genuine_constructor_authorized','genuine_forward_authorized',
            'genuine_backward_authorized','genuine_updates_authorized','genuine_predictor_fit_authorized',
            'training_enabled','native_training_enabled','full_epoch_training_allowed','native_launch_authorized'):
        expected = not recovery or key in ('execution_authorized','data_reads_authorized',
            'genuine_constructor_authorized','genuine_forward_authorized','native_launch_authorized')
        _require(gate.get(key) is expected, 'READINESS_GATE:' + key)
    if recovery:
        _require(gate.get('checkpoint_reads_authorized') is True
            and gate.get('zero_update_scoring_recovery_authorized') is True, 'READINESS_RECOVERY_GATE')
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
            'fold':recipe['fold'], 'max_windows':1 if recovery else MAX_WINDOWS, 'full_epoch_training_allowed':not recovery,
            'optimizer_step_ceiling':0 if recovery else recipe['numerical_recipe']['max_updates'], 'window_policy_file_sha256':window_file_sha256}}


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
        if 'family_same_boot' in plan:
            campaign.require_family_idle_admission(marker.get('idle_admission'), plan=plan,
                invocation=invocation, previous=previous[-1] if previous else None, current_boot=marker['boot'],
                at_utc=marker['started_utc'], expected_plan_binding=plan_ref)
            policy = campaign.read_bound_json(CANONICAL_REPO/'NEXT_RUN_POLICY.json', campaign.file_sha256(CANONICAL_REPO/'NEXT_RUN_POLICY.json'))
            _require(policy.get(launch['admission_stage'], {}).get('same_boot_after_idle_review_authorized') is True, 'FAMILY_BOOT_READINESS')
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
    _require(type(train_passes) is int and train_passes in (1, 2, 3), 'CURSOR_PASSES')
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
    _require(type(passes) is int and ((multipass and passes in (2, 3)) or (not multipass and passes == 1)), 'CURSOR_PASSES')
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
            bound = result['feature_view_identity']['prefix_normalization']
            if launch.get('run_scope') == DYNAMIC_STATE_SCOPE:
                manifest = require_dynamic_state_input_manifest(launch['input_manifest'],
                    fold=fold, source_owners=launch['source_owners'])
                ref = manifest['prefix_normalization']
                fitted = campaign.read_bound_json(Path(ref['path']), ref['sha256'])
                expected = {'binding':ref, 'contract_sha256':fitted['contract_sha256'],
                    'base_sha256':fitted['base_normalization']['contract_sha256'],
                    'gc_layout_sha256':fitted['gc_mtf_history']['layout']['contract_sha256'],
                    'gc_normalization_sha256':fitted['gc_mtf_history']['normalization']['contract_sha256'],
                    'qualification_sha256':fitted['prefix_qualification']['contract_sha256']}
                matches = bound == expected
            else:
                prefix = _input_manifest(launch)['prefix_normalization']
                matches = bound['binding'] == prefix['output'] and all(bound[target] == prefix[source] for target,source in (
                    ('contract_sha256','combined_contract_sha256'),('base_sha256','base_normalization_sha256'),
                    ('gc_layout_sha256','gc_layout_sha256'),('gc_normalization_sha256','gc_normalization_sha256')))
            _require(matches and result['feature_view_identity']['research_objective_sha256'] ==
                launch['numerical_recipe']['research_objective_sha256'], 'CURSOR_INPUT_VIEW_IDENTITY')
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
    _require(type(passes) is int and ((multipass and passes in (2, 3)) or (not multipass and passes == 1)), 'PROGRESS_PASSES')
    fold = _fold(result['fold']); n = fold['prefix_rows']; quarter = fold['oof_end_row']-n
    recovery = invocation.get('direction_prefix_fit_scoring_recovery') is True
    _require(invocation['optimizer_step_budget'] == (0 if recovery else passes*((n+15)//16)), 'PROGRESS_PASS_BUDGET')
    if recovery:
        _require(passes==1 and invocation['invocation_number']==invocation['window_count']==1
            and invocation['expected_success_outcome']=='COMPLETE' and result['outcome'] in ('COMPLETE','FAILED'),
            'RECOVERY_PROGRESS_SCOPE')
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
