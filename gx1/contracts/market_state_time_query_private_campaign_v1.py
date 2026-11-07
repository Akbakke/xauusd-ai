"""Exact metadata and receipt contract for a four-step time-query private probe.

This is a specialization of the existing guarded native research campaign. The
probe proves optimizer/resume/timing mechanics on fixed TRAIN rows; it cannot
select a predictor, score OOF, read TEST, or authorize Entry/Exit/trading.
Metadata validation remains stdlib-only. CUDA and model state live in the
separate runner/state owner behind the canonical campaign guard.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts.entry_model_native_train_recipe_v1 import native_direction_research_recipe

SCOPE = 'market_state_time_query_private_4step_v1'
STAGE = 'market_state_time_query_private_4step_20261107_002'
RECIPE_SCHEMA = 'gx1_market_state_time_query_private_launch_recipe_v1'
NUMERICAL_SCHEMA = 'gx1_market_state_time_query_private_numerical_v1'
CURSOR_SCHEMA = 'gx1_market_state_time_query_private_cursor_v1'
RESULT_SCHEMA = 'gx1_market_state_time_query_private_resume_result_v1'
STATE_SCHEMA = 'gx1_market_state_time_query_private_state_v1'
CANONICAL_REPO = Path('/home/andre2/src/GX1_CURRENT')
RUNTIME_ROOT = Path('/home/andre2/GX1_RUNS') / STAGE.upper() / 'RUNTIME'
FOLD_ID = 1
PARENT_ROWS = tuple(range(2, 34))
BATCH_SIZE = 16
PASS_INDEX = 0
MAX_PRIVATE_UPDATES = 4
MAX_MODEL_CONSTRUCTIONS = 3
MAX_FORWARDS = 6
MAX_BACKWARDS = 4
MAX_CHECKPOINT_BYTES = 1073741824
MAX_PUBLISHED_JSON_BYTES = 33554432
MODEL_CONTRACT_SHA256 = 'bf7acb658c37f7c461bc2617da794c51050f84f0576a7370ebabce15894dba75'
LOSS_CONTRACT_SHA256 = '8a0d11358cb32955a11d1376b785309322d081d2aee1c8759e7532acebee79d8'
ZERO_STEP_ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TIME_QUERY_ZERO_STEP_20261107_002')
ZERO_STEP_REFS = {
    'plan': {'path': str(CANONICAL_REPO / 'configs/research/MARKET_STATE_TIME_QUERY_ZERO_STEP_20261107_002.json'),
             'sha256': '41b94d01e1cae7633946c565ab8975ce9d150d4809506df5dc19f59a77e54764'},
    'claim': {'path': str(ZERO_STEP_ROOT / 'CLAIM.json'),
              'sha256': 'a7791f921cda20dd71b5d8c751a7d9126aee7bc001fab44879307a342395c0b9'},
    'result': {'path': str(ZERO_STEP_ROOT / 'RESULT.json'),
               'sha256': '7183c00a5dc97ce6199705f885c8626e3b190d70b4de5d16a8f2d597518ae906'},
    'terminal': {'path': str(ZERO_STEP_ROOT / 'TERMINAL.json'),
                 'sha256': 'abbe714eab252622bc1a12e95f3d00bdc7546e55045d37c1e56e403e0f8fd791'},
}
# Additional current source owners beyond the existing native research campaign.
# The original 13 are the physically qualified time-query input/model owners.
EXTRA_SOURCE_PATHS = (
    'gx1/audit/fit_market_state_mature_fold1_units_v1.py',
    'gx1/audit/qualify_market_state_mature_prefix_normalization_v1.py',
    'gx1/audit/verify_market_state_five_endpoint_zero_step_v1.py',
    'gx1/audit/verify_market_state_time_query_zero_step_v1.py',
    'gx1/contracts/entry_native_direction_prefix_units_v1.py',
    'gx1/contracts/market_state_five_endpoint_contract_v1.py',
    'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
    'gx1/contracts/market_state_time_query_batch_v1.py',
    'gx1/contracts/market_state_time_query_binding_v1.py',
    'gx1/contracts/market_state_time_query_loss_v1.py',
    'gx1/contracts/market_state_time_query_model_v1.py',
    'gx1/contracts/market_state_time_query_target_v1.py',
    'gx1/scripts/validate_entry_model_native_technical_checkpoint_v1.py',
    'gx1/contracts/market_state_time_query_private_campaign_v1.py',
    'gx1/contracts/market_state_time_query_private_state_v1.py',
    'gx1/scripts/run_market_state_time_query_private_window_v1.py',
)


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise campaign.RandomAccessCampaignError('TIME_QUERY_PRIVATE_' + reason)


def _identity(value: Any, fields: set[str], schema: str, seal: str) -> dict[str, Any]:
    _need(isinstance(value, Mapping) and set(value) == fields | {seal}, 'FIELDS:' + schema)
    result = dict(value)
    claimed = campaign._sha(result.pop(seal), seal)
    _need(result.get('schema_version') == schema
          and claimed == campaign.canonical_sha256(result), 'IDENTITY:' + schema)
    result[seal] = claimed
    return result


def _read(ref: Mapping[str, str], label: str) -> dict:
    checked = campaign.require_binding(ref, label=label, verify_file=True)
    return campaign.read_bound_json(Path(checked['path']), checked['sha256'])


def _bridge() -> tuple[dict, dict]:
    """Reuse the completed zero-step's exact 15-input lineage, without Torch."""
    plan = _read(ZERO_STEP_REFS['plan'], 'time-query zero-step plan')
    claim = _read(ZERO_STEP_REFS['claim'], 'time-query zero-step claim')
    result = _read(ZERO_STEP_REFS['result'], 'time-query zero-step result')
    terminal = _read(ZERO_STEP_REFS['terminal'], 'time-query zero-step terminal')
    _need(plan.get('stage') == 'market_state_time_query_zero_step_20261107_002'
          and claim.get('plan') == result.get('plan') == terminal.get('plan') == ZERO_STEP_REFS['plan']
          and result.get('claim') == terminal.get('claim') == ZERO_STEP_REFS['claim']
          and terminal.get('result') == ZERO_STEP_REFS['result']
          and result.get('status') == terminal.get('status') == 'COMPLETE'
          and terminal.get('exit_code') == 0 and terminal.get('no_relaunch_under_this_plan') is True
          and result.get('model_forwards') == result.get('backward_calls') == 2
          and result.get('optimizer_steps') == 0 and result.get('test_data_used') is False
          and terminal.get('test_data_used') is False
          and isinstance(plan.get('input_refs'), dict) and len(plan['input_refs']) == 15,
          'ZERO_STEP_CHAIN')
    folds = result.get('fold_results')
    _need(isinstance(folds, dict) and set(folds) == {'1', '2'}, 'ZERO_STEP_FOLDS')
    for fold_id, item in folds.items():
        _need(item.get('fold_id') == int(fold_id)
              and item.get('model_contract_sha256') == MODEL_CONTRACT_SHA256
              and item.get('loss_contract_sha256') == LOSS_CONTRACT_SHA256
              and item.get('initial_model_state_sha256') == item.get('final_model_state_sha256')
              and item.get('model_forwards') == item.get('backward_calls') == 1
              and item.get('optimizer_steps') == 0 and item.get('test_data_used') is False,
              'ZERO_STEP_FOLD:' + fold_id)
    _need(folds['1'].get('normalization') == plan['input_refs'].get('fold1_normalization')
          and folds['2'].get('normalization') == plan['input_refs'].get('fold2_normalization')
          and folds['1'].get('view_sha256') == plan['expected_view_sha256_by_fold']['1']
          and folds['2'].get('view_sha256') == plan['expected_view_sha256_by_fold']['2'],
          'ZERO_STEP_VIEWS')
    return plan, result


def bound_input_refs() -> dict[str, dict[str, str]]:
    """The new optimizer may read only previously qualified TRAIN artifacts."""
    plan, _ = _bridge()
    result = dict(plan['input_refs'])
    result.update({'zero_step_' + key: dict(ref) for key, ref in ZERO_STEP_REFS.items()})
    return result


def time_query_private_numerical_recipe() -> dict[str, Any]:
    """Keep the existing B16 AdamW/FP32 numeric values; replace only objective."""
    base = dict(native_direction_research_recipe(batch_size=BATCH_SIZE, max_updates=2))
    base.pop('contract_sha256')
    base['schema_version'] = NUMERICAL_SCHEMA
    base['research_objective_sha256'] = MODEL_CONTRACT_SHA256
    base['time_query_loss_contract_sha256'] = LOSS_CONTRACT_SHA256
    raw = json.dumps(base, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                     allow_nan=False).encode()
    return {**base, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def require_time_query_private_recipe(value: Any, *, source_repo: Path,
                                      source_commit: str,
                                      verify_files: bool = True) -> dict[str, Any]:
    _need(verify_files, 'PHYSICAL_METADATA_REQUIRED')
    fields = {'schema_version', 'source_repo', 'runtime_root', 'admission_stage',
              'run_scope', 'fold_id', 'parent_rows', 'pass_index',
              'numerical_recipe', 'input_manifest', 'input_refs',
              'expected_view_sha256', 'expected_model_contract_sha256',
              'expected_loss_contract_sha256', 'source_owners',
              'execution_binding_path', 'max_private_updates',
              'max_model_constructions', 'max_forwards', 'max_backwards',
              'max_checkpoint_bytes', 'max_published_json_bytes',
              'training_enabled', 'full_epoch_training_allowed',
              'test_data_used', 'predictive_value_claimed',
              'qualification_weights_for_main_fit'}
    recipe = _identity(value, fields, RECIPE_SCHEMA, 'recipe_sha256')
    _need(source_repo == CANONICAL_REPO and recipe['source_repo'] == str(source_repo)
          and isinstance(source_commit, str) and campaign._COMMIT.fullmatch(source_commit) is not None
          and recipe['runtime_root'] == str(RUNTIME_ROOT)
          and recipe['execution_binding_path'] == str(RUNTIME_ROOT / 'EXECUTION_BINDING.json')
          and recipe['admission_stage'] == STAGE and recipe['run_scope'] == SCOPE
          and type(recipe['fold_id']) is int and recipe['fold_id'] == FOLD_ID
          and recipe['parent_rows'] == list(PARENT_ROWS)
          and type(recipe['pass_index']) is int and recipe['pass_index'] == PASS_INDEX
          and recipe['numerical_recipe'] == time_query_private_numerical_recipe()
          and recipe['max_private_updates'] == MAX_PRIVATE_UPDATES
          and recipe['max_model_constructions'] == MAX_MODEL_CONSTRUCTIONS
          and recipe['max_forwards'] == MAX_FORWARDS
          and recipe['max_backwards'] == MAX_BACKWARDS
          and recipe['max_checkpoint_bytes'] == MAX_CHECKPOINT_BYTES
          and recipe['max_published_json_bytes'] == MAX_PUBLISHED_JSON_BYTES
          and recipe['training_enabled'] is True
          and recipe['full_epoch_training_allowed'] is False
          and recipe['test_data_used'] is False
          and recipe['predictive_value_claimed'] is False
          and recipe['qualification_weights_for_main_fit'] is False,
          'RECIPE_SCOPE')
    plan, result = _bridge()
    _need(recipe['input_refs'] == bound_input_refs()
          and recipe['input_manifest'] == plan['input_refs']['mature_plan']
          and recipe['expected_view_sha256'] == result['fold_results']['1']['view_sha256']
          and recipe['expected_model_contract_sha256'] == MODEL_CONTRACT_SHA256
          and recipe['expected_loss_contract_sha256'] == LOSS_CONTRACT_SHA256,
          'RECIPE_INPUTS')
    from gx1.contracts import entry_native_direction_research_campaign_v1 as outer
    required = set(outer.REQUIRED_SOURCE_PATHS) | set(EXTRA_SOURCE_PATHS)
    owners = recipe['source_owners']
    _need(isinstance(owners, Mapping) and set(owners) == required,
          'SOURCE_OWNER_SET')
    for name, item in owners.items():
        ref = campaign.require_binding(item, label='time-query current source',
                                       verify_file=True)
        _need(ref['path'] == str(source_repo / name), 'SOURCE_PATH:' + name)
    for name, item in recipe['input_refs'].items():
        campaign.require_binding(item, label='time-query TRAIN input ' + name,
                                 verify_file=True)
    return recipe


def require_time_query_private_gate(gate: Any, recipe: Mapping[str, Any]) -> None:
    _need(isinstance(gate, Mapping) and gate.get('decision') == 'ADMITTED'
          and gate.get('status') == 'ADMITTED'
          and gate.get('time_query_private_probe_authorized') is True
          and gate.get('fold_id') == FOLD_ID
          and gate.get('sample_parent_rows') == list(PARENT_ROWS)
          and gate.get('model_forwards_authorized') == MAX_FORWARDS
          and gate.get('backward_calls_authorized') == MAX_BACKWARDS
          and gate.get('optimizer_steps_authorized') == MAX_PRIVATE_UPDATES
          and gate.get('genuine_predictor_fit_authorized') is False
          and gate.get('test_data_used') is False
          and recipe.get('run_scope') == SCOPE
          and recipe.get('admission_stage') == STAGE,
          'POLICY_GATE')


def build_time_query_private_cursor(*, recipe_ref: Mapping[str, str],
                                    outcome: str, optimizer_updates: int,
                                    parent_cursor: int, parent_order_sha256: str,
                                    full_state: Mapping[str, str] | None,
                                    result: Mapping[str, str]) -> dict[str, Any]:
    value = {'schema_version': CURSOR_SCHEMA, 'recipe': dict(recipe_ref),
             'outcome': outcome, 'optimizer_updates': optimizer_updates,
             'parent_cursor': parent_cursor,
             'parent_order_sha256': parent_order_sha256,
             'full_state': dict(full_state) if full_state is not None else None,
             'result': dict(result), 'test_data_used': False}
    value['cursor_sha256'] = campaign.canonical_sha256(value)
    return require_time_query_private_cursor(value, expected_recipe=recipe_ref,
                                             verify_files=False, batch_size=BATCH_SIZE)


def require_time_query_private_cursor(value: Any, *, expected_recipe: Mapping[str, str],
                                      verify_files: bool = True,
                                      batch_size: int = BATCH_SIZE) -> dict[str, Any]:
    fields = {'schema_version', 'recipe', 'outcome', 'optimizer_updates',
              'parent_cursor', 'parent_order_sha256', 'full_state',
              'result', 'test_data_used'}
    result = _identity(value, fields, CURSOR_SCHEMA, 'cursor_sha256')
    _need(type(batch_size) is int and batch_size == BATCH_SIZE
          and result['recipe'] == expected_recipe
          and result['outcome'] in {'COMPLETE', 'FAILED'}
          and type(result['optimizer_updates']) is int
          and 0 <= result['optimizer_updates'] <= MAX_PRIVATE_UPDATES
          and type(result['parent_cursor']) is int
          and 0 <= result['parent_cursor'] <= len(PARENT_ROWS)
          and result['parent_cursor'] % BATCH_SIZE == 0
          and result['test_data_used'] is False,
          'CURSOR_ROLE')
    campaign._sha(result['parent_order_sha256'], 'time-query parent order')
    campaign.require_binding(result['result'], label='time-query result',
                             verify_file=verify_files)
    if result['full_state'] is not None:
        campaign.require_binding(result['full_state'], label='time-query full state',
                                 verify_file=verify_files)
    _need(result['outcome'] != 'COMPLETE' or
          (result['optimizer_updates'] == MAX_PRIVATE_UPDATES
           and result['parent_cursor'] == len(PARENT_ROWS)
           and result['full_state'] is not None), 'CURSOR_COMPLETE')
    if verify_files:
        report = campaign.read_bound_json(Path(result['result']['path']),
                                          result['result']['sha256'])
        _need(report.get('schema_version') == RESULT_SCHEMA
              and report.get('recipe') == expected_recipe
              and report.get('parent_order_sha256') == result['parent_order_sha256']
              and report.get('parent_cursor') == result['parent_cursor']
              and report.get('physical_optimizer_updates') == result['optimizer_updates']
              and report.get('final_state') == result['full_state']
              and report.get('test_data_used') is False
              and (result['outcome'] != 'COMPLETE' or
                   report.get('decision') == 'PASS_PRIVATE_MARKET_STATE_TIME_QUERY_RESUME'
                   and report.get('bit_exact_resume') is True),
              'CURSOR_RESULT')
    return result
