"""Pure metadata gate for one guarded native S/F AdamW resume qualification.

No Torch import, feature payload, training, TEST or trading authority here.
The existing research campaign owns the trainer launch and GPU safety guards.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts.entry_model_native_train_recipe_v1 import native_direction_research_recipe
from gx1.contracts import market_state_time_query_private_campaign_v1 as previous

SCOPE = 'market_state_native_F_private_4step_v1'
STAGE = 'market_state_native_F_private_4step_20261108_001'
RECIPE_SCHEMA = 'gx1_market_state_native_F_private_launch_recipe_v1'
NUMERICAL_SCHEMA = 'gx1_market_state_native_F_private_numerical_v1'
CURSOR_SCHEMA = 'gx1_market_state_native_F_private_cursor_v1'
RESULT_SCHEMA = 'gx1_market_state_native_F_private_result_v1'
CANONICAL_REPO = Path('/home/andre2/src/GX1_CURRENT')
ROOT = Path('/home/andre2/GX1_RUNS') / STAGE.upper()
RUNTIME_ROOT = ROOT / 'RUNTIME'
PARENT_ROWS = tuple(range(2, 34))
BATCH_SIZE = 16
PASS_INDEX = 0
MAX_PRIVATE_UPDATES = 4
MAX_MODEL_CONSTRUCTIONS = 3
MAX_FORWARDS = 6
MAX_BACKWARDS = 4
MAX_CHECKPOINT_BYTES = 1073741824
MAX_PUBLISHED_JSON_BYTES = 33554432
MODEL_CONTRACT_SHA256 = 'ab6b8163d99213abd348a91ad97de87ca0e62b8704b4df0cb2086e47614ba811'
LOSS_CONTRACT_SHA256 = '06c8bffc81cbfe2da114499d3428ee94682a38609de67788854fd6cd2a1606dd'
VIEW_SHA256 = '5ceac9557d37234090805b8b0edde0f1444ab4e6a140f7b956c6120811e592ac'
ZERO_STEP_ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_NATIVE_F_CPU_ZERO_STEP_20261108_002')
ZERO_STEP_REFS = {
    'plan': {'path': str(CANONICAL_REPO / 'configs/research/MARKET_STATE_NATIVE_F_CPU_ZERO_STEP_20261108_002.json'),
             'sha256': '8842fad76deb86fb3ce2de7e7c22eb9483d7a86ff5a8124e5219cc86dc738f6b'},
    'claim': {'path': str(ZERO_STEP_ROOT / 'CLAIM.json'),
              'sha256': '3272900652a1c710d214a7cf2225d4e26e39a21b2ab85227eb38610aaf587fc7'},
    'result': {'path': str(ZERO_STEP_ROOT / 'RESULT.json'),
               'sha256': '7302d760741ae0329bd936a3bc0461dfedc2ada3a337fb23700bb2be7207b84b'},
    'terminal': {'path': str(ZERO_STEP_ROOT / 'TERMINAL.json'),
                 'sha256': '514ca93aa8b9f21287751e5db58bd8a00d5c6c1ca2c017a00eceda9195171c49'},
}
EXTRA_SOURCE_PATHS = (
    'gx1/contracts/market_state_native_F_private_campaign_v1.py',
    'gx1/contracts/market_state_native_F_private_state_v1.py',
    'gx1/contracts/market_state_time_query_native_F_model_v1.py',
    'gx1/contracts/market_state_time_query_native_F_objective_v1.py',
    'gx1/contracts/market_state_time_query_side_normalized_loss_v1.py',
    'gx1/contracts/market_state_time_query_native_F_batch_v1.py',
    'gx1/contracts/market_state_future_displacement_candidate_v1.py',
    'gx1/scripts/run_market_state_native_F_private_window_v1.py',
)


def _need(ok, reason):
    if not ok:
        raise campaign.RandomAccessCampaignError('NATIVE_F_PRIVATE_' + reason)


def _read(ref: Mapping[str, str], label: str) -> dict:
    checked = campaign.require_binding(ref, label=label, verify_file=True)
    return campaign.read_bound_json(Path(checked['path']), checked['sha256'])


def _identity(value: Any, fields: set[str], schema: str, seal: str) -> dict:
    _need(isinstance(value, Mapping) and set(value) == fields | {seal}, 'FIELDS:' + schema)
    result = dict(value)
    claimed = campaign._sha(result.pop(seal), seal)
    _need(result.get('schema_version') == schema
          and claimed == campaign.canonical_sha256(result), 'IDENTITY:' + schema)
    result[seal] = claimed
    return result


def _bridge() -> tuple[dict, dict]:
    plan = _read(ZERO_STEP_REFS['plan'], 'native F zero-step plan')
    claim = _read(ZERO_STEP_REFS['claim'], 'native F zero-step claim')
    result = _read(ZERO_STEP_REFS['result'], 'native F zero-step result')
    terminal = _read(ZERO_STEP_REFS['terminal'], 'native F zero-step terminal')
    _need(plan.get('stage') == 'market_state_native_F_cpu_zero_step_20261108_002'
          and claim.get('plan') == result.get('plan') == terminal.get('plan') == ZERO_STEP_REFS['plan']
          and result.get('claim') == terminal.get('claim') == ZERO_STEP_REFS['claim']
          and terminal.get('result') == ZERO_STEP_REFS['result']
          and result.get('status') == terminal.get('status') == 'COMPLETE'
          and terminal.get('exit_code') == 0
          and result.get('model_forwards') == result.get('backward_calls') == 1
          and result.get('optimizer_steps') == 0
          and result.get('view_sha256') == VIEW_SHA256
          and result.get('model_contract_sha256') == MODEL_CONTRACT_SHA256
          and result.get('loss_contract_sha256') == LOSS_CONTRACT_SHA256
          and result.get('test_data_used') is False
          and terminal.get('test_data_used') is False
          and terminal.get('no_relaunch_under_this_plan') is True,
          'ZERO_STEP_CHAIN')
    return plan, result


def bound_input_refs() -> dict[str, dict[str, str]]:
    plan, _ = _bridge()
    v2_plan = _read(plan['inputs']['completed_v2_plan'], 'completed v2 input plan')
    _need(isinstance(v2_plan.get('input_refs'), dict)
          and len(v2_plan['input_refs']) == 15, 'V2_INPUTS')
    refs = {key: dict(ref) for key, ref in v2_plan['input_refs'].items()}
    refs.update({'zero_step_' + key: dict(ref) for key, ref in ZERO_STEP_REFS.items()})
    refs['F_result'] = dict(plan['inputs']['F_result'])
    refs['F_tape'] = dict(plan['inputs']['F_tape'])
    return refs


def numerical_recipe() -> dict:
    base = dict(native_direction_research_recipe(batch_size=BATCH_SIZE, max_updates=2))
    base.pop('contract_sha256')
    base['schema_version'] = NUMERICAL_SCHEMA
    base['research_objective_sha256'] = MODEL_CONTRACT_SHA256
    base['native_F_loss_contract_sha256'] = LOSS_CONTRACT_SHA256
    base['future_weight'] = 1.0
    raw = json.dumps(base, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**base, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def require_recipe(value: Any, *, source_repo: Path,
                   source_commit: str, verify_files: bool = True) -> dict:
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
          and type(recipe['fold_id']) is int and recipe['fold_id'] == 1
          and recipe['parent_rows'] == list(PARENT_ROWS)
          and type(recipe['pass_index']) is int and recipe['pass_index'] == PASS_INDEX
          and recipe['numerical_recipe'] == numerical_recipe()
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
          and recipe['input_manifest'] == _read(plan['inputs']['completed_v2_plan'],
              'v2 plan')['input_refs']['mature_plan']
          and recipe['expected_view_sha256'] == result['view_sha256'] == VIEW_SHA256
          and recipe['expected_model_contract_sha256'] == MODEL_CONTRACT_SHA256
          and recipe['expected_loss_contract_sha256'] == LOSS_CONTRACT_SHA256,
          'RECIPE_INPUTS')
    from gx1.contracts import entry_native_direction_research_campaign_v1 as outer
    required = (set(outer.REQUIRED_SOURCE_PATHS) |
                set(previous.EXTRA_SOURCE_PATHS) | set(EXTRA_SOURCE_PATHS))
    owners = recipe['source_owners']
    _need(isinstance(owners, Mapping) and set(owners) == required, 'SOURCE_OWNER_SET')
    for name, item in owners.items():
        ref = campaign.require_binding(item, label='native F source', verify_file=True)
        _need(ref['path'] == str(source_repo / name), 'SOURCE_PATH:' + name)
    for name, item in recipe['input_refs'].items():
        campaign.require_binding(item, label='native F TRAIN input ' + name,
                                 verify_file=True)
    return recipe


def require_gate(gate: Any, recipe: Mapping[str, Any]) -> None:
    _need(isinstance(gate, Mapping) and gate.get('decision') == 'ADMITTED'
          and gate.get('status') == 'ADMITTED'
          and gate.get('native_F_private_probe_authorized') is True
          and gate.get('fold_id') == 1
          and gate.get('sample_parent_rows') == list(PARENT_ROWS)
          and gate.get('model_forwards_authorized') == MAX_FORWARDS
          and gate.get('backward_calls_authorized') == MAX_BACKWARDS
          and gate.get('optimizer_steps_authorized') == MAX_PRIVATE_UPDATES
          and gate.get('genuine_predictor_fit_authorized') is False
          and gate.get('test_data_used') is False
          and recipe.get('run_scope') == SCOPE
          and recipe.get('admission_stage') == STAGE,
          'POLICY_GATE')


def build_cursor(*, recipe_ref: Mapping[str, str], outcome: str,
                 optimizer_updates: int, parent_cursor: int,
                 parent_order_sha256: str, full_state: Mapping[str, str] | None,
                 result: Mapping[str, str]) -> dict:
    value = {'schema_version': CURSOR_SCHEMA, 'recipe': dict(recipe_ref),
             'outcome': outcome, 'optimizer_updates': optimizer_updates,
             'parent_cursor': parent_cursor,
             'parent_order_sha256': parent_order_sha256,
             'full_state': dict(full_state) if full_state is not None else None,
             'result': dict(result), 'test_data_used': False}
    value['cursor_sha256'] = campaign.canonical_sha256(value)
    return require_cursor(value, expected_recipe=recipe_ref, verify_files=False,
                          batch_size=BATCH_SIZE)


def require_cursor(value: Any, *, expected_recipe: Mapping[str, str],
                   verify_files: bool = True, batch_size: int = BATCH_SIZE) -> dict:
    fields = {'schema_version', 'recipe', 'outcome', 'optimizer_updates',
              'parent_cursor', 'parent_order_sha256', 'full_state',
              'result', 'test_data_used'}
    result = _identity(value, fields, CURSOR_SCHEMA, 'cursor_sha256')
    _need(batch_size == BATCH_SIZE and result['recipe'] == expected_recipe
          and result['outcome'] in {'COMPLETE', 'FAILED'}
          and type(result['optimizer_updates']) is int
          and 0 <= result['optimizer_updates'] <= MAX_PRIVATE_UPDATES
          and type(result['parent_cursor']) is int
          and 0 <= result['parent_cursor'] <= len(PARENT_ROWS)
          and result['parent_cursor'] % BATCH_SIZE == 0
          and result['test_data_used'] is False, 'CURSOR_ROLE')
    campaign._sha(result['parent_order_sha256'], 'native F parent order')
    campaign.require_binding(result['result'], label='native F result',
                             verify_file=verify_files)
    if result['full_state'] is not None:
        campaign.require_binding(result['full_state'], label='native F full state',
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
                   report.get('decision') == 'PASS_NATIVE_F_PRIVATE_RESUME'
                   and report.get('bit_exact_resume') is True),
              'CURSOR_RESULT')
    return result
