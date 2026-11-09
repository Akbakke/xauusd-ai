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



TA_SCOPE = 'market_state_ta_sf_mechanical_v1'
TA_STAGE = 'market_state_ta_sf_mechanical_20261108_001'
TA_RECIPE_SCHEMA = 'gx1_market_state_ta_sf_mechanical_launch_recipe_v1'
TA_RESULT_SCHEMA = 'gx1_market_state_ta_sf_mechanical_result_v1'
TA_CURSOR_SCHEMA = 'gx1_market_state_ta_sf_mechanical_cursor_v1'
TA_ROOT = Path('/home/andre2/GX1_RUNS') / TA_STAGE.upper()
TA_RUNTIME_ROOT = TA_ROOT / 'RUNTIME'
TA_RECIPE_PATH = TA_ROOT / 'RECIPE.json'
TA_PARENT_SELECTION = 'first_two_batches_of_exact_mature_fold1_order'
TA_SELECTED_FIT_ROWS = 2 * BATCH_SIZE


def ta_input_refs() -> dict[str, dict[str, str]]:
    """Only immutable fold-1 ancestry plus this TA candidate's input proof."""
    from gx1.audit import verify_market_state_time_query_zero_step_v1 as old
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as full
    from gx1.contracts.market_state_future_displacement_candidate_v1 import REFS
    refs = {key: old._fixed_ref(key) for key in (
        'fold1_plan', 'fold1_claim', 'fold1_result', 'fold1_terminal',
        'fold1_qualification', 'fold1_normalization', 'mature_plan',
        'constructor', 'base_view', 'candidate_tape')}
    refs.update({'F_' + key: {'path': str(path), 'sha256': digest}
                 for key, (path, digest) in REFS.items()})
    refs.update({name: full._ta_ref(name) for name in (
        'ta_candidate_view', 'ta_input_audit', 'ta_endpoint_priors')})
    terminal = full.TA_INPUT_ROOT / 'TERMINAL.json'
    refs['ta_input_terminal'] = {'path': str(terminal), 'sha256': campaign.file_sha256(terminal)}
    return refs


def _require_ta_input_proof(refs):
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as full
    view = _read(refs['ta_candidate_view'], 'TA candidate view')
    audit = _read(refs['ta_input_audit'], 'TA physical input audit')
    terminal = _read(refs['ta_input_terminal'], 'TA physical input terminal')
    _need(terminal.get('status') == 'COMPLETE' and terminal.get('exit_code') == 0
          and terminal.get('result') == refs['ta_input_audit']
          and terminal.get('test_data_used') is False
          and terminal.get('no_relaunch_under_this_plan') is True, 'TA_INPUT_TERMINAL')
    evidence = audit.get('input_audit', {})
    _need(audit.get('status') == 'COMPLETE' and audit.get('test_data_used') is False
          and audit.get('candidate_view') == refs['ta_candidate_view']
          and view.get('full_ta_feature_profile') is True
          and evidence.get('candidate_view_sha256') == view.get('contract_sha256')
          and evidence.get('gather', {}).get('fit_rows') == full.FIT_ROWS
          and evidence.get('gather', {}).get('score_rows') == full.SCORE_ROWS
          and evidence.get('gather', {}).get('all_fields_all_window_positions_bit_identical') is True
          and evidence.get('normalization_fits') == 0
          and evidence.get('model_constructions') == evidence.get('model_forwards') == 1
          and evidence.get('backward_calls') == evidence.get('optimizer_steps') == 0,
          'TA_FRESH_INPUT_QUALIFICATION')
    return view, audit


def build_ta_sf_mechanical_recipe(*, source_commit: str) -> dict:
    """Build reviewable metadata after the new physical TA input proof exists."""
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as full
    refs = ta_input_refs()
    view, _ = _require_ta_input_proof(refs)
    numerical = full.ta_sf_numerical_recipe()
    owners = {path: {'path': str(CANONICAL_REPO / path),
                     'sha256': campaign.file_sha256(CANONICAL_REPO / path)}
              for path in sorted(full.required_source_paths(ta_sf=True))}
    value = {
        'schema_version': TA_RECIPE_SCHEMA, 'source_repo': str(CANONICAL_REPO),
        'runtime_root': str(TA_RUNTIME_ROOT), 'admission_stage': TA_STAGE,
        'run_scope': TA_SCOPE, 'fold_id': 1, 'parent_selection': TA_PARENT_SELECTION,
        'parent_order_rows': full.FIT_ROWS, 'selected_fit_rows': TA_SELECTED_FIT_ROWS,
        'pass_index': PASS_INDEX, 'numerical_recipe': numerical,
        'input_manifest': refs['mature_plan'], 'input_refs': refs,
        'expected_view_sha256': view['contract_sha256'],
        'expected_model_contract_sha256': numerical['research_objective_sha256'],
        'expected_loss_contract_sha256': numerical['native_F_loss_contract_sha256'],
        'source_owners': owners,
        'execution_binding_path': str(TA_RUNTIME_ROOT / 'EXECUTION_BINDING.json'),
        'max_private_updates': MAX_PRIVATE_UPDATES,
        'max_model_constructions': MAX_MODEL_CONSTRUCTIONS,
        'max_forwards': MAX_FORWARDS, 'max_backwards': MAX_BACKWARDS,
        'max_checkpoint_bytes': MAX_CHECKPOINT_BYTES,
        'max_published_json_bytes': MAX_PUBLISHED_JSON_BYTES,
        'training_enabled': True, 'full_epoch_training_allowed': False,
        'test_data_used': False, 'predictive_value_claimed': False,
        'qualification_weights_for_main_fit': False,
    }
    value['recipe_sha256'] = campaign.canonical_sha256(value)
    return require_ta_sf_mechanical_recipe(value, source_repo=CANONICAL_REPO,
                                           source_commit=source_commit)


def require_ta_sf_mechanical_recipe(value, *, source_repo, source_commit,
                                    verify_files=True):
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as full
    _need(verify_files, 'TA_PHYSICAL_METADATA_REQUIRED')
    fields = {'schema_version', 'source_repo', 'runtime_root', 'admission_stage',
        'run_scope', 'fold_id', 'parent_selection', 'parent_order_rows', 'selected_fit_rows',
        'pass_index', 'numerical_recipe', 'input_manifest', 'input_refs',
        'expected_view_sha256', 'expected_model_contract_sha256',
        'expected_loss_contract_sha256', 'source_owners', 'execution_binding_path',
        'max_private_updates', 'max_model_constructions', 'max_forwards', 'max_backwards',
        'max_checkpoint_bytes', 'max_published_json_bytes', 'training_enabled',
        'full_epoch_training_allowed', 'test_data_used', 'predictive_value_claimed',
        'qualification_weights_for_main_fit'}
    value = _identity(value, fields, TA_RECIPE_SCHEMA, 'recipe_sha256')
    numerical = full.ta_sf_numerical_recipe()
    refs = ta_input_refs()
    _need(source_repo == CANONICAL_REPO and value['source_repo'] == str(CANONICAL_REPO)
          and isinstance(source_commit, str) and campaign._COMMIT.fullmatch(source_commit) is not None
          and TA_ROOT == full.TA_MECHANICAL_ROOT
          and value['runtime_root'] == str(TA_RUNTIME_ROOT)
          and value['execution_binding_path'] == str(TA_RUNTIME_ROOT / 'EXECUTION_BINDING.json')
          and value['admission_stage'] == TA_STAGE and value['run_scope'] == TA_SCOPE
          and type(value['fold_id']) is int and value['fold_id'] == 1
          and value['parent_selection'] == TA_PARENT_SELECTION
          and value['parent_order_rows'] == full.FIT_ROWS
          and value['selected_fit_rows'] == TA_SELECTED_FIT_ROWS
          and type(value['pass_index']) is int and value['pass_index'] == PASS_INDEX
          and value['numerical_recipe'] == numerical and value['input_refs'] == refs
          and value['input_manifest'] == refs['mature_plan']
          and value['expected_model_contract_sha256'] == numerical['research_objective_sha256']
          and value['expected_loss_contract_sha256'] == numerical['native_F_loss_contract_sha256']
          and value['max_private_updates'] == MAX_PRIVATE_UPDATES
          and value['max_model_constructions'] == MAX_MODEL_CONSTRUCTIONS
          and value['max_forwards'] == MAX_FORWARDS and value['max_backwards'] == MAX_BACKWARDS
          and value['max_checkpoint_bytes'] == MAX_CHECKPOINT_BYTES
          and value['max_published_json_bytes'] == MAX_PUBLISHED_JSON_BYTES
          and value['training_enabled'] is True and value['full_epoch_training_allowed'] is False
          and value['test_data_used'] is False and value['predictive_value_claimed'] is False
          and value['qualification_weights_for_main_fit'] is False, 'TA_RECIPE_SCOPE')
    view, audit = _require_ta_input_proof(refs)
    _need(value['expected_view_sha256'] == view['contract_sha256'], 'TA_RECIPE_VIEW')
    owners = value['source_owners']
    _need(isinstance(owners, Mapping)
          and set(owners) == full.required_source_paths(ta_sf=True), 'TA_SOURCE_OWNER_SET')
    for name, item in owners.items():
        ref = campaign.require_binding(item, label='TA mechanical source', verify_file=True)
        _need(ref['path'] == str(source_repo / name), 'TA_SOURCE_PATH:' + name)
    for name, ref in refs.items():
        campaign.require_binding(ref, label='TA mechanical input ' + name, verify_file=True)
    audited = audit.get('source_owners')
    from gx1.audit.verify_market_state_time_query_zero_step_v1 import TA_SOURCE_PATHS
    _need(isinstance(audited, Mapping) and TA_SOURCE_PATHS <= set(audited)
          and TA_SOURCE_PATHS <= set(owners)
          and all(owners[path]['sha256'] == audited[path]
                  for path in set(owners) & set(audited)), 'TA_INPUT_SOURCE_DRIFT')
    return value


def require_ta_sf_mechanical_gate(gate, recipe):
    _need(isinstance(gate, Mapping) and gate.get('decision') == gate.get('status') == 'ADMITTED'
          and gate.get('ta_sf_mechanical_authorized') is True and gate.get('fold_id') == 1
          and gate.get('parent_selection') == TA_PARENT_SELECTION
          and gate.get('selected_fit_rows') == TA_SELECTED_FIT_ROWS
          and gate.get('model_forwards_authorized') == MAX_FORWARDS
          and gate.get('backward_calls_authorized') == MAX_BACKWARDS
          and gate.get('optimizer_steps_authorized') == MAX_PRIVATE_UPDATES
          and gate.get('genuine_predictor_fit_authorized') is False
          and gate.get('test_data_used') is False
          and recipe.get('run_scope') == TA_SCOPE
          and recipe.get('admission_stage') == TA_STAGE, 'TA_POLICY_GATE')


# New one-use TA/M5 qualification; the completed TA/SF campaign stays immutable.
TA_M5_STAGE = 'market_state_ta_dedicated_m5_mechanical_20261109_002'
TA_M5_SCOPE = 'market_state_ta_dedicated_m5_mechanical_v1'
TA_M5_RECIPE_SCHEMA = 'gx1_market_state_ta_m5_mechanical_launch_recipe_v1'
TA_M5_RESULT_SCHEMA = 'gx1_market_state_ta_m5_mechanical_result_v1'
TA_M5_CURSOR_SCHEMA = 'gx1_market_state_ta_m5_mechanical_cursor_v1'
TA_M5_ROOT = Path('/home/andre2/GX1_RUNS') / TA_M5_STAGE.upper()
TA_M5_RUNTIME_ROOT = TA_M5_ROOT / 'RUNTIME'
TA_M5_RECIPE_PATH = TA_M5_ROOT / 'RECIPE.json'
TA_M5_MAX_MODEL_CONSTRUCTIONS = 4
TA_M5_MAX_OPTIMIZER_CONSTRUCTIONS = 4
TA_M5_PARITY_ATOL = 2e-6
TA_M5_PARITY_RTOL = 1e-6
TA_M5_REVIEWED_COMMIT = '95bb3df1b4592286aade63cf56fa7528cb9582ee'
TA_M5_ORIGINAL_COMMIT = '78cfce7fea40e41f9d3373ee025c8d535bdc4876'
TA_M5_ORIGINAL_MODEL_SHA256 = '7c86ae356af80db0e67141311f1b51c1551e3927ec22a2673496e315d6d13f03'
TA_M5_ORIGINAL_CURSOR = 29574
TA_M5_ORIGINAL_UPDATES = 1849
TA_M5_ORIGINAL_SKIPPED = 0
_TA_M5_ORIGINAL_ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_NATIVE_FOLD1_RECOVERY_20261108_001')
_TA_M5_REVIEW_ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_DEDICATED_M5_SOURCE_20261109_001')
TA_M5_FIXED_REFS = {
    'source_review': {'path': str(_TA_M5_REVIEW_ROOT / 'RESULT.json'),
        'sha256': '78ede8c1112ccc63e185749a3b28ef8ed6aaaba634163b653fbed1243a2ee295'},
    'original_recipe': {'path': str(_TA_M5_ORIGINAL_ROOT / 'RECIPE.json'),
        'sha256': '7e572eca2c2f70d36eae90d4cc8b339df0f921436f64b5ecc7b2c16a382e4299'},
    'original_result': {'path': str(_TA_M5_ORIGINAL_ROOT / 'RUNTIME/prefix-fit/window-0001/RESULT.json'),
        'sha256': '503bf914961e316dde327c37f6df9455781fa6f26e8fd623f02e9db893d9a7b6'},
    'original_checkpoint': {'path': str(_TA_M5_ORIGINAL_ROOT / 'RUNTIME/prefix-fit/states/FINAL_FOLD1_FIT_STATE.pt'),
        'sha256': 'f5ea915eb6803af22906c39580f9b7ceb10c94cd663659130cdbe6570b7a1f60'},
    'original_final_measurement': {'path': str(_TA_M5_ORIGINAL_ROOT / 'RUNTIME/prefix-fit/window-0001/FINAL_TRAIN.json'),
        'sha256': 'a608b856ede83c348642dc6875686a25e237a42d56bad0cdcd44f85d1afcce74'},
}
TA_M5_REVIEWED_SOURCE_PATHS = frozenset({
    'gx1/audit/verify_market_state_time_query_zero_step_v1.py',
    'gx1/contracts/entry_native_direction_feature_view_v1.py',
    'gx1/contracts/market_state_time_query_fold1_fit_state_v1.py',
    'gx1/contracts/market_state_time_query_fold1_fit_v1.py',
    'gx1/contracts/market_state_time_query_model_v2.py',
    'gx1/contracts/market_state_time_query_native_F_batch_v1.py',
    'gx1/contracts/market_state_time_query_native_F_model_v1.py',
    'gx1/contracts/market_state_time_query_native_F_objective_v1.py',
})
TA_M5_ORCHESTRATION_SOURCE_PATHS = frozenset({
    'gx1/contracts/market_state_native_F_private_campaign_v1.py',
    'gx1/scripts/run_market_state_native_F_private_window_v1.py',
    'gx1/contracts/entry_native_direction_research_campaign_v1.py',
    'gx1/scripts/run_entry_native_direction_research_window_v1.py',
    'gx1/scripts/materialize_market_state_time_query_fold1_fit_campaign_v1.py',
})


def ta_m5_input_refs() -> dict[str, dict[str, str]]:
    refs = ta_input_refs()
    original = _read(TA_M5_FIXED_REFS['original_recipe'], 'TA M5 original recipe')
    review = _read(TA_M5_FIXED_REFS['source_review'], 'TA M5 source review')
    refs.update({name: dict(ref) for name, ref in TA_M5_FIXED_REFS.items()})
    refs['original_candidate_view'] = dict(original['input_refs']['ta_candidate_view'])
    refs['original_endpoint_priors'] = dict(original['input_refs']['ta_endpoint_priors'])
    refs['ta_candidate_view'] = dict(review['candidate_view'])
    refs['ta_endpoint_priors'] = dict(review['rebound_priors'])
    refs['source_metadata_check'] = dict(review['real_metadata_check'])
    return refs


def _require_ta_m5_input_proof(refs):
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as full
    from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
    from gx1.contracts.entry_native_direction_feature_view_v1 import _sha as feature_view_sha
    from gx1.contracts.market_state_time_query_model_v2 import factorized_time_query_model_contract
    old_refs = dict(refs, ta_candidate_view=refs['original_candidate_view'],
                    ta_endpoint_priors=refs['original_endpoint_priors'])
    old_view, audit = _require_ta_input_proof(old_refs)
    view = _read(refs['ta_candidate_view'], 'TA M5 candidate view')
    review = _read(refs['source_review'], 'TA M5 reviewed source bridge')
    metadata = _read(refs['source_metadata_check'], 'TA M5 real metadata check')
    original = _read(refs['original_recipe'], 'TA M5 original launch recipe')
    model_path = 'gx1/contracts/market_state_time_query_model_v2.py'
    model_key = 'time_query_v2_model_source'
    old_sources, sources = old_view.get('source_bindings'), view.get('source_bindings')
    reviewed_sha = review.get('source_owners', {}).get(model_path)
    _need(isinstance(old_sources, Mapping) and isinstance(sources, Mapping)
          and set(old_sources) == set(sources)
          and old_sources.get(model_key) == original.get('source_owners', {}).get(model_path)
          and sources.get(model_key) == {'path': str(CANONICAL_REPO / model_path),
                                        'sha256': reviewed_sha}
          and reviewed_sha == _ta_m5_git_source_sha(TA_M5_REVIEWED_COMMIT, model_path)
          and {k: v for k, v in sources.items() if k != model_key}
              == {k: v for k, v in old_sources.items() if k != model_key},
          'TA_M5_INPUT_SOURCE_BRIDGE')
    changed = {'contract_sha256', 'dedicated_m5_side', 'research_objective'}
    _need(review.get('status') == 'IMPLEMENTED_FOCUSED_CHECKS_PASS_NATIVE_QUALIFICATION_PENDING'
          and review.get('candidate_view') == refs['ta_candidate_view']
          and review.get('rebound_priors') == refs['ta_endpoint_priors']
          and review.get('real_metadata_check') == refs['source_metadata_check']
          and review.get('native_model_forwards') == review.get('native_optimizer_updates') == 0
          and review.get('test_data_used') is False
          and view.get('dedicated_m5_side') is True
          and not old_view.get('dedicated_m5_side', False)
          and view.get('research_objective') == factorized_time_query_model_contract(
              ta_only=True, dedicated_m5_side=True)
          and view.get('contract_sha256') == feature_view_sha({
              k: v for k, v in view.items() if k != 'contract_sha256'})
          and {k: v for k, v in view.items() if k not in changed | {'source_bindings'}}
              == {k: v for k, v in old_view.items() if k not in changed | {'source_bindings'}},
          'TA_M5_EXACT_INPUT_IDENTITY')
    _need(metadata.get('status') == 'PASS_REAL_METADATA_BINDING_NO_MODEL_EXECUTION'
          and set(metadata.get('changed_current_view_fields', [])) == changed
          and metadata.get('view_sha256') == view['contract_sha256']
          and metadata.get('legacy_default_numerical_recipe_unchanged') is True
          and metadata.get('original_prior_counts_and_logits_unchanged') is True
          and metadata.get('model_constructions') == metadata.get('native_forwards')
              == metadata.get('optimizer_updates') == 0
          and metadata.get('test_data_used') is False, 'TA_M5_METADATA_REVIEW')
    old_prior = _read(refs['original_endpoint_priors'], 'TA M5 original priors')
    prior = _read(refs['ta_endpoint_priors'], 'TA M5 rebound priors')
    objective.require_endpoint_priors(old_prior, ta_only=True)
    objective.require_endpoint_priors(prior, ta_only=True, dedicated_m5_side=True)
    excluded = {'contract_sha256', 'loss_contract_sha256'}
    _need({k: v for k, v in prior.items() if k not in excluded}
          == {k: v for k, v in old_prior.items() if k not in excluded},
          'TA_M5_PRIOR_VALUES_CHANGED')
    result = _read(refs['original_result'], 'TA M5 completed original fit')
    measured = _read(refs['original_final_measurement'], 'TA M5 original FINAL measurement')
    _need(original.get('recipe_sha256') == campaign.canonical_sha256({
              k: v for k, v in original.items() if k != 'recipe_sha256'})
          and original.get('run_scope') == full.TA_RECOVERY_SCOPE
          and original.get('numerical_recipe') == full.ta_sf_numerical_recipe()
          and original.get('expected_view_sha256') == old_view['contract_sha256']
          and original.get('input_refs', {}).get('ta_candidate_view') == refs['original_candidate_view']
          and original.get('input_refs', {}).get('ta_endpoint_priors') == refs['original_endpoint_priors']
          and result.get('recipe') == refs['original_recipe']
          and result.get('full_state') == refs['original_checkpoint']
          and result.get('final_train_measurement') == refs['original_final_measurement']
          and result.get('source_commit') == TA_M5_ORIGINAL_COMMIT
          and result.get('source_unchanged') is True
          and result.get('outcome') == result.get('phase_state') == 'COMPLETE'
          and result.get('parent_cursor') == TA_M5_ORIGINAL_CURSOR
          and result.get('optimizer_updates') == TA_M5_ORIGINAL_UPDATES
          and result.get('skipped_batches') == TA_M5_ORIGINAL_SKIPPED
          and result.get('test_data_used') is False and result.get('error') is None
          and result.get('feature_view_identity', {}).get('contract_sha256') == old_view['contract_sha256']
          and measured.get('evaluated_checkpoint') == refs['original_checkpoint']
          and measured.get('evaluated_model_state_sha256') == TA_M5_ORIGINAL_MODEL_SHA256
          and measured.get('optimizer_updates') == TA_M5_ORIGINAL_UPDATES
          and measured.get('test_data_used') is False, 'TA_M5_ORIGINAL_FINAL_BINDING')
    campaign._sha(result.get('parent_order_sha256'), 'TA M5 original parent order')
    origin = {name: dict(refs[name]) for name in (
        'original_recipe', 'original_result', 'original_candidate_view', 'original_checkpoint')}
    origin.update(source_commit=TA_M5_ORIGINAL_COMMIT,
        model_state_sha256=TA_M5_ORIGINAL_MODEL_SHA256,
        parent_order_sha256=result['parent_order_sha256'],
        cursor=TA_M5_ORIGINAL_CURSOR, optimizer_updates=TA_M5_ORIGINAL_UPDATES,
        skipped_batches=TA_M5_ORIGINAL_SKIPPED,
        optimizer_migration='fresh_empty_AdamW_cursor0')
    return view, review, original, origin


def _ta_m5_git_source_sha(commit, name):
    import subprocess
    return hashlib.sha256(subprocess.check_output(
        ['git', 'show', commit + ':' + name], cwd=CANONICAL_REPO)).hexdigest()


def _require_ta_m5_source_bridge(owners, review, original, *, source_commit):
    import subprocess
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as full
    paths = full.required_source_paths(ta_sf=True)
    _need(isinstance(owners, Mapping) and set(owners) == paths
          and TA_M5_REVIEWED_SOURCE_PATHS | TA_M5_ORCHESTRATION_SOURCE_PATHS <= paths
          and isinstance(original.get('source_owners'), Mapping)
          and set(original['source_owners']) == paths, 'TA_M5_SOURCE_OWNER_SET')
    reviewed = review.get('source_owners', {})
    _need({name for name in reviewed if name.startswith('gx1/')} == TA_M5_REVIEWED_SOURCE_PATHS,
          'TA_M5_REVIEWED_OWNER_SET')
    _need(subprocess.run(['git', 'merge-base', '--is-ancestor', TA_M5_REVIEWED_COMMIT,
                          source_commit], cwd=CANONICAL_REPO, capture_output=True).returncode == 0,
          'TA_M5_REVIEWED_COMMIT_ANCESTRY')
    for name, ref in owners.items():
        checked = campaign.require_binding(ref, label='TA M5 current source', verify_file=True)
        _need(checked['path'] == str(CANONICAL_REPO / name), 'TA_M5_SOURCE_PATH:' + name)
        if name in TA_M5_ORCHESTRATION_SOURCE_PATHS:
            continue  # These five current bytes are explicitly recipe/plan-bound.
        expected = (original['source_owners'][name]['sha256'] if name.startswith('.venv/')
                    else _ta_m5_git_source_sha(TA_M5_REVIEWED_COMMIT, name))
        if name in TA_M5_REVIEWED_SOURCE_PATHS:
            _need(reviewed[name] == expected, 'TA_M5_REVIEWED_SOURCE_SHA:' + name)
        _need(checked['sha256'] == expected, 'TA_M5_UNREVIEWED_SOURCE_DRIFT:' + name)


def build_ta_m5_mechanical_recipe(*, source_commit: str) -> dict:
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as full
    refs = ta_m5_input_refs()
    view, _, _, origin = _require_ta_m5_input_proof(refs)
    numerical = full.ta_sf_numerical_recipe(dedicated_m5_side=True)
    value = {
        'schema_version': TA_M5_RECIPE_SCHEMA, 'source_repo': str(CANONICAL_REPO),
        'runtime_root': str(TA_M5_RUNTIME_ROOT), 'admission_stage': TA_M5_STAGE,
        'run_scope': TA_M5_SCOPE, 'fold_id': 1, 'parent_selection': TA_PARENT_SELECTION,
        'parent_order_rows': full.FIT_ROWS, 'selected_fit_rows': TA_SELECTED_FIT_ROWS,
        'pass_index': PASS_INDEX, 'numerical_recipe': numerical,
        'input_manifest': refs['mature_plan'], 'input_refs': refs,
        'expected_view_sha256': view['contract_sha256'],
        'expected_model_contract_sha256': numerical['research_objective_sha256'],
        'expected_loss_contract_sha256': numerical['native_F_loss_contract_sha256'],
        'source_owners': {name: {'path': str(CANONICAL_REPO / name),
            'sha256': campaign.file_sha256(CANONICAL_REPO / name)}
            for name in sorted(full.required_source_paths(ta_sf=True))},
        'reviewed_source_commit': TA_M5_REVIEWED_COMMIT, 'warmstart_origin': origin,
        'execution_binding_path': str(TA_M5_RUNTIME_ROOT / 'EXECUTION_BINDING.json'),
        'max_private_updates': MAX_PRIVATE_UPDATES,
        'max_model_constructions': TA_M5_MAX_MODEL_CONSTRUCTIONS,
        'max_optimizer_constructions': TA_M5_MAX_OPTIMIZER_CONSTRUCTIONS,
        'max_forwards': MAX_FORWARDS, 'max_backwards': MAX_BACKWARDS,
        'm5_side_parity_atol': TA_M5_PARITY_ATOL, 'm5_side_parity_rtol': TA_M5_PARITY_RTOL,
        'max_checkpoint_bytes': MAX_CHECKPOINT_BYTES,
        'max_published_json_bytes': MAX_PUBLISHED_JSON_BYTES,
        'training_enabled': True, 'full_epoch_training_allowed': False,
        'test_data_used': False, 'predictive_value_claimed': False,
        'qualification_weights_for_main_fit': False,
    }
    value['recipe_sha256'] = campaign.canonical_sha256(value)
    return require_ta_m5_mechanical_recipe(value, source_repo=CANONICAL_REPO,
                                         source_commit=source_commit)


def require_ta_m5_mechanical_recipe(value, *, source_repo, source_commit, verify_files=True):
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as full
    _need(verify_files, 'TA_M5_PHYSICAL_METADATA_REQUIRED')
    fields = {'schema_version', 'source_repo', 'runtime_root', 'admission_stage',
        'run_scope', 'fold_id', 'parent_selection', 'parent_order_rows', 'selected_fit_rows',
        'pass_index', 'numerical_recipe', 'input_manifest', 'input_refs',
        'expected_view_sha256', 'expected_model_contract_sha256', 'expected_loss_contract_sha256',
        'source_owners', 'reviewed_source_commit', 'warmstart_origin', 'execution_binding_path',
        'max_private_updates', 'max_model_constructions', 'max_optimizer_constructions',
        'max_forwards', 'max_backwards', 'm5_side_parity_atol', 'm5_side_parity_rtol',
        'max_checkpoint_bytes', 'max_published_json_bytes', 'training_enabled',
        'full_epoch_training_allowed', 'test_data_used', 'predictive_value_claimed',
        'qualification_weights_for_main_fit'}
    value = _identity(value, fields, TA_M5_RECIPE_SCHEMA, 'recipe_sha256')
    numerical = full.ta_sf_numerical_recipe(dedicated_m5_side=True)
    refs = ta_m5_input_refs()
    expected = {
        'source_repo': str(CANONICAL_REPO), 'runtime_root': str(TA_M5_RUNTIME_ROOT),
        'admission_stage': TA_M5_STAGE, 'run_scope': TA_M5_SCOPE, 'fold_id': 1,
        'parent_selection': TA_PARENT_SELECTION, 'parent_order_rows': full.FIT_ROWS,
        'selected_fit_rows': TA_SELECTED_FIT_ROWS, 'pass_index': PASS_INDEX,
        'numerical_recipe': numerical, 'input_refs': refs, 'input_manifest': refs['mature_plan'],
        'expected_model_contract_sha256': numerical['research_objective_sha256'],
        'expected_loss_contract_sha256': numerical['native_F_loss_contract_sha256'],
        'reviewed_source_commit': TA_M5_REVIEWED_COMMIT,
        'execution_binding_path': str(TA_M5_RUNTIME_ROOT / 'EXECUTION_BINDING.json'),
        'max_private_updates': MAX_PRIVATE_UPDATES,
        'max_model_constructions': TA_M5_MAX_MODEL_CONSTRUCTIONS,
        'max_optimizer_constructions': TA_M5_MAX_OPTIMIZER_CONSTRUCTIONS,
        'max_forwards': MAX_FORWARDS, 'max_backwards': MAX_BACKWARDS,
        'm5_side_parity_atol': TA_M5_PARITY_ATOL, 'm5_side_parity_rtol': TA_M5_PARITY_RTOL,
        'max_checkpoint_bytes': MAX_CHECKPOINT_BYTES,
        'max_published_json_bytes': MAX_PUBLISHED_JSON_BYTES,
        'training_enabled': True, 'full_epoch_training_allowed': False,
        'test_data_used': False, 'predictive_value_claimed': False,
        'qualification_weights_for_main_fit': False,
    }
    _need(source_repo == CANONICAL_REPO and isinstance(source_commit, str)
          and campaign._COMMIT.fullmatch(source_commit) is not None
          and all(value[key] == item and type(value[key]) is type(item)
                  for key, item in expected.items()), 'TA_M5_RECIPE_SCOPE')
    view, review, original, origin = _require_ta_m5_input_proof(refs)
    _need(value['expected_view_sha256'] == view['contract_sha256']
          and value['warmstart_origin'] == origin, 'TA_M5_RECIPE_WARMSTART')
    _require_ta_m5_source_bridge(value['source_owners'], review, original,
                               source_commit=source_commit)
    for name, ref in refs.items():
        campaign.require_binding(ref, label='TA M5 input ' + name, verify_file=True)
    return value


def require_ta_m5_mechanical_gate(gate, recipe):
    _need(isinstance(gate, Mapping) and gate.get('decision') == gate.get('status') == 'ADMITTED'
          and gate.get('ta_m5_mechanical_authorized') is True
          and gate.get('fold_id') == 1 and gate.get('parent_selection') == TA_PARENT_SELECTION
          and gate.get('selected_fit_rows') == TA_SELECTED_FIT_ROWS
          and gate.get('model_constructions_authorized') == TA_M5_MAX_MODEL_CONSTRUCTIONS
          and gate.get('optimizer_constructions_authorized') == TA_M5_MAX_OPTIMIZER_CONSTRUCTIONS
          and gate.get('model_forwards_authorized') == MAX_FORWARDS
          and gate.get('backward_calls_authorized') == MAX_BACKWARDS
          and gate.get('optimizer_steps_authorized') == MAX_PRIVATE_UPDATES
          and gate.get('genuine_predictor_fit_authorized') is False
          and gate.get('test_data_used') is False
          and recipe.get('run_scope') == TA_M5_SCOPE
          and recipe.get('admission_stage') == TA_M5_STAGE, 'TA_M5_POLICY_GATE')


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
                 result: Mapping[str, str], ta_sf: bool = False,
                 dedicated_m5_side: bool = False,
                 m5_equal_observation: bool = False) -> dict:
    _need(type(dedicated_m5_side) is bool and (not dedicated_m5_side or ta_sf is True), 'TA_M5_CURSOR_MODE')
    _need(type(m5_equal_observation) is bool and (not m5_equal_observation or (
        ta_sf is True and dedicated_m5_side is True)), 'TA_M5_EQUAL_LOSS_CURSOR_MODE')
    schema = (TA_M5_EQUAL_LOSS_CURSOR_SCHEMA if m5_equal_observation
              else TA_M5_CURSOR_SCHEMA if dedicated_m5_side else TA_CURSOR_SCHEMA if ta_sf else CURSOR_SCHEMA)
    value = {'schema_version': schema, 'recipe': dict(recipe_ref),
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
    equal_loss = isinstance(value, Mapping) and value.get('schema_version') == TA_M5_EQUAL_LOSS_CURSOR_SCHEMA
    if equal_loss or expected_recipe.get('path') == str(TA_M5_EQUAL_LOSS_RECIPE_PATH):
        _need(equal_loss and expected_recipe.get('path') == str(TA_M5_EQUAL_LOSS_RECIPE_PATH),
              'TA_M5_EQUAL_LOSS_CURSOR_RECIPE_PATH')
    dedicated = isinstance(value, Mapping) and value.get('schema_version') in (TA_M5_CURSOR_SCHEMA, TA_M5_EQUAL_LOSS_CURSOR_SCHEMA)
    ta_sf = isinstance(value, Mapping) and value.get('schema_version') in (TA_CURSOR_SCHEMA, TA_M5_CURSOR_SCHEMA, TA_M5_EQUAL_LOSS_CURSOR_SCHEMA)
    schema = (TA_M5_EQUAL_LOSS_CURSOR_SCHEMA if equal_loss else TA_M5_CURSOR_SCHEMA
              if dedicated else TA_CURSOR_SCHEMA if ta_sf else CURSOR_SCHEMA)
    result = _identity(value, fields, schema, 'cursor_sha256')
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
        _need(report.get('schema_version') == (TA_M5_EQUAL_LOSS_RESULT_SCHEMA if equal_loss else TA_M5_RESULT_SCHEMA if dedicated else TA_RESULT_SCHEMA if ta_sf else RESULT_SCHEMA)
              and report.get('recipe') == expected_recipe
              and report.get('parent_order_sha256') == result['parent_order_sha256']
              and report.get('parent_cursor') == result['parent_cursor']
              and report.get('physical_optimizer_updates') == result['optimizer_updates']
              and report.get('final_state') == result['full_state']
              and report.get('test_data_used') is False
              and (result['outcome'] != 'COMPLETE' or
                   report.get('decision') == ('PASS_TA_M5_EQUAL_LOSS_MECHANICAL' if equal_loss else 'PASS_TA_M5_MECHANICAL' if dedicated else 'PASS_TA_SF_MECHANICAL' if ta_sf else 'PASS_NATIVE_F_PRIVATE_RESUME')
                   and report.get('bit_exact_resume') is True),
              'CURSOR_RESULT')
    return result

# Zero-update import qualification is separate from the consumed four-step run.
TA_M5_IMPORT_SCOPE = 'market_state_ta_m5_equal_weight_import_v1'
TA_M5_IMPORT_STAGE = 'market_state_ta_m5_equal_weight_import_20261109_001'
TA_M5_IMPORT_ROOT = Path('/home/andre2/GX1_RUNS') / TA_M5_IMPORT_STAGE.upper()
TA_M5_IMPORT_RUNTIME_ROOT = TA_M5_IMPORT_ROOT / 'RUNTIME'
TA_M5_IMPORT_RECIPE_PATH = TA_M5_IMPORT_ROOT / 'RECIPE.json'
TA_M5_IMPORT_RECIPE_SCHEMA = 'gx1_market_state_ta_m5_equal_weight_import_recipe_v1'
TA_M5_IMPORT_CURSOR_SCHEMA = 'gx1_market_state_ta_m5_equal_weight_import_cursor_v1'
TA_M5_IMPORT_RESULT_SCHEMA = 'gx1_market_state_ta_m5_equal_weight_import_result_v1'
TA_M5_IMPORT_BASE_COMMIT = '9007e811f8196432620fc455d82e98e457357555'
TA_M5_IMPORT_INITIAL_SHA = '71278ead438bb991343ec3c7e77d9a77dd9a4c2f91f308b5478d594693766ba4'
_EQ_ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_DIRECTION_EQUAL_WEIGHT_20261109_001')
TA_M5_IMPORT_FIXED_REFS = {
    'equal_plan': {'path': str(_EQ_ROOT / 'PLAN.json'), 'sha256': '2271dbfffb0ae2fae0196f0cd59c514057a36b50ccb114d4618cd5f3b72b4c32'},
    'equal_result': {'path': str(_EQ_ROOT / 'RESULT.json'), 'sha256': '4d0189738896d76a7a459f74a0c1a262ce5cf16f19432f3513015c9bd45fef5c'},
    'equal_verification': {'path': str(_EQ_ROOT / 'VERIFICATION.json'), 'sha256': '85482fd6fabd6d97ec14769a50f44b47e95d9b4102f0d9c4a06e6b5ae4a3c9fe'},
    'equal_coefficients': {'path': str(_EQ_ROOT / 'PRIVATE_READOUTS_AND_PREDICTIONS.npz'), 'sha256': '3754df052ccc130b2e9b41086ed09dcb940d30a70da5d2cd2380e52454bd670b'},
}


def _ta_m5_import_metadata():
    """Reuse completed physical metadata proofs, without loading tensor payload."""
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as full
    refs = ta_m5_input_refs()
    refs.update(TA_M5_IMPORT_FIXED_REFS)
    fitted = _read(refs['equal_result'], 'equal-weight completed result')
    checked = _read(refs['equal_verification'], 'equal-weight independent check')
    plan = _read(refs['equal_plan'], 'equal-weight coefficient recipe')
    _need(fitted['status'] == 'COMPLETE' and fitted['source_unchanged'] is True
          and fitted['plan'] == refs['equal_plan']
          and fitted['private_outputs'] == refs['equal_coefficients']
          and fitted['alpha'] == plan['alpha'] == 0.01
          and fitted['native_forwards'] == fitted['native_parameter_updates'] == 0
          and fitted['test_data_used'] is False
          and checked['status'] == 'PASS' and checked['result'] == refs['equal_result']
          and checked['test_data_used'] is False, 'M5_IMPORT_COEFFICIENT_PROVENANCE')
    refs.update(cache_result=plan['cache_result'], fit_cache=plan['cache']['fit'],
                score_cache=plan['cache']['score'])
    cached = _read(refs['cache_result'], 'original representation cache')
    _need(cached['status'] == 'COMPLETE' and cached['model_state_unchanged'] is True
          and cached['test_data_used'] is False
          and cached['checkpoint']['checkpoint'] == refs['original_checkpoint']
          and cached['checkpoint']['model_sha256'] == TA_M5_ORIGINAL_MODEL_SHA256
          and all(cached['cache_partitions'][p]['artifact'] == refs[p + '_cache']
                  for p in ('fit', 'score')), 'M5_IMPORT_CACHE_ORIGIN')
    view, _, _, origin = _require_ta_m5_input_proof(refs)
    for name, ref in refs.items():
        campaign.require_binding(ref, label='M5 import ' + name, verify_file=True)
    return refs, view, origin, full.ta_sf_numerical_recipe(dedicated_m5_side=True)


def _ta_m5_import_expected():
    refs, view, origin, numerical = _ta_m5_import_metadata()
    return {
        'schema_version': TA_M5_IMPORT_RECIPE_SCHEMA,
        'source_repo': str(CANONICAL_REPO), 'runtime_root': str(TA_M5_IMPORT_RUNTIME_ROOT),
        'admission_stage': TA_M5_IMPORT_STAGE, 'run_scope': TA_M5_IMPORT_SCOPE,
        'fold_id': 1, 'parent_selection': TA_PARENT_SELECTION,
        'parent_order_rows': 29574, 'selected_fit_rows': 32, 'pass_index': 0,
        'numerical_recipe': numerical, 'input_manifest': refs['mature_plan'],
        'input_refs': refs, 'expected_view_sha256': view['contract_sha256'],
        'expected_model_contract_sha256': numerical['research_objective_sha256'],
        'expected_loss_contract_sha256': numerical['native_F_loss_contract_sha256'],
        'warmstart_origin': origin, 'initial_model_state_sha256': TA_M5_IMPORT_INITIAL_SHA,
        'execution_binding_path': str(TA_M5_IMPORT_RUNTIME_ROOT / 'EXECUTION_BINDING.json'),
        'max_private_updates': 0, 'max_backwards': 0, 'max_forwards': 6,
        'max_model_constructions': 3, 'max_optimizer_constructions': 3,
        'cached_head_rows': {'fit': 29574, 'score': 17753},
        'output_parity_atol': 2e-6, 'output_parity_rtol': 1e-6,
        'max_checkpoint_bytes': MAX_CHECKPOINT_BYTES,
        'max_published_json_bytes': MAX_PUBLISHED_JSON_BYTES,
        'training_enabled': False, 'full_epoch_training_allowed': False,
        'test_data_used': False, 'predictive_value_claimed': False,
        'full_TRAIN_encoder_baseline_qualified': False,
        'source_base_commit': TA_M5_IMPORT_BASE_COMMIT,
    }


def build_ta_m5_import_recipe(*, source_commit):
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as full
    value = _ta_m5_import_expected()
    value['source_owners'] = {name: {'path': str(CANONICAL_REPO / name),
        'sha256': campaign.file_sha256(CANONICAL_REPO / name)}
        for name in sorted(full.required_source_paths(ta_sf=True))}
    value['recipe_sha256'] = campaign.canonical_sha256(value)
    return require_ta_m5_import_recipe(value, source_repo=CANONICAL_REPO,
                                      source_commit=source_commit)


def require_ta_m5_import_recipe(value, *, source_repo, source_commit, verify_files=True):
    import subprocess
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as full
    _need(verify_files and source_repo == CANONICAL_REPO, 'M5_IMPORT_PHYSICAL_SOURCE')
    expected = _ta_m5_import_expected()
    value = _identity(value, set(expected) | {'source_owners'},
                      TA_M5_IMPORT_RECIPE_SCHEMA, 'recipe_sha256')
    _need(all(value[k] == v and type(value[k]) is type(v) for k, v in expected.items()),
          'M5_IMPORT_RECIPE_SCOPE')
    _need(isinstance(source_commit, str) and campaign._COMMIT.fullmatch(source_commit)
          and subprocess.run(['git', 'merge-base', '--is-ancestor', TA_M5_IMPORT_BASE_COMMIT,
                              source_commit], cwd=CANONICAL_REPO, capture_output=True).returncode == 0,
          'M5_IMPORT_SOURCE_ANCESTRY')
    owners = value['source_owners']
    _need(set(owners) == full.required_source_paths(ta_sf=True), 'M5_IMPORT_OWNER_SET')
    state_path = 'gx1/contracts/market_state_time_query_fold1_fit_state_v1.py'
    for name, ref in owners.items():
        campaign.require_binding(ref, label='M5 import source ' + name, verify_file=True)
        _need(ref['path'] == str(CANONICAL_REPO / name), 'M5_IMPORT_SOURCE_PATH')
        if name in TA_M5_ORCHESTRATION_SOURCE_PATHS | {'gx1/contracts/local_random_access_campaign_v2.py'}:
            continue  # Exact new routing bytes are bound by the clean campaign.
        if name == state_path:
            old = subprocess.check_output(['git', 'show', TA_M5_IMPORT_BASE_COMMIT + ':' + name],
                                          cwd=CANONICAL_REPO)
            _need((CANONICAL_REPO / name).read_bytes().startswith(old),
                  'M5_IMPORT_OLD_STATE_METHODS_CHANGED')
            continue
        expected_sha = (_read(value['input_refs']['original_recipe'], 'original source')['source_owners'][name]['sha256']
            if name.startswith('.venv/') else _ta_m5_git_source_sha(TA_M5_IMPORT_BASE_COMMIT, name))
        _need(ref['sha256'] == expected_sha, 'M5_IMPORT_UNREVIEWED_DRIFT:' + name)
    return value


def require_ta_m5_import_gate(gate, recipe):
    expected = {
        'decision': 'ADMITTED', 'status': 'ADMITTED', 'ta_m5_import_authorized': True,
        'fold_id': 1, 'parent_selection': TA_PARENT_SELECTION, 'selected_fit_rows': 32,
        'model_constructions_authorized': 3, 'optimizer_constructions_authorized': 3,
        'model_forwards_authorized': 6, 'backward_calls_authorized': 0,
        'optimizer_steps_authorized': 0, 'training_enabled': False,
        'native_training_enabled': False, 'genuine_predictor_fit_authorized': False,
        'genuine_updates_authorized': False, 'genuine_backward_authorized': False,
        'test_data_used': False,
    }
    _need(isinstance(gate, Mapping) and all(gate.get(k) == v and type(gate[k]) is type(v)
          for k, v in expected.items()) and recipe['run_scope'] == TA_M5_IMPORT_SCOPE
          and recipe['admission_stage'] == TA_M5_IMPORT_STAGE, 'M5_IMPORT_POLICY_GATE')


def require_ta_m5_import_cursor(value, *, expected_recipe, verify_files=True, batch_size=16):
    fields = {'schema_version', 'recipe', 'outcome', 'optimizer_updates', 'parent_cursor',
              'parent_order_sha256', 'full_state', 'result', 'test_data_used'}
    value = _identity(value, fields, TA_M5_IMPORT_CURSOR_SCHEMA, 'cursor_sha256')
    _need(expected_recipe['path'] == str(TA_M5_IMPORT_RECIPE_PATH)
          and value['recipe'] == expected_recipe and type(batch_size) is int and batch_size == 16
          and value['outcome'] in ('COMPLETE', 'FAILED') and value['test_data_used'] is False
          and type(value['optimizer_updates']) is int and value['optimizer_updates'] == 0
          and type(value['parent_cursor']) is int and value['parent_cursor'] == 0,
          'M5_IMPORT_CURSOR_SCOPE')
    campaign._sha(value['parent_order_sha256'], 'import parent order')
    campaign.require_binding(value['result'], label='import result', verify_file=verify_files)
    if value['full_state'] is not None:
        campaign.require_binding(value['full_state'], label='import state', verify_file=verify_files)
    _need(value['outcome'] != 'COMPLETE' or value['full_state'] is not None,
          'M5_IMPORT_COMPLETE_STATE')
    if verify_files:
        report = _read(value['result'], 'import result')
        _need(report['schema_version'] == TA_M5_IMPORT_RESULT_SCHEMA
              and all(report[k] == value[k] for k in
                      ('recipe', 'parent_order_sha256', 'parent_cursor', 'test_data_used'))
              and report['physical_optimizer_updates'] == 0
              and report['final_state'] == value['full_state']
              and (value['outcome'] != 'COMPLETE' or
                   report['decision'] == 'PASS_TA_M5_EQUAL_WEIGHT_IMPORT'
                   and report['bit_exact_resume'] is True), 'M5_IMPORT_CURSOR_RESULT')
    return value


# New-loss resume qualification starts from the independently verified imported
# predictor; it never reclones the old shared head or replays the import plan.
TA_M5_EQUAL_LOSS_STAGE = 'market_state_ta_m5_equal_loss_mechanical_20261109_001'
TA_M5_EQUAL_LOSS_SCOPE = 'market_state_ta_m5_equal_loss_mechanical_v1'
TA_M5_EQUAL_LOSS_ROOT = Path('/home/andre2/GX1_RUNS') / TA_M5_EQUAL_LOSS_STAGE.upper()
TA_M5_EQUAL_LOSS_RUNTIME_ROOT = TA_M5_EQUAL_LOSS_ROOT / 'RUNTIME'
TA_M5_EQUAL_LOSS_RECIPE_PATH = TA_M5_EQUAL_LOSS_ROOT / 'RECIPE.json'
TA_M5_EQUAL_LOSS_RECIPE_SCHEMA = 'gx1_market_state_ta_m5_equal_loss_mechanical_recipe_v1'
TA_M5_EQUAL_LOSS_CURSOR_SCHEMA = 'gx1_market_state_ta_m5_equal_loss_mechanical_cursor_v1'
TA_M5_EQUAL_LOSS_RESULT_SCHEMA = 'gx1_market_state_ta_m5_equal_loss_mechanical_result_v1'
TA_M5_EQUAL_LOSS_IMPORT_COMMIT = '338f8b3f16d3621dd5cd8d4d31942ae8424d9c78'
TA_M5_EQUAL_LOSS_AUDIT_COMMIT = '269d9004005fe488be822f1c5ae51c7464d15182'
TA_M5_EQUAL_LOSS_MODEL_SHA256 = 'ae772fd755cbfa07b1b7bcca8bccd2c8a8889806989a289ddb8267972c281b37'
TA_M5_EQUAL_LOSS_RNG_SHA256 = 'eff47232fb6c3273f3ce779d5089a9c1ea1ad6e3c43506a2e9e8b55f91be57bf'
TA_M5_EQUAL_LOSS_ORDER_SHA256 = 'db8e77282a7318873530888a2a636a728b1db109405da3f59c16b74aa9f4c3fb'
_TA_M5_EQUAL_LOSS_AUDIT_ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_EQUAL_OBSERVATION_LOSS_20261109_001')
TA_M5_EQUAL_LOSS_FIXED_REFS = {
    'import_recipe': {'path': str(TA_M5_IMPORT_RECIPE_PATH), 'sha256': '5d07910b5fe4ef5fcc16f68c4656f3a06f4001a02a768f104c3f04302e885aa1'},
    'import_result': {'path': str(TA_M5_IMPORT_ROOT / 'RESULT.json'), 'sha256': '46e94426529cb37384109cc9af32bfeaadbfd8eb07c3f6b0d0fa5a7a28a786de'},
    'import_terminal': {'path': str(TA_M5_IMPORT_ROOT / 'TERMINAL.json'), 'sha256': '5b1d89fdf6bf7d46968d64d2064aa29b09aa2c1005f9aecf95621c5616682efc'},
    'import_verification': {'path': str(TA_M5_IMPORT_ROOT / 'ARTIFACT_VERIFICATION.json'), 'sha256': 'acd0c3be92a6dc7d0ee32e42b3b3dfa7ee2c3406d62736080524c73f870ef44c'},
    'import_review': {'path': str(TA_M5_IMPORT_ROOT / 'REVIEW.json'), 'sha256': 'b683ad03d535d1c5ddb1d4ecfa5c2a692c231d86c60892de24025772ca67342b'},
    'imported_checkpoint': {'path': str(TA_M5_IMPORT_RUNTIME_ROOT / 'private-research/IMPORTED_M5_STATE.pt'), 'sha256': '616a22e467ebf35bded61705e00f99531d28420db90c30b8d755d447c612679f'},
    'saved_loss_review': {'path': str(_TA_M5_EQUAL_LOSS_AUDIT_ROOT / 'REVIEW.json'), 'sha256': '651e4aa6b44da41a0570a0b06fe32ef83d93dfd2b305dcd8a631a46e66c7caca'},
}
TA_M5_EQUAL_LOSS_ORCHESTRATION_PATHS = frozenset({
    'gx1/contracts/market_state_native_F_private_campaign_v1.py',
    'gx1/contracts/entry_native_direction_research_campaign_v1.py',
    'gx1/scripts/materialize_market_state_time_query_fold1_fit_campaign_v1.py',
    'gx1/scripts/run_entry_native_direction_research_window_v1.py',
    'gx1/scripts/run_market_state_native_F_private_window_v1.py',
})
TA_M5_EQUAL_LOSS_AUDITED_PATHS = frozenset({
    'gx1/contracts/market_state_time_query_fold1_fit_state_v1.py',
    'gx1/contracts/market_state_time_query_fold1_fit_v1.py',
    'gx1/contracts/market_state_time_query_native_F_batch_v1.py',
    'gx1/contracts/market_state_time_query_native_F_objective_v1.py',
    'gx1/contracts/market_state_time_query_side_normalized_loss_v1.py',
    'gx1/scripts/run_market_state_time_query_fold1_fit_window_v1.py',
})


def _ta_m5_equal_loss_import_origin(refs, imported, checked, review, terminal):
    head = imported.get('head_import', {})
    checkpoint = refs['imported_checkpoint']
    _need(imported.get('schema_version') == TA_M5_IMPORT_RESULT_SCHEMA
          and imported.get('decision') == 'PASS_TA_M5_EQUAL_WEIGHT_IMPORT'
          and imported.get('recipe') == refs['import_recipe']
          and imported.get('final_state') == checkpoint
          and imported.get('source_commit') == TA_M5_EQUAL_LOSS_IMPORT_COMMIT
          and imported.get('source_unchanged') is True
          and imported.get('error') is None and imported.get('test_data_used') is False
          and imported.get('bit_exact_resume') is True
          and type(imported.get('physical_optimizer_updates')) is int
          and imported['physical_optimizer_updates'] == 0
          and type(imported.get('parent_cursor')) is int and imported['parent_cursor'] == 0
          and imported.get('parent_order_sha256') == TA_M5_EQUAL_LOSS_ORDER_SHA256
          and head.get('model_state_sha256') == TA_M5_EQUAL_LOSS_MODEL_SHA256
          and head.get('rng_state_sha256') == TA_M5_EQUAL_LOSS_RNG_SHA256,
          'M5_EQUAL_LOSS_IMPORTED_ORIGIN')
    _need(checked.get('status') == 'PASS_INDEPENDENT_MODEL_ARTIFACTS_CAMPAIGN_CONTROLLER_FAILED'
          and checked.get('result') == refs['import_result']
          and checked.get('checkpoint') == checkpoint
          and checked.get('imported_model_state_sha256') == TA_M5_EQUAL_LOSS_MODEL_SHA256
          and checked.get('unchanged_buffers_optimizer_rng_modes_parent_order') is True
          and checked.get('coefficient_tensors_bit_equal') is True
          and checked.get('original_campaign_receipt_present') is False
          and checked.get('test_data_used') is False
          and review.get('status') == 'MODEL_IMPORT_VERIFIED_CAMPAIGN_CONTROLLER_FAILED_CONSUMED'
          and review.get('result') == refs['import_result']
          and review.get('artifact_verification') == refs['import_verification']
          and review.get('native_terminal') == refs['import_terminal']
          and review.get('imported_checkpoint') == checkpoint
          and review.get('imported_model_state_sha256') == TA_M5_EQUAL_LOSS_MODEL_SHA256
          and review.get('checkpoint_resume_bit_exact') is True
          and review.get('test_data_used') is False
          and terminal.get('status') == 'COMPLETE' and terminal.get('exit_code') == 0
          and terminal.get('result') == refs['import_result']
          and terminal.get('recipe') == refs['import_recipe']
          and terminal.get('no_relaunch_under_this_plan') is True
          and terminal.get('test_data_used') is False,
          'M5_EQUAL_LOSS_IMPORTED_PROOF')
    return {'imported_checkpoint': dict(checkpoint),
            'source_commit': TA_M5_EQUAL_LOSS_IMPORT_COMMIT,
            'model_state_sha256': TA_M5_EQUAL_LOSS_MODEL_SHA256,
            'rng_state_sha256': TA_M5_EQUAL_LOSS_RNG_SHA256,
            'parent_order_sha256': TA_M5_EQUAL_LOSS_ORDER_SHA256}


def _ta_m5_equal_loss_metadata():
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as full
    base_refs = ta_m5_input_refs()
    view, _, _, _ = _require_ta_m5_input_proof(base_refs)
    refs = {name: dict(ref) for name, ref in base_refs.items()}
    refs.update({name: dict(ref) for name, ref in TA_M5_EQUAL_LOSS_FIXED_REFS.items()})
    original = _read(refs['import_recipe'], 'M5 equal-loss original import recipe')
    imported = _read(refs['import_result'], 'M5 equal-loss imported result')
    origin = _ta_m5_equal_loss_import_origin(refs, imported,
        _read(refs['import_verification'], 'M5 equal-loss imported verification'),
        _read(refs['import_review'], 'M5 equal-loss imported review'),
        _read(refs['import_terminal'], 'M5 equal-loss imported native terminal'))
    numerical = full.ta_sf_numerical_recipe(dedicated_m5_side=True, m5_equal_observation=True)
    _need(original.get('numerical_recipe') == full.ta_sf_numerical_recipe(dedicated_m5_side=True)
          and original.get('expected_view_sha256') == view['contract_sha256']
          and original.get('expected_model_contract_sha256') == numerical['research_objective_sha256']
          and all(original['input_refs'][name] == ref for name, ref in base_refs.items()),
          'M5_EQUAL_LOSS_UNCHANGED_INPUTS_AND_MODEL')
    review = _read(refs['saved_loss_review'], 'M5 equal-loss saved-logit review')
    refs.update({name: dict(review[key]) for name, key in (
        ('saved_loss_result', 'result'), ('saved_loss_terminal', 'terminal'),
        ('saved_loss_plan', 'plan'))})
    result = _read(refs['saved_loss_result'], 'M5 equal-loss full saved-logit result')
    terminal = _read(refs['saved_loss_terminal'], 'M5 equal-loss saved-logit terminal')
    plan = _read(refs['saved_loss_plan'], 'M5 equal-loss saved-logit plan')
    refs['saved_loss_source_validation'] = dict(plan['inputs']['source_validation'])
    validation = _read(refs['saved_loss_source_validation'], 'M5 equal-loss audited source')
    _need(review.get('status') == 'PASS' and review.get('actual_exit_code') == 0
          and review.get('native_training_authorized') is False
          and terminal.get('status') == 'COMPLETE' and terminal.get('exit_code') == 0
          and terminal.get('result') == refs['saved_loss_result']
          and result.get('status') == 'COMPLETE'
          and result.get('plan') == refs['saved_loss_plan']
          and result.get('execution_head') == TA_M5_EQUAL_LOSS_AUDIT_COMMIT
          and result.get('batches') == 1849 and result.get('raw_logit_autograd_calls') == 3698
          and all(result.get(key) is True for key in (
              'activity_gradients_bitexact', 'other_TF_gradients_and_losses_bitexact',
              'masks_counts_skips_unchanged', 'unknown_M5_side_gradients_zero',
              'default_contracts_unchanged', 'source_unchanged'))
          and all(result.get(key) == 0 for key in (
              'native_model_constructions', 'model_forwards', 'model_backwards', 'optimizer_steps'))
          and result.get('test_data_used') is False
          and plan.get('weighting_profile') == numerical['m5_side_weighting_profile'],
          'M5_EQUAL_LOSS_SAVED_AUDIT')
    for task in ('state', 'future'):
        counts = numerical['m5_side_weighting_profile'][task]
        summary = result['summaries'][task]
        _need(summary['known_rows'] == counts['known_side_rows']
              and summary['nonempty_batches'] == counts['nonempty_batches']
              and summary['tail_rows'] == 6, 'M5_EQUAL_LOSS_AUDITED_COUNTS')
    return refs, view, origin, original, validation, numerical


def _ta_m5_equal_loss_expected():
    refs, view, origin, original, validation, numerical = _ta_m5_equal_loss_metadata()
    value = {
        'schema_version': TA_M5_EQUAL_LOSS_RECIPE_SCHEMA,
        'source_repo': str(CANONICAL_REPO), 'runtime_root': str(TA_M5_EQUAL_LOSS_RUNTIME_ROOT),
        'admission_stage': TA_M5_EQUAL_LOSS_STAGE, 'run_scope': TA_M5_EQUAL_LOSS_SCOPE,
        'fold_id': 1, 'parent_selection': TA_PARENT_SELECTION,
        'parent_order_rows': 29574, 'selected_fit_rows': 32, 'pass_index': 0,
        'numerical_recipe': numerical, 'input_manifest': refs['mature_plan'], 'input_refs': refs,
        'expected_view_sha256': view['contract_sha256'],
        'expected_model_contract_sha256': numerical['research_objective_sha256'],
        'expected_loss_contract_sha256': numerical['native_F_loss_contract_sha256'],
        'expected_initial_model_state_sha256': TA_M5_EQUAL_LOSS_MODEL_SHA256,
        'imported_origin': origin,
        'execution_binding_path': str(TA_M5_EQUAL_LOSS_RUNTIME_ROOT / 'EXECUTION_BINDING.json'),
        'source_base_commit': TA_M5_EQUAL_LOSS_AUDIT_COMMIT,
        'max_private_updates': 4, 'max_backwards': 4, 'max_forwards': 6,
        'max_model_constructions': 3, 'max_optimizer_constructions': 3,
        'max_checkpoint_bytes': MAX_CHECKPOINT_BYTES,
        'max_published_json_bytes': MAX_PUBLISHED_JSON_BYTES,
        'training_enabled': True, 'full_epoch_training_allowed': False,
        'fresh_initial_baseline_required': True, 'full_TRAIN_encoder_baseline_qualified': False,
        'qualification_weights_for_main_fit': False, 'test_data_used': False,
        'predictive_value_claimed': False,
    }
    return value, original, validation


def build_ta_m5_equal_loss_mechanical_recipe(*, source_commit):
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as full
    value, _, _ = _ta_m5_equal_loss_expected()
    value['source_owners'] = {name: {'path': str(CANONICAL_REPO / name),
        'sha256': campaign.file_sha256(CANONICAL_REPO / name)}
        for name in sorted(full.required_source_paths(ta_sf=True))}
    value['recipe_sha256'] = campaign.canonical_sha256(value)
    return require_ta_m5_equal_loss_mechanical_recipe(value,
        source_repo=CANONICAL_REPO, source_commit=source_commit)


def require_ta_m5_equal_loss_mechanical_recipe(value, *, source_repo, source_commit, verify_files=True):
    import subprocess
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as full
    _need(verify_files and source_repo == CANONICAL_REPO, 'M5_EQUAL_LOSS_PHYSICAL_SOURCE')
    expected, original, validation = _ta_m5_equal_loss_expected()
    value = _identity(value, set(expected) | {'source_owners'},
                      TA_M5_EQUAL_LOSS_RECIPE_SCHEMA, 'recipe_sha256')
    _need(all(value[k] == v and type(value[k]) is type(v) for k, v in expected.items()),
          'M5_EQUAL_LOSS_RECIPE_SCOPE')
    _need(isinstance(source_commit, str) and campaign._COMMIT.fullmatch(source_commit)
          and subprocess.run(['git', 'merge-base', '--is-ancestor', TA_M5_EQUAL_LOSS_AUDIT_COMMIT,
                              source_commit], cwd=CANONICAL_REPO, capture_output=True).returncode == 0,
          'M5_EQUAL_LOSS_SOURCE_ANCESTRY')
    owners = value['source_owners']
    _need(set(owners) == set(original['source_owners']) == full.required_source_paths(ta_sf=True)
          and len(owners) == 85, 'M5_EQUAL_LOSS_OWNER_SET')
    audited = {name: ref for name, ref in validation['source_files'].items() if name.startswith('gx1/')}
    _need(set(audited) == TA_M5_EQUAL_LOSS_AUDITED_PATHS, 'M5_EQUAL_LOSS_AUDITED_OWNER_SET')
    state_path = 'gx1/contracts/market_state_time_query_fold1_fit_state_v1.py'
    for name, ref in owners.items():
        campaign.require_binding(ref, label='M5 equal-loss source ' + name, verify_file=True)
        _need(ref['path'] == str(CANONICAL_REPO / name), 'M5_EQUAL_LOSS_SOURCE_PATH')
        if name in TA_M5_EQUAL_LOSS_ORCHESTRATION_PATHS:
            continue  # New routing bytes are bound to the clean one-shot campaign.
        if name == state_path:
            old = subprocess.check_output(['git', 'show', TA_M5_EQUAL_LOSS_AUDIT_COMMIT + ':' + name],
                                          cwd=CANONICAL_REPO)
            _need(hashlib.sha256(old).hexdigest() == audited[name]['sha256']
                  and (CANONICAL_REPO / name).read_bytes().startswith(old),
                  'M5_EQUAL_LOSS_OLD_STATE_METHODS_CHANGED')
            continue
        expected_ref = audited[name] if name in audited else original['source_owners'][name]
        _need(ref == expected_ref, 'M5_EQUAL_LOSS_UNREVIEWED_DRIFT:' + name)
    for name, ref in value['input_refs'].items():
        campaign.require_binding(ref, label='M5 equal-loss input ' + name, verify_file=True)
    return value


def require_ta_m5_equal_loss_mechanical_gate(gate, recipe):
    expected = {
        'decision': 'ADMITTED', 'status': 'ADMITTED', 'ta_m5_equal_loss_mechanical_authorized': True,
        'fold_id': 1, 'parent_selection': TA_PARENT_SELECTION, 'selected_fit_rows': 32,
        'model_constructions_authorized': 3, 'optimizer_constructions_authorized': 3,
        'model_forwards_authorized': 6, 'backward_calls_authorized': 4,
        'optimizer_steps_authorized': 4, 'training_enabled': True,
        'native_training_enabled': True, 'genuine_predictor_fit_authorized': False,
        'genuine_updates_authorized': True, 'genuine_backward_authorized': True,
        'test_data_used': False,
    }
    _need(isinstance(gate, Mapping) and all(gate.get(k) == v and type(gate[k]) is type(v)
          for k, v in expected.items()) and recipe['run_scope'] == TA_M5_EQUAL_LOSS_SCOPE
          and recipe['admission_stage'] == TA_M5_EQUAL_LOSS_STAGE, 'M5_EQUAL_LOSS_POLICY_GATE')
