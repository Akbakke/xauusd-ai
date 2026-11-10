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



TA_M5_FINAL_IMPORT_STAGE = 'market_state_ta_m5_final_readout_import_20261109_001'
TA_M5_FINAL_IMPORT_ROOT = Path('/home/andre2/GX1_RUNS') / TA_M5_FINAL_IMPORT_STAGE.upper()
TA_M5_FINAL_IMPORT_RUNTIME_ROOT = TA_M5_FINAL_IMPORT_ROOT / 'RUNTIME'
TA_M5_FINAL_IMPORT_RECIPE_PATH = TA_M5_FINAL_IMPORT_ROOT / 'RECIPE.json'
TA_M5_FINAL_IMPORT_BASE_COMMIT = '3e504a5d93e0fcadc3966aa51709217f6e93c9b9'
TA_M5_FINAL_IMPORT_CHANGED_SOURCES = frozenset({
    'gx1/contracts/market_state_native_F_private_campaign_v1.py',
    'gx1/contracts/entry_native_direction_research_campaign_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
    'gx1/scripts/run_market_state_native_F_private_window_v1.py',
    'gx1/scripts/materialize_market_state_time_query_fold1_fit_campaign_v1.py',
})
_FINAL_READOUT_ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_FINAL_READOUT_20261109_001')
TA_M5_FINAL_IMPORT_FIXED_REFS = {
    'equal_plan': {'path': str(_FINAL_READOUT_ROOT / 'PLAN.json'), 'sha256': 'fe5bbbd7b6faba84335b47b3d3e4525f5566ea1dc04359e4f7ed04b4188b8972'},
    'equal_result': {'path': str(_FINAL_READOUT_ROOT / 'RESULT.json'), 'sha256': '623e66c9baba2a3cf6910ffb10a11ba2e14a4c2e514e3660a83528c361c16ae6'},
    'equal_verification': {'path': str(_FINAL_READOUT_ROOT / 'REVIEW.json'), 'sha256': 'deff9b4774ad68e6306d598d2073f43e36b126cc29069122965b5319d4ef310d'},
}


def ta_m5_import_paths(stage):
    if stage == TA_M5_IMPORT_STAGE:
        return TA_M5_IMPORT_ROOT, TA_M5_IMPORT_RUNTIME_ROOT, TA_M5_IMPORT_RECIPE_PATH
    _need(stage == TA_M5_FINAL_IMPORT_STAGE, 'M5_IMPORT_STAGE')
    return TA_M5_FINAL_IMPORT_ROOT, TA_M5_FINAL_IMPORT_RUNTIME_ROOT, TA_M5_FINAL_IMPORT_RECIPE_PATH


def _ta_m5_final_import_metadata():
    """Bind the completed dedicated FINAL and fixed readouts without tensor loading."""
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as full
    from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero
    fixed = {k: dict(v) for k, v in TA_M5_FINAL_IMPORT_FIXED_REFS.items()}
    fitted = _read(fixed['equal_result'], 'FINAL readout result')
    checked = _read(fixed['equal_verification'], 'FINAL readout independent check')
    readout_plan = _read(fixed['equal_plan'], 'FINAL readout plan')
    _need(fitted['status'] == 'COMPLETE' and fitted['source_unchanged'] is True
          and fitted['plan'] == fixed['equal_plan']
          and fitted['alpha'] == readout_plan['alpha'] == 0.01
          and fitted['readout_fits'] == 2 and all(v['converged'] for v in fitted['solvers'].values())
          and fitted['native_forwards'] == fitted['native_parameter_updates'] == 0
          and fitted['test_data_used'] is False
          and checked['status'] == 'COMPLETE_CONSUMED_READOUT_RECOVERS_TRAIN_NOT_LATER_EDGE'
          and checked['result'] == fixed['equal_result']
          and checked['independent_saved_prediction_metric_max_error'] == 0
          and checked['test_data_used'] is False, 'FINAL_IMPORT_READOUT_PROVENANCE')
    cached = _read(readout_plan['inputs']['new_cache_result'], 'dedicated FINAL cache')
    cache_plan = _read(cached['plan'], 'dedicated FINAL cache plan')
    original = _read(cache_plan['recipe'], 'dedicated FINAL recipe')
    result = _read(cache_plan['result'], 'dedicated FINAL native result')
    measurement = _read(result['final_train_measurement'], 'dedicated FINAL measurement')
    numerical = full.ta_sf_numerical_recipe(dedicated_m5_side=True, m5_equal_observation=True)
    _need(cache_plan['recipe']['path'] == str(full.TA_M5_EQUAL_LOSS_FIT_RECIPE_PATH)
          and original['numerical_recipe'] == numerical
          and result['recipe'] == cache_plan['recipe']
          and result['outcome'] == result['phase_state'] == 'COMPLETE'
          and result['source_unchanged'] is True and result['test_data_used'] is False
          and result['dedicated_m5_side'] is True and result['m5_equal_observation'] is True
          and result['parent_cursor'] == 29574 and result['optimizer_updates'] == 1849
          and result['skipped_batches'] == 0
          and cached['status'] == 'COMPLETE' and cached['model_state_unchanged'] is True
          and cached['source_unchanged'] is True and cached['test_data_used'] is False
          and cached['checkpoint']['checkpoint'] == result['full_state']
          and cached['checkpoint']['source_commit'] == result['source_commit']
          and measurement['evaluated_checkpoint'] == result['full_state']
          and measurement['evaluated_model_state_sha256'] == cached['checkpoint']['model_sha256'],
          'FINAL_IMPORT_ORIGIN')
    refs = dict(original['input_refs'], **fixed)
    refs.update(equal_coefficients=fitted['private_outputs'],
        original_recipe=cache_plan['recipe'], original_result=cache_plan['result'],
        original_checkpoint=result['full_state'], original_measurement=result['final_train_measurement'],
        original_candidate_view=original['input_refs']['ta_candidate_view'],
        cache_result=readout_plan['inputs']['new_cache_result'], cache_plan=cached['plan'])
    for part in ('fit', 'score'):
        refs[part + '_cache'] = readout_plan['inputs']['new_' + part + '_cache']
        _need(cached['cache_partitions'][part]['artifact'] == refs[part + '_cache']
              and cached['cache_partitions'][part]['partial'] is False
              and cached['cache_partitions'][part]['rows'] == (29574 if part == 'fit' else 17753),
              'FINAL_IMPORT_CACHE_PARTITION')
    current = {name: campaign.file_sha256(CANONICAL_REPO / name)
               for name in full.required_source_paths(ta_sf=True)}
    receipts = zero.load_ta_input_receipts(current_source_owners=current)
    view = zero._bind_ta_view(receipts, dedicated_m5_side=True)
    _need(view['contract_sha256'] == original['expected_view_sha256'] == cached['view_sha256'],
          'FINAL_IMPORT_ACTUAL_VIEW')
    full.require_checkpoint_view(result['feature_view_identity'], recipe=original, ta_sf=True)
    origin = dict(original_recipe=refs['original_recipe'],
        original_candidate_view=refs['original_candidate_view'],
        original_checkpoint=refs['original_checkpoint'], source_commit=result['source_commit'],
        model_state_sha256=cached['checkpoint']['model_sha256'],
        parent_order_sha256=result['parent_order_sha256'],
        cursor=result['parent_cursor'], optimizer_updates=result['optimizer_updates'],
        skipped_batches=result['skipped_batches'])
    for name, ref in refs.items():
        campaign.require_binding(ref, label='FINAL import ' + name, verify_file=True)
    return refs, view, origin, numerical


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


def _ta_m5_import_expected(*, final_readout=False):
    _need(type(final_readout) is bool, 'FINAL_IMPORT_MODE')
    refs, view, origin, numerical = (_ta_m5_final_import_metadata() if final_readout
                                    else _ta_m5_import_metadata())
    stage = TA_M5_FINAL_IMPORT_STAGE if final_readout else TA_M5_IMPORT_STAGE
    _, runtime, _ = ta_m5_import_paths(stage)
    value = {
        'schema_version': TA_M5_IMPORT_RECIPE_SCHEMA,
        'source_repo': str(CANONICAL_REPO), 'runtime_root': str(runtime),
        'admission_stage': stage, 'run_scope': TA_M5_IMPORT_SCOPE,
        'fold_id': 1, 'parent_selection': TA_PARENT_SELECTION,
        'parent_order_rows': 29574, 'selected_fit_rows': 32, 'pass_index': 0,
        'numerical_recipe': numerical, 'input_manifest': refs['mature_plan'],
        'input_refs': refs, 'expected_view_sha256': view['contract_sha256'],
        'expected_model_contract_sha256': numerical['research_objective_sha256'],
        'expected_loss_contract_sha256': numerical['native_F_loss_contract_sha256'],
        'warmstart_origin': origin, 'initial_model_state_sha256': (
            origin['model_state_sha256'] if final_readout else TA_M5_IMPORT_INITIAL_SHA),
        'execution_binding_path': str(runtime / 'EXECUTION_BINDING.json'),
        'max_private_updates': 0, 'max_backwards': 0, 'max_forwards': 6,
        'max_model_constructions': 3, 'max_optimizer_constructions': 3,
        'cached_head_rows': {'fit': 29574, 'score': 17753},
        'output_parity_atol': 2e-6, 'output_parity_rtol': 1e-6,
        'max_checkpoint_bytes': MAX_CHECKPOINT_BYTES,
        'max_published_json_bytes': MAX_PUBLISHED_JSON_BYTES,
        'training_enabled': False, 'full_epoch_training_allowed': False,
        'test_data_used': False, 'predictive_value_claimed': False,
        'full_TRAIN_encoder_baseline_qualified': False,
        'source_base_commit': TA_M5_FINAL_IMPORT_BASE_COMMIT if final_readout else TA_M5_IMPORT_BASE_COMMIT,
    }
    if final_readout:
        value['final_readout_import'] = True
    return value


def build_ta_m5_import_recipe(*, source_commit, final_readout=False):
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as full
    value = _ta_m5_import_expected(final_readout=final_readout)
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
    final_readout = value.get('admission_stage') == TA_M5_FINAL_IMPORT_STAGE
    expected = _ta_m5_import_expected(final_readout=final_readout)
    value = _identity(value, set(expected) | {'source_owners'},
                      TA_M5_IMPORT_RECIPE_SCHEMA, 'recipe_sha256')
    _need(all(value[k] == v and type(value[k]) is type(v) for k, v in expected.items()),
          'M5_IMPORT_RECIPE_SCOPE')
    _need(isinstance(source_commit, str) and campaign._COMMIT.fullmatch(source_commit)
          and subprocess.run(['git', 'merge-base', '--is-ancestor', expected['source_base_commit'],
                              source_commit], cwd=CANONICAL_REPO, capture_output=True).returncode == 0,
          'M5_IMPORT_SOURCE_ANCESTRY')
    owners = value['source_owners']
    _need(set(owners) == full.required_source_paths(ta_sf=True), 'M5_IMPORT_OWNER_SET')
    state_path = 'gx1/contracts/market_state_time_query_fold1_fit_state_v1.py'
    for name, ref in owners.items():
        campaign.require_binding(ref, label='M5 import source ' + name, verify_file=True)
        _need(ref['path'] == str(CANONICAL_REPO / name), 'M5_IMPORT_SOURCE_PATH')
        if final_readout:
            if name not in TA_M5_FINAL_IMPORT_CHANGED_SOURCES:
                if name.startswith('.venv/'):
                    original = _read(value['input_refs']['original_recipe'], 'FINAL installed runtime')
                    expected_sha = original['source_owners'][name]['sha256']
                else:
                    expected_sha = _ta_m5_git_source_sha(TA_M5_FINAL_IMPORT_BASE_COMMIT, name)
                _need(ref['sha256'] == expected_sha, 'FINAL_IMPORT_UNREVIEWED_DRIFT:' + name)
            continue
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
          and recipe['admission_stage'] in (TA_M5_IMPORT_STAGE, TA_M5_FINAL_IMPORT_STAGE), 'M5_IMPORT_POLICY_GATE')


def require_ta_m5_import_cursor(value, *, expected_recipe, verify_files=True, batch_size=16):
    fields = {'schema_version', 'recipe', 'outcome', 'optimizer_updates', 'parent_cursor',
              'parent_order_sha256', 'full_state', 'result', 'test_data_used'}
    value = _identity(value, fields, TA_M5_IMPORT_CURSOR_SCHEMA, 'cursor_sha256')
    _need(expected_recipe['path'] in (str(TA_M5_IMPORT_RECIPE_PATH), str(TA_M5_FINAL_IMPORT_RECIPE_PATH))
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


# One exact historical CUDA prefix replay. The old four-step and import scopes
# remain immutable; this scope observes unchanged original training numerics.
TA_M5_ORIGINAL_CUDA_PREFIX_STAGE = 'market_state_ta_m5_original_cuda_prefix_20261110_001'
TA_M5_ORIGINAL_CUDA_PREFIX_SCOPE = 'market_state_ta_m5_original_cuda_prefix_v1'
TA_M5_ORIGINAL_CUDA_PREFIX_ROOT = Path('/home/andre2/GX1_RUNS') / TA_M5_ORIGINAL_CUDA_PREFIX_STAGE.upper()
TA_M5_ORIGINAL_CUDA_PREFIX_RUNTIME_ROOT = TA_M5_ORIGINAL_CUDA_PREFIX_ROOT / 'RUNTIME'
TA_M5_ORIGINAL_CUDA_PREFIX_RECIPE_PATH = TA_M5_ORIGINAL_CUDA_PREFIX_ROOT / 'RECIPE.json'
TA_M5_ORIGINAL_CUDA_PREFIX_RECIPE_SCHEMA = 'gx1_market_state_ta_m5_original_cuda_prefix_recipe_v1'
TA_M5_ORIGINAL_CUDA_PREFIX_RESULT_SCHEMA = 'gx1_market_state_ta_m5_original_cuda_prefix_result_v1'
TA_M5_ORIGINAL_CUDA_PREFIX_CURSOR_SCHEMA = 'gx1_market_state_ta_m5_original_cuda_prefix_cursor_v1'
TA_M5_ORIGINAL_CUDA_PREFIX_SOURCE = '84f0f47c4e0c02b9c5287b1930ffb30bbdf98e81'
TA_M5_ORIGINAL_CUDA_PREFIX_REFERENCE_SOURCE = 'b02fc5d46191d550f7b3f4682e65c098515c7ddf'
TA_M5_ORIGINAL_CUDA_PREFIX_WORKER = 'gx1/audit/trace_market_state_ta_m5_original_cuda_prefix_v1.py'
TA_M5_ORIGINAL_CUDA_PREFIX_ROUTING_PATHS = frozenset({
    'gx1/contracts/market_state_native_F_private_campaign_v1.py',
    'gx1/contracts/entry_native_direction_research_campaign_v1.py',
    'gx1/scripts/run_entry_native_direction_research_window_v1.py',
    'gx1/scripts/materialize_market_state_time_query_fold1_fit_campaign_v1.py',
})
_TA_ORIGINAL_FIT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_EQUAL_LOSS_FOLD1_FIT_20261109_002')
_TA_ORIGINAL_MIX = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_MIXED_BATCH_NATIVE_20261109_001')
TA_M5_ORIGINAL_CUDA_PREFIX_REFS = {
    'original_recipe': {'path': str(_TA_ORIGINAL_FIT / 'RECIPE.json'), 'sha256': 'b52a8ac46b3bf7e4827024431e6bccec77542ce0be09ebe9680edda15266b697'},
    'original_result': {'path': str(_TA_ORIGINAL_FIT / 'RUNTIME/prefix-fit/window-0001/RESULT.json'), 'sha256': '07fc4b631616e36fd7bc52a80375dfafe857eed51ca742965326886881844d55'},
    'initial_measurement': {'path': str(_TA_ORIGINAL_FIT / 'RUNTIME/prefix-fit/window-0001/INITIAL_TRAIN.json'), 'sha256': 'e4998ab3686d17e1273ef80f56cd483a5ca486ce45698b45773f553ab90acd92'},
    'original_initial_checkpoint': {'path': str(_TA_ORIGINAL_FIT / 'RUNTIME/prefix-fit/states/INITIAL_FOLD1_STATE.pt'), 'sha256': '80fc8029200c8076bc2faa38014bfd24c75ee71632fb1f45138249ccaca8d484'},
    'original_p4096_checkpoint': {'path': str(_TA_ORIGINAL_FIT / 'RUNTIME/prefix-fit/states/w0001_p004096.pt'), 'sha256': '2a8fb565228212f029c45712e0ad20484fdb1bacd6f9a6afead4d70af59a1497'},
    'source_reference_recipe': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_SCORE_RECOVERY_20261109_002/RECIPE.json', 'sha256': '4fc09408227deddfa3f73256d1a458e9b61d606995633e0aa4c6c1afdb73c120'},
    'contiguous_batch_review': {'path': str(CANONICAL_REPO / 'docs/TA_M5_MIXED_BATCH_NATIVE_REVIEW.json'), 'sha256': '972ede64bf9063e7e06bfd127411af968ed2635ee293715b7af6ba4e406a46af'},
    'contiguous_batch_tests': {'path': str(_TA_ORIGINAL_MIX / 'BATCH_API_TESTS.json'), 'sha256': '78da9219c574b6bbdc3bcef21176a497c574a8f610d730952bbc4b8a84eb6d60'},
    'mixed_plan': {'path': str(_TA_ORIGINAL_MIX / 'PLAN.json'), 'sha256': 'c14dc33b9ed94a5b15be7baa301a3c713dee22d074b9078b3ac563333a73ea42'},
    'mixed_result': {'path': str(_TA_ORIGINAL_MIX / 'RESULT.json'), 'sha256': '6895ee933974dae209c04251719033ffdff0ab70dbe1179e529a80cc60767cea'},
    'mixed_review': {'path': str(_TA_ORIGINAL_MIX / 'REVIEW.json'), 'sha256': 'd4a3bbcd3c9fdce924f6400b76944f141d8227a01e1dcf7bed9a7b87ca73fdfb'},
}
TA_M5_ORIGINAL_CUDA_PREFIX_CONSUMED_REF_KEYS = (
    'candidate_tape', 'F_tape', 'F_result', 'F_terminal',
    'mature_plan', 'fold1_plan', 'fold1_claim', 'fold1_result', 'fold1_terminal',
    'fold1_qualification', 'fold1_normalization', 'constructor', 'base_view',
    'ta_candidate_view', 'ta_input_audit', 'ta_input_terminal',
)
_TA_PREFIX_BATCH_PATH = 'gx1/contracts/market_state_time_query_native_F_batch_v1.py'
_TA_PREFIX_BATCH_OLD = '1167021db3c87d7b24a7d5f912267814b8dcc1640f81674ba9f6c6880cbca97b'
_TA_PREFIX_BATCH_CURRENT = 'ec62b841a866960c273fdbbacae2598d675d427339cfcd5c6942630c2b739fb5'


def _ta_m5_original_cuda_prefix_input_refs(original):
    refs = {key: dict(value) for key, value in TA_M5_ORIGINAL_CUDA_PREFIX_REFS.items()}
    refs.update({'original_input_' + key: dict(original['input_refs'][key])
                 for key in TA_M5_ORIGINAL_CUDA_PREFIX_CONSUMED_REF_KEYS})
    refs['original_input_manifest'] = dict(original['input_manifest'])
    return refs


def read_ta_m5_original_cuda_prefix_origin(launch):
    refs = launch['input_refs']
    _need({key: refs.get(key) for key in TA_M5_ORIGINAL_CUDA_PREFIX_REFS}
          == TA_M5_ORIGINAL_CUDA_PREFIX_REFS, 'ORIGINAL_PREFIX_FIXED_REFS')
    original = _read(refs['original_recipe'], 'original prefix recipe')
    _need(refs == _ta_m5_original_cuda_prefix_input_refs(original),
          'ORIGINAL_PREFIX_CONSUMED_INPUT_REFS')
    # Explicit original bindings are checked before decoding and automatically
    # included in the worker's existing input_refs bookend. No unused weights.
    for key, ref in refs.items():
        if key.startswith('original_input_'):
            campaign.require_binding(ref, label=key, verify_file=True)
    result = _read(refs['original_result'], 'original prefix result')
    initial = _read(refs['initial_measurement'], 'original prefix initial measurement')
    qualified = _read(refs['source_reference_recipe'], 'original prefix source reference')
    _need(result.get('source_commit') == TA_M5_ORIGINAL_CUDA_PREFIX_SOURCE
          and result.get('outcome') == result.get('phase_state') == 'COMPLETE'
          and result.get('source_unchanged') is True and result.get('test_data_used') is False
          and result.get('optimizer_updates') == 1849 and result.get('parent_cursor') == 29574
          and result.get('skipped_batches') == 0
          and result.get('recipe') == refs['original_recipe']
          and result.get('own_initial_state') == refs['original_initial_checkpoint']
          and result.get('initial_measurement') == refs['initial_measurement']
          and result.get('parent_order_sha256') == TA_M5_EQUAL_LOSS_ORDER_SHA256,
          'ORIGINAL_PREFIX_ORIGIN')
    _need(initial.get('evaluated_checkpoint') == refs['original_initial_checkpoint']
          and initial.get('evaluated_model_state_sha256') == TA_M5_EQUAL_LOSS_MODEL_SHA256
          and initial.get('optimizer_updates') == 0 and initial.get('test_data_used') is False,
          'ORIGINAL_PREFIX_INITIAL')
    numerical = original.get('numerical_recipe', {})
    _need(numerical.get('contract_sha256') == '5085579ff8ad90257412ad1d42406c6c481a1d272f0cf2849ae293da87974361'
          and numerical.get('max_updates') == 1849 and numerical.get('batch_size') == 16
          and numerical.get('grad_accum_steps') == 1 and numerical.get('m5_equal_observation') is True
          and original.get('expected_initial_model_state_sha256') == TA_M5_EQUAL_LOSS_MODEL_SHA256,
          'ORIGINAL_PREFIX_UNCHANGED_NUMERICAL')
    for key in ('expected_view_sha256', 'expected_model_contract_sha256', 'expected_loss_contract_sha256'):
        _need(original[key] == qualified[key], 'ORIGINAL_PREFIX_REFERENCE_IDENTITY:' + key)
    _need(original['input_refs']['ta_candidate_view'] == qualified['input_refs']['ta_candidate_view'],
          'ORIGINAL_PREFIX_REFERENCE_VIEW')
    return {'original_recipe': original, 'original_result': result,
            'initial_measurement': initial, 'source_reference_recipe': qualified}


def _ta_m5_original_cuda_prefix_sources(owners, original, qualified, trace_worker, *, paired_order_guard=False):
    _need(isinstance(owners, Mapping) and set(owners) == set(original['source_owners'])
          == set(qualified['source_owners']) and len(owners) == 85, 'ORIGINAL_PREFIX_OWNER_SET')
    historical, current = {}, {}
    for name, ref in owners.items():
        old, base = original['source_owners'][name], qualified['source_owners'][name]
        expected_path = str(CANONICAL_REPO / name)
        _need(ref.get('path') == old.get('path') == base.get('path') == expected_path,
              'ORIGINAL_PREFIX_OWNER_PATH')
        campaign.require_binding(ref, label='original prefix source ' + name, verify_file=True)
        if name.startswith('.venv/'):
            _need(ref == old == base, 'ORIGINAL_PREFIX_INSTALLED_RUNTIME_CHANGED:' + name)
        else:
            _need(_ta_m5_git_source_sha(TA_M5_ORIGINAL_CUDA_PREFIX_SOURCE, name) == old['sha256']
                  and _ta_m5_git_source_sha(TA_M5_ORIGINAL_CUDA_PREFIX_REFERENCE_SOURCE, name) == base['sha256'],
                  'ORIGINAL_PREFIX_HISTORICAL_SOURCE:' + name)
        if name == _TA_PREFIX_BATCH_PATH:
            _need(base['sha256'] == _TA_PREFIX_BATCH_OLD and ref['sha256'] == _TA_PREFIX_BATCH_CURRENT,
                  'ORIGINAL_PREFIX_BATCH_PAIR')
        elif paired_order_guard and name == 'scripts/gx1_capped_run.sh':
            _need(base['sha256'] == TA_M5_PAIRED_BATCH_ORDER_GUARD_OLD_SHA256
                  and ref['sha256'] == TA_M5_PAIRED_BATCH_ORDER_GUARD_CURRENT_SHA256,
                  'PAIRED_ORDER_GUARD_SOURCE_PAIR')
        elif name not in TA_M5_ORIGINAL_CUDA_PREFIX_ROUTING_PATHS:
            _need(ref == base, 'ORIGINAL_PREFIX_UNREVIEWED_SOURCE:' + name)
        if old != base:
            historical[name] = {'original': old, 'qualified': base}
        if ref != base:
            current[name] = {'qualified': base, 'current': dict(ref)}
    _need(trace_worker.get('path') == str(CANONICAL_REPO / TA_M5_ORIGINAL_CUDA_PREFIX_WORKER),
          'ORIGINAL_PREFIX_WORKER_PATH')
    campaign.require_binding(trace_worker, label='original prefix trace worker', verify_file=True)
    return {'original_source_commit': TA_M5_ORIGINAL_CUDA_PREFIX_SOURCE,
            'qualified_source_commit': TA_M5_ORIGINAL_CUDA_PREFIX_REFERENCE_SOURCE,
            'historical_changes': historical, 'current_changes': current}


def _ta_m5_original_cuda_prefix_expected():
    original = _read(TA_M5_ORIGINAL_CUDA_PREFIX_REFS['original_recipe'], 'original prefix recipe')
    refs = _ta_m5_original_cuda_prefix_input_refs(original)
    origin = read_ta_m5_original_cuda_prefix_origin({'input_refs': refs})
    original, qualified = origin['original_recipe'], origin['source_reference_recipe']
    for key in ('original_initial_checkpoint', 'original_p4096_checkpoint'):
        campaign.require_binding(refs[key], label=key, verify_file=True)
    review = _read(refs['contiguous_batch_review'], 'qualified contiguous batch review')
    tests = _read(refs['contiguous_batch_tests'], 'qualified contiguous batch tests')
    mixed_plan = _read(refs['mixed_plan'], 'qualified mixed plan')
    _read(refs['mixed_result'], 'qualified mixed result')
    _read(refs['mixed_review'], 'qualified mixed review')
    override = {'original_ref': {'path': str(CANONICAL_REPO / _TA_PREFIX_BATCH_PATH), 'sha256': _TA_PREFIX_BATCH_OLD},
                'current_ref': {'path': str(CANONICAL_REPO / _TA_PREFIX_BATCH_PATH), 'sha256': _TA_PREFIX_BATCH_CURRENT}}
    proof = review.get('source_change', {})
    _need(review.get('plan') == refs['mixed_plan'] and review.get('result') == refs['mixed_result']
          and review.get('review') == refs['mixed_review']
          and review.get('focused_tests') == refs['contiguous_batch_tests']
          and proof.get('focused_tests_passed') == 50
          and proof.get('legacy_contiguous_route_preserved') is True
          and proof.get('query_forward_targets_loss_suffix_byte_identical') is True
          and proof.get('runtime_source_override') == mixed_plan.get('runtime_source_override') == override
          and tests.get('exit_code') == 0 and tests.get('source') == override['current_ref']
          and tests.get('query_forward_targets_loss_suffix_byte_identical') is True,
          'ORIGINAL_PREFIX_CONTIGUOUS_BATCH_PROOF')
    value = {
        'schema_version': TA_M5_ORIGINAL_CUDA_PREFIX_RECIPE_SCHEMA,
        'source_repo': str(CANONICAL_REPO), 'runtime_root': str(TA_M5_ORIGINAL_CUDA_PREFIX_RUNTIME_ROOT),
        'admission_stage': TA_M5_ORIGINAL_CUDA_PREFIX_STAGE, 'run_scope': TA_M5_ORIGINAL_CUDA_PREFIX_SCOPE,
        'execution_binding_path': str(TA_M5_ORIGINAL_CUDA_PREFIX_RUNTIME_ROOT / 'EXECUTION_BINDING.json'),
        'original_source_commit': TA_M5_ORIGINAL_CUDA_PREFIX_SOURCE,
        'input_refs': refs, 'input_manifest': original['input_manifest'],
        'numerical_recipe': original['numerical_recipe'],
        'expected_view_sha256': original['expected_view_sha256'],
        'expected_model_contract_sha256': original['expected_model_contract_sha256'],
        'expected_loss_contract_sha256': original['expected_loss_contract_sha256'],
        'expected_initial_model_state_sha256': TA_M5_EQUAL_LOSS_MODEL_SHA256,
        'parent_order_sha256': TA_M5_EQUAL_LOSS_ORDER_SHA256,
        'fold_id': 1, 'pass_index': 0, 'parent_order_rows': 29574,
        'selected_fit_rows': 4096, 'parent_selection': 'original_chronological_fit_prefix_0_4096',
        'device': 'cuda', 'max_private_updates': 256, 'max_forwards': 256, 'max_backwards': 256,
        'max_model_constructions': 1, 'max_optimizer_constructions': 1,
        'max_checkpoint_loads': 2, 'max_full_state_restores': 1, 'max_checkpoint_writes': 0,
        'max_model_seconds': 720, 'max_trace_bytes': 67108864,
        'max_checkpoint_bytes': MAX_CHECKPOINT_BYTES, 'max_published_json_bytes': MAX_PUBLISHED_JSON_BYTES,
        'first_step_exact_required': True, 'final_full_state_exact_required': True,
        'initial_evaluation_forwards': 0, 'later_evaluation_forwards': 0,
        'training_enabled': True, 'full_epoch_training_allowed': False,
        'canonical_resume_authority': False, 'candidate_selection': False,
        'test_data_used': False, 'predictive_value_claimed': False,
    }
    return value, original, qualified


def build_ta_m5_original_cuda_prefix_recipe(*, source_commit):
    value, original, qualified = _ta_m5_original_cuda_prefix_expected()
    owners = {name: {'path': str(CANONICAL_REPO / name),
                     'sha256': campaign.file_sha256(CANONICAL_REPO / name)}
              for name in original['source_owners']}
    worker = {'path': str(CANONICAL_REPO / TA_M5_ORIGINAL_CUDA_PREFIX_WORKER),
              'sha256': campaign.file_sha256(CANONICAL_REPO / TA_M5_ORIGINAL_CUDA_PREFIX_WORKER)}
    value.update(source_owners=owners, trace_worker=worker,
                 source_bridge=_ta_m5_original_cuda_prefix_sources(owners, original, qualified, worker))
    value['recipe_sha256'] = campaign.canonical_sha256(value)
    return require_ta_m5_original_cuda_prefix_recipe(value, source_repo=CANONICAL_REPO,
                                                     source_commit=source_commit)


def require_ta_m5_original_cuda_prefix_recipe(value, *, source_repo, source_commit, verify_files=True):
    import subprocess
    _need(verify_files and source_repo == CANONICAL_REPO, 'ORIGINAL_PREFIX_PHYSICAL_SOURCE')
    expected, original, qualified = _ta_m5_original_cuda_prefix_expected()
    value = _identity(value, set(expected) | {'source_owners', 'trace_worker', 'source_bridge'},
                      TA_M5_ORIGINAL_CUDA_PREFIX_RECIPE_SCHEMA, 'recipe_sha256')
    _need(all(value[k] == v and type(value[k]) is type(v) for k, v in expected.items()),
          'ORIGINAL_PREFIX_RECIPE_SCOPE')
    _need(isinstance(source_commit, str) and campaign._COMMIT.fullmatch(source_commit)
          and subprocess.run(['git', 'merge-base', '--is-ancestor',
              TA_M5_ORIGINAL_CUDA_PREFIX_REFERENCE_SOURCE, source_commit],
              cwd=CANONICAL_REPO, capture_output=True).returncode == 0,
          'ORIGINAL_PREFIX_SOURCE_ANCESTRY')
    expected_bridge = _ta_m5_original_cuda_prefix_sources(
        value['source_owners'], original, qualified, value['trace_worker'])
    _need(value['source_bridge'] == expected_bridge, 'ORIGINAL_PREFIX_SOURCE_BRIDGE')
    return value


def require_ta_m5_original_cuda_prefix_gate(gate, recipe):
    expected = {
        'decision': 'ADMITTED', 'status': 'ADMITTED', 'ta_m5_original_cuda_prefix_authorized': True,
        'fold_id': 1, 'selected_fit_rows': 4096,
        'model_constructions_authorized': 1, 'optimizer_constructions_authorized': 1,
        'model_forwards_authorized': 256, 'backward_calls_authorized': 256,
        'optimizer_steps_authorized': 256, 'training_enabled': True, 'native_training_enabled': True,
        'genuine_updates_authorized': True, 'genuine_backward_authorized': True,
        'genuine_predictor_fit_authorized': False, 'canonical_resume_authority': False,
        'candidate_selection': False, 'test_data_used': False,
    }
    _need(isinstance(gate, Mapping) and all(gate.get(k) == v and type(gate[k]) is type(v)
          for k, v in expected.items()) and recipe['run_scope'] == TA_M5_ORIGINAL_CUDA_PREFIX_SCOPE
          and recipe['admission_stage'] == TA_M5_ORIGINAL_CUDA_PREFIX_STAGE,
          'ORIGINAL_PREFIX_POLICY_GATE')


def require_ta_m5_original_cuda_prefix_cursor(value, *, expected_recipe, verify_files=True, batch_size=16):
    fields = {'recipe', 'outcome', 'optimizer_updates', 'parent_cursor', 'parent_order_sha256',
              'full_state', 'result', 'test_data_used', 'canonical_resume_authority', 'schema_version'}
    cursor = _identity(value, fields, TA_M5_ORIGINAL_CUDA_PREFIX_CURSOR_SCHEMA, 'cursor_sha256')
    _need(batch_size == 16 and expected_recipe.get('path') == str(TA_M5_ORIGINAL_CUDA_PREFIX_RECIPE_PATH)
          and cursor['recipe'] == expected_recipe and cursor['outcome'] in ('COMPLETE', 'FAILED')
          and type(cursor['optimizer_updates']) is int and 0 <= cursor['optimizer_updates'] <= 256
          and type(cursor['parent_cursor']) is int and cursor['parent_cursor'] == 16 * cursor['optimizer_updates']
          and cursor['parent_order_sha256'] == TA_M5_EQUAL_LOSS_ORDER_SHA256
          and cursor['test_data_used'] is False and cursor['canonical_resume_authority'] is False,
          'ORIGINAL_PREFIX_CURSOR_SCOPE')
    complete = cursor['outcome'] == 'COMPLETE'
    target = TA_M5_ORIGINAL_CUDA_PREFIX_REFS['original_p4096_checkpoint']
    _need(cursor['full_state'] == (target if complete else None)
          and (not complete or cursor['optimizer_updates'] == 256),
          'ORIGINAL_PREFIX_CURSOR_COMPLETE')
    _need(cursor['result'].get('path') == str(TA_M5_ORIGINAL_CUDA_PREFIX_ROOT / 'RESULT.json'),
          'ORIGINAL_PREFIX_RESULT_PATH')
    campaign.require_binding(cursor['result'], label='original prefix result', verify_file=verify_files)
    if complete:
        campaign.require_binding(target, label='original prefix reference checkpoint', verify_file=verify_files)
    if verify_files:
        report = _read(cursor['result'], 'original prefix completed result')
        _need(report.get('schema_version') == TA_M5_ORIGINAL_CUDA_PREFIX_RESULT_SCHEMA
              and report.get('recipe') == expected_recipe
              and report.get('parent_cursor') == cursor['parent_cursor']
              and report.get('physical_optimizer_updates') == cursor['optimizer_updates']
              and report.get('parent_order_sha256') == cursor['parent_order_sha256']
              and report.get('final_state') == cursor['full_state']
              and report.get('canonical_resume_authority') is False
              and report.get('checkpoint_writes') == 0 and report.get('test_data_used') is False,
              'ORIGINAL_PREFIX_CURSOR_RESULT')
        if complete:
            counts = report.get('counts', {})
            exact = {'model_constructions': 1, 'optimizer_constructions': 1,
                     'forwards': 256, 'backwards': 256, 'optimizer_step_calls': 256,
                     'optimizer_updates': 256, 'checkpoint_loads': 2,
                     'strict_full_state_restore_calls': 1, 'checkpoint_writes': 0}
            _need(report.get('decision') == 'PASS_TA_M5_ORIGINAL_CUDA_PREFIX_TRACE'
                  and report.get('first_step_exact') is True and report.get('final_full_state_exact') is True
                  and all(type(counts.get(k)) is int and counts[k] == v for k, v in exact.items()),
                  'ORIGINAL_PREFIX_CURSOR_PARITY')
    return cursor


# One fixed-mean coordinate intervention; all legacy plans retain their scopes.
TA_M5_CENTERED_HEAD_COORDINATES_STAGE = 'market_state_ta_m5_centered_head_coordinates_20261110_001'
TA_M5_CENTERED_HEAD_COORDINATES_SCOPE = 'market_state_ta_m5_centered_head_coordinates_v1'
TA_M5_CENTERED_HEAD_COORDINATES_ROOT = Path('/home/andre2/GX1_RUNS') / TA_M5_CENTERED_HEAD_COORDINATES_STAGE.upper()
TA_M5_CENTERED_HEAD_COORDINATES_RUNTIME_ROOT = TA_M5_CENTERED_HEAD_COORDINATES_ROOT / 'RUNTIME'
TA_M5_CENTERED_HEAD_COORDINATES_RECIPE_PATH = TA_M5_CENTERED_HEAD_COORDINATES_ROOT / 'RECIPE.json'
TA_M5_CENTERED_HEAD_COORDINATES_RECIPE_SCHEMA = 'gx1_market_state_ta_m5_centered_head_coordinates_recipe_v1'
TA_M5_CENTERED_HEAD_COORDINATES_RESULT_SCHEMA = 'gx1_market_state_ta_m5_centered_head_coordinates_result_v1'
TA_M5_CENTERED_HEAD_COORDINATES_CURSOR_SCHEMA = 'gx1_market_state_ta_m5_centered_head_coordinates_cursor_v1'
TA_M5_CENTERED_HEAD_COORDINATES_WORKER = 'gx1/audit/run_market_state_ta_m5_centered_head_coordinates_v1.py'
TA_M5_CENTERED_HEAD_COORDINATES_TRAIN_RANGES = [[0, 64], [14272, 14336], [29504, 29568]]
TA_M5_CENTERED_HEAD_COORDINATES_SCORE_RANGES = [[0, 64], [8352, 8416], [17680, 17744]]
TA_M5_CENTERED_HEAD_COORDINATES_REFS = {
    'original_prefix_recipe': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_ORIGINAL_CUDA_PREFIX_20261110_001/RECIPE.json', 'sha256': '696ec0baa96a8d719cf1de8fadcf44f36d5ce3da985637e03dd1c06a77088f69'},
    'original_prefix_result': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_ORIGINAL_CUDA_PREFIX_20261110_001/RESULT.json', 'sha256': 'c82c5723ac47af260234eca3ee6a08f62ef0358861c3d39fc19f31e8a7243483'},
    'original_prefix_terminal': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_ORIGINAL_CUDA_PREFIX_20261110_001/TERMINAL.json', 'sha256': 'b0fbce5386dcbed5b694d9654e09ff72e3d25e67490ab6c46522fcca3b1c83f2'},
    'original_trace': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_ORIGINAL_CUDA_PREFIX_20261110_001/TRACE.npz', 'sha256': '486ed93b763539d606749bb4511f9c004c5d0cfe42cc645c0cf155f04e2118b8'},
    'original_trace_helper': {'path': '/home/andre2/src/GX1_CURRENT/gx1/audit/trace_market_state_ta_m5_original_cuda_prefix_v1.py', 'sha256': '593e3e17e235acd3e6ee09d63aa7478c9eacb15523765a6322af99e30a3294dc'},
    'geometry_plan': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_HEAD_CENTERING_GEOMETRY_20261110_002/PLAN.json', 'sha256': '20c6a23597f35abd0fdfe056a459d8318bb2f2a81eac3afa56f9c9604cad95c8'},
    'geometry_result': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_HEAD_CENTERING_GEOMETRY_20261110_002/RESULT.json', 'sha256': '2d24a83a27b486228c783c7803f5c88a7f6a0686f8b797e1768ac85db10e674e'},
    'geometry_terminal': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_HEAD_CENTERING_GEOMETRY_20261110_002/TERMINAL.json', 'sha256': 'd8a861bc4c7a251d5d6478344df9abdab593ab88328e7772729680f84d1b2b6d'},
    'frozen_means': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_HEAD_CENTERING_GEOMETRY_20261110_002/MEANS.npz', 'sha256': '1a20692cf7319f14ef216dd9d5828038bd1b20096e64e9eb7517e9e89e38a71f'},
    'short_panel_plan': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_SHORT_NATIVE_20261109_001/PLAN.json', 'sha256': '126c9fb6b1059576870145ec9fe5ce652dd7791f5a39964665e438d65e79b15e'},
    'short_panel_helper': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_SHORT_NATIVE_20261109_001/probe.py', 'sha256': '167e8e11675df0762a7dd91803243b82e610fe916b5542b34997bcd6332c77da'},
    'trajectory_plan': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_FULLPASS_TRAJECTORY_20261109_002/PLAN.json', 'sha256': 'a82e18305c38e492b0ff779d6a3531b9605d642474e28b75fb56ea2d5baa02fd'},
    'trajectory_result': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_FULLPASS_TRAJECTORY_20261109_002/RESULT.json', 'sha256': '02b686f0ae9a864892f9ac9500f1dd54110a654c708bc3d484ffbdac4a4df113'},
    'trajectory_review': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_FULLPASS_TRAJECTORY_20261109_002/REVIEW.json', 'sha256': 'cc7a642c965e8e388b6f1f89ec6b865b0c3c99d4ea3e54e3e3c8af6f7eca0992'},
    'panel_initial_measurement': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_FULLPASS_TRAJECTORY_20261109_001/INITIAL_TRAIN.json', 'sha256': '6ee23ca371003f58957bb0aaa2f374b6d7d8400090d19a1e9024f20b4d49c2c1'},
    'panel_initial_capture': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_FULLPASS_TRAJECTORY_20261109_001/INITIAL_M5_READOUT.npz', 'sha256': '821a93d52269dec040073069842254f34d32a3725838952e929711820e381904'},
    'panel_p4096_measurement': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_FULLPASS_TRAJECTORY_20261109_002/P004096_TRAIN.json', 'sha256': '025344e9897a5a7f59453a0ccf3df4f0d09cac2e8a7a5a0493e47a727349a031'},
    'panel_p4096_capture': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_FULLPASS_TRAJECTORY_20261109_002/P004096_M5_READOUT.npz', 'sha256': 'a888aae1e973a067f58c549b541e46a53690e855e4ba61c046eb5cf1f1253943'},
    'endpoint_priors': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_DEDICATED_M5_SOURCE_20261109_001/ENDPOINT_PRIORS.json', 'sha256': '9f2cc89f5abc886fa4a89a1c3b0d0f9c39f877d8ab1374af651aee678c7a425c'},
}

def read_ta_m5_centered_head_coordinates_origin(launch):
    refs = launch['input_refs']
    extra = TA_M5_CENTERED_HEAD_COORDINATES_REFS
    _need({key: refs.get(key) for key in extra} == extra, 'CENTERED_FIXED_REFS')
    original_refs = {key: value for key, value in refs.items() if key not in extra}
    origin = read_ta_m5_original_cuda_prefix_origin({'input_refs': original_refs})
    for key, ref in extra.items():
        campaign.require_binding(ref, label='centered ' + key, verify_file=True)
    replay = _read(refs['original_prefix_result'], 'qualified original prefix replay')
    terminal = _read(refs['original_prefix_terminal'], 'qualified original prefix terminal')
    _need(replay.get('decision') == 'PASS_TA_M5_ORIGINAL_CUDA_PREFIX_TRACE'
          and replay.get('recipe') == refs['original_prefix_recipe']
          and replay.get('first_step_exact') is True and replay.get('final_full_state_exact') is True
          and replay.get('physical_optimizer_updates') == 256 and replay.get('parent_cursor') == 4096
          and replay.get('skipped_batches') == 0 and replay.get('source_unchanged') is True
          and replay.get('trace') == refs['original_trace'] and replay.get('test_data_used') is False
          and terminal.get('status') == 'COMPLETE' and terminal.get('exit_code') == 0
          and terminal.get('result') == refs['original_prefix_result']
          and terminal.get('no_relaunch_under_this_plan') is True
          and terminal.get('test_data_used') is False, 'CENTERED_ORIGINAL_CONTROL')
    geometry = _read(refs['geometry_result'], 'qualified fixed TRAIN means')
    geometry_terminal = _read(refs['geometry_terminal'], 'qualified fixed TRAIN mean terminal')
    _need(geometry.get('status') == 'COMPLETE' and geometry.get('source_unchanged') is True
          and geometry.get('means') == refs['frozen_means']
          and geometry.get('mean_population') == 'all29574initialTRAINrows;neverlabelconditioned'
          and geometry.get('mean_dtype') == 'float64' and geometry.get('mean_estimates') == 2
          and geometry.get('later_arrays_consumed') is False
          and geometry.get('test_data_used') is False
          and geometry_terminal.get('status') == 'COMPLETE' and geometry_terminal.get('exit_code') == 0
          and geometry_terminal.get('result') == refs['geometry_result'],
          'CENTERED_MEAN_POPULATION')
    panels = _read(refs['short_panel_plan'], 'fixed panel definition only')
    _need(panels.get('fit_eval_ranges') == TA_M5_CENTERED_HEAD_COORDINATES_TRAIN_RANGES
          and panels.get('score_eval_ranges') == TA_M5_CENTERED_HEAD_COORDINATES_SCORE_RANGES
          and panels.get('worker') == refs['short_panel_helper'], 'CENTERED_PANEL_DEFINITION')
    trajectory = _read(refs['trajectory_result'], 'original fixed TRAIN panel baselines')
    _need(trajectory.get('status') == 'COMPLETE' and trajectory.get('source_unchanged') is True
          and trajectory.get('test_data_used') is False, 'CENTERED_TRAJECTORY_STATUS')
    for index, prefix, cursor in ((0, 'initial', 0), (1, 'p4096', 4096)):
        entry = trajectory['trajectory'][index]
        identity = entry['checkpoint_identity']
        checkpoint = refs['original_initial_checkpoint' if index == 0 else 'original_p4096_checkpoint']
        _need(entry['measurement'] == refs['panel_' + prefix + '_measurement']
              and entry['capture'] == refs['panel_' + prefix + '_capture']
              and identity['checkpoint'] == checkpoint and identity['cursor'] == cursor
              and identity['optimizer_updates'] == cursor // 16
              and identity['source_commit'] == TA_M5_ORIGINAL_CUDA_PREFIX_SOURCE,
              'CENTERED_ORIGINAL_PANEL_BINDING')
    return {**origin, 'geometry_result': geometry, 'trajectory_result': trajectory}


def _ta_m5_centered_head_coordinates_expected():
    value, original, qualified = _ta_m5_original_cuda_prefix_expected()
    value['input_refs'].update({key: dict(ref) for key, ref in TA_M5_CENTERED_HEAD_COORDINATES_REFS.items()})
    read_ta_m5_centered_head_coordinates_origin(value)
    value.pop('first_step_exact_required')
    value.pop('final_full_state_exact_required')
    value.update(
        schema_version=TA_M5_CENTERED_HEAD_COORDINATES_RECIPE_SCHEMA,
        runtime_root=str(TA_M5_CENTERED_HEAD_COORDINATES_RUNTIME_ROOT),
        admission_stage=TA_M5_CENTERED_HEAD_COORDINATES_STAGE,
        run_scope=TA_M5_CENTERED_HEAD_COORDINATES_SCOPE,
        execution_binding_path=str(TA_M5_CENTERED_HEAD_COORDINATES_RUNTIME_ROOT / 'EXECUTION_BINDING.json'),
        max_forwards=328, max_training_forwards=256, max_evaluation_forwards=72,
        max_full_state_restores=3, max_model_seconds=900,
        max_candidate_evidence_writes=1, max_candidate_evidence_bytes=MAX_CHECKPOINT_BYTES,
        initial_evaluation_forwards=24, later_evaluation_forwards=36,
        first_forward_raw_gradient_parity_required=True, control_p4096_restore_parity_required=True,
        function_equivalent_initial_required=True, evaluation_state_unchanged_required=True,
        fit_eval_ranges=TA_M5_CENTERED_HEAD_COORDINATES_TRAIN_RANGES,
        score_eval_ranges=TA_M5_CENTERED_HEAD_COORDINATES_SCORE_RANGES,
        evaluation_states=['original_initial', 'original_p4096', 'centered_candidate'],
        evaluate_both_panels_unconditionally=True,
        primary_baselines='fresh_CUDA_original_INITIAL_and_P4096_same_fixed_panels',
        historical_panel_parity={'raw_atol': 2e-5, 'probability_atol': 1e-6, 'score_atol': 1e-6},
        head_coordinate_intervention={
            'kind': 'fixed_initial_TRAIN_mean_head_coordinate_AdamW',
            'mean_reference': value['input_refs']['frozen_means'],
            'mean_population': 'all29574initialTRAINrows;neverlabelconditioned',
            'heads': ['state_model.m5_side_head', 'future_m5_side_head'],
            'centered_intercept': 'beta=b+w_dot_mu',
            'centered_weight_gradient': 'g_w_centered=g_w-g_b_times_mu',
            'optimizer_moments': 'centered_weight_and_beta_coordinates',
            'global_clipping': 'canonical_norm_after_coordinate_gradient_transform',
            'mean_updates': 0, 'loss_and_architecture_unchanged': True,
        })
    return value, original, qualified


def build_ta_m5_centered_head_coordinates_recipe(*, source_commit):
    value, original, qualified = _ta_m5_centered_head_coordinates_expected()
    owners = {name: {'path': str(CANONICAL_REPO / name),
                     'sha256': campaign.file_sha256(CANONICAL_REPO / name)}
              for name in original['source_owners']}
    worker = CANONICAL_REPO / TA_M5_CENTERED_HEAD_COORDINATES_WORKER
    value.update(source_owners=owners,
        trace_worker={'path': str(worker), 'sha256': campaign.file_sha256(worker)},
        source_bridge=_ta_m5_original_cuda_prefix_sources(
            owners, original, qualified, value['input_refs']['original_trace_helper']))
    value['recipe_sha256'] = campaign.canonical_sha256(value)
    return require_ta_m5_centered_head_coordinates_recipe(
        value, source_repo=CANONICAL_REPO, source_commit=source_commit)


def require_ta_m5_centered_head_coordinates_recipe(value, *, source_repo, source_commit, verify_files=True):
    import subprocess
    _need(verify_files and source_repo == CANONICAL_REPO, 'CENTERED_PHYSICAL_SOURCE')
    expected, original, qualified = _ta_m5_centered_head_coordinates_expected()
    value = _identity(value, set(expected) | {'source_owners', 'trace_worker', 'source_bridge'},
                      TA_M5_CENTERED_HEAD_COORDINATES_RECIPE_SCHEMA, 'recipe_sha256')
    _need(all(value[k] == v and type(value[k]) is type(v) for k, v in expected.items()),
          'CENTERED_RECIPE_SCOPE')
    _need(isinstance(source_commit, str) and campaign._COMMIT.fullmatch(source_commit)
          and subprocess.run(['git', 'merge-base', '--is-ancestor',
              TA_M5_ORIGINAL_CUDA_PREFIX_REFERENCE_SOURCE, source_commit],
              cwd=CANONICAL_REPO, capture_output=True).returncode == 0,
          'CENTERED_SOURCE_ANCESTRY')
    _need(value['trace_worker'].get('path') == str(CANONICAL_REPO / TA_M5_CENTERED_HEAD_COORDINATES_WORKER),
          'CENTERED_WORKER_PATH')
    campaign.require_binding(value['trace_worker'], label='centered coordinate worker', verify_file=True)
    expected_bridge = _ta_m5_original_cuda_prefix_sources(
        value['source_owners'], original, qualified, value['input_refs']['original_trace_helper'])
    _need(value['source_bridge'] == expected_bridge, 'CENTERED_SOURCE_BRIDGE')
    return value


def require_ta_m5_centered_head_coordinates_gate(gate, recipe):
    expected = {
        'decision': 'ADMITTED', 'status': 'ADMITTED', 'ta_m5_centered_head_coordinates_authorized': True,
        'fold_id': 1, 'selected_fit_rows': 4096,
        'model_constructions_authorized': 1, 'optimizer_constructions_authorized': 1,
        'model_forwards_authorized': 328, 'backward_calls_authorized': 256,
        'optimizer_steps_authorized': 256, 'training_enabled': True, 'native_training_enabled': True,
        'genuine_updates_authorized': True, 'genuine_backward_authorized': True,
        'genuine_predictor_fit_authorized': False, 'canonical_resume_authority': False,
        'candidate_selection': False, 'test_data_used': False,
        'evaluate_both_panels_unconditionally': True,
    }
    _need(isinstance(gate, Mapping) and all(gate.get(k) == v and type(gate[k]) is type(v)
          for k, v in expected.items()) and recipe['run_scope'] == TA_M5_CENTERED_HEAD_COORDINATES_SCOPE
          and recipe['admission_stage'] == TA_M5_CENTERED_HEAD_COORDINATES_STAGE,
          'CENTERED_POLICY_GATE')


def require_ta_m5_centered_head_coordinates_cursor(value, *, expected_recipe, verify_files=True, batch_size=16):
    fields = {'recipe', 'outcome', 'optimizer_updates', 'parent_cursor', 'parent_order_sha256',
              'full_state', 'result', 'test_data_used', 'canonical_resume_authority', 'schema_version'}
    cursor = _identity(value, fields, TA_M5_CENTERED_HEAD_COORDINATES_CURSOR_SCHEMA, 'cursor_sha256')
    _need(batch_size == 16 and expected_recipe.get('path') == str(TA_M5_CENTERED_HEAD_COORDINATES_RECIPE_PATH)
          and cursor['recipe'] == expected_recipe and cursor['outcome'] in ('COMPLETE', 'FAILED')
          and type(cursor['optimizer_updates']) is int and 0 <= cursor['optimizer_updates'] <= 256
          and type(cursor['parent_cursor']) is int and cursor['parent_cursor'] == 16 * cursor['optimizer_updates']
          and cursor['parent_order_sha256'] == TA_M5_EQUAL_LOSS_ORDER_SHA256
          and cursor['test_data_used'] is False and cursor['canonical_resume_authority'] is False
          and cursor['full_state'] is None, 'CENTERED_CURSOR_SCOPE')
    complete = cursor['outcome'] == 'COMPLETE'
    _need(not complete or cursor['optimizer_updates'] == 256, 'CENTERED_CURSOR_COMPLETE')
    _need(cursor['result'].get('path') == str(TA_M5_CENTERED_HEAD_COORDINATES_ROOT / 'RESULT.json'),
          'CENTERED_RESULT_PATH')
    campaign.require_binding(cursor['result'], label='centered result', verify_file=verify_files)
    if verify_files:
        report = _read(cursor['result'], 'centered result')
        _need(report.get('schema_version') == TA_M5_CENTERED_HEAD_COORDINATES_RESULT_SCHEMA
              and report.get('recipe') == expected_recipe
              and report.get('parent_cursor') == cursor['parent_cursor']
              and report.get('physical_optimizer_updates') == cursor['optimizer_updates']
              and report.get('parent_order_sha256') == cursor['parent_order_sha256']
              and report.get('final_state') is None and report.get('canonical_resume_authority') is False
              and report.get('checkpoint_writes') == 0 and report.get('test_data_used') is False,
              'CENTERED_CURSOR_RESULT')
        if complete:
            exact = {'model_constructions': 1, 'optimizer_constructions': 1,
                     'forwards': 328, 'training_forwards': 256, 'evaluation_forwards': 72,
                     'backwards': 256, 'optimizer_step_calls': 256, 'optimizer_updates': 256,
                     'checkpoint_loads': 2, 'strict_full_state_restore_calls': 3,
                     'checkpoint_writes': 0, 'candidate_evidence_writes': 1}
            _need(report.get('decision') == 'PASS_TA_M5_CENTERED_HEAD_COORDINATES_TECHNICAL'
                  and report.get('first_forward_raw_gradient_exact') is True
                  and report.get('control_p4096_restore_exact') is True
                  and report.get('function_equivalent_initial') is True
                  and report.get('evaluation_state_unchanged') is True
                  and report.get('all_panels_complete') is True and report.get('skipped_batches') == 0
                  and all(type(report.get('counts', {}).get(k)) is int
                          and report['counts'][k] == v for k, v in exact.items()),
                  'CENTERED_TECHNICAL_COMPLETION')
            evidence = report.get('candidate_evidence', {})
            _need(evidence.get('path') == str(TA_M5_CENTERED_HEAD_COORDINATES_ROOT / 'CANDIDATE_EVIDENCE.pt'),
                  'CENTERED_EVIDENCE_PATH')
            campaign.require_binding(evidence, label='private candidate evidence', verify_file=True)
    return cursor


# One paired intact-batch order comparison; no legacy cursor or resume semantics.
TA_M5_PAIRED_BATCH_ORDER_GUARD_OLD_SHA256 = '30e1f89dfcaeb9ea72c821031b9ca8b452020aa754d0ba23c1efdb68e3e1ba89'
TA_M5_PAIRED_BATCH_ORDER_GUARD_CURRENT_SHA256 = 'f6c2e2c8de55af14047ed2377c44736f7612edd71f8686b47ecc2771f1056ae7'
TA_M5_PAIRED_BATCH_ORDER_STAGE = 'market_state_ta_m5_paired_batch_order_20261110_002'
TA_M5_PAIRED_BATCH_ORDER_SCOPE = 'market_state_ta_m5_paired_batch_order_v1'
TA_M5_PAIRED_BATCH_ORDER_ROOT = Path('/home/andre2/GX1_RUNS') / TA_M5_PAIRED_BATCH_ORDER_STAGE.upper()
TA_M5_PAIRED_BATCH_ORDER_RUNTIME_ROOT = TA_M5_PAIRED_BATCH_ORDER_ROOT / 'RUNTIME'
TA_M5_PAIRED_BATCH_ORDER_RECIPE_PATH = TA_M5_PAIRED_BATCH_ORDER_ROOT / 'RECIPE.json'
TA_M5_PAIRED_BATCH_ORDER_RECIPE_SCHEMA = 'gx1_market_state_ta_m5_paired_batch_order_recipe_v1'
TA_M5_PAIRED_BATCH_ORDER_RESULT_SCHEMA = 'gx1_market_state_ta_m5_paired_batch_order_result_v1'
TA_M5_PAIRED_BATCH_ORDER_CURSOR_SCHEMA = 'gx1_market_state_ta_m5_paired_batch_order_cursor_v1'
TA_M5_PAIRED_BATCH_ORDER_WORKER = 'gx1/audit/run_market_state_ta_m5_paired_batch_order_v1.py'
TA_M5_PAIRED_BATCH_ORDER_ARMS = ('CONTROL', 'INTERLEAVED')
TA_M5_PAIRED_BATCH_ORDER_SCORE_RANGES = [[0, 64], [8352, 8416], [17680, 17744]]
TA_M5_PAIRED_BATCH_ORDER_BATCH_INDICES = {
    'CONTROL': [(i * 1847) // 255 for i in range(256)],
    'INTERLEAVED': [((r * 16 + c) * 1847) // 255 for c in range(16) for r in range(16)],
}
TA_M5_PAIRED_BATCH_ORDER_COMPLETE_COUNTS = {
    'model_constructions': 1, 'optimizer_constructions': 1, 'checkpoint_loads': 1,
    'strict_full_state_restore_calls': 2, 'forwards': 1048, 'training_forwards': 512,
    'evaluation_forwards': 536, 'backwards': 512, 'optimizer_step_calls': 512,
    'optimizer_updates': 512, 'checkpoint_writes': 0, 'candidate_evidence_writes': 2,
}
_TA_PAIRED_OBJECTIVE = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_SF_OBJECTIVE_WEIGHT_AUDIT_20261109_001')
TA_M5_PAIRED_BATCH_ORDER_REFS = {
    key: dict(TA_M5_CENTERED_HEAD_COORDINATES_REFS[key]) for key in (
        'original_prefix_recipe', 'original_prefix_result', 'original_prefix_terminal',
        'original_trace', 'original_trace_helper', 'endpoint_priors')
}
TA_M5_PAIRED_BATCH_ORDER_REFS.update({
    'objective_audit_plan': {'path': str(_TA_PAIRED_OBJECTIVE / 'PLAN.json'), 'sha256': '199f7c1f0d702cd47762d0db27ccf66bf2bf510304ba12ace2686afb0fe4597b'},
    'objective_audit_result': {'path': str(_TA_PAIRED_OBJECTIVE / 'RESULT.json'), 'sha256': '59511a3f8cddc0afbc4e6dceef609419bf3a6b6bdf4463fff080fbc703879707'},
    'objective_audit_review': {'path': str(_TA_PAIRED_OBJECTIVE / 'REVIEW.json'), 'sha256': '48586cc5feb9d9edfc2dfad3e23078747d6b15f3d3b45360cd443e09005b58f1'},
    'initial_fit_cache': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_DIRECTION_FULL_CACHE_20261108_001/FIT_CACHE.npz', 'sha256': 'e8a562314d81b5821931df0ff78ff563b0968015eac0eaa141e909c80d6fc22f'},
    'imported_predictions': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_FINAL_READOUT_20261109_001/PRIVATE_READOUTS_AND_PREDICTIONS.npz', 'sha256': '317deaca134460b73e8ac200dff2741483c7729aa13d6d6870b4aeecf136846b'},
    'centered_recipe': {'path': str(TA_M5_CENTERED_HEAD_COORDINATES_RECIPE_PATH), 'sha256': '5fd06f688bfc2e924a1d7e59755b6d8be6a4347b2d218e005edcdaa656a18a5d'},
    'centered_result': {'path': str(TA_M5_CENTERED_HEAD_COORDINATES_ROOT / 'RESULT.json'), 'sha256': 'e2053949378c915de02dd916e1db7f3eac26a24b69e7bde2d27dc80b2ffdfadd'},
    'centered_terminal': {'path': str(TA_M5_CENTERED_HEAD_COORDINATES_ROOT / 'TERMINAL.json'), 'sha256': 'c56c8ff988b584cb80a4717980c320c2767587f92dc63f0aecd604570df8a150'},
    'centered_score_review': {'path': str(TA_M5_CENTERED_HEAD_COORDINATES_ROOT / 'CACHED_SCORE_REVIEW.json'), 'sha256': 'b21848b3ab4ea236e106bbc53bdf8c7e658346c766d46a8f4a0b71c9dbe66b5c'},
    'initial_later_measurement': {'path': str(TA_M5_CENTERED_HEAD_COORDINATES_ROOT / 'INITIAL_LATER.json'), 'sha256': 'e1f5a1694bf571a55bc8daab8570abc60597f3d58e7e9421d4ab283f4310b6d9'},
    'initial_later_capture': {'path': str(TA_M5_CENTERED_HEAD_COORDINATES_ROOT / 'INITIAL_LATER_READOUT.npz'), 'sha256': 'ad895e24f4f321b47c7077243b57e769a84eea32b043868271714604aba36b5b'},
})


def read_ta_m5_paired_batch_order_origin(launch):
    refs = launch['input_refs']
    extra = TA_M5_PAIRED_BATCH_ORDER_REFS
    _need({key: refs.get(key) for key in extra} == extra, 'PAIRED_ORDER_FIXED_REFS')
    origin = read_ta_m5_original_cuda_prefix_origin(
        {'input_refs': {key: ref for key, ref in refs.items() if key not in extra}})
    for key, ref in extra.items():
        campaign.require_binding(ref, label='paired order ' + key, verify_file=True)
    replay = _read(refs['original_prefix_result'], 'paired original replay')
    terminal = _read(refs['original_prefix_terminal'], 'paired original replay terminal')
    _need(replay.get('decision') == 'PASS_TA_M5_ORIGINAL_CUDA_PREFIX_TRACE'
          and replay.get('recipe') == refs['original_prefix_recipe']
          and replay.get('first_step_exact') is True and replay.get('final_full_state_exact') is True
          and replay.get('physical_optimizer_updates') == 256 and replay.get('parent_cursor') == 4096
          and replay.get('skipped_batches') == 0 and replay.get('source_unchanged') is True
          and replay.get('trace') == refs['original_trace'] and replay.get('test_data_used') is False
          and terminal.get('status') == 'COMPLETE' and terminal.get('exit_code') == 0
          and terminal.get('result') == refs['original_prefix_result']
          and terminal.get('no_relaunch_under_this_plan') is True
          and terminal.get('test_data_used') is False, 'PAIRED_ORDER_ORIGINAL_REPLAY')
    audit = _read(refs['objective_audit_result'], 'paired original TRAIN baseline proof')
    review = _read(refs['objective_audit_review'], 'paired TRAIN baseline independent review')
    _need(audit.get('status') == 'COMPLETE' and audit.get('source_unchanged') is True
          and audit.get('test_data_used') is False and audit.get('later_arrays_consumed') is False
          and audit.get('plan') == review.get('plan') == refs['objective_audit_plan']
          and review.get('status') == 'PASS_INDEPENDENT_CACHED_OBJECTIVE_REVIEW'
          and review.get('result') == refs['objective_audit_result']
          and review.get('test_data_used') is False and review.get('later_arrays_consumed') is False
          and review.get('rows') == 29574 and review.get('batches') == 1849
          and review.get('last_batch_rows') == 6
          and audit.get('baseline') == review.get('baseline')
          and audit['baseline'].get('initial_model_sha256') == TA_M5_EQUAL_LOSS_MODEL_SHA256
          and audit['baseline'].get('short_INITIAL0_used') is False
          and audit.get('inputs', {}).get('old_fit_cache') == refs['initial_fit_cache']
          and audit['inputs'].get('readout_outputs') == refs['imported_predictions']
          and audit['inputs'].get('fullfit_recipe') == refs['original_recipe']
          and audit['inputs'].get('initial_measurement') == refs['initial_measurement'],
          'PAIRED_ORDER_TRAIN_BASELINE')
    profile = audit.get('profile', {})
    _need(profile.get('fit_rows') == 29574 and profile.get('fit_batches') == 1849
          and profile.get('batch_size') == 16 and profile.get('pass_index') == 0
          and profile.get('fit_cache_sha256') == refs['initial_fit_cache']['sha256']
          and profile.get('later_data_used_for_normalization') is False,
          'PAIRED_ORDER_TRAIN_PROFILE')
    centered = _read(refs['centered_result'], 'paired original later CUDA baseline proof')
    centered_terminal = _read(refs['centered_terminal'], 'paired CUDA baseline terminal')
    centered_review = _read(refs['centered_score_review'], 'paired CUDA baseline score review')
    later = _read(refs['initial_later_measurement'], 'paired original later measurement')
    _need(centered.get('decision') == 'PASS_TA_M5_CENTERED_HEAD_COORDINATES_TECHNICAL'
          and centered.get('recipe') == refs['centered_recipe']
          and centered.get('source_unchanged') is True and centered.get('test_data_used') is False
          and centered.get('evaluation_state_unchanged') is True
          and centered.get('measurement_refs', {}).get('INITIAL_LATER') == refs['initial_later_measurement']
          and centered.get('captures', {}).get('INITIAL_LATER') == refs['initial_later_capture']
          and centered.get('restore_proofs', {}).get('INITIAL_BASELINE', {}).get('all_numerical_state_exact') is True
          and centered_terminal.get('status') == 'COMPLETE' and centered_terminal.get('exit_code') == 0
          and centered_terminal.get('result') == refs['centered_result']
          and centered_terminal.get('no_relaunch_under_this_plan') is True
          and centered_review.get('status') == 'PASS_INDEPENDENT_CACHED_SCORES_NEGATIVE_COMPARISON'
          and centered_review.get('result') == refs['centered_result']
          and centered_review.get('test_data_used') is False,
          'PAIRED_ORDER_LATER_BASELINE_PROOF')
    positions = [i for start, stop in TA_M5_PAIRED_BATCH_ORDER_SCORE_RANGES for i in range(start, stop)]
    _need(later.get('evaluated_checkpoint') == refs['original_initial_checkpoint']
          and later.get('evaluated_model_state_sha256') == TA_M5_EQUAL_LOSS_MODEL_SHA256
          and later.get('capture') == refs['initial_later_capture']
          and later.get('fixed_subset_positions') == positions and later.get('optimizer_updates') == 0
          and later.get('fold_id') == 1 and later.get('test_data_used') is False,
          'PAIRED_ORDER_LATER_BASELINE_IDENTITY')
    return {**origin, 'input_refs': refs, 'original_prefix_result': replay,
            'objective_audit_result': audit, 'objective_audit_review': review,
            'initial_later_measurement': later, 'centered_result': centered}


def _ta_m5_paired_batch_order_expected():
    value, original, qualified = _ta_m5_original_cuda_prefix_expected()
    value['input_refs'].update({key: dict(ref) for key, ref in TA_M5_PAIRED_BATCH_ORDER_REFS.items()})
    read_ta_m5_paired_batch_order_origin(value)
    value.pop('first_step_exact_required')
    value.pop('final_full_state_exact_required')
    value.update(
        schema_version=TA_M5_PAIRED_BATCH_ORDER_RECIPE_SCHEMA,
        runtime_root=str(TA_M5_PAIRED_BATCH_ORDER_RUNTIME_ROOT),
        admission_stage=TA_M5_PAIRED_BATCH_ORDER_STAGE, run_scope=TA_M5_PAIRED_BATCH_ORDER_SCOPE,
        execution_binding_path=str(TA_M5_PAIRED_BATCH_ORDER_RUNTIME_ROOT / 'EXECUTION_BINDING.json'),
        parent_selection='256_intact_original_B16_indices_floor_i_times_1847_over_255',
        arm_batch_indices={arm: list(indices) for arm, indices in TA_M5_PAIRED_BATCH_ORDER_BATCH_INDICES.items()},
        arm_batch_order_sha256={arm: campaign.canonical_sha256(indices)
                               for arm, indices in TA_M5_PAIRED_BATCH_ORDER_BATCH_INDICES.items()},
        arms=list(TA_M5_PAIRED_BATCH_ORDER_ARMS), updates_per_arm=256, selected_fit_rows=4096,
        original_complete_batches=1848, excluded_partial_batch_rows=6,
        max_private_updates=512, max_backwards=512, max_forwards=1048,
        max_training_forwards=512, max_evaluation_forwards=536,
        max_checkpoint_loads=1, max_full_state_restores=2,
        max_model_seconds=2400, max_invocation_seconds=3000,
        max_candidate_evidence_writes=2, max_candidate_evidence_bytes=MAX_CHECKPOINT_BYTES,
        initial_evaluation_forwards=0, later_evaluation_forwards=24,
        selected_TRAIN_evaluation_forwards=512,
        score_eval_ranges=TA_M5_PAIRED_BATCH_ORDER_SCORE_RANGES,
        fit_evaluation_query_domain='original_pass0_sampled_S_and_native_F_same_intact_batches',
        later_evaluation_query_domain='native_endpoints_all_five_timeframes',
        primary_baselines='qualified_cached_original_INITIAL_same4096_and_CUDA_later192',
        both_arms_same_original_INITIAL_required=True, same_training_queries_required=True,
        same_selected_rows_required=True, rng_stream_parity_required=True,
        evaluation_state_unchanged_required=True, evaluate_both_panels_unconditionally=True,
        ordinary_optimizer_unchanged=True, per_update_pareto_requirement=False,
        statistical_significance_claimed=False, canonical_resume_authority=False,
        candidate_selection=False, predictive_value_claimed=False,
        baseline_parity={'raw_atol': 2e-5, 'probability_atol': 1e-6, 'score_atol': 1e-6})
    return value, original, qualified


def build_ta_m5_paired_batch_order_recipe(*, source_commit):
    value, original, qualified = _ta_m5_paired_batch_order_expected()
    owners = {name: {'path': str(CANONICAL_REPO / name),
                     'sha256': campaign.file_sha256(CANONICAL_REPO / name)}
              for name in original['source_owners']}
    worker = CANONICAL_REPO / TA_M5_PAIRED_BATCH_ORDER_WORKER
    value.update(source_owners=owners,
        trace_worker={'path': str(worker), 'sha256': campaign.file_sha256(worker)},
        source_bridge=_ta_m5_original_cuda_prefix_sources(
            owners, original, qualified, value['input_refs']['original_trace_helper'], paired_order_guard=True))
    value['recipe_sha256'] = campaign.canonical_sha256(value)
    return require_ta_m5_paired_batch_order_recipe(value, source_repo=CANONICAL_REPO,
                                                 source_commit=source_commit)


def require_ta_m5_paired_batch_order_recipe(value, *, source_repo, source_commit, verify_files=True):
    import subprocess
    _need(verify_files and source_repo == CANONICAL_REPO, 'PAIRED_ORDER_PHYSICAL_SOURCE')
    expected, original, qualified = _ta_m5_paired_batch_order_expected()
    value = _identity(value, set(expected) | {'source_owners', 'trace_worker', 'source_bridge'},
                      TA_M5_PAIRED_BATCH_ORDER_RECIPE_SCHEMA, 'recipe_sha256')
    _need(campaign.canonical_sha256({key: value[key] for key in expected})
          == campaign.canonical_sha256(expected), 'PAIRED_ORDER_RECIPE_SCOPE')
    _need(isinstance(source_commit, str) and campaign._COMMIT.fullmatch(source_commit)
          and subprocess.run(['git', 'merge-base', '--is-ancestor',
              TA_M5_ORIGINAL_CUDA_PREFIX_REFERENCE_SOURCE, source_commit],
              cwd=CANONICAL_REPO, capture_output=True).returncode == 0,
          'PAIRED_ORDER_SOURCE_ANCESTRY')
    _need(value['trace_worker'].get('path') == str(CANONICAL_REPO / TA_M5_PAIRED_BATCH_ORDER_WORKER),
          'PAIRED_ORDER_WORKER_PATH')
    campaign.require_binding(value['trace_worker'], label='paired batch order worker', verify_file=True)
    expected_bridge = _ta_m5_original_cuda_prefix_sources(
        value['source_owners'], original, qualified, value['input_refs']['original_trace_helper'], paired_order_guard=True)
    _need(value['source_bridge'] == expected_bridge, 'PAIRED_ORDER_SOURCE_BRIDGE')
    return value


def require_ta_m5_paired_batch_order_gate(gate, recipe):
    expected = {
        'decision': 'ADMITTED', 'status': 'ADMITTED', 'ta_m5_paired_batch_order_authorized': True,
        'fold_id': 1, 'selected_fit_rows': 4096, 'arms': list(TA_M5_PAIRED_BATCH_ORDER_ARMS),
        'model_constructions_authorized': 1, 'optimizer_constructions_authorized': 1,
        'model_forwards_authorized': 1048, 'backward_calls_authorized': 512,
        'optimizer_steps_authorized': 512, 'training_enabled': True, 'native_training_enabled': True,
        'genuine_updates_authorized': True, 'genuine_backward_authorized': True,
        'genuine_predictor_fit_authorized': False, 'canonical_resume_authority': False,
        'candidate_selection': False, 'test_data_used': False,
        'evaluate_both_panels_unconditionally': True, 'per_update_pareto_requirement': False,
    }
    _need(isinstance(gate, Mapping) and all(gate.get(k) == v and type(gate[k]) is type(v)
          for k, v in expected.items()) and recipe['run_scope'] == TA_M5_PAIRED_BATCH_ORDER_SCOPE
          and recipe['admission_stage'] == TA_M5_PAIRED_BATCH_ORDER_STAGE,
          'PAIRED_ORDER_POLICY_GATE')


def build_ta_m5_paired_batch_order_cursor(*, recipe_ref, outcome, optimizer_updates,
                                         arm_progress, result, batch_size=16):
    value = dict(schema_version=TA_M5_PAIRED_BATCH_ORDER_CURSOR_SCHEMA, recipe=recipe_ref,
                 outcome=outcome, optimizer_updates=optimizer_updates, parent_cursor=None,
                 parent_order_sha256=None, arm_progress=arm_progress, full_state=None,
                 result=result, test_data_used=False, canonical_resume_authority=False)
    value['cursor_sha256'] = campaign.canonical_sha256(value)
    return require_ta_m5_paired_batch_order_cursor(
        value, expected_recipe=recipe_ref, verify_files=False, batch_size=batch_size)


def require_ta_m5_paired_batch_order_cursor(value, *, expected_recipe, verify_files=True, batch_size=16):
    fields = {'recipe', 'outcome', 'optimizer_updates', 'parent_cursor', 'parent_order_sha256',
              'arm_progress', 'full_state', 'result', 'test_data_used', 'canonical_resume_authority',
              'schema_version'}
    cursor = _identity(value, fields, TA_M5_PAIRED_BATCH_ORDER_CURSOR_SCHEMA, 'cursor_sha256')
    _need(batch_size == 16 and expected_recipe.get('path') == str(TA_M5_PAIRED_BATCH_ORDER_RECIPE_PATH)
          and cursor['recipe'] == expected_recipe and cursor['outcome'] in ('COMPLETE', 'FAILED')
          and type(cursor['optimizer_updates']) is int and 0 <= cursor['optimizer_updates'] <= 512
          and cursor['parent_cursor'] is None and cursor['parent_order_sha256'] is None
          and cursor['full_state'] is None and cursor['test_data_used'] is False
          and cursor['canonical_resume_authority'] is False, 'PAIRED_ORDER_CURSOR_SCOPE')
    arms = cursor['arm_progress']
    _need(isinstance(arms, Mapping) and set(arms) == set(TA_M5_PAIRED_BATCH_ORDER_ARMS),
          'PAIRED_ORDER_CURSOR_ARMS')
    for arm in TA_M5_PAIRED_BATCH_ORDER_ARMS:
        progress = arms[arm]
        _need(isinstance(progress, Mapping) and set(progress) == {
            'optimizer_updates', 'parent_cursor', 'parent_order_sha256', 'batch_order_sha256'},
            'PAIRED_ORDER_CURSOR_ARM_FIELDS')
        updates = progress['optimizer_updates']
        _need(type(updates) is int and 0 <= updates <= 256
              and type(progress['parent_cursor']) is int and progress['parent_cursor'] == 16 * updates
              and progress['batch_order_sha256'] == campaign.canonical_sha256(
                  TA_M5_PAIRED_BATCH_ORDER_BATCH_INDICES[arm]), 'PAIRED_ORDER_CURSOR_ARM_PROGRESS')
        if progress['parent_order_sha256'] is None:
            _need(cursor['outcome'] == 'FAILED' and updates == 0, 'PAIRED_ORDER_CURSOR_MISSING_ORDER')
        else:
            campaign._sha(progress['parent_order_sha256'], 'paired ' + arm + ' parent order')
    _need(cursor['optimizer_updates'] == sum(arm['optimizer_updates'] for arm in arms.values())
          and (arms['INTERLEAVED']['optimizer_updates'] == 0 or arms['CONTROL']['optimizer_updates'] == 256),
          'PAIRED_ORDER_CURSOR_PHYSICAL_COUNT')
    complete = cursor['outcome'] == 'COMPLETE'
    _need(not complete or (cursor['optimizer_updates'] == 512
          and arms['CONTROL']['parent_order_sha256'] != arms['INTERLEAVED']['parent_order_sha256']),
          'PAIRED_ORDER_CURSOR_COMPLETE')
    _need(cursor['result'].get('path') == str(TA_M5_PAIRED_BATCH_ORDER_ROOT / 'RESULT.json'),
          'PAIRED_ORDER_RESULT_PATH')
    campaign.require_binding(cursor['result'], label='paired order result', verify_file=verify_files)
    if verify_files:
        report = _read(cursor['result'], 'paired order result')
        _need(report.get('schema_version') == TA_M5_PAIRED_BATCH_ORDER_RESULT_SCHEMA
              and report.get('recipe') == expected_recipe and report.get('outcome') == cursor['outcome']
              and report.get('physical_optimizer_updates') == report.get('optimizer_updates') == cursor['optimizer_updates']
              and report.get('arm_progress') == arms
              and report.get('parent_cursor') is None and report.get('parent_order_sha256') is None
              and report.get('full_state') is None and report.get('final_state') is None
              and report.get('canonical_resume_authority') is False
              and report.get('checkpoint_writes') == 0 and report.get('test_data_used') is False,
              'PAIRED_ORDER_CURSOR_RESULT')
        if complete:
            _need(report.get('decision') == 'PASS_TA_M5_PAIRED_BATCH_ORDER_TECHNICAL'
                  and report.get('source_unchanged') is True and report.get('skipped_batches') == 0
                  and all(report.get(key) is True for key in (
                      'same_selected_rows', 'same_initial_state', 'same_training_queries',
                      'rng_stream_parity', 'evaluation_state_unchanged', 'all_panels_complete'))
                  and all(type(report.get('counts', {}).get(key)) is int
                          and report['counts'][key] == count
                          for key, count in TA_M5_PAIRED_BATCH_ORDER_COMPLETE_COUNTS.items()),
                  'PAIRED_ORDER_TECHNICAL_COMPLETION')
            evidence = report.get('candidate_evidence', {})
            _need(isinstance(evidence, Mapping) and set(evidence) == set(TA_M5_PAIRED_BATCH_ORDER_ARMS),
                  'PAIRED_ORDER_EVIDENCE_ARMS')
            for arm, ref in evidence.items():
                _need(ref.get('path') == str(TA_M5_PAIRED_BATCH_ORDER_ROOT / (arm + '_EVIDENCE.pt')),
                      'PAIRED_ORDER_EVIDENCE_PATH')
                campaign.require_binding(ref, label='paired ' + arm + ' private evidence', verify_file=True)
    return cursor
# Two fixed additional passes over the already-qualified CONTROL population.
TA_M5_REPEATED_PASS_STAGE = 'market_state_ta_m5_repeated_pass_20261110_001'
TA_M5_REPEATED_PASS_SCOPE = 'market_state_ta_m5_repeated_pass_v1'
TA_M5_REPEATED_PASS_ROOT = Path('/home/andre2/GX1_RUNS') / TA_M5_REPEATED_PASS_STAGE.upper()
TA_M5_REPEATED_PASS_RUNTIME_ROOT = TA_M5_REPEATED_PASS_ROOT / 'RUNTIME'
TA_M5_REPEATED_PASS_RECIPE_PATH = TA_M5_REPEATED_PASS_ROOT / 'RECIPE.json'
TA_M5_REPEATED_PASS_RECIPE_SCHEMA = 'gx1_market_state_ta_m5_repeated_pass_recipe_v1'
TA_M5_REPEATED_PASS_RESULT_SCHEMA = 'gx1_market_state_ta_m5_repeated_pass_result_v1'
TA_M5_REPEATED_PASS_CURSOR_SCHEMA = 'gx1_market_state_ta_m5_repeated_pass_cursor_v1'
TA_M5_REPEATED_PASS_WORKER = 'gx1/audit/run_market_state_ta_m5_repeated_pass_v1.py'
TA_M5_REPEATED_PASS_COMPLETE_COUNTS = {
    'model_constructions': 1, 'optimizer_constructions': 1, 'checkpoint_loads': 1,
    'strict_full_state_restore_calls': 1, 'forwards': 1050, 'training_forwards': 512,
    'evaluation_forwards': 538, 'backwards': 512, 'optimizer_step_calls': 512,
    'optimizer_updates': 512, 'checkpoint_writes': 0, 'candidate_evidence_writes': 1,
}
TA_M5_REPEATED_PASS_REFS = {
    'paired_recipe': {'path': str(TA_M5_PAIRED_BATCH_ORDER_RECIPE_PATH), 'sha256': 'e503cf1f17a58391b7286b51ee5423f90fb47fe45d06a860f47553b9f1d8d667'},
    'paired_result': {'path': str(TA_M5_PAIRED_BATCH_ORDER_ROOT / 'RESULT.json'), 'sha256': 'b11a1932c924b22555bf9e848c8dea7507291c5584d0e7626de1a0f660840438'},
    'paired_terminal': {'path': str(TA_M5_PAIRED_BATCH_ORDER_ROOT / 'TERMINAL.json'), 'sha256': '6df60cb3966721d71e0d72f654c5268d0ddb2a15215db1c63c8a9f36ac3d2b3a'},
    'paired_review': {'path': str(TA_M5_PAIRED_BATCH_ORDER_ROOT / 'CACHED_SCORE_REVIEW_002.json'), 'sha256': '5635c3b7bcbfe6169e4b3766f8c093d3a9c29a6e5d8638a8adb9f2f0e961a3f9'},
    'paired_worker': {'path': str(CANONICAL_REPO / TA_M5_PAIRED_BATCH_ORDER_WORKER), 'sha256': 'd8f42bb3133e7ca33d026699a07bc178f3e110827398d99b36e8c0ca7f3b5cc8'},
    'control_evidence': {'path': str(TA_M5_PAIRED_BATCH_ORDER_ROOT / 'CONTROL_EVIDENCE.pt'), 'sha256': '6780622d9bc81204e046e4670d0b7091b4630864997def74420e19d21259ec9f'},
    'initial_train_measurement': {'path': str(TA_M5_PAIRED_BATCH_ORDER_ROOT / 'INITIAL_TRAIN.json'), 'sha256': '56eb106742bb2336747dfbd26cfa6d81aca78c18b7214b033158f1e8aad51d84'},
    'initial_train_capture': {'path': str(TA_M5_PAIRED_BATCH_ORDER_ROOT / 'INITIAL_TRAIN_READOUT.npz'), 'sha256': '5f205ecea562be6315f8bddbfea4da7f6fbec672ef8983b028cd3220d132cac7'},
    'control_train_measurement': {'path': str(TA_M5_PAIRED_BATCH_ORDER_ROOT / 'CONTROL_TRAIN.json'), 'sha256': 'fc42a5786d94785254df833de44934a498840c7bedee42daf6ed8b4458219c22'},
    'control_train_capture': {'path': str(TA_M5_PAIRED_BATCH_ORDER_ROOT / 'CONTROL_TRAIN_READOUT.npz'), 'sha256': 'cfe978111848add3eeecc2b0d01716ef3c58412edddd782032a228413d7bd8ed'},
    'control_later_measurement': {'path': str(TA_M5_PAIRED_BATCH_ORDER_ROOT / 'CONTROL_LATER.json'), 'sha256': '94d695306dc6691cf4578dac1420e70ee7c99ced9eaa656df46745c480ea6e36'},
    'control_later_capture': {'path': str(TA_M5_PAIRED_BATCH_ORDER_ROOT / 'CONTROL_LATER_READOUT.npz'), 'sha256': '4e79d0fa588dcfa07d47e307bec28fa3286aa149381f5d5e55707ff76bcd5839'},
}


def read_ta_m5_repeated_pass_origin(launch):
    refs = launch['input_refs']
    extra = TA_M5_REPEATED_PASS_REFS
    _need({key: refs.get(key) for key in extra} == extra, 'REPEATED_PASS_FIXED_REFS')
    base_refs = {key: ref for key, ref in refs.items() if key not in extra}
    origin = read_ta_m5_paired_batch_order_origin({'input_refs': base_refs})
    for key, ref in extra.items():
        campaign.require_binding(ref, label='repeated pass ' + key, verify_file=True)
    paired = _read(refs['paired_recipe'], 'repeated pass qualified paired recipe')
    report = _read(refs['paired_result'], 'repeated pass qualified paired result')
    terminal = _read(refs['paired_terminal'], 'repeated pass qualified paired terminal')
    review = _read(refs['paired_review'], 'repeated pass cached scores review')
    _need(paired.get('schema_version') == TA_M5_PAIRED_BATCH_ORDER_RECIPE_SCHEMA
          and paired.get('run_scope') == TA_M5_PAIRED_BATCH_ORDER_SCOPE
          and paired.get('admission_stage') == TA_M5_PAIRED_BATCH_ORDER_STAGE
          and paired.get('input_refs') == base_refs
          and paired.get('numerical_recipe') == origin['original_recipe']['numerical_recipe']
          and paired.get('trace_worker') == refs['paired_worker']
          and paired.get('arm_batch_indices') == TA_M5_PAIRED_BATCH_ORDER_BATCH_INDICES
          and paired.get('pass_index') == 0 and paired.get('canonical_resume_authority') is False,
          'REPEATED_PASS_PAIRED_RECIPE')
    _need(report.get('schema_version') == TA_M5_PAIRED_BATCH_ORDER_RESULT_SCHEMA
          and report.get('recipe') == refs['paired_recipe'] and report.get('outcome') == 'COMPLETE'
          and report.get('decision') == 'PASS_TA_M5_PAIRED_BATCH_ORDER_TECHNICAL'
          and report.get('physical_optimizer_updates') == report.get('optimizer_updates') == 512
          and report.get('checkpoint_writes') == 0 and report.get('skipped_batches') == 0
          and report.get('source_unchanged') is True and report.get('test_data_used') is False
          and report.get('canonical_resume_authority') is False
          and report.get('candidate_evidence', {}).get('CONTROL') == refs['control_evidence']
          and all(report.get(key) is True for key in (
              'same_selected_rows', 'same_initial_state', 'same_training_queries',
              'rng_stream_parity', 'evaluation_state_unchanged', 'all_panels_complete'))
          and all(type(report.get('counts', {}).get(key)) is int and report['counts'][key] == count
                  for key, count in TA_M5_PAIRED_BATCH_ORDER_COMPLETE_COUNTS.items()),
          'REPEATED_PASS_PAIRED_COMPLETION')
    _need(terminal.get('status') == 'COMPLETE' and terminal.get('exit_code') == 0
          and terminal.get('recipe') == refs['paired_recipe']
          and terminal.get('result') == refs['paired_result']
          and terminal.get('source_unchanged') is True
          and terminal.get('no_relaunch_under_this_plan') is True
          and terminal.get('test_data_used') is False
          and review.get('status') == 'PASS_INDEPENDENT_CACHED_SCORES_CONTROL_LOCAL_GAIN_ORDER_CHANGE_FAIL'
          and review.get('result') == refs['paired_result']
          and review.get('matched_immutable_axes_queries_truth') is True
          and review.get('test_data_used') is False, 'REPEATED_PASS_PAIRED_REVIEW')
    progress = report.get('arm_progress', {}).get('CONTROL', {})
    _need(progress.get('optimizer_updates') == 256 and progress.get('parent_cursor') == 4096
          and progress.get('batch_order_sha256') == campaign.canonical_sha256(
              TA_M5_PAIRED_BATCH_ORDER_BATCH_INDICES['CONTROL']), 'REPEATED_PASS_CONTROL_PROGRESS')
    campaign._sha(progress.get('parent_order_sha256'), 'repeated CONTROL parent order')
    measurements = {}
    for label in ('INITIAL_TRAIN', 'INITIAL_LATER', 'CONTROL_TRAIN', 'CONTROL_LATER'):
        key = label.lower()
        value = _read(refs[key + '_measurement'], 'repeated ' + label)
        measure = value.get('measurement', {})
        is_train, is_control = label.endswith('TRAIN'), label.startswith('CONTROL')
        _need(report.get('measurement_refs', {}).get(label) == refs[key + '_measurement']
              and report.get('captures', {}).get(label) == refs[key + '_capture']
              and value.get('capture') == refs[key + '_capture']
              and value.get('optimizer_updates') == (256 if is_control else 0)
              and value.get('test_data_used') is False
              and measure.get('rows') == (4096 if is_train else 192)
              and measure.get('batches') == (256 if is_train else 12)
              and measure.get('feature_view_sha256') == paired['expected_view_sha256']
              and measure.get('model_contract_sha256') == paired['expected_model_contract_sha256'],
              'REPEATED_PASS_MEASUREMENT:' + label)
        if is_train:
            _need(measure.get('pass_index') == 0 and measure.get('seed') == 20260911
                  and measure.get('query_domain') == {'state': 'sampled_pass0', 'future': 'native_endpoints'}
                  and measure.get('loss_contract_sha256') == paired['expected_loss_contract_sha256'],
                  'REPEATED_PASS_TRAIN_QUERY:' + label)
        if not is_control:
            _need(value.get('evaluated_model_state_sha256') == TA_M5_EQUAL_LOSS_MODEL_SHA256,
                  'REPEATED_PASS_INITIAL_MODEL')
        measurements[label] = value
    control_model = measurements['CONTROL_TRAIN'].get('evaluated_model_state_sha256')
    campaign._sha(control_model, 'repeated CONTROL model')
    _need(measurements['CONTROL_LATER'].get('evaluated_model_state_sha256') == control_model,
          'REPEATED_PASS_CONTROL_MODEL')
    return {**origin, 'paired_recipe': paired, 'paired_result': report, 'measurements': measurements}


def _ta_m5_repeated_pass_expected():
    value, original, qualified = _ta_m5_paired_batch_order_expected()
    value['input_refs'].update({key: dict(ref) for key, ref in TA_M5_REPEATED_PASS_REFS.items()})
    origin = read_ta_m5_repeated_pass_origin(value)
    for key in ('arms', 'arm_batch_indices', 'arm_batch_order_sha256', 'updates_per_arm',
                'both_arms_same_original_INITIAL_required', 'rng_stream_parity_required'):
        value.pop(key)
    value.update(
        schema_version=TA_M5_REPEATED_PASS_RECIPE_SCHEMA,
        runtime_root=str(TA_M5_REPEATED_PASS_RUNTIME_ROOT),
        admission_stage=TA_M5_REPEATED_PASS_STAGE, run_scope=TA_M5_REPEATED_PASS_SCOPE,
        execution_binding_path=str(TA_M5_REPEATED_PASS_RUNTIME_ROOT / 'EXECUTION_BINDING.json'),
        expected_initial_model_state_sha256=origin['measurements']['CONTROL_TRAIN']['evaluated_model_state_sha256'],
        parent_order_sha256=origin['paired_result']['arm_progress']['CONTROL']['parent_order_sha256'],
        batch_indices=list(TA_M5_PAIRED_BATCH_ORDER_BATCH_INDICES['CONTROL']),
        batch_order_sha256=campaign.canonical_sha256(TA_M5_PAIRED_BATCH_ORDER_BATCH_INDICES['CONTROL']),
        passes=[2, 3], updates_per_pass=256, starting_optimizer_updates=256, final_optimizer_updates=768,
        max_private_updates=512, max_backwards=512, max_forwards=1050,
        max_training_forwards=512, max_evaluation_forwards=538,
        max_checkpoint_loads=1, max_full_state_restores=1, max_candidate_evidence_writes=1,
        max_model_seconds=2400, max_invocation_seconds=12000,
        initial_evaluation_forwards=0, baseline_parity_forwards=2,
        selected_TRAIN_evaluation_forwards=512, later_evaluation_forwards=24,
        primary_baselines='qualified_cached_original_INITIAL_and_CONTROL_E1_same4096_and_later192',
        strict_control_restore_required=True, optimizer_state_continued_required=True,
        rng_state_continued_required=True, unchanged_query_pass_index=0,
        unselected_suffix_consumed=False, automatic_continuation=False)
    return value, original, qualified


def build_ta_m5_repeated_pass_recipe(*, source_commit):
    value, original, qualified = _ta_m5_repeated_pass_expected()
    owners = {name: {'path': str(CANONICAL_REPO / name),
                     'sha256': campaign.file_sha256(CANONICAL_REPO / name)}
              for name in original['source_owners']}
    worker = CANONICAL_REPO / TA_M5_REPEATED_PASS_WORKER
    value.update(source_owners=owners,
        trace_worker={'path': str(worker), 'sha256': campaign.file_sha256(worker)},
        source_bridge=_ta_m5_original_cuda_prefix_sources(
            owners, original, qualified, value['input_refs']['original_trace_helper'], paired_order_guard=True))
    value['recipe_sha256'] = campaign.canonical_sha256(value)
    return require_ta_m5_repeated_pass_recipe(value, source_repo=CANONICAL_REPO, source_commit=source_commit)


def require_ta_m5_repeated_pass_recipe(value, *, source_repo, source_commit, verify_files=True):
    import subprocess
    _need(verify_files and source_repo == CANONICAL_REPO, 'REPEATED_PASS_PHYSICAL_SOURCE')
    expected, original, qualified = _ta_m5_repeated_pass_expected()
    value = _identity(value, set(expected) | {'source_owners', 'trace_worker', 'source_bridge'},
                      TA_M5_REPEATED_PASS_RECIPE_SCHEMA, 'recipe_sha256')
    _need(campaign.canonical_sha256({key: value[key] for key in expected})
          == campaign.canonical_sha256(expected), 'REPEATED_PASS_RECIPE_SCOPE')
    _need(isinstance(source_commit, str) and campaign._COMMIT.fullmatch(source_commit)
          and subprocess.run(['git', 'merge-base', '--is-ancestor',
              TA_M5_ORIGINAL_CUDA_PREFIX_REFERENCE_SOURCE, source_commit],
              cwd=CANONICAL_REPO, capture_output=True).returncode == 0,
          'REPEATED_PASS_SOURCE_ANCESTRY')
    _need(value['trace_worker'].get('path') == str(CANONICAL_REPO / TA_M5_REPEATED_PASS_WORKER),
          'REPEATED_PASS_WORKER_PATH')
    campaign.require_binding(value['trace_worker'], label='repeated pass worker', verify_file=True)
    expected_bridge = _ta_m5_original_cuda_prefix_sources(
        value['source_owners'], original, qualified, value['input_refs']['original_trace_helper'], paired_order_guard=True)
    _need(value['source_bridge'] == expected_bridge, 'REPEATED_PASS_SOURCE_BRIDGE')
    return value


def require_ta_m5_repeated_pass_gate(gate, recipe):
    expected = {
        'decision': 'ADMITTED', 'status': 'ADMITTED', 'ta_m5_repeated_pass_authorized': True,
        'fold_id': 1, 'selected_fit_rows': 4096, 'passes': [2, 3],
        'starting_optimizer_updates': 256, 'final_optimizer_updates': 768,
        'model_constructions_authorized': 1, 'optimizer_constructions_authorized': 1,
        'model_forwards_authorized': 1050, 'backward_calls_authorized': 512,
        'optimizer_steps_authorized': 512, 'training_enabled': True, 'native_training_enabled': True,
        'genuine_updates_authorized': True, 'genuine_backward_authorized': True,
        'genuine_predictor_fit_authorized': False, 'canonical_resume_authority': False,
        'candidate_selection': False, 'test_data_used': False,
        'evaluate_both_panels_unconditionally': True, 'per_update_pareto_requirement': False,
    }
    _need(isinstance(gate, Mapping) and all(gate.get(k) == v and type(gate[k]) is type(v)
          for k, v in expected.items()) and recipe['run_scope'] == TA_M5_REPEATED_PASS_SCOPE
          and recipe['admission_stage'] == TA_M5_REPEATED_PASS_STAGE, 'REPEATED_PASS_POLICY_GATE')


def build_ta_m5_repeated_pass_cursor(*, recipe_ref, outcome, optimizer_updates,
                                    current_pass, batch_index, cumulative_optimizer_updates,
                                    result, batch_size=16):
    value = dict(schema_version=TA_M5_REPEATED_PASS_CURSOR_SCHEMA, recipe=recipe_ref,
                 outcome=outcome, optimizer_updates=optimizer_updates, current_pass=current_pass,
                 batch_index=batch_index, cumulative_optimizer_updates=cumulative_optimizer_updates,
                 parent_cursor=None, parent_order_sha256=None, full_state=None,
                 result=result, test_data_used=False, canonical_resume_authority=False)
    value['cursor_sha256'] = campaign.canonical_sha256(value)
    return require_ta_m5_repeated_pass_cursor(
        value, expected_recipe=recipe_ref, verify_files=False, batch_size=batch_size)


def require_ta_m5_repeated_pass_cursor(value, *, expected_recipe, verify_files=True, batch_size=16):
    fields = {'recipe', 'outcome', 'optimizer_updates', 'current_pass', 'batch_index',
              'cumulative_optimizer_updates', 'parent_cursor', 'parent_order_sha256', 'full_state',
              'result', 'test_data_used', 'canonical_resume_authority', 'schema_version'}
    cursor = _identity(value, fields, TA_M5_REPEATED_PASS_CURSOR_SCHEMA, 'cursor_sha256')
    _need(type(batch_size) is int and batch_size == 16
          and expected_recipe.get('path') == str(TA_M5_REPEATED_PASS_RECIPE_PATH)
          and cursor['recipe'] == expected_recipe and cursor['outcome'] in ('COMPLETE', 'FAILED')
          and type(cursor['optimizer_updates']) is int and 0 <= cursor['optimizer_updates'] <= 512
          and cursor['parent_cursor'] is None and cursor['parent_order_sha256'] is None
          and cursor['full_state'] is None and cursor['test_data_used'] is False
          and cursor['canonical_resume_authority'] is False, 'REPEATED_PASS_CURSOR_SCOPE')
    _need(type(cursor['current_pass']) is int and cursor['current_pass'] in (2, 3)
          and type(cursor['batch_index']) is int and 0 <= cursor['batch_index'] <= 256
          and type(cursor['cumulative_optimizer_updates']) is int
          and cursor['optimizer_updates'] == (cursor['current_pass'] - 2) * 256 + cursor['batch_index']
          and cursor['cumulative_optimizer_updates'] == 256 + cursor['optimizer_updates'],
          'REPEATED_PASS_CURSOR_PROGRESS')
    complete = cursor['outcome'] == 'COMPLETE'
    _need(not complete or (cursor['current_pass'] == 3 and cursor['batch_index'] == 256
          and cursor['optimizer_updates'] == 512 and cursor['cumulative_optimizer_updates'] == 768),
          'REPEATED_PASS_CURSOR_COMPLETE')
    _need(cursor['result'].get('path') == str(TA_M5_REPEATED_PASS_ROOT / 'RESULT.json'),
          'REPEATED_PASS_RESULT_PATH')
    campaign.require_binding(cursor['result'], label='repeated pass result', verify_file=verify_files)
    if verify_files:
        report = _read(cursor['result'], 'repeated pass result')
        _need(report.get('schema_version') == TA_M5_REPEATED_PASS_RESULT_SCHEMA
              and report.get('recipe') == expected_recipe and report.get('outcome') == cursor['outcome']
              and type(report.get('physical_optimizer_updates')) is int
              and type(report.get('optimizer_updates')) is int
              and report.get('physical_optimizer_updates') == report.get('optimizer_updates') == cursor['optimizer_updates']
              and all(report.get(key) == cursor[key] and type(report.get(key)) is int for key in (
                  'current_pass', 'batch_index', 'cumulative_optimizer_updates'))
              and report.get('parent_cursor') is None and report.get('parent_order_sha256') is None
              and report.get('full_state') is None and report.get('final_state') is None
              and report.get('canonical_resume_authority') is False
              and report.get('checkpoint_writes') == 0 and report.get('test_data_used') is False,
              'REPEATED_PASS_CURSOR_RESULT')
        if complete:
            _need(report.get('decision') == 'PASS_TA_M5_REPEATED_PASS_TECHNICAL'
                  and report.get('source_unchanged') is True and report.get('skipped_batches') == 0
                  and all(report.get(key) is True for key in (
                      'control_restore_exact', 'baseline_output_parity', 'evaluation_state_unchanged',
                      'all_measurements_complete', 'optimizer_state_continued'))
                  and all(type(report.get('counts', {}).get(key)) is int
                          and report['counts'][key] == count
                          for key, count in TA_M5_REPEATED_PASS_COMPLETE_COUNTS.items()),
                  'REPEATED_PASS_TECHNICAL_COMPLETION')
            evidence = report.get('candidate_evidence', {})
            _need(isinstance(evidence, Mapping)
                  and evidence.get('path') == str(TA_M5_REPEATED_PASS_ROOT / 'E3_EVIDENCE.pt'),
                  'REPEATED_PASS_EVIDENCE_PATH')
            campaign.require_binding(evidence, label='repeated pass private E3 evidence', verify_file=True)
    return cursor

# One full-population order contrast, with the original six-row tail last.
TA_M5_FULL_REVERSE_ORDER_STAGE = 'market_state_ta_m5_full_reverse_order_20261110_001'
TA_M5_FULL_REVERSE_ORDER_SCOPE = 'market_state_ta_m5_full_reverse_order_v1'
TA_M5_FULL_REVERSE_ORDER_ROOT = Path('/home/andre2/GX1_RUNS') / TA_M5_FULL_REVERSE_ORDER_STAGE.upper()
TA_M5_FULL_REVERSE_ORDER_RUNTIME_ROOT = TA_M5_FULL_REVERSE_ORDER_ROOT / 'RUNTIME'
TA_M5_FULL_REVERSE_ORDER_RECIPE_PATH = TA_M5_FULL_REVERSE_ORDER_ROOT / 'RECIPE.json'
TA_M5_FULL_REVERSE_ORDER_RECIPE_SCHEMA = 'gx1_market_state_ta_m5_full_reverse_order_recipe_v1'
TA_M5_FULL_REVERSE_ORDER_RESULT_SCHEMA = 'gx1_market_state_ta_m5_full_reverse_order_result_v1'
TA_M5_FULL_REVERSE_ORDER_CURSOR_SCHEMA = 'gx1_market_state_ta_m5_full_reverse_order_cursor_v1'
TA_M5_FULL_REVERSE_ORDER_WORKER = 'gx1/audit/run_market_state_ta_m5_full_reverse_order_v1.py'
TA_M5_FULL_REVERSE_ORDER_BATCH_INDICES = list(range(1847, -1, -1)) + [1848]
TA_M5_FULL_REVERSE_ORDER_COMPLETE_COUNTS = {
    'model_constructions': 1, 'optimizer_constructions': 1, 'checkpoint_loads': 1,
    'strict_full_state_restore_calls': 1, 'forwards': 3710, 'training_forwards': 1849,
    'evaluation_forwards': 1861, 'backwards': 1849, 'optimizer_step_calls': 1849,
    'optimizer_updates': 1849, 'checkpoint_writes': 0, 'candidate_evidence_writes': 1,
}
_TA_REVERSE_CACHE = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_EQUAL_LOSS_FINAL_CACHE_20261109_001')
_TA_REVERSE_READOUT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_FINAL_READOUT_20261109_001')
TA_M5_FULL_REVERSE_ORDER_REFS = {
    'initial_score_cache': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_DIRECTION_FULL_CACHE_20261108_001/SCORE_CACHE.npz', 'sha256': '8cc964f1ff229bc3c1140a47ed66e256ce90b9ecfbc18c4c4509ba60f63bfee9'},
    'final_fit_cache': {'path': str(_TA_REVERSE_CACHE / 'FIT_CACHE.npz'), 'sha256': 'b41874cd1c99d371f3899e4f017c06c1757f4b92dece140c34846bf8ba1772a4'},
    'final_score_cache': {'path': str(_TA_REVERSE_CACHE / 'SCORE_CACHE.npz'), 'sha256': '299a279072cb139224190e93aeb190f167c9e047fce55b16a95e3438616147a0'},
    'final_cache_result': {'path': str(_TA_REVERSE_CACHE / 'RESULT.json'), 'sha256': 'fb337666ed23ff4892f7c0362484e25c1fa49f7c8ed121c3ff2a65f035925687'},
    'final_cache_review': {'path': str(_TA_REVERSE_CACHE / 'REVIEW.json'), 'sha256': '1638e131a6c092a1aa2157263f284fdf6e084343906f44a1e7abdf67e8add36b'},
    'readout_plan': {'path': str(_TA_REVERSE_READOUT / 'PLAN.json'), 'sha256': 'fe5bbbd7b6faba84335b47b3d3e4525f5566ea1dc04359e4f7ed04b4188b8972'},
    'readout_result': {'path': str(_TA_REVERSE_READOUT / 'RESULT.json'), 'sha256': '623e66c9baba2a3cf6910ffb10a11ba2e14a4c2e514e3660a83528c361c16ae6'},
    'readout_review': {'path': str(_TA_REVERSE_READOUT / 'REVIEW.json'), 'sha256': 'deff9b4774ad68e6306d598d2073f43e36b126cc29069122965b5319d4ef310d'},
    'paired_worker': dict(TA_M5_REPEATED_PASS_REFS['paired_worker']),
}


def read_ta_m5_full_reverse_order_origin(launch):
    refs = launch['input_refs']
    extra = TA_M5_FULL_REVERSE_ORDER_REFS
    _need({key: refs.get(key) for key in extra} == extra, 'FULL_REVERSE_FIXED_REFS')
    origin = read_ta_m5_paired_batch_order_origin(
        {'input_refs': {key: ref for key, ref in refs.items() if key not in extra}})
    for key, ref in extra.items():
        campaign.require_binding(ref, label='full reverse ' + key, verify_file=True)
    cached = _read(refs['final_cache_result'], 'full reverse chronological FINAL cache')
    review = _read(refs['final_cache_review'], 'full reverse FINAL cache qualification')
    _need(cached.get('status') == 'COMPLETE' and cached.get('source_unchanged') is True
          and cached.get('model_state_unchanged') is True and cached.get('test_data_used') is False
          and cached.get('sampled_S_pass_index') == 0 and cached.get('sampled_F_reuses_native_1') is True
          and cached.get('checkpoint', {}).get('source_commit') == TA_M5_ORIGINAL_CUDA_PREFIX_SOURCE
          and cached['checkpoint'].get('model_sha256') == '23ff344b183a2256f2c8a1a8e32b811ed14f10c1cb1423becca14b36d5340d5b'
          and review.get('status') == 'COMPLETE_CONSUMED_CACHE_QUALIFIED'
          and review.get('result') == refs['final_cache_result']
          and review.get('source_unchanged') is True and review.get('test_data_used') is False
          and review.get('all_twenty_partition_task_TF_pairs_identical') is True,
          'FULL_REVERSE_FINAL_BASELINE')
    for part, rows in (('fit', 29574), ('score', 17753)):
        item = cached.get('cache_partitions', {}).get(part, {})
        _need(item.get('artifact') == refs['final_' + part + '_cache']
              and item.get('rows') == rows and item.get('partial') is False,
              'FULL_REVERSE_FINAL_CACHE:' + part)
    readout_plan = _read(refs['readout_plan'], 'full reverse baseline output plan')
    readout = _read(refs['readout_result'], 'full reverse baseline outputs')
    readout_review = _read(refs['readout_review'], 'full reverse baseline output qualification')
    _need(readout.get('status') == 'COMPLETE' and readout.get('source_unchanged') is True
          and readout.get('test_data_used') is False and readout.get('plan') == refs['readout_plan']
          and readout.get('private_outputs') == refs['imported_predictions']
          and readout_review.get('status') == 'COMPLETE_CONSUMED_READOUT_RECOVERS_TRAIN_NOT_LATER_EDGE'
          and readout_review.get('result') == refs['readout_result']
          and readout_review.get('source_unchanged') is True and readout_review.get('test_data_used') is False,
          'FULL_REVERSE_READOUT_PROOF')
    _need(all(readout_plan.get('inputs', {}).get(key) == refs[name] for key, name in (
        ('old_fit_cache', 'initial_fit_cache'), ('old_score_cache', 'initial_score_cache'),
        ('new_fit_cache', 'final_fit_cache'), ('new_score_cache', 'final_score_cache'),
        ('new_cache_result', 'final_cache_result'), ('new_cache_review', 'final_cache_review'))),
        'FULL_REVERSE_BASELINE_AXES_BINDING')
    return {**origin, 'final_cache_result': cached, 'readout_result': readout, 'readout_plan': readout_plan}


def _ta_m5_full_reverse_order_expected():
    value, original, qualified = _ta_m5_paired_batch_order_expected()
    value['input_refs'].update({key: dict(ref) for key, ref in TA_M5_FULL_REVERSE_ORDER_REFS.items()})
    read_ta_m5_full_reverse_order_origin(value)
    for key in ('arms', 'arm_batch_indices', 'arm_batch_order_sha256', 'updates_per_arm',
                'both_arms_same_original_INITIAL_required', 'rng_stream_parity_required',
                'same_selected_rows_required'):
        value.pop(key)
    value.update(
        schema_version=TA_M5_FULL_REVERSE_ORDER_RECIPE_SCHEMA,
        runtime_root=str(TA_M5_FULL_REVERSE_ORDER_RUNTIME_ROOT),
        admission_stage=TA_M5_FULL_REVERSE_ORDER_STAGE, run_scope=TA_M5_FULL_REVERSE_ORDER_SCOPE,
        execution_binding_path=str(TA_M5_FULL_REVERSE_ORDER_RUNTIME_ROOT / 'EXECUTION_BINDING.json'),
        arm='REVERSED', batch_indices=list(TA_M5_FULL_REVERSE_ORDER_BATCH_INDICES),
        batch_order_sha256=campaign.canonical_sha256(TA_M5_FULL_REVERSE_ORDER_BATCH_INDICES),
        parent_selection='all_original_fold1_fit_rows_once',
        parent_order_role='original_INITIAL_full_order_before_explicit_permutation',
        selected_fit_rows=29574, original_complete_batches=1848, excluded_partial_batch_rows=0,
        tail_batch_index=1848, tail_rows=6, tail_policy='original_tail_last',
        max_private_updates=1849, max_backwards=1849, max_forwards=3710,
        max_training_forwards=1849, max_evaluation_forwards=1861,
        max_checkpoint_loads=1, max_full_state_restores=1, max_candidate_evidence_writes=1,
        max_model_seconds=7200, max_invocation_seconds=12000,
        initial_evaluation_forwards=0, selected_TRAIN_evaluation_forwards=1849,
        later_evaluation_forwards=12,
        primary_baselines='cached_original_chronological_FINAL_and_INITIAL_full_TRAIN_and_later192',
        same_initial_full_state_required=True, all_original_rows_once_required=True,
        tail6_preserved_required=True, original_chronological_control_replayed=False,
        full_epoch_training_allowed=True, genuine_predictor_fit_authorized=True,
        unchanged_query_pass_index=0, automatic_continuation=False)
    return value, original, qualified


def build_ta_m5_full_reverse_order_recipe(*, source_commit):
    value, original, qualified = _ta_m5_full_reverse_order_expected()
    owners = {name: {'path': str(CANONICAL_REPO / name),
                     'sha256': campaign.file_sha256(CANONICAL_REPO / name)}
              for name in original['source_owners']}
    worker = CANONICAL_REPO / TA_M5_FULL_REVERSE_ORDER_WORKER
    value.update(source_owners=owners,
        trace_worker={'path': str(worker), 'sha256': campaign.file_sha256(worker)},
        source_bridge=_ta_m5_original_cuda_prefix_sources(
            owners, original, qualified, value['input_refs']['original_trace_helper'], paired_order_guard=True))
    value['recipe_sha256'] = campaign.canonical_sha256(value)
    return require_ta_m5_full_reverse_order_recipe(value, source_repo=CANONICAL_REPO, source_commit=source_commit)


def require_ta_m5_full_reverse_order_recipe(value, *, source_repo, source_commit, verify_files=True):
    import subprocess
    _need(verify_files and source_repo == CANONICAL_REPO, 'FULL_REVERSE_PHYSICAL_SOURCE')
    expected, original, qualified = _ta_m5_full_reverse_order_expected()
    value = _identity(value, set(expected) | {'source_owners', 'trace_worker', 'source_bridge'},
                      TA_M5_FULL_REVERSE_ORDER_RECIPE_SCHEMA, 'recipe_sha256')
    _need(campaign.canonical_sha256({key: value[key] for key in expected})
          == campaign.canonical_sha256(expected), 'FULL_REVERSE_RECIPE_SCOPE')
    _need(isinstance(source_commit, str) and campaign._COMMIT.fullmatch(source_commit)
          and subprocess.run(['git', 'merge-base', '--is-ancestor',
              TA_M5_ORIGINAL_CUDA_PREFIX_REFERENCE_SOURCE, source_commit],
              cwd=CANONICAL_REPO, capture_output=True).returncode == 0,
          'FULL_REVERSE_SOURCE_ANCESTRY')
    _need(value['trace_worker'].get('path') == str(CANONICAL_REPO / TA_M5_FULL_REVERSE_ORDER_WORKER),
          'FULL_REVERSE_WORKER_PATH')
    campaign.require_binding(value['trace_worker'], label='full reverse worker', verify_file=True)
    expected_bridge = _ta_m5_original_cuda_prefix_sources(
        value['source_owners'], original, qualified, value['input_refs']['original_trace_helper'], paired_order_guard=True)
    _need(value['source_bridge'] == expected_bridge, 'FULL_REVERSE_SOURCE_BRIDGE')
    return value


def require_ta_m5_full_reverse_order_gate(gate, recipe):
    expected = {
        'decision': 'ADMITTED', 'status': 'ADMITTED', 'ta_m5_full_reverse_order_authorized': True,
        'fold_id': 1, 'selected_fit_rows': 29574, 'arm': 'REVERSED', 'tail_rows': 6,
        'model_constructions_authorized': 1, 'optimizer_constructions_authorized': 1,
        'model_forwards_authorized': 3710, 'backward_calls_authorized': 1849,
        'optimizer_steps_authorized': 1849, 'training_enabled': True, 'native_training_enabled': True,
        'genuine_updates_authorized': True, 'genuine_backward_authorized': True,
        'genuine_predictor_fit_authorized': True, 'full_epoch_training_allowed': True,
        'canonical_resume_authority': False,
        'candidate_selection': False, 'test_data_used': False,
        'evaluate_both_panels_unconditionally': True, 'per_update_pareto_requirement': False,
    }
    _need(isinstance(gate, Mapping) and all(gate.get(k) == v and type(gate[k]) is type(v)
          for k, v in expected.items()) and recipe['run_scope'] == TA_M5_FULL_REVERSE_ORDER_SCOPE
          and recipe['admission_stage'] == TA_M5_FULL_REVERSE_ORDER_STAGE, 'FULL_REVERSE_POLICY_GATE')


def build_ta_m5_full_reverse_order_cursor(*, recipe_ref, outcome, optimizer_updates,
                                         parent_cursor, parent_order_sha256, result, batch_size=16):
    value = dict(schema_version=TA_M5_FULL_REVERSE_ORDER_CURSOR_SCHEMA, recipe=recipe_ref,
                 outcome=outcome, optimizer_updates=optimizer_updates, parent_cursor=parent_cursor,
                 parent_order_sha256=parent_order_sha256, full_state=None,
                 result=result, test_data_used=False, canonical_resume_authority=False)
    value['cursor_sha256'] = campaign.canonical_sha256(value)
    return require_ta_m5_full_reverse_order_cursor(
        value, expected_recipe=recipe_ref, verify_files=False, batch_size=batch_size)


def require_ta_m5_full_reverse_order_cursor(value, *, expected_recipe, verify_files=True, batch_size=16):
    fields = {'recipe', 'outcome', 'optimizer_updates', 'parent_cursor', 'parent_order_sha256',
              'full_state', 'result', 'test_data_used', 'canonical_resume_authority', 'schema_version'}
    cursor = _identity(value, fields, TA_M5_FULL_REVERSE_ORDER_CURSOR_SCHEMA, 'cursor_sha256')
    _need(type(batch_size) is int and batch_size == 16
          and expected_recipe.get('path') == str(TA_M5_FULL_REVERSE_ORDER_RECIPE_PATH)
          and cursor['recipe'] == expected_recipe and cursor['outcome'] in ('COMPLETE', 'FAILED')
          and type(cursor['optimizer_updates']) is int and 0 <= cursor['optimizer_updates'] <= 1849
          and type(cursor['parent_cursor']) is int
          and cursor['parent_cursor'] == min(16 * cursor['optimizer_updates'], 29574)
          and cursor['full_state'] is None and cursor['test_data_used'] is False
          and cursor['canonical_resume_authority'] is False, 'FULL_REVERSE_CURSOR_SCOPE')
    if cursor['parent_order_sha256'] is None:
        _need(cursor['outcome'] == 'FAILED' and cursor['optimizer_updates'] == 0, 'FULL_REVERSE_MISSING_ORDER')
    else:
        campaign._sha(cursor['parent_order_sha256'], 'full reverse parent order')
        _need(cursor['parent_order_sha256'] != TA_M5_EQUAL_LOSS_ORDER_SHA256, 'FULL_REVERSE_UNCHANGED_ORDER')
    complete = cursor['outcome'] == 'COMPLETE'
    _need(not complete or cursor['optimizer_updates'] == 1849, 'FULL_REVERSE_CURSOR_COMPLETE')
    _need(cursor['result'].get('path') == str(TA_M5_FULL_REVERSE_ORDER_ROOT / 'RESULT.json'),
          'FULL_REVERSE_RESULT_PATH')
    campaign.require_binding(cursor['result'], label='full reverse result', verify_file=verify_files)
    if verify_files:
        report = _read(cursor['result'], 'full reverse result')
        _need(report.get('schema_version') == TA_M5_FULL_REVERSE_ORDER_RESULT_SCHEMA
              and report.get('recipe') == expected_recipe and report.get('outcome') == cursor['outcome']
              and type(report.get('physical_optimizer_updates')) is int
              and type(report.get('optimizer_updates')) is int
              and report.get('physical_optimizer_updates') == report.get('optimizer_updates') == cursor['optimizer_updates']
              and type(report.get('parent_cursor')) is int and report['parent_cursor'] == cursor['parent_cursor']
              and report.get('parent_order_sha256') == cursor['parent_order_sha256']
              and report.get('full_state') is None and report.get('final_state') is None
              and report.get('canonical_resume_authority') is False
              and report.get('checkpoint_writes') == 0 and report.get('test_data_used') is False,
              'FULL_REVERSE_CURSOR_RESULT')
        if complete:
            _need(report.get('decision') == 'PASS_TA_M5_FULL_REVERSE_ORDER_TECHNICAL'
                  and report.get('source_unchanged') is True and report.get('skipped_batches') == 0
                  and all(report.get(key) is True for key in (
                      'same_initial_state', 'same_training_queries', 'evaluation_state_unchanged',
                      'all_panels_complete', 'all_original_rows_once', 'tail6_preserved'))
                  and all(type(report.get('counts', {}).get(key)) is int
                          and report['counts'][key] == count
                          for key, count in TA_M5_FULL_REVERSE_ORDER_COMPLETE_COUNTS.items()),
                  'FULL_REVERSE_TECHNICAL_COMPLETION')
            evidence = report.get('candidate_evidence', {})
            _need(isinstance(evidence, Mapping)
                  and evidence.get('path') == str(TA_M5_FULL_REVERSE_ORDER_ROOT / 'REVERSED_EVIDENCE.pt'),
                  'FULL_REVERSE_EVIDENCE_PATH')
            campaign.require_binding(evidence, label='full reverse private evidence', verify_file=True)
    return cursor
