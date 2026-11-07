"""Exact TRAIN-only fold-1 time-query fit recipe and result boundaries.

The guarded prefix-fit campaign owns invocation, boot, power, cursor and
capacity gates. This contract admits only the mature fold-1 population and
its immediately later TRAIN score slice; it does not authorize a model run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts.entry_model_native_train_recipe_v1 import native_direction_research_recipe
from gx1.contracts import market_state_time_query_private_campaign_v1 as private

SCOPE = 'market_state_time_query_fold1_one_pass_fit_v1'
STAGE = 'market_state_time_query_fold1_fit_20261107_001'
RECIPE_SCHEMA = 'gx1_native_direction_prefix_fit_launch_recipe_v1'
NUMERICAL_SCHEMA = 'gx1_market_state_time_query_fold1_fit_numerical_v1'
RESULT_SCHEMA = 'gx1_market_state_time_query_fold1_fit_result_v1'
CANONICAL_REPO = Path('/home/andre2/src/GX1_CURRENT')
RUNTIME_ROOT = Path('/home/andre2/GX1_RUNS') / STAGE.upper() / 'RUNTIME'
RECIPE_PATH = RUNTIME_ROOT.parent / 'RECIPE.json'
FOLD = {
    'fold_id': 1,
    'prefix_rows': 29574,
    'oof_start_row': 29849,
    'oof_end_row': 47602,
    'train_cutoff_ns': 1727654400000000000,
    'oof_cutoff_ns': 1735689600000000000,
}
FIT_ROWS = 29574
SCORE_ROWS = 17753
BATCH_SIZE = 16
MAX_UPDATES = 1849
MAX_WINDOWS = 4
PASS_INDEX = 0
CHECKPOINT_EVERY_UPDATES = 256
MAX_CHECKPOINT_BYTES = 1073741824
MAX_CHECKPOINT_TOTAL_BYTES = 4294967296
MAX_PUBLISHED_JSON_BYTES = 67108864
VIEW_SHA256 = '7e0d13f1b4f58cc4d3dd294bf869d3195b48a69c022fe570e6b4db02f1581f15'
MODEL_CONTRACT_SHA256 = private.MODEL_CONTRACT_SHA256
LOSS_CONTRACT_SHA256 = private.LOSS_CONTRACT_SHA256
REFERENCE_RESULT = {
    'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TIME_QUERY_REFERENCES_20261107_001/RESULT.json',
    'sha256': '1ef8e55ebf38a7571cd98e812491c37d2ab7498403734f7701d1f9ee442f4d4b',
}
EXTRA_SOURCE_PATHS = frozenset({
    'gx1/contracts/entry_native_direction_prefix_fit_campaign_v1.py',
    'gx1/scripts/run_entry_native_direction_prefix_fit_window_v1.py',
    'gx1/contracts/market_state_time_query_fold1_fit_v1.py',
    'gx1/contracts/market_state_time_query_fold1_fit_state_v1.py',
    'gx1/scripts/run_market_state_time_query_fold1_fit_window_v1.py',
    'gx1/scripts/materialize_market_state_time_query_fold1_fit_campaign_v1.py',
    'gx1/audit/score_market_state_time_query_fold1_oof_v1.py',
    'gx1/contracts/market_state_time_query_scorer_v1.py',
    'gx1/contracts/model_state_digest_v1.py',
})


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise campaign.RandomAccessCampaignError('TIME_QUERY_FOLD1_FIT_' + reason)


def _identity(value: Any, fields: set[str]) -> dict[str, Any]:
    _need(isinstance(value, Mapping) and set(value) == fields | {'recipe_sha256'},
          'RECIPE_FIELDS')
    result = dict(value)
    claimed = campaign._sha(result.pop('recipe_sha256'), 'recipe seal')
    _need(result.get('schema_version') == RECIPE_SCHEMA
          and claimed == campaign.canonical_sha256(result), 'RECIPE_IDENTITY')
    result['recipe_sha256'] = claimed
    return result


def fold1_numerical_recipe() -> dict[str, Any]:
    """Retain qualified AdamW/FP32 values; change only objective and full-pass cap."""
    base = dict(native_direction_research_recipe(
        batch_size=BATCH_SIZE, max_updates=MAX_UPDATES))
    base.pop('contract_sha256')
    base['schema_version'] = NUMERICAL_SCHEMA
    base['research_objective_sha256'] = MODEL_CONTRACT_SHA256
    base['time_query_loss_contract_sha256'] = LOSS_CONTRACT_SHA256
    raw = json.dumps(base, sort_keys=True, separators=(',', ':'),
                     ensure_ascii=True, allow_nan=False).encode()
    return {**base, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def required_source_paths() -> frozenset[str]:
    """Exact executable and lineage owners for this one candidate."""
    from gx1.contracts import entry_native_direction_research_campaign_v1 as research
    return frozenset(research.REQUIRED_SOURCE_PATHS) | frozenset(
        private.EXTRA_SOURCE_PATHS) | EXTRA_SOURCE_PATHS


def require_checkpoint_view(value: Any, *, recipe: Mapping[str, Any] | None = None,
                            verify_files: bool = True) -> dict[str, Any]:
    """Require the exact matured fold-1 feature and normalization identity."""
    _need(isinstance(value, Mapping)
          and set(value) == {'contract_sha256', 'prefix_normalization',
                             'research_objective_sha256'}
          and value['contract_sha256'] == VIEW_SHA256
          and value['research_objective_sha256'] == MODEL_CONTRACT_SHA256,
          'CHECKPOINT_VIEW')
    prefix = value['prefix_normalization']
    _need(isinstance(prefix, Mapping)
          and set(prefix) == {'binding', 'contract_sha256', 'base_sha256',
                              'gc_layout_sha256', 'gc_normalization_sha256',
                              'qualification_sha256'},
          'CHECKPOINT_NORMALIZATION_SHAPE')
    expected = private.bound_input_refs()['fold1_normalization']
    _need(prefix['binding'] == expected, 'CHECKPOINT_FOLD1_NORMALIZATION')
    if recipe is not None:
        _need(recipe.get('run_scope') == SCOPE
              and recipe.get('expected_view_sha256') == VIEW_SHA256
              and recipe.get('input_refs', {}).get('fold1_normalization') == expected,
              'CHECKPOINT_RECIPE_VIEW')
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix_fit
    return prefix_fit._view_identity(value, verify_files=verify_files)


def _reference_report() -> dict[str, Any]:
    checked = campaign.require_binding(
        REFERENCE_RESULT, label='frozen TRAIN reference', verify_file=True)
    report = campaign.read_bound_json(Path(checked['path']), checked['sha256'])
    _need(report.get('test_data_used') is False
          and report.get('model_fits') == 0
          and report.get('read_only') is True,
          'FROZEN_REFERENCE_ROLE')
    folds = report.get('references', {}).get('folds')
    _need(isinstance(folds, list) and len(folds) == 5
          and folds[0].get('fold_id') == 1
          and folds[0].get('fit_rows') == FIT_ROWS
          and folds[0].get('score_rows') == SCORE_ROWS
          and folds[0].get('effective_fit_cutoff_ns') == FOLD['train_cutoff_ns']
          and folds[0].get('official_fit_cutoff_ns') == 1727740800000000000
          and folds[0].get('score_end_exclusive_ns') == FOLD['oof_cutoff_ns']
          and isinstance(folds[0].get('reference', {}).get('per_tf_k', {})
                         .get('M5', {}).get('1', {}).get('prior'), dict),
          'FROZEN_REFERENCE_FOLD1')
    return report


def require_launch_recipe(value: Any, *, source_repo: Path,
                          source_commit: str,
                          verify_files: bool = True) -> dict[str, Any]:
    """Validate a distinct fit admission without the legacy prefix fold registry."""
    _need(verify_files, 'PHYSICAL_METADATA_REQUIRED')
    fields = {
        'schema_version', 'source_repo', 'runtime_root', 'run_id',
        'admission_stage', 'run_scope', 'fold', 'train_passes', 'max_windows',
        'numerical_recipe', 'input_manifest', 'input_refs', 'reference_result',
        'expected_view_sha256', 'expected_model_contract_sha256',
        'expected_loss_contract_sha256', 'source_owners',
        'execution_binding_path', 'checkpoint_every_updates',
        'max_checkpoint_bytes', 'max_checkpoint_total_bytes',
        'max_published_json_bytes', 'training_enabled',
        'full_epoch_training_allowed', 'test_data_used',
        'predictive_value_claimed', 'qualification_weights_for_main_fit',
    }
    recipe = _identity(value, fields)
    _need(source_repo == CANONICAL_REPO
          and type(source_commit) is str
          and campaign._COMMIT.fullmatch(source_commit) is not None
          and recipe['source_repo'] == str(CANONICAL_REPO)
          and recipe['runtime_root'] == str(RUNTIME_ROOT)
          and recipe['run_id'] == STAGE
          and recipe['admission_stage'] == STAGE
          and recipe['run_scope'] == SCOPE
          and recipe['fold'] == FOLD
          and type(recipe['train_passes']) is int and recipe['train_passes'] == 1
          and type(recipe['max_windows']) is int and recipe['max_windows'] == MAX_WINDOWS
          and recipe['numerical_recipe'] == fold1_numerical_recipe()
          and recipe['input_manifest'] == private.bound_input_refs()['mature_plan']
          and recipe['input_refs'] == private.bound_input_refs()
          and recipe['reference_result'] == REFERENCE_RESULT
          and recipe['expected_view_sha256'] == VIEW_SHA256
          and recipe['expected_model_contract_sha256'] == MODEL_CONTRACT_SHA256
          and recipe['expected_loss_contract_sha256'] == LOSS_CONTRACT_SHA256
          and recipe['execution_binding_path'] == str(RUNTIME_ROOT / 'EXECUTION_BINDING.json')
          and type(recipe['checkpoint_every_updates']) is int
          and recipe['checkpoint_every_updates'] == CHECKPOINT_EVERY_UPDATES
          and type(recipe['max_checkpoint_bytes']) is int
          and recipe['max_checkpoint_bytes'] == MAX_CHECKPOINT_BYTES
          and type(recipe['max_checkpoint_total_bytes']) is int
          and recipe['max_checkpoint_total_bytes'] == MAX_CHECKPOINT_TOTAL_BYTES
          and type(recipe['max_published_json_bytes']) is int
          and recipe['max_published_json_bytes'] == MAX_PUBLISHED_JSON_BYTES
          and recipe['training_enabled'] is True
          and recipe['full_epoch_training_allowed'] is True
          and recipe['test_data_used'] is False
          and recipe['predictive_value_claimed'] is False
          and recipe['qualification_weights_for_main_fit'] is False,
          'RECIPE_SCOPE')
    _reference_report()
    owners = recipe['source_owners']
    _need(isinstance(owners, Mapping) and set(owners) == required_source_paths(),
          'SOURCE_OWNER_SET')
    for name, ref in owners.items():
        checked = campaign.require_binding(
            ref, label='time-query fold1 fit source', verify_file=True)
        _need(checked['path'] == str(CANONICAL_REPO / name),
              'SOURCE_OWNER_PATH:' + name)
    for name, ref in recipe['input_refs'].items():
        campaign.require_binding(
            ref, label='time-query fold1 TRAIN input ' + name, verify_file=True)
    campaign.require_binding(
        REFERENCE_RESULT, label='frozen TRAIN reference', verify_file=True)
    return recipe


def require_fit_gate(gate: Any, recipe: Mapping[str, Any]) -> None:
    """Separate positive fit gate; the completed private probe remains closed."""
    _need(isinstance(gate, Mapping)
          and gate.get('decision') == gate.get('status') == 'ADMITTED'
          and gate.get('time_query_fold1_fit_authorized') is True
          and gate.get('fold_id') == 1
          and gate.get('fit_rows') == FIT_ROWS
          and gate.get('score_rows') == SCORE_ROWS
          and gate.get('max_optimizer_updates') == MAX_UPDATES
          and gate.get('genuine_predictor_fit_authorized') is True
          and gate.get('test_data_used') is False
          and recipe.get('run_scope') == SCOPE
          and recipe.get('admission_stage') == STAGE,
          'POLICY_GATE')
