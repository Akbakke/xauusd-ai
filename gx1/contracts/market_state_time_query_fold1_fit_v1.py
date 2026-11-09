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
STAGE = 'market_state_time_query_fold1_fit_20261107_002'
RECIPE_SCHEMA = 'gx1_native_direction_prefix_fit_launch_recipe_v1'
NUMERICAL_SCHEMA = 'gx1_market_state_time_query_fold1_fit_numerical_v1'
RESULT_SCHEMA = 'gx1_market_state_time_query_fold1_fit_result_v1'
CANONICAL_REPO = Path('/home/andre2/src/GX1_CURRENT')
RUNTIME_ROOT = Path('/home/andre2/GX1_RUNS') / STAGE.upper() / 'RUNTIME'
RECIPE_PATH = RUNTIME_ROOT.parent / 'RECIPE.json'
TA_SF_SCOPE = 'market_state_ta_sf_fold1_one_pass_fit_v1'
TA_SF_STAGE = 'market_state_ta_native_fold1_fit_20261108_001'
TA_SF_RUNTIME_ROOT = Path('/home/andre2/GX1_RUNS') / TA_SF_STAGE.upper() / 'RUNTIME'
TA_SF_RECIPE_PATH = TA_SF_RUNTIME_ROOT.parent / 'RECIPE.json'
TA_RECOVERY_SCOPE = 'market_state_ta_sf_fold1_recovery_fit_v1'
TA_RECOVERY_STAGE = 'market_state_ta_native_fold1_recovery_20261108_001'
TA_RECOVERY_RUNTIME_ROOT = Path('/home/andre2/GX1_RUNS') / TA_RECOVERY_STAGE.upper() / 'RUNTIME'
TA_RECOVERY_RECIPE_PATH = TA_RECOVERY_RUNTIME_ROOT.parent / 'RECIPE.json'
TA_M5_FIT_SCOPE = 'market_state_ta_dedicated_m5_fold1_fit_v1'
TA_M5_FIT_STAGE = 'market_state_ta_dedicated_m5_fold1_fit_20261109_001'
TA_M5_FIT_ROOT = Path('/home/andre2/GX1_RUNS') / TA_M5_FIT_STAGE.upper()
TA_M5_FIT_RUNTIME_ROOT = TA_M5_FIT_ROOT / 'RUNTIME'
TA_M5_FIT_RECIPE_PATH = TA_M5_FIT_ROOT / 'RECIPE.json'
TA_M5_FIT_MECHANICAL_ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_DEDICATED_M5_MECHANICAL_20261109_002')
TA_M5_FIT_MECHANICAL_COMMIT = '855eb7a1629da29336fab76577aa8ca069140a61'
TA_M5_FIT_MECHANICAL_REFS = {
    'ta_mechanical_recipe': {'path': str(TA_M5_FIT_MECHANICAL_ROOT / 'RECIPE.json'),
        'sha256': '957ed6790200634bd820a37fb316199bc13319e7268a224261d88dccce589339'},
    'ta_mechanical_qualification': {'path': str(TA_M5_FIT_MECHANICAL_ROOT / 'RESULT.json'),
        'sha256': '7c0c111c6db3cfcfaa4390f7074532a164c0007dedc5a1dac9f8379c29d2cf8b'},
    'ta_mechanical_terminal': {'path': str(TA_M5_FIT_MECHANICAL_ROOT / 'TERMINAL.json'),
        'sha256': '1c43c43f419eb4eccec66d5caa33ab62f74e7721ab7b67d41e3d1e7cc9f5f136'},
    'ta_mechanical_plan': {'path': str(TA_M5_FIT_MECHANICAL_ROOT / 'CAMPAIGN/CAMPAIGN_PLAN.json'),
        'sha256': '29a78e159979436c1199a224b03f2dbc9961598fa5eada1b0ab09c8893cb568b'},
}
# Fit-only orchestration bridge; the consumed mechanical source rules stay closed.
TA_M5_FIT_CHANGED_SOURCE_PATHS = frozenset({
    'gx1/contracts/market_state_time_query_fold1_fit_v1.py',
    'gx1/scripts/run_market_state_time_query_fold1_fit_window_v1.py',
    'gx1/contracts/entry_native_direction_prefix_fit_campaign_v1.py',
    'gx1/scripts/run_entry_native_direction_prefix_fit_window_v1.py',
    'gx1/scripts/materialize_market_state_time_query_fold1_fit_campaign_v1.py',
})

TA_M5_EQUAL_LOSS_FIT_SCOPE = 'market_state_ta_m5_equal_loss_fold1_fit_v1'
TA_M5_EQUAL_LOSS_FIT_STAGE = 'market_state_ta_m5_equal_loss_fold1_fit_20261109_002'
TA_M5_EQUAL_LOSS_FIT_ROOT = Path('/home/andre2/GX1_RUNS') / TA_M5_EQUAL_LOSS_FIT_STAGE.upper()
TA_M5_EQUAL_LOSS_FIT_RUNTIME_ROOT = TA_M5_EQUAL_LOSS_FIT_ROOT / 'RUNTIME'
TA_M5_EQUAL_LOSS_FIT_RECIPE_PATH = TA_M5_EQUAL_LOSS_FIT_ROOT / 'RECIPE.json'
TA_M5_EQUAL_LOSS_FIT_MAX_WINDOWS = 1
TA_M5_EQUAL_LOSS_FIT_MECHANICAL_COMMIT = '619c79da3e1478742766c07908103f04ef630f8d'
TA_M5_EQUAL_LOSS_FIT_MECHANICAL_ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_M5_EQUAL_LOSS_MECHANICAL_20261109_001')
TA_M5_EQUAL_LOSS_FIT_MECHANICAL_REFS = {
    name: {'path': str(TA_M5_EQUAL_LOSS_FIT_MECHANICAL_ROOT / path), 'sha256': sha}
    for name, path, sha in (
        ('ta_mechanical_recipe', 'RECIPE.json', '93693fff5236861e1a2b2bd2e6a86a3ee7862c2a74e9da5127f537e7fab3c742'),
        ('ta_mechanical_qualification', 'RESULT.json', '58fa9767f531f458c95e375d3f86e7c8efd956c58970bc0b976f743483277ef7'),
        ('ta_mechanical_terminal', 'TERMINAL.json', '6eac64d7b0eddb8aa330e434133350ef099edebce994347017610d9a9bb3ccac'),
        ('ta_mechanical_plan', 'CAMPAIGN/CAMPAIGN_PLAN.json', '905dd8efd3cd7fae95f2dc90457b710d673f5a23399a61831eea9063516fcd9d'),
        ('ta_mechanical_review', 'REVIEW.json', 'c2d0114a1375c82e35b15231be195f1ff2e14d1831fdd6d1ae2ffc88ff978e4d'),
        ('ta_mechanical_artifact_verification', 'ARTIFACT_VERIFICATION.json', 'e84a33f7cffe489b5fa43d0f519a29d2db8f8f5608a2dff20e41eb57ab3de18a'),
        ('ta_mechanical_campaign_receipt', 'RUNTIME/receipts/invocation-0001.json', '683261526b538366d4ba1589d625c0bea918cbb9e1805a3bd289d46255b80869'),
    )
}
TA_M5_EQUAL_LOSS_FIT_CHANGED_SOURCE_PATHS = TA_M5_FIT_CHANGED_SOURCE_PATHS

TA_RECOVERY_ORIGIN_SCHEMA = 'gx1_market_state_ta_fold1_recovery_origin_v1'
TA_RECOVERY_SOURCE_COMMIT = 'c7c1f4f5a5f21e1b8ac5255d65e1646b7ea99410'
TA_RECOVERY_PARENT_CURSOR = 4096
TA_RECOVERY_UPDATES = 256
# Only the measured Counter replacement and this recovery's orchestration differ.
TA_RECOVERY_CHANGED_SOURCE_PATHS = frozenset({
    'gx1/audit/verify_market_state_time_query_zero_step_v1.py',
    'gx1/contracts/entry_model_native_signal_v1.py',
    'gx1/contracts/entry_native_direction_prefix_fit_campaign_v1.py',
    'gx1/contracts/market_state_time_query_fold1_fit_v1.py',
    'gx1/scripts/run_market_state_time_query_fold1_fit_window_v1.py',
    'gx1/scripts/run_entry_native_direction_prefix_fit_window_v1.py',
    'gx1/scripts/materialize_market_state_time_query_fold1_fit_campaign_v1.py',
})
TA_RECOVERY_SIGNAL_SHA256 = '2390e52858e953886a60c414bd8fc080564beec2868984359ef4ebc4b3e2671b'
TA_INPUT_STAGE = 'market_state_ta_native_input_completion_20261108_001'
TA_INPUT_ROOT = Path('/home/andre2/GX1_RUNS') / TA_INPUT_STAGE.upper()
TA_MECHANICAL_ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_SF_MECHANICAL_20261108_001')
TA_NUMERICAL_SCHEMA = 'gx1_market_state_ta_sf_fold1_fit_numerical_v1'
TA_RESULT_SCHEMA = 'gx1_market_state_ta_sf_fold1_fit_result_v1'
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


def required_source_paths(*, ta_sf: bool = False) -> frozenset[str]:
    """Exact executable and lineage owners for this one candidate."""
    from gx1.contracts import entry_native_direction_research_campaign_v1 as research
    paths = frozenset(research.REQUIRED_SOURCE_PATHS) | frozenset(
        private.EXTRA_SOURCE_PATHS) | EXTRA_SOURCE_PATHS
    if ta_sf:
        from gx1.contracts import market_state_native_F_private_campaign_v1 as sf
        from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero
        paths |= zero.TA_SOURCE_PATHS | frozenset(sf.EXTRA_SOURCE_PATHS) | frozenset({
            'gx1/contracts/market_state_time_query_model_v2.py',
            'gx1/contracts/market_state_time_query_batch_v2.py',
            'gx1/contracts/market_state_time_query_loss_v2.py',
            'gx1/contracts/market_state_time_query_scorer_v2.py',
            'gx1/contracts/market_state_five_endpoint_prior_v1.py',
            'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
            'gx1/audit/verify_market_state_ta_native_input_path_v1.py',
        })
    return paths


def require_checkpoint_view(value: Any, *, recipe: Mapping[str, Any] | None = None,
                            verify_files: bool = True, ta_sf: bool = False,
                            dedicated_m5_side: bool = False) -> dict[str, Any]:
    """Require the exact matured fold-1 feature and normalization identity."""
    _need(type(dedicated_m5_side) is bool, 'TA_M5_VIEW_MODE')
    if recipe is not None:
        m5_recipe = recipe.get('run_scope') in (TA_M5_FIT_SCOPE, TA_M5_EQUAL_LOSS_FIT_SCOPE)
        _need(not dedicated_m5_side or m5_recipe, 'TA_M5_VIEW_RECIPE_MODE')
        dedicated_m5_side = m5_recipe
        if m5_recipe:
            _need(recipe.get('numerical_recipe') == ta_sf_numerical_recipe(
                dedicated_m5_side=True,
                m5_equal_observation=recipe.get('run_scope') == TA_M5_EQUAL_LOSS_FIT_SCOPE),
                'TA_M5_VIEW_NUMERICAL')
    if ta_sf or dedicated_m5_side or (recipe is not None and recipe.get("run_scope") in (TA_SF_SCOPE, TA_RECOVERY_SCOPE)):
        return _require_ta_checkpoint_view(value, recipe=recipe, verify_files=verify_files,
                                           dedicated_m5_side=dedicated_m5_side)
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
                              'qualification_sha256', 'unobserved_units_profile_sha256'},
          'CHECKPOINT_NORMALIZATION_SHAPE')
    from gx1.audit.fit_market_state_mature_fold1_units_v1 import PROFILE_SHA256
    _need(prefix['unobserved_units_profile_sha256'] == PROFILE_SHA256,
          'CHECKPOINT_UNOBSERVED_UNITS_PROFILE')
    expected = private.bound_input_refs()['fold1_normalization']
    _need(prefix['binding'] == expected, 'CHECKPOINT_FOLD1_NORMALIZATION')
    if recipe is not None:
        _need(recipe.get('run_scope') == SCOPE
              and recipe.get('expected_view_sha256') == VIEW_SHA256
              and recipe.get('input_refs', {}).get('fold1_normalization') == expected,
              'CHECKPOINT_RECIPE_VIEW')
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix_fit
    return prefix_fit._view_identity(value, verify_files=verify_files,
                                     allow_unobserved_units_profile=True)


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
    if isinstance(value, Mapping) and value.get("run_scope") == TA_M5_EQUAL_LOSS_FIT_SCOPE:
        return _require_ta_m5_equal_loss_fit_launch_recipe(value, source_repo=source_repo,
            source_commit=source_commit, verify_files=verify_files)
    if isinstance(value, Mapping) and value.get("run_scope") == TA_M5_FIT_SCOPE:
        return _require_ta_m5_fit_launch_recipe(value, source_repo=source_repo,
            source_commit=source_commit, verify_files=verify_files)
    if isinstance(value, Mapping) and value.get("run_scope") == TA_RECOVERY_SCOPE:
        return _require_ta_recovery_recipe(value, source_repo=source_repo,
            source_commit=source_commit, verify_files=verify_files)
    if isinstance(value, Mapping) and value.get("run_scope") == TA_SF_SCOPE:
        return _require_ta_launch_recipe(value, source_repo=source_repo,
            source_commit=source_commit, verify_files=verify_files)
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
    if recipe.get("run_scope") == TA_M5_EQUAL_LOSS_FIT_SCOPE:
        _need(isinstance(gate, Mapping)
              and gate.get("ta_m5_equal_loss_fold1_fit_authorized") is True
              and recipe.get("admission_stage") == TA_M5_EQUAL_LOSS_FIT_STAGE
              and gate.get("imported_origin") == recipe["imported_origin"]
              and gate.get("mechanical_review") == recipe["input_refs"]["ta_mechanical_review"]
              and gate.get("physical_power_limit_w") == 250,
              "M5_EQUAL_LOSS_FIT_POLICY")
    if recipe.get("run_scope") == TA_M5_FIT_SCOPE:
        _need(isinstance(gate, Mapping)
              and gate.get("ta_m5_fold1_fit_authorized") is True
              and recipe.get("admission_stage") == TA_M5_FIT_STAGE
              and gate.get("warmstart_origin") == recipe["warmstart_origin"],
              "TA_M5_FIT_POLICY")
    if recipe.get("run_scope") == TA_RECOVERY_SCOPE:
        _need(isinstance(gate, Mapping)
              and gate.get("ta_sf_fold1_recovery_authorized") is True
              and gate.get("recovery_origin") == recipe["recovery_origin"],
              "TA_RECOVERY_POLICY")
    if recipe.get("run_scope") in (TA_SF_SCOPE, TA_RECOVERY_SCOPE, TA_M5_FIT_SCOPE, TA_M5_EQUAL_LOSS_FIT_SCOPE):
        _need(isinstance(gate, Mapping)
              and gate.get("ta_sf_fold1_fit_authorized") is True
              and gate.get("input_audit") == recipe["input_refs"]["ta_input_audit"]
              and gate.get("mechanical_qualification") == recipe["input_refs"]["ta_mechanical_qualification"],
              "TA_POLICY_QUALIFICATION")
        recipe = {**recipe, "run_scope": SCOPE, "admission_stage": STAGE}
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


def ta_sf_numerical_recipe(*, dedicated_m5_side: bool = False,
                           m5_equal_observation: bool = False) -> dict[str, Any]:
    """Same mature one-pass optimizer; explicitly bind corrected TA S/F."""
    from gx1.contracts.market_state_time_query_native_F_model_v1 import contract as model_contract
    from gx1.contracts.market_state_time_query_native_F_objective_v1 import contract as loss_contract
    base = dict(native_direction_research_recipe(batch_size=BATCH_SIZE, max_updates=MAX_UPDATES))
    base.pop('contract_sha256')
    base.update(schema_version=TA_NUMERICAL_SCHEMA,
        research_objective_sha256=model_contract(ta_only=True, dedicated_m5_side=dedicated_m5_side)['contract_sha256'],
        native_F_loss_contract_sha256=loss_contract(ta_only=True,
            dedicated_m5_side=dedicated_m5_side,
            m5_equal_observation=m5_equal_observation)['contract_sha256'],
        future_weight=1.0)
    if m5_equal_observation:
        from gx1.contracts.market_state_time_query_native_F_objective_v1 import m5_equal_observation_profile
        base.update(m5_equal_observation=True,
                    m5_side_weighting_profile=m5_equal_observation_profile())
    return {**base, 'contract_sha256': campaign.canonical_sha256(base)}


def _ta_ref(name: str) -> dict[str, str]:
    roots = {'ta_candidate_view': TA_INPUT_ROOT / 'CANDIDATE_VIEW.json',
             'ta_input_audit': TA_INPUT_ROOT / 'RESULT.json',
             'ta_endpoint_priors': TA_INPUT_ROOT / 'ENDPOINT_PRIORS.json',
             'ta_mechanical_qualification': TA_MECHANICAL_ROOT / 'RESULT.json'}
    path = roots[name]
    return {'path': str(path), 'sha256': campaign.file_sha256(path)}


def ta_sf_bound_input_refs() -> dict[str, dict[str, str]]:
    from gx1.contracts import market_state_native_F_private_campaign_v1 as sf
    refs = sf.ta_input_refs()
    refs.update({name: _ta_ref(name) for name in (
        'ta_candidate_view', 'ta_input_audit', 'ta_endpoint_priors',
        'ta_mechanical_qualification')})
    return refs


def _require_ta_checkpoint_view(value: Any, *, recipe=None, verify_files=True,
                                dedicated_m5_side=False):
    from gx1.contracts.market_state_time_query_model_v2 import factorized_time_query_model_contract
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix_fit
    from gx1.contracts import market_state_native_F_private_campaign_v1 as sf
    from gx1.audit.fit_market_state_mature_fold1_units_v1 import PROFILE_SHA256
    _need(isinstance(value, Mapping) and set(value) == {
        'contract_sha256', 'prefix_normalization', 'research_objective_sha256'}, 'TA_VIEW_FIELDS')
    prefix = value['prefix_normalization']
    _need(value['research_objective_sha256'] == factorized_time_query_model_contract(
        ta_only=True, dedicated_m5_side=dedicated_m5_side)['contract_sha256']
        and isinstance(prefix, Mapping)
        and set(prefix) == {'binding', 'contract_sha256', 'base_sha256',
            'gc_layout_sha256', 'gc_normalization_sha256', 'qualification_sha256',
            'unobserved_units_profile_sha256', 'ta_base_normalization_sha256',
            'source_owner_profile_sha256'}
        and prefix['binding'] == sf.bound_input_refs()['fold1_normalization']
        and prefix['unobserved_units_profile_sha256'] == PROFILE_SHA256,
        'TA_VIEW_NORMALIZATION')
    if recipe is not None:
        _need(recipe.get('run_scope') in (TA_SF_SCOPE, TA_RECOVERY_SCOPE, TA_M5_FIT_SCOPE, TA_M5_EQUAL_LOSS_FIT_SCOPE)
            and recipe.get('expected_view_sha256') == value['contract_sha256'],
            'TA_CHECKPOINT_RECIPE_VIEW')
    return prefix_fit._view_identity(value, verify_files=verify_files,
        allow_unobserved_units_profile=True, allow_ta_profile=True)


def _require_ta_launch_recipe(value, *, source_repo, source_commit, verify_files,
                              historical_source_commit=None):
    """New candidate requires new physical inputs and new mechanical proof."""
    _need(verify_files, 'TA_PHYSICAL_METADATA_REQUIRED')
    fields = {'schema_version', 'source_repo', 'runtime_root', 'run_id',
        'admission_stage', 'run_scope', 'fold', 'train_passes', 'max_windows',
        'numerical_recipe', 'input_manifest', 'input_refs', 'reference_result',
        'expected_view_sha256', 'expected_model_contract_sha256',
        'expected_loss_contract_sha256', 'source_owners', 'execution_binding_path',
        'checkpoint_every_updates', 'max_checkpoint_bytes', 'max_checkpoint_total_bytes',
        'max_published_json_bytes', 'training_enabled', 'full_epoch_training_allowed',
        'test_data_used', 'predictive_value_claimed', 'qualification_weights_for_main_fit'}
    recipe = _identity(value, fields)
    refs = ta_sf_bound_input_refs()
    numerical = ta_sf_numerical_recipe()
    _need(source_repo == CANONICAL_REPO and isinstance(source_commit, str)
        and campaign._COMMIT.fullmatch(source_commit) is not None
        and recipe['source_repo'] == str(CANONICAL_REPO)
        and recipe['runtime_root'] == str(TA_SF_RUNTIME_ROOT)
        and recipe['run_id'] == recipe['admission_stage'] == TA_SF_STAGE
        and recipe['run_scope'] == TA_SF_SCOPE and recipe['fold'] == FOLD
        and type(recipe['train_passes']) is int and recipe['train_passes'] == 1
        and type(recipe['max_windows']) is int and recipe['max_windows'] == MAX_WINDOWS
        and recipe['numerical_recipe'] == numerical and recipe['input_refs'] == refs
        and recipe['input_manifest'] == refs['mature_plan']
        and recipe['reference_result'] == refs['ta_endpoint_priors']
        and recipe['expected_model_contract_sha256'] == numerical['research_objective_sha256']
        and recipe['expected_loss_contract_sha256'] == numerical['native_F_loss_contract_sha256']
        and recipe['execution_binding_path'] == str(TA_SF_RUNTIME_ROOT / 'EXECUTION_BINDING.json')
        and recipe['checkpoint_every_updates'] == CHECKPOINT_EVERY_UPDATES
        and recipe['max_checkpoint_bytes'] == MAX_CHECKPOINT_BYTES
        and recipe['max_checkpoint_total_bytes'] == MAX_CHECKPOINT_TOTAL_BYTES
        and recipe['max_published_json_bytes'] == MAX_PUBLISHED_JSON_BYTES
        and recipe['training_enabled'] is True and recipe['full_epoch_training_allowed'] is True
        and recipe['test_data_used'] is False and recipe['predictive_value_claimed'] is False
        and recipe['qualification_weights_for_main_fit'] is False, 'TA_RECIPE_SCOPE')
    for name, ref in refs.items():
        campaign.require_binding(ref, label='TA S/F input ' + name, verify_file=True)
    def read(name):
        ref = refs[name]
        return campaign.read_bound_json(Path(ref['path']), ref['sha256'])
    view, audit, mechanical, priors = (read(name) for name in (
        'ta_candidate_view', 'ta_input_audit', 'ta_mechanical_qualification', 'ta_endpoint_priors'))
    _need(view['contract_sha256'] == recipe['expected_view_sha256']
        and view.get('full_ta_feature_profile') is True
        and audit.get('status') == 'COMPLETE' and audit.get('test_data_used') is False
        and audit.get('candidate_view') == refs['ta_candidate_view']
        and audit['input_audit']['candidate_view_sha256'] == view['contract_sha256']
        and audit['input_audit']['gather']['fit_rows'] == FIT_ROWS
        and audit['input_audit']['gather']['score_rows'] == SCORE_ROWS
        and audit['input_audit']['gather']['all_fields_all_window_positions_bit_identical'] is True
        and mechanical.get('decision') == 'PASS_TA_SF_MECHANICAL'
        and mechanical.get('candidate_view') == refs['ta_candidate_view']
        and mechanical.get('input_audit') == refs['ta_input_audit']
        and mechanical.get('bit_exact_resume') is True
        and mechanical.get('fresh_initial_baseline') is True
        and mechanical.get('test_data_used') is False
        and mechanical.get('numerical_recipe') == numerical,
        'TA_NEW_QUALIFICATION')
    from gx1.contracts.market_state_time_query_native_F_objective_v1 import require_endpoint_priors
    require_endpoint_priors(priors, fold_id=1, ta_only=True)
    owners = recipe['source_owners']
    _need(isinstance(owners, Mapping) and set(owners) == required_source_paths(ta_sf=True),
        'TA_SOURCE_OWNER_SET')
    _require_ta_source_owners(owners, historical_source_commit=historical_source_commit)
    _need(all(mechanical.get('source_owners', {}).get(name) == ref
        for name, ref in owners.items()), 'TA_QUALIFIED_SOURCE')
    audit_owners = audit.get('source_owners', {})
    _need(isinstance(audit_owners, dict) and all(
        audit_owners[name] == ref['sha256']
        for name, ref in owners.items() if name in audit_owners),
        'TA_AUDITED_INPUT_SOURCE')
    terminal_path = TA_MECHANICAL_ROOT / 'TERMINAL.json'
    terminal = campaign.read_bound_json(terminal_path, campaign.file_sha256(terminal_path))
    _need(terminal.get('status') == 'COMPLETE' and terminal.get('exit_code') == 0
        and terminal.get('result') == refs['ta_mechanical_qualification']
        and terminal.get('no_relaunch_under_this_plan') is True
        and terminal.get('test_data_used') is False, 'TA_MECHANICAL_TERMINAL')
    return recipe


def _require_ta_source_owners(owners, *, historical_source_commit=None):
    """Validate old qualification bytes without pretending they are current."""
    for name, ref in owners.items():
        checked = campaign.require_binding(ref, label='TA S/F source',
            verify_file=historical_source_commit is None)
        _need(checked['path'] == str(CANONICAL_REPO / name), 'TA_SOURCE_PATH:' + name)
        if historical_source_commit is not None:
            _need(historical_source_commit == TA_RECOVERY_SOURCE_COMMIT,
                  'TA_RECOVERY_ORIGINAL_SOURCE')
            if campaign.file_sha256(Path(checked['path'])) != checked['sha256']:
                _need(name in TA_RECOVERY_CHANGED_SOURCE_PATHS,
                      'TA_RECOVERY_UNQUALIFIED_SOURCE:' + name)
                import subprocess
                old = subprocess.run(['git', 'show', historical_source_commit + ':' + name],
                    cwd=CANONICAL_REPO, check=True, capture_output=True).stdout
                _need(hashlib.sha256(old).hexdigest() == checked['sha256'],
                      'TA_RECOVERY_HISTORICAL_SOURCE:' + name)


def read_ta_recovery_origin(value, *, expected_view_sha256):
    """Read the one consumed origin; no checkpoint deserialization or new cursor."""
    paths = {
        'original_recipe': TA_SF_RECIPE_PATH,
        'original_plan': TA_SF_RUNTIME_ROOT.parent / 'CAMPAIGN/CAMPAIGN_PLAN.json',
        'interruption_terminal': TA_SF_RUNTIME_ROOT.parent / 'OPERATOR_INTERRUPTION_TERMINAL.json',
        'initial_state': TA_SF_RUNTIME_ROOT / 'prefix-fit/states/INITIAL_FOLD1_STATE.pt',
        'initial_measurement': TA_SF_RUNTIME_ROOT / 'prefix-fit/window-0001/INITIAL_TRAIN.json',
        'checkpoint': TA_SF_RUNTIME_ROOT / 'prefix-fit/states/w0001_p004096.pt',
    }
    _need(isinstance(value, Mapping)
          and set(value) == set(paths) | {'schema_version', 'source_commit'}
          and value['schema_version'] == TA_RECOVERY_ORIGIN_SCHEMA
          and value['source_commit'] == TA_RECOVERY_SOURCE_COMMIT,
          'TA_RECOVERY_ORIGIN_FIELDS')
    for name, path in paths.items():
        ref = campaign.require_binding(value[name], label='TA recovery ' + name,
                                       verify_file=True)
        _need(ref['path'] == str(path), 'TA_RECOVERY_ORIGIN_PATH:' + name)
    def read(name):
        ref = value[name]
        return campaign.read_bound_json(Path(ref['path']), ref['sha256'])
    original, plan, terminal, initial = (
        read(name) for name in ('original_recipe', 'original_plan',
                               'interruption_terminal', 'initial_measurement'))
    _need(plan.get('source_commit') == value['source_commit']
          and plan.get('source_repo') == str(CANONICAL_REPO)
          and plan.get('runtime_root') == str(TA_SF_RUNTIME_ROOT)
          and plan.get('recipe') == value['original_recipe']
          and plan.get('plan_sha256') == campaign.canonical_sha256({
              key: item for key, item in plan.items() if key != 'plan_sha256'}),
          'TA_RECOVERY_ORIGINAL_PLAN')
    _need(terminal.get('schema_version') == 'gx1_operator_interruption_terminal_v1'
          and terminal.get('status') == 'ABORTED_CONSUMED'
          and terminal.get('source_commit_at_interruption') == value['source_commit']
          and terminal.get('source_changed_during_trainer') is False
          and terminal.get('old_plan_relaunch_allowed') is False
          and terminal.get('originals_modified') is False
          and terminal.get('test_data_used') is False
          and terminal.get('full_fit_complete') is False
          and terminal.get('final_train_measurement_complete') is False
          and terminal.get('later_train_score_complete') is False
          and terminal.get('last_preserved_parent_cursor') == TA_RECOVERY_PARENT_CURSOR
          and terminal.get('last_preserved_optimizer_updates') == TA_RECOVERY_UPDATES
          and terminal.get('initial_measurement_optimizer_updates') == 0,
          'TA_RECOVERY_INTERRUPTION')
    preserved = terminal.get('preserved_checkpoints')
    _need(isinstance(preserved, list) and len(preserved) == 2,
          'TA_RECOVERY_PRESERVED_STATES')
    metadata = {}
    for name, cursor, updates in (
            ('initial_state', 0, 0),
            ('checkpoint', TA_RECOVERY_PARENT_CURSOR, TA_RECOVERY_UPDATES)):
        matches = [item for item in preserved if item.get('path') == value[name]['path']]
        _need(len(matches) == 1, 'TA_RECOVERY_STATE_REFERENCE:' + name)
        item = matches[0]
        _need({key: item.get(key) for key in ('path', 'sha256')} == value[name]
              and item.get('schema_version') == 'gx1_market_state_ta_sf_fold1_fit_state_v1'
              and item.get('source_commit') == value['source_commit']
              and type(item.get('cursor')) is int and item['cursor'] == cursor
              and type(item.get('optimizer_updates')) is int and item['optimizer_updates'] == updates
              and type(item.get('skipped_batches')) is int and item['skipped_batches'] == 0
              and item.get('feature_view_sha256') == expected_view_sha256
              and item.get('state_and_component_digest_checks') == 'PASS',
              'TA_RECOVERY_STATE_METADATA:' + name)
        for key in ('contract_sha256', 'model_state_sha256'):
            campaign._sha(item.get(key), 'TA recovery ' + key)
        metadata[name] = item
    _need({key: terminal.get('initial_measurement', {}).get(key)
           for key in ('path', 'sha256')} == value['initial_measurement'],
          'TA_RECOVERY_INITIAL_REFERENCE')
    from gx1.contracts.market_state_time_query_native_F_objective_v1 import _seal as endpoint_seal
    measured = initial.get('measurement', {})
    _need(initial.get('schema_version') == 'gx1_market_state_ta_sf_fold1_evaluation_v1'
          and initial.get('fold_id') == 1 and initial.get('optimizer_updates') == 0
          and initial.get('test_data_used') is False
          and initial.get('predictive_value_claimed') is False
          and initial.get('evaluated_checkpoint') == value['initial_state']
          and initial.get('evaluated_model_state_sha256') == metadata['initial_state']['model_state_sha256']
          and measured.get('schema_version') == 'gx1_native_S_F_endpoint_measurement_v1'
          and measured.get('partition') == 'fit' and measured.get('fold_id') == 1
          and measured.get('rows') == FIT_ROWS and measured.get('batches') == MAX_UPDATES
          and measured.get('feature_view_sha256') == expected_view_sha256
          and measured.get('model_contract_sha256') == original.get('expected_model_contract_sha256')
          and measured.get('loss_contract_sha256') == original.get('expected_loss_contract_sha256')
          and measured == endpoint_seal({
              key: item for key, item in measured.items() if key != 'contract_sha256'}),
          'TA_RECOVERY_COMPLETE_INITIAL')
    return {'origin': dict(value), 'original_recipe': original, 'original_plan': plan,
            'initial_metadata': metadata['initial_state'],
            'checkpoint_metadata': metadata['checkpoint'], 'initial_payload': initial}


def _require_ta_recovery_recipe(value, *, source_repo, source_commit, verify_files):
    """Reuse qualified evidence only across the declared source bridge."""
    _need(verify_files, 'TA_RECOVERY_PHYSICAL_METADATA_REQUIRED')
    fields = set(value) - {'recipe_sha256'} if isinstance(value, Mapping) else set()
    recipe = _identity(value, fields)
    _need(source_repo == CANONICAL_REPO
          and isinstance(source_commit, str)
          and campaign._COMMIT.fullmatch(source_commit) is not None
          and source_commit != TA_RECOVERY_SOURCE_COMMIT,
          'TA_RECOVERY_CURRENT_SOURCE')
    origin = read_ta_recovery_origin(recipe.get('recovery_origin'),
                                    expected_view_sha256=recipe.get('expected_view_sha256'))
    original = _require_ta_launch_recipe(origin['original_recipe'],
        source_repo=source_repo, source_commit=TA_RECOVERY_SOURCE_COMMIT,
        verify_files=True, historical_source_commit=TA_RECOVERY_SOURCE_COMMIT)
    expected = {**original, 'run_id': TA_RECOVERY_STAGE,
        'admission_stage': TA_RECOVERY_STAGE, 'run_scope': TA_RECOVERY_SCOPE,
        'runtime_root': str(TA_RECOVERY_RUNTIME_ROOT),
        'execution_binding_path': str(TA_RECOVERY_RUNTIME_ROOT / 'EXECUTION_BINDING.json'),
        'source_owners': recipe['source_owners'], 'recovery_origin': recipe['recovery_origin']}
    expected.pop('recipe_sha256')
    _need(recipe == {**expected, 'recipe_sha256': campaign.canonical_sha256(expected)},
          'TA_RECOVERY_RECIPE_SCOPE')
    owners = recipe['source_owners']
    _need(isinstance(owners, Mapping) and set(owners) == set(original['source_owners']),
          'TA_RECOVERY_SOURCE_OWNER_SET')
    _require_ta_source_owners(owners)
    changed = {name for name, ref in owners.items() if ref != original['source_owners'][name]}
    _need(changed <= TA_RECOVERY_CHANGED_SOURCE_PATHS
          and owners['gx1/contracts/entry_model_native_signal_v1.py']['sha256']
              == TA_RECOVERY_SIGNAL_SHA256, 'TA_RECOVERY_SOURCE_BRIDGE')
    return recipe


def ta_m5_fit_bound_input_refs() -> dict[str, dict[str, str]]:
    """The completed mechanical inputs plus its four immutable qualification refs."""
    from gx1.contracts import market_state_native_F_private_campaign_v1 as sf
    qualified = sf._read(TA_M5_FIT_MECHANICAL_REFS['ta_mechanical_recipe'],
                         'TA M5 qualified mechanical recipe')
    return {**qualified['input_refs'],
            **{name: dict(ref) for name, ref in TA_M5_FIT_MECHANICAL_REFS.items()}}


def _ta_m5_fit_qualification(refs):
    from gx1.contracts import market_state_native_F_private_campaign_v1 as sf
    qualified, report, terminal, plan = (sf._read(refs[name], name) for name in (
        'ta_mechanical_recipe', 'ta_mechanical_qualification',
        'ta_mechanical_terminal', 'ta_mechanical_plan'))
    numerical = ta_sf_numerical_recipe(dedicated_m5_side=True)
    view, _, _, origin = sf._require_ta_m5_input_proof(refs)
    _need(qualified.get('recipe_sha256') == campaign.canonical_sha256({
              k: v for k, v in qualified.items() if k != 'recipe_sha256'})
          and qualified.get('schema_version') == sf.TA_M5_RECIPE_SCHEMA
          and qualified.get('run_scope') == sf.TA_M5_SCOPE
          and qualified.get('runtime_root') == str(TA_M5_FIT_MECHANICAL_ROOT / 'RUNTIME')
          and qualified.get('numerical_recipe') == numerical
          and qualified.get('warmstart_origin') == origin
          and qualified.get('input_refs') == {
              k: v for k, v in refs.items() if k not in TA_M5_FIT_MECHANICAL_REFS}
          and report.get('schema_version') == sf.TA_M5_RESULT_SCHEMA
          and report.get('decision') == 'PASS_TA_M5_MECHANICAL'
          and report.get('recipe') == refs['ta_mechanical_recipe']
          and report.get('source_owners') == qualified.get('source_owners')
          and report.get('source_unchanged') is True
          and report.get('numerical_recipe') == numerical
          and report.get('warmstart_origin') == origin
          and report.get('candidate_view') == refs['ta_candidate_view']
          and report.get('input_audit') == refs['ta_input_audit']
          and report.get('bit_exact_resume') is True
          and report.get('fresh_initial_baseline') is True
          and report.get('full_TRAIN_initialbaseline_qualified') is False
          and report.get('qualification_weights_for_main_fit') is False
          and report.get('private_weights_for_main_fit') is False
          and report.get('native_fit_authorized') is False
          and report.get('physical_optimizer_updates') == 4
          and report.get('test_data_used') is False and report.get('error') is None
          and plan.get('source_commit') == TA_M5_FIT_MECHANICAL_COMMIT
          and plan.get('recipe') == refs['ta_mechanical_recipe']
          and terminal.get('status') == 'COMPLETE' and terminal.get('exit_code') == 0
          and terminal.get('recipe') == refs['ta_mechanical_recipe']
          and terminal.get('result') == refs['ta_mechanical_qualification']
          and terminal.get('no_relaunch_under_this_plan') is True
          and terminal.get('test_data_used') is False, 'TA_M5_FIT_QUALIFICATION')
    imported = report.get('legacy_import', {})
    _need(imported.get('strict_full_restore') is True
          and imported.get('source_commit') == origin['source_commit']
          and imported.get('original_checkpoint') == origin['original_checkpoint']
          and imported.get('model_state_sha256') == origin['model_state_sha256']
          and imported.get('restored_progress') == {key: origin[key] for key in (
              'cursor', 'optimizer_updates', 'skipped_batches')}
          and imported.get('optimizer_history_transferred_to_candidate') is False,
          'TA_M5_FIT_ORIGINAL_RESTORE')
    migrations = report.get('weight_migrations')
    _need(isinstance(migrations, list) and len(migrations) == 3
          and all(m == migrations[0] for m in migrations), 'TA_M5_FIT_INITIAL_PARITY')
    initial = migrations[0]
    initial_sha = campaign._sha(initial.get('model_state_sha256'), 'TA M5 initial model')
    _need(initial.get('source_model_state_sha256') == origin['model_state_sha256']
          and initial.get('model_contract_sha256') == numerical['research_objective_sha256']
          and initial.get('cursor') == initial.get('optimizer_updates') == 0
          and initial.get('optimizer_policy') == 'fresh_AdamW_empty_history_no_inherited_updates'
          and initial.get('fresh_actual_initial_measurement_required') is True,
          'TA_M5_FIT_INITIAL_STATE')
    parity = report.get('baseline_output_parity')
    _need(isinstance(parity, list) and len(parity) == 2 and all(
        item.get('atol') == sf.TA_M5_PARITY_ATOL
        and item.get('rtol') == sf.TA_M5_PARITY_RTOL
        and item.get('extra_native_forwards') == 0
        and all(item.get('tasks', {}).get(task, {}).get('m5_side_within_fp32_tolerance') is True
            and item['tasks'][task].get('unchanged_coordinates_bit_equal') is True
            for task in ('state', 'future')) for item in parity), 'TA_M5_FIT_OUTPUT_PARITY')
    return view, qualified, origin, initial_sha


def _require_ta_m5_fit_sources(owners, qualified, *, source_commit):
    """Permit only this fit's five named orchestration changes after mechanical002."""
    import subprocess
    from gx1.contracts import market_state_native_F_private_campaign_v1 as sf
    paths = required_source_paths(ta_sf=True)
    _need(isinstance(owners, Mapping) and set(owners) == paths
          and isinstance(qualified, Mapping) and set(qualified) == paths
          and TA_M5_FIT_CHANGED_SOURCE_PATHS <= paths, 'TA_M5_FIT_SOURCE_SET')
    _need(subprocess.run(['git', 'merge-base', '--is-ancestor',
        TA_M5_FIT_MECHANICAL_COMMIT, source_commit], cwd=CANONICAL_REPO,
        capture_output=True).returncode == 0, 'TA_M5_FIT_SOURCE_ANCESTRY')
    for name, ref in owners.items():
        current = campaign.require_binding(ref, label='TA M5 fit current source', verify_file=True)
        old = campaign.require_binding(qualified[name], label='TA M5 qualified source', verify_file=False)
        _need(current['path'] == old['path'] == str(CANONICAL_REPO / name),
              'TA_M5_FIT_SOURCE_PATH:' + name)
        if name in TA_M5_FIT_CHANGED_SOURCE_PATHS:
            _need(old['sha256'] == sf._ta_m5_git_source_sha(TA_M5_FIT_MECHANICAL_COMMIT, name),
                  'TA_M5_FIT_HISTORICAL_SOURCE:' + name)
        else:
            _need(current == old, 'TA_M5_FIT_UNQUALIFIED_SOURCE:' + name)


def _ta_m5_fit_recipe_fields(refs, view, origin, initial_sha):
    numerical = ta_sf_numerical_recipe(dedicated_m5_side=True)
    return {
        'schema_version': RECIPE_SCHEMA, 'source_repo': str(CANONICAL_REPO),
        'runtime_root': str(TA_M5_FIT_RUNTIME_ROOT), 'run_id': TA_M5_FIT_STAGE,
        'admission_stage': TA_M5_FIT_STAGE, 'run_scope': TA_M5_FIT_SCOPE,
        'fold': dict(FOLD), 'train_passes': 1, 'max_windows': MAX_WINDOWS,
        'numerical_recipe': numerical, 'input_manifest': refs['mature_plan'],
        'input_refs': refs, 'reference_result': refs['ta_endpoint_priors'],
        'expected_view_sha256': view['contract_sha256'],
        'expected_model_contract_sha256': numerical['research_objective_sha256'],
        'expected_loss_contract_sha256': numerical['native_F_loss_contract_sha256'],
        'expected_initial_model_state_sha256': initial_sha,
        'mechanical_source_commit': TA_M5_FIT_MECHANICAL_COMMIT,
        'warmstart_origin': origin,
        'execution_binding_path': str(TA_M5_FIT_RUNTIME_ROOT / 'EXECUTION_BINDING.json'),
        'checkpoint_every_updates': CHECKPOINT_EVERY_UPDATES,
        'max_checkpoint_bytes': MAX_CHECKPOINT_BYTES,
        'max_checkpoint_total_bytes': MAX_CHECKPOINT_TOTAL_BYTES,
        'max_published_json_bytes': MAX_PUBLISHED_JSON_BYTES,
        'training_enabled': True, 'full_epoch_training_allowed': True,
        'test_data_used': False, 'predictive_value_claimed': False,
        'qualification_weights_for_main_fit': False,
    }


def build_ta_m5_fit_launch_recipe(*, source_commit: str) -> dict[str, Any]:
    """Build metadata only; original FINAL weights, fresh optimizer and full baseline."""
    refs = ta_m5_fit_bound_input_refs()
    view, _, origin, initial_sha = _ta_m5_fit_qualification(refs)
    value = _ta_m5_fit_recipe_fields(refs, view, origin, initial_sha)
    value['source_owners'] = {name: {'path': str(CANONICAL_REPO / name),
        'sha256': campaign.file_sha256(CANONICAL_REPO / name)}
        for name in sorted(required_source_paths(ta_sf=True))}
    value['recipe_sha256'] = campaign.canonical_sha256(value)
    return _require_ta_m5_fit_launch_recipe(value, source_repo=CANONICAL_REPO,
        source_commit=source_commit, verify_files=True)


def _require_ta_m5_fit_launch_recipe(value, *, source_repo, source_commit, verify_files):
    _need(verify_files, 'TA_M5_FIT_PHYSICAL_METADATA_REQUIRED')
    refs = ta_m5_fit_bound_input_refs()
    view, qualified, origin, initial_sha = _ta_m5_fit_qualification(refs)
    expected = _ta_m5_fit_recipe_fields(refs, view, origin, initial_sha)
    recipe = _identity(value, set(expected) | {'source_owners'})
    _need(source_repo == CANONICAL_REPO and isinstance(source_commit, str)
          and campaign._COMMIT.fullmatch(source_commit) is not None
          and all(recipe[key] == item and type(recipe[key]) is type(item)
                  for key, item in expected.items()), 'TA_M5_FIT_RECIPE_SCOPE')
    _require_ta_m5_fit_sources(recipe['source_owners'], qualified['source_owners'],
                              source_commit=source_commit)
    for name, ref in refs.items():
        campaign.require_binding(ref, label='TA M5 fit input ' + name, verify_file=True)
    return recipe


def _ta_m5_equal_loss_fit_qualification():
    """Reuse the completed native zero-state rebind and exact-resume proof."""
    from gx1.contracts import market_state_native_F_private_campaign_v1 as sf
    expected, _, _ = sf._ta_m5_equal_loss_expected()
    refs = {**expected['input_refs'],
            **{name: dict(ref) for name, ref in TA_M5_EQUAL_LOSS_FIT_MECHANICAL_REFS.items()}}
    qualified, report, terminal, plan, review, checked, receipt = (
        sf._read(refs[name], name) for name in TA_M5_EQUAL_LOSS_FIT_MECHANICAL_REFS)
    qualified = sf._identity(qualified, set(expected) | {'source_owners'},
        sf.TA_M5_EQUAL_LOSS_RECIPE_SCHEMA, 'recipe_sha256')
    _need(all(qualified[k] == value and type(qualified[k]) is type(value)
              for k, value in expected.items()), 'M5_EQUAL_LOSS_FIT_MECHANICAL_RECIPE')
    origin, numerical = expected['imported_origin'], expected['numerical_recipe']
    _need(report.get('schema_version') == sf.TA_M5_EQUAL_LOSS_RESULT_SCHEMA
          and report.get('decision') == 'PASS_TA_M5_EQUAL_LOSS_MECHANICAL'
          and report.get('recipe') == refs['ta_mechanical_recipe']
          and report.get('source_owners') == qualified['source_owners']
          and report.get('source_unchanged') is True
          and report.get('numerical_recipe') == numerical
          and report.get('imported_origin') == origin
          and report.get('candidate_view') == refs['ta_candidate_view']
          and report.get('input_audit') == refs['ta_input_audit']
          and report.get('m5_equal_observation') is True
          and report.get('bit_exact_resume') is True
          and report.get('fresh_initial_baseline') is True
          and report.get('full_TRAIN_initialbaseline_qualified') is False
          and report.get('qualification_weights_for_main_fit') is False
          and report.get('private_weights_for_main_fit') is False
          and report.get('native_fit_authorized') is False
          and report.get('physical_optimizer_updates') == 4
          and report.get('parent_cursor') == 32
          and report.get('parent_order_rows') == FIT_ROWS
          and report.get('parent_order') == list(range(32))
          and report.get('parent_order_sha256') == origin['parent_order_sha256']
          and report.get('test_data_used') is False and report.get('error') is None,
          'M5_EQUAL_LOSS_FIT_MECHANICAL_RESULT')
    _need(plan.get('source_commit') == TA_M5_EQUAL_LOSS_FIT_MECHANICAL_COMMIT
          and plan.get('recipe') == refs['ta_mechanical_recipe']
          and terminal.get('status') == 'COMPLETE' and terminal.get('exit_code') == 0
          and terminal.get('recipe') == refs['ta_mechanical_recipe']
          and terminal.get('result') == refs['ta_mechanical_qualification']
          and terminal.get('no_relaunch_under_this_plan') is True
          and terminal.get('test_data_used') is False
          and receipt.get('schema_version') == 'gx1_local_random_access_invocation_receipt_v2'
          and receipt.get('plan_sha256') == plan.get('plan_sha256')
          and receipt.get('invocation_number') == 1 and receipt.get('outcome') == 'COMPLETE'
          and receipt.get('guard_decision') == 'PASS'
          and receipt.get('trainer_guard_exit_code') == receipt.get('progress_observer_exit_code') == 0
          and receipt.get('test_data_used') is False, 'M5_EQUAL_LOSS_FIT_TERMINAL')
    _need(review.get('status') == 'NATIVE_EQUAL_LOSS_REBINDING_AND_RESUME_QUALIFIED_CONSUMED'
          and review.get('native_source_commit') == TA_M5_EQUAL_LOSS_FIT_MECHANICAL_COMMIT
          and review.get('imported_checkpoint') == origin['imported_checkpoint']
          and review.get('initial_model_state_sha256') == origin['model_state_sha256']
          and review.get('result') == checked.get('result') == refs['ta_mechanical_qualification']
          and review.get('terminal') == checked.get('terminal') == refs['ta_mechanical_terminal']
          and review.get('campaign_receipt') == checked.get('campaign_receipt')
              == refs['ta_mechanical_campaign_receipt']
          and review.get('artifact_verification') == refs['ta_mechanical_artifact_verification']
          and review.get('model_buffers_optimizer_rng_modes_order_preserved_at_rebinding') is True
          and review.get('bit_exact_resume') is True
          and review.get('both_M5_heads_receive_gradient') is True
          and review.get('physical_power_limit_w') == 250
          and review.get('full_TRAIN_initialbaseline_qualified') is False
          and review.get('qualified_initial_rows') == 32
          and review.get('private_updated_weights_for_main_fit') is False
          and checked.get('status') == 'PASS_ACTUAL_SAVED_STATE_AND_CONTROLLER'
          and checked.get('all_model_buffers_optimizer_rng_modes_order_bitexact') is True
          and checked.get('actual_final_equals_native_reference_full_state') is True
          and checked.get('original_checkpoint_unchanged') is True
          and checked.get('original_import_for_main_fit') is True
          and checked.get('private_updated_weights_for_main_fit') is False
          and checked.get('actual_state_cursors') == [0, 0, 16, 32]
          and review.get('test_data_used') is checked.get('test_data_used') is False,
          'M5_EQUAL_LOSS_FIT_REVIEW')
    proofs = report.get('loss_rebindings')
    _need(isinstance(proofs, list) and len(proofs) == 3
          and all(p == proofs[0] for p in proofs), 'M5_EQUAL_LOSS_FIT_REBINDING_PARITY')
    proof = proofs[0]
    _need(proof.get('schema_version') == 'gx1_ta_m5_equal_observation_initial_rebinding_v1'
          and proof.get('model_state_sha256') == origin['model_state_sha256']
          and proof.get('rng_state_sha256') == origin['rng_state_sha256']
          and proof.get('parent_order_sha256') == origin['parent_order_sha256']
          and proof.get('source_commit') == origin['source_commit']
          and proof.get('new_source_commit') == TA_M5_EQUAL_LOSS_FIT_MECHANICAL_COMMIT
          and proof.get('old_numerical_recipe_sha256') == ta_sf_numerical_recipe(
              dedicated_m5_side=True)['contract_sha256']
          and proof.get('new_numerical_recipe_sha256') == numerical['contract_sha256']
          and proof.get('state_sha256') == report.get('initial_state_sha256')
          and all(type(proof.get(k)) is int and proof[k] == 0 for k in (
              'cursor', 'optimizer_updates', 'skipped_batches', 'optimizer_steps_performed'))
          and proof.get('parameter_identity_unchanged') is proof.get('buffer_identity_unchanged') is True
          and proof.get('changed_state_fields') == ['binding', 'source_commit', 'contract_sha256']
          and proof.get('changed_binding_fields') == ['numerical_recipe', 'contract_sha256'],
          'M5_EQUAL_LOSS_FIT_REBINDING_STATE')
    return refs, qualified


def _require_ta_m5_equal_loss_fit_sources(owners, qualified, *, source_commit):
    """Only five full-fit orchestration owners may differ from native qualification."""
    import subprocess
    from gx1.contracts import market_state_native_F_private_campaign_v1 as sf
    paths = required_source_paths(ta_sf=True)
    _need(isinstance(owners, Mapping) and isinstance(qualified, Mapping)
          and set(owners) == set(qualified) == paths
          and len(paths) == 85 and TA_M5_EQUAL_LOSS_FIT_CHANGED_SOURCE_PATHS <= paths,
          'M5_EQUAL_LOSS_FIT_SOURCE_SET')
    _need(subprocess.run(['git', 'merge-base', '--is-ancestor',
        TA_M5_EQUAL_LOSS_FIT_MECHANICAL_COMMIT, source_commit], cwd=CANONICAL_REPO,
        capture_output=True).returncode == 0, 'M5_EQUAL_LOSS_FIT_SOURCE_ANCESTRY')
    for name, ref in owners.items():
        current = campaign.require_binding(ref, label='equal-loss fit current source', verify_file=True)
        old = campaign.require_binding(qualified[name], label='equal-loss qualified source', verify_file=False)
        _need(current['path'] == old['path'] == str(CANONICAL_REPO / name),
              'M5_EQUAL_LOSS_FIT_SOURCE_PATH:' + name)
        if name in TA_M5_EQUAL_LOSS_FIT_CHANGED_SOURCE_PATHS:
            _need(old['sha256'] == sf._ta_m5_git_source_sha(TA_M5_EQUAL_LOSS_FIT_MECHANICAL_COMMIT, name),
                  'M5_EQUAL_LOSS_FIT_HISTORICAL_SOURCE:' + name)
        else:
            _need(current == old, 'M5_EQUAL_LOSS_FIT_UNQUALIFIED_SOURCE:' + name)


def _ta_m5_equal_loss_fit_recipe_fields(refs, qualified):
    numerical = ta_sf_numerical_recipe(dedicated_m5_side=True, m5_equal_observation=True)
    return {
        'schema_version': RECIPE_SCHEMA, 'source_repo': str(CANONICAL_REPO),
        'runtime_root': str(TA_M5_EQUAL_LOSS_FIT_RUNTIME_ROOT),
        'run_id': TA_M5_EQUAL_LOSS_FIT_STAGE, 'admission_stage': TA_M5_EQUAL_LOSS_FIT_STAGE,
        'run_scope': TA_M5_EQUAL_LOSS_FIT_SCOPE,
        'fold': dict(FOLD), 'train_passes': 1, 'max_windows': TA_M5_EQUAL_LOSS_FIT_MAX_WINDOWS,
        'numerical_recipe': numerical, 'input_manifest': refs['mature_plan'],
        'input_refs': refs, 'reference_result': refs['ta_endpoint_priors'],
        'expected_view_sha256': qualified['expected_view_sha256'],
        'expected_model_contract_sha256': numerical['research_objective_sha256'],
        'expected_loss_contract_sha256': numerical['native_F_loss_contract_sha256'],
        'expected_initial_model_state_sha256': qualified['expected_initial_model_state_sha256'],
        'mechanical_source_commit': TA_M5_EQUAL_LOSS_FIT_MECHANICAL_COMMIT,
        'imported_origin': qualified['imported_origin'],
        'measurement_scope': {'initial_fit_rows': FIT_ROWS, 'final_fit_rows': FIT_ROWS,
                              'later_train_rows': SCORE_ROWS, 'matched_initial_final': True},
        'execution_binding_path': str(TA_M5_EQUAL_LOSS_FIT_RUNTIME_ROOT / 'EXECUTION_BINDING.json'),
        'checkpoint_every_updates': CHECKPOINT_EVERY_UPDATES,
        'max_checkpoint_bytes': MAX_CHECKPOINT_BYTES,
        'max_checkpoint_total_bytes': MAX_CHECKPOINT_TOTAL_BYTES,
        'max_published_json_bytes': MAX_PUBLISHED_JSON_BYTES,
        'training_enabled': True, 'full_epoch_training_allowed': True,
        'test_data_used': False, 'predictive_value_claimed': False,
        'qualification_weights_for_main_fit': False,
    }


def build_ta_m5_equal_loss_fit_launch_recipe(*, source_commit: str) -> dict[str, Any]:
    """Build one full-fold admission from the original zero-update import only."""
    refs, qualified = _ta_m5_equal_loss_fit_qualification()
    value = _ta_m5_equal_loss_fit_recipe_fields(refs, qualified)
    value['source_owners'] = {name: {'path': str(CANONICAL_REPO / name),
        'sha256': campaign.file_sha256(CANONICAL_REPO / name)}
        for name in sorted(required_source_paths(ta_sf=True))}
    value['recipe_sha256'] = campaign.canonical_sha256(value)
    return _require_ta_m5_equal_loss_fit_launch_recipe(value, source_repo=CANONICAL_REPO,
        source_commit=source_commit, verify_files=True)


def _require_ta_m5_equal_loss_fit_launch_recipe(value, *, source_repo, source_commit, verify_files):
    _need(verify_files, 'M5_EQUAL_LOSS_FIT_PHYSICAL_METADATA_REQUIRED')
    refs, qualified = _ta_m5_equal_loss_fit_qualification()
    expected = _ta_m5_equal_loss_fit_recipe_fields(refs, qualified)
    recipe = _identity(value, set(expected) | {'source_owners'})
    _need(source_repo == CANONICAL_REPO and isinstance(source_commit, str)
          and campaign._COMMIT.fullmatch(source_commit) is not None
          and all(recipe[k] == v and type(recipe[k]) is type(v) for k, v in expected.items()),
          'M5_EQUAL_LOSS_FIT_RECIPE_SCOPE')
    _require_ta_m5_equal_loss_fit_sources(recipe['source_owners'], qualified['source_owners'],
                                        source_commit=source_commit)
    for name, ref in refs.items():
        campaign.require_binding(ref, label='equal-loss full-fit input ' + name, verify_file=True)
    return recipe
