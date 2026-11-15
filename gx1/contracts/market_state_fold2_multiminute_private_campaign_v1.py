"""One-use guarded Fold 2 S-only 25-cell optimizer/resume metadata.

A candidate recipe is not admission. Four physical private updates compare a
fresh two-B16 trajectory with an interrupted/resumed trajectory. The private
weights cannot seed the later full FIT. No Entry, TEST, broker or trade role.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any, Mapping

from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts import market_state_fold2_selected_native_private_v1 as old
from gx1.contracts import market_state_native_F_private_campaign_v1 as base
from gx1.contracts.market_state_fold2_multiminute_private_v1 import (
    numerical_recipe, MODEL_CONTRACT_SHA256, LOSS_CONTRACT_SHA256,
)

SCOPE = 'market_state_fold2_25_cell_S_private_two_batch_v1'
STAGE = 'market_state_fold2_25_cell_S_private_two_batch_20261115_001'
ROOT = Path('/home/andre2/GX1_RUNS') / STAGE.upper()
RUNTIME_ROOT = ROOT / 'RUNTIME'
RECIPE_PATH = ROOT / 'RECIPE.json'
RECIPE_SCHEMA = 'gx1_market_state_fold2_25_cell_S_private_recipe_v1'
CURSOR_SCHEMA = 'gx1_market_state_fold2_25_cell_S_private_cursor_v1'
RESULT_SCHEMA = 'gx1_market_state_fold2_25_cell_S_private_result_v1'
PASS_DECISION = 'PASS_FOLD2_25_CELL_S_PRIVATE_RESUME'
FOLD_ID = 2
PARENT_ROWS = base.PARENT_ROWS
BATCH_SIZE = base.BATCH_SIZE
PASS_INDEX = 0
MAX_PRIVATE_UPDATES = base.MAX_PRIVATE_UPDATES
MAX_MODEL_CONSTRUCTIONS = base.MAX_MODEL_CONSTRUCTIONS
MAX_FORWARDS = base.MAX_FORWARDS
MAX_BACKWARDS = base.MAX_BACKWARDS
MAX_CHECKPOINT_BYTES = base.MAX_CHECKPOINT_BYTES
MAX_PUBLISHED_JSON_BYTES = base.MAX_PUBLISHED_JSON_BYTES
INITIAL_REVIEW = base.CANONICAL_REPO / 'docs/S_FOLD2_MULTIMINUTE_INITIAL_REVIEW_20261115.json'
SOURCE_PATHS = (
    'gx1/contracts/market_state_fold2_multiminute_private_campaign_v1.py',
    'gx1/contracts/market_state_fold2_multiminute_private_v1.py',
    'gx1/contracts/market_state_fold2_multiminute_path_v1.py',
    'gx1/contracts/market_state_fold2_multiminute_side_loss_v1.py',
    'gx1/contracts/market_state_minute_query_candidate_v1.py',
    'gx1/contracts/market_state_time_query_model_v2.py',
    'gx1/contracts/market_state_time_query_private_state_v1.py',
    'gx1/audit/build_market_state_time_query_tape_v1.py',
    'gx1/scripts/run_market_state_time_query_private_window_v1.py',
    'gx1/scripts/prepare_market_state_fold2_multiminute_private_recipe_v1.py',
)


def _need(ok: object, reason: str) -> None:
    if not ok:
        raise campaign.RandomAccessCampaignError('FOLD2_S_PRIVATE_' + reason)


def time_query_private_numerical_recipe() -> dict:
    return numerical_recipe()


def input_refs() -> dict[str, dict[str, str]]:
    refs = old.input_refs()
    review_ref = {'path': str(INITIAL_REVIEW),
                  'sha256': campaign.file_sha256(INITIAL_REVIEW)}
    review = campaign.read_bound_json(INITIAL_REVIEW, review_ref['sha256'])
    _need(review.get('status', '').startswith('COMPLETE_CONSUMED_INITIAL_MECHANICS_PASS')
          and review.get('measured', {}).get('optimizer_steps') == 0
          and review.get('measured', {}).get('query_cells_per_row') == 25
          and review.get('measured', {}).get('test_data_used') is False,
          'INITIAL_S_REVIEW')
    plan_ref = review['plan']
    plan = campaign.read_bound_json(Path(plan_ref['path']), plan_ref['sha256'])
    _need(plan.get('scope', {}).get('fold_id') == FOLD_ID
          and plan['scope']['optimizer_steps'] == 0
          and plan['scope']['batch_size'] == BATCH_SIZE
          and plan['scope']['loss_contract_sha256'] == LOSS_CONTRACT_SHA256
          and plan['scope']['test_data_used'] is False,
          'INITIAL_S_PLAN')
    refs.update(s_initial_review=review_ref, s_initial_plan=dict(plan_ref),
                s_initial_result=dict(review['result']),
                s_initial_terminal=dict(review['terminal']),
                candidate_tape={key: plan['input_refs']['candidate_tape'][key]
                                for key in ('path', 'sha256')},
                fit_support=dict(plan['input_refs']['fit_support']))
    return refs


def required_source_paths() -> set[str]:
    return old.required_source_paths() | set(SOURCE_PATHS)


def build_recipe(*, source_commit: str) -> dict:
    proof = base.fold2_selected_native_source_evidence()
    _need(proof['parent_rows'] == list(PARENT_ROWS)
          and proof['batch_size'] == BATCH_SIZE
          and type(proof['view_sha256']) is str
          and len(proof['view_sha256']) == 64,
          'FOLD2_FULL_GC_SOURCE')
    repo = base.CANONICAL_REPO
    value = {
        'schema_version': RECIPE_SCHEMA,
        'source_repo': str(repo), 'mechanical_source_commit': source_commit,
        'runtime_root': str(RUNTIME_ROOT),
        'execution_binding_path': str(RUNTIME_ROOT / 'EXECUTION_BINDING.json'),
        'admission_stage': STAGE, 'run_scope': SCOPE,
        'fold_id': FOLD_ID, 'partition': 'fit',
        'parent_rows': list(PARENT_ROWS), 'pass_index': PASS_INDEX,
        's_only_25_cell': True, 'selected_state_endpoints': False,
        'numerical_recipe': numerical_recipe(), 'input_refs': input_refs(),
        'source_owners': {name: {'path': str(repo / name),
                                'sha256': campaign.file_sha256(repo / name)}
                          for name in sorted(required_source_paths())},
        'expected_view_sha256': proof['view_sha256'],
        'expected_model_contract_sha256': MODEL_CONTRACT_SHA256,
        'expected_loss_contract_sha256': LOSS_CONTRACT_SHA256,
        'max_private_updates': MAX_PRIVATE_UPDATES,
        'max_model_constructions': MAX_MODEL_CONSTRUCTIONS,
        'max_forwards': MAX_FORWARDS, 'max_backwards': MAX_BACKWARDS,
        'max_checkpoint_bytes': MAX_CHECKPOINT_BYTES,
        'max_published_json_bytes': MAX_PUBLISHED_JSON_BYTES,
        'training_enabled': True, 'full_epoch_training_allowed': False,
        'test_data_used': False, 'predictive_value_claimed': False,
        'qualification_weights_for_main_fit': False,
    }
    return {**value, 'recipe_sha256': campaign.canonical_sha256(value)}


def require_recipe(value: Any, *, source_repo: Path, source_commit: str,
                   verify_files: bool = True) -> dict:
    _need(verify_files and source_repo == base.CANONICAL_REPO
          and isinstance(value, Mapping)
          and type(source_commit) is str
          and campaign._COMMIT.fullmatch(source_commit) is not None,
          'PHYSICAL_SOURCE_REQUIRED')
    recipe = dict(value)
    seal = recipe.pop('recipe_sha256', None)
    mechanical = recipe.get('mechanical_source_commit')
    _need(seal == campaign.canonical_sha256(recipe)
          and type(mechanical) is str
          and campaign._COMMIT.fullmatch(mechanical) is not None
          and subprocess.run(['git', 'merge-base', '--is-ancestor', mechanical,
                              source_commit], cwd=source_repo,
                             capture_output=True).returncode == 0,
          'EXACT_COMMITTED_RECIPE')
    _need(value == build_recipe(source_commit=mechanical), 'RECIPE_BINDING')
    for name, ref in value['input_refs'].items():
        campaign.require_binding(ref, label='Fold2 S TRAIN input ' + name,
                                 verify_file=True)
    for name, ref in value['source_owners'].items():
        campaign.require_binding(ref, label='Fold2 S source ' + name,
                                 verify_file=True)
    return dict(value)


def require_gate(gate: Any, recipe: Mapping[str, Any]) -> None:
    _need(isinstance(gate, Mapping)
          and gate.get('decision') == gate.get('status') == 'ADMITTED'
          and gate.get('fold2_25_cell_s_private_authorized') is True
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


def build_time_query_private_cursor(*, recipe_ref: Mapping[str, str], outcome: str,
                                    optimizer_updates: int, parent_cursor: int,
                                    parent_order_sha256: str,
                                    full_state: Mapping[str, str] | None,
                                    result: Mapping[str, str]) -> dict:
    value = {'schema_version': CURSOR_SCHEMA, 'recipe': dict(recipe_ref),
             'outcome': outcome, 'optimizer_updates': optimizer_updates,
             'parent_cursor': parent_cursor,
             'parent_order_sha256': parent_order_sha256,
             'full_state': dict(full_state) if full_state is not None else None,
             'result': dict(result), 'test_data_used': False}
    return require_cursor({**value, 'cursor_sha256': campaign.canonical_sha256(value)},
                          expected_recipe=recipe_ref, verify_files=False,
                          batch_size=BATCH_SIZE)


def require_cursor(value: Any, *, expected_recipe: Mapping[str, str],
                   verify_files: bool = True, batch_size: int = BATCH_SIZE) -> dict:
    fields = {'schema_version', 'recipe', 'outcome', 'optimizer_updates',
              'parent_cursor', 'parent_order_sha256', 'full_state', 'result',
              'test_data_used'}
    _need(isinstance(value, Mapping) and set(value) == fields | {'cursor_sha256'},
          'CURSOR_FIELDS')
    cursor = dict(value)
    seal = cursor.pop('cursor_sha256')
    _need(seal == campaign.canonical_sha256(cursor)
          and cursor['schema_version'] == CURSOR_SCHEMA
          and cursor['recipe'] == expected_recipe
          and expected_recipe['path'] == str(RECIPE_PATH)
          and batch_size == BATCH_SIZE
          and cursor['outcome'] in ('COMPLETE', 'FAILED')
          and type(cursor['optimizer_updates']) is int
          and 0 <= cursor['optimizer_updates'] <= MAX_PRIVATE_UPDATES
          and type(cursor['parent_cursor']) is int
          and cursor['parent_cursor'] in (0, 16, 32)
          and cursor['test_data_used'] is False,
          'CURSOR_SCOPE')
    campaign._sha(cursor['parent_order_sha256'], 'Fold2 S parent order')
    campaign.require_binding(cursor['result'], label='Fold2 S private result',
                             verify_file=verify_files)
    if cursor['full_state'] is not None:
        campaign.require_binding(cursor['full_state'], label='Fold2 S private state',
                                 verify_file=verify_files)
    _need(cursor['outcome'] != 'COMPLETE'
          or (cursor['optimizer_updates'] == MAX_PRIVATE_UPDATES
              and cursor['parent_cursor'] == len(PARENT_ROWS)
              and cursor['full_state'] is not None), 'CURSOR_COMPLETE')
    if verify_files:
        report = campaign.read_bound_json(Path(cursor['result']['path']),
                                          cursor['result']['sha256'])
        _need(report.get('schema_version') == RESULT_SCHEMA
              and report.get('recipe') == expected_recipe
              and report.get('parent_order_sha256') == cursor['parent_order_sha256']
              and report.get('parent_cursor') == cursor['parent_cursor']
              and report.get('physical_optimizer_updates') == cursor['optimizer_updates']
              and report.get('final_state') == cursor['full_state']
              and report.get('test_data_used') is False
              and (cursor['outcome'] != 'COMPLETE' or
                   (report.get('decision') == PASS_DECISION
                    and report.get('bit_exact_resume') is True)),
              'CURSOR_RESULT')
    return dict(value)
