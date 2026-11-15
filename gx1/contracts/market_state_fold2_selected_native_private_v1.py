"""One guarded Fold 2 full-GC selected S/F AdamW resume qualification.

Metadata only. This probe has four physical updates on two fixed TRAIN batches.
It grants no model fit, later score, TEST, Entry or trading authority.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any, Mapping

from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts import market_state_native_F_private_campaign_v1 as old

SCOPE = 'market_state_native_F_selected_fold2_two_batch_v1'
STAGE = 'market_state_native_F_selected_fold2_two_batch_20261115_001'
ROOT = Path('/home/andre2/GX1_RUNS') / STAGE.upper()
RUNTIME_ROOT = ROOT / 'RUNTIME'
RECIPE_PATH = ROOT / 'RECIPE.json'
RECIPE_SCHEMA = 'gx1_market_state_native_F_selected_fold2_private_recipe_v1'
CURSOR_SCHEMA = 'gx1_market_state_native_F_selected_fold2_private_cursor_v1'
RESULT_SCHEMA = 'gx1_market_state_native_F_selected_fold2_private_result_v1'
PASS_DECISION = 'PASS_FOLD2_SELECTED_NATIVE_F_PRIVATE_RESUME'
SOURCE_PATHS = (
    'gx1/contracts/market_state_fold2_selected_native_private_v1.py',
    'gx1/contracts/market_state_fold2_selected_native_batches_v1.py',
    'gx1/contracts/market_state_native_F_private_state_v1.py',
    'gx1/contracts/market_state_native_F_private_campaign_v1.py',
    'gx1/contracts/market_state_time_query_native_F_batch_v1.py',
    'gx1/scripts/run_market_state_native_F_private_window_v1.py',
    'gx1/scripts/run_entry_native_direction_research_window_v1.py',
    'gx1/scripts/materialize_market_state_time_query_fold1_fit_campaign_v1.py',
    'gx1/scripts/prepare_market_state_fold2_selected_native_private_recipe_v1.py',
)


def _need(ok, reason):
    if not ok:
        raise campaign.RandomAccessCampaignError('FOLD2_SELECTED_NATIVE_PRIVATE_' + reason)


def input_refs() -> dict[str, dict[str, str]]:
    review = old._read(old.FOLD2_SELECTED_INITIAL_REVIEW, 'Fold 2 initial review')
    plan = old._read(review['plan'], 'Fold 2 initial plan')
    later = old._read(old.FOLD2_FULL_GC_LATER_REVIEW, 'Fold 2 later review')
    prior = old._read(old.FOLD2_EXACT_PRIOR_REVIEW, 'Fold 2 frozen prior review')
    _need(plan.get('scope', {}).get('fold_id') == 2
          and plan['scope']['selected_state_endpoints'] is True
          and plan['scope']['optimizer_steps'] == 0,
          'INITIAL_PLAN')
    refs = {name: dict(ref) for name, ref in plan['inputs'].items()}
    refs.update(initial_review=dict(old.FOLD2_SELECTED_INITIAL_REVIEW),
                initial_plan=dict(review['plan']), initial_result=dict(review['result']),
                initial_terminal=dict(review['terminal']),
                later_review=dict(old.FOLD2_FULL_GC_LATER_REVIEW),
                later_result=dict(later['result']), later_terminal=dict(later['terminal']),
                prior_review=dict(old.FOLD2_EXACT_PRIOR_REVIEW),
                frozen_priors=dict(prior['frozen_priors']))
    return refs


def required_source_paths() -> set[str]:
    from gx1.audit.verify_market_state_native_F_cpu_zero_step_v1 import SOURCE_PATHS as initial_paths
    from gx1.contracts.entry_native_direction_research_campaign_v1 import REQUIRED_SOURCE_PATHS
    return set(initial_paths) | set(REQUIRED_SOURCE_PATHS) | set(SOURCE_PATHS)


def build_recipe(*, source_commit: str) -> dict:
    from gx1.contracts.market_state_native_F_private_state_v1 import numerical_recipe
    proof = old.fold2_selected_native_source_evidence()
    _need(proof['parent_rows'] == list(old.PARENT_ROWS)
          and proof['batch_size'] == old.BATCH_SIZE
          and proof['target_support']['state']['informative_by_batch'] == [80, 80]
          and proof['target_support']['future']['informative_by_batch'] == [80, 80],
          'SOURCE_PROOF')
    value = {
        'schema_version': RECIPE_SCHEMA,
        'source_repo': str(old.CANONICAL_REPO),
        'mechanical_source_commit': source_commit,
        'runtime_root': str(RUNTIME_ROOT),
        'execution_binding_path': str(RUNTIME_ROOT / 'EXECUTION_BINDING.json'),
        'admission_stage': STAGE, 'run_scope': SCOPE,
        'fold_id': 2, 'partition': 'fit', 'parent_rows': list(old.PARENT_ROWS),
        'pass_index': 0, 'selected_state_endpoints': True,
        'numerical_recipe': numerical_recipe(),
        'input_refs': input_refs(),
        'source_owners': {name: {'path': str(old.CANONICAL_REPO / name),
                                'sha256': campaign.file_sha256(old.CANONICAL_REPO / name)}
                          for name in sorted(required_source_paths())},
        'expected_view_sha256': proof['view_sha256'],
        'expected_model_contract_sha256': proof['model_contract_sha256'],
        'expected_loss_contract_sha256': proof['loss_contract_sha256'],
        'max_private_updates': old.MAX_PRIVATE_UPDATES,
        'max_model_constructions': old.MAX_MODEL_CONSTRUCTIONS,
        'max_forwards': old.MAX_FORWARDS, 'max_backwards': old.MAX_BACKWARDS,
        'max_checkpoint_bytes': old.MAX_CHECKPOINT_BYTES,
        'max_published_json_bytes': old.MAX_PUBLISHED_JSON_BYTES,
        'training_enabled': True, 'full_epoch_training_allowed': False,
        'test_data_used': False, 'predictive_value_claimed': False,
        'qualification_weights_for_main_fit': False,
    }
    return {**value, 'recipe_sha256': campaign.canonical_sha256(value)}


def require_recipe(value: Any, *, source_repo: Path, source_commit: str,
                   verify_files: bool = True) -> dict:
    _need(verify_files and source_repo == old.CANONICAL_REPO
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
    expected = build_recipe(source_commit=mechanical)
    _need(value == expected, 'RECIPE_BINDING')
    for name, ref in value['input_refs'].items():
        campaign.require_binding(ref, label='Fold 2 selected TRAIN ' + name,
                                 verify_file=True)
    for name, ref in value['source_owners'].items():
        campaign.require_binding(ref, label='Fold 2 selected source ' + name,
                                 verify_file=True)
    return dict(value)


def require_gate(gate: Any, recipe: Mapping[str, Any]) -> None:
    _need(isinstance(gate, Mapping)
          and gate.get('decision') == gate.get('status') == 'ADMITTED'
          and gate.get('fold2_selected_native_private_authorized') is True
          and gate.get('fold_id') == 2
          and gate.get('sample_parent_rows') == list(old.PARENT_ROWS)
          and gate.get('model_forwards_authorized') == old.MAX_FORWARDS
          and gate.get('backward_calls_authorized') == old.MAX_BACKWARDS
          and gate.get('optimizer_steps_authorized') == old.MAX_PRIVATE_UPDATES
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
    return require_cursor({**value, 'cursor_sha256': campaign.canonical_sha256(value)},
                          expected_recipe=recipe_ref, verify_files=False,
                          batch_size=old.BATCH_SIZE)


def require_cursor(value: Any, *, expected_recipe: Mapping[str, str],
                   verify_files: bool = True, batch_size: int = old.BATCH_SIZE) -> dict:
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
          and batch_size == old.BATCH_SIZE
          and cursor['outcome'] in ('COMPLETE', 'FAILED')
          and type(cursor['optimizer_updates']) is int
          and 0 <= cursor['optimizer_updates'] <= old.MAX_PRIVATE_UPDATES
          and type(cursor['parent_cursor']) is int
          and cursor['parent_cursor'] in (0, 16, 32)
          and cursor['test_data_used'] is False,
          'CURSOR_SCOPE')
    campaign._sha(cursor['parent_order_sha256'], 'Fold 2 parent order')
    campaign.require_binding(cursor['result'], label='Fold 2 private result',
                             verify_file=verify_files)
    if cursor['full_state'] is not None:
        campaign.require_binding(cursor['full_state'], label='Fold 2 private state',
                                 verify_file=verify_files)
    _need(cursor['outcome'] != 'COMPLETE'
          or (cursor['optimizer_updates'] == old.MAX_PRIVATE_UPDATES
              and cursor['parent_cursor'] == len(old.PARENT_ROWS)
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
