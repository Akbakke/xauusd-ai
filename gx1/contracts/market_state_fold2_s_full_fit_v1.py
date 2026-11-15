"""One chronological full-GC Fold 2 S fit; recipe is not execution authority."""
from __future__ import annotations

from pathlib import Path
import subprocess

from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts import market_state_fold2_multiminute_private_campaign_v1 as source

REPO = Path('/home/andre2/src/GX1_CURRENT')
STAGE = 'market_state_fold2_25_cell_S_full_fit_20261115_003'
SCOPE = 'market_state_fold2_25_cell_S_full_fit_v1'
ROOT = Path('/home/andre2/GX1_RUNS') / STAGE.upper()
RUNTIME_ROOT = ROOT / 'RUNTIME'
RECIPE_PATH = ROOT / 'RECIPE.json'
DESIGN_PATH = REPO / 'docs/S_FOLD2_S_ONLY_FULL_FIT_DESIGN_20261115.json'
DESIGN_SHA256 = 'ffead067330fd38406ba78ebda589d5b20d71906d4048b15236b188d425220f6'
RECIPE_SCHEMA = 'gx1_fold2_s_full_fit_recipe_v1'
CURSOR_SCHEMA = 'gx1_fold2_s_full_fit_cursor_v1'
RESULT_SCHEMA = 'gx1_fold2_s_full_fit_result_v1'
FIT_ROWS, BATCH_SIZE, BATCH_VISITS, LATER_ROWS = 47338, 16, 2959, 4096
INITIAL_MODEL_SHA256 = 'd47a4624835ab811f115007af88b346d0597ef4fd682806db6351c7a6ade4679'
TRAIN_PARENT_ORDER_SHA256 = 'db43609ad6bd10e8d1a13b216243855c499189f8d2f08ef0a67d49de66e29fe9'
TRAIN_ISSUES_SHA256 = '380184a9c95a02f2716326765783746b1834371a7266545113fb1eae6722da47'
LATER_PARENT_ORDER_SHA256 = '1e4a137f13f9217641999e9730a0eec152472fe26aef517bd83da04463d926b6'
LATER_ISSUES_SHA256 = 'c03ec2ae8b418e21739f9f7966b946615878344364e6dc4e4d80087a0fee99d3'
MAX_FORWARDS = BATCH_VISITS + 512
ONE_WINDOW = {'max_invocation_seconds': 12000, 'max_model_seconds': 10500,
              'wrapper_wall_seconds': 13800, 'gpu_power_cap_w': 250,
              'checkpoint_writes_max': 9, 'resume_allowed': False,
              'failure_consumes_one_use_plan': True}
SOURCE_PATHS = tuple(sorted(source.required_source_paths() | {
    'gx1/contracts/market_state_fold2_s_full_fit_v1.py',
    'gx1/contracts/market_state_fold2_selected_native_batches_v1.py',
    'gx1/scripts/run_market_state_fold2_s_full_fit_window_v1.py',
    'gx1/contracts/entry_native_direction_research_campaign_v1.py',
    'gx1/scripts/run_entry_native_direction_research_window_v1.py',
    'gx1/scripts/materialize_market_state_time_query_fold1_fit_campaign_v1.py',
    'gx1/scripts/prepare_market_state_fold2_s_full_fit_recipe_v1.py',
    'gx1/contracts/market_state_fold2_multiminute_path_v1.py',
    'gx1/contracts/market_state_fold2_multiminute_side_loss_v1.py',
    'gx1/contracts/market_state_minute_query_candidate_v1.py',
    'gx1/contracts/market_state_time_query_model_v2.py',
    'gx1/contracts/market_state_time_query_private_state_v1.py',
    'gx1/models/entry_v10/entry_v10_ctx_train_v3.py',
    'scripts/gx1_capped_run.sh',
}))


def need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise campaign.RandomAccessCampaignError('FOLD2_S_FULL_FIT_' + reason)


def batch_bounds(index: int) -> tuple[int, int]:
    need(type(index) is int and 0 <= index < BATCH_VISITS, 'BATCH_INDEX')
    left = index * BATCH_SIZE
    return left, min(left + BATCH_SIZE, FIT_ROWS)


def cursor_from_visits(visits: int) -> int:
    need(type(visits) is int and 0 <= visits <= BATCH_VISITS, 'VISITS')
    return min(visits * BATCH_SIZE, FIT_ROWS)


def visits_from_cursor(cursor: int) -> int:
    need(type(cursor) is int and 0 <= cursor <= FIT_ROWS
         and (cursor == FIT_ROWS or cursor % BATCH_SIZE == 0), 'CURSOR')
    visits = (cursor + BATCH_SIZE - 1) // BATCH_SIZE
    need(cursor_from_visits(visits) == cursor, 'CURSOR_PARTITION')
    return visits


def _ref(path: Path) -> dict:
    need(path.is_absolute() and path.is_file() and not path.is_symlink(), 'FILE')
    return {'path': str(path), 'sha256': campaign.file_sha256(path)}


def _expected_recipe() -> dict:
    from gx1.contracts.market_state_fold2_multiminute_path_v1 import QUERY_SHA256
    from gx1.contracts.market_state_fold2_multiminute_side_loss_v1 import contract
    from gx1.contracts.market_state_fold2_multiminute_private_v1 import MODEL_CONTRACT_SHA256
    design_ref = _ref(DESIGN_PATH)
    need(design_ref['sha256'] == DESIGN_SHA256, 'FROZEN_DESIGN')
    design = campaign.read_bound_json(DESIGN_PATH, DESIGN_SHA256)
    need(design['schema_version'] == 'gx1_fold2_s_only_25_cell_full_fit_design_v1'
         and design['fixed_inputs']['fit_rows'] == FIT_ROWS
         and design['fixed_inputs']['fit_batches_b16'] == BATCH_VISITS
         and design['fixed_inputs']['later_rows'] == LATER_ROWS
         and design['fixed_inputs']['query_sha256'] == QUERY_SHA256
         and design['authority']['test'] is False
         and design['authority']['trade'] is False, 'DESIGN_SCOPE')
    refs = source.input_refs()
    refs['frozen_design'] = design_ref
    for name, value in design['references'].items():
        need(_ref(Path(value['path'])) == value, 'FROZEN_REFERENCE:' + name)
        refs[name] = value
    proof = source.base.fold2_selected_native_source_evidence()
    need(proof['decision'] == 'PASS_FOLD2_FULL_GC_SELECTED_S_F_SOURCE_TARGET_ONLY',
         'SOURCE_PROOF')
    import hashlib
    import json
    numerical = dict(source.time_query_private_numerical_recipe())
    numerical.pop('contract_sha256')
    numerical['schema_version'] = 'gx1_fold2_s_full_fit_numerical_v1'
    numerical['max_updates'] = BATCH_VISITS
    numerical['contract_sha256'] = hashlib.sha256(json.dumps(
        numerical, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    return {
        'schema_version': RECIPE_SCHEMA, 'source_repo': str(REPO),
        'runtime_root': str(RUNTIME_ROOT),
        'execution_binding_path': str(RUNTIME_ROOT / 'EXECUTION_BINDING.json'),
        'admission_stage': STAGE, 'run_scope': SCOPE,
        'fold_id': 2, 'partition': 'fit', 'pass_index': 0,
        'feature_fit_cutoff_ns': source.FEATURE_FIT_CUTOFF_NS,
        's_target_fold_upper_ns': source.S_TARGET_FOLD_UPPER_NS,
        's_only_25_cell': True, 'selected_state_endpoints': False,
        'numerical_recipe': numerical, 'input_refs': refs,
        'source_owners': {name: _ref(REPO / name) for name in SOURCE_PATHS},
        'expected_view_sha256': proof['view_sha256'],
        'expected_model_contract_sha256': MODEL_CONTRACT_SHA256,
        'expected_loss_contract_sha256': contract()['contract_sha256'],
        'expected_initial_model_sha256': INITIAL_MODEL_SHA256,
        'selected_fit_rows': FIT_ROWS, 'selected_score_rows': LATER_ROWS,
        'max_private_updates': BATCH_VISITS,
        'max_model_constructions': 1, 'max_optimizer_constructions': 1,
        'max_forwards': MAX_FORWARDS, 'max_backwards': BATCH_VISITS,
        'max_checkpoint_writes': ONE_WINDOW['checkpoint_writes_max'],
        'max_checkpoint_bytes': 192 * 1024 * 1024,
        'max_published_json_bytes': 4 * 1024 * 1024,
        'max_model_seconds': ONE_WINDOW['max_model_seconds'],
        'one_window': ONE_WINDOW,
        'training_enabled': True, 'full_epoch_training_allowed': True,
        'canonical_resume_authority': False, 'relaunch_allowed': False,
        'predictive_value_claimed': False,
        'qualification_weights_for_main_fit': False,
        'test_data_used': False,
    }


def build_recipe(*, source_commit: str) -> dict:
    need(isinstance(source_commit, str)
         and campaign._COMMIT.fullmatch(source_commit), 'SOURCE_COMMIT')
    value = _expected_recipe()
    value['mechanical_source_commit'] = source_commit
    value['recipe_sha256'] = campaign.canonical_sha256(value)
    return require_recipe(value, source_repo=REPO, source_commit=source_commit)


def require_recipe(value: dict, *, source_repo: Path, source_commit: str,
                   verify_files: bool = True) -> dict:
    need(verify_files and source_repo == REPO and isinstance(value, dict),
         'PHYSICAL_SOURCE')
    expected = _expected_recipe()
    need(set(value) == set(expected) | {'mechanical_source_commit', 'recipe_sha256'}
         and all(value[key] == expected[key] for key in expected)
         and value['recipe_sha256'] == campaign.canonical_sha256({
             k: v for k, v in value.items() if k != 'recipe_sha256'}),
         'EXACT_RECIPE')
    for revision in (source_commit, value['mechanical_source_commit']):
        need(isinstance(revision, str)
             and campaign._COMMIT.fullmatch(revision)
             and subprocess.run(['git', 'merge-base', '--is-ancestor',
                 '9aed2d906fb4511a6a099ba796e632a29dc4755e', revision],
                 cwd=REPO, capture_output=True).returncode == 0,
             'SOURCE_ANCESTRY')
    need(subprocess.run(['git', 'merge-base', '--is-ancestor',
         value['mechanical_source_commit'], source_commit], cwd=REPO,
         capture_output=True).returncode == 0, 'RECIPE_COMMIT')
    return dict(value)


def gate_fields() -> dict:
    return {'decision': 'ADMITTED', 'status': 'ADMITTED', 'fold_id': 2,
            'selected_fit_rows': FIT_ROWS, 'selected_score_rows': LATER_ROWS,
            'model_constructions_authorized': 1, 'optimizer_constructions_authorized': 1,
            'model_forwards_authorized': MAX_FORWARDS,
            'backward_calls_authorized': BATCH_VISITS,
            'optimizer_steps_authorized': BATCH_VISITS,
            'training_enabled': True, 'native_training_enabled': True,
            'genuine_predictor_fit_authorized': True,
            'genuine_backward_authorized': True, 'genuine_updates_authorized': True,
            'full_epoch_training_allowed': True,
            'canonical_resume_authority': False, 'candidate_selection': False,
            'native_launch_authorized': False, 'normalization_fit_authorized': False,
            'synthetic_test_execution_authorized': False, 'full_val_allowed': False,
            'exit_evaluation': False, 'broker_access_allowed': False,
            'test_data_used': False}


def require_gate(gate: dict, recipe: dict) -> None:
    need(isinstance(gate, dict)
         and all(type(gate.get(k)) is type(v) and gate[k] == v
                 for k, v in gate_fields().items())
         and recipe['run_scope'] == SCOPE
         and recipe['admission_stage'] == STAGE, 'POLICY_GATE')


def build_cursor(*, recipe_ref: dict, outcome: str, optimizer_updates: int,
                 parent_cursor: int, parent_order_sha256: str,
                 full_state, result: dict, batch_size: int = BATCH_SIZE) -> dict:
    need(batch_size == BATCH_SIZE, 'CURSOR_BATCH_SIZE')
    value = {'schema_version': CURSOR_SCHEMA, 'recipe': recipe_ref,
             'outcome': outcome, 'batch_visits': visits_from_cursor(parent_cursor),
             'optimizer_updates': optimizer_updates, 'parent_cursor': parent_cursor,
             'parent_order_sha256': parent_order_sha256, 'full_state': full_state,
             'result': result, 'canonical_resume_authority': False,
             'test_data_used': False}
    value['cursor_sha256'] = campaign.canonical_sha256(value)
    return require_cursor(value, expected_recipe=recipe_ref, verify_files=False,
                          batch_size=batch_size)


def require_cursor(value: dict, *, expected_recipe: dict,
                   verify_files: bool = True, batch_size: int = BATCH_SIZE) -> dict:
    fields = {'schema_version', 'recipe', 'outcome', 'batch_visits',
              'optimizer_updates', 'parent_cursor', 'parent_order_sha256',
              'full_state', 'result', 'canonical_resume_authority', 'test_data_used'}
    need(batch_size == BATCH_SIZE
         and expected_recipe.get('path') == str(RECIPE_PATH)
         and isinstance(value, dict)
         and set(value) == fields | {'cursor_sha256'}
         and value['schema_version'] == CURSOR_SCHEMA
         and value['cursor_sha256'] == campaign.canonical_sha256({
             k: v for k, v in value.items() if k != 'cursor_sha256'})
         and value['recipe'] == expected_recipe
         and value['outcome'] in ('COMPLETE', 'FAILED')
         and value['parent_order_sha256'] == TRAIN_PARENT_ORDER_SHA256
         and value['parent_cursor'] == cursor_from_visits(value['batch_visits'])
         and type(value['optimizer_updates']) is int
         and 0 <= value['optimizer_updates'] <= value['batch_visits']
         and value['canonical_resume_authority'] is False
         and value['test_data_used'] is False, 'CURSOR_SCOPE')
    need(value['outcome'] != 'COMPLETE'
         or value['batch_visits'] == BATCH_VISITS, 'COMPLETE_VISITS')
    need(value['result'].get('path') == str(ROOT / 'RESULT.json'), 'RESULT_PATH')
    campaign.require_binding(value['result'], label='S full fit result',
                             verify_file=verify_files)
    if value['full_state'] is not None:
        campaign.require_binding(value['full_state'], label='S full state',
                                 verify_file=verify_files)
    if verify_files:
        report = campaign.read_bound_json(
            Path(value['result']['path']), value['result']['sha256'])
        need(report.get('schema_version') == RESULT_SCHEMA
             and report.get('outcome') == value['outcome']
             and report.get('batch_visits') == value['batch_visits']
             and report.get('optimizer_updates') == value['optimizer_updates']
             and report.get('parent_cursor') == value['parent_cursor']
             and report.get('parent_order_sha256') == TRAIN_PARENT_ORDER_SHA256
             and report.get('canonical_resume_authority') is False
             and report.get('predictive_edge_proven') is False
             and report.get('economic_value_measured') is False
             and report.get('test_data_used') is False
             and 0 <= report.get('checkpoint_writes', -1)
                 <= ONE_WINDOW['checkpoint_writes_max']
             and (report['checkpoints'][-1]['state']
                  if report.get('checkpoints') else None) == value['full_state'],
             'RESULT_IDENTITY')
        if value['outcome'] == 'COMPLETE':
            need(report.get('status') == 'PASS_FOLD2_S_FULL_FIT_AND_LATER_READOUT'
                 and report.get('source_unchanged') is True
                 and report.get('batch_visits') == BATCH_VISITS
                 and report.get('optimizer_updates')
                     + report.get('skipped_batches') == BATCH_VISITS
                 and report.get('batch_visit_in_flight_at_terminal') is False
                 and report.get('optimizer_updates_is_lower_bound') is False
                 and set(report.get('captures', {})) == {'INITIAL', 'FINAL'}
                 and set(report.get('readouts', {})) == {'INITIAL', 'FINAL'}
                 and isinstance(report.get('primary_comparison'), dict)
                 and report['primary_comparison'].get('same_certified_rows') == 1392
                 and len(report.get('training_records', [])) == BATCH_VISITS,
                 'COMPLETE_RESULT')
            for binding in report['captures'].values():
                campaign.require_binding(binding, label='S later capture',
                                         verify_file=True)
    return dict(value)
