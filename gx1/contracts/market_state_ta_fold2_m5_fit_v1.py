"""Fixed chronological fold 2 M5 direction-fit schedule; no run authority."""
from __future__ import annotations

from pathlib import Path

REPO = Path('/home/andre2/src/GX1_CURRENT')
STAGE = 'market_state_ta_fold2_m5_direction_fit_20261114_004'
SCOPE = 'market_state_ta_fold2_m5_direction_fit_v1'
ROOT = Path('/home/andre2/GX1_RUNS') / STAGE.upper()
RUNTIME_ROOT = ROOT / 'RUNTIME'
RECIPE_PATH = ROOT / 'RECIPE.json'
TRAIN_ROWS = 47338
BATCH_SIZE = 16
BATCH_VISITS = 2959
TAIL_ROWS = 10
PANEL_ROWS = 4096
PANEL_BATCHES = 256
MAX_UPDATES = BATCH_VISITS
INITIAL_MODEL_SHA256 = '3c172867694217f240a076c34220a1b7fafb2464af5738de544cf7a17068984b'
VIEW_SHA256 = '388fadd9ce6b6d8673724f776eb0f945711ab76cc3b3650f875f3a9e5431c4ab'
PROFILE_SHA256 = '734a2bfb6b1c796205b9a77bd91989bf1f0652e6bf6594927c74ff580c634f75'
LOSS_SHA256 = 'da042c0f02d104bf5533f9d82688b986a29117436517485272de0853e7bd6150'
MODEL_CONTRACT_SHA256 = '4d6fa668262b2cf51e78d015b81a462dc53debf606121b859898053e6fd12e38'
BOOTSTRAP_SEEDS = {'UTC_DAY': 20260911, 'UTC_MONDAY_WEEK': 20260912}
ADAMW_DEFAULTS = {'lr': 1e-4, 'betas': (0.9, 0.999), 'eps': 1e-8,
                  'weight_decay': 1e-4, 'amsgrad': False, 'foreach': None,
                  'maximize': False, 'capturable': False,
                  'differentiable': False, 'fused': None}
ONE_WINDOW = {'max_invocation_seconds': 12000, 'max_model_seconds': 10500,
              'wrapper_wall_seconds': 13800, 'gpu_power_cap_w': 250,
              'checkpoint_writes': 0, 'resume_allowed': False,
              'failure_consumes_one_use_plan': True}


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('FOLD2_M5_FIT_' + reason)


def batch_bounds(index: int) -> tuple[int, int]:
    """Use every TRAIN row once, in axis order; the final batch has ten rows."""
    _need(type(index) is int and 0 <= index < BATCH_VISITS, 'BATCH_INDEX')
    left = BATCH_SIZE * index
    return left, min(left + BATCH_SIZE, TRAIN_ROWS)


def require_fixed_design(hypothesis: dict) -> None:
    fit = hypothesis['fixed_fit']
    readout = hypothesis['fixed_readout']
    _need((fit['fold_id'], fit['pass_index'], fit['seed'], fit['train_rows'],
           fit['ordered_batches'], fit['batch_size'], fit['last_batch_rows'],
           fit['maximum_optimizer_updates'])
          == (2, 0, 20260911, TRAIN_ROWS, BATCH_VISITS, BATCH_SIZE, TAIL_ROWS,
              MAX_UPDATES), 'FIT_SCHEDULE')
    _need(fit['fresh_start_requires_exact_initial_model_state_sha256'] == INITIAL_MODEL_SHA256
          and fit['view_contract_sha256'] == VIEW_SHA256
          and fit['m5_equal_observation_profile_sha256'] == PROFILE_SHA256
          and fit['loss_contract_sha256'] == LOSS_SHA256
          and fit['m5_direction_only'] is True and fit['future_weight'] == 1.0,
          'FIXED_FUNCTION')
    _need(fit['optimizer_candidate_reused_without_search']
          == {'name': 'torch.optim.AdamW',
              'kwargs': {key: list(value) if key == 'betas' else value
                         for key, value in ADAMW_DEFAULTS.items()}},
          'FIXED_OPTIMIZER')
    _need(readout['bootstrap_binding']['replicates'] == 2000
          and readout['bootstrap_binding']['seeds'] == BOOTSTRAP_SEEDS
          and all(panel['selected_rows'] == PANEL_ROWS
                  and panel['batch_size'] == BATCH_SIZE
                  for panel in readout['train_and_later_panels'].values()),
          'FIXED_READOUT')
    _need(hypothesis['resource_gate']['proposed_one_window'] == ONE_WINDOW,
          'ONE_WINDOW')
    _need(batch_bounds(0) == (0, 16)
          and batch_bounds(BATCH_VISITS - 2) == (47312, 47328)
          and batch_bounds(BATCH_VISITS - 1) == (47328, TRAIN_ROWS),
          'TAIL_PARTITION')


INITIAL_PLAN_REF = {
    'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_FOLD2_FRESH_INITIAL_20261114_004/PLAN.json',
    'sha256': '6fe63d5e06812122cf5a7c9a7e816fa547c718fd65039fac2588b2988542390a',
}


def require_reused_input_refs(initial_plan_ref: dict) -> dict:
    """Bind completed input receipts; the evolving design is separately frozen."""
    from gx1.audit.observe_market_state_ta_fold2_fresh_initial_v1 import read
    _need(initial_plan_ref == INITIAL_PLAN_REF, 'INITIAL_PLAN_BINDING')
    plan = read(initial_plan_ref)
    inputs = plan['inputs']
    _need(plan['stage'] == 'market_state_ta_fold2_fresh_initial_20261114_004'
          and plan['source_repo'] == '/home/andre2/src/GX1_CURRENT'
          and plan['train_rows'] == plan['later_rows'] == 4096
          and plan['forwards'] == 512 and plan['optimizer_updates'] == 0
          and plan['checkpoint_loads'] == 0 and plan['test_data_used'] is False,
          'INITIAL_PLAN_STAGE')
    expected = {'design', 'train_plan', 'later_plan', 'view', 'normalization',
                'constructor', 'priors', 'packet', 'provider_plan',
                'physical_result', 'train_gather_result', 'later_result',
                'later_terminal', 'later_gather_result', 'later_coverage',
                'train_validation', 'later_validation', 'target_validation'}
    _need(set(inputs) == expected, 'INITIAL_INPUT_SET')
    for name, binding in inputs.items():
        if name != 'design':
            read(binding)
    for name, status in (('train_validation', 'PASS_FULL_FOLD2_TRAIN_INPUT_GATHER'),
                         ('later_validation', 'PASS_JSON_HASH_JOIN_AND_EXACT_COUNTS'),
                         ('target_validation', 'PASS_JSON_PREFIX_MASK_BATCH_AND_PRIOR_JOINS')):
        receipt = read(inputs[name])
        _need(receipt['status'] == status and receipt['test_data_used'] is False,
              'VALIDATION_' + name.upper())
    return inputs


# Candidate metadata for the existing native research campaign. Building this
# recipe does not change NEXT_RUN_POLICY or grant execution.
DESIGN_PATH = REPO / 'configs/research/MARKET_STATE_TA_FOLD2_M5_FIT_DESIGN_20261114_004.json'
RECIPE_SCHEMA = 'gx1_ta_fold2_m5_direction_fit_recipe_v1'
RESULT_SCHEMA = 'gx1_ta_fold2_m5_direction_fit_result_v1'
CURSOR_SCHEMA = 'gx1_ta_fold2_m5_direction_fit_cursor_v1'
TRAIN_PARENT_ORDER_SHA256 = 'db43609ad6bd10e8d1a13b216243855c499189f8d2f08ef0a67d49de66e29fe9'
BASE_COMMIT = '678c0b19241d86074850bee7ad839e4b41097bef'
SOURCE_PATHS = (
    'gx1/contracts/market_state_ta_fold2_m5_fit_v1.py',
    'gx1/scripts/run_market_state_ta_fold2_m5_fit_window_v1.py',
    'gx1/audit/score_market_state_ta_fold2_m5_fit_v1.py',
    'gx1/audit/preflight_market_state_ta_fold2_m5_fit_v1.py',
    'gx1/contracts/entry_native_direction_research_campaign_v1.py',
    'gx1/scripts/run_entry_native_direction_research_window_v1.py',
    'gx1/scripts/materialize_market_state_time_query_fold1_fit_campaign_v1.py',
    'gx1/scripts/prepare_market_state_ta_fold2_m5_fit_recipe_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/entry_native_direction_feature_view_v1.py',
    'gx1/contracts/market_state_time_query_native_F_model_v1.py',
    'gx1/contracts/market_state_time_query_native_F_batch_v1.py',
    'gx1/contracts/market_state_time_query_native_F_objective_v1.py',
    'gx1/contracts/market_state_time_query_fold1_fit_v1.py',
    'gx1/contracts/model_state_digest_v1.py',
    'gx1/audit/observe_market_state_ta_fold2_fresh_initial_v1.py',
    'gx1/contracts/entry_anchor_span_feature_inputs_v1.py',
    'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
    'gx1/contracts/market_state_future_displacement_candidate_v1.py',
    'gx1/audit/qualify_market_state_mature_prefix_normalization_v1.py',
    'gx1/models/entry_v10/entry_v10_ctx_train_v3.py',
    'scripts/gx1_capped_run.sh',
)


def _source_ref(path: Path) -> dict:
    import hashlib
    _need(path.is_absolute() and path.is_file() and not path.is_symlink(), 'SOURCE_FILE')
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def _fit_recipe_expected() -> dict:
    import hashlib
    import json
    import subprocess
    from gx1.audit.preflight_market_state_ta_fold2_m5_fit_v1 import read_ref
    from gx1.contracts import local_random_access_campaign_v2 as campaign

    design_ref = _source_ref(DESIGN_PATH)
    design = read_ref(design_ref, base=REPO)
    _need(set(design) == {'schema_version', 'admission_stage', 'run_scope',
          'fresh_initial_source_base_commit', 'fresh_initial_original_design',
          'fresh_initial_design', 'fold2_m5_fit_hypothesis', 'initial_plan',
          'initial_result', 'initial_validation', 'test_data_used'}
          and design['schema_version'] == 'gx1_ta_fold2_m5_direction_fit_frozen_design_v1'
          and design['admission_stage'] == STAGE and design['run_scope'] == SCOPE
          and design['test_data_used'] is False, 'FROZEN_DESIGN_IDENTITY')
    hypothesis = design['fold2_m5_fit_hypothesis']
    _need(set(hypothesis) == {'observed_basis', 'fixed_fit', 'fixed_readout', 'resource_gate'},
          'HYPOTHESIS_FIELDS')
    require_fixed_design(hypothesis)
    _need(design['fresh_initial_design']['panels']
          == hypothesis['fixed_readout']['train_and_later_panels']
          and design['initial_result'] == hypothesis['observed_basis']['fresh_fold2_initial_result']
          and design['initial_validation'] == hypothesis['observed_basis']['fresh_fold2_initial_validation'],
          'INITIAL_DESIGN_BINDING')
    runs = Path('/home/andre2/GX1_RUNS')
    plan = read_ref(design['initial_plan'], base=runs)
    initial = read_ref(design['initial_result'], base=runs)
    validation = read_ref(design['initial_validation'], base=runs)
    _need(plan['source_base_commit'] == design['fresh_initial_source_base_commit']
          and plan['inputs']['design'] == design['fresh_initial_original_design']
          and initial['plan'] == design['initial_plan']
          and initial['status'] == 'COMPLETE_FRESH_INITIAL'
          and initial['optimizer_updates'] == validation['training_updates'] == 0
          and initial['model_state_sha256'] == INITIAL_MODEL_SHA256
          and validation['status'] == 'PASS_INDEPENDENT_CAPTURE_READOUT'
          and initial['test_data_used'] is validation['test_data_used'] is False,
          'FRESH_INITIAL_BASIS')
    original = subprocess.check_output(['git', 'show',
        plan['source_base_commit'] + ':docs/TA_FOLD2_INPUT_REVIEW.json'], cwd=REPO)
    _need(hashlib.sha256(original).hexdigest() == plan['inputs']['design']['sha256']
          and json.loads(original)['fresh_initial_design'] == design['fresh_initial_design'],
          'ORIGINAL_INITIAL_DESIGN')
    require_reused_input_refs(design['initial_plan'])
    fixed = hypothesis['fixed_fit']
    numerical = {'fold_id': 2, 'pass_index': 0, 'seed': fixed['seed'],
                 'batch_size': BATCH_SIZE, 'max_updates': MAX_UPDATES,
                 'loss_contract_sha256': LOSS_SHA256,
                 'model_contract_sha256': MODEL_CONTRACT_SHA256,
                 'initial_model_state_sha256': INITIAL_MODEL_SHA256,
                 'm5_direction_only': True}
    value = {'schema_version': RECIPE_SCHEMA, 'source_repo': str(REPO),
             'admission_stage': STAGE, 'run_scope': SCOPE,
             'runtime_root': str(RUNTIME_ROOT),
             'execution_binding_path': str(RUNTIME_ROOT / 'EXECUTION_BINDING.json'),
             'input_refs': {'frozen_design': design_ref,
                            'initial_plan': design['initial_plan'],
                            'initial_result': design['initial_result'],
                            'initial_validation': design['initial_validation']},
             'source_owners': {name: _source_ref(REPO / name) for name in SOURCE_PATHS},
             'numerical_recipe': numerical,
             'selected_fit_rows': TRAIN_ROWS, 'selected_score_rows': PANEL_ROWS,
             'max_model_constructions': 1, 'max_optimizer_constructions': 1,
             'max_training_forwards': BATCH_VISITS,
             'max_evaluation_forwards': 2 * PANEL_BATCHES,
             'max_forwards': BATCH_VISITS + 2 * PANEL_BATCHES,
             'max_backwards': BATCH_VISITS, 'max_private_updates': MAX_UPDATES,
             'max_batch_visits': BATCH_VISITS,
             'max_checkpoint_writes': 0, 'max_model_seconds': 10500,
             'max_published_json_bytes': 4 * 1024 * 1024,
             'one_window': ONE_WINDOW, 'training_enabled': True,
             'full_epoch_training_allowed': True,
             'canonical_resume_authority': False, 'relaunch_allowed': False,
             'predictive_value_claimed': False,
             'qualification_weights_for_main_fit': False,
             'test_data_used': False}
    return value


def build_recipe(*, source_commit: str) -> dict:
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    value = _fit_recipe_expected()
    value['mechanical_source_commit'] = source_commit
    value['recipe_sha256'] = campaign.canonical_sha256(value)
    return require_recipe(value, source_repo=REPO, source_commit=source_commit)


def require_recipe(value: dict, *, source_repo: Path, source_commit: str,
                   verify_files: bool = True) -> dict:
    import subprocess
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    expected = _fit_recipe_expected()
    _need(verify_files and source_repo == REPO and isinstance(value, dict)
          and set(value) == set(expected) | {'mechanical_source_commit', 'recipe_sha256'}
          and {key: value[key] for key in expected} == expected
          and value['recipe_sha256'] == campaign.canonical_sha256({
              key: item for key, item in value.items() if key != 'recipe_sha256'}),
          'EXACT_RECIPE')
    for revision in (source_commit, value['mechanical_source_commit']):
        _need(isinstance(revision, str) and campaign._COMMIT.fullmatch(revision)
              and subprocess.run(['git', 'merge-base', '--is-ancestor', BASE_COMMIT, revision],
                  cwd=REPO, capture_output=True).returncode == 0, 'SOURCE_ANCESTRY')
    _need(subprocess.run(['git', 'merge-base', '--is-ancestor',
          value['mechanical_source_commit'], source_commit],
          cwd=REPO, capture_output=True).returncode == 0, 'RECIPE_SOURCE_ANCESTRY')
    return dict(value)


def gate_fields() -> dict:
    return {'decision': 'ADMITTED', 'status': 'ADMITTED', 'fold_id': 2,
            'selected_fit_rows': TRAIN_ROWS, 'selected_score_rows': PANEL_ROWS,
            'model_constructions_authorized': 1, 'optimizer_constructions_authorized': 1,
            'model_forwards_authorized': BATCH_VISITS + 2 * PANEL_BATCHES,
            'backward_calls_authorized': BATCH_VISITS,
            'optimizer_steps_authorized': MAX_UPDATES,
            'training_enabled': True, 'native_training_enabled': True,
            'genuine_predictor_fit_authorized': True,
            'genuine_backward_authorized': True,
            'genuine_updates_authorized': True,
            'full_epoch_training_allowed': True,
            'canonical_resume_authority': False, 'candidate_selection': False,
            'native_launch_authorized': False, 'normalization_fit_authorized': False,
            'synthetic_test_execution_authorized': False,
            'full_val_allowed': False, 'exit_evaluation': False,
            'broker_access_allowed': False, 'test_data_used': False}


def require_gate(gate: dict, recipe: dict) -> None:
    _need(isinstance(gate, dict)
          and all(type(gate.get(key)) is type(expected) and gate[key] == expected
                  for key, expected in gate_fields().items())
          and recipe['run_scope'] == SCOPE and recipe['admission_stage'] == STAGE,
          'POLICY_GATE')


def cursor_from_batch_visits(visits: int) -> int:
    _need(type(visits) is int and 0 <= visits <= BATCH_VISITS, 'CURSOR_VISITS')
    return min(BATCH_SIZE * visits, TRAIN_ROWS)


def batch_visits_from_cursor(parent_cursor: int) -> int:
    _need(type(parent_cursor) is int and 0 <= parent_cursor <= TRAIN_ROWS
          and (parent_cursor == TRAIN_ROWS or parent_cursor % BATCH_SIZE == 0),
          'PARENT_CURSOR')
    visits = (parent_cursor + BATCH_SIZE - 1) // BATCH_SIZE
    _need(cursor_from_batch_visits(visits) == parent_cursor, 'CURSOR_PARTITION')
    return visits


def build_cursor(*, recipe_ref: dict, outcome: str, optimizer_updates: int,
                 parent_cursor: int, parent_order_sha256: str,
                 full_state, result: dict, batch_size: int = BATCH_SIZE) -> dict:
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    value = {'schema_version': CURSOR_SCHEMA, 'recipe': recipe_ref,
             'outcome': outcome, 'batch_visits': batch_visits_from_cursor(parent_cursor),
             'optimizer_updates': optimizer_updates, 'parent_cursor': parent_cursor,
             'parent_order_sha256': parent_order_sha256, 'full_state': full_state,
             'result': result, 'canonical_resume_authority': False,
             'test_data_used': False}
    value['cursor_sha256'] = campaign.canonical_sha256(value)
    return require_cursor(value, expected_recipe=recipe_ref, verify_files=False,
                          batch_size=batch_size)


def require_cursor(value: dict, *, expected_recipe: dict,
                   verify_files: bool = True, batch_size: int = BATCH_SIZE) -> dict:
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    fields = {'schema_version', 'recipe', 'outcome', 'batch_visits',
              'optimizer_updates', 'parent_cursor', 'parent_order_sha256',
              'full_state', 'result', 'canonical_resume_authority', 'test_data_used'}
    _need(type(batch_size) is int and batch_size == BATCH_SIZE
          and isinstance(value, dict) and set(value) == fields | {'cursor_sha256'}
          and value['schema_version'] == CURSOR_SCHEMA
          and value['cursor_sha256'] == campaign.canonical_sha256({
              key: item for key, item in value.items() if key != 'cursor_sha256'}),
          'CURSOR_IDENTITY')
    _need(expected_recipe.get('path') == str(RECIPE_PATH)
          and value['recipe'] == expected_recipe
          and value['outcome'] in ('COMPLETE', 'FAILED')
          and type(value['batch_visits']) is int
          and 0 <= value['batch_visits'] <= BATCH_VISITS
          and value['parent_cursor'] == cursor_from_batch_visits(value['batch_visits'])
          and type(value['optimizer_updates']) is int
          and 0 <= value['optimizer_updates'] <= value['batch_visits']
          and value['parent_order_sha256'] == TRAIN_PARENT_ORDER_SHA256
          and value['full_state'] is None
          and value['canonical_resume_authority'] is False
          and value['test_data_used'] is False,
          'CURSOR_SCOPE')
    _need(value['outcome'] != 'COMPLETE'
          or (value['batch_visits'] == BATCH_VISITS
              and value['parent_cursor'] == TRAIN_ROWS), 'CURSOR_COMPLETE')
    _need(value['result'].get('path') == str(ROOT / 'RESULT.json'), 'RESULT_PATH')
    campaign.require_binding(value['result'], label='fold2 fit result',
                             verify_file=verify_files)
    if verify_files:
        report = campaign.read_bound_json(Path(value['result']['path']),
                                          value['result']['sha256'])
        _need(report.get('schema_version') == RESULT_SCHEMA
              and report.get('recipe') == expected_recipe
              and report.get('outcome') == value['outcome']
              and report.get('batch_visits') == value['batch_visits']
              and report.get('optimizer_updates') == value['optimizer_updates']
              and report.get('parent_cursor') == value['parent_cursor']
              and report.get('parent_order_sha256') == TRAIN_PARENT_ORDER_SHA256
              and report.get('checkpoint_writes') == 0
              and report.get('canonical_resume_authority') is False
              and report.get('test_data_used') is False,
              'CURSOR_RESULT')
        if value['outcome'] == 'COMPLETE':
            _need(report.get('status') == 'PASS_FOLD2_M5_FIT_AND_READOUT'
                  and report.get('source_unchanged') is True
                  and report.get('batch_visit_in_flight_at_terminal') is False
                  and report.get('optimizer_updates_is_lower_bound') is False
                  and type(report.get('skipped_batches')) is int
                  and report['optimizer_updates'] + report['skipped_batches'] == BATCH_VISITS
                  and report.get('initial_result') == report.get('recipe_inputs', {}).get('initial_result')
                  and set(report.get('final_captures', {})) == {'TRAIN', 'LATER'}
                  and set(report.get('readouts', {})) == {'TRAIN', 'LATER'}
                  and set(report.get('measurements', {})) == {'TRAIN', 'LATER'}
                  and report.get('paired_readout', {}).get('path') == str(ROOT / 'PAIRED_READOUT.json')
                  and isinstance(report.get('fit'), dict)
                  and report['fit'].get('batch_visits') == BATCH_VISITS
                  and report['fit'].get('optimizer_updates') == report['optimizer_updates']
                  and report['fit'].get('skipped_batches') == report['skipped_batches']
                  and report['fit'].get('parent_cursor') == TRAIN_ROWS
                  and len(report['fit'].get('training_records', [])) == BATCH_VISITS
                  and report.get('predictive_edge_proven') is False
                  and report.get('economic_value_measured') is False,
                  'COMPLETE_RESULT')
            for label, binding in report['final_captures'].items():
                _need(binding.get('path') == str(ROOT / ('FINAL_' + label + '_READOUT.npz')),
                      'FINAL_CAPTURE_PATH')
                campaign.require_binding(binding, label='final ' + label, verify_file=True)
            campaign.require_binding(report['paired_readout'], label='paired readout',
                                     verify_file=True)
            paired = campaign.read_bound_json(Path(report['paired_readout']['path']),
                                              report['paired_readout']['sha256'])
            _need(set(paired.get('schemes', {})) == {'UTC_DAY', 'UTC_MONDAY_WEEK'}
                  and type(paired.get('development_ranking_signal_qualified')) is bool
                  and paired.get('test_data_used') is False,
                  'PAIRED_READOUT_RESULT')
    return dict(value)
