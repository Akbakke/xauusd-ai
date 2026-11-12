"""Two bounded passes over the original mature fold-1 prefix, including tail6.

Reuse the activity-only COVERAGE INITIAL and frozen inputs. This is repeated
training of an already seen prefix, not full-history training or resume authority.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
from pathlib import Path
import subprocess

from gx1.contracts import market_state_ta_coverage_benchmark_v1 as coverage
from gx1.contracts import market_state_ta_combined_decoder_benchmark_v1 as qualified

base, campaign, anchor, REPO = coverage.base, coverage.campaign, coverage.anchor, coverage.REPO
BASE_COMMIT = 'edba13492c8a5cffe8ea205acac26c83a0a220bf'
STAGE = 'market_state_ta_full_prefix_learning_20261112_001'
SCOPE = 'market_state_ta_full_prefix_learning_v1'
ROOT = coverage.ROOT.parent / STAGE.upper()
RUNTIME_ROOT, RECIPE_PATH = ROOT / 'RUNTIME', ROOT / 'RECIPE.json'
RECIPE_SCHEMA = 'gx1_ta_full_prefix_learning_recipe_v1'
RESULT_SCHEMA = 'gx1_ta_full_prefix_learning_result_v1'
CURSOR_SCHEMA = 'gx1_ta_full_prefix_learning_cursor_v1'
OWNER = 'gx1/contracts/market_state_ta_full_prefix_learning_v1.py'
WORKER = base.WORKER
ARMS, PASSES, EVALUATION_PASSES = ('FULL_PREFIX',), (1, 2), (0, 1, 2)
PREFIX_ROWS, BATCHES_PER_PASS, TRAINING_PRESENTATIONS = 29574, 1849, 59148
TRAIN_BATCH_INDICES, LATER_BATCH_INDICES = list(base.TRAIN_BATCH_INDICES), list(base.LATER_BATCH_INDICES)
FIXED_PASS0_QUERY_POLICY = base.FIXED_PASS0_QUERY_POLICY
MODEL_SHA256, SOURCE_MODEL_SHA256 = base.MODEL_SHA256, base.SOURCE_MODEL_SHA256
INITIAL_RNG_SHA256, VIEW_REF, LATER_REFS = base.INITIAL_RNG_SHA256, base.VIEW_REF, base.LATER_REFS
CONTROL_RESULT_REF, CONTROL_RECIPE_REF = qualified.CONTROL_RESULT_REF, qualified.CONTROL_RECIPE_REF
CONTROL_TERMINAL_REF = qualified.CONTROL_TERMINAL_REF
PREPARATION_REF = dict(path=str(ROOT.parent / 'MARKET_STATE_TA_FULL_PREFIX_LEARNING_PREPARATION_20261112_001' / 'RESULT.json'),
    sha256='f741414d8893709b15d248236bcc79f5e26865d755d9bad066abaf0c27584522')
CONTROL_REFS = dict(full_prefix_control_result=CONTROL_RESULT_REF,
    full_prefix_control_recipe=CONTROL_RECIPE_REF, full_prefix_control_terminal=CONTROL_TERMINAL_REF,
    full_prefix_preparation=PREPARATION_REF,
    full_prefix_source_qualification=qualified.SOURCE_QUALIFICATION_REF,
    full_prefix_metadata_qualification=qualified.METADATA_QUALIFICATION_REF)
EXTRA_OWNERS = frozenset({OWNER, qualified.OWNER})
MEASUREMENT_LABELS = ('INITIAL_TRAIN', 'INITIAL_LATER') + tuple(
    f'FULL_PREFIX_PASS{p}_{panel}' for p in PASSES for panel in ('TRAIN', 'LATER'))
BUDGETS = dict(max_model_constructions=1, max_optimizer_constructions=1,
    max_checkpoint_loads=1, max_checkpoint_writes=0, max_full_state_restores=1,
    max_initial_rng_imports=1, max_training_forwards=3698, max_evaluation_forwards=1536,
    max_forwards=5234, max_backwards=3698, max_private_updates=3698,
    max_batch_visits=3698, max_candidate_evidence_writes=1,
    max_control_capture_archive_decodes=2, max_candidate_evidence_bytes=1024**3,
    max_model_seconds=10500)
_read, _ref = base._read, base._ref


def _need(ok, reason):
    if not bool(ok):
        raise RuntimeError('TA_FULL_PREFIX_LEARNING_' + reason)


def training_batch_indices(pass_number):
    _need(type(pass_number) is int and pass_number in PASSES, 'PASS_NUMBER')
    return list(range(BATCHES_PER_PASS))


def batch_rows(batch_index):
    _need(type(batch_index) is int and 0 <= batch_index < BATCHES_PER_PASS, 'BATCH_INDEX')
    return min(16, PREFIX_ROWS - 16 * batch_index)


def cursor_from_batch_visits(visits):
    _need(type(visits) is int and 0 <= visits <= 3698, 'BATCH_VISITS')
    passes, batches = divmod(visits, BATCHES_PER_PASS)
    return passes * PREFIX_ROWS + batches * 16


def _visits_from_cursor(cursor):
    _need(type(cursor) is int and 0 <= cursor <= TRAINING_PRESENTATIONS, 'CURSOR_BOUNDARY')
    passes, rows = divmod(cursor, PREFIX_ROWS)
    _need(rows % 16 == 0, 'CURSOR_BOUNDARY')
    return passes * BATCHES_PER_PASS + rows // 16


def _require_preparation(metadata):
    _need(metadata['schema_version'] == 'gx1_ta_full_prefix_learning_preparation_v1'
          and metadata['status'] == 'METADATA_QUALIFIED_SOURCE_IMPLEMENTATION_AND_MODEL_ADMISSION_PENDING'
          and metadata['control_result'] == CONTROL_RESULT_REF and metadata['control_recipe'] == CONTROL_RECIPE_REF
          and metadata['feature_view'] == VIEW_REF
          and metadata['training_authorized'] is metadata['execution_authorized'] is metadata['test_data_used'] is False,
          'PREPARATION_ORIGIN')
    schedule = {str(p): training_batch_indices(p) for p in PASSES}
    _need(campaign.canonical_sha256(metadata['batch_indices_by_pass']) == campaign.canonical_sha256(schedule)
          and metadata['evaluation_passes'] == list(EVALUATION_PASSES) and type(metadata['primary_pass']) is int
          and metadata['primary_pass'] == 2, 'FIXED_FULL_PREFIX_SCHEDULE')
    counts = dict(unique_training_rows=29574, training_presentations=59148, passes=2,
        batches_per_pass=1849, batch_visits=3698, required_optimizer_updates=3698,
        required_skips=0, tail_rows_each_pass=6, previous_complement_rows_included=736)
    _need(campaign.canonical_sha256(metadata['counts']) == campaign.canonical_sha256(counts), 'PREPARATION_COUNTS')
    shared = metadata['shared_numerics']
    _need(shared['initial_model_state_sha256'] == MODEL_SHA256
          and shared['initial_rng_state_sha256'] == INITIAL_RNG_SHA256
          and shared['optimizer_initialization'] == 'fresh_adamw_empty_state'
          and shared['fixed_pass0_query_policy'] is True, 'PREPARATION_INITIAL')
    _need(metadata['unchanged_measurement_panels'] == dict(TRAIN=TRAIN_BATCH_INDICES, LATER=LATER_BATCH_INDICES),
          'UNCHANGED_PANELS')
    return metadata


def _expected():
    control = _read(CONTROL_RECIPE_REF)
    metadata = _require_preparation(_read(PREPARATION_REF))
    full = control['numerical_recipe']
    _need(full == metadata['shared_numerics']['objective_recipe']
          == control['objective_recipes']['UNIQUE_COVERAGE'] and not full.get('m5_direction_only', False),
          'UNCHANGED_FULL_OBJECTIVE')
    value = copy.deepcopy(control)
    for key in ('source_owners', 'trace_worker', 'mechanical_source_commit', 'recipe_sha256',
                'common_pass1_model_state_sha256', 'common_pass1_rng_state_sha256'):
        value.pop(key, None)
    refs = {**copy.deepcopy(control['input_refs']), **copy.deepcopy(CONTROL_REFS)}
    value.update(schema_version=RECIPE_SCHEMA, admission_stage=STAGE, run_scope=SCOPE,
        runtime_root=str(RUNTIME_ROOT), execution_binding_path=str(RUNTIME_ROOT / 'EXECUTION_BINDING.json'),
        input_refs=refs, candidate_inputs=copy.deepcopy(refs), objective_recipes={'FULL_PREFIX': copy.deepcopy(full)},
        arms=list(ARMS), passes=list(PASSES), evaluation_passes=list(EVALUATION_PASSES), primary_pass=2,
        batch_indices_by_pass={str(p): training_batch_indices(p) for p in PASSES},
        train_batch_indices=training_batch_indices(1), excluded_complement_batch_indices=[],
        selected_fit_rows=29574, unique_training_rows=29574, training_presentations=59148,
        selected_score_rows=4096, parent_order_rows=29574,
        parent_selection='all_original_mature_fold1_TRAIN_rows_twice_chronological_including_tail6',
        empty_support_policy='abort_before_update_on_empty_support_without_replacement',
        initial_evaluation_forwards=512, later_evaluation_forwards=768,
        dedicated_tf_activity=True, dedicated_non_m5_side=False, **BUDGETS)
    return value, control


def build_recipe(*, source_commit):
    value, control = _expected()
    value.update(source_owners={n: _ref(REPO / n) for n in set(control['source_owners']) | EXTRA_OWNERS},
        trace_worker=_ref(REPO / WORKER), mechanical_source_commit=source_commit)
    value['recipe_sha256'] = campaign.canonical_sha256(value)
    return require_recipe(value, source_repo=REPO, source_commit=source_commit)


def require_recipe(value, *, source_repo, source_commit, verify_files=True):
    expected, control = _expected()
    _need(verify_files and source_repo == REPO and isinstance(value, Mapping)
          and set(value) == set(expected) | {'source_owners', 'trace_worker', 'mechanical_source_commit', 'recipe_sha256'},
          'RECIPE_FIELDS')
    _need(campaign.canonical_sha256({k: value[k] for k in expected}) == campaign.canonical_sha256(expected)
          and value['recipe_sha256'] == campaign.canonical_sha256({k: v for k, v in value.items() if k != 'recipe_sha256'}),
          'FIXED_RECIPE')
    _need(set(value['source_owners']) == set(control['source_owners']) | EXTRA_OWNERS, 'SOURCE_INVENTORY')
    for name, ref in value['source_owners'].items():
        _need(ref['path'] == str(REPO / name), 'SOURCE_PATH')
        campaign.require_binding(ref, label=name, verify_file=True)
    _need(value['trace_worker'] == value['source_owners'][WORKER], 'WORKER_BINDING')
    for revision in (source_commit, value['mechanical_source_commit']):
        _need(isinstance(revision, str) and campaign._COMMIT.fullmatch(revision)
              and subprocess.run(['git', 'merge-base', '--is-ancestor', BASE_COMMIT, revision],
                  cwd=REPO, capture_output=True).returncode == 0, 'SOURCE_ANCESTRY')
    read_origin(value)
    return dict(value)


def read_origin(recipe):
    control_recipe, source_qualification, _, _, _ = qualified._metadata()
    metadata = _require_preparation(_read(PREPARATION_REF))
    _need(all(recipe['input_refs'].get(k) == v for k, v in CONTROL_REFS.items())
          and recipe['input_refs']['feature_view'] == VIEW_REF
          and recipe['numerical_recipe'] == control_recipe['numerical_recipe'] == metadata['shared_numerics']['objective_recipe']
          and recipe['objective_recipes'] == {'FULL_PREFIX': recipe['numerical_recipe']}
          and recipe['expected_view_sha256'] == control_recipe['expected_view_sha256']
          and recipe['dedicated_tf_activity'] is True and recipe['dedicated_non_m5_side'] is False,
          'ORIGIN_REFS_AND_NUMERICS')
    # Reuse the qualified default-compatible source bridge, not its model or gate.
    qualified._require_source_bridge(recipe, control_recipe, source_qualification)
    origin = base.read_origin(control_recipe)
    control, terminal = _read(CONTROL_RESULT_REF), _read(CONTROL_TERMINAL_REF)
    _need(control['outcome'] == terminal['status'] == 'COMPLETE' and terminal['exit_code'] == 0
          and terminal['result'] == CONTROL_RESULT_REF and control['recipe'] == CONTROL_RECIPE_REF
          and control['source_unchanged'] is control['initial_restore_exact'] is control['evaluation_state_unchanged']
              is control['same_training_queries'] is True and control['test_data_used'] is False
          and control['optimizer_updates'] == 768 and control['skipped_batches'] == 0
          and control['initial_model_state_sha256'] == MODEL_SHA256, 'CONTROL_COMPLETE')
    _need(control['arms']['UNIQUE_COVERAGE']['initial_rng_state_sha256'] == INITIAL_RNG_SHA256, 'CONTROL_INITIAL_RNG')
    for panel in ('TRAIN', 'LATER'):
        label = 'INITIAL_' + panel
        ref = control['measurement_refs'][label]
        _need(metadata['initial_measurements'][panel] == ref, 'CONTROL_INITIAL_REF')
        measured = _read(ref)
        _need(measured['evaluated_model_state_sha256'] == MODEL_SHA256 and measured['optimizer_updates'] == 0
              and measured['evaluation_state_unchanged'] is True and measured['test_data_used'] is False
              and measured['measurement']['rows'] == 4096
              and measured['measurement'] == control['measurements'][label]
              and measured['capture'] == control['panel_captures'][label], 'CONTROL_INITIAL_MEASUREMENT')
    origin.update(full_prefix_control=control, full_prefix_metadata=metadata)
    return origin


def gate_fields():
    return dict(decision='ADMITTED', status='ADMITTED', ta_full_prefix_learning_authorized=True,
        fold_id=1, selected_fit_rows=29574, selected_score_rows=4096, training_presentations=59148,
        model_constructions_authorized=1, optimizer_constructions_authorized=1,
        model_forwards_authorized=5234, backward_calls_authorized=3698, optimizer_steps_authorized=3698,
        initial_rng_imports_authorized=1, training_enabled=True, native_training_enabled=True,
        genuine_updates_authorized=True, genuine_backward_authorized=True,
        genuine_predictor_fit_authorized=False, canonical_resume_authority=False,
        candidate_selection=False, test_data_used=False)


def require_gate(gate, recipe):
    _need(isinstance(gate, Mapping) and all(type(gate.get(k)) is type(v) and gate[k] == v
          for k, v in gate_fields().items()) and recipe['run_scope'] == SCOPE
          and recipe['admission_stage'] == STAGE, 'POLICY_GATE')


def preflight_read_only(recipe, *, source_commit=None):
    source_commit = source_commit or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    require_recipe(recipe, source_repo=REPO, source_commit=source_commit)
    return dict(status='PASS_METADATA_ONLY_FULL_PREFIX_LEARNING', run_scope=SCOPE,
        model_constructions=0, data_payload_reads=0, checkpoint_loads=0, optimizer_updates=0,
        unique_training_rows=29574, training_presentations=59148, passes=2, tail_rows_each_pass=6,
        reused_control_initial=True, train_normalization_unchanged=True, test_data_used=False)


def build_cursor(*, recipe_ref, outcome, optimizer_updates, parent_cursor, parent_order_sha256,
                 full_state, result, batch_size=16):
    value = dict(schema_version=CURSOR_SCHEMA, recipe=recipe_ref, outcome=outcome,
        optimizer_updates=optimizer_updates, parent_cursor=parent_cursor,
        parent_order_sha256=parent_order_sha256, full_state=full_state, result=result,
        test_data_used=False, canonical_resume_authority=False)
    value['cursor_sha256'] = campaign.canonical_sha256(value)
    return require_cursor(value, expected_recipe=recipe_ref, verify_files=False, batch_size=batch_size)


def require_cursor(value, *, expected_recipe, verify_files=True, batch_size=16):
    fields = {'schema_version', 'recipe', 'outcome', 'optimizer_updates', 'parent_cursor',
        'parent_order_sha256', 'full_state', 'result', 'test_data_used', 'canonical_resume_authority', 'cursor_sha256'}
    _need(isinstance(value, Mapping) and set(value) == fields and value['schema_version'] == CURSOR_SCHEMA
          and value['cursor_sha256'] == campaign.canonical_sha256({k: v for k, v in value.items() if k != 'cursor_sha256'}),
          'CURSOR_FIELDS')
    visits = _visits_from_cursor(value['parent_cursor'])
    _need(type(batch_size) is int and batch_size == 16 and expected_recipe.get('path') == str(RECIPE_PATH)
          and value['recipe'] == expected_recipe and value['outcome'] in ('COMPLETE', 'FAILED')
          and type(value['optimizer_updates']) is int and 0 <= value['optimizer_updates'] <= visits
          and value['parent_order_sha256'] is value['full_state'] is None
          and value['test_data_used'] is value['canonical_resume_authority'] is False, 'CURSOR_SCOPE')
    _need(isinstance(value['result'], Mapping) and value['result'].get('path') == str(ROOT / 'RESULT.json'), 'RESULT_PATH')
    campaign.require_binding(value['result'], label='full prefix result', verify_file=verify_files)
    if value['outcome'] == 'COMPLETE':
        _need(value['parent_cursor'] == 59148 and value['optimizer_updates'] == 3698, 'COMPLETE_FIXED_UPDATES')
    if verify_files:
        report = _read(value['result'])
        _need(report['schema_version'] == RESULT_SCHEMA and report['outcome'] == value['outcome']
              and report['recipe'] == expected_recipe and report['parent_cursor'] == value['parent_cursor']
              and report['training_presentations'] == value['parent_cursor'] and report['batch_visits'] == visits
              and report['optimizer_updates'] == value['optimizer_updates']
              and report['canonical_resume_authority'] is report['test_data_used'] is False, 'CURSOR_RESULT')
        if value['outcome'] == 'COMPLETE':
            _require_complete(report)
    return dict(value)


def _require_initial_parity(proof):
    _need(isinstance(proof, Mapping) and set(proof) == {'initial'}
          and isinstance(proof['initial'], Mapping) and set(proof['initial']) == {'TRAIN', 'LATER'}, 'CONTROL_PARITY')
    for item in proof['initial'].values():
        _need(isinstance(item, Mapping) and set(item) == {'arrays_exact', 'model_state_exact', 'rows'}
              and item['arrays_exact'] is item['model_state_exact'] is True
              and type(item['rows']) is int and item['rows'] == 4096, 'INITIAL_PANEL_PARITY')


def _require_complete(report):
    expected = dict(model_constructions=1, model_construction_returns=1, optimizer_constructions=1,
        checkpoint_loads=1, resume_state_load_calls=1, resume_state_load_returns=1,
        strict_full_state_restore_calls=1, strict_full_state_restore_returns=1, initial_rng_imports=1,
        training_forwards=3698, evaluation_forwards=1536, forwards=5234, forward_returns=5234,
        candidate_evidence_writes=1, checkpoint_writes=0, control_capture_archive_decodes=2,
        input_sidecar_archive_decodes=12, training_presentations=59148, skipped_batches=0)
    expected.update({k: 3698 for k in ('backwards', 'backward_returns', 'optimizer_step_calls',
        'optimizer_step_returns', 'optimizer_updates', 'batch_visits')})
    _need(all(type(report['counts'].get(k)) is int and report['counts'][k] == v for k, v in expected.items()), 'COMPLETE_COUNTS')
    _need(all(type(report.get(k)) is int and report[k] == v for k, v in dict(optimizer_updates=3698,
        batch_visits=3698, skipped_batches=0, parent_cursor=59148, training_presentations=59148).items()), 'COMPLETE_FIXED_UPDATES')
    _need(all(report.get(k) is True for k in ('initial_restore_exact', 'evaluation_state_unchanged',
        'source_unchanged', 'same_training_queries')) and report['rng_stream_parity'] is None
        and report['optimizer_initialization'] == 'fresh_adamw_empty_state'
        and report['initial_model_state_sha256'] == MODEL_SHA256, 'COMPLETE_INITIAL_AND_PASSIVITY')
    _require_initial_parity(report['full_prefix_control_parity'])
    _need(set(report['arms']) == set(ARMS) and set(report['measurement_refs']) == set(report['panel_captures'])
          == set(MEASUREMENT_LABELS) and set(report['training_records']) == set(ARMS), 'COMPLETE_ARM_AND_PANELS')
    arm = report['arms']['FULL_PREFIX']
    _need(all(type(arm.get(k)) is int and arm[k] == v for k, v in dict(batch_visits=3698,
        optimizer_updates=3698, skipped_batches=0, passes_completed=2, training_presentations=59148).items())
        and arm['initial_model_state_sha256'] == MODEL_SHA256 and arm['initial_rng_state_sha256'] == INITIAL_RNG_SHA256
        and arm['fixed_pass0_query_policy'] is True and set(arm['training_rng_by_pass']) == {'1', '2'}, 'ARM_FIXED_PREFIX')
    for digest in arm['training_rng_by_pass'].values():
        campaign._sha(digest, 'observed full prefix pass RNG')
    campaign._sha(arm['final_model_state_sha256'], 'full prefix final model')
    evidence = arm['evidence_capsule']
    _need(evidence['path'] == str(ROOT / 'FULL_PREFIX_EVIDENCE.pt'), 'EVIDENCE_PATH')
    campaign.require_binding(evidence, label='full prefix evidence only', verify_file=True)
    records = report['training_records']['FULL_PREFIX']
    _need(len(records) == 3698 and [(r['pass_number'], r['original_batch_index'], r['rows']) for r in records]
          == [(p, b, batch_rows(b)) for p in PASSES for b in training_batch_indices(p)]
          and all(type(r['pass_number']) is type(r['original_batch_index']) is type(r['rows']) is int
              and r['skip_update'] is False for r in records), 'ACTUAL_TRAINING_SCHEDULE')
    _need(sum(r['rows'] for r in records) == 59148 and sum(r['rows'] == 6 for r in records) == 2, 'ACTUAL_TAIL_AND_PRESENTATIONS')
    control, chains, state_digests = _read(CONTROL_RESULT_REF), {}, {}
    for label in MEASUREMENT_LABELS:
        ref, capture = report['measurement_refs'][label], report['panel_captures'][label]
        _need(ref['path'] == str(ROOT / (label + '.json'))
              and capture['path'] == str(ROOT / (label + '_READOUT.npz')), 'PANEL_PATH')
        campaign.require_binding(ref, label=label + ' measurement', verify_file=True)
        campaign.require_binding(capture, label=label + ' capture', verify_file=True)
        measured = _read(ref); panel = label.rsplit('_', 1)[1]
        phase = 0 if label.startswith('INITIAL_') else int(label.split('_PASS')[1].split('_')[0])
        _need(measured['panel_batch_indices'] == (LATER_BATCH_INDICES if panel == 'LATER' else TRAIN_BATCH_INDICES)
              and measured['capture'] == capture and measured['evaluation_objective'] == 'FULL_OBJECTIVE'
              and measured['evaluation_state_unchanged'] is True and measured['test_data_used'] is False
              and measured['measurement']['rows'] == 4096 and measured['measurement']['batches'] == 256,
              'MEASUREMENT_SCOPE')
        _need(type(measured['optimizer_updates']) is int and measured['optimizer_updates'] == phase * 1849, 'MEASUREMENT_UPDATES')
        digest = measured['evaluated_model_state_sha256']; campaign._sha(digest, label + ' model')
        if phase == 0:
            _need(digest == MODEL_SHA256 and measured['measurement'] == control['measurements'][label], 'INITIAL_MEASUREMENT_EXACT')
        elif phase == 2:
            _need(digest == arm['final_model_state_sha256'], 'FINAL_MEASUREMENT_STATE')
        _need(phase not in state_digests or state_digests[phase] == digest, 'SAME_PHASE_MODEL')
        state_digests[phase] = digest
        chain = measured['measurement']['observation_chain_sha256']; campaign._sha(chain, label + ' observations')
        _need(panel not in chains or chains[panel] == chain, 'SAME_PANEL_OBSERVATIONS')
        chains[panel] = chain
