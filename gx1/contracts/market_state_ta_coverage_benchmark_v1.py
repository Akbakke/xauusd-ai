"""One fixed exposure-schedule contrast against the completed FULL reference.

The three passes contain disjoint original TRAIN batches. This is not a pure
unique-count intervention, a holdout experiment, or canonical resume authority.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
from pathlib import Path
import subprocess

from gx1.contracts import market_state_ta_m5_objective_benchmark_v1 as base

campaign, anchor, REPO = base.campaign, base.anchor, base.REPO
BASE_COMMIT = 'b5eea3146065b48a4ea521c5deba3c6ec729c792'
STAGE = 'market_state_ta_coverage_benchmark_20261112_001'
SCOPE = 'market_state_ta_coverage_benchmark_v1'
ROOT = base.ROOT.parent / STAGE.upper()
RUNTIME_ROOT = ROOT / 'RUNTIME'
RECIPE_PATH = ROOT / 'RECIPE.json'
RECIPE_SCHEMA = 'gx1_ta_coverage_benchmark_recipe_v1'
RESULT_SCHEMA = 'gx1_ta_coverage_benchmark_result_v1'
CURSOR_SCHEMA = 'gx1_ta_coverage_benchmark_cursor_v1'
WORKER = base.WORKER
ARMS = ('UNIQUE_COVERAGE',)
PASSES, EVALUATION_PASSES = base.PASSES, base.EVALUATION_PASSES
TRAIN_BATCH_INDICES = list(base.TRAIN_BATCH_INDICES)  # The unchanged measurement panel.
LATER_BATCH_INDICES = list(base.LATER_BATCH_INDICES)
FIXED_PASS0_QUERY_POLICY = base.FIXED_PASS0_QUERY_POLICY
MODEL_SHA256, SOURCE_MODEL_SHA256 = base.MODEL_SHA256, base.SOURCE_MODEL_SHA256
INITIAL_RNG_SHA256, VIEW_REF = base.INITIAL_RNG_SHA256, base.VIEW_REF
LATER_REFS = base.LATER_REFS
COMMON_PASS1_MODEL_SHA256 = '9dfd2be522800a946599eb87cb14a7da7bef8df276bdec25c16ae91ec0e0f41c'
COMMON_PASS1_RNG_SHA256 = 'b06f3644120220289779083e7591cd3b066fa048c0b55d389c834855c0986a60'
COVERAGE_METADATA_REF = dict(path=str(ROOT.parent / 'MARKET_STATE_TA_COVERAGE_PREPARATION_20261112_001' / 'RESULT.json'),
    sha256='394328000f4974e2f4409c3eadfa2db78d275487f00c8d7ab6d6b3351da2d721')
CONTROL_RESULT_REF = dict(path=str(base.ROOT / 'RESULT.json'), sha256='069d4d7cbb088474016c78be2abcb0a1d1fb20f240413cc35e9d9003730b3a5d')
CONTROL_RECIPE_REF = dict(path=str(base.RECIPE_PATH), sha256='690ae74195017c21f0aa9cdd3558c6cb6a8f4a09e7a797b3ce591fb1ea86026b')
CONTROL_REFS = dict(coverage_metadata=COVERAGE_METADATA_REF,
    coverage_control_result=CONTROL_RESULT_REF, coverage_control_recipe=CONTROL_RECIPE_REF)
EXTRA_OWNERS = base.EXTRA_OWNERS | {'gx1/contracts/market_state_ta_coverage_benchmark_v1.py'}
# Only orchestration changes for this new scope; numerical owners stay exact.
CONTROL_ORCHESTRATION_OWNERS = frozenset({WORKER,
    'gx1/contracts/entry_native_direction_research_campaign_v1.py',
    'gx1/scripts/run_entry_native_direction_research_window_v1.py',
    'gx1/scripts/materialize_market_state_time_query_fold1_fit_campaign_v1.py',
    'scripts/gx1_capped_run.sh'})
MEASUREMENT_LABELS = ('INITIAL_TRAIN', 'INITIAL_LATER') + tuple(
    f'UNIQUE_COVERAGE_PASS{p}_{panel}' for p in (1, 3) for panel in ('TRAIN', 'LATER'))
BUDGETS = dict(max_model_constructions=1, max_optimizer_constructions=1,
    max_checkpoint_loads=1, max_checkpoint_writes=0, max_full_state_restores=1,
    max_initial_rng_imports=1, max_training_forwards=768, max_evaluation_forwards=1536,
    max_forwards=2304, max_backwards=768, max_private_updates=768,
    max_batch_visits=768, max_candidate_evidence_writes=1,
    max_candidate_evidence_bytes=1024**3, max_model_seconds=6500)
_read, _ref = base._read, base._ref


def _need(ok, reason):
    if not bool(ok):
        raise RuntimeError('TA_COVERAGE_BENCHMARK_' + reason)


def _selection():
    first = [i * 1847 // 255 for i in range(256)]
    missing = [i for i in range(1848) if i not in set(first)]
    excluded = [missing[i * 1591 // 45] for i in range(46)]
    pool = [i for i in missing if i not in set(excluded)]
    chosen = [pool[i * 1545 // 511] for i in range(512)]
    return {'1': first, '2': chosen[::2], '3': chosen[1::2]}, excluded


def training_batch_indices(pass_number):
    _need(type(pass_number) is int and pass_number in PASSES, 'PASS_NUMBER')
    return _selection()[0][str(pass_number)]


def _require_preparation(metadata):
    batches, excluded = _selection()
    _need(metadata['status'] == 'METADATA_QUALIFIED_SOURCE_IMPLEMENTATION_AND_MODEL_ADMISSION_PENDING'
          and metadata['benchmark_result'] == CONTROL_RESULT_REF
          and metadata['benchmark_recipe'] == CONTROL_RECIPE_REF
          and metadata['training_authorized'] is metadata['execution_authorized'] is metadata['test_data_used'] is False,
          'METADATA_ORIGIN')
    _need(campaign.canonical_sha256(metadata['batch_indices_by_pass']) == campaign.canonical_sha256(batches)
          and campaign.canonical_sha256(metadata['excluded_complement_batch_indices']) == campaign.canonical_sha256(excluded),
          'FIXED_EXPOSURE_SCHEDULE')
    _need(metadata['counts'] == dict(control_unique_rows=4096, candidate_unique_rows=12288,
        presentations_each=12288, batch_visits_each=768, required_actual_optimizer_updates_each=768,
        required_skips_each=0, excluded_complement_rows=736, partial_last_batch_rows_not_selected=6), 'EXPOSURE_COUNTS')
    shared, common = metadata['shared_numerics'], metadata['common_pass1']
    _need(shared['initial_model_state_sha256'] == MODEL_SHA256
          and shared['initial_rng_state_sha256'] == INITIAL_RNG_SHA256
          and shared['optimizer_initialization'] == 'fresh_adamw_empty_state'
          and shared['fixed_pass0_query_policy'] is True
          and common['model_state_sha256'] == COMMON_PASS1_MODEL_SHA256
          and common['training_rng_sha256'] == COMMON_PASS1_RNG_SHA256
          and common['checkpoint_or_optimizer_state_retained'] is False, 'COMMON_INITIAL_AND_PASS1')
    _need(metadata['unchanged_measurement_panels'] == dict(TRAIN=TRAIN_BATCH_INDICES, LATER=LATER_BATCH_INDICES), 'UNCHANGED_PANELS')
    return metadata


def _expected():
    value, old = base._expected()
    metadata = _require_preparation(_read(COVERAGE_METADATA_REF))
    full = value['numerical_recipe']
    _need(full == metadata['shared_numerics']['objective_recipe'] and not full.get('m5_direction_only', False), 'UNCHANGED_FULL_OBJECTIVE')
    refs = {**value['input_refs'], **copy.deepcopy(CONTROL_REFS)}
    batches, excluded = _selection()
    value.update(schema_version=RECIPE_SCHEMA, admission_stage=STAGE, run_scope=SCOPE,
        runtime_root=str(RUNTIME_ROOT), execution_binding_path=str(RUNTIME_ROOT / 'EXECUTION_BINDING.json'),
        input_refs=refs, candidate_inputs=copy.deepcopy(refs), objective_recipes={'UNIQUE_COVERAGE': copy.deepcopy(full)},
        arms=list(ARMS), batch_indices_by_pass=batches, excluded_complement_batch_indices=excluded,
        selected_fit_rows=12288, unique_training_rows=12288, selected_score_rows=4096,
        parent_selection='common_original_pass1_then_two_disjoint_chronological_TRAIN_sets_excluding_fixed736',
        empty_support_policy='abort_before_update_on_empty_support_without_replacement',
        initial_evaluation_forwards=512, later_evaluation_forwards=768,
        common_pass1_model_state_sha256=COMMON_PASS1_MODEL_SHA256,
        common_pass1_rng_state_sha256=COMMON_PASS1_RNG_SHA256, **BUDGETS)
    return value, old


def build_recipe(*, source_commit):
    value, old = _expected()
    value.update(source_owners={name: _ref(REPO / name) for name in set(old['source_owners']) | EXTRA_OWNERS},
        trace_worker=_ref(REPO / WORKER), mechanical_source_commit=source_commit)
    value['recipe_sha256'] = campaign.canonical_sha256(value)
    return require_recipe(value, source_repo=REPO, source_commit=source_commit)


def require_recipe(value, *, source_repo, source_commit, verify_files=True):
    expected, old = _expected()
    _need(verify_files and source_repo == REPO and isinstance(value, Mapping)
          and set(value) == set(expected) | {'source_owners', 'trace_worker', 'mechanical_source_commit', 'recipe_sha256'}, 'RECIPE_FIELDS')
    _need(campaign.canonical_sha256({k: value[k] for k in expected}) == campaign.canonical_sha256(expected)
          and value['recipe_sha256'] == campaign.canonical_sha256({k: v for k, v in value.items() if k != 'recipe_sha256'}), 'FIXED_RECIPE')
    _need(set(value['source_owners']) == set(old['source_owners']) | EXTRA_OWNERS, 'SOURCE_INVENTORY')
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
    _need(all(recipe['input_refs'].get(k) == v for k, v in CONTROL_REFS.items()), 'CONTROL_REFS')
    origin = base.read_origin(recipe)
    metadata = _require_preparation(_read(COVERAGE_METADATA_REF))
    control, control_recipe = _read(CONTROL_RESULT_REF), _read(CONTROL_RECIPE_REF)
    _need(control['outcome'] == 'COMPLETE' and control['source_unchanged'] is True
          and control['initial_restore_exact'] is control['evaluation_state_unchanged'] is control['same_training_queries'] is True
          and control['rng_stream_parity'] is True and control['test_data_used'] is False
          and control['recipe'] == CONTROL_RECIPE_REF, 'CONTROL_COMPLETE')
    _need(control_recipe['numerical_recipe'] == recipe['numerical_recipe']
          == metadata['shared_numerics']['objective_recipe']
          == control_recipe['objective_recipes']['FULL_OBJECTIVE']
          and control_recipe['expected_view_sha256'] == recipe['expected_view_sha256']
          and recipe['objective_recipes'] == {'UNIQUE_COVERAGE': recipe['numerical_recipe']}, 'CONTROL_NUMERICS_VIEW')
    for name, ref in control_recipe['source_owners'].items():
        _need(ref['path'] == str(REPO / name), 'CONTROL_SOURCE_PATH')
        if name not in CONTROL_ORCHESTRATION_OWNERS:
            campaign.require_binding(ref, label='unchanged control numerical owner ' + name, verify_file=True)
    arm, common = control['arms']['FULL_OBJECTIVE'], metadata['common_pass1']
    _need(arm['optimizer_updates'] == arm['batch_visits'] == 768 and arm['skipped_batches'] == 0
          and arm['initial_model_state_sha256'] == MODEL_SHA256
          and arm['initial_rng_state_sha256'] == INITIAL_RNG_SHA256
          and arm['fixed_pass0_query_policy'] is True and arm['training_rng_by_pass']['1'] == COMMON_PASS1_RNG_SHA256,
          'CONTROL_FULL_ARM')
    records = control['training_records']['FULL_OBJECTIVE']
    _need(len(records) == 768 and len(common['records']) == 256 and records[:256] == common['records']
          and [x['original_batch_index'] for x in records[:256]] == TRAIN_BATCH_INDICES
          and all(x['pass_number'] == 1 and x['skip_update'] is False for x in records[:256]), 'COMMON_PASS1_RECORDS')
    for panel in ('TRAIN', 'LATER'):
        ref = control['measurement_refs']['FULL_OBJECTIVE_PASS1_' + panel]
        _need(common['measurements'][panel] == ref, 'COMMON_PASS1_MEASUREMENT_REF')
        measured = _read(ref)
        _need(measured['evaluated_model_state_sha256'] == COMMON_PASS1_MODEL_SHA256
              and measured['optimizer_updates'] == 256 and measured['evaluation_state_unchanged'] is True,
              'COMMON_PASS1_MEASUREMENT')
    origin.update(coverage_control=control, coverage_metadata=metadata)
    return origin


def gate_fields():
    return dict(decision='ADMITTED', status='ADMITTED', ta_coverage_benchmark_authorized=True,
        fold_id=1, selected_fit_rows=12288, selected_score_rows=4096,
        model_constructions_authorized=1, optimizer_constructions_authorized=1,
        model_forwards_authorized=2304, backward_calls_authorized=768, optimizer_steps_authorized=768,
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
    return dict(status='PASS_METADATA_ONLY_COVERAGE_BENCHMARK', run_scope=SCOPE,
        model_constructions=0, data_payload_reads=0, checkpoint_loads=0, optimizer_updates=0,
        unique_training_rows=12288, excluded_complement_rows=736, reused_control=True,
        train_normalization_unchanged=True, separately_qualified_later_panel=True, test_data_used=False)


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
          and value['cursor_sha256'] == campaign.canonical_sha256({k: v for k, v in value.items() if k != 'cursor_sha256'}), 'CURSOR_FIELDS')
    _need(type(batch_size) is int and batch_size == 16 and expected_recipe.get('path') == str(RECIPE_PATH)
          and value['recipe'] == expected_recipe and value['outcome'] in ('COMPLETE', 'FAILED')
          and type(value['optimizer_updates']) is type(value['parent_cursor']) is int
          and 0 <= value['parent_cursor'] <= 12288 and value['parent_cursor'] % 16 == 0
          and 0 <= value['optimizer_updates'] <= value['parent_cursor'] // 16
          and value['parent_order_sha256'] is value['full_state'] is None
          and value['test_data_used'] is value['canonical_resume_authority'] is False, 'CURSOR_SCOPE')
    _need(isinstance(value['result'], Mapping) and value['result'].get('path') == str(ROOT / 'RESULT.json'), 'RESULT_PATH')
    campaign.require_binding(value['result'], label='coverage result', verify_file=verify_files)
    if value['outcome'] == 'COMPLETE':
        _need(value['parent_cursor'] == 12288 and value['optimizer_updates'] == 768, 'COMPLETE_FIXED_UPDATES')
    if verify_files:
        report = _read(value['result'])
        _need(report['schema_version'] == RESULT_SCHEMA and report['outcome'] == value['outcome']
              and report['recipe'] == expected_recipe and report['parent_cursor'] == value['parent_cursor']
              and report['optimizer_updates'] == value['optimizer_updates']
              and report['batch_visits'] == value['parent_cursor'] // 16
              and report['canonical_resume_authority'] is report['test_data_used'] is False, 'CURSOR_RESULT')
        if value['outcome'] == 'COMPLETE':
            _require_complete(report)
    return dict(value)


def _require_control_parity(proof):
    _need(isinstance(proof, Mapping) and set(proof) == {'initial', 'pass1', 'rng_after_each_pass_exact'}
          and proof['rng_after_each_pass_exact'] is True, 'CONTROL_PARITY')
    _need(set(proof['initial']) == {'TRAIN', 'LATER'} and set(proof['pass1']) == {
        'model_state_exact', 'rng_exact', 'training_records_exact', 'TRAIN', 'LATER'}, 'CONTROL_PROOF_FIELDS')
    _need(all(proof['pass1'][k] is True for k in ('model_state_exact', 'rng_exact', 'training_records_exact')), 'COMMON_PASS1_PARITY')
    for phase in ('initial', 'pass1'):
        for panel in ('TRAIN', 'LATER'):
            item = proof[phase][panel]
            _need(isinstance(item, Mapping) and set(item) == {'arrays_exact', 'model_state_exact', 'rows'}
                  and item['arrays_exact'] is item['model_state_exact'] is True
                  and type(item['rows']) is int and item['rows'] == 4096, 'CONTROL_PANEL_PARITY')


def _require_complete(report):
    expected = dict(model_constructions=1, model_construction_returns=1, optimizer_constructions=1,
        checkpoint_loads=1, resume_state_load_calls=1, resume_state_load_returns=1,
        strict_full_state_restore_calls=1, strict_full_state_restore_returns=1, initial_rng_imports=1,
        training_forwards=768, evaluation_forwards=1536, forwards=2304, forward_returns=2304,
        candidate_evidence_writes=1, checkpoint_writes=0, control_capture_archive_decodes=4,
        input_sidecar_archive_decodes=12)
    expected.update({k: 768 for k in ('backwards', 'backward_returns', 'optimizer_step_calls',
        'optimizer_step_returns', 'optimizer_updates', 'batch_visits')})
    expected['skipped_batches'] = 0
    _need(all(type(report['counts'].get(k)) is int and report['counts'][k] == v for k, v in expected.items()), 'COMPLETE_COUNTS')
    _need(all(type(report.get(k)) is int and report[k] == v for k, v in
        dict(optimizer_updates=768, batch_visits=768, skipped_batches=0, parent_cursor=12288).items()), 'COMPLETE_FIXED_UPDATES')
    _need(all(report.get(k) is True for k in ('initial_restore_exact', 'evaluation_state_unchanged',
        'source_unchanged', 'same_training_queries', 'rng_stream_parity'))
        and report['optimizer_initialization'] == 'fresh_adamw_empty_state'
        and report['initial_model_state_sha256'] == MODEL_SHA256, 'COMPLETE_INITIAL_AND_PASSIVITY')
    _require_control_parity(report['coverage_control_parity'])
    _need(set(report['arms']) == set(ARMS) and set(report['measurement_refs']) == set(report['panel_captures']) == set(MEASUREMENT_LABELS), 'COMPLETE_ARM_AND_PANELS')
    arm = report['arms']['UNIQUE_COVERAGE']; control = _read(CONTROL_RESULT_REF)
    _need(all(type(arm.get(k)) is int and arm[k] == v for k, v in
        dict(batch_visits=768, optimizer_updates=768, skipped_batches=0, passes_completed=3).items())
        and arm['initial_model_state_sha256'] == MODEL_SHA256 and arm['initial_rng_state_sha256'] == INITIAL_RNG_SHA256
        and arm['fixed_pass0_query_policy'] is True
        and arm['training_rng_by_pass'] == control['arms']['FULL_OBJECTIVE']['training_rng_by_pass'], 'ARM_FIXED_EXPOSURE')
    campaign._sha(arm['final_model_state_sha256'], 'coverage final model')
    evidence = arm['evidence_capsule']
    _need(evidence['path'] == str(ROOT / 'UNIQUE_COVERAGE_EVIDENCE.pt'), 'EVIDENCE_PATH')
    campaign.require_binding(evidence, label='coverage evidence only', verify_file=True)
    records = report['training_records']['UNIQUE_COVERAGE']
    _need(len(records) == 768 and records[:256] == control['training_records']['FULL_OBJECTIVE'][:256]
          and [(r['pass_number'], r['original_batch_index']) for r in records]
              == [(p, b) for p in PASSES for b in training_batch_indices(p)]
          and all(r['skip_update'] is False for r in records), 'ACTUAL_TRAINING_SCHEDULE')
    chains = {}
    for label in MEASUREMENT_LABELS:
        ref, capture = report['measurement_refs'][label], report['panel_captures'][label]
        _need(ref['path'] == str(ROOT / (label + '.json')) and capture['path'] == str(ROOT / (label + '_READOUT.npz')), 'PANEL_PATH')
        campaign.require_binding(ref, label=label + ' measurement', verify_file=True)
        campaign.require_binding(capture, label=label + ' capture', verify_file=True)
        measured = _read(ref); panel = label.rsplit('_', 1)[1]
        _need(measured['panel_batch_indices'] == (LATER_BATCH_INDICES if panel == 'LATER' else TRAIN_BATCH_INDICES)
              and measured['capture'] == capture and measured['evaluation_objective'] == 'FULL_OBJECTIVE'
              and measured['evaluation_state_unchanged'] is True and measured['test_data_used'] is False
              and measured['measurement']['rows'] == 4096 and measured['measurement']['batches'] == 256, 'MEASUREMENT_SCOPE')
        updates, digest = ((0, MODEL_SHA256) if label.startswith('INITIAL_') else
            (256, COMMON_PASS1_MODEL_SHA256) if '_PASS1_' in label else (768, arm['final_model_state_sha256']))
        _need(type(measured['optimizer_updates']) is int and measured['optimizer_updates'] == updates
              and measured['evaluated_model_state_sha256'] == digest, 'MEASUREMENT_STATE')
        chain = measured['measurement']['observation_chain_sha256']; campaign._sha(chain, label + ' observations')
        if panel in chains:
            _need(chain == chains[panel], 'SAME_PANEL_OBSERVATIONS')
        else:
            chains[panel] = chain
