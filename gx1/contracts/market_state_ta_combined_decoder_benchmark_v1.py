"""One combined-decoder comparison against the completed coverage control.

Reuse its fixed inputs, exposure schedule and loss. The target INITIAL digest is
measured after strict side-row cloning; this contract grants no run or resume.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import math
from pathlib import Path
import subprocess

from gx1.contracts import market_state_ta_coverage_benchmark_v1 as coverage

base, campaign, anchor, REPO = coverage.base, coverage.campaign, coverage.anchor, coverage.REPO
BASE_COMMIT = '06227e6c11f4ce4300f9496a4b66a747bbf52b7f'
STAGE = 'market_state_ta_combined_decoder_benchmark_20261112_001'
SCOPE = 'market_state_ta_combined_decoder_benchmark_v1'
ROOT = coverage.ROOT.parent / STAGE.upper()
RUNTIME_ROOT, RECIPE_PATH = ROOT / 'RUNTIME', ROOT / 'RECIPE.json'
RECIPE_SCHEMA = 'gx1_ta_combined_decoder_benchmark_recipe_v1'
RESULT_SCHEMA = 'gx1_ta_combined_decoder_benchmark_result_v1'
CURSOR_SCHEMA = 'gx1_ta_combined_decoder_benchmark_cursor_v1'
OWNER = 'gx1/contracts/market_state_ta_combined_decoder_benchmark_v1.py'
WORKER = coverage.WORKER
ARMS = ('COMBINED_DECODER',)
PASSES, EVALUATION_PASSES = coverage.PASSES, coverage.EVALUATION_PASSES
TRAIN_BATCH_INDICES, LATER_BATCH_INDICES = list(coverage.TRAIN_BATCH_INDICES), list(coverage.LATER_BATCH_INDICES)
FIXED_PASS0_QUERY_POLICY = coverage.FIXED_PASS0_QUERY_POLICY
SOURCE_INITIAL_SHA256, SOURCE_MODEL_SHA256 = coverage.MODEL_SHA256, coverage.SOURCE_MODEL_SHA256
MODEL_SHA256 = None  # The new state inventory has no pre-measured market INITIAL.
INITIAL_RNG_SHA256 = coverage.INITIAL_RNG_SHA256
QUALIFICATION_ROOT = ROOT.parent / 'MARKET_STATE_TA_COMBINED_DECODER_SOURCE_20261112_001'
SOURCE_QUALIFICATION_REF = dict(path=str(QUALIFICATION_ROOT / 'SOURCE_QUALIFICATION.json'),
    sha256='4cc3c3869cceff5a7b856253d96baab969e131d1a85f426b4612279503826327')
METADATA_QUALIFICATION_REF = dict(path=str(QUALIFICATION_ROOT / 'METADATA_QUALIFICATION.json'),
    sha256='05df7e2eb54d87c35c50ec87f21d6ec236d7132b16259b419d6392c9f3abe947')
SOURCE_VIEW_REF = dict(path=str(QUALIFICATION_ROOT / 'SOURCE_FEATURE_VIEW.json'),
    sha256='668c70a079b7b835ea77225c47fd6cfd044ab65aaaa4617b9a4094f9752923c2')
VIEW_REF = TARGET_VIEW_REF = dict(path=str(QUALIFICATION_ROOT / 'TARGET_FEATURE_VIEW.json'),
    sha256='10446fab1b61ead0c5ead71d97f80d7bd3fb26c4ffaadfa3ee59e67fb75dfefb')
NUMERICAL_REF = dict(path=str(QUALIFICATION_ROOT / 'NUMERICAL_RECIPE.json'),
    sha256='c896fdb33715189b2a6199f549481c39ee25edc9bb688ed9f7b6577d6eab76b4')
CONTROL_RESULT_REF = dict(path=str(coverage.ROOT / 'RESULT.json'),
    sha256='d5ba8fa99c87ee3f639770c5b042fe93a442ef4f200733f13fe21cb30a8ac9e0')
CONTROL_RECIPE_REF = dict(path=str(coverage.RECIPE_PATH),
    sha256='673fcc4a2f8d8273324b935ec7078df637ebc7c8425eff62d0248bb0d4fea5ad')
CONTROL_TERMINAL_REF = dict(path=str(coverage.ROOT / 'TERMINAL.json'),
    sha256='e43be2ee9e46085039ecf7a995ccc77e38c52713375bfdf54c238e24ae9b49cd')
CONTROL_REFS = dict(combined_control_result=CONTROL_RESULT_REF, combined_control_recipe=CONTROL_RECIPE_REF,
    combined_control_terminal=CONTROL_TERMINAL_REF, combined_source_qualification=SOURCE_QUALIFICATION_REF,
    combined_metadata_qualification=METADATA_QUALIFICATION_REF, source_initial_view=SOURCE_VIEW_REF,
    target_numerical_recipe=NUMERICAL_REF)
QUALIFIED_NUMERICAL_OWNERS = frozenset({
    'gx1/contracts/market_state_time_query_model_v2.py',
    'gx1/contracts/market_state_time_query_native_F_model_v1.py',
    'gx1/contracts/market_state_time_query_fold1_fit_state_v1.py',
    'gx1/contracts/entry_native_direction_feature_view_v1.py',
    'gx1/contracts/entry_anchor_span_feature_view_v1.py'})
ORCHESTRATION_OWNERS = coverage.CONTROL_ORCHESTRATION_OWNERS | {
    'gx1/audit/trace_market_state_ta_m5_original_cuda_prefix_v1.py'}
EXTRA_OWNERS = coverage.EXTRA_OWNERS | {OWNER}
MEASUREMENT_LABELS = ('INITIAL_TRAIN', 'INITIAL_LATER') + tuple(
    f'COMBINED_DECODER_PASS{p}_{panel}' for p in (1, 3) for panel in ('TRAIN', 'LATER'))
BUDGETS = dict(coverage.BUDGETS, max_model_constructions=2, max_control_capture_archive_decodes=2)
INITIAL_PARITY = dict(rtol=1e-6, atol=2e-6, loss_atol=5e-6)
_read, _ref = base._read, base._ref
training_batch_indices = coverage.training_batch_indices


def _need(ok, reason):
    if not bool(ok):
        raise RuntimeError('TA_COMBINED_DECODER_BENCHMARK_' + reason)


def _differences(a, b):
    return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}


def _metadata():
    control, qualified, metadata = (_read(r) for r in
        (CONTROL_RECIPE_REF, SOURCE_QUALIFICATION_REF, METADATA_QUALIFICATION_REF))
    original, source, target, numerical = (_read(r) for r in
        (base.VIEW_REF, SOURCE_VIEW_REF, TARGET_VIEW_REF, NUMERICAL_REF))
    _need(qualified['status'] == 'SOURCE_QUALIFIED_NATIVE_BASELINE_AND_MATCHED_COMPARISON_PENDING'
          and qualified['metadata'] == METADATA_QUALIFICATION_REF
          and qualified['production_files_changed'] == 5
          and {n for n in qualified['source_hashes'] if not n.startswith('tests/')} == QUALIFIED_NUMERICAL_OWNERS
          and qualified['forward_and_loss_arithmetic_changed'] is qualified['training_authorized']
              is qualified['test_data_used'] is False
          and qualified['initialization']['new_tensors'] == 16
          and qualified['initialization']['new_parameters'] == 1032, 'SOURCE_QUALIFICATION')
    _need(metadata['status'] == 'PASS_CURRENT_REAL_METADATA_BINDINGS'
          and metadata['original_recipe'] == CONTROL_RECIPE_REF and metadata['original_view'] == base.VIEW_REF
          and metadata['SOURCE_FEATURE_VIEW.json'] == SOURCE_VIEW_REF
          and metadata['TARGET_FEATURE_VIEW.json'] == TARGET_VIEW_REF
          and metadata['NUMERICAL_RECIPE.json'] == NUMERICAL_REF
          and all(metadata[k] is True for k in ('old_view_rebuilt_exact',
              'source_numerical_rebuilt_exact', 'input_normalization_and_constructor_unchanged'))
          and metadata['changed_source_bindings'] == ['time_query_v2_model_source']
          and metadata['test_data_used'] is False, 'METADATA_QUALIFICATION')
    _need(_differences(original, source) == {'source_bindings', 'contract_sha256'}
          and _differences(original['source_bindings'], source['source_bindings']) == {'time_query_v2_model_source'}
          and _differences(source, target) == {'dedicated_non_m5_side', 'research_objective', 'contract_sha256'}
          and 'dedicated_non_m5_side' not in source and target['dedicated_non_m5_side'] is True
          and source['dedicated_tf_activity'] is target['dedicated_tf_activity'] is True,
          'ONLY_QUALIFIED_DECODER_VIEW_CHANGES')
    identities = {'research_objective_sha256', 'native_F_loss_contract_sha256', 'contract_sha256'}
    _need(set(numerical) == set(control['numerical_recipe'])
          and _differences(numerical, control['numerical_recipe']) == identities
          and numerical['research_objective_sha256'] == metadata['target_dual_contract']
          and numerical['native_F_loss_contract_sha256'] == metadata['target_loss_contract']
          and numerical['contract_sha256'] == metadata['target_numerical_contract']
          and target['research_objective']['contract_sha256'] == metadata['target_model_contract']
          and source['research_objective']['contract_sha256'] == metadata['source_model_contract'],
          'UNCHANGED_NUMERICS_AND_NEW_IDENTITIES')
    return control, qualified, source, target, numerical


def _expected():
    control, _, source, target, numerical = _metadata()
    value = copy.deepcopy(control)
    for key in ('source_owners', 'trace_worker', 'mechanical_source_commit', 'recipe_sha256',
                'common_pass1_model_state_sha256', 'common_pass1_rng_state_sha256'):
        value.pop(key, None)
    refs = {**copy.deepcopy(control['input_refs']), **copy.deepcopy(CONTROL_REFS), 'feature_view': TARGET_VIEW_REF}
    value.update(schema_version=RECIPE_SCHEMA, admission_stage=STAGE, run_scope=SCOPE,
        runtime_root=str(RUNTIME_ROOT), execution_binding_path=str(RUNTIME_ROOT / 'EXECUTION_BINDING.json'),
        input_refs=refs, candidate_inputs=copy.deepcopy(refs), numerical_recipe=numerical,
        source_numerical_recipe=copy.deepcopy(control['numerical_recipe']),
        objective_recipes={'COMBINED_DECODER': copy.deepcopy(numerical)}, arms=list(ARMS),
        dedicated_non_m5_side=True, dedicated_tf_activity=True,
        expected_initial_model_state_sha256=None, expected_source_initial_model_state_sha256=SOURCE_INITIAL_SHA256,
        expected_source_initial_view_sha256=source['contract_sha256'], expected_view_sha256=target['contract_sha256'],
        expected_model_contract_sha256=numerical['research_objective_sha256'],
        expected_loss_contract_sha256=numerical['native_F_loss_contract_sha256'],
        initial_parity=copy.deepcopy(INITIAL_PARITY), new_side_parameter_tensors=16, new_side_parameters=1032,
        **BUDGETS)
    return value, control


def build_recipe(*, source_commit):
    value, control = _expected()
    value.update(source_owners={n: _ref(REPO / n) for n in set(control['source_owners']) | {OWNER}},
        trace_worker=_ref(REPO / WORKER), mechanical_source_commit=source_commit)
    value['recipe_sha256'] = campaign.canonical_sha256(value)
    return require_recipe(value, source_repo=REPO, source_commit=source_commit)


def require_recipe(value, *, source_repo, source_commit, verify_files=True):
    expected, control = _expected()
    _need(verify_files and source_repo == REPO and isinstance(value, Mapping)
          and set(value) == set(expected) | {'source_owners', 'trace_worker', 'mechanical_source_commit', 'recipe_sha256'}, 'RECIPE_FIELDS')
    _need(campaign.canonical_sha256({k: value[k] for k in expected}) == campaign.canonical_sha256(expected)
          and value['recipe_sha256'] == campaign.canonical_sha256({k: v for k, v in value.items() if k != 'recipe_sha256'}), 'FIXED_RECIPE')
    _need(set(value['source_owners']) == set(control['source_owners']) | {OWNER}, 'SOURCE_INVENTORY')
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


def _require_source_bridge(recipe, control, qualified):
    _need(QUALIFIED_NUMERICAL_OWNERS <= set(control['source_owners']), 'QUALIFIED_OWNER_INVENTORY')
    for name, original in control['source_owners'].items():
        current = recipe['source_owners'][name]
        _need(original['path'] == current['path'] == str(REPO / name), 'CONTROL_SOURCE_PATH')
        if name in QUALIFIED_NUMERICAL_OWNERS:
            _need(current['sha256'] == qualified['source_hashes'][name], 'QUALIFIED_NUMERICAL_SOURCE:' + name)
        elif name not in ORCHESTRATION_OWNERS:
            _need(current == original, 'UNCHANGED_CONTROL_SOURCE:' + name)
        campaign.require_binding(current, label='combined current owner ' + name, verify_file=True)


def _bind_views(origin, source, target):
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts.market_state_time_query_fold1_fit_v1 import ta_sf_numerical_recipe
    normal = origin['receipts']['fold1']['normal']
    for view, non_m5 in ((source, False), (target, True)):
        views.bind_native_direction_feature_view(prefix_normalization=normal,
            prefix_normalization_ref=view['prefix_normalization']['binding'],
            constructor_metadata=origin['receipts']['constructor'],
            constructor_metadata_ref=view['constructor_metadata']['binding'], source_bindings=view['source_bindings'],
            max_batch_rows=16, unobserved_units_profile=normal['unobserved_units_profile'],
            full_feature_profile=True, full_ta_feature_profile=True, market_state_time_query_v2=True,
            dedicated_m5_side=True, dedicated_tf_activity=True, dedicated_non_m5_side=non_m5, expected=view)
    return (ta_sf_numerical_recipe(dedicated_m5_side=True, dedicated_tf_activity=True, m5_equal_observation=True),
        ta_sf_numerical_recipe(dedicated_m5_side=True, dedicated_tf_activity=True,
            dedicated_non_m5_side=True, m5_equal_observation=True))


def _later_binding(original, old_view, source, target):
    _need(original['view_sha256'] == old_view['contract_sha256']
          and all(old_view[k] == source[k] == target[k] for k in
              ('full_axes', 'prefix', 'prefix_normalization', 'constructor_metadata',
               'physical_field_names_sha256', 'anchor_span_input_extension')), 'LATER_UNCHANGED_INPUTS')
    current = copy.deepcopy(original)
    current['view_sha256'] = target['contract_sha256']
    proof = dict(original_coverage=base.LATER_REFS['later_coverage'], original_view=base.VIEW_REF,
        source_view=SOURCE_VIEW_REF, target_view=TARGET_VIEW_REF,
        original_view_sha256=old_view['contract_sha256'], target_view_sha256=target['contract_sha256'],
        changed_fields=['view_sha256'], inputs_unchanged=True)
    return current, proof


def read_origin(recipe):
    control_recipe, qualified, source, target, numerical = _metadata()
    _need(all(recipe['input_refs'].get(k) == v for k, v in CONTROL_REFS.items())
          and recipe['input_refs']['feature_view'] == TARGET_VIEW_REF
          and recipe['source_numerical_recipe'] == control_recipe['numerical_recipe']
          and recipe['numerical_recipe'] == numerical
          and recipe['objective_recipes'] == {'COMBINED_DECODER': numerical}, 'ORIGIN_REFS_AND_NUMERICS')
    _require_source_bridge(recipe, control_recipe, qualified)
    origin = base.read_origin(control_recipe)  # Qualifies the immutable old physical coverage first.
    actual_source, actual_target = _bind_views(origin, source, target)
    _need(actual_source == control_recipe['numerical_recipe'] and actual_target == numerical, 'ACTUAL_RECIPE_IDENTITIES')
    metadata = coverage._require_preparation(_read(coverage.COVERAGE_METADATA_REF))
    control, terminal = _read(CONTROL_RESULT_REF), _read(CONTROL_TERMINAL_REF)
    _need(control['outcome'] == terminal['status'] == 'COMPLETE' and terminal['exit_code'] == 0
          and terminal['result'] == CONTROL_RESULT_REF and control['recipe'] == terminal['recipe'] == CONTROL_RECIPE_REF
          and all(control[k] is True for k in ('source_unchanged', 'initial_restore_exact',
              'evaluation_state_unchanged', 'same_training_queries', 'rng_stream_parity'))
          and terminal['source_unchanged'] is True and control['test_data_used'] is terminal['test_data_used'] is False
          and control['optimizer_updates'] == control['batch_visits'] == 768
          and control['skipped_batches'] == 0 and control['initial_model_state_sha256'] == SOURCE_INITIAL_SHA256,
          'CONTROL_COMPLETE')
    _need(control_recipe['batch_indices_by_pass'] == metadata['batch_indices_by_pass']
          and control_recipe['excluded_complement_batch_indices'] == metadata['excluded_complement_batch_indices']
          and control_recipe['numerical_recipe'] == metadata['shared_numerics']['objective_recipe'], 'CONTROL_FIXED_SCHEDULE')
    original_coverage = origin['later_coverage']
    later, proof = _later_binding(original_coverage, origin['view'], source, target)
    origin.update(view=target, source_initial_view=source,
        source_numerical_recipe=actual_source, target_numerical_recipe=actual_target,
        combined_control=control, coverage_metadata=metadata, original_later_coverage=original_coverage,
        later_coverage=later, later_coverage_binding_proof=proof)
    return origin


def gate_fields():
    value = coverage.gate_fields()
    value.pop('ta_coverage_benchmark_authorized')
    return dict(value, ta_combined_decoder_benchmark_authorized=True, model_constructions_authorized=2)


def require_gate(gate, recipe):
    _need(isinstance(gate, Mapping) and all(type(gate.get(k)) is type(v) and gate[k] == v
          for k, v in gate_fields().items()) and recipe['run_scope'] == SCOPE
          and recipe['admission_stage'] == STAGE, 'POLICY_GATE')


def preflight_read_only(recipe, *, source_commit=None):
    source_commit = source_commit or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    require_recipe(recipe, source_repo=REPO, source_commit=source_commit)
    return dict(status='PASS_METADATA_ONLY_COMBINED_DECODER_BENCHMARK', run_scope=SCOPE,
        model_constructions=0, data_payload_reads=0, checkpoint_loads=0, optimizer_updates=0,
        actual_initial_parity_still_required_before_updates=True, reused_control=True,
        train_normalization_unchanged=True, test_data_used=False)


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
    campaign.require_binding(value['result'], label='combined decoder result', verify_file=verify_files)
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


def _nonnegative_finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _require_control_parity(proof):
    _need(isinstance(proof, Mapping) and set(proof) == {'initial', 'rng_after_each_pass_exact'}
          and proof['rng_after_each_pass_exact'] is True and set(proof['initial']) == {'TRAIN', 'LATER'}, 'CONTROL_PARITY')
    keys = {'rows', 'axes_targets_exact', 'm5_side_exact', 'activity_exact', 'm5_representations_exact',
        'm5_representation_arrays_checked', 'non_m5_side_close', 'canonical_loss_close',
        'non_m5_side_max_abs', 'non_m5_side_max_scaled_error', 'loss_max_abs', *INITIAL_PARITY}
    for panel, item in proof['initial'].items():
        _need(isinstance(item, Mapping) and set(item) == keys
              and type(item['rows']) is int and item['rows'] == 4096
              and all(item[k] is True for k in ('axes_targets_exact', 'm5_side_exact', 'activity_exact',
                  'non_m5_side_close', 'canonical_loss_close'))
              and all(type(item[k]) is float and item[k] == v for k, v in INITIAL_PARITY.items())
              and item['m5_representations_exact'] is (True if panel == 'LATER' else None)
              and type(item['m5_representation_arrays_checked']) is int
              and item['m5_representation_arrays_checked'] == (2 if panel == 'LATER' else 0), 'INITIAL_PANEL_PROOF')
        for key in ('non_m5_side_max_abs', 'non_m5_side_max_scaled_error'):
            _need(isinstance(item[key], Mapping) and set(item[key]) == {'state', 'future'}
                  and all(_nonnegative_finite(v) for v in item[key].values()), 'INITIAL_ERROR_VALUES')
        _need(max(item['non_m5_side_max_scaled_error'].values()) <= 1
              and _nonnegative_finite(item['loss_max_abs']) and item['loss_max_abs'] <= INITIAL_PARITY['loss_atol'],
              'INITIAL_TOLERANCE')


def _require_initial_import(report):
    target = report['initial_model_state_sha256']; campaign._sha(target, 'measured combined INITIAL')
    proof = report['import_proof']; migration, side = proof['migration'], proof['side_migration']
    source_checkpoint = _read(CONTROL_RECIPE_REF)['input_refs']['short_initial_checkpoint']
    _need(target != SOURCE_INITIAL_SHA256 and proof['source_checkpoint'] == source_checkpoint
          and proof['source_initial_model_state_sha256'] == SOURCE_INITIAL_SHA256
          and proof['original_rng_restored'] is True and proof['additional_checkpoint_decode'] is False
          and migration['model_state_sha256'] == SOURCE_INITIAL_SHA256
          and migration['expanded_parameter_tensors'] == 14 and migration['added_parameters'] == 2314
          and migration['buffers_preserved'] is True, 'SPAN_SOURCE_INITIAL')
    clones = {prefix + '.' + tf + '.' + kind: shared + '.' + kind
        for prefix, shared in (('state_model.non_m5_side_heads', 'state_model.query_head'),
                               ('future_non_m5_side_heads', 'future_head'))
        for tf in ('M15', 'H1', 'H4', 'D1') for kind in ('weight', 'bias')}
    _need(side['source_model_state_sha256'] == SOURCE_INITIAL_SHA256 and side['model_state_sha256'] == target
          and side['copied_side_rows'] == clones and side['added_parameters'] == 1032
          and type(side['cursor']) is type(side['optimizer_updates']) is int
          and side['cursor'] == side['optimizer_updates'] == 0
          and side['optimizer_policy'] == 'fresh_AdamW_empty_history_no_inherited_updates', 'SIDE_INITIAL_IMPORT')
    return target


def _require_complete(report):
    expected = dict(model_constructions=2, model_construction_returns=2, optimizer_constructions=1,
        checkpoint_loads=1, resume_state_load_calls=1, resume_state_load_returns=1,
        strict_full_state_restore_calls=1, strict_full_state_restore_returns=1, initial_rng_imports=1,
        training_forwards=768, evaluation_forwards=1536, forwards=2304, forward_returns=2304,
        candidate_evidence_writes=1, checkpoint_writes=0, control_capture_archive_decodes=2,
        input_sidecar_archive_decodes=12, skipped_batches=0)
    expected.update({k: 768 for k in ('backwards', 'backward_returns', 'optimizer_step_calls',
        'optimizer_step_returns', 'optimizer_updates', 'batch_visits')})
    _need(all(type(report['counts'].get(k)) is int and report['counts'][k] == v for k, v in expected.items()), 'COMPLETE_COUNTS')
    _need(all(type(report.get(k)) is int and report[k] == v for k, v in
        dict(optimizer_updates=768, batch_visits=768, skipped_batches=0, parent_cursor=12288).items()), 'COMPLETE_FIXED_UPDATES')
    _need(all(report.get(k) is True for k in ('initial_restore_exact', 'evaluation_state_unchanged',
        'source_unchanged', 'same_training_queries', 'rng_stream_parity'))
        and report['optimizer_initialization'] == 'fresh_adamw_empty_state', 'COMPLETE_PASSIVITY')
    initial = _require_initial_import(report)
    _require_control_parity(report['combined_control_parity'])
    _, later_proof = _later_binding(_read(base.LATER_REFS['later_coverage']),
        _read(base.VIEW_REF), _read(SOURCE_VIEW_REF), _read(TARGET_VIEW_REF))
    _need(report['import_proof']['later_coverage_binding_proof'] == later_proof, 'RETAINED_LATER_BINDING')
    _need(set(report['arms']) == set(ARMS) and set(report['measurement_refs']) == set(report['panel_captures']) == set(MEASUREMENT_LABELS), 'COMPLETE_ARM_AND_PANELS')
    arm = report['arms']['COMBINED_DECODER']; control = _read(CONTROL_RESULT_REF)
    _need(all(type(arm.get(k)) is int and arm[k] == v for k, v in
        dict(batch_visits=768, optimizer_updates=768, skipped_batches=0, passes_completed=3).items())
        and arm['initial_model_state_sha256'] == initial and arm['initial_rng_state_sha256'] == INITIAL_RNG_SHA256
        and arm['fixed_pass0_query_policy'] is True
        and arm['training_rng_by_pass'] == control['arms']['UNIQUE_COVERAGE']['training_rng_by_pass'], 'ARM_MATCHED_EXPOSURE')
    campaign._sha(arm['final_model_state_sha256'], 'combined final model')
    _need(arm['evidence_capsule']['path'] == str(ROOT / 'COMBINED_DECODER_EVIDENCE.pt'), 'EVIDENCE_PATH')
    campaign.require_binding(arm['evidence_capsule'], label='combined evidence only', verify_file=True)
    records = report['training_records']['COMBINED_DECODER']
    _need(len(records) == 768 and [(r['pass_number'], r['original_batch_index']) for r in records]
          == [(p, b) for p in PASSES for b in training_batch_indices(p)]
          and all(r['skip_update'] is False for r in records), 'ACTUAL_TRAINING_SCHEDULE')
    chains, state_by_pass = {}, {0: initial, 3: arm['final_model_state_sha256']}
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
        p = 0 if label.startswith('INITIAL_') else 1 if '_PASS1_' in label else 3
        digest = measured['evaluated_model_state_sha256']; campaign._sha(digest, label + ' model')
        state_by_pass.setdefault(p, digest)
        _need(type(measured['optimizer_updates']) is int and measured['optimizer_updates'] == 256 * p
              and digest == state_by_pass[p], 'MEASUREMENT_STATE')
        chain = measured['measurement']['observation_chain_sha256']; campaign._sha(chain, label + ' observations')
        chains.setdefault(panel, chain)
        _need(chain == chains[panel], 'SAME_PANEL_OBSERVATIONS')
