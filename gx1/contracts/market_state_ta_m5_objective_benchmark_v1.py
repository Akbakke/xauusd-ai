"""One fixed TRAIN/development comparison; no canonical resume or selection.

Both arms start from the same preserved activity-only INITIAL and use the same
pass-zero queries. The later panel is separately qualified and never fitted.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
from pathlib import Path
import subprocess

from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts import market_state_ta_anchor_span_short_learning_v1 as anchor

REPO = anchor.REPO
BASE_COMMIT = '596970f5bb462d31b720dbc5a50670c90938dbb4'
STAGE = 'market_state_ta_m5_objective_benchmark_20261112_001'
SCOPE = 'market_state_ta_m5_objective_benchmark_v1'
ROOT = REPO.parent.parent / 'GX1_RUNS' / STAGE.upper()
RUNTIME_ROOT = ROOT / 'RUNTIME'
RECIPE_PATH = ROOT / 'RECIPE.json'
RECIPE_SCHEMA = 'gx1_ta_m5_objective_benchmark_recipe_v1'
RESULT_SCHEMA = 'gx1_ta_m5_objective_benchmark_result_v1'
CURSOR_SCHEMA = 'gx1_ta_m5_objective_benchmark_cursor_v1'
WORKER = 'gx1/audit/run_market_state_ta_m5_objective_benchmark_v1.py'
TRAIN_BATCH_INDICES = [i * 1847 // 255 for i in range(256)]
LATER_BATCH_INDICES = [i * 1108 // 255 for i in range(256)]
ARMS = ('FULL_OBJECTIVE', 'M5_DIRECTION_ONLY')
PASSES = (1, 2, 3)
EVALUATION_PASSES = (0, 1, 3)
FIXED_PASS0_QUERY_POLICY = 'repeat_original_pass0_queries_for_every_arm_and_pass'
MODEL_SHA256 = 'bef35dafc43f0b3450cca161049e9daacee55d92d7f1c3187bd82c9c4de972fc'
SOURCE_MODEL_SHA256 = anchor.PARENT_MODEL_SHA
INITIAL_RNG_SHA256 = '67daad3d6238c809023f066b4599c7634db931c73d9229990a3f76a1845289ae'
VIEW_REF = dict(path=str(ROOT / 'FEATURE_VIEW.json'), sha256='2b15bcfe7fec09d83dfc6af71fe406b6c8153aa5d0a5d9262d112fea3ace3d60')
SPAN_RESULT = dict(path=str(anchor.ROOT / 'RESULT.json'), sha256='0ddae87d37b056964740ef48fd56e94ff194082685b5b09a39696563ca9fc3f8')
LATER_ROOT = ROOT.parent / 'MARKET_STATE_TA_M5_OBJECTIVE_LATER_INPUTS_20261112_001'
LATER_REFS = {
    'later_result': dict(path=str(LATER_ROOT / 'RESULT.json'), sha256='f5f8a0e6da728863ceb2efee33c20b2b371cc1555df89524bfb4ba1712a655b6'),
    'later_coverage': dict(path=str(LATER_ROOT / 'COVERAGE.json'), sha256='307f930b304535a4673703a9c4815baf7985d5cd53e63142aa2ac71f58995091'),
    'later_terminal': dict(path=str(LATER_ROOT / 'TERMINAL.json'), sha256='dde568a695041fee8e3a7bc2afc5dddf0a5da48751cf361b8ff90c3f67bff735'),
    'later_plan': dict(path=str(REPO / 'configs/research/MARKET_STATE_TA_M5_OBJECTIVE_LATER_INPUTS_20261112_001.json'), sha256='3f8c1a9b3fa5d694a2dd63c12c75dc486e6fdbc85296bdc3a8a5cfdbbc06bbe3'),
}
EXTRA_OWNERS = anchor.EXTRA_OWNERS | {
    'gx1/contracts/market_state_ta_m5_objective_benchmark_v1.py', WORKER,
    'gx1/audit/qualify_market_state_ta_benchmark_later_inputs_v1.py',
}
MEASUREMENT_LABELS = ('INITIAL_TRAIN', 'INITIAL_LATER') + tuple(
    f'{arm}_PASS{p}_{panel}' for arm in ARMS for p in (1, 3) for panel in ('TRAIN', 'LATER'))
BUDGETS = dict(max_model_constructions=1, max_optimizer_constructions=1,
    max_checkpoint_loads=1, max_checkpoint_writes=0, max_full_state_restores=2,
    max_initial_rng_imports=1, max_training_forwards=1536, max_evaluation_forwards=2560,
    max_forwards=4096, max_backwards=1536, max_private_updates=1536,
    max_batch_visits=1536, max_candidate_evidence_writes=2,
    max_candidate_evidence_bytes=1024**3, max_model_seconds=10000)


def _need(ok, reason):
    if not bool(ok):
        raise RuntimeError('M5_OBJECTIVE_BENCHMARK_' + reason)


def _read(ref):
    return campaign.read_bound_json(Path(ref['path']), ref['sha256'])


def _ref(path):
    p = Path(path)
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def _expected():
    from gx1.contracts.market_state_time_query_fold1_fit_v1 import ta_sf_numerical_recipe
    old, result, view = _read(anchor.PARENT_RECIPE), _read(anchor.PARENT_RESULT), _read(VIEW_REF)
    physical = _read(anchor.PHYSICAL_RESULT)
    full = ta_sf_numerical_recipe(dedicated_m5_side=True, m5_equal_observation=True, dedicated_tf_activity=True)
    only = ta_sf_numerical_recipe(dedicated_m5_side=True, m5_equal_observation=True,
                                 dedicated_tf_activity=True, m5_direction_only=True)
    anchor.require_unchanged_numerics(full, old['numerical_recipe'])
    identities = {'m5_direction_only', 'native_F_loss_contract_sha256', 'contract_sha256'}
    _need(set(only) == set(full) | {'m5_direction_only'} and only['m5_direction_only'] is True
          and all(only[k] == v for k, v in full.items() if k not in identities), 'ONLY_OBJECTIVE_DIFFERS')
    inputs = dict(parent_recipe=anchor.PARENT_RECIPE, parent_result=anchor.PARENT_RESULT,
        physical_result=anchor.PHYSICAL_RESULT, physical_plan=physical['plan'],
        feature_view=VIEW_REF, prefix_packet=view['prefix_normalization']['binding'],
        constructor=view['constructor_metadata']['binding'],
        short_initial_checkpoint=result['initial_state'], endpoint_priors=anchor.PRIORS,
        source_view=old['input_refs']['original_input_ta_candidate_view'],
        span_initial_result=SPAN_RESULT, **LATER_REFS)
    value = copy.deepcopy(old)
    for key in ('source_owners', 'trace_worker', 'mechanical_source_commit', 'recipe_sha256',
                'terminal_rng_state_sha256', 'parent_order_sha256'):
        value.pop(key, None)
    value.update(schema_version=RECIPE_SCHEMA, admission_stage=STAGE, run_scope=SCOPE,
        runtime_root=str(RUNTIME_ROOT), execution_binding_path=str(RUNTIME_ROOT / 'EXECUTION_BINDING.json'),
        input_refs=inputs, candidate_inputs=copy.deepcopy(inputs), numerical_recipe=full,
        objective_recipes=dict(zip(ARMS, (full, only))), expected_view_sha256=view['contract_sha256'],
        expected_model_contract_sha256=full['research_objective_sha256'],
        expected_loss_contract_sha256=full['native_F_loss_contract_sha256'],
        expected_initial_model_state_sha256=MODEL_SHA256, expected_source_model_state_sha256=SOURCE_MODEL_SHA256,
        initial_rng_state_sha256=INITIAL_RNG_SHA256, anchor_span_inputs=True,
        original_source_commit=result['source_commit'], panel_batch_indices=TRAIN_BATCH_INDICES,
        train_batch_indices=TRAIN_BATCH_INDICES, later_batch_indices=LATER_BATCH_INDICES,
        parent_selection='fixed_evenly_spaced_intact_B16_TRAIN_and_development_panels',
        selected_fit_rows=4096, selected_score_rows=4096, parent_order_rows=29574,
        later_parent_start=29849, later_parent_rows=17753, pass_index=0,
        passes=list(PASSES), evaluation_passes=list(EVALUATION_PASSES), primary_pass=3,
        arms=list(ARMS), fixed_pass0_query_policy=FIXED_PASS0_QUERY_POLICY,
        empty_support_policy='skip_optimizer_update_and_preserve_batch_visit',
        initial_evaluation_forwards=512, later_evaluation_forwards=1280,
        optimizer_initialization='fresh_adamw_empty_state', canonical_resume_authority=False,
        candidate_selection=False, test_data_used=False, **BUDGETS)
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
    q = anchor.read_origin(recipe)
    refs = recipe['input_refs']
    _need(all(refs.get(k) == v for k, v in LATER_REFS.items())
          and refs.get('span_initial_result') == SPAN_RESULT and refs.get('feature_view') == VIEW_REF, 'ORIGIN_REFS')
    later = {k: _read(refs[k]) for k in LATER_REFS}
    result, coverage, terminal, plan = (later[k] for k in ('later_result', 'later_coverage', 'later_terminal', 'later_plan'))
    _need(result['status'] == 'COMPLETE' and result['source_unchanged'] is True
          and terminal['status'] == 'COMPLETE' and terminal['source_unchanged'] is True
          and terminal['exit_code'] == 0 and terminal['result'] == LATER_REFS['later_result']
          and terminal['coverage'] == LATER_REFS['later_coverage']
          and result['plan'] == terminal['plan'] == LATER_REFS['later_plan'], 'LATER_COMPLETE')
    _need(coverage['schema_version'] == 'entry_anchor_span_later_coverage_v1'
          and coverage['status'] == 'COMPLETE' and coverage['qualification'] == LATER_REFS['later_result']
          and coverage['view_sha256'] == q['view']['contract_sha256']
          and coverage['full_axes'] == q['view']['full_axes']
          and coverage['normalization_prefix'] == q['view']['prefix']
          and coverage['batch_indices'] == LATER_BATCH_INDICES
          and coverage['score_parent_start'] == 29849 and coverage['score_rows'] == 17753
          and coverage['batch_size'] == 16, 'LATER_FIXED_AXES')
    _need(result['counts']['local_windows'] == 4096 and result['counts']['mtf_windows'] == 20480
          and result['counts']['producer_calls'] == 6
          and all(v['parity']['fields'] == 31 and v['parity']['float32_bit_mismatched_cells'] == 0
                  for v in result['surfaces'].values()) and len(result['surfaces']) == 6, 'LATER_ACTUAL_WINDOW_PROOF')
    span = _read(SPAN_RESULT)
    _need(span['outcome'] == 'COMPLETE' and span['source_unchanged'] is True
          and span['initial_model_state_sha256'] == MODEL_SHA256, 'PRESERVED_EXPANDED_INITIAL')
    _need(q['original_result']['initial_state'] == refs['short_initial_checkpoint']
          and q['original_result']['initial_model_state_sha256'] == SOURCE_MODEL_SHA256
          and q['original_recipe']['initial_rng_state_sha256'] == INITIAL_RNG_SHA256, 'PRESERVED_SOURCE_INITIAL')
    q.update(later)
    return q


def gate_fields():
    return dict(decision='ADMITTED', status='ADMITTED', ta_m5_objective_benchmark_authorized=True,
        fold_id=1, selected_fit_rows=4096, selected_score_rows=4096,
        model_constructions_authorized=1, optimizer_constructions_authorized=1,
        model_forwards_authorized=4096, backward_calls_authorized=1536, optimizer_steps_authorized=1536,
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
    return dict(status='PASS_METADATA_ONLY_M5_OBJECTIVE_BENCHMARK', run_scope=SCOPE,
        model_constructions=0, data_payload_reads=0, checkpoint_loads=0, optimizer_updates=0,
        test_data_used=False, train_normalization_unchanged=True, separately_qualified_later_panel=True)


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
    _need(batch_size == 16 and expected_recipe.get('path') == str(RECIPE_PATH) and value['recipe'] == expected_recipe
          and value['outcome'] in ('COMPLETE', 'FAILED') and type(value['optimizer_updates']) is int
          and type(value['parent_cursor']) is int and 0 <= value['parent_cursor'] <= 24576
          and value['parent_cursor'] % 16 == 0 and 0 <= value['optimizer_updates'] <= value['parent_cursor'] // 16
          and value['parent_order_sha256'] is value['full_state'] is None
          and value['test_data_used'] is value['canonical_resume_authority'] is False, 'CURSOR_SCOPE')
    _need(isinstance(value['result'], Mapping) and value['result'].get('path') == str(ROOT / 'RESULT.json'), 'RESULT_PATH')
    campaign.require_binding(value['result'], label='objective comparison result', verify_file=verify_files)
    if value['outcome'] == 'COMPLETE':
        _need(value['parent_cursor'] == 24576, 'COMPLETE_BATCH_VISITS')
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


def _require_complete(report):
    expected = dict(model_constructions=1, optimizer_constructions=1, checkpoint_loads=1,
        strict_full_state_restore_calls=2, strict_full_state_restore_returns=2, initial_rng_imports=1,
        training_forwards=1536, evaluation_forwards=2560, forwards=4096, forward_returns=4096,
        candidate_evidence_writes=2, checkpoint_writes=0)
    counts = report['counts']
    _need(all(type(counts.get(k)) is int and counts[k] == v for k, v in expected.items()), 'COMPLETE_COUNTS')
    updates = report['optimizer_updates']
    _need(type(updates) is int and 0 <= updates <= 1536
          and all(type(counts.get(k)) is int and counts[k] == updates for k in
                  ('backwards', 'backward_returns', 'optimizer_step_calls', 'optimizer_step_returns', 'optimizer_updates'))
          and report['batch_visits'] == 1536 and type(report['skipped_batches']) is int
          and report['skipped_batches'] == 1536 - updates, 'COMPLETE_UPDATES_AND_SKIPS')
    _need(all(report.get(k) is True for k in ('initial_restore_exact', 'evaluation_state_unchanged',
          'source_unchanged', 'same_training_queries')) and report['optimizer_initialization'] == 'fresh_adamw_empty_state'
          and report['initial_model_state_sha256'] == MODEL_SHA256, 'COMPLETE_INITIAL_AND_PASSIVITY')
    _need(set(report['arms']) == set(ARMS) and set(report['measurement_refs']) == set(MEASUREMENT_LABELS)
          and set(report['panel_captures']) == set(MEASUREMENT_LABELS), 'COMPLETE_ARMS_AND_PANELS')
    for arm in ARMS:
        item = report['arms'][arm]
        _need(item['batch_visits'] == 768 and type(item['optimizer_updates']) is int
              and 0 <= item['optimizer_updates'] <= 768 and item['skipped_batches'] == 768 - item['optimizer_updates']
              and item['passes_completed'] == 3 and item['initial_model_state_sha256'] == MODEL_SHA256
              and item['initial_rng_state_sha256'] == INITIAL_RNG_SHA256
              and item['fixed_pass0_query_policy'] is True, 'ARM_MATCHED_INITIAL_AND_PROGRESS')
        campaign._sha(item['final_model_state_sha256'], arm + ' final model')
        ref = item['evidence_capsule']
        _need(ref['path'] == str(ROOT / (arm + '_EVIDENCE.pt')), 'EVIDENCE_PATH')
        campaign.require_binding(ref, label=arm + ' evidence only', verify_file=True)
    _need(sum(v['optimizer_updates'] for v in report['arms'].values()) == updates
          and sum(v['skipped_batches'] for v in report['arms'].values()) == report['skipped_batches'], 'ARM_TOTALS')
    chains = {}
    for label in MEASUREMENT_LABELS:
        ref, capture = report['measurement_refs'][label], report['panel_captures'][label]
        _need(ref['path'] == str(ROOT / (label + '.json')) and capture['path'] == str(ROOT / (label + '_READOUT.npz')), 'PANEL_PATH')
        campaign.require_binding(ref, label=label + ' measurement', verify_file=True)
        campaign.require_binding(capture, label=label + ' capture', verify_file=True)
        measured = _read(ref)
        panel = 'LATER' if label.endswith('_LATER') else 'TRAIN'
        _need(measured['panel_batch_indices'] == (LATER_BATCH_INDICES if panel == 'LATER' else TRAIN_BATCH_INDICES)
              and measured['evaluation_state_unchanged'] is True and measured['test_data_used'] is False
              and measured['measurement']['rows'] == 4096 and measured['measurement']['batches'] == 256, 'MEASUREMENT_SCOPE')
        if label.startswith('INITIAL_'):
            _need(measured['optimizer_updates'] == 0 and measured['evaluated_model_state_sha256'] == MODEL_SHA256, 'INITIAL_MEASUREMENT')
        else:
            arm = next(a for a in ARMS if label.startswith(a + '_'))
            _need(type(measured['optimizer_updates']) is int and 0 <= measured['optimizer_updates'] <= report['arms'][arm]['optimizer_updates'], 'ARM_MEASUREMENT_UPDATES')
            if '_PASS3_' in label:
                _need(measured['optimizer_updates'] == report['arms'][arm]['optimizer_updates']
                      and measured['evaluated_model_state_sha256'] == report['arms'][arm]['final_model_state_sha256'], 'FINAL_MEASUREMENT')
        chain = measured['measurement']['observation_chain_sha256']
        campaign._sha(chain, label + ' observations')
        if panel in chains:
            _need(chain == chains[panel], 'SAME_PANEL_OBSERVATIONS')
        else:
            chains[panel] = chain
