"""One CPU-capped qualification of fresh native M5 feature/model bindings."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# The executable guard precedes every GX1 source/feature/model import.
_IO_FIREWALL = None
if __name__ == '__main__':
    from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
    _IO_FIREWALL = install_native_m5_direction_io_firewall()

from gx1.audit import build_native_m5_direction_targets_v1 as data
from gx1.contracts.gx1_capped_execution_v1 import (
    require_capped_cpu_research_producer_execution, require_capped_lock_ancestry,
)
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO = data.REPO
POLICY = REPO / 'NEXT_RUN_POLICY.json'
STAGE = 'native_m5_pipeline_20261119_005'
PLAN = REPO / 'configs/research/NATIVE_M5_PIPELINE_20261119_005.json'
ROOT = Path('/home/andre2/GX1_RUNS/NATIVE_M5_PIPELINE_20261119_005')
FIT_START_NS = 1714521600000000000
FIT_END_NS = 1727740800000000000
PARAMETER_START_NS = 1243814400000000000
PARAMETER_END_NS = 1357077600000000000
INNER_END = '2012-04-11T22:00:00+00:00'
SEED, BATCH = 20260911, 16
ARCHITECTURE = {'dropout': .05, 'multi_tf_num_layers': 2, 'multi_tf_scale': .5,
    'specialist_num_layers': 1, 'specialist_fusion_scale': .25,
    'cross_family_fusion_scale': .25}
EXECUTION = {'class': 'producer', 'memory': '12G', 'swap': '512M',
    'outer_seconds': 3600, 'internal_seconds': 3570, 'kill_after_seconds': 10,
    'numerical_threads': 1, 'model_constructions': 1, 'model_forwards': 1,
    'optimizer_steps': 0}
INPUT_KEYS = {'squeeze_fit_plan', 'registry_fit_plan', 'native_source_plan',
              'macro_source_plan', 'target_result', 'target_terminal', 'target_tape',
              'history_verification', 'prefitted_squeeze_manifest', 'prefitted_registry_constants'}
SOURCE_PATHS = (
    'gx1/audit/qualify_market_state_native_m5_pipeline_v1.py',
    'gx1/audit/build_native_m5_direction_targets_v1.py',
    'gx1/contracts/market_state_native_m5_targets_v1.py',
    'gx1/contracts/market_state_native_m5_feature_inputs_v1.py',
    'gx1/contracts/market_state_ta_integrated_model_v1.py',
    'gx1/contracts/market_state_ta_integrated_objective_v1.py',
    'gx1/contracts/entry_model_native_input_normalization_v1.py',
    'gx1/contracts/entry_exit_production_architecture_v1.py',
    'gx1/contracts/entry_model_native_signal_v1.py',
    'gx1/contracts/registry_hyperparameter_fit_v1.py',
    'gx1/contracts/model_state_digest_v1.py',
    'gx1/features/volatility_squeeze_state_v1.py',
    'gx1/features/htf_features.py',
    'gx1/features/entry_specialist_feature_groups_v1.py',
    'gx1/features/entry_model_native_feature_layers_v1.py',
    'gx1/features/d1_macro_context_v1.py',
    'gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py',
    'gx1/scripts/fit_volatility_squeeze_artifacts_v1.py',
    'gx1/scripts/research_ta_campaign_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_qualify_market_state_native_m5_pipeline_v1.py',
)
sha, ref, read, stamp = data.sha, data.ref, data.read, data.stamp


BASE_INPUT_KEYS = INPUT_KEYS.copy()
CANDIDATE_ROOT = Path('/home/andre2/GX1_RUNS/NATIVE_M5_PIPELINE_20261119_003')
CANDIDATE_ROLES = {'candidate_snapshot', 'candidate_archive', 'candidate_support', 'candidate_terminal'}
INPUT_KEYS |= CANDIDATE_ROLES
# These owners produce physical values. Imports/docs and catalogue-only profiles
# are separate contracts; numerical statements and helper bodies must match.
NUMERIC_OWNER_PATHS = (
    'gx1/contracts/market_state_native_m5_feature_inputs_v1.py',
    'gx1/features/htf_features.py', 'gx1/features/entry_model_native_feature_layers_v1.py',
    'gx1/features/technical_indicators_v1.py', 'gx1/features/basic_v1.py',
    'gx1/features/micro_structure_v1.py', 'gx1/features/volume_features.py',
    'gx1/features/entry_foundation_structure_v1.py', 'gx1/features/smc_v1.py',
    'gx1/features/level_registry_v1.py', 'gx1/features/trendline_registry_v1.py',
    'gx1/features/event_age_v1.py', 'gx1/features/volatility_squeeze_state_v1.py',
    'gx1/features/d1_macro_context_v1.py', 'gx1/time/session_detector.py',
    'gx1/scripts/materialize_canonical_v3_augment.py',
    'gx1/scripts/materialize_build_canonical_features_v1.py',
)

SOURCE_PATHS = tuple(dict.fromkeys((*SOURCE_PATHS, *NUMERIC_OWNER_PATHS,
    'gx1/features/gc_native_field_catalog_v1.py',
    'gx1/contracts/native_m5_direction_io_firewall_v1.py',
    'gx1/features/gc_native_context_v1.py',
    'gx1/contracts/entry_gc_mtf_history_v1.py')))


def numeric_ast_identity(source, *, excluded_functions=()):
    import ast
    tree = ast.parse(source)
    # Ignore only import wiring and documentation. Constant assignments and
    # every function/class body remain, including unlisted helper dependencies.
    tree.body = [node for node in tree.body
                 if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                         and node.name in excluded_functions)
                 and not isinstance(node, (ast.Import, ast.ImportFrom))
                 and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                          and isinstance(node.value.value, str))]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node.body = [part for part in node.body
                         if not (isinstance(part, ast.Expr) and isinstance(part.value, ast.Constant)
                                 and isinstance(part.value.value, str))]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def numeric_owner_identities(old_plan):
    result = {}
    for relative in NUMERIC_OWNER_PATHS:
        original = subprocess.check_output(
            ['git', 'show', old_plan['source_base_commit'] + ':' + relative], cwd=REPO)
        if relative in old_plan['source_owners']:
            need(hashlib.sha256(original).hexdigest() == old_plan['source_owners'][relative],
                 'CANDIDATE_ORIGINAL_SOURCE')
        excluded = ('native_m5_feature_provider_contract', 'fit_native_m5_feature_normalization') if (
            relative == 'gx1/contracts/market_state_native_m5_feature_inputs_v1.py') else ()
        old = numeric_ast_identity(original.decode(), excluded_functions=excluded)
        current = numeric_ast_identity((REPO / relative).read_text(), excluded_functions=excluded)
        need(old == current, 'CANDIDATE_NUMERIC_OWNER_CHANGED:' + relative)
        result[relative] = current
    return result


def firewall_state():
    from gx1.contracts.native_m5_direction_io_firewall_v1 import contract
    need(_IO_FIREWALL is not None and _IO_FIREWALL['contract'] == contract()
         and _IO_FIREWALL['denied_attempts'] == 0, 'CLEAN_PROCESS_IO_FIREWALL')
    return json.loads(json.dumps(_IO_FIREWALL))


def validate_candidate_clock_domains(starts, tf_clocks, truth):
    """Prove feature-owner IDs and target IDs in their separate native domains."""
    from gx1.features import htf_features as htf
    issues, source = truth['issue_ns'], truth['source_row']
    n = len(starts)
    need(starts.dtype == np.int64 and starts.ndim == 1 and n > 0
         and np.all(np.diff(starts) > 0) and len(issues) >= n
         and np.array_equal(source[:n], np.arange(n, dtype=np.int64))
         and np.array_equal(starts + data.M5_NS, issues[:n])
         and np.all(starts + data.M5_NS < FIT_END_NS), 'CANDIDATE_NATIVE_SOURCE_AXIS')
    # Feature geometry excludes an initial bin opening before source start.
    # Observed bar facts intentionally retain that partial first bin. Their row
    # numbers therefore cannot be used interchangeably, even on identical bars.
    owner_axes = htf.build_multi_tf_v4_closed_timestamp_indices(pd.to_datetime(starts, utc=True))
    report = {}
    digest = lambda a: hashlib.sha256(np.ascontiguousarray(a, dtype='<i8').tobytes()).hexdigest()
    need(set(tf_clocks) == set(owner_axes), 'CANDIDATE_FIVE_CLOCK_DOMAINS')
    for tf, clock in tf_clocks.items():
        ids, times = clock['owner_row_indices'], clock['ts_int64']
        owner = owner_axes[tf].asi8
        need(ids.dtype == times.dtype == np.int64 and ids.ndim == times.ndim == 1
             and len(ids) == len(times) > 0 and ids[0] >= 0 and ids[-1] < len(owner)
             and np.all(np.diff(ids) > 0) and np.all(np.diff(times) > 0)
             and np.array_equal(owner[ids], times), 'CANDIDATE_FEATURE_OWNER_AXIS:' + tf)
        target = truth['bars_' + tf + '_bar_start_ns']
        available = truth['bars_' + tf + '_available_ns']
        need(target.dtype == available.dtype == np.int64 and target.ndim == 1
             and available.shape == target.shape and np.all(np.diff(target) > 0)
             and np.all(np.diff(available) > 0), 'CANDIDATE_TARGET_CLOCK:' + tf)
        joined = np.searchsorted(target, times)
        need(np.all(joined < len(target)) and np.array_equal(target[joined], times)
             and len(np.unique(joined)) == len(times), 'CANDIDATE_TARGET_TIME_JOIN:' + tf)
        expected_available = times + int(htf.MULTI_TF_SHIFT[tf].value)
        need(np.array_equal(available[joined], expected_available),
             'CANDIDATE_TARGET_AVAILABILITY_JOIN:' + tf)
        offsets, counts = np.unique(joined - ids, return_counts=True)
        report[tf] = {'feature_owner_axis_rows': len(owner), 'retained_rows': len(ids),
            'feature_owner_axis_sha256': digest(owner),
            'retained_feature_owner_ids_sha256': digest(ids), 'target_joined_rows_sha256': digest(joined),
            'bar_start_ns_sha256': digest(times), 'available_ns_sha256': digest(expected_available),
            'target_minus_feature_owner_id_counts': {str(int(k)): int(v) for k, v in zip(offsets, counts)},
            'target_rows_before_feature_axis_start': int(np.searchsorted(target, owner[0])),
            'feature_owner_ids_preserved': True, 'exact_bijective_retained_time_and_availability_join': True}
    return {'schema_version': 'gx1_native_feature_and_bar_fact_clock_domain_join_v1',
        'native_M5_prefix_rows': n, 'native_M5_prefix_exact': True,
        'per_timeframe': report, 'feature_and_target_row_domains_kept_separate': True}


def candidate_clock_proof(inputs, saved):
    """Preclaim clock-only decode; no cached feature matrix or raw price read."""
    need(ref(inputs['candidate_archive']['path']) == inputs['candidate_archive']
         and ref(inputs['target_tape']['path']) == inputs['target_tape'], 'CANDIDATE_CLOCK_ARCHIVE_IDENTITY')
    names = ['bar_start_ns']
    for tf in ('M5', 'M15', 'H1', 'H4', 'D1'):
        names += ['tf_' + tf + '_ts_int64', 'tf_' + tf + '_owner_row_indices']
    clocks = {}
    with np.load(inputs['candidate_archive']['path'], allow_pickle=False) as archive:
        for name in names:
            value = archive[name]
            header = saved['arrays'][name]
            need(value.dtype == np.int64 and value.ndim == 1 and value.dtype.str == header['dtype']
                 and list(value.shape) == header['shape']
                 and hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == header['sha256'],
                 'CANDIDATE_CLOCK_BYTES:' + name)
            clocks[name] = value
    tf_clocks = {tf: {key: clocks['tf_' + tf + '_' + key]
                     for key in ('ts_int64', 'owner_row_indices')}
                 for tf in ('M5', 'M15', 'H1', 'H4', 'D1')}
    with np.load(inputs['target_tape']['path'], allow_pickle=False) as truth:
        proof = validate_candidate_clock_domains(clocks['bar_start_ns'], tf_clocks, truth)
    return {**proof, 'cached_clock_arrays_decoded': len(names),
            'cached_feature_value_arrays_decoded': 0, 'raw_price_values_decoded': 0}


def candidate_metadata(inputs, feature_contract):
    from gx1.contracts.entry_model_native_input_normalization_v1 import native_m5_input_domain_contract
    paths = {'candidate_snapshot': 'FEATURE_INPUTS.json', 'candidate_archive': 'FEATURE_INPUTS.npz',
             'candidate_support': 'FEATURE_SUPPORT.json', 'candidate_terminal': 'TERMINAL.json'}
    need(all(inputs[key]['path'] == str(CANDIDATE_ROOT / name)
             for key, name in paths.items()), 'CANDIDATE_PATHS')
    saved, support, terminal = (read(inputs[key]) for key in
        ('candidate_snapshot', 'candidate_support', 'candidate_terminal'))
    old_plan_ref, old_claim_ref = terminal['plan'], terminal['claim']
    old_plan, old_claim = read(old_plan_ref), read(old_claim_ref)
    need(old_plan_ref['path'] == str(REPO / 'configs/research/NATIVE_M5_PIPELINE_20261119_003.json')
         and old_claim_ref['path'] == str(CANDIDATE_ROOT / 'CLAIM.json')
         and terminal['stage'] == old_plan['stage'] == old_claim['stage']
             == 'native_m5_pipeline_20261119_003'
         and terminal['status'] == 'FAILED_CONSUMED' and terminal['exit_code'] == 1
         and terminal['result'] is None and terminal['relaunch_allowed'] is False
         and terminal['test_data_used'] is False and old_claim['plan'] == old_plan_ref
         and terminal['error'].startswith('RuntimeError:[ENTRY_INPUT_NORMALIZATION_UNSCALEABLE]')
         and all(old_plan['inputs'][key] == inputs[key] for key in BASE_INPUT_KEYS),
         'CANDIDATE_ORIGIN')
    need(saved['schema_version'] == 'gx1_native_m5_bounded_feature_snapshot_v1'
         and saved['archive'] == inputs['candidate_archive']
         and saved['reload_all_arrays_dtype_shape_equal_nan_and_bytes_sha256'] is True
         and saved['model_forwards'] == saved['optimizer_steps'] == 0
         and len(saved['arrays']) == 27 and set(saved['timeframes']) == {'M5','M15','H1','H4','D1'}
         and saved['surfaces_metadata']['provider_contract'] == old_plan['feature_contract']
         and support['feature_source_spec'] == old_plan['feature_source_spec']
         and support['model_forwards'] == support['optimizer_steps'] == 0,
         'CANDIDATE_SNAPSHOT')
    before = saved['surfaces_metadata']['provider_contract']
    strip = lambda value: {key: item for key, item in value.items()
                          if key not in ('execution_profile', 'contract_sha256', 'schema_version',
                                         'input_semantic_domains')}
    need(before['schema_version'] == 'gx1_native_m5_feature_provider_v1'
         and feature_contract['schema_version'] == 'gx1_native_m5_feature_provider_v2'
         and 'input_semantic_domains' not in before
         and feature_contract['input_semantic_domains'] == native_m5_input_domain_contract()
         and strip(before) == strip(feature_contract)
         and before['execution_profile']['name'] == feature_contract['execution_profile']['name'],
         'CANDIDATE_PROVIDER_NUMERIC_CONTRACT')
    return {'original_plan': old_plan_ref, 'original_claim': old_claim_ref,
        'candidate_dependencies': {'plan': old_plan_ref, 'claim': old_claim_ref},
        'cached_clock_admission': candidate_clock_proof(inputs, saved),
        'original_provider_contract_sha256': before['contract_sha256'],
        'admitted_provider_contract_sha256': feature_contract['contract_sha256'],
        'original_execution_profile': before['execution_profile'],
        'admitted_execution_profile': feature_contract['execution_profile'],
        'original_provider_schema': before['schema_version'],
        'admitted_provider_schema': feature_contract['schema_version'],
        'fresh_normalization_semantic_domains': native_m5_input_domain_contract(),
        'permitted_changed_domains': ['explicit_provider_v1_to_v2_semantic_contract',
                                      'independent_static_field_catalog_and_profile_identity',
                                      'fresh_native_binary_and_signed_ternary_normalization_domain_policy'],
        'unchanged_provider_contract_domains': sorted(strip(before)),
        'numeric_owner_ast_sha256': numeric_owner_identities(old_plan),
        'excluded_non_raw_functions': ['native_m5_feature_provider_contract',
                                       'fit_native_m5_feature_normalization'],
        'original_full_process_M1_free': False,
        'new_admission_requires_clean_process_and_exact_physical_array_checks': True}


def need(ok, reason):
    if not bool(ok):
        raise ValueError('NATIVE_M5_PIPELINE_' + reason)


def registry_provenance(binding, plan):
    from gx1.features.volatility_squeeze_state_v1 import VOLATILITY_SQUEEZE_DIRECTION_CLOCKS
    native = plan['native_m5_manifest']
    return {tf: {
        'source_artifact': native['path'], 'source_sha256': native['sha256'],
        'source_schema_version': 'xau_canonical_native_source_v3', 'source_lane': tf,
        'native_m5_manifest_artifact': native['path'],
        'native_m5_manifest_sha256': native['sha256'],
        'fit_plan_artifact': binding['path'], 'fit_plan_sha256': binding['sha256'],
        'source_clock_contract': plan['source_clock_contract'], 'train_split_id': 'TRAIN',
        'declared_train_window_start': plan['declared_train_window_start'],
        'declared_train_window_end': plan['declared_train_window_end'],
    } for tf in VOLATILITY_SQUEEZE_DIRECTION_CLOCKS}


def fit_plans(inputs):
    from gx1.features.volatility_squeeze_state_v1 import (
        _require_native_m5_provenance, volatility_squeeze_bar_grid,
    )
    from gx1.contracts.registry_hyperparameter_fit_v1 import require_registry_fit_source_v1
    squeeze, registry = (read(inputs[key]) for key in ('squeeze_fit_plan', 'registry_fit_plan'))
    need(squeeze['fit_start_ns'] == registry['fit_start_ns'] == PARAMETER_START_NS
         and squeeze['fit_cutoff_ns'] == registry['fit_cutoff_ns'] == PARAMETER_END_NS
         and registry['declared_inner_fit_window_end'] == INNER_END
         and squeeze['source_spec'] == registry['source_spec'], 'PARAMETER_POPULATION')
    provenance = registry_provenance(inputs['registry_fit_plan'], registry)
    for tf, item in provenance.items():
        require_registry_fit_source_v1(item, clock=tf)
        squeeze_item = {k: v for k, v in item.items()
                        if k not in ('source_clock_contract', 'declared_train_window_start',
                                     'declared_train_window_end')}
        squeeze_item.update(clock_contract=squeeze['source_clock_contract'],
            bar_grid=volatility_squeeze_bar_grid(tf),
            fit_plan_artifact=inputs['squeeze_fit_plan']['path'],
            fit_plan_sha256=inputs['squeeze_fit_plan']['sha256'])
        _require_native_m5_provenance(squeeze_item, timeframe=tf,
            fit_cutoff_ns=PARAMETER_END_NS, fit_start_ns=PARAMETER_START_NS)
    return squeeze, registry


def macro_metadata(binding):
    from gx1.features import d1_macro_context_v1 as macro
    spec = macro.load_d1_macro_context_source_spec(REPO)
    need(binding == ref(REPO / macro.D1_MACRO_CONTEXT_SOURCE_SPEC['path'])
         and spec['test_seal_utc'] == '2026-07-01T00:00:00Z', 'MACRO_SPEC')
    dependencies = {}
    results = [spec['cot']['result'], spec['oanda_panel']['result'],
               *(spec['drivers'][key] for key in ('treasury', 'vix', 'gld'))]
    for result_ref in results:
        result = read(result_ref)
        dependencies[result_ref['path']] = result_ref
        need(type(result.get('records')) is list and len(result['records']) > 0,
             'MACRO_RECORDS')
        for record in result['records']:
            need(record['status'] == 'RETRIEVED_NOT_ADMITTED', 'MACRO_INCOMPLETE')
            pair = ('parquet_path', 'parquet_sha256') if 'parquet_path' in record else ('raw_path', 'raw_sha256')
            path, digest = record[pair[0]], record[pair[1]]
            stamp(path)
            need(isinstance(digest, str) and len(digest) == 64
                 and all(c in '0123456789abcdef' for c in digest), 'MACRO_DIGEST')
            value = {'path': path, 'sha256': digest}
            need(path not in dependencies or dependencies[path] == value, 'MACRO_SPLIT_BINDING')
            dependencies[path] = value
    clock = dict(spec['cot']['publication_clock'])
    if not Path(clock['path']).is_absolute():
        clock['path'] = str(REPO / clock['path'])
    read(clock)
    dependencies[clock['path']] = clock
    panel = read(spec['oanda_panel']['result'])
    need(set(spec['oanda_panel']['instruments']).issubset(
        {record['item'] for record in panel['records']}), 'MACRO_PANEL_COMPONENTS')
    return dependencies


def feature_spec(source_plan):
    original = source_plan['source_spec']
    need(original['issue_start_ns'] <= FIT_START_NS
         and original['cutoff_ns'] >= FIT_END_NS, 'FEATURE_SOURCE_SCOPE')
    parts = []
    for part in original['parts']:
        groups = [item['index'] for item in source_plan['source_metadata'][part['path']]['selected_groups']
                  if item['first_bar_ns'] + data.M5_NS < FIT_END_NS]
        if groups:
            parts.append({**part, 'row_groups': groups})
    need(len(parts) > 0, 'FEATURE_SOURCE_EMPTY')
    return {'manifest': original['manifest'], 'parts': parts, 'cutoff_ns': FIT_END_NS}


def reused_history_verification(inputs):
    binding = inputs['history_verification']
    previous_root = Path('/home/andre2/GX1_RUNS/NATIVE_M5_PIPELINE_20261119_001')
    need(binding['path'] == str(previous_root / 'HISTORY_VERIFICATION.json'),
         'HISTORY_REUSE_PATH')
    saved = read(binding)
    old_plan = read(saved['plan'])
    old_claim = read(saved['claim'])
    target = read(inputs['target_result'])
    source = read(inputs['native_source_plan'])
    proof = saved['verification']
    need(saved['plan']['path'] == str(REPO / 'configs/research/NATIVE_M5_PIPELINE_20261119_001.json')
         and saved['claim']['path'] == str(previous_root / 'CLAIM.json')
         and old_plan['stage'] == old_claim['stage'] == 'native_m5_pipeline_20261119_001'
         and old_claim['plan'] == saved['plan']
         and saved['target_tape'] == old_plan['inputs']['target_tape'] == inputs['target_tape']
         and old_plan['inputs']['native_source_plan'] == inputs['native_source_plan']
         and old_plan['target_contract'] == target['target_contract']
         and old_plan['target_source_spec'] == source['source_spec'],
         'HISTORY_REUSE_LINEAGE')
    report = target['report']
    need(proof['schema_version'] == 'native_m5_independent_all_history_facts_verification_v1'
         and target['target_contract']['schema_version'] == 'gx1_market_state_native_m5_bar_directions_v2'
         and proof['array_count'] == len(proof['compared_arrays']) == 76
         and set(proof['compared_arrays']) == set(report['array_shapes'])
         and proof['all_arrays_exact_equal_nan'] is True
         and proof['raw_and_archive_hash_bookends_unchanged'] is True
         and proof['raw_parts'] == len(source['source_spec']['parts'])
         and proof['retained_source_rows'] == report['source_rows']
         and proof['issue_rows'] == report['selected_issue_rows']
         and proof['strict_exclusive_close_cutoff_ns'] == source['source_spec']['cutoff_ns'],
         'HISTORY_REUSE_COMPLETENESS')
    fields = {'observed_bars': 'observed_bars', 'missing_first_previous': 'missing_previous_bars',
              'cross_year_previous_bars': 'cross_year_previous_bars',
              'future_present': 'future_target_present', 'future_mature': 'future_mature',
              'future_eligible': 'future_eligible',
              'future_unique_mature_rows': 'future_unique_mature_bar_rows'}
    need(set(proof['coverage_by_timeframe']) == set(report['coverage_by_timeframe'])
         and all(observed[left] == report['coverage_by_timeframe'][tf][right]
                 for tf, observed in proof['coverage_by_timeframe'].items()
                 for left, right in fields.items()), 'HISTORY_REUSE_COVERAGE')
    return saved


def parameter_artifact_metadata(inputs, geometry):
    """Reload completed native artifacts; validate metadata without decoding prices."""
    from gx1.features.volatility_squeeze_state_v1 import (
        load_volatility_squeeze_artifact_manifest, VOLATILITY_SQUEEZE_DIRECTION_CLOCKS,
        volatility_squeeze_bar_grid,
    )
    from gx1.features.htf_features import require_native_m5_registry_constants
    previous = Path('/home/andre2/GX1_RUNS/NATIVE_M5_PIPELINE_20261119_002')
    squeeze_ref, registry_ref = (inputs[key] for key in (
        'prefitted_squeeze_manifest', 'prefitted_registry_constants'))
    need(squeeze_ref['path'] == str(previous / 'SQUEEZE/manifest.json')
         and registry_ref['path'] == str(previous / 'REGISTRY_CONSTANTS.json'),
         'PARAMETER_REUSE_PATH')
    squeeze_plan, registry_plan = fit_plans(inputs)
    manifest = read(squeeze_ref)
    carrier = load_volatility_squeeze_artifact_manifest(Path(squeeze_ref['path']),
        expected_sha256=squeeze_ref['sha256'], clocks=VOLATILITY_SQUEEZE_DIRECTION_CLOCKS)
    constants = require_native_m5_registry_constants(read(registry_ref))
    start = pd.Timestamp(PARAMETER_START_NS, tz='UTC').isoformat()
    end = pd.Timestamp(PARAMETER_END_NS, tz='UTC').isoformat()
    need(constants['declared_train_window_start'] == start
         and constants['declared_train_window_end'] == end
         and constants['declared_inner_fit_window_end'] == INNER_END
         and constants['entry_m5']['seq_len'] == geometry['local']
         and constants['per_tf_seq_lens'] == {tf: geometry[tf]
             for tf in VOLATILITY_SQUEEZE_DIRECTION_CLOCKS}, 'PARAMETER_REUSE_SCOPE')
    dependencies = {}
    frames = {}
    registry_sources = registry_provenance(inputs['registry_fit_plan'], registry_plan)
    for tf in VOLATILITY_SQUEEZE_DIRECTION_CLOCKS:
        entry = manifest['artifacts'][tf]
        need(entry['params_artifact'] == str(previous / 'SQUEEZE' / (tf.lower() + '_params.json')),
             'PARAMETER_REUSE_INVENTORY')
        dependencies[entry['params_artifact']] = {
            'path': entry['params_artifact'], 'sha256': entry['params_file_sha256']}
        params = carrier.require_params(tf)
        expected = {k: v for k, v in registry_sources[tf].items()
            if k not in ('source_clock_contract', 'declared_train_window_start',
                         'declared_train_window_end')}
        expected.update(clock_contract=squeeze_plan['source_clock_contract'],
            bar_grid=volatility_squeeze_bar_grid(tf),
            fit_plan_artifact=inputs['squeeze_fit_plan']['path'],
            fit_plan_sha256=inputs['squeeze_fit_plan']['sha256'])
        need(params['source_provenance'] == expected
             and params['declared_train_window_start'] == start
             and params['declared_train_window_end'] == end
             and params['train_ohlcv_rows'] == entry['train_ohlcv_rows']
             and params['train_ohlcv_sha256'] == entry['train_ohlcv_sha256'],
             'PARAMETER_REUSE_SQUEEZE_FRAME')
        level = constants['provenance']['level_recurrence_threshold'][tf]
        trend = constants['provenance']['trendline_band'][tf]
        need(level['source_provenance'] == trend['source_provenance'] == registry_sources[tf]
             and level['outer_train_rows'] == trend['outer_train_rows'] > 0
             and level['frame_sha256'] == trend['frame_sha256'],
             'PARAMETER_REUSE_REGISTRY_FRAME')
        frames[tf] = {'squeeze_rows': entry['train_ohlcv_rows'],
            'squeeze_ohlcv_sha256': entry['train_ohlcv_sha256'],
            'registry_rows': level['outer_train_rows'],
            'registry_frame_sha256': level['frame_sha256']}
    entry = constants['provenance']['entry_m5_trendline_band']
    need(entry['source_provenance'] == registry_sources['M5']
         and entry['outer_train_rows'] == frames['M5']['registry_rows']
         == constants['provenance']['n_train_m5_rows']
         and entry['frame_sha256'] == frames['M5']['registry_frame_sha256'],
         'PARAMETER_REUSE_ENTRY_FRAME')
    old_claim_ref = ref(previous / 'CLAIM.json')
    old_claim = read(old_claim_ref)
    old_plan_ref = old_claim['plan']
    old_plan = read(old_plan_ref)
    need(old_plan_ref['path'] == str(REPO / 'configs/research/NATIVE_M5_PIPELINE_20261119_002.json')
         and old_plan['stage'] == old_claim['stage'] == 'native_m5_pipeline_20261119_002'
         and old_plan['parameter_source_spec'] == squeeze_plan['source_spec']
         and all(old_plan['inputs'][key] == inputs[key]
                 for key in ('squeeze_fit_plan', 'registry_fit_plan', 'native_source_plan'))
         and old_plan['geometry'] == geometry, 'PARAMETER_REUSE_ORIGIN')
    dependencies[old_plan_ref['path']] = old_plan_ref
    dependencies[old_claim_ref['path']] = old_claim_ref
    return {'squeeze_manifest': squeeze_ref, 'registry_artifact': registry_ref,
        'squeeze_binding': carrier.binding(), 'registry_contract_sha256': constants['contract_sha256'],
        'original_plan': old_plan_ref, 'original_claim': old_claim_ref,
        'recorded_fit_frames': frames, 'parameter_dependencies': dependencies,
        'parameter_fits': 0,
        'identity_scope': 'recorded_owner_fit_frames_bound_to_original_raw_file_hashes_no_price_decode',
        'frame_domains': 'owner_specific_squeeze_observed_bins_registry_closed_bins_not_equalized'}, constants


def build_plan(*, inputs, source_base_commit=None):
    from gx1.contracts.market_state_native_m5_targets_v1 import contract as target_contract
    from gx1.contracts.market_state_native_m5_feature_inputs_v1 import (
        RAW_COLUMNS, native_m5_feature_provider_contract,
    )
    from gx1.contracts.market_state_ta_integrated_model_v1 import native_bar_backbone_geometry
    from gx1.contracts.market_state_ta_integrated_objective_v1 import native_bar_objective_contract
    from gx1.contracts.native_m5_direction_io_firewall_v1 import contract as firewall_contract
    need(type(inputs) is dict and set(inputs) == INPUT_KEYS, 'INPUT_ROLES')
    squeeze, registry = fit_plans(inputs)
    source = read(inputs['native_source_plan'])
    need(inputs['native_source_plan']['path'] == str(data.PLAN)
         and source.get('stage') == data.STAGE
         and source.get('source_spec', {}).get('issue_start_ns') == data.ISSUE_START_NS
         and source.get('source_spec', {}).get('cutoff_ns') == data.CUTOFF_NS,
         'FULL_HISTORY_003_IDENTITY')
    spec = feature_spec(source)
    target = read(inputs['target_result'])
    terminal = read(inputs['target_terminal'])
    need(target['target_tape'] == terminal['target_tape'] == inputs['target_tape']
         and target['plan'] == terminal['plan'] == inputs['native_source_plan']
         and terminal['result'] == inputs['target_result'] and terminal['status'] == 'COMPLETE'
         and terminal['exit_code'] == 0 and target['source_unchanged'] is True
         and target['target_contract'] == target_contract()
         and target['test_data_used'] is terminal['test_data_used'] is False,
         'TARGET_CHAIN')
    history = reused_history_verification(inputs)
    geometry = native_bar_backbone_geometry()
    parameters, _ = parameter_artifact_metadata(inputs, geometry)
    dependencies = macro_metadata(inputs['macro_source_plan'])
    metadata = {part['path']: data.part_metadata(part) for part in spec['parts']}
    for part in spec['parts']:
        need(all(key in pq.ParquetFile(part['path']).schema_arrow.names
                 for key in ('time', *RAW_COLUMNS)), 'RAW_FEATURE_COLUMNS')
    feature_contract = native_m5_feature_provider_contract()
    candidate = candidate_metadata(inputs, feature_contract)
    old_plan = read(candidate['original_plan'])
    need(old_plan['feature_source_spec'] == spec
         and old_plan['macro_dependencies'] == dependencies
         and old_plan['geometry'] == geometry, 'CANDIDATE_SCOPE')
    head = source_base_commit or subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    return {
        'schema_version': 'gx1_native_m5_pipeline_plan_v1', 'stage': STAGE,
        'status': 'PREPARED_NOT_ADMITTED', 'output_root': str(ROOT),
        'source_base_commit': head, 'source_owners': {p: sha(REPO / p) for p in SOURCE_PATHS},
        'inputs': inputs, 'macro_dependencies': dependencies, 'feature_source_spec': spec,
        'history_lineage': {'plan': history['plan'], 'claim': history['claim']},
        'history_reuse': 'original_001_all_76_array_proof_for_identical_003_target_bytes',
        'parameter_reuse': parameters, 'candidate_requalification': candidate,
        'feature_source_metadata': metadata, 'parameter_source_spec': squeeze['source_spec'],
        'target_source_spec': source['source_spec'], 'target_source_metadata': source['source_metadata'],
        'fit_start_ns': FIT_START_NS, 'fit_end_ns': FIT_END_NS,
        'fit_axis': 'native_M5_issue_in_window_and_input_ready_and_any_mature_supervised_target',
        'normalization_population': 'whole_exact_consumption_union_of_selected_TRAIN_issues',
        'geometry': geometry, 'architecture': ARCHITECTURE,
        'seed': SEED, 'batch_rows': BATCH, 'execution': EXECUTION,
        'io_firewall': firewall_contract(),
        'feature_contract': feature_contract,
        'target_contract': target_contract(), 'objective_contract': native_bar_objective_contract(),
        'metadata_scope': 'genuine_receipts_and_11_cached_clock_arrays_no_feature_or_raw_price_decode',
        'model_forwards': 1, 'model_constructions': 1, 'optimizer_steps': 0,
        'training_authorized': False, 'test_data_used': False, 'economic_edge_proven': False,
    }


def source_clean(plan):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    return (subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip()
            == 'work/gx1-current'
        and not subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)
        and subprocess.run(['git', 'merge-base', '--is-ancestor', plan['source_base_commit'], head],
                           cwd=REPO, capture_output=True).returncode == 0
        and subprocess.run(['git', 'diff', '--quiet', plan['source_base_commit'], head,
                           '--', 'gx1', 'scripts'], cwd=REPO).returncode == 0
        and set(plan['source_owners']) == set(SOURCE_PATHS)
        and all(sha(REPO / path) == digest for path, digest in plan['source_owners'].items()))


def checked_inputs(plan):
    bindings = (list(plan['inputs'].values()) + list(plan['macro_dependencies'].values())
                + list(plan['history_lineage'].values()))
    bindings.extend(plan['parameter_reuse']['parameter_dependencies'].values())
    bindings.extend(plan['candidate_requalification']['candidate_dependencies'].values())
    bindings.append(plan['feature_source_spec']['manifest'])
    for spec in (plan['feature_source_spec'], plan['parameter_source_spec']):
        bindings.extend({'path': part['path'], 'sha256': part['sha256']} for part in spec['parts'])
    seen = {}
    for binding in bindings:
        path = binding['path']
        need(path not in seen or seen[path] == binding, 'INPUT_SPLIT_BINDING')
        if path not in seen:
            need(ref(path) == binding, 'INPUT_IDENTITY')
            seen[path] = binding
    for part in plan['feature_source_spec']['parts']:
        need(data.part_metadata(part) == plan['feature_source_metadata'][part['path']],
             'FEATURE_METADATA')


def inspect(plan_sha, *, admitted):
    need(Path.cwd().resolve() == REPO and not ROOT.exists(), 'CWD_OUTPUT')
    plan_ref = {'path': str(PLAN), 'sha256': plan_sha}
    plan = read(plan_ref)
    need(plan == build_plan(inputs=plan['inputs'], source_base_commit=plan['source_base_commit'])
         and source_clean(plan), 'PLAN_SOURCE')
    policy_ref = ref(POLICY)
    policy = read(policy_ref)
    gate = policy.get(STAGE, {})
    need(all(policy.get(key) is False for key in data.CLOSED_FLAGS)
         and gate.get('plan') == plan_ref and gate.get('output_root') == str(ROOT)
         and gate.get('relaunch_allowed') is False and gate.get('max_invocations') == 1
         and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
         and gate.get('training_authorized') is False and gate.get('test_authorized') is False
         and gate.get('broker_access_allowed') is False
         and gate.get('model_forwards') == 1 and gate.get('optimizer_steps') == 0, 'CLOSED_POLICY')
    expected = ('ADMITTED_ONE_USE', True, 1) if admitted else ('PREPARED_NOT_ADMITTED', False, 0)
    need((gate.get('status'), gate.get('execution_authorized'),
          gate.get('remaining_invocations')) == expected, 'ONE_USE_GATE')
    checked_inputs(plan)
    return plan, plan_ref, policy_ref


def publish(name, value):
    raw = value if isinstance(value, bytes) else (
        json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    _publish_bytes(ROOT / name, raw)
    return {'path': str(ROOT / name), 'sha256': hashlib.sha256(raw).hexdigest()}


def load_parameters(plan):
    metadata, constants = parameter_artifact_metadata(plan['inputs'], plan['geometry'])
    need(metadata == plan['parameter_reuse'], 'PARAMETER_REUSE_BOOKEND')
    proof = publish('PARAMETER_REUSE.json', metadata)
    return {'squeeze_manifest': metadata['squeeze_manifest'],
        'registry_artifact': metadata['registry_artifact'], 'registry_constants': constants,
        'reuse_proof': proof, 'parameter_fits': 0}


def read_feature_source(plan):
    from gx1.contracts.market_state_native_m5_feature_inputs_v1 import RAW_COLUMNS
    spec = plan['feature_source_spec']
    pieces = []
    for part in spec['parts']:
        before = stamp(part['path'])
        need(before == plan['feature_source_metadata'][part['path']]['stat'], 'FEATURE_SOURCE_STAT')
        parquet = pq.ParquetFile(part['path'])
        for group in part['row_groups']:
            table = parquet.read_row_group(group, columns=['time', *RAW_COLUMNS], use_threads=False)
            need(all(table[name].null_count == 0 for name in table.column_names), 'RAW_NULL')
            frame = table.to_pandas()
            starts = pd.to_datetime(frame.pop('time'), utc=True, errors='raise')
            frame.index = pd.DatetimeIndex(starts).as_unit('ns')
            pieces.append(frame.loc[frame.index.asi8 < spec['cutoff_ns'] - data.M5_NS,
                                    list(RAW_COLUMNS)])
        need(stamp(part['path']) == before, 'FEATURE_SOURCE_CHANGED')
    need(len(pieces) > 0, 'FEATURE_SOURCE_EMPTY')
    frame = pd.concat(pieces)
    need(frame.index.is_monotonic_increasing and not frame.index.has_duplicates, 'FEATURE_SOURCE_ORDER')
    return frame


def fold_truth(truth, upper):
    value = {key: array.copy() for key, array in truth.items()}
    mature = value['target_present'] & (value['validity_ns'] < upper)
    value['mature'] = mature
    value['observed_previous'] &= mature
    value['eligible'] = mature & value['observed_previous']
    value['target_row'][~mature] = -1
    for key in ('open_sign', 'close_sign'):
        value[key][~mature] = 0
    return value


def target_axis(plan, provider):
    with np.load(plan['inputs']['target_tape']['path'], allow_pickle=False) as archive:
        source = archive['source_row']
        issues = archive['issue_ns']
        wanted = np.flatnonzero((provider.issue_ns >= FIT_START_NS) & (provider.issue_ns < FIT_END_NS))
        need(len(wanted) > 0, 'EMPTY_TRAIN_WINDOW')
        at = np.searchsorted(source, provider.source_row[wanted])
        need(np.all(at < len(source)) and np.array_equal(source[at], provider.source_row[wanted])
             and np.array_equal(issues[at], provider.issue_ns[wanted]), 'TARGET_AXIS_JOIN')
        keys = ('issue_ns', 'target_row', 'validity_ns', 'target_present', 'observed_previous',
                'mature', 'eligible', 'open_known_at_issue', 'open_sign', 'close_sign')
        truth = {key: archive['future_' + key][at] for key in keys}
    truth = fold_truth(truth, FIT_END_NS)
    supervised = truth['eligible'].any(axis=1)
    ready = provider.input_ready[wanted]
    chosen = ready & supervised
    positions = wanted[chosen]
    rows = provider.source_row[positions]
    fit_issues = provider.issue_ns[positions]
    need(len(rows) >= BATCH and np.all(np.diff(rows) > 0), 'INSUFFICIENT_GENUINE_FIT')
    report = {
        'window_issue_rows': len(wanted), 'input_not_ready_rows': int((~ready).sum()),
        'no_mature_supervision_rows': int((~supervised).sum()), 'fit_issue_rows': len(rows),
        'fit_first_issue_ns': int(fit_issues[0]),
        'fit_last_issue_ns': int(fit_issues[-1]),
        'fit_source_rows_sha256': hashlib.sha256(np.ascontiguousarray(rows, dtype='<i8').tobytes()).hexdigest(),
        'fit_issue_ns_sha256': hashlib.sha256(np.ascontiguousarray(fit_issues, dtype='<i8').tobytes()).hexdigest(),
        'per_tf_mature': truth['mature'][chosen].sum(axis=0).tolist(),
        'per_tf_eligible': truth['eligible'][chosen].sum(axis=0).tolist(),
    }
    return rows, {key: value[chosen] for key, value in truth.items()}, report


def normalized_batch_report(model, tensors):
    report = {}
    for key, tensor in tensors.items():
        if key == 'ctx_cat':
            normalized = tensor
            role = 'categorical_domain_verified_by_native_forward'
        else:
            surface = 'signal' if key in ('seq_x', 'snap_x') else (
                'ctx_cont' if key == 'ctx_cont' else 'mtf_' + key.removeprefix('seq_'))
            normalized = model.backbone._normalize_input_surface(tensor, surface=surface)
            role = 'same_native_backbone_input_normalization'
        raw = tensor.detach().cpu().numpy()
        values = normalized.detach().cpu().numpy()
        need(np.isfinite(raw).all() and np.isfinite(values).all(), 'NORMALIZED_NONFINITE_' + key)
        report[key] = {'shape': list(raw.shape), 'dtype': str(raw.dtype),
            'finite_values': int(values.size), 'role': role,
            'raw_sha256': hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
            'normalized_sha256': hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()}
    need(len(report) == 9, 'NINE_NATIVE_SURFACES')
    return report


def publish_feature_snapshot(provider):
    """Persist the bounded physical inputs, with exact array reload verification."""
    surfaces = provider.surfaces
    top_arrays = ('bar_start_ns', 'retained_source_rows', 'retained_issue_rows',
                  'signal', 'ctx_cont', 'ctx_cat', 'observed')
    arrays = {name: surfaces[name] for name in top_arrays}
    metadata = {key: value for key, value in surfaces.items()
                if key not in (*top_arrays, 'multi_tf_sources')}
    timeframes = {}
    for tf, frame in surfaces['multi_tf_sources'].items():
        descriptors = {}
        scalars = {}
        for name, value in frame.attrs.items():
            if isinstance(value, np.ndarray):
                key = 'tf_' + tf + '_' + name
                arrays[key] = value
                descriptors[name] = key
            else:
                scalars[name] = value
        need(set(descriptors) == {'ts_int64', 'feats_np', 'owner_row_indices',
             'model_native_mtf_scalars_np_v4'}, 'SNAPSHOT_TF_ARRAYS')
        need(np.array_equal(frame.index.asi8, frame.attrs['ts_int64'])
             and np.shares_memory(frame.to_numpy(copy=False), frame.attrs['feats_np']),
             'SNAPSHOT_TF_FRAME_IDENTITY')
        timeframes[tf] = {'columns': list(frame.columns), 'array_attributes': descriptors,
                          'scalar_attributes': scalars}
    need(set(timeframes) == {'M5', 'M15', 'H1', 'H4', 'D1'}, 'SNAPSHOT_CLOCKS')
    descriptors = {}
    for name, value in arrays.items():
        need(isinstance(value, np.ndarray) and value.dtype.kind in 'biuf',
             'SNAPSHOT_ARRAY_TYPE')
        descriptors[name] = {'dtype': value.dtype.str, 'shape': list(value.shape),
            'sha256': hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    archive_ref = publish('FEATURE_INPUTS.npz', stream.getvalue())
    stream.close()
    with np.load(archive_ref['path'], allow_pickle=False) as saved:
        need(set(saved.files) == set(arrays), 'SNAPSHOT_ARRAY_INVENTORY')
        for name, original in arrays.items():
            loaded = saved[name]
            need(loaded.dtype == original.dtype and loaded.shape == original.shape
                 and np.array_equal(loaded, original, equal_nan=True)
                 and hashlib.sha256(np.ascontiguousarray(loaded).tobytes()).hexdigest()
                     == descriptors[name]['sha256'], 'SNAPSHOT_ARRAY_ROUNDTRIP')
    return publish('FEATURE_INPUTS.json', {
        'schema_version': 'gx1_native_m5_bounded_feature_snapshot_v1',
        'archive': archive_ref, 'surfaces_metadata': metadata,
        'timeframes': timeframes, 'arrays': descriptors,
        'reload_all_arrays_dtype_shape_equal_nan_and_bytes_sha256': True,
        'scope': 'full_native_clock_axes_and_bounded_physical_values_no_recomputed_state',
        'raw_history_prices_stored': False, 'model_forwards': 0, 'optimizer_steps': 0})


def load_candidate_provider(plan):
    from gx1.contracts import market_state_native_m5_feature_inputs_v1 as features
    from gx1.features import htf_features as htf
    saved = read(plan['inputs']['candidate_snapshot'])
    need(candidate_metadata(plan['inputs'], plan['feature_contract'])
         == plan['candidate_requalification'], 'CANDIDATE_METADATA_BOOKEND')
    need(ref(plan['inputs']['candidate_archive']['path']) == plan['inputs']['candidate_archive'],
         'CANDIDATE_ARCHIVE_IDENTITY')
    arrays = {}
    expected = {'bar_start_ns', 'retained_source_rows', 'retained_issue_rows',
                'signal', 'ctx_cont', 'ctx_cat', 'observed'}
    attrs = {'ts_int64', 'feats_np', 'owner_row_indices', 'model_native_mtf_scalars_np_v4'}
    for tf in saved['timeframes']:
        expected.update('tf_' + tf + '_' + name for name in attrs)
    with np.load(plan['inputs']['candidate_archive']['path'], allow_pickle=False) as archive:
        need(set(archive.files) == set(saved['arrays']) == expected and len(expected) == 27,
             'CANDIDATE_ARRAY_INVENTORY')
        for name in sorted(expected):
            value = archive[name]
            header = saved['arrays'][name]
            dtype = np.dtype(np.bool_ if name == 'observed' else np.float32 if
                name in ('signal', 'ctx_cont') or name.endswith(('_feats_np', '_scalars_np_v4'))
                else np.int64)
            need(value.dtype == dtype and value.dtype.str == header['dtype']
                 and list(value.shape) == header['shape']
                 and hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == header['sha256'],
                 'CANDIDATE_ARRAY_BYTES:' + name)
            value.flags.writeable = False
            arrays[name] = value
    need(ref(plan['inputs']['candidate_archive']['path']) == plan['inputs']['candidate_archive'],
         'CANDIDATE_ARCHIVE_BOOKEND')
    surfaces = dict(saved['surfaces_metadata'])
    need(surfaces['source_binding']['source_spec'] == plan['feature_source_spec']
         and surfaces['source_binding']['native_m5_manifest'] == plan['feature_source_spec']['manifest']
         and surfaces['source_binding']['raw_columns'] == list(features.RAW_COLUMNS)
         and surfaces['registry_contract_sha256'] == plan['parameter_reuse']['registry_contract_sha256']
         and surfaces['squeeze_binding'] == plan['parameter_reuse']['squeeze_binding']
         and surfaces['parameter_available_ns'] == PARAMETER_END_NS,
         'CANDIDATE_PHYSICAL_SOURCE_BINDINGS')
    macro_binding = surfaces['macro_binding']
    need(macro_binding['contract'] == plan['feature_contract']['macro_contract']
         and macro_binding['binding_sha256'] == features._sha(
             {key: value for key, value in macro_binding.items() if key != 'binding_sha256'}),
         'CANDIDATE_MACRO_BINDING')
    surfaces.update({name: arrays[name] for name in expected if not name.startswith('tf_')})
    multi_tf = {}
    for tf, descriptor in saved['timeframes'].items():
        need(descriptor['array_attributes'] == {name: 'tf_' + tf + '_' + name for name in attrs}
             and descriptor['columns'] == plan['feature_contract']['field_names']['mtf'],
             'CANDIDATE_TF_ORDER:' + tf)
        scalar = descriptor['scalar_attributes']
        need(scalar['htf_feature_contract'] == htf.HTF_V4_MATRIX_CONTRACT
             and scalar['model_native_mtf_scalar_contract_v4'] == htf.MODEL_NATIVE_MTF_SCALAR_CONTRACT_V4
             and scalar['model_native_mtf_scalar_fields_v4']
                 == list(htf.MODEL_NATIVE_MTF_SCALAR_FIELDS_BY_TIMEFRAME_V4[tf]),
             'CANDIDATE_SCALAR_CONTRACT:' + tf)
        values, times = arrays['tf_' + tf + '_feats_np'], arrays['tf_' + tf + '_ts_int64']
        scalar_values = arrays['tf_' + tf + '_model_native_mtf_scalars_np_v4']
        need(times.ndim == 1 and values.shape == (len(times), len(descriptor['columns']))
             and scalar_values.shape == (len(times), len(scalar['model_native_mtf_scalar_fields_v4'])),
             'CANDIDATE_TF_ARRAY_SHAPES:' + tf)
        frame = pd.DataFrame(values, index=pd.to_datetime(times, utc=True),
                             columns=descriptor['columns'], copy=False)
        frame.attrs = dict(scalar)
        frame.attrs.update({name: arrays[key] for name, key in descriptor['array_attributes'].items()})
        multi_tf[tf] = frame
    surfaces['multi_tf_sources'] = multi_tf
    # The old receipt remains unchanged. Only this new in-memory representation
    # uses the clean catalogue identity, after contract and byte equivalence.
    surfaces['provider_contract'] = plan['feature_contract']
    need(len(surfaces['bar_start_ns']) == surfaces['source_binding']['rows'],
         'CANDIDATE_NATIVE_SOURCE_ROWS')
    with np.load(plan['inputs']['target_tape']['path'], allow_pickle=False) as truth:
        domain_proof = validate_candidate_clock_domains(surfaces['bar_start_ns'],
            {tf: {key: frame.attrs[key] for key in ('ts_int64', 'owner_row_indices')}
             for tf, frame in multi_tf.items()}, truth)
    expected_proof = plan['candidate_requalification'].get('cached_clock_admission')
    need(expected_proof is not None and domain_proof == {key: value for key, value in expected_proof.items()
         if key not in ('cached_clock_arrays_decoded', 'cached_feature_value_arrays_decoded',
                        'raw_price_values_decoded')}, 'CANDIDATE_CLOCK_ADMISSION_PARITY')
    provider = features.NativeM5FeatureProvider(surfaces)
    aliases = dict(plan['feature_contract']['local_value_owners']['same_M5_exact_aliases'])
    from gx1.features import entry_model_native_feature_layers_v1 as layers
    trend = dict(zip(layers.TRENDLINE_REGISTRY_M5_LAYER_FEATURE_NAMES,
                     layers.TRENDLINE_REGISTRY_FEATURE_NAMES_V1))
    macro = set(provider.profile['macro_context_fields'])
    lineage = {}
    for name in provider.profile['physical_field_names']['signal']:
        if name in aliases:
            lineage[name] = 'same_native_M5_owner:' + aliases[name]
        elif name in trend:
            lineage[name] = 'local_native_M5_trendline:' + trend[name]
        else:
            need(name.startswith('ctx_cont.'), 'CANDIDATE_UNOWNED_FIELD:' + name)
            lineage[name] = ('asof_macro:' if name[9:] in macro
                             else 'closed_context_or_session:') + name[9:]
    need(surfaces['field_lineage'] == lineage, 'CANDIDATE_FIELD_LINEAGE')
    proof = {**plan['candidate_requalification'],
        'candidate_snapshot': plan['inputs']['candidate_snapshot'],
        'candidate_archive': plan['inputs']['candidate_archive'],
        'array_count': len(arrays), 'all_array_bytes_dtype_shape_exact': True,
        'native_M5_axis_and_separate_TF_owner_domains_exactly_joined_to_verified_targets': True,
        'ordered_fields_aliases_and_formula_lineage_equal': True,
        'source_axes_sha256_unchanged': surfaces['axes']['contract_sha256'],
        'raw_numeric_values_recomputed': False, 'old_statistics_or_weights_loaded': False}
    return provider, proof


def qualify_physical_pipeline(plan, parameters):
    import torch
    from gx1.features import d1_macro_context_v1 as macro
    from gx1.features.volatility_squeeze_state_v1 import (
        load_volatility_squeeze_artifact_manifest, VOLATILITY_SQUEEZE_DIRECTION_CLOCKS,
    )
    from gx1.contracts import market_state_native_m5_feature_inputs_v1 as features
    from gx1.contracts.market_state_ta_integrated_model_v1 import build_native_bar_direction_model
    from gx1.contracts.market_state_ta_integrated_objective_v1 import (
        native_bar_direction_loss, native_bar_direction_score_sums,
    )
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256

    provider, admission = load_candidate_provider(plan)
    feature_snapshot_ref = plan['inputs']['candidate_snapshot']
    rows, truth, population = target_axis(plan, provider)
    window_rows = np.flatnonzero((provider.issue_ns >= FIT_START_NS) & (provider.issue_ns < FIT_END_NS))
    readiness = provider.readiness_report(provider.source_row[window_rows])
    support = provider.support_report(rows)
    need(all(all(count == surface['physical_rows'] for count in surface['observed_counts'])
             for surface in support['surfaces'].values())
         and len(support['families']) == 8 and len(support['macro_fields']) == 61,
         'GENUINE_CONSUMED_SUPPORT')
    original_support = read(plan['inputs']['candidate_support'])
    need(population == original_support['fit_population']
         and readiness == original_support['input_readiness']
         and support == original_support['consumed_support'], 'CANDIDATE_CONSUMPTION_PARITY')
    admission_ref = publish('FEATURE_INPUTS_REQUALIFICATION.json', {
        **admission, 'exact_fit_population_readiness_and_consumed_support_equal': True,
        'clean_process_io_firewall': firewall_state()})
    support_ref = publish('FEATURE_SUPPORT.json', {
        'fit_population': population, 'input_readiness': readiness,
        'consumed_support': support,
        'feature_source_spec': plan['feature_source_spec'],
        'parameter_reuse_proof': parameters['reuse_proof'],
        'scope': 'bounded_exact_consumption_arrays_after_full_native_state_prefix',
        'model_forwards': 0, 'optimizer_steps': 0})
    normalization = features.fit_native_m5_feature_normalization(
        provider, fit_source_rows=rows, cutoff_ns=FIT_END_NS)
    normal_ref = publish('NORMALIZATION.json', normalization)
    input_binding = provider.input_binding(normalization)
    batch_rows = rows[:BATCH]
    raw = provider.gather(batch_rows, max_rows=BATCH)
    batch_truth = {key: value[:BATCH] for key, value in truth.items()}
    torch.set_num_threads(plan['execution']['numerical_threads'])
    torch.manual_seed(plan['seed'])
    model = build_native_bar_direction_model(input_normalization=normalization,
        input_binding=input_binding, architecture=plan['architecture'], device=torch.device('cpu')).eval()
    initial_sha = canonical_model_state_sha256(model.state_dict())
    tensors = {key: torch.from_numpy(value) for key, value in raw.items()}
    with torch.inference_mode():
        normalized = normalized_batch_report(model, tensors)
        output = model(tensors)
        loss = native_bar_direction_loss(model_output=output, truth=batch_truth,
            fold_upper_ns=FIT_END_NS, target_contract_sha256=plan['target_contract']['contract_sha256'])
        score = native_bar_direction_score_sums(model_output=output, truth=batch_truth,
            fold_upper_ns=FIT_END_NS, target_contract_sha256=plan['target_contract']['contract_sha256'])
    need(not loss['skip_update'] and bool(torch.isfinite(loss['loss']))
         and all(bool(torch.isfinite(value).all()) for value in output.values())
         and canonical_model_state_sha256(model.state_dict()) == initial_sha, 'FRESH_FORWARD_IDENTITY')
    checkpoint_binding = {
        'input': input_binding, 'model': model.research_objective,
        'target': plan['target_contract'], 'normalization': normal_ref,
        'architecture': plan['architecture'], 'seed': plan['seed'],
        'model_state_sha256': initial_sha}
    stream = io.BytesIO()
    torch.save({'model_state': model.state_dict(), 'bindings': checkpoint_binding}, stream)
    payload = stream.getvalue()
    loaded = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    need(loaded['bindings'] == checkpoint_binding
         and canonical_model_state_sha256(loaded['model_state']) == initial_sha, 'CHECKPOINT_ROUNDTRIP')
    model.load_state_dict(loaded['model_state'], strict=True)
    need(canonical_model_state_sha256(model.state_dict()) == initial_sha, 'CHECKPOINT_STRICT_LOAD')
    checkpoint_ref = publish('INITIAL_MODEL.pt', payload)
    capture = {'source_row': batch_rows, **{'truth_' + key: value for key, value in batch_truth.items()},
               **{key: value.detach().cpu().numpy() for key, value in output.items()}}
    stream = io.BytesIO()
    np.savez_compressed(stream, **capture)
    capture_ref = publish('INITIAL_CAPTURE.npz', stream.getvalue())
    fields = provider.profile['physical_field_names']['signal']
    missing = {field: int((~np.isfinite(provider.surfaces['signal'][:, j])).sum())
               for j, field in enumerate(fields) if not np.isfinite(provider.surfaces['signal'][:, j]).all()}
    return {
        'status': 'TECHNICAL_INITIAL_BATCH_ONLY', 'model_constructions': 1,
        'model_forwards': 1, 'optimizer_steps': 0, 'checkpoint': checkpoint_ref,
        'capture': capture_ref, 'normalization': normal_ref, 'input_binding': input_binding,
        'model_contract': model.research_objective, 'initial_state_sha256': initial_sha,
        'strict_load_state_sha256': initial_sha, 'forward_surfaces': normalized,
        'fit_population': population, 'normalization_population': normalization['fit_population'],
        'input_readiness': readiness, 'consumed_support': support,
        'feature_support_artifact': support_ref, 'feature_inputs_artifact': feature_snapshot_ref,
        'feature_inputs_requalification': admission_ref,
        'routed_families': support['families'],
        'field_lineage': provider.surfaces['field_lineage'], 'macro_field_count': len(macro.D1_MACRO_CONTEXT_FIELDS),
        'physical_missing_value_counts': missing, 'input_ready_rows': int(provider.input_ready.sum()),
        'batch_loss': float(loss['loss']), 'score_sums': score,
        'full_TRAIN_baseline_qualified': False, 'predictive_quality_measured': False,
        'economic_edge_proven': False,
    }


def run(plan_sha):
    firewall_state()
    started = time.monotonic()
    cap = require_capped_cpu_research_producer_execution()
    lock = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 12 * 1024**3
         and cap['swap'] == 512 * 1024**2 and cap['numerical_threads'] == 1
         and lock['scope'] == cap['cgroup_path'][len('/sys/fs/cgroup'):]
         and signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'CAP_LIMITS')
    plan, plan_ref, policy_ref = inspect(plan_sha, admitted=True)
    ROOT.mkdir(parents=True, exist_ok=False)
    claim = publish('CLAIM.json', {'status': 'CLAIMED_ONE_USE', 'stage': STAGE,
        'plan': plan_ref, 'policy_at_claim': policy_ref, 'remaining_invocations': 0,
        'max_invocations': 1, 'test_data_used': False})
    result_ref = None
    old_handler = signal.getsignal(signal.SIGALRM)
    def expired(*args):
        raise TimeoutError('NATIVE_M5_PIPELINE_INTERNAL_LIMIT')
    try:
        remaining = plan['execution']['internal_seconds'] - (time.monotonic() - started)
        need(remaining > 0, 'TIME_BEFORE_PHYSICAL_WORK')
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        original_history = reused_history_verification(plan['inputs'])
        history = original_history['verification']
        history_ref = plan['inputs']['history_verification']
        parameters = load_parameters(plan)
        report = qualify_physical_pipeline(plan, parameters)
        checked_inputs(plan)
        need(source_clean(plan) and ref(POLICY) == policy_ref and ref(PLAN) == plan_ref,
             'SOURCE_INPUT_POLICY_BOOKEND')
        result_ref = publish('RESULT.json', {
            'status': 'COMPLETE_NATIVE_M5_PIPELINE_TECHNICAL_QUALIFICATION', 'stage': STAGE,
            'plan': plan_ref, 'claim': claim, 'policy_at_claim': policy_ref,
            'source_unchanged': True, 'parameters': {key: value for key, value in parameters.items()
                                                   if key != 'registry_constants'},
            'history_verification': history, 'history_verification_artifact': history_ref,
            'history_verification_role': 'reused_original_001_proof_with_original_plan_and_claim',
            'report': report, 'model_forwards': 1, 'optimizer_steps': 0,
            'normalization_fits': 1, 'clean_process_io_firewall': firewall_state(),
            'test_data_used': False, 'economic_edge_proven': False,
            'elapsed_seconds': time.monotonic() - started})
        terminal = publish('TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0,
            'stage': STAGE, 'plan': plan_ref, 'claim': claim, 'result': result_ref,
            'remaining_invocations': 0, 'source_unchanged': True,
            'relaunch_allowed': False, 'test_data_used': False})
        return {'result': result_ref, 'terminal': terminal}
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        publish('TERMINAL.json', {'status': 'FAILED_CONSUMED', 'exit_code': 1,
            'stage': STAGE, 'plan': plan_ref, 'claim': claim, 'result': result_ref,
            'error': type(error).__name__ + ':' + str(error)[:2000],
            'remaining_invocations': 0, 'relaunch_allowed': False, 'test_data_used': False})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    firewall_state()
    need(args.plan == str(PLAN) and args.out_dir == str(ROOT), 'CLI_SCOPE')
    if args.preflight:
        plan, plan_ref, policy_ref = inspect(args.plan_sha256, admitted=False)
        print(json.dumps({'status': 'PREFLIGHT_PASS_GENUINE_METADATA_AND_CACHED_CLOCKS', 'plan': plan_ref,
            'policy': policy_ref, 'claim_created': False, 'output_root_absent': True,
            'parquet_values_decoded': 0, 'cached_clock_arrays_decoded': 11,
            'cached_feature_value_arrays_decoded': 0, 'model_constructions': 0,
            'model_forwards': 0, 'clean_process_io_firewall': firewall_state(),
            'test_data_used': False}))
    else:
        print(json.dumps(run(args.plan_sha256), sort_keys=True))


if __name__ == '__main__':
    main()
