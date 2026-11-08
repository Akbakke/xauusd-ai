"""One-use local-31 source/oracle diagnosis; no gather, target or model work."""
from __future__ import annotations

import ast
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from gx1.audit import verify_market_state_ta_native_input_path_v1 as inputs
from gx1.audit import verify_market_state_time_query_zero_step_v1 as io
from gx1.audit import verify_h4_trendline_lifecycle_oracle_v1 as files
from gx1.audit.verify_d1_trendline_lifecycle_oracle_v1 import NAMES, oracle_rows
from gx1.audit.verify_m5_m15_h1_trendline_oracle_v1 import _compare

REPO = Path('/home/andre2/src/GX1_CURRENT')
STAGE = 'market_state_local_trendline_formula_diagnostic_20261108_001'
SCHEMA = 'gx1_market_state_local_trendline_formula_diagnostic_v1'
PLAN_PATH = REPO / 'configs/research' / (STAGE.upper() + '.json')
OUT_ROOT = Path('/home/andre2/GX1_RUNS') / STAGE.upper()
POLICY_PATH = REPO / 'NEXT_RUN_POLICY.json'
OLD_ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_NATIVE_INPUT_20261108_003')
DATA_ROOT = Path('/home/andre2/GX1_DATA/data/data/prebuilt/XAU_GC_NATIVE_26M_20261018_003')
OLD_HEAD = '7128684281792657431df09e8c844c8ec935bec9'
PHASE_REF = {'path': str(OLD_ROOT / 'INPUT_TENSOR_PHASE.json'),
             'sha256': '57d17dc576d3e22ee7c5ba1e4a25c1e7f07f3cb57bed2bc60365d9fe349ae79d'}
FIXED = {
    'input_phase': PHASE_REF,
    'enriched_manifest': {'path': str(DATA_ROOT / 'M5_ENRICHED.parquet.manifest.json'),
        'sha256': 'd1c0f94bcfa7c837143b287fc8a4355e3ea564cffcbae2d96c53c64e7bb62de6'},
    'enriched': {'path': str(DATA_ROOT / 'M5_ENRICHED.parquet'),
        'sha256': '5e5cb256fc60a5e9a7ffe1a161aa875660f01806b8e8de2bf601fc2f528e7f99'},
    'original_manifest': {'path': str(DATA_ROOT / 'M5_FEATURE_BASE_039.parquet.manifest.json'),
        'sha256': '992339d96956822e889ceccecd8b300ea4891c0660bd4876a372795405730d3c'},
    'source_manifest': {'path': str(DATA_ROOT / 'SURFACES_050/M5_FEATURE_BASE_050.parquet.manifest.json'),
        'sha256': '2faacdca663e3192e0fd17ef877ef2ca6527b6fc0b9e156e714251a13fc19283'},
    'original_receipt': {'path': '/home/andre2/GX1_RUNS/NATIVE_GC_DYNAMIC_DATASET_BUILD_20261018_039/04_m5_feature_base_RECEIPT.json',
        'sha256': 'e1805cfe2851baa20dfee83f86bf741177b0d6968bd778276ceafb74e6532a76'},
}
HISTORICAL_FILES = (
    'gx1/features/trendline_registry_v1.py',
    'gx1/features/entry_model_native_feature_layers_v1.py',
    'gx1/features/technical_indicators_v1.py',
    'gx1/scripts/materialize_entry_exit_m1_feature_base_v1.py',
)
WALL_SECONDS = 3600
LIMITS = {'memory_max': 12 * 1024**3, 'memory_high': 12 * 1024**3,
          'swap': 512 * 1024**2, 'pids': 64, 'numerical_threads': 1,
          'wall_seconds': WALL_SECONDS}
EXECUTION = {'class': 'producer', 'memory': '12G', 'swap': '512M',
    'runner': 'scripts/gx1_capped_run.sh',
    'module': 'gx1.audit.diagnose_market_state_local_trendline_formula_v1',
    'timeout_seconds': WALL_SECONDS, 'kill_after_seconds': 10,
    'device': 'cpu', 'cuda_producer': False}
SCOPE = {'local_rows': 47697, 'fields': 31, 'fit_parent_rows': 29574,
    'score_parent_rows': 17753, 'local_seq_len': 96,
    'history_end_exclusive_ns': 1735689600000000000,
    'local_source_row_intervals': [[987027, 1016809], [1016990, 1034905]],
    'local_times_ns_sha256': '2f662630b053eb426b39b81eeca31e629e131452c8509cf53f62a0186fd5fde9',
    'comparisons': ['native2009_oracle_vs_cache', 'actual2010_oracle_vs_cache',
                    'actual2010_producer_vs_cache', 'actual2010_oracle_vs_producer'],
    'conditional_native_truncated2010': 'only_if_native_vs_actual_2010_OHLC_or_clock_not_exact',
    'model_constructions': 0, 'model_forwards': 0, 'gather_calls': 0,
    'normalization_calls': 0, 'target_arrays_decoded': 0,
    'optimizer_steps': 0, 'test_data_used': False}
MAX_JSON_BYTES = 16 * 1024**2


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError('LOCAL_TRENDLINE_DIAGNOSTIC_' + reason)


def _ref(path):
    return {'path': str(path), 'sha256': io._sha(Path(path))}


def _historical_sources():
    report = {}
    for relative in HISTORICAL_FILES:
        original = subprocess.check_output(['git', 'show', OLD_HEAD + ':' + relative], cwd=REPO)
        current = (REPO / relative).read_bytes()
        need(original == current, 'ORIGINAL_PRODUCER_SOURCE:' + relative)
        report[relative] = hashlib.sha256(current).hexdigest()
    relative = 'gx1/features/htf_features.py'
    original = subprocess.check_output(['git', 'show', OLD_HEAD + ':' + relative], cwd=REPO, text=True)
    def atr_ast(source):
        matches = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == '_atr']
        need(len(matches) == 1, 'ATR_AST_OWNER')
        return ast.dump(matches[0], include_attributes=False)
    old_ast, current_ast = atr_ast(original), atr_ast((REPO / relative).read_text())
    need(old_ast == current_ast, 'ORIGINAL_ATR_AST')
    return {'original_source_head': OLD_HEAD, 'byte_identical_files': report,
            'htf_atr_ast_sha256': hashlib.sha256(current_ast.encode()).hexdigest()}


def _bound_local_parameters(registry_ref, qualified_ref, manifest, binding, enriched_ref):
    "Reuse a verified manifest only when it is the original producer artifact."
    need(registry_ref == qualified_ref, 'REGISTRY_MANIFEST_REF')
    constants = manifest['v29_registry_constants']
    need(binding['artifact_path'] == registry_ref['path']
         and binding['artifact_sha256'] == registry_ref['sha256']
         and binding['lane'] == 'M5'
         and binding['params_schema_version'] == constants['schema_version']
         and binding['params_module'] == constants['provenance']['module']
         and binding['params_contract_sha256'] == constants['contract_sha256']
         and manifest['m5_prebuilt_source'] == enriched_ref['path']
         and manifest['m5_prebuilt_source_sha256'] == enriched_ref['sha256'],
         'HISTORICAL_REGISTRY_BINDING')
    from gx1.features.entry_model_native_feature_layers_v1 import model_native_m5_trendline_parameters_from_registry
    parameters = model_native_m5_trendline_parameters_from_registry(constants)['local']
    need(parameters['seq_len'] == 96, 'LOCAL_PARAMETERS')
    return parameters


def _small_inputs():
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as mature
    phase = io._read_ref(PHASE_REF)
    old_plan = io._read_ref(phase['plan'])
    enriched = io._read_ref(FIXED['enriched_manifest'])
    original = io._read_ref(FIXED['original_manifest'])
    source = io._read_ref(FIXED['source_manifest'])
    receipt = io._read_ref(FIXED['original_receipt'])
    terminal_ref = _ref(OLD_ROOT / 'TERMINAL.json')
    terminal = io._read_ref(terminal_ref)
    failure = io._read_ref(terminal['failure'])
    need(phase['status'] == 'COMPLETE_INPUT_TENSOR_PHASE_ONLY'
         and phase['full_input_audit_pass'] is False
         and terminal['status'] == 'FAILED_CONSUMED'
         and terminal['completed_input_tensor_phase'] == PHASE_REF
         and terminal['plan'] == phase['plan']
         and failure['error'] == 'TA_NATIVE_INPUT_PATH_LOCAL_TRENDLINE_FORMULA_MISMATCH', 'PHASE_003_CHAIN')
    need(phase['gather']['local_source_row_intervals'] == SCOPE['local_source_row_intervals']
         and phase['local_context_formula_evidence']['times_ns_sha256'] == SCOPE['local_times_ns_sha256']
         and phase['normalization']['signal']['unique_rows'] == SCOPE['local_rows']
         and phase['gather']['fit_rows'] == SCOPE['fit_parent_rows']
         and phase['gather']['score_rows'] == SCOPE['score_parent_rows'], 'PHASE_003_SCOPE')
    need(receipt['exit_code'] == 0 and receipt['source_head'] == OLD_HEAD
         and original['source_parquet'] == source['source_parquet'] == FIXED['enriched']['path']
         and original['source_sha256'] == source['source_sha256'] == FIXED['enriched']['sha256']
         and original['source_manifest_sha256'] == source['source_manifest_sha256'] == FIXED['enriched_manifest']['sha256']
         and original['output_parquet_sha256'] == source['output_parquet_sha256']
         and original['registry_fit_binding'] == source['registry_fit_binding']
         and enriched['output_parquet'] == FIXED['enriched']['path']
         and enriched['output_parquet_sha256'] == FIXED['enriched']['sha256'], 'ORIGINAL_039_050_LINEAGE')
    original_stat, source_stat = Path(original['output_parquet']).stat(), Path(source['output_parquet']).stat()
    need((original_stat.st_dev, original_stat.st_ino, original_stat.st_size) ==
         (source_stat.st_dev, source_stat.st_ino, source_stat.st_size), '039_050_HARDLINK_IDENTITY')
    refs = dict(FIXED)
    refs.update({'input_plan': phase['plan'], 'input_claim': phase['claim'],
        'input_terminal': terminal_ref, 'input_failure': terminal['failure'],
        'candidate_view': phase['candidate_view'],
        'provider_plan': mature.INPUT_REFS['provider_plan'],
        'provider_worker': mature.INPUT_REFS['provider_worker'],
        'registry_manifest': {'path': source['registry_fit_binding']['artifact_path'],
                              'sha256': source['registry_fit_binding']['artifact_sha256']}})
    native_inputs, registry_manifest, _ = inputs.inspect_formula_inputs(old_plan['formula_inputs'])
    native_refs = native_inputs['native_parts']
    provenance = enriched['native_m5_source']
    need(all(ref['path'] in provenance['part_paths']
         and provenance['part_sha256'][ref['path']] == ref['sha256']
         and int(Path(ref['path']).parent.name.removeprefix('year=')) <= 2024
         for ref in native_refs), 'ORIGINAL_NATIVE_PREFIX_REFS')
    refs['native_parts'] = native_refs
    refs['native_manifest'] = {'path': provenance['manifest_path'], 'sha256': provenance['manifest_sha256']}
    native_manifest = io._read_ref(refs['native_manifest'], limit=MAX_JSON_BYTES)
    need(native_manifest['time_min_utc'] == '2009-06-01T00:00:00+00:00', 'BOUND_NATIVE_FIRST_CLOCK')
    # inspect_formula_inputs already byte-verifies this 40 MB manifest through
    # its existing bounded owner. Its ref must also be the original039/050
    # producer binding; no fresh hash or relaxed JSON limit is substituted.
    parameters = _bound_local_parameters(refs['registry_manifest'], native_inputs['cache_manifest'],
        registry_manifest, source['registry_fit_binding'], refs['enriched'])
    return refs, phase, old_plan, source, parameters


def _metadata_interfaces(refs, source, parameters):
    "Check the real footer and callable signatures without decoding rows."
    import inspect as signatures
    import pyarrow.parquet as pq
    from gx1.scripts.audit_direction_d1_trendline_replay_v1 import load_native_parts_prefix, load_m5_prefix
    from gx1.features.trendline_registry_v1 import compute_trendline_registry_features_v1
    def footer(path, first, expected_rows=None, prices=False):
        with Path(path).open('rb') as stream:
            parquet = pq.ParquetFile(stream)
            schema = parquet.schema_arrow
            need(str(schema.field('time').type) == 'timestamp[ns, tz=UTC]', 'FOOTER_TIME_TYPE')
            column = parquet.schema.names.index('time')
            stats = parquet.metadata.row_group(0).column(column).statistics
            need(stats is not None and stats.has_min_max and stats.min.isoformat() == first, 'FOOTER_FIRST_CLOCK')
            need(expected_rows is None or parquet.metadata.num_rows == expected_rows, 'FOOTER_ROWS')
            if prices:
                need(all(str(schema.field(name).type) == 'double' for name in ('open', 'high', 'low', 'close'))
                     and str(schema.field('volume').type) == 'int64', 'FOOTER_OHLCV_TYPES')
            return {'rows': parquet.metadata.num_rows, 'row_groups': parquet.num_row_groups,
                    'first_time_min': stats.min.isoformat(), 'time_type': str(schema.field('time').type),
                    'price_columns_double': prices}
    footers = {
        'enriched': footer(refs['enriched']['path'], '2010-06-13T22:00:00+00:00', 1140982, True),
        'native_first_part': footer(refs['native_parts'][0]['path'], '2009-06-01T00:00:00+00:00', prices=True),
        'source050': footer(source['output_parquet'], '2010-06-14T16:15:00+00:00', source['rows']),
    }
    fields = source['feature_field_order']
    columns = [fields.index('chart.' + name) for name in NAMES]
    need(len(fields) == 2725 and len(set(columns)) == 31, 'SOURCE_FIELD_INDEX_INTERFACE')
    calls = ((load_m5_prefix, (Path(refs['enriched']['path']), 0), {}),
             (load_native_parts_prefix, (refs['native_parts'], 0), {}),
             (oracle_rows, ([], [], [], []), {**parameters, 'witness_positions': []}),
             (compute_trendline_registry_features_v1, (None,), {'timeframe': 'M5', **parameters}))
    names = []
    for function, args, kwargs in calls:
        signature = signatures.signature(function)
        signature.bind(*args, **kwargs)
        names.append({'owner': function.__module__ + '.' + function.__name__, 'signature': str(signature)})
    return {'footers': footers, 'signal_column_indices': columns,
            'call_signatures_bound_without_execution': names,
            'feature_rows_decoded': 0, 'target_arrays_decoded': 0, 'model_forwards': 0}


def build_plan():
    """Hash-bound metadata only; no feature/target payload, writes or claim."""
    refs, _, old_plan, source, parameters = _small_inputs()
    paths = set(old_plan['source_owners']) | set(HISTORICAL_FILES) | {str(Path(__file__).resolve().relative_to(REPO))}
    for path, digest in old_plan['source_owners'].items():
        need(io._sha(REPO / path) == digest, 'CORE_PHASE_SOURCE_CHANGED:' + path)
    return {'schema_version': SCHEMA + '_plan', 'stage': STAGE,
        'status': 'PREPARED_NOT_ADMITTED',
        'source_base_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'source_owners': {path: io._sha(REPO / path) for path in sorted(paths)},
        'inputs': refs, 'scope': dict(SCOPE), 'parameters': parameters,
        'historical_source_binding': _historical_sources(),
        'metadata_interfaces': _metadata_interfaces(refs, source, parameters),
        'limits': dict(LIMITS), 'execution': dict(EXECUTION),
        'output_root': str(OUT_ROOT), 'max_json_bytes': MAX_JSON_BYTES}


def _provider(ref):
    import importlib.util
    need(io._sha(Path(ref['path'])) == ref['sha256'], 'PROVIDER_SHA')
    spec = importlib.util.spec_from_file_location('gx1_local_trendline_provider264', ref['path'])
    need(spec is not None and spec.loader is not None, 'PROVIDER_IMPORT')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    need(io._sha(Path(ref['path'])) == ref['sha256'], 'PROVIDER_IMPORT_BOOKEND')
    return module


def _policy_gate(policy, plan_ref):
    gate = policy.get(STAGE, {})
    need(policy.get('training_enabled') is False
         and policy.get('native_training_enabled') is False
         and policy.get('native_launch_authorized') is False
         and policy.get('full_val_allowed') is False
         and gate.get('status') == 'ADMITTED' and gate.get('plan') == plan_ref
         and gate.get('worker') == _ref(Path(__file__).resolve())
         and gate.get('execution_authorized') is True and gate.get('data_reads_authorized') is True
         and gate.get('training_authorized') is False
         and gate.get('model_constructions_authorized') == 0 and gate.get('model_forwards_authorized') == 0
         and gate.get('target_arrays_decoded_authorized') == 0 and gate.get('optimizer_steps_authorized') == 0
         and gate.get('normalization_fits_authorized') == 0 and gate.get('test_data_used') is False
         and gate.get('relaunch_allowed') is False and gate.get('max_invocations') == 1
         and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
         and gate.get('output_root') == str(OUT_ROOT), 'EXACT_DIAGNOSTIC_POLICY')


def inspect(plan_ref, *, require_policy):
    head = io._clean_head()
    need(plan_ref['path'] == str(PLAN_PATH), 'PLAN_PATH')
    plan = io._read_ref(plan_ref, limit=MAX_JSON_BYTES)
    expected = build_plan()
    need(set(plan) == set(expected)
         and all(plan[key] == expected[key] for key in expected if key != 'source_base_head')
         and subprocess.run(['git', 'merge-base', '--is-ancestor', plan['source_base_head'], head],
             cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0,
         'PLAN_SCOPE_OR_BINDING')
    need(OUT_ROOT.is_dir() and OUT_ROOT.resolve(strict=True) == OUT_ROOT
         and not list(OUT_ROOT.iterdir()), 'UNCONSUMED_OUTPUT_ROOT')
    policy_ref = _ref(POLICY_PATH)
    if require_policy:
        _policy_gate(io._read_ref(policy_ref, limit=MAX_JSON_BYTES), plan_ref)
    provider = _provider(plan['inputs']['provider_worker'])
    provider_plan = io._read_ref(plan['inputs']['provider_plan'])
    info = provider.validate_metadata(provider_plan)
    need(provider_plan['m5_feature_manifest'] == plan['inputs']['source_manifest'], 'PROVIDER_SOURCE_MANIFEST')
    specs = provider.reuse_specs(info['reuse'])
    source_specs = [item for item in specs.values() if item['group'] == 'source050' and item['name'] == 'signal']
    need(len(source_specs) == 1 and source_specs[0]['shape'] == [provider.SOURCE_ROWS, 2725]
         and source_specs[0]['dtype'] == 'float32', 'PROVIDER_SIGNAL_INTERFACE')
    phase = io._read_ref(plan['inputs']['input_phase'])
    return plan, policy_ref, head, provider, provider_plan, info, phase


def preflight(plan_ref):
    plan, _, head, _, _, info, _ = inspect(plan_ref, require_policy=False)
    return {'status': 'READY_NO_CLAIM_NO_FEATURE_PAYLOAD_NO_MODEL', 'plan': plan_ref,
        'source_head': head, 'scope': plan['scope'], 'parameters': plan['parameters'],
        'historical_source_binding': plan['historical_source_binding'],
        'metadata_interfaces': plan['metadata_interfaces'],
        'readonly_decode_files_stat_checked': len(info['reuse']['scratch_files']),
        'target_arrays_decoded': 0, 'feature_payloads_read': False, 'model_forwards': 0}


def compare_all(oracle, observed, rows, times):
    """Preserve all fields and both consumed intervals before any decision."""
    need(oracle.shape == observed.shape == (len(rows), 31), 'COMPARE_SHAPE')
    reports = []
    start = 0
    for part in inputs.contiguous_parts(rows):
        stop = start + len(part)
        report = _compare(oracle[start:stop], observed[start:stop], part, times[start:stop], set())
        a, b = np.asarray(oracle[start:stop]), np.asarray(observed[start:stop])
        report['mask_mismatched_cells'] = int(np.count_nonzero(np.isnan(a) != np.isnan(b)))
        reports.append(report)
        start = stop
    per_field = []
    for j, name in enumerate(NAMES):
        fields = [report['per_field'][j] for report in reports]
        errors = [field['max_finite_absolute_error'] for field in fields if field['max_finite_absolute_error'] is not None]
        per_field.append({'field': name,
            'mismatched_rows': sum(field['mismatched_rows'] for field in fields),
            'oracle_finite_rows': sum(field['oracle_finite_rows'] for field in fields),
            'cache_finite_rows': sum(field['cache_finite_rows'] for field in fields),
            'max_finite_absolute_error': max(errors) if errors else None,
            'mask_mismatched_rows': int(np.count_nonzero(np.isnan(oracle[:, j]) != np.isnan(observed[:, j])))})
    total = sum(report['mismatched_cells'] for report in reports)
    examples = [example for report in reports for example in report['mismatch_examples']][:128]
    return {'rows': len(rows), 'fields': 31, 'compared_cells': len(rows) * 31,
        'mismatched_cells': total,
        'mask_mismatched_cells': sum(report['mask_mismatched_cells'] for report in reports),
        'lifecycle_mismatched_cells': sum(report['lifecycle_mismatched_cells'] for report in reports),
        'per_field': per_field, 'mismatch_examples': examples,
        'mismatch_examples_truncated': total > len(examples),
        'float_abs_tolerance': reports[0]['float_abs_tolerance'],
        'float_rel_tolerance': reports[0]['float_rel_tolerance'],
        'discrete_tolerance': 0, 'all_intervals_compared': len(reports)}


def price_parity(native, actual):
    """Measure the actual source, never assume it equals a truncated native tape."""
    clock_a, clock_b = native.index.asi8, actual.index.asi8
    keep = clock_a >= clock_b[0]
    a = native.loc[keep]
    clock_a = a.index.asi8
    common, ia, ib = np.intersect1d(clock_a, clock_b, assume_unique=True, return_indices=True)
    clocks_equal = np.array_equal(clock_a, clock_b)
    fields = {}
    for name in ('open', 'high', 'low', 'close'):
        x = a[name].to_numpy(dtype=np.float64)[ia]
        y = actual[name].to_numpy(dtype=np.float64)[ib]
        bad = np.flatnonzero(x.view(np.uint64) != y.view(np.uint64))
        fields[name] = {'mismatched_rows': len(bad),
            'max_absolute_error': float(np.max(np.abs(x-y))) if len(x) else None,
            'examples': [{'time_ns': int(common[i]), 'native': float(x[i]), 'actual': float(y[i])} for i in bad[:8]]}
    return {'native_rows_from_actual_start': len(a), 'actual_rows': len(actual),
        'common_clock_rows': len(common), 'clocks_exact': bool(clocks_equal),
        'native_only_clock_rows': len(clock_a) - len(common),
        'actual_only_clock_rows': len(clock_b) - len(common), 'OHLC': fields,
        'all_OHLC_bits_and_clocks_exact': bool(clocks_equal and all(x['mismatched_rows'] == 0 for x in fields.values()))}


def _positions(frame, times):
    clock = frame.index.asi8
    positions = np.searchsorted(clock, times).astype(np.int64)
    need(len(positions) > 0 and np.all(positions < len(clock))
         and np.array_equal(clock[positions], times) and np.all(np.diff(positions) > 0), 'PRICE_WITNESS_CLOCK')
    return positions


def _oracle(frame, times, parameters):
    from gx1.features.technical_indicators_v1 import wilder_atr
    positions = _positions(frame, times)
    atr = wilder_atr(frame['high'], frame['low'], frame['close'], 14).to_numpy(dtype=np.float64)
    positive = np.where(atr > 0., atr, np.nan)
    result, events = oracle_rows(frame['high'].to_numpy(dtype=np.float64),
        frame['low'].to_numpy(dtype=np.float64), frame['close'].to_numpy(dtype=np.float64), positive,
        seq_len=parameters['seq_len'], band_atr=parameters['band_atr'], witness_positions=positions)
    return result, atr, positions, events


def _source_clock_prefix(path, stop, expected_rows):
    import pandas as pd
    import pyarrow.parquet as pq
    with Path(path).open('rb') as stream:
        parquet = pq.ParquetFile(stream)
        need(parquet.metadata.num_rows == expected_rows, 'SOURCE_PARQUET_ROWS')
        groups, count = [], 0
        for i in range(parquet.num_row_groups):
            if count >= stop:
                break
            groups.append(i)
            count += parquet.metadata.row_group(i).num_rows
        clock = parquet.read_row_groups(groups, columns=['time'], use_threads=False).to_pandas(use_threads=False)['time']
    clock = pd.DatetimeIndex(clock.iloc[:stop]).as_unit('ns')
    need(len(clock) == stop and str(clock.tz) == 'UTC' and clock.is_unique and clock.is_monotonic_increasing,
         'SOURCE_CLOCK_PREFIX')
    return clock.asi8


def measure(plan, provider, provider_plan, info, phase, *, publish_measurement):
    from gx1.scripts.audit_direction_d1_trendline_replay_v1 import load_native_parts_prefix, load_m5_prefix
    from gx1.features.trendline_registry_v1 import compute_trendline_registry_features_v1
    source_rows = np.concatenate([np.arange(left, right, dtype=np.int64)
        for left, right in phase['gather']['local_source_row_intervals']])
    need(len(source_rows) == SCOPE['local_rows'], 'CONSUMED_ROWS')
    source = info['source']
    clock = _source_clock_prefix(source['output_parquet'], int(source_rows[-1]) + 1, provider.SOURCE_ROWS)
    times = np.ascontiguousarray(clock[source_rows], dtype=np.int64)
    need(inputs.axis_sha(times) == SCOPE['local_times_ns_sha256']
         and times[-1] + 300_000_000_000 <= SCOPE['history_end_exclusive_ns'], 'CONSUMED_CLOCK')
    specs = provider.reuse_specs(info['reuse'])
    maps = [(path, spec) for path, spec in specs.items() if spec['group'] == 'source050' and spec['name'] == 'signal']
    need(len(maps) == 1, 'SOURCE_SIGNAL_MAP')
    path, spec = maps[0]
    need(provider.regular(path) == info['reuse']['scratch_files'][path], 'SOURCE_SIGNAL_STAT')
    mapped = np.memmap(path, mode='r', dtype=spec['dtype'], shape=tuple(spec['shape']))
    names = source['feature_field_order']
    columns = [names.index('chart.' + name) for name in NAMES]
    need(len(set(columns)) == 31 and mapped.dtype == np.float32 and not mapped.flags.writeable, 'SIGNAL_COLUMNS')
    observed = np.ascontiguousarray(mapped[np.ix_(source_rows, columns)], dtype=np.float32)
    need(observed.shape == (SCOPE['local_rows'], 31) and np.isfinite(observed).all(), 'CONSUMED_VALUES')
    provider.flush_pages(mapped)
    del mapped, clock
    snapshots = {}
    for ref in [*plan['inputs']['native_parts'], plan['inputs']['enriched']]:
        checked, stat = files.checked(ref)
        snapshots[str(checked)] = stat
    cutoff = int(times[-1]) + 300_000_000_000
    native = load_native_parts_prefix(plan['inputs']['native_parts'], cutoff)
    actual, _ = load_m5_prefix(Path(plan['inputs']['enriched']['path']), cutoff)
    need(str(native.index.tz) == str(actual.index.tz) == 'UTC'
         and native.index[0].isoformat() == '2009-06-01T00:00:00+00:00'
         and actual.index[0].isoformat() == '2010-06-13T22:00:00+00:00'
         and all(actual[name].dtype == np.float64 for name in ('open', 'high', 'low', 'close'))
         and np.all(native.index.asi8 % 300_000_000_000 == 0), 'PRICE_PREFIX_SCHEMA_OR_ORIGIN')
    source_report = price_parity(native, actual)
    source_report.update({'source_row_ids_sha256': inputs.axis_sha(source_rows),
        'times_ns_sha256': inputs.axis_sha(times), 'cache_values_sha256': hashlib.sha256(observed.tobytes()).hexdigest(),
        'native_first_ns': int(native.index[0].value), 'actual_first_ns': int(actual.index[0].value),
        'native_prefix_rows': len(native), 'actual_prefix_rows': len(actual), 'cutoff_exclusive_ns': cutoff,
        'parameters': plan['parameters'], 'raw_HLC_dtype': 'float64',
        'ATR_owner': 'technical_indicators_v1.wilder_atr:14:recomputed_from_each_full_source_prefix',
        'stored_ATR_used': False, 'local_model_fields': ['chart.' + name for name in NAMES],
        'model_forwards': 0, 'target_arrays_decoded': 0})
    published = {}
    def report(name, value):
        for filename, stat in snapshots.items():
            need(files.regular(Path(filename)) == stat, 'PRICE_PAYLOAD_CHANGED')
        provider.validate_metadata(provider_plan)
        published[name] = publish_measurement(name + '.json', value)
    report('SOURCE_COMPARISON', source_report)
    measurements = {}
    actual_oracle = actual_atr = actual_positions = None
    for name, frame in (('native2009', native), ('actual2010', actual)):
        tic = time.monotonic()
        oracle, atr, positions, events = _oracle(frame, times, plan['parameters'])
        comparison = compare_all(oracle, observed, source_rows, times)
        comparison.update({'variant': name, 'native_prefix_rows': len(frame),
            'first_ns': int(frame.index[0].value), 'events': events,
            'elapsed_seconds': time.monotonic() - tic, 'fresh_state_at_prefix_row_zero': True,
            'ATR_at_witness_sha256': hashlib.sha256(np.ascontiguousarray(atr[positions]).tobytes()).hexdigest()})
        measurements[name] = comparison
        report(name.upper() + '_ORACLE', comparison)
        if name == 'actual2010':
            actual_oracle, actual_atr, actual_positions = oracle, atr, positions
        else:
            del oracle, atr, positions
        gc.collect()
    if not source_report['all_OHLC_bits_and_clocks_exact']:
        truncated = native.loc[native.index >= actual.index[0]]
        tic = time.monotonic()
        oracle, atr, positions, events = _oracle(truncated, times, plan['parameters'])
        comparison = compare_all(oracle, observed, source_rows, times)
        comparison.update({'variant': 'native_truncated2010', 'native_prefix_rows': len(truncated),
            'first_ns': int(truncated.index[0].value), 'events': events,
            'elapsed_seconds': time.monotonic() - tic,
            'predeclared_condition': 'native_vs_actual_2010_OHLC_or_clock_not_exact'})
        measurements['native_truncated2010'] = comparison
        report('NATIVE_TRUNCATED2010_ORACLE', comparison)
        del truncated, oracle, atr, positions
    del native
    producer_input = actual[['high', 'low', 'close']].copy()
    producer_input['atr'] = actual_atr
    tic = time.monotonic()
    produced, state = compute_trendline_registry_features_v1(producer_input,
        timeframe='M5', seq_len=plan['parameters']['seq_len'], band_atr=plan['parameters']['band_atr'])
    need(tuple(produced.columns) == NAMES and state.bar_count == len(actual), 'PRODUCER_INTERFACE')
    producer_values = np.ascontiguousarray(produced.iloc[actual_positions].to_numpy(), dtype=np.float32)
    producer_cache = compare_all(producer_values, observed, source_rows, times)
    producer_cache['float32_bit_mismatched_cells'] = int(np.count_nonzero(producer_values.view(np.uint32) != observed.view(np.uint32)))
    producer_oracle = compare_all(actual_oracle, producer_values, source_rows, times)
    measurements['actual2010_producer'] = {'producer_vs_cache': producer_cache,
        'oracle_vs_producer': producer_oracle, 'elapsed_seconds': time.monotonic() - tic,
        'producer_owner': 'gx1.features.trendline_registry_v1.compute_trendline_registry_features_v1',
        'identical_HLC_ATR_and_origin_for_producer_and_oracle': True}
    report('ACTUAL2010_PRODUCER', measurements['actual2010_producer'])
    actual_pass = (measurements['actual2010']['mismatched_cells'] == 0
        and producer_cache['float32_bit_mismatched_cells'] == 0
        and producer_oracle['mismatched_cells'] == 0)
    return {'decision': 'ACTUAL_SOURCE_MATCH' if actual_pass else 'MISMATCH',
        'measurements': measurements, 'source_comparison': source_report,
        'measurement_refs': published,
        'original_2009_reference_mismatched_cells': measurements['native2009']['mismatched_cells'],
        'input_tensor_phase_reused': plan['inputs']['input_phase'],
        'full_input_audit_pass': False, 'predictive_value_claimed': False,
        'limits_of_evidence': ['Only local31 is diagnosed; remaining MTF/macro qualification is not completed here.',
            'Matching the actual historical source does not establish predictive utility or stable line identity at model slots.',
            'Local mmap integrity reuses completed decode provenance and exact immutable stats, not a new body SHA.',
            'Different histories also change ATR initialization and absolute registry bar origins; a source match does not isolate each mechanism.']}


def run(plan_ref):
    import os
    import resource
    import signal
    from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_research_producer_execution
    cap = require_capped_cpu_research_producer_execution()
    need(all(cap[key] == LIMITS[key] for key in ('memory_max', 'memory_high', 'swap', 'pids', 'numerical_threads')), 'CAPS')
    began = time.monotonic()
    plan, policy_ref, head, provider, provider_plan, info, phase = inspect(plan_ref, require_policy=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    need(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'AMBIENT_TIMER')
    published, bytes_written = {}, 0
    def publish(name, value):
        nonlocal bytes_written
        size = len((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())
        need(name in ('FAILED.json', 'TERMINAL.json') or bytes_written + size <= MAX_JSON_BYTES, 'JSON_BUDGET')
        ref = io._write(OUT_ROOT, name, value, limit=MAX_JSON_BYTES)
        bytes_written += size
        published[name] = ref
        return ref
    def bookend():
        need(io._clean_head() == head and io._sha(PLAN_PATH) == plan_ref['sha256']
             and io._sha(POLICY_PATH) == policy_ref['sha256']
             and all(io._sha(REPO / path) == digest for path, digest in plan['source_owners'].items())
             and all(io._sha(Path(ref['path'])) == ref['sha256'] for name, ref in plan['inputs'].items()
                     if name not in ('native_parts', 'enriched')), 'SOURCE_BOOKEND')
    remaining = WALL_SECONDS - 30 - (time.monotonic() - began)
    need(remaining > 0., 'PRECLAIM_WALL_BUDGET')
    claim = publish('CLAIM.json', {'schema_version': SCHEMA + '_claim', 'plan': plan_ref,
        'policy': policy_ref, 'source_head': head, 'scope': plan['scope'], 'cap': cap,
        'input_tensor_phase': plan['inputs']['input_phase'], 'test_data_used': False})
    old_alarm = signal.getsignal(signal.SIGALRM)
    def expired(_signum, _frame):
        raise TimeoutError('LOCAL_TRENDLINE_DIAGNOSTIC_3600_SECONDS')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    def publish_measurement(name, measurement):
        bookend()
        return publish(name, {'schema_version': SCHEMA + '_measurement',
            'status': 'COMPLETE_DIAGNOSTIC_MEASUREMENT', 'plan': plan_ref,
            'claim': claim, 'source_head': head, 'measurement': measurement,
            'full_input_audit_pass': False, 'test_data_used': False})
    try:
        measurement = measure(plan, provider, provider_plan, info, phase, publish_measurement=publish_measurement)
        provider.validate_metadata(provider_plan)
        bookend()
        result = publish('RESULT.json', {'schema_version': SCHEMA + '_result',
            'status': 'COMPLETE_DIAGNOSTIC', 'plan': plan_ref, 'claim': claim,
            'policy': policy_ref, 'source_head': head, 'source_owners': plan['source_owners'],
            'inputs': plan['inputs'], 'historical_source_binding': plan['historical_source_binding'],
            'diagnostic': measurement, 'decision': measurement['decision'],
            'elapsed_seconds': time.monotonic() - began,
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'model_forwards': 0, 'target_arrays_decoded': 0, 'gather_calls': 0,
            'normalization_calls': 0, 'optimizer_steps': 0, 'full_input_audit_pass': False,
            'test_data_used': False})
        terminal = publish('TERMINAL.json', {'schema_version': SCHEMA + '_terminal',
            'status': 'COMPLETE_DIAGNOSTIC', 'exit_code': 0, 'plan': plan_ref,
            'claim': claim, 'result': result, 'decision': measurement['decision'],
            'no_relaunch_under_this_plan': True, 'test_data_used': False})
        return {'result': result, 'terminal': terminal, 'decision': measurement['decision']}
    except BaseException as exc:
        completed = {name: ref for name, ref in published.items() if name not in ('CLAIM.json', 'FAILED.json', 'TERMINAL.json')}
        failed = publish('FAILED.json', {'schema_version': SCHEMA + '_failure',
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'completed_measurements': completed, 'full_input_audit_pass': False,
            'elapsed_seconds': time.monotonic() - began, 'test_data_used': False})
        publish('TERMINAL.json', {'schema_version': SCHEMA + '_terminal',
            'status': 'FAILED_CONSUMED', 'exit_code': 1, 'plan': plan_ref, 'claim': claim,
            'failure': failed, 'completed_measurements': completed,
            'no_relaunch_under_this_plan': True, 'test_data_used': False})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, old_alarm)


def main():
    import argparse
    import sys
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    ref = {'path': args.plan, 'sha256': args.plan_sha256}
    print(json.dumps(preflight(ref) if args.preflight else run(ref), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
