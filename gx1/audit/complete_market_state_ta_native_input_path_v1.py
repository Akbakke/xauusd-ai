"""Complete TA input evidence from immutable tensor/local31 phases and MTF gaps.

No predictor, feature gather or normalizer is executed. The new computation is
raw census, missing MTF formula comparisons and exact mature-fit endpoint priors.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from gx1.audit import verify_market_state_ta_native_input_path_v1 as original
from gx1.audit import diagnose_market_state_local_trendline_formula_v1 as local31
from gx1.audit import verify_market_state_time_query_zero_step_v1 as io
from gx1.contracts import market_state_time_query_fold1_fit_v1 as fold

REPO = fold.CANONICAL_REPO
STAGE = 'market_state_ta_native_input_completion_20261108_001'
SCHEMA = 'gx1_market_state_ta_native_input_completion_v1'
PLAN_PATH = REPO / 'configs/research' / (STAGE.upper() + '.json')
OUT_ROOT = Path('/home/andre2/GX1_RUNS') / STAGE.upper()
POLICY_PATH = REPO / 'NEXT_RUN_POLICY.json'
SCOPE_FILE = 'gx1/contracts/market_state_time_query_fold1_fit_v1.py'
OLD_ASSIGNMENT = b"TA_INPUT_STAGE = 'market_state_ta_native_input_20261108_003'"
NEW_ASSIGNMENT = ("TA_INPUT_STAGE = '" + STAGE + "'").encode()
PHASE_REF = local31.PHASE_REF
LOCAL_ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_LOCAL_TRENDLINE_FORMULA_DIAGNOSTIC_20261108_001')
LOCAL_RESULT = {'path': str(LOCAL_ROOT / 'RESULT.json'),
                'sha256': '6efc0712d48b5ddb7929519c047185ad7fc6a031e9fadb126c555f10cfcd9e59'}
LOCAL_TERMINAL = {'path': str(LOCAL_ROOT / 'TERMINAL.json'),
                  'sha256': '07bff1dabde75a2fd4493092389a7ea497e8460b702bcbc1decf04fef11dbae1'}
LIMITS = dict(original.LIMITS)
EXECUTION = {**original.EXECUTION, 'module': 'gx1.audit.complete_market_state_ta_native_input_path_v1'}
MAX_JSON_BYTES = 16 * 1024**2
THIS_INVOCATION = {'model_constructions': 0, 'model_forwards': 0,
    'gather_calls': 0, 'normalization_calls': 0, 'normalization_fits': 0,
    'backward_calls': 0, 'optimizer_steps': 0, 'local_trendline_replays': 0,
    'score_fit_rows': 0,
    'provider_axes_target_labels_decoded': 0, 'test_data_used': False}


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError('TA_INPUT_COMPLETION_' + reason)


def _ref(path):
    return {'path': str(path), 'sha256': io._sha(Path(path))}


def require_stage_only_transition(old_bytes, current_bytes):
    """Exactly one known assignment changes; no other byte is excepted."""
    need(old_bytes.count(OLD_ASSIGNMENT) == 1
         and current_bytes.count(NEW_ASSIGNMENT) == 1
         and old_bytes.replace(OLD_ASSIGNMENT, NEW_ASSIGNMENT, 1) == current_bytes,
         'SCOPE_STAGE_ONLY_BYTE_PARITY')
    return {'old_sha256': hashlib.sha256(old_bytes).hexdigest(),
            'current_sha256': hashlib.sha256(current_bytes).hexdigest(),
            'old_assignment': OLD_ASSIGNMENT.decode(),
            'new_assignment': NEW_ASSIGNMENT.decode(),
            'all_other_bytes_identical': True}


def _source_reuse_binding(phase, diagnostic):
    old = dict(phase['source_owners'])
    for name, digest in diagnostic['source_owners'].items():
        need(name not in old or old[name] == digest, 'REUSED_SOURCE_DISAGREEMENT:' + name)
        old[name] = digest
    need(SCOPE_FILE in old, 'ORIGINAL_SCOPE_OWNER')
    archived = subprocess.check_output(['git', 'show', phase['source_head'] + ':' + SCOPE_FILE], cwd=REPO)
    need(hashlib.sha256(archived).hexdigest() == old[SCOPE_FILE], 'ORIGINAL_SCOPE_GIT_BLOB')
    transition = require_stage_only_transition(archived, (REPO / SCOPE_FILE).read_bytes())
    for name, digest in old.items():
        if name != SCOPE_FILE:
            need(io._sha(REPO / name) == digest, 'REUSED_SOURCE_CHANGED:' + name)
    paths = set(old) | {str(Path(__file__).resolve().relative_to(REPO))}
    current = {name: io._sha(REPO / name) for name in sorted(paths)}
    return current, {'tensor_phase_source_head': phase['source_head'],
        'local31_source_head': diagnostic['source_head'],
        'stage_only_scope_transition': transition,
        'all_other_reused_source_files_byte_identical': True,
        'unchanged_reused_source_files': len(old) - 1}


def _reuse_inputs():
    phase = io._read_ref(PHASE_REF, limit=MAX_JSON_BYTES)
    old_plan = io._read_ref(phase['plan'], limit=MAX_JSON_BYTES)
    diagnostic = io._read_ref(LOCAL_RESULT, limit=MAX_JSON_BYTES)
    terminal = io._read_ref(LOCAL_TERMINAL)
    diag = diagnostic['diagnostic']
    source = diag['source_comparison']
    actual = diag['measurements']['actual2010']
    producer = diag['measurements']['actual2010_producer']
    old_terminal_ref = diagnostic['inputs']['input_terminal']
    old_terminal = io._read_ref(old_terminal_ref)
    need(phase['status'] == 'COMPLETE_INPUT_TENSOR_PHASE_ONLY'
         and phase['full_input_audit_pass'] is False
         and phase['model_constructions'] == phase['model_forwards'] == 1
         and phase['backward_calls'] == phase['optimizer_steps'] == phase['normalization_fits'] == 0
         and phase['test_data_used'] is False
         and old_terminal['status'] == 'FAILED_CONSUMED'
         and old_terminal['completed_input_tensor_phase'] == PHASE_REF
         and old_terminal['plan'] == phase['plan'], 'REUSED_TENSOR_PHASE')
    need(diagnostic['status'] == terminal['status'] == 'COMPLETE_DIAGNOSTIC'
         and diagnostic['decision'] == terminal['decision'] == 'ACTUAL_SOURCE_MATCH'
         and terminal['exit_code'] == 0 and terminal['result'] == LOCAL_RESULT
         and diagnostic['plan'] == terminal['plan']
         and diag['input_tensor_phase_reused'] == PHASE_REF
         and diagnostic['test_data_used'] is False
         and actual['rows'] == phase['normalization']['signal']['unique_rows'] == local31.SCOPE['local_rows']
         and actual['fields'] == 31 and actual['compared_cells'] == actual['rows'] * 31
         and actual['mismatched_cells'] == actual['mask_mismatched_cells'] == 0
         and producer['producer_vs_cache']['float32_bit_mismatched_cells'] == 0
         and producer['oracle_vs_producer']['mismatched_cells'] == 0
         and producer['identical_HLC_ATR_and_origin_for_producer_and_oracle'] is True
         and source['times_ns_sha256'] == phase['local_context_formula_evidence']['times_ns_sha256'],
         'REUSED_ACTUAL_SOURCE_LOCAL31')
    need(fold.TA_INPUT_STAGE == STAGE and fold.TA_INPUT_ROOT == OUT_ROOT
         and phase['audit_scope'] == original.audit_scope(), 'UNCHANGED_INPUT_SCOPE')
    current_sources, reuse = _source_reuse_binding(phase, diagnostic)
    refs = {'tensor_phase': PHASE_REF, 'tensor_plan': phase['plan'], 'tensor_claim': phase['claim'],
        'tensor_terminal': old_terminal_ref, 'tensor_failure': old_terminal['failure'],
        'local31_result': LOCAL_RESULT, 'local31_terminal': LOCAL_TERMINAL,
        'local31_plan': diagnostic['plan'], 'original_candidate_view': phase['candidate_view'],
        'candidate_view': _ref(OUT_ROOT / 'CANDIDATE_VIEW.json'),
        'provider_plan': diagnostic['inputs']['provider_plan'],
        'provider_worker': diagnostic['inputs']['provider_worker']}
    need(refs['candidate_view']['sha256'] == refs['original_candidate_view']['sha256'], 'UNCHANGED_VIEW_BYTES')
    receipts = io.load_ta_input_receipts(current_source_owners=current_sources)
    view = io._bind_ta_view(receipts, max_batch_rows=fold.BATCH_SIZE)
    need(view == io._read_ref(refs['candidate_view'], limit=MAX_JSON_BYTES)
         and view['contract_sha256'] == phase['candidate_view_sha256']
         and view.get('full_ta_feature_profile') is True, 'UNCHANGED_BOUND_VIEW')
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as mature
    need(refs['provider_plan'] == mature.INPUT_REFS['provider_plan']
         and refs['provider_worker'] == mature.INPUT_REFS['provider_worker'], 'EXACT_PROVIDER')
    original.inspect_formula_inputs(old_plan['formula_inputs'])
    return refs, phase, diagnostic, old_plan, current_sources, reuse, receipts, view


def build_plan(*, census_rows_per_batch=1024):
    """Metadata-only construction, with no writes, claims or array decoding."""
    need(type(census_rows_per_batch) is int and census_rows_per_batch > 0, 'CENSUS_BATCH_ROWS')
    refs, phase, _, old_plan, owners, reuse, _, _ = _reuse_inputs()
    return {'schema_version': SCHEMA + '_plan', 'stage': STAGE, 'status': 'PREPARED_NOT_ADMITTED',
        'source_base_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'source_owners': owners, 'source_reuse_binding': reuse,
        'inputs': refs, 'formula_inputs': old_plan['formula_inputs'],
        'endpoint_inputs': original.target_input_refs(), 'scope': phase['audit_scope'],
        'endpoint_prior_fit_scope': {'fold_id': 1, 'fit_rows': fold.FIT_ROWS, 'score_fit_rows': 0},
        'this_invocation': dict(THIS_INVOCATION), 'census_rows_per_batch': census_rows_per_batch,
        'raw_census_statistics': ['finite_rows', 'missing_rows', 'raw_min', 'raw_max', 'raw_nonzero_rows'],
        'limits': dict(LIMITS), 'execution': dict(EXECUTION),
        'output_root': str(OUT_ROOT), 'max_json_bytes': MAX_JSON_BYTES}


def _policy_gate(policy, plan_ref):
    gate = policy.get(STAGE, {})
    zeros = ('model_constructions_authorized', 'model_forwards_authorized', 'backward_calls_authorized',
        'optimizer_steps_authorized', 'normalization_fits_authorized', 'normalization_calls_authorized',
        'gather_calls_authorized', 'local_trendline_replays_authorized')
    need(policy.get('training_enabled') is False and policy.get('native_training_enabled') is False
         and policy.get('native_launch_authorized') is False and policy.get('full_val_allowed') is False
         and gate.get('status') == 'ADMITTED' and gate.get('plan') == plan_ref
         and gate.get('worker') == _ref(Path(__file__).resolve())
         and gate.get('execution_authorized') is True and gate.get('data_reads_authorized') is True
         and gate.get('endpoint_prior_fit_authorized') is True and gate.get('training_authorized') is False
         and all(gate.get(key) == 0 for key in zeros)
         and gate.get('endpoint_prior_fit_rows') == fold.FIT_ROWS and gate.get('score_fit_rows') == 0
         and gate.get('test_data_used') is False and gate.get('relaunch_allowed') is False
         and gate.get('max_invocations') == 1 and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
         and gate.get('output_root') == str(OUT_ROOT), 'EXACT_COMPLETION_POLICY')


def inspect(plan_ref, *, require_policy):
    head = io._clean_head()
    need(plan_ref['path'] == str(PLAN_PATH), 'PLAN_PATH')
    plan = io._read_ref(plan_ref, limit=MAX_JSON_BYTES)
    expected = build_plan(census_rows_per_batch=plan.get('census_rows_per_batch'))
    need(set(plan) == set(expected)
         and all(plan[key] == expected[key] for key in expected if key != 'source_base_head')
         and subprocess.run(['git', 'merge-base', '--is-ancestor', plan['source_base_head'], head],
             cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0, 'PLAN_SCOPE')
    need(OUT_ROOT.is_dir() and OUT_ROOT.resolve(strict=True) == OUT_ROOT
         and {path.name for path in OUT_ROOT.iterdir()} == {'CANDIDATE_VIEW.json'}, 'UNCONSUMED_OUTPUT_ROOT')
    policy_ref = _ref(POLICY_PATH)
    if require_policy:
        _policy_gate(io._read_ref(policy_ref, limit=MAX_JSON_BYTES), plan_ref)
    _, phase, diagnostic, _, _, _, receipts, view = _reuse_inputs()
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as mature
    provider = mature._provider_module()
    integrity = original._provider_integrity_metadata(provider, receipts['provider_chain']['provider_plan'])
    return plan, policy_ref, head, receipts, view, phase, diagnostic, integrity


def preflight(plan_ref):
    plan, _, head, _, view, _, _, integrity = inspect(plan_ref, require_policy=False)
    return {'status': 'READY_NO_CLAIM_NO_FEATURE_PAYLOAD_NO_MODEL', 'plan': plan_ref,
        'source_head': head, 'candidate_view_sha256': view['contract_sha256'],
        'source_reuse_binding': plan['source_reuse_binding'], 'provider_integrity_metadata': integrity,
        'this_invocation': plan['this_invocation'], 'reused_model_forwards': 1,
        'feature_payloads_read': False, 'target_arrays_decoded': 0, 'test_data_used': False}


def rows_from_intervals(intervals, *, row_count):
    need(isinstance(intervals, list) and len(intervals) > 0 and type(row_count) is int and row_count > 0,
         'INTERVAL_SCOPE')
    previous = -1
    parts = []
    for pair in intervals:
        need(isinstance(pair, list) and len(pair) == 2 and all(type(value) is int for value in pair)
             and 0 <= pair[0] < pair[1] <= row_count and pair[0] > previous, 'INTERVAL_ORDER')
        parts.append(np.arange(pair[0], pair[1], dtype=np.int64))
        previous = pair[1]
    return np.concatenate(parts)


def raw_census(matrix, rows, columns, names, *, batch_rows, digest_columns=None):
    """Deduplicated raw rows only; missing data is reported, never imputed."""
    rows, columns = np.asarray(rows), np.asarray(columns)
    need(rows.dtype == columns.dtype == np.int64 and rows.ndim == columns.ndim == 1
         and len(rows) > 0 and len(columns) == len(names) > 0
         and np.all(np.diff(rows) > 0) and rows[0] >= 0 and rows[-1] < len(matrix)
         and np.all(columns >= 0) and np.all(columns < matrix.shape[1])
         and len(np.unique(columns)) == len(columns)
         and type(batch_rows) is int and batch_rows > 0, 'CENSUS_INPUT')
    width = len(columns)
    finite_rows = np.zeros(width, np.int64)
    nonzero = np.zeros(width, np.int64)
    lower, upper = np.full(width, np.inf), np.full(width, -np.inf)
    digest = hashlib.sha256() if digest_columns is not None else None
    if digest_columns is not None:
        need(isinstance(digest_columns, list) and len(digest_columns) > 0
             and all(type(i) is int and 0 <= i < width for i in digest_columns), 'CENSUS_DIGEST_COLUMNS')
    for left in range(0, len(rows), batch_rows):
        raw = np.ascontiguousarray(matrix[np.ix_(rows[left:left + batch_rows], columns)])
        finite = np.isfinite(raw)
        finite_rows += np.count_nonzero(finite, axis=0)
        nonzero += np.count_nonzero(finite & (raw != 0), axis=0)
        lower = np.minimum(lower, np.where(finite, raw, np.inf).min(0))
        upper = np.maximum(upper, np.where(finite, raw, -np.inf).max(0))
        if digest is not None:
            digest.update(np.ascontiguousarray(raw[:, digest_columns]).tobytes())
    result = {'unique_rows': len(rows), 'row_ids_sha256': original.axis_sha(rows),
        'fields': width, 'checked_values': len(rows) * width, 'field_names': list(names),
        'finite_rows': finite_rows.tolist(), 'missing_rows': (len(rows) - finite_rows).tolist(),
        'raw_min': [float(value) if np.isfinite(value) else None for value in lower],
        'raw_max': [float(value) if np.isfinite(value) else None for value in upper],
        'raw_nonzero_rows': nonzero.tolist(),
        'population': 'unique consumed rows; not weighted by overlapping model windows'}
    if digest is not None:
        result['selected_fields_sha256'] = digest.hexdigest()
        result['selected_digest_fields'] = [names[i] for i in digest_columns]
    return result


def census_and_unions(plan, prepared, phase, diagnostic, *, publish_measurement):
    from gx1.features.entry_specialist_feature_groups_v1 import compile_full_ta_entry_execution_profile
    from gx1.features.htf_features import MULTI_TF_SHIFT
    profile = compile_full_ta_entry_execution_profile()
    axes, source, current, cache = (prepared[key] for key in ('axes', 'entry_sequence_source', 'current', 'cache'))
    at = np.r_[np.arange(fold.FIT_ROWS, dtype=np.int64),
               np.arange(fold.FOLD['oof_start_row'], fold.FOLD['oof_end_row'], dtype=np.int64)]
    parents, issues = axes['parent_rows'][at], axes['decision_ns'][at]
    need(original.axis_sha(parents) == phase['gather']['parent_rows_sha256']
         and original.axis_sha(issues) == phase['gather']['decision_ns_sha256'], 'REUSED_PARENT_AXES')
    local_rows, intervals = original.interval_union(source['sample_positions'][at], 96)
    need(intervals == phase['gather']['local_source_row_intervals']
         and original.axis_sha(local_rows) == diagnostic['diagnostic']['source_comparison']['source_row_ids_sha256'],
         'REUSED_LOCAL_ROW_UNION')
    times = source['times_ns'][local_rows]
    need(original.axis_sha(times) == phase['local_context_formula_evidence']['times_ns_sha256']
         and len(local_rows) == phase['normalization']['signal']['unique_rows'], 'REUSED_LOCAL_TIME_AXIS')
    fields = profile['physical_field_names']
    need(list(fields['signal']) == phase['normalization']['signal']['field_names']
         and list(fields['ctx_cont']) == phase['normalization']['ctx_cont']['field_names']
         and (len(fields['signal']), len(fields['ctx_cont']), len(fields['ctx_cat']), len(fields['mtf'])) == (315, 132, 1, 253),
         'REUSED_PROFILE_FIELDS')
    signal_cols = np.asarray(profile['source_signal_indices'], dtype=np.int64)
    ctx_cols = np.asarray(profile['source_ctx_cont_indices'], dtype=np.int64)
    specs = {
        'seq_x': (source['signal'], local_rows, signal_cols, fields['signal']),
        'snap_x': (current['snap'], at, signal_cols, fields['signal']),
        'ctx_cont': (current['ctx_cont'], at, ctx_cols, fields['ctx_cont']),
        'ctx_cat': (current['ctx_cat'], at, np.arange(len(fields['ctx_cat']), dtype=np.int64), fields['ctx_cat']),
    }
    formula_unions, union_report = {}, {}
    for tf in original.TIMEFRAMES:
        frame = cache[tf]
        key = 'mtf_' + tf.lower()
        clock = np.asarray(frame.attrs['ts_int64'])
        gather_rows = rows_from_intervals(phase['gather']['mtf_cache_row_intervals'][tf], row_count=len(clock))
        normal = phase['normalization'][key]
        need(original.axis_sha(gather_rows) == normal['row_ids_sha256']
             and len(gather_rows) == normal['unique_rows']
             and list(frame.columns) == list(fields['mtf']) == normal['field_names'], 'REUSED_MTF_GATHER_UNION:' + tf)
        context_rows = np.unique(original.closed_windows(clock, times + 300_000_000_000,
            duration_ns=int(MULTI_TF_SHIFT[tf].value), length=1,
            warmup=int(frame.attrs['causal_warmup_rows']))[:, 0])
        bound = phase['local_context_formula_evidence']['required_mtf_rows'][key]
        need(len(context_rows) == bound['rows'] and original.axis_sha(context_rows) == bound['row_ids_sha256'],
             'REUSED_CONTEXT_FORMULA_UNION:' + tf)
        union = np.union1d(gather_rows, context_rows).astype(np.int64)
        if tf == 'M5':
            aliases = np.searchsorted(clock, times).astype(np.int64)
            need(np.all(aliases < len(clock)) and np.array_equal(clock[aliases], times)
                 and np.array_equal(aliases, context_rows)
                 and len(aliases) == phase['local_formula_alias_bridge']['rows']
                 and phase['local_formula_alias_bridge']['fields'] == 186, 'REUSED_LOCAL_ALIAS_AXIS')
            union = np.union1d(union, aliases).astype(np.int64)
        formula_unions[key] = union
        union_report[tf] = {'gather_rows': len(gather_rows), 'gather_row_ids_sha256': original.axis_sha(gather_rows),
            'context_rows': len(context_rows), 'context_row_ids_sha256': original.axis_sha(context_rows),
            'formula_rows': len(union), 'formula_row_ids_sha256': original.axis_sha(union)}
        specs['seq_' + tf.lower()] = (frame.attrs['feats_np'], gather_rows,
            np.arange(len(fields['mtf']), dtype=np.int64), fields['mtf'])
    need(set(specs) == set(profile['feature_keys']) and len(specs) == 9, 'NINE_CENSUS_SURFACES')
    census = {}
    for name, (matrix, rows, columns, names) in specs.items():
        digest_columns = ([list(names).index('chart.' + field) for field in local31.NAMES] if name == 'seq_x' else None)
        census[name] = raw_census(matrix, rows, columns, names,
            batch_rows=plan['census_rows_per_batch'], digest_columns=digest_columns)
    result = {'surfaces': census, 'formula_unions': union_report,
        'local_times_ns_sha256': original.axis_sha(times), 'fit_parent_rows': fold.FIT_ROWS,
        'score_parent_rows': fold.SCORE_ROWS, 'statistics_fitted': False,
        'normalized_extrema_recomputed_or_inferred': False, 'model_forwards': 0, 'gather_calls': 0,
        'normalization_calls': 0, 'test_data_used': False}
    ref = publish_measurement('RAW_CENSUS_AND_FORMULA_UNIONS.json', result)
    need(all(not any(report['missing_rows']) for report in census.values()), 'CENSUS_NONFINITE')
    need(census['seq_x']['selected_fields_sha256'] == diagnostic['diagnostic']['source_comparison']['cache_values_sha256'],
         'REUSED_LOCAL31_VALUE_BYTES')
    return formula_unions, result, ref


def complete_mtf_gaps(formula_inputs, cache, unions, *, publish_measurement):
    import pandas as pd
    from gx1.audit import verify_market_state_remaining_mtf_numeric_v1 as residual
    from gx1.audit import verify_m5_m15_h1_trendline_oracle_v1 as lines
    from gx1.audit import verify_h4_trendline_lifecycle_oracle_v1 as files
    from gx1.audit.verify_d1_trendline_lifecycle_oracle_v1 import oracle_rows
    from gx1.scripts.audit_direction_d1_trendline_replay_v1 import load_native_parts_prefix
    from gx1.scripts import audit_entry_gc_foundation_v1 as independent
    from gx1.features.technical_indicators_v1 import wilder_rsi, wilder_atr
    from gx1.features.volatility_squeeze_state_v1 import require_volatility_squeeze_artifact_binding
    from gx1.features.htf_features import MultiTFV4DiskCache
    bound, manifest, qualified = original.inspect_formula_inputs(formula_inputs)
    need(type(cache) is MultiTFV4DiskCache and cache.manifest_sha256 == bound['cache_manifest']['sha256'], 'FORMULA_CACHE')
    names = residual._field_contract(manifest, qualified)
    missing = {}
    for tf in original.TIMEFRAMES:
        rows = unions['mtf_' + tf.lower()]
        frame = cache[tf]
        values, clock = frame.attrs['feats_np'], frame.attrs['ts_int64']
        need(len(rows) > 0 and rows[-1] <= residual.TF_SCOPE[tf]['last_row']
             and list(frame.columns) == manifest['feature_names']
             and values.dtype == np.float32 and values.shape == (len(clock), 253)
             and not values.flags.writeable
             and any(np.shares_memory(values, backing) for backing in cache._disk_backing_arrays), 'FORMULA_ROWS:' + tf)
        missing[tf] = rows[rows < residual.TF_SCOPE[tf]['first_row']]
    cutoffs = [int(cache[tf].attrs['ts_int64'][rows[-1]]) + residual.TF_SCOPE[tf]['bar_ns']
               for tf, rows in missing.items() if len(rows)]
    need(len(cutoffs) > 0, 'DECLARED_HISTORY_GAPS')
    cutoff = max(cutoffs)
    snapshots = {}
    for ref in bound['native_parts']:
        path, stat = files.checked(ref)
        snapshots[str(path)] = stat
    source = load_native_parts_prefix(bound['native_parts'], cutoff)
    volume = source['volume'].to_numpy()
    need(np.isfinite(volume).all() and np.all(volume > 0) and np.all(volume == np.floor(volume))
         and np.all(volume < np.iinfo(np.int64).max), 'SOURCE_QUOTE_COUNTS')
    m5 = {'bar_start_ns': source.index.asi8,
        **{name: source[name].to_numpy(dtype=np.float64) for name in ('open', 'high', 'low', 'close')},
        'volume': source['volume'].to_numpy(dtype=np.int64)}
    frozen = require_volatility_squeeze_artifact_binding(manifest['volatility_squeeze_artifact_set'])
    reports, refs = {}, {}
    for tf, rows in missing.items():
        tic = time.monotonic()
        report = {'additional_rows': len(rows), 'newly_compared_fields': 94,
            'formula_union_rows': len(unions['mtf_' + tf.lower()]),
            'formula_union_row_ids_sha256': original.axis_sha(unions['mtf_' + tf.lower()]),
            'mismatched_cells': 0, 'local31_replayed': False}
        if len(rows):
            resampled = independent.native_resample_calendar_frame(m5, tf)
            times = np.asarray(cache[tf].attrs['ts_int64'][rows], dtype=np.int64)
            positions = np.searchsorted(resampled['bar_start_ns'], times).astype(np.int64)
            need(np.all(positions < len(resampled['bar_start_ns']))
                 and np.array_equal(resampled['bar_start_ns'][positions], times), 'ORACLE_SOURCE_CLOCK:' + tf)
            bars = pd.DataFrame({name: values for name, values in resampled.items() if name != 'bar_start_ns'},
                index=pd.to_datetime(resampled['bar_start_ns'], utc=True))
            rsi = wilder_rsi(bars['close'], 14).to_numpy(dtype=np.float64)
            atr = wilder_atr(bars['high'], bars['low'], bars['close'], 14).to_numpy(dtype=np.float64)
            atr = np.where(atr > 0., atr, np.nan)
            prerequisites = residual._require_prerequisites(bars['high'].to_numpy()[positions],
                bars['low'].to_numpy()[positions], bars['close'].to_numpy()[positions], rsi[positions], atr[positions],
                np.asarray(cache[tf].attrs['feats_np'][np.ix_(rows, [0, 1])]))
            bars, rsi, atr = original._formula_prefix(bars, positions, rsi, atr)
            oracle63 = residual._oracle_matrix(bars, positions, tf, frozen.require_params(tf)['fit'], names, rsi, atr)
            config = residual.TF_SCOPE[tf]
            oracle31, events = oracle_rows(bars['high'].to_numpy(), bars['low'].to_numpy(), bars['close'].to_numpy(), atr,
                seq_len=config['seq_len'], band_atr=config['band_atr'], witness_positions=positions)
            comparisons, offset = [], 0
            for part in original.contiguous_parts(rows):
                right = offset + len(part)
                observed = cache[tf].attrs['feats_np']
                a = residual._compare(oracle63[offset:right], np.asarray(observed[np.ix_(part, residual.EXPECTED_COLUMNS)]),
                    part, times[offset:right], names)
                b = lines._compare(oracle31[offset:right], np.asarray(observed[part, 159:190]), part, times[offset:right], set())
                comparisons.append({'remaining63': a, 'trendline31': b})
                offset = right
            report.update({'cache_row_first': int(rows[0]), 'cache_row_last': int(rows[-1]),
                'row_ids_sha256': original.axis_sha(rows), 'native_prefix_rows': len(bars),
                'prerequisites': prerequisites, 'comparisons': comparisons, 'trendline_events': events,
                'mismatched_cells': sum(item['remaining63']['mismatched_cells'] + item['trendline31']['mismatched_cells']
                    for item in comparisons)})
            del bars, resampled, rsi, atr, oracle63, oracle31
        report['elapsed_seconds'] = time.monotonic() - tic
        for path, stat in snapshots.items():
            need(files.regular(Path(path)) == stat, 'NATIVE_PREFIX_CHANGED')
        reports[tf] = report
        refs[tf] = publish_measurement('MTF_HISTORY_' + tf + '.json', report)
        need(report['mismatched_cells'] == 0, 'MTF_HISTORY_FORMULA_MISMATCH:' + tf)
        gc.collect()
    return {'prior_union_fields': 253, 'additional_rows': reports, 'measurement_refs': refs,
        'prior_158_full_history_reused': True, 'prior_close236_full_history_reused': True,
        'prior_31_63_may_dec2024_reused': True, 'all_consumed_mtf_rows_covered': True,
        'native_source_cutoff_exclusive_ns': cutoff, 'no_reseed_or_gap_reset': True,
        'OHLC_resampler': 'independent.native_resample_calendar_frame',
        'local31_replayed': False, 'test_data_used': False}


def compose_input_audit(phase, diagnostic, view, mtf, census_ref):
    """Counters describe the bound evidence; this invocation is separate."""
    result = {'schema_version': original.SCHEMA,
        'candidate_view_sha256': view['contract_sha256'], 'profile': view['entry_execution_profile'],
        'scope': phase['audit_scope'],
        **{key: phase[key] for key in ('gather', 'normalization', 'family_consumption',
            'local_formula_alias_bridge', 'local_context_formula_evidence')},
        'mtf_formula_evidence': mtf,
        'local_trendline_formula_evidence': {'result': LOCAL_RESULT, 'terminal': LOCAL_TERMINAL,
            'actual_source_comparison': diagnostic['diagnostic']['measurements']['actual2010'],
            'producer_comparisons': diagnostic['diagnostic']['measurements']['actual2010_producer'],
            'source_comparison': diagnostic['diagnostic']['source_comparison'],
            'source_origin_is_actual_enriched_2010': True, 'replayed_in_completion': False},
        'raw_census': census_ref,
        'evidence_execution': {'reused_tensor_phase': PHASE_REF, 'reused_local31': LOCAL_RESULT,
            'model_constructions_in_reused_tensor_phase': 1, 'model_forwards_in_reused_tensor_phase': 1,
            'this_invocation': dict(THIS_INVOCATION)},
        'limits_of_evidence': list(phase['limits_of_evidence']) + [
            'Normalized extrema omitted from the original phase receipt remain unavailable; they were not reconstructed.',
            'A passing local31 actual-source comparison does not restore line identity to nearest-line model slots.'],
        'model_constructions': 1, 'model_forwards': 1, 'normalization_fits': 0,
        'backward_calls': 0, 'optimizer_steps': 0, 'targets_used_as_inputs': False,
        'full_input_audit_pass': True, 'test_data_used': False, 'predictive_value_claimed': False}
    need(phase['model_constructions'] == phase['model_forwards'] == 1
         and result['candidate_view_sha256'] == phase['candidate_view_sha256']
         and mtf['all_consumed_mtf_rows_covered'] is True, 'COMPOSED_EVIDENCE_SCOPE')
    return result


def _fit_priors(fit_rows, issues):
    from gx1.contracts import market_state_five_endpoint_target_binding_v1 as state
    from gx1.contracts import market_state_future_displacement_candidate_v1 as future
    from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
    priors = objective.fit_endpoint_priors(state.load_bound_five_endpoint_targets(),
        future.load_bound_future_displacement(), axes={'fit_parent_rows': fit_rows, 'fit_decision_ns': issues},
        fold_id=1, ta_only=True)
    need(priors['fit_rows'] == fold.FIT_ROWS
         and priors['fit_parent_rows_sha256'] == original.axis_sha(fit_rows)
         and priors['fit_decision_ns_sha256'] == original.axis_sha(issues), 'EXACT_FIT_PRIOR_AXIS')
    objective.require_endpoint_priors(priors, fold_id=1, ta_only=True)
    return priors


def run(plan_ref):
    import os
    import resource
    import signal
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as mature
    from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner
    from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_research_producer_execution
    cap = require_capped_cpu_research_producer_execution()
    need(all(cap[key] == LIMITS[key] for key in ('memory_max', 'memory_high', 'swap', 'pids', 'numerical_threads')), 'CAPS')
    began = time.monotonic()
    plan, policy_ref, head, receipts, view, phase, diagnostic, _ = inspect(plan_ref, require_policy=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    need(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'AMBIENT_TIMER')
    published, bytes_written = {}, 0
    provider = provider_plan = prepared = None
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
             and all(io._sha(Path(ref['path'])) == ref['sha256'] for ref in plan['inputs'].values()), 'SOURCE_OR_REUSE_BOOKEND')
        if provider is not None:
            original._provider_integrity_metadata(provider, provider_plan)
    remaining = LIMITS['wall_seconds'] - 30 - (time.monotonic() - began)
    need(remaining > 0., 'PRECLAIM_WALL_BUDGET')
    claim = publish('CLAIM.json', {'schema_version': SCHEMA + '_claim', 'plan': plan_ref, 'policy': policy_ref,
        'source_head': head, 'source_reuse_binding': plan['source_reuse_binding'], 'cap': cap,
        'candidate_view': plan['inputs']['candidate_view'], 'reused_tensor_phase': PHASE_REF,
        'reused_local31': LOCAL_RESULT, 'this_invocation': plan['this_invocation'], 'test_data_used': False})
    old_alarm = signal.getsignal(signal.SIGALRM)
    def expired(_signum, _frame):
        raise TimeoutError('TA_INPUT_COMPLETION_3600_SECONDS')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    def publish_measurement(name, measurement):
        bookend()
        return publish(name, {'schema_version': SCHEMA + '_measurement', 'plan': plan_ref, 'claim': claim,
            'source_head': head, 'measurement': measurement, 'full_input_audit_pass': False,
            'this_invocation': plan['this_invocation'], 'test_data_used': False})
    try:
        setup = time.monotonic()
        provider, provider_plan, prepared, fit_rows, issues = runner._prepared_fold1(receipts, view, np=np)
        setup_seconds = time.monotonic() - setup
        unions, census, census_ref = census_and_unions(plan, prepared, phase, diagnostic,
            publish_measurement=publish_measurement)
        mtf = complete_mtf_gaps(plan['formula_inputs'], prepared['cache'], unions, publish_measurement=publish_measurement)
        priors = _fit_priors(fit_rows, issues)
        bookend()
        priors_ref = publish('ENDPOINT_PRIORS.json', priors)
        measurement = compose_input_audit(phase, diagnostic, view, mtf, census_ref)
        mature._release_pages(provider, prepared)
        prepared = None
        bookend()
        result = publish('RESULT.json', {'schema_version': SCHEMA + '_result', 'status': 'COMPLETE',
            'completion_mode': 'immutable_tensor_and_actual_local31_reuse_plus_new_mtf_gaps_and_fit_priors',
            'plan': plan_ref, 'claim': claim, 'policy': policy_ref, 'source_head': head,
            'source_owners': plan['source_owners'], 'source_reuse_binding': plan['source_reuse_binding'],
            'candidate_view': plan['inputs']['candidate_view'], 'endpoint_priors': priors_ref,
            'endpoint_prior_fit_rows': priors['fit_rows'],
            'input_tensor_phase': PHASE_REF, 'local31_qualification': LOCAL_RESULT,
            'raw_census': census_ref, 'mtf_history_measurements': mtf['measurement_refs'],
            'input_audit': measurement, 'limits_of_evidence': measurement['limits_of_evidence'],
            'this_invocation': plan['this_invocation'], **THIS_INVOCATION,
            'input_integrity': {'provider_metadata_before_and_after': True,
                'axes_NPZ_members_decoded': ['parent_rows', 'bar_start_ns', 'decision_ns'],
                'provider_axes_target_labels_decoded': 0,
                'S_F_targets_separately_loaded_for_fit_only_priors': True,
                'native_MTF_payloads_fresh_SHA_verified_by_canonical_loader': True,
                'local_decode_fresh_body_hash_claimed': False},
            'input_setup_seconds': setup_seconds, 'elapsed_seconds': time.monotonic() - began,
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'source_unchanged': True, 'full_input_audit_pass': True, 'predictive_value_claimed': False})
        terminal = publish('TERMINAL.json', {'schema_version': SCHEMA + '_terminal', 'status': 'COMPLETE',
            'exit_code': 0, 'plan': plan_ref, 'claim': claim, 'result': result, 'endpoint_priors': priors_ref,
            'input_tensor_phase': PHASE_REF, 'local31_qualification': LOCAL_RESULT,
            'this_invocation': plan['this_invocation'], 'no_relaunch_under_this_plan': True, 'test_data_used': False})
        return {'result': result, 'terminal': terminal, 'endpoint_priors': priors_ref}
    except BaseException as exc:
        completed = {name: ref for name, ref in published.items() if name not in ('CLAIM.json', 'FAILED.json', 'TERMINAL.json')}
        failed = publish('FAILED.json', {'schema_version': SCHEMA + '_failure', 'plan': plan_ref, 'claim': claim,
            'source_head': head, 'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'completed_measurements': completed, 'this_invocation': plan['this_invocation'],
            'elapsed_seconds': time.monotonic() - began, 'test_data_used': False})
        publish('TERMINAL.json', {'schema_version': SCHEMA + '_terminal', 'status': 'FAILED_CONSUMED',
            'exit_code': 1, 'plan': plan_ref, 'claim': claim, 'failure': failed,
            'completed_measurements': completed, 'this_invocation': plan['this_invocation'],
            'no_relaunch_under_this_plan': True, 'test_data_used': False})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, old_alarm)
        if prepared is not None:
            mature._release_pages(provider, prepared)


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
