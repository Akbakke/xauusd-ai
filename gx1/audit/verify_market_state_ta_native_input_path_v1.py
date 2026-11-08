"""Measure the exact TA fold-1 source -> gather -> normalizer -> family path.

The caller owns admission, immutable input/receipt loading, capped execution,
publication and target priors. This owner never loads labels or fits anything.
Formula parity, tensor delivery and one genuine forward are separate evidence.
"""
from __future__ import annotations

import gc
import hashlib
import time
from pathlib import Path

import numpy as np

from gx1.contracts import market_state_time_query_fold1_fit_v1 as fold
from gx1.audit import verify_market_state_time_query_v2_zero_step_v1 as witness_owner
from gx1.scripts import audit_dynamic_m5_october_native_postnorm_v1 as norm_evidence

TIMEFRAMES = ('M5', 'M15', 'H1', 'H4', 'D1')
SCHEMA = 'gx1_market_state_ta_native_input_path_v1'


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError('TA_NATIVE_INPUT_PATH_' + reason)


def axis_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype='<i8').tobytes()).hexdigest()


def exact(actual, expected, *, label):
    """Require storage parity, including float32 signed zero and field order."""
    a, b = np.asarray(actual), np.asarray(expected)
    need(a.shape == b.shape and a.dtype == b.dtype, 'SHAPE_DTYPE:' + label)
    need(np.isfinite(a).all() and np.isfinite(b).all(), 'NONFINITE:' + label)
    if a.dtype == np.float32:
        a, b = a.view(np.uint32), b.view(np.uint32)
    if not np.array_equal(a, b):
        first = tuple(int(i) for i in np.argwhere(a != b)[0])
        need(False, 'BIT_PARITY:' + label + ':' + str(first))


def closed_windows(clock, decisions, *, duration_ns, length, warmup):
    """Independent integer as-of selection; sparse observed clocks stay sparse."""
    clock, decisions = np.asarray(clock), np.asarray(decisions)
    need(clock.dtype == decisions.dtype == np.int64 and clock.ndim == decisions.ndim == 1
         and len(clock) > 0 and len(decisions) > 0
         and np.all(np.diff(clock) > 0) and np.all(np.diff(decisions) > 0)
         and type(duration_ns) is int and duration_ns > 0
         and type(length) is int and length > 0
         and type(warmup) is int and 0 <= warmup < len(clock), 'CLOCK_INPUT')
    need(np.all(decisions >= duration_ns)
         and np.all(clock <= np.iinfo(np.int64).max - duration_ns), 'CLOCK_OVERFLOW')
    end = np.searchsorted(clock, decisions - duration_ns, side='right') - 1
    need(np.all(end - length + 1 >= warmup) and np.all(end < len(clock)), 'HISTORY_WARMUP')
    ids = end[:, None] + np.arange(1 - length, 1, dtype=np.int64)[None, :]
    need(np.all(clock[ids] + duration_ns <= decisions[:, None]), 'FUTURE_BAR')
    after = end + 1
    present = after < len(clock)
    need(np.all(clock[after[present]] + duration_ns > decisions[present]), 'NOT_LATEST_CLOSED')
    return ids


def interval_union(endpoints, length):
    """Deduplicate complete consumed windows without filling unconsumed gaps."""
    raw = np.asarray(endpoints)
    need(raw.dtype == np.int64, 'INTERVAL_DTYPE')
    ends = np.unique(raw)
    need(ends.ndim == 1 and len(ends) > 0 and type(length) is int
         and length > 0 and ends[0] >= length - 1, 'INTERVAL_INPUT')
    intervals = []
    for end in ends:
        left, right = int(end) - length + 1, int(end) + 1
        if intervals and left <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], right)
        else:
            intervals.append([left, right])
    ids = np.concatenate([np.arange(left, right, dtype=np.int64) for left, right in intervals])
    return ids, intervals


def contiguous_parts(rows):
    rows = np.asarray(rows, dtype=np.int64)
    need(rows.ndim == 1 and len(rows) > 0 and np.all(np.diff(rows) > 0), 'ROW_UNION')
    return np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1)


def audit_scope():
    """Exact existing population and preselected genuine forward witnesses."""
    return {'fold': dict(fold.FOLD), 'fit_rows': fold.FIT_ROWS,
            'score_rows': fold.SCORE_ROWS, 'gather_batch_size': fold.BATCH_SIZE,
            'witness_parent_rows': list(witness_owner.PARENT_ROWS),
            'witness_issue_ns': list(witness_owner.ISSUE_NS), 'seed': witness_owner.SEED,
            'model_constructions': 1, 'model_forwards': 1, 'backward_calls': 0,
            'optimizer_steps': 0, 'normalization_fits': 0, 'test_data_used': False}


def inspect_formula_inputs(formula_inputs):
    """Small existing receipts only; no old one-shot plan is relaunched."""
    from gx1.audit import verify_market_state_remaining_mtf_numeric_v1 as residual
    from gx1.audit import verify_h4_trendline_lifecycle_oracle_v1 as files
    need(set(formula_inputs) == {'remaining_plan', 'remaining_result', 'remaining_terminal'},
         'FORMULA_INPUT_KEYS')
    previous = {key: files.bound_json(ref) for key, ref in formula_inputs.items()}
    plan, result, terminal = (previous[key] for key in
                             ('remaining_plan', 'remaining_result', 'remaining_terminal'))
    need(formula_inputs['remaining_plan']['path'] == str(residual.PLAN_PATH)
         and result['plan'] == formula_inputs['remaining_plan']
         and terminal['result'] == formula_inputs['remaining_result']
         and result['status'] == terminal['status'] == 'COMPLETE'
         and terminal['exit_code'] == 0 and result['total_mismatched_cells'] == 0
         and result['input_refs'] == plan['input_refs']
         and result['test_data_used'] is False and terminal['test_data_used'] is False,
         'COMPLETED_FORMULA_CHAIN')
    inputs, manifest, _, qualified = residual._small_receipts()
    need(plan['input_refs']['qualified_index'] == residual.QUALIFIED_INDEX,
         'QUALIFIED_INDEX_BINDING')
    for path, digest in plan['source_owners'].items():
        if path.startswith(('gx1/features/', 'gx1/audit/', 'gx1/scripts/')):
            need(residual._sha(Path(residual.REPO) / path) == digest, 'PRIOR_NUMERIC_OWNER_CHANGED:' + path)
    for tf in TIMEFRAMES:
        comparison = result['timeframes'][tf]['comparison']
        need(comparison['mismatched_cells'] == 0 and comparison['fields'] == 63
             and comparison['rows'] == residual.TF_SCOPE[tf]['last_row']
                 - residual.TF_SCOPE[tf]['first_row'] + 1, 'PRIOR_FORMULA_SCOPE:' + tf)
    return inputs, manifest, qualified


def _normalization_measure(model, normalized, *, matrices, unions, fields, batch_rows):
    import torch
    from gx1.contracts import entry_model_native_input_normalization_v1 as norm
    report = {}
    with norm_evidence.forbid_model_work(torch) as counts, torch.no_grad():
        for surface, matrix in matrices.items():
            rows = unions[surface]
            contract = normalized['surfaces'][surface]
            need(contract['field_names'] == list(fields[surface]), 'NORMALIZATION_FIELDS:' + surface)
            width = len(contract['field_names'])
            raw_min, raw_max = np.full(width, np.inf), np.full(width, -np.inf)
            norm_min, norm_max = raw_min.copy(), raw_max.copy()
            nonzero = np.zeros(width, dtype=np.int64)
            maximum_error = 0.
            identity = np.asarray(contract['binary_mask'], bool) | np.asarray(contract['categorical_mask'], bool)
            for start in range(0, len(rows), batch_rows):
                raw = np.ascontiguousarray(matrix[rows[start:start + batch_rows]], dtype=np.float32)
                expected = norm.apply_surface_normalization(raw, contract)
                actual = model.backbone._normalize_input_surface(
                    torch.from_numpy(raw), surface=surface).cpu().numpy()
                need(actual.shape == raw.shape and np.isfinite(actual).all()
                     and np.allclose(actual, expected, rtol=norm_evidence.RTOL,
                                     atol=norm_evidence.ATOL), 'NORMALIZER_PARITY:' + surface)
                exact(actual[:, identity], raw[:, identity], label='IDENTITY:' + surface)
                maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
                raw_min = np.minimum(raw_min, raw.min(0)); raw_max = np.maximum(raw_max, raw.max(0))
                norm_min = np.minimum(norm_min, actual.min(0)); norm_max = np.maximum(norm_max, actual.max(0))
                nonzero += np.count_nonzero(raw, axis=0)
            report[surface] = {'unique_rows': len(rows), 'row_ids_sha256': axis_sha(rows),
                'checked_values': len(rows) * width, 'maximum_absolute_error': maximum_error,
                'field_names': list(fields[surface]), 'raw_min': raw_min.tolist(),
                'raw_max': raw_max.tolist(), 'normalized_min': norm_min.tolist(),
                'normalized_max': norm_max.tolist(), 'raw_nonzero_rows': nonzero.tolist()}
    need(all(value == 0 for value in counts.values()), 'NORMALIZER_NO_FORWARD')
    return report


def _family_witness(model, raw, normalized, profile):
    """Observe actual pre-projection tensors on one native representation call."""
    import torch
    from gx1.contracts import entry_model_native_input_normalization_v1 as norm
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    backbone = model.backbone
    families = tuple(profile['local_specialists'])
    need(len(families) == 8 and tuple(backbone._specialist_names) == families
         and tuple(backbone._mtf_specialist_names) == families
         and backbone.gc_mtf_history is None, 'RUNTIME_FAMILY_PROFILE')
    captured, gates, nominal, handles = {}, {}, {}, []
    def input_hook(key):
        def hook(_module, args):
            need(len(args) == 1 and isinstance(args[0], torch.Tensor), 'HOOK_INPUT:' + key)
            captured.setdefault(key, []).append(args[0].detach().cpu().numpy().copy())
        return hook
    def output_hook(key):
        def hook(_module, _args, output):
            gates[key] = output.detach().cpu().numpy().copy()
        return hook
    def nominal_hook(key):
        def hook(_module, args):
            nominal.setdefault(key, []).append(args[0].detach().cpu().numpy().copy())
        return hook
    for family in families:
        handles.append(backbone.specialist_proj[family].register_forward_pre_hook(input_hook('local:' + family)))
        handles.append(backbone.mtf_family_proj[family].register_forward_pre_hook(input_hook('mtf:' + family)))
        if family in backbone.specialist_ctx_cont_proj:
            handles.append(backbone.specialist_ctx_cont_proj[family].register_forward_pre_hook(input_hook('ctx:' + family)))
        for tf in TIMEFRAMES:
            key = tf.lower() + '__' + family
            handles.append(backbone.mtf_feature_context_gate[key].register_forward_hook(output_hook(key)))
    for index, embedding in backbone.signal_nominal_embeddings.items():
        handles.append(embedding.register_forward_pre_hook(nominal_hook('signal:' + index)))
    for key, embedding in backbone.mtf_nominal_embeddings.items():
        handles.append(embedding.register_forward_pre_hook(nominal_hook('mtf:' + key)))
    for index, embedding in backbone.specialist_ctx_cont_nominal_embeddings.items():
        handles.append(embedding.register_forward_pre_hook(nominal_hook('ctx:' + index)))
    for index, embedding in enumerate(backbone.ctx_cat_embeddings):
        handles.append(embedding.register_forward_pre_hook(nominal_hook('cat:' + str(index))))
    before = canonical_model_state_sha256(model.state_dict())
    tensors = {key: torch.from_numpy(value.copy()) for key, value in raw.items()}
    model.eval()
    calls = []
    handles.append(backbone.register_forward_hook(lambda _m, _a, _o: calls.append(1)))
    try:
        with torch.no_grad(), trainer._training_mha_fastpath_context(backbone), \
             trainer._training_model_finite_check_context():
            output = backbone(**tensors, research_representation_only=True)
        need(len(calls) == 1 and torch.isfinite(output['entry_q_joint_hidden']).all(), 'GENUINE_FORWARD')
    finally:
        for handle in handles:
            handle.remove()
    need(canonical_model_state_sha256(model.state_dict()) == before, 'FORWARD_MUTATED_STATE')
    expected = {}
    for key, surface in (('seq_x', 'signal'), ('ctx_cont', 'ctx_cont'),
                         *(('seq_' + tf.lower(), 'mtf_' + tf.lower()) for tf in TIMEFRAMES)):
        expected[key] = norm.apply_surface_normalization(raw[key], normalized['surfaces'][surface])
    seq_numeric = expected['seq_x'].copy()
    seq_numeric[..., np.asarray(normalized['surfaces']['signal']['categorical_mask'], bool)] = 0.
    report = {}
    for family in families:
        local_idx = np.asarray(profile['local_signal_indices_by_specialist'][family], dtype=np.int64)
        mtf_idx = np.asarray(profile['mtf_indices_by_specialist'][family], dtype=np.int64)
        exact(getattr(backbone, 'specialist_idx_' + family).cpu().numpy(), local_idx, label='LOCAL_ROUTE:' + family)
        exact(getattr(backbone, 'multi_tf_specialist_idx_' + family).cpu().numpy(), mtf_idx, label='MTF_ROUTE:' + family)
        need(len(captured.get('local:' + family, ())) == 1
             and len(captured.get('mtf:' + family, ())) == len(TIMEFRAMES), 'FAMILY_HOOK_COUNTS:' + family)
        need(np.allclose(captured['local:' + family][0], seq_numeric[..., local_idx],
                         rtol=norm_evidence.RTOL, atol=norm_evidence.ATOL), 'LOCAL_PREPROJ:' + family)
        ctx_idx = getattr(backbone, 'specialist_ctx_cont_idx_' + family).cpu().numpy()
        nominal_idx = getattr(backbone, 'specialist_ctx_cont_nominal_idx_' + family).cpu().numpy()
        need(sorted([*ctx_idx.tolist(), *nominal_idx.tolist()])
             == sorted(profile['ctx_cont_indices_by_specialist'][family]), 'CONTEXT_ROUTE:' + family)
        if len(ctx_idx):
            need(len(captured.get('ctx:' + family, ())) == 1
                 and np.allclose(captured['ctx:' + family][0], expected['ctx_cont'][:, ctx_idx],
                                 rtol=norm_evidence.RTOL, atol=norm_evidence.ATOL), 'CONTEXT_PREPROJ:' + family)
        tf_report = {}
        for index, tf in enumerate(TIMEFRAMES):
            suffix = tf.lower(); surface = 'mtf_' + suffix
            values = expected['seq_' + suffix].copy()
            values[..., np.asarray(normalized['surfaces'][surface]['categorical_mask'], bool)] = 0.
            # The learned gate is observed; its declared sigmoid and learned TF
            # scale are applied to independently normalized source values.
            logit = gates[suffix + '__' + family]
            gate = (2. / (1. + np.exp(-logit.astype(np.float64)))).astype(np.float32)
            scale = float(backbone._effective_tf_input_scale(suffix).detach().cpu())
            reference = values[..., mtf_idx] * gate[:, None, :] * np.float32(scale)
            actual = captured['mtf:' + family][index]
            need(np.isfinite(actual).all() and np.allclose(actual, reference,
                 rtol=norm_evidence.RTOL, atol=norm_evidence.ATOL), 'MTF_PREPROJ:' + tf + ':' + family)
            tf_report[tf] = {'values': int(actual.size), 'maximum_absolute_error':
                            float(np.max(np.abs(actual - reference)))}
        report[family] = {'local_fields': len(local_idx), 'context_fields': len(ctx_idx) + len(nominal_idx),
                          'mtf_fields': len(mtf_idx), 'timeframes': tf_report}
    for key, values in nominal.items():
        kind, index = key.split(':', 1)
        if kind == 'signal': reference = raw['seq_x'][..., int(index)].astype(np.int64)
        elif kind == 'ctx': reference = raw['ctx_cont'][:, int(index)].astype(np.int64)
        elif kind == 'cat': reference = raw['ctx_cat'][:, int(index)]
        else:
            tf, column = index.rsplit('_', 1)
            reference = raw['seq_' + tf][..., int(column)].astype(np.int64)
        # Signal embeddings can also receive the current snapshot in the base
        # path. Each invocation must match its genuine sequence or last row.
        for actual in values:
            selected = reference[:, -1] if kind == 'signal' and actual.ndim == 1 else reference
            exact(actual, selected, label='NOMINAL:' + key)
    need(len(nominal) == len(backbone.signal_nominal_embeddings)
         + len(backbone.mtf_nominal_embeddings)
         + len(backbone.specialist_ctx_cont_nominal_embeddings)
         + len(backbone.ctx_cat_embeddings), 'ALL_NOMINAL_ROUTES_CALLED')
    return {'families': report, 'model_constructions': 1, 'backbone_forwards': 1,
            'backward_calls': 0, 'optimizer_steps': 0, 'state_sha256_before_after': before,
            'nominal_routes_checked': len(nominal), 'macro_fields_preserved': len(profile['macro_context_fields']),
            'predictive_influence_or_value_measured': False}


def _formula_prefix(bars, positions, rsi, atr):
    "Return the complete causal prefix ending at this oracle's last witness."
    positions = np.asarray(positions)
    need(positions.dtype == np.int64 and positions.ndim == 1 and len(positions) > 0
         and positions[0] >= 0 and np.all(np.diff(positions) > 0)
         and positions[-1] < len(bars)
         and rsi.dtype == atr.dtype == np.float64
         and rsi.shape == atr.shape == (len(bars),), 'ORACLE_PREFIX_INPUT')
    stop = int(positions[-1]) + 1
    # Only the unused tail is removed. Index zero and every state/warmup row
    # remain present; RSI/ATR retain their original full-history initialization.
    return bars.iloc[:stop], rsi[:stop], atr[:stop]


def _history_formula_measure(formula_inputs, cache, unions, *, local_trendline):
    """Extend MTF history gaps and qualify all consumed genuine local-96 lines."""
    import pandas as pd
    from gx1.audit import verify_market_state_remaining_mtf_numeric_v1 as residual
    from gx1.audit import verify_m5_m15_h1_trendline_oracle_v1 as line_compare
    from gx1.audit import verify_h4_trendline_lifecycle_oracle_v1 as files
    from gx1.audit.verify_d1_trendline_lifecycle_oracle_v1 import NAMES, oracle_rows
    from gx1.scripts.audit_direction_d1_trendline_replay_v1 import load_native_parts_prefix
    from gx1.scripts import audit_entry_gc_foundation_v1 as independent
    from gx1.features.volatility_squeeze_state_v1 import require_volatility_squeeze_artifact_binding
    from gx1.features.technical_indicators_v1 import wilder_rsi, wilder_atr
    from gx1.features.entry_model_native_feature_layers_v1 import model_native_m5_trendline_parameters_from_registry
    inputs, manifest, qualified = inspect_formula_inputs(formula_inputs)
    from gx1.features.htf_features import MultiTFV4DiskCache
    need(type(cache) is MultiTFV4DiskCache
         and cache.manifest_sha256 == inputs['cache_manifest']['sha256'], 'VERIFIED_FORMULA_CACHE')
    names = residual._field_contract(manifest, qualified)
    local_parameters = model_native_m5_trendline_parameters_from_registry(cache.v29_registry_constants)['local']
    local_rows = np.asarray(local_trendline['rows'])
    local_times = np.asarray(local_trendline['times_ns'])
    local_values = np.asarray(local_trendline['values'])
    need(set(local_trendline) == {'rows', 'times_ns', 'values', 'fields'}
         and local_trendline['fields'] == ['chart.' + name for name in NAMES]
         and local_rows.dtype == local_times.dtype == np.int64
         and local_rows.ndim == local_times.ndim == 1 and len(local_rows) > 0
         and local_times.shape == local_rows.shape and np.all(np.diff(local_rows) > 0)
         and np.all(np.diff(local_times) > 0) and local_rows[0] >= 0
         and local_rows[-1] <= residual.TF_SCOPE['M5']['last_row']
         and local_values.dtype == np.float32 and local_values.shape == (len(local_rows), 31)
         and np.isfinite(local_values).all(), 'LOCAL_TRENDLINE_INPUT')
    need(np.array_equal(cache['M5'].attrs['ts_int64'][local_rows], local_times), 'LOCAL_TRENDLINE_CLOCK')
    missing = {}
    for tf in TIMEFRAMES:
        c = residual.TF_SCOPE[tf]; rows = unions['mtf_' + tf.lower()]
        need(list(cache[tf].columns) == manifest['feature_names'], 'FORMULA_CACHE_FIELDS:' + tf)
        matrix = cache[tf].attrs['feats_np']; clock = cache[tf].attrs['ts_int64']
        need(matrix.dtype == np.float32 and matrix.shape == (len(clock), manifest['feature_count'])
             and not matrix.flags.writeable
             and any(np.shares_memory(matrix, backing) for backing in cache._disk_backing_arrays)
             and len(clock) == manifest['tfs'][tf]['n_bars'], 'FORMULA_CACHE_BINDING:' + tf)
        need(rows[-1] <= c['last_row'], 'UNBOUND_POST2024_FORMULA_SCOPE:' + tf)
        missing[tf] = rows[rows < c['first_row']]
    cutoffs = [int(local_times[-1]) + residual.TF_SCOPE['M5']['bar_ns']]
    cutoffs.extend(int(cache[tf].attrs['ts_int64'][rows[-1]]) + residual.TF_SCOPE[tf]['bar_ns']
                   for tf, rows in missing.items() if len(rows))
    cutoff = max(cutoffs)
    # One longest source prefix supplies both parameterizations. State replays
    # remain separate because local-96 and per-TF M5-16 are genuine differences.
    snapshots = {}
    for ref in inputs['native_parts']:
        path, stat = files.checked(ref)
        snapshots[str(path)] = stat
    source = load_native_parts_prefix(inputs['native_parts'], cutoff)
    volume = source['volume'].to_numpy()
    need(np.isfinite(volume).all() and np.all(volume > 0)
         and np.all(volume == np.floor(volume))
         and np.all(volume < np.iinfo(np.int64).max), 'SOURCE_QUOTE_COUNTS')
    m5 = {'bar_start_ns': source.index.asi8,
          **{name: source[name].to_numpy(dtype=np.float64) for name in ('open', 'high', 'low', 'close')},
          'volume': source['volume'].to_numpy(dtype=np.int64)}
    frozen = require_volatility_squeeze_artifact_binding(manifest['volatility_squeeze_artifact_set'])
    reports = {}; local_report = None
    for tf, rows in missing.items():
        if not len(rows) and tf != 'M5':
            reports[tf] = {'additional_rows': 0}; continue
        resampled = independent.native_resample_calendar_frame(m5, tf)
        times = np.asarray(cache[tf].attrs['ts_int64'][rows], dtype=np.int64)
        positions = np.searchsorted(resampled['bar_start_ns'], times)
        need(np.all(positions < len(resampled['bar_start_ns']))
             and np.array_equal(resampled['bar_start_ns'][positions], times), 'ORACLE_SOURCE_CLOCK:' + tf)
        prerequisite_rows = np.union1d(rows, local_rows).astype(np.int64) if tf == 'M5' else rows
        prerequisite_times = cache[tf].attrs['ts_int64'][prerequisite_rows]
        prerequisite_positions = np.searchsorted(resampled['bar_start_ns'], prerequisite_times)
        need(np.all(prerequisite_positions < len(resampled['bar_start_ns']))
             and np.array_equal(resampled['bar_start_ns'][prerequisite_positions], prerequisite_times),
             'ORACLE_PREREQUISITE_CLOCK:' + tf)
        end = int(prerequisite_positions[-1]) + 1
        bars = pd.DataFrame({name: values[:end] for name, values in resampled.items() if name != 'bar_start_ns'},
                            index=pd.to_datetime(resampled['bar_start_ns'][:end], utc=True))
        rsi = wilder_rsi(bars['close'], 14).to_numpy(dtype=np.float64)
        atr = wilder_atr(bars['high'], bars['low'], bars['close'], 14).to_numpy(dtype=np.float64)
        positive_atr = np.where(atr > 0., atr, np.nan)
        # ATR/RSI prerequisites retain the independent full-history 158-field
        # proof. This bridge includes every newly checked local-96 row.
        p = prerequisite_positions
        prerequisites = residual._require_prerequisites(
            bars['high'].to_numpy()[p], bars['low'].to_numpy()[p],
            bars['close'].to_numpy()[p], rsi[p], positive_atr[p],
            np.asarray(cache[tf].attrs['feats_np'][np.ix_(prerequisite_rows, [0, 1])]))
        if len(rows):
            gap_bars, gap_rsi, gap_atr = _formula_prefix(bars, positions, rsi, positive_atr)
            oracle63 = residual._oracle_matrix(gap_bars, positions, tf, frozen.require_params(tf)['fit'],
                                               names, gap_rsi, gap_atr)
            c = residual.TF_SCOPE[tf]
            oracle31, events = oracle_rows(gap_bars['high'].to_numpy(), gap_bars['low'].to_numpy(),
                gap_bars['close'].to_numpy(), gap_atr, seq_len=c['seq_len'], band_atr=c['band_atr'],
                witness_positions=positions)
            comparisons = []; offset = 0
            for part in contiguous_parts(rows):
                right = offset + len(part); observed = cache[tf].attrs['feats_np']
                a = residual._compare(oracle63[offset:right], np.asarray(observed[np.ix_(part, residual.EXPECTED_COLUMNS)]),
                                      part, times[offset:right], names)
                b = line_compare._compare(oracle31[offset:right], np.asarray(observed[part, 159:190]),
                                          part, times[offset:right], set())
                need(a['mismatched_cells'] == b['mismatched_cells'] == 0, 'HISTORY_FORMULA_MISMATCH:' + tf)
                comparisons.append({'remaining63': a, 'trendline31': b}); offset = right
            reports[tf] = {'additional_rows': len(rows), 'cache_row_first': int(rows[0]),
                'cache_row_last': int(rows[-1]), 'row_ids_sha256': axis_sha(rows),
                'native_prefix_rows': len(gap_bars), 'shared_source_prefix_rows': end, 'prerequisites': prerequisites,
                'comparisons': comparisons, 'trendline_events': events}
            del oracle63, oracle31, gap_bars, gap_rsi, gap_atr
        else:
            reports[tf] = {'additional_rows': 0}
        if tf == 'M5':
            local_positions = np.searchsorted(resampled['bar_start_ns'], local_times)
            need(np.all(local_positions < end)
                 and np.array_equal(resampled['bar_start_ns'][local_positions], local_times), 'LOCAL_TRENDLINE_SOURCE_CLOCK')
            tic = time.monotonic()
            local_bars, _, local_atr = _formula_prefix(bars, local_positions, rsi, positive_atr)
            local_oracle, local_events = oracle_rows(
                local_bars['high'].to_numpy(), local_bars['low'].to_numpy(), local_bars['close'].to_numpy(), local_atr,
                seq_len=local_parameters['seq_len'], band_atr=local_parameters['band_atr'],
                witness_positions=local_positions)
            local_comparisons = []; offset = 0
            for part in contiguous_parts(local_rows):
                right = offset + len(part)
                compared = line_compare._compare(local_oracle[offset:right], local_values[offset:right],
                                                 part, local_times[offset:right], set())
                need(compared['mismatched_cells'] == 0, 'LOCAL_TRENDLINE_FORMULA_MISMATCH')
                local_comparisons.append(compared); offset = right
            local_report = {'rows': len(local_rows), 'fields': 31, 'checked_values': len(local_rows) * 31,
                'local_field_names': local_trendline['fields'], 'parameters': local_parameters,
                'row_ids_sha256': axis_sha(local_rows), 'times_ns_sha256': axis_sha(local_times),
                'native_prefix_rows': len(local_bars), 'prerequisites': prerequisites,
                'comparisons': local_comparisons, 'trendline_events': local_events,
                'oracle_elapsed_seconds': time.monotonic() - tic,
                'independent_oracle_owner': 'gx1.audit.verify_d1_trendline_lifecycle_oracle_v1.oracle_rows',
                'all_consumed_local_trendline_rows_covered': True,
                'shared_prefix_with_mtf_gap_audit': True, 'no_reseed_or_gap_reset': True}
            del local_oracle, local_bars, local_atr
        del bars, resampled, rsi, atr, positive_atr
        gc.collect()
    for path, stat in snapshots.items():
        need(files.regular(Path(path)) == stat, 'NATIVE_PREFIX_CHANGED')
    need(local_report is not None, 'LOCAL_TRENDLINE_REPORT_MISSING')
    return {'prior_union_fields': 253, 'additional_rows': reports,
            'prior_158_full_history_reused': True, 'prior_close236_full_history_reused': True,
            'prior_31_63_may_dec2024_reused': True, 'all_consumed_mtf_rows_covered': True,
            'OHLC_resampler': 'independent.native_resample_calendar_frame',
            'no_reseed_or_gap_reset': True, 'local31': local_report, 'test_data_used': False}


def required_source_paths():
    from gx1.audit import verify_market_state_remaining_mtf_numeric_v1 as residual
    return residual.SOURCE_PATHS | frozenset({
        'gx1/audit/verify_market_state_ta_native_input_path_v1.py',
        'gx1/audit/verify_market_state_time_query_v2_zero_step_v1.py',
        'gx1/scripts/audit_dynamic_m5_october_native_postnorm_v1.py',
        'gx1/scripts/audit_entry_gc_foundation_v1.py',
        'gx1/contracts/entry_native_direction_feature_view_v1.py',
        'gx1/contracts/entry_model_native_input_normalization_v1.py',
        'gx1/contracts/entry_model_native_signal_v1.py',
        'gx1/contracts/entry_gc_mtf_history_v1.py',
        'gx1/contracts/market_state_time_query_fold1_fit_v1.py',
        'gx1/contracts/model_state_digest_v1.py',
        'gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py',
        'gx1/models/entry_v10/entry_v10_ctx_train_v3.py',
        'gx1/features/entry_model_native_feature_layers_v1.py',
        'gx1/features/entry_specialist_feature_groups_v1.py',
        'gx1/time/session_detector.py',
        'gx1/scripts/materialize_canonical_v3_augment.py',
        'gx1/execution/v12_ctx_augment_live.py',
    })


def _local_formula_aliases(local_matrix, local_times, cache, profile):
    from gx1.features.entry_model_native_feature_layers_v1 import (
        model_native_m5_local_per_bar_exact_aliases,
        model_native_m5_trendline_parameters_from_registry,
    )
    parameters = model_native_m5_trendline_parameters_from_registry(cache.v29_registry_constants)
    pairs = model_native_m5_local_per_bar_exact_aliases(trendline_parameters=parameters)
    names = list(profile['physical_field_names']['signal'])
    mtf_names = list(profile['physical_field_names']['mtf'])
    need(all(local in names and mtf in mtf_names for local, mtf in pairs), 'LOCAL_FORMULA_ALIAS_NAMES')
    clock = np.asarray(cache['M5'].attrs['ts_int64'])
    ids = np.searchsorted(clock, local_times)
    need(np.all(ids < len(clock)) and np.array_equal(clock[ids], local_times), 'LOCAL_FORMULA_ALIAS_CLOCK')
    local_cols = [names.index(local) for local, _ in pairs]
    mtf_cols = [mtf_names.index(mtf) for _, mtf in pairs]
    exact(local_matrix[:, local_cols], cache['M5'].attrs['feats_np'][np.ix_(ids, mtf_cols)],
          label='LOCAL_M5_FORMULA_ALIAS')
    mapped = {local for local, _ in pairs}
    return ids, {'rows': len(ids), 'fields': len(pairs), 'checked_values': len(ids) * len(pairs),
        'exact_alias_pairs': [list(pair) for pair in pairs],
        'signal_fields_without_this_independent_mtf_oracle_bridge': [name for name in names if name not in mapped],
        'formula_qualification_requires_mtf_coverage_of_same_rows': True}


def _local_context_formula_bridges(local_matrix, local_times, cache, profile):
    """Independently select closed TF values and check 37 local context fields."""
    from gx1.features.htf_features import (
        MODEL_NATIVE_MTF_SCALAR_PER_BAR_EXACT_ALIASES_V4,
        MODEL_NATIVE_CONTEXT_MTF_PROJECTION, MULTI_TF_SHIFT,
    )
    from gx1.time.session_detector import SESSION_BOUNDARIES
    local_times = np.asarray(local_times)
    need(local_times.dtype == np.int64 and local_times.ndim == 1 and len(local_times) > 0
         and np.all(np.diff(local_times) > 0)
         and np.all(local_times <= np.iinfo(np.int64).max - 300_000_000_000), 'LOCAL_CONTEXT_CLOCK')
    decisions = local_times + 300_000_000_000
    names = list(profile['physical_field_names']['signal'])
    need(local_matrix.dtype == np.float32 and local_matrix.shape == (len(local_times), len(names)),
         'LOCAL_CONTEXT_MATRIX')
    report = {}; required_rows = {}
    regime_map = tuple(pair for pair in MODEL_NATIVE_CONTEXT_MTF_PROJECTION
                       if pair[0] in ('ema_stack_aligned', 'trend_state_age_bars'))
    need(len(regime_map) == 2, 'LOCAL_CONTEXT_REGIME_MAP')
    def record(name, expected, *, tf=None, tolerance=None):
        local_name = 'ctx_cont.' + name
        need(local_name in names and name not in report, 'LOCAL_CONTEXT_FIELD:' + name)
        actual = local_matrix[:, names.index(local_name)]
        need(actual.shape == expected.shape and np.isfinite(actual).all()
             and np.isfinite(expected).all(), 'LOCAL_CONTEXT_AVAILABILITY:' + name)
        error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        if tolerance is None:
            exact(actual, expected, label='LOCAL_CONTEXT:' + name)
        else:
            need(np.all(error <= tolerance), 'LOCAL_CONTEXT_RSI_ROUNDING:' + name)
        report[name] = {'rows': len(actual), 'available_rows': len(actual),
            'timeframe': tf, 'maximum_absolute_error': float(error.max()),
            'bit_identical': tolerance is None,
            'maximum_allowed_rounding_error': 0. if tolerance is None else float(tolerance.max())}
    for tf in TIMEFRAMES:
        frame = cache[tf]; clock = np.asarray(frame.attrs['ts_int64'])
        ids = closed_windows(clock, decisions, duration_ns=int(MULTI_TF_SHIFT[tf].value),
            length=1, warmup=int(frame.attrs['causal_warmup_rows']))[:, 0]
        required_rows['mtf_' + tf.lower()] = np.unique(ids)
        columns = list(frame.columns); values = frame.attrs['feats_np']
        def selected(field):
            need(field in columns, 'LOCAL_CONTEXT_CACHE_FIELD:' + field)
            return np.asarray(values[ids, columns.index(field)], dtype=np.float32)
        for output, source in MODEL_NATIVE_MTF_SCALAR_PER_BAR_EXACT_ALIASES_V4[tf]:
            record(output, selected(source), tf=tf)
        if tf != 'M5':
            for output, source in regime_map:
                record(tf.lower() + '_' + output + '_v2', selected(source), tf=tf)
        # Both wires round the same raw RSI64 separately. Centered float32
        # cannot be inverted bit-exactly. Half the larger adjacent spacing
        # bounds each round-to-nearest operation (including binade edges).
        name = tf.lower() + '_rsi14_canon_v2'
        need('ctx_cont.' + name in names, 'LOCAL_CONTEXT_RSI_FIELD:' + tf)
        raw = local_matrix[:, names.index('ctx_cont.' + name)]
        centered = selected('rsi14_centered')
        need(np.isfinite(raw).all() and np.isfinite(centered).all()
             and np.all((raw >= 0.) & (raw <= 100.))
             and np.all((centered >= -1.) & (centered <= 1.)), 'LOCAL_CONTEXT_RSI_DOMAIN:' + tf)
        def half_spacing(value):
            center = value.astype(np.float64)
            upper = np.nextafter(value, np.float32(np.inf)).astype(np.float64)
            lower = np.nextafter(value, np.float32(-np.inf)).astype(np.float64)
            return np.maximum(upper - center, center - lower) * .5
        expected = centered.astype(np.float64) * 50. + 50.
        # Two binary64 arithmetic operations in the centered producer are
        # bounded conservatively by two ulps at the raw RSI domain maximum.
        tolerance = 50. * half_spacing(centered) + half_spacing(raw) + 2. * np.spacing(np.float64(100.))
        record(name, expected, tf=tf, tolerance=tolerance)
    # Integer UTC calendar independent of the pandas producer. 1970-01-01
    # was Thursday (Monday=0). Published trig operations retain float32 order.
    day_ns = 86_400_000_000_000; minute_ns = 60_000_000_000
    minute = (decisions % day_ns) // minute_ns
    hour = (minute // 60).astype(np.float32)
    weekday = ((decisions // day_ns + 3) % 7).astype(np.float32)
    record('hour_sin', np.sin(2 * np.pi * hour / 24).astype(np.float32))
    record('hour_cos', np.cos(2 * np.pi * hour / 24).astype(np.float32))
    record('dow_sin', np.sin(2 * np.pi * weekday / 7).astype(np.float32))
    need(SESSION_BOUNDARIES == {'ASIA': (22, 7), 'EU': (7, 12),
                                'OVERLAP': (12, 16), 'US': (16, 22)}, 'LOCAL_CONTEXT_SESSION_CONTRACT')
    start = np.where((minute >= 22 * 60) | (minute < 7 * 60), 22 * 60,
                     np.where(minute < 12 * 60, 7 * 60,
                              np.where(minute < 16 * 60, 12 * 60, 16 * 60)))
    record('minutes_since_session_open', ((minute - start) % (24 * 60)).astype(np.float32))
    need(len(report) == 37, 'LOCAL_CONTEXT_37_FIELDS')
    return required_rows, {'rows': len(local_times), 'fields': 37,
        'checked_values': len(local_times) * 37,
        'exact_scalar_alias_fields': 20, 'exact_regime_projection_fields': 8,
        'raw_rsi_rounding_checked_fields': 5, 'UTC_calendar_session_fields': 4,
        'decision_clock': 'M5 label + 5 minutes; closed TF availability <= decision; fixed UTC sessions',
        'times_ns_sha256': axis_sha(local_times), 'per_field': report,
        'rsi_tolerance': '50*half_max_adjacent_float32_spacing(centered)+half_max_adjacent_float32_spacing(raw)+2*ulp64(100)',
        'formula_qualification_requires_mtf_coverage_of_same_rows': True,
        'required_mtf_rows': {key: {'rows': len(rows), 'row_ids_sha256': axis_sha(rows)}
                              for key, rows in required_rows.items()}}


def audit_inputs(*, plan, receipts, view, prepared, on_input_checks_complete=None):
    """Caller must execute under the new hash-bound capped producer admission."""
    import torch
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts import entry_model_native_input_normalization_v1 as norm
    from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_SEQ_LEN
    from gx1.contracts.entry_gc_mtf_history_v1 import WINDOW_ROWS
    from gx1.features.htf_features import MULTI_TF_SHIFT
    from gx1.features.entry_specialist_feature_groups_v1 import compile_full_ta_entry_execution_profile
    need(plan['audit_scope'] == audit_scope() and view.get('full_ta_feature_profile') is True,
         'EXACT_TA_SCOPE')
    batch_rows = plan['normalization_rows_per_batch']
    need(type(batch_rows) is int and batch_rows > 0, 'NORMALIZATION_BATCH_ROWS')
    views._require_view(view, unobserved_units_profile=receipts['fold1']['profile'])
    profile = compile_full_ta_entry_execution_profile()
    need(view['physical_widths'] == profile['physical_widths']
         and view['entry_execution_profile'] == {key: profile[key] for key in ('name', 'contract_sha256')},
         'PROFILE_VIEW')
    axes = prepared['axes']; source = prepared['entry_sequence_source']; current = prepared['current']
    at = np.r_[np.arange(fold.FIT_ROWS, dtype=np.int64),
               np.arange(fold.FOLD['oof_start_row'], fold.FOLD['oof_end_row'], dtype=np.int64)]
    parents, issues = axes['parent_rows'][at], axes['decision_ns'][at]
    need(len(at) == fold.FIT_ROWS + fold.SCORE_ROWS and np.all(np.diff(parents) > 0)
         and np.all(issues[:fold.FIT_ROWS] < fold.FOLD['train_cutoff_ns'])
         and np.all(issues[fold.FIT_ROWS:] < fold.FOLD['oof_cutoff_ns'])
         and axes['decision_ns'][fold.FOLD['oof_start_row']] >= receipts['fold1']['cohort']['official_fit_cutoff_ns'],
         'FIT_SCORE_AXES')
    need(axis_sha(parents[:fold.FIT_ROWS]) == receipts['fold1']['cohort']['fit_parent_rows_sha256']
         and axis_sha(issues[:fold.FIT_ROWS]) == receipts['fold1']['cohort']['fit_decision_ns_sha256'],
         'FIT_AXIS_BINDING')
    positions = np.asarray(source['sample_positions'][at], dtype=np.int64)
    need(np.all(positions >= MODEL_NATIVE_SEQ_LEN - 1)
         and np.array_equal(source['times_ns'][positions], axes['bar_start_ns'][at])
         and np.array_equal(issues, axes['bar_start_ns'][at] + 300_000_000_000), 'LOCAL_CLOCK')
    local_rows, local_intervals = interval_union(positions, MODEL_NATIVE_SEQ_LEN)
    take_signal = list(profile['source_signal_indices']); take_ctx = list(profile['source_ctx_cont_indices'])
    # Select only consumed local rows; later unconsumed source payload is not scanned.
    local_matrix = np.ascontiguousarray(source['signal'][local_rows][:, take_signal], dtype=np.float32)
    current_ctx = np.ascontiguousarray(current['ctx_cont'][at][:, take_ctx], dtype=np.float32)
    local_m5_ids, local_alias_report = _local_formula_aliases(
        local_matrix, source['times_ns'][local_rows], prepared['cache'], profile)
    context_formula_rows, context_formula_report = _local_context_formula_bridges(
        local_matrix, source['times_ns'][local_rows], prepared['cache'], profile)
    from gx1.features.entry_model_native_feature_layers_v1 import TRENDLINE_REGISTRY_M5_LAYER_FEATURE_NAMES
    line_fields = list(TRENDLINE_REGISTRY_M5_LAYER_FEATURE_NAMES)
    signal_fields = list(profile['physical_field_names']['signal'])
    need(len(line_fields) == 31 and all(name in signal_fields for name in line_fields), 'LOCAL_TRENDLINE_FIELDS')
    local_trendline = {'rows': local_m5_ids, 'times_ns': source['times_ns'][local_rows],
        'values': np.ascontiguousarray(local_matrix[:, [signal_fields.index(name) for name in line_fields]]),
        'fields': line_fields}
    matrices = {'signal': local_matrix, 'ctx_cont': current_ctx}
    unions = {'signal': np.arange(len(local_rows), dtype=np.int64),
              'ctx_cont': np.arange(len(at), dtype=np.int64)}
    fields = {'signal': profile['physical_field_names']['signal'],
              'ctx_cont': profile['physical_field_names']['ctx_cont']}
    tf_ids, tf_intervals = {}, {}
    for tf in TIMEFRAMES:
        frame = prepared['cache'][tf]
        ids = closed_windows(np.asarray(frame.attrs['ts_int64']), issues,
            duration_ns=int(MULTI_TF_SHIFT[tf].value), length=int(WINDOW_ROWS[tf]),
            warmup=int(frame.attrs['causal_warmup_rows']))
        key = 'mtf_' + tf.lower()
        tf_ids[tf] = ids
        unions[key], tf_intervals[tf] = interval_union(ids[:, -1], int(WINDOW_ROWS[tf]))
        matrices[key] = frame.attrs['feats_np']; fields[key] = profile['physical_field_names']['mtf']
    tic = time.monotonic(); checked_values = 0; witness_raw = None
    for left in range(0, len(at), fold.BATCH_SIZE):
        right = min(left + fold.BATCH_SIZE, len(at)); selected = parents[left:right]
        got = views.gather_native_direction_feature_batch(view,
            parent_rows=axes['parent_rows'], bar_start_ns=axes['bar_start_ns'], decision_ns=axes['decision_ns'],
            entry_sequence_source=source, snap_x=current['snap'], ctx_cont=current['ctx_cont'], ctx_cat=current['ctx_cat'],
            multi_tf_sources=prepared['cache'], gc_surfaces=None, gc_layout=None, batch_parent_rows=selected,
            unobserved_units_profile=receipts['fold1']['profile'])
        need(set(got['features']) == set(profile['feature_keys'])
             and got['witness']['parent_rows'] == selected.tolist()
             and got['witness']['decision_ns'] == issues[left:right].tolist(), 'GATHER_IDENTITY')
        local_ids = positions[left:right, None] + np.arange(1 - MODEL_NATIVE_SEQ_LEN, 1)[None, :]
        local_at = np.searchsorted(local_rows, local_ids)
        need(np.array_equal(local_rows[local_at], local_ids)
             and np.all(source['times_ns'][local_ids] + 300_000_000_000 <= issues[left:right, None]),
             'LOCAL_HISTORY_MAPPING')
        expected = {'seq_x': local_matrix[local_at],
            'snap_x': np.ascontiguousarray(current['snap'][at[left:right]][:, take_signal]),
            'ctx_cont': current_ctx[left:right], 'ctx_cat': current['ctx_cat'][at[left:right]]}
        for tf in TIMEFRAMES:
            ids = tf_ids[tf][left:right]
            expected['seq_' + tf.lower()] = prepared['cache'][tf].attrs['feats_np'][ids]
            need([item['native_row_indices'] for item in got['witness']['multi_tf'][tf]] == ids.tolist(),
                 'GATHER_ROW_WITNESS:' + tf)
        for key in expected:
            exact(got['features'][key], expected[key], label=key)
            checked_values += int(expected[key].size)
        if left == 0:
            witness_at = np.searchsorted(selected, np.asarray(witness_owner.PARENT_ROWS, dtype=np.int64))
            need(np.array_equal(selected[witness_at], witness_owner.PARENT_ROWS)
                 and np.array_equal(issues[left:right][witness_at], witness_owner.ISSUE_NS), 'FIXED_WITNESSES')
            witness_raw = {key: values[witness_at].copy() for key, values in got['features'].items()}
    gather_seconds = time.monotonic() - tic
    need(witness_raw is not None and not torch.cuda.is_initialized(), 'CPU_WITNESS')
    torch.set_num_threads(1); torch.manual_seed(witness_owner.SEED)
    with norm_evidence.forbid_model_work(torch) as construction_counts:
        model = views.build_native_direction_research_model(view,
            prefix_normalization=receipts['fold1']['normal'], constructor_metadata=receipts['constructor'],
            device=torch.device('cpu'), unobserved_units_profile=receipts['fold1']['profile'])
    need(all(value == 0 for value in construction_counts.values()), 'CONSTRUCTOR_NO_FORWARD')
    normalized = norm.project_ta_input_normalization_contract(
        receipts['fold1']['normal']['base_normalization'], unobserved_units_profile=receipts['fold1']['profile'])
    need(model.backbone._input_normalization_contract == normalized
         and normalized['contract_sha256'] == view['prefix_normalization']['ta_base_normalization_sha256'],
         'RUNTIME_NORMALIZATION_BINDING')
    normalized_report = _normalization_measure(model, normalized, matrices=matrices,
        unions=unions, fields=fields, batch_rows=batch_rows)
    family_report = _family_witness(model, witness_raw, normalized, profile)
    del model, local_matrix, current_ctx
    gc.collect()
    measurement = {'schema_version': SCHEMA, 'candidate_view_sha256': view['contract_sha256'],
        'profile': view['entry_execution_profile'], 'scope': audit_scope(),
        'gather': {'parent_rows': len(at), 'fit_rows': fold.FIT_ROWS, 'score_rows': fold.SCORE_ROWS,
            'parent_rows_sha256': axis_sha(parents), 'decision_ns_sha256': axis_sha(issues),
            'checked_values': checked_values, 'elapsed_seconds': gather_seconds,
            'local_source_row_intervals': local_intervals, 'mtf_cache_row_intervals': tf_intervals,
            'all_fields_all_window_positions_bit_identical': True},
        'normalization': normalized_report, 'family_consumption': family_report,
        'local_formula_alias_bridge': local_alias_report,
        'local_context_formula_evidence': context_formula_report,
        'limits_of_evidence': ['The 61 macro fields retain separate prior formula/asof evidence; this audit does not independently rederive every macro formula.',
            'COT 2024 release/vintage provenance is UNKNOWN: as_of + 7 calendar days is a proxy without per-report historical release/vintage qualification. No 2024 leakage has been demonstrated; that does not qualify publication timing.',
            'The single genuine forward proves routed pre-projection values, not predictive utility or per-field influence.',
            'No TRAIN fit, OOF prediction quality, serving parity, TEST or trading evidence.'],
        'normalization_fits': 0, 'model_constructions': 1, 'model_forwards': 1,
        'backward_calls': 0, 'optimizer_steps': 0, 'targets_used_as_inputs': False,
        'test_data_used': False, 'predictive_value_claimed': False}
    if on_input_checks_complete is not None:
        on_input_checks_complete(measurement)
    formula_unions = dict(unions)
    formula_unions['mtf_m5'] = np.union1d(unions['mtf_m5'], local_m5_ids).astype(np.int64)
    for key, rows in context_formula_rows.items():
        formula_unions[key] = np.union1d(formula_unions[key], rows).astype(np.int64)
    formula_report = _history_formula_measure(plan['formula_inputs'], prepared['cache'], formula_unions,
        local_trendline=local_trendline)
    local31_report = formula_report.pop('local31')
    measurement.update({'mtf_formula_evidence': formula_report,
                        'local_trendline_formula_evidence': local31_report})
    return measurement


# One newly admitted producer invocation owns this audit and separate TRAIN-only
# endpoint priors. Historical plans remain immutable inputs, never launch keys.
STAGE = fold.TA_INPUT_STAGE
REPO = fold.CANONICAL_REPO
PLAN_PATH = REPO / 'configs/research' / (STAGE.upper() + '.json')
OUT_ROOT = fold.TA_INPUT_ROOT
POLICY_PATH = REPO / 'NEXT_RUN_POLICY.json'
PLAN_SCHEMA = 'gx1_market_state_ta_native_input_plan_v1'
WALL_SECONDS = 3600
LIMITS = {'memory_max': 12 * 1024**3, 'memory_high': 12 * 1024**3,
          'swap': 512 * 1024**2, 'pids': 64, 'numerical_threads': 1,
          'wall_seconds': WALL_SECONDS}
EXECUTION = {'class': 'producer', 'memory': '12G', 'swap': '512M',
             'runner': 'scripts/gx1_capped_run.sh',
             'module': 'gx1.audit.verify_market_state_ta_native_input_path_v1',
             'timeout_seconds': WALL_SECONDS, 'kill_after_seconds': 10,
             'device': 'cpu', 'cuda_producer': False}


def target_input_refs():
    from gx1.contracts import market_state_five_endpoint_target_binding_v1 as state
    from gx1.contracts import market_state_future_displacement_candidate_v1 as future
    return {'S_endpoint_refs': {name: {'path': str(path), 'sha256': digest}
                                for name, (path, digest) in state.REFS.items()},
            'F_endpoint_refs': {name: {'path': str(path), 'sha256': digest}
                                for name, (path, digest) in future.REFS.items()}}


def producer_source_paths():
    from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero
    return required_source_paths() | zero.TA_SOURCE_PATHS | frozenset({
        'gx1/scripts/run_market_state_time_query_fold1_fit_window_v1.py',
        'gx1/contracts/market_state_time_query_native_F_objective_v1.py',
        'gx1/contracts/market_state_time_query_scorer_v2.py',
        'gx1/contracts/market_state_five_endpoint_prior_v1.py',
        'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
        'gx1/contracts/market_state_five_endpoint_contract_v1.py',
        'gx1/contracts/market_state_future_displacement_candidate_v1.py',
        'gx1/contracts/market_state_time_query_side_normalized_loss_v1.py',
    })


def build_plan(*, normalization_rows_per_batch):
    # Pure preparation: hashes small bound files and current sources, writes
    # nothing, and does not import/read feature payloads or instantiate a model.
    import subprocess
    from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero
    from gx1.audit import verify_market_state_remaining_mtf_numeric_v1 as residual
    need(type(normalization_rows_per_batch) is int and normalization_rows_per_batch > 0,
         'NORMALIZATION_BATCH_ROWS')
    def ref(path):
        return {'path': str(path), 'sha256': zero._sha(path)}
    return {'schema_version': PLAN_SCHEMA, 'stage': STAGE,
        'status': 'PREPARED_NOT_ADMITTED',
        'source_base_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'source_owners': {path: zero._sha(REPO / path) for path in sorted(producer_source_paths())},
        'inputs': {'candidate_view': ref(OUT_ROOT / 'CANDIDATE_VIEW.json'), **target_input_refs()},
        'audit_scope': audit_scope(), 'normalization_rows_per_batch': normalization_rows_per_batch,
        'formula_inputs': {'remaining_plan': ref(residual.PLAN_PATH),
            'remaining_result': ref(residual.OUT_ROOT / 'RESULT.json'),
            'remaining_terminal': ref(residual.OUT_ROOT / 'TERMINAL.json')},
        'limits': dict(LIMITS), 'execution': dict(EXECUTION),
        'output_root': str(OUT_ROOT), 'max_json_bytes': zero.MAX_JSON_BYTES}


def _policy_gate(policy, plan_ref, worker_ref):
    gate = policy.get(STAGE, {})
    need(policy.get('training_enabled') is False
         and policy.get('native_training_enabled') is False
         and policy.get('native_launch_authorized') is False
         and policy.get('full_val_allowed') is False
         and gate.get('status') == 'ADMITTED'
         and gate.get('plan') == plan_ref and gate.get('worker') == worker_ref
         and gate.get('execution_authorized') is True
         and gate.get('data_reads_authorized') is True
         and gate.get('endpoint_prior_fit_authorized') is True
         and gate.get('model_constructions_authorized') == 1
         and gate.get('model_forwards_authorized') == 1
         and gate.get('backward_calls_authorized') == 0
         and gate.get('optimizer_steps_authorized') == 0
         and gate.get('normalization_fits_authorized') == 0
         and gate.get('training_authorized') is False
         and gate.get('test_data_used') is False
         and gate.get('relaunch_allowed') is False
         and gate.get('max_invocations') == 1
         and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
         and gate.get('output_root') == str(OUT_ROOT), 'EXACT_PRODUCER_POLICY')


def inspect(plan_ref, *, require_policy):
    import subprocess
    from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero
    from gx1.audit import verify_h4_trendline_lifecycle_oracle_v1 as files
    head = zero._clean_head()
    need(plan_ref['path'] == str(PLAN_PATH), 'PRODUCER_PLAN_PATH')
    plan = zero._read_ref(plan_ref)
    need(set(plan) == {'schema_version', 'stage', 'status', 'source_base_head',
         'source_owners', 'inputs', 'audit_scope', 'normalization_rows_per_batch',
         'formula_inputs', 'limits', 'execution', 'output_root', 'max_json_bytes'}
         and plan['schema_version'] == PLAN_SCHEMA and plan['stage'] == STAGE
         and plan['status'] == 'PREPARED_NOT_ADMITTED'
         and plan['audit_scope'] == audit_scope() and plan['limits'] == LIMITS
         and plan['execution'] == EXECUTION and plan['output_root'] == str(OUT_ROOT)
         and plan['max_json_bytes'] == zero.MAX_JSON_BYTES
         and type(plan['normalization_rows_per_batch']) is int
         and plan['normalization_rows_per_batch'] > 0
         and subprocess.run(['git', 'merge-base', '--is-ancestor', plan['source_base_head'], head],
             cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0,
         'PRODUCER_PLAN_SCOPE')
    owners = plan['source_owners']
    need(isinstance(owners, dict) and producer_source_paths() <= set(owners), 'PRODUCER_SOURCE_SET')
    for relative, digest in owners.items():
        path = REPO / relative
        need(not Path(relative).is_absolute() and '..' not in Path(relative).parts
             and path.is_file() and not path.is_symlink()
             and zero._sha(path) == digest, 'PRODUCER_SOURCE:' + relative)
    need(OUT_ROOT.is_dir() and OUT_ROOT.resolve(strict=True) == OUT_ROOT
         and {p.name for p in OUT_ROOT.iterdir()} == {'CANDIDATE_VIEW.json'}, 'UNCONSUMED_OUTPUT_ROOT')
    candidate_ref = plan['inputs'].get('candidate_view')
    need(set(plan['inputs']) == {'candidate_view', *target_input_refs()}
         and candidate_ref['path'] == str(OUT_ROOT / 'CANDIDATE_VIEW.json')
         and all(plan['inputs'][key] == value for key, value in target_input_refs().items()),
         'PRODUCER_INPUT_REFS')
    candidate = zero._read_ref(candidate_ref)
    receipts = zero.load_ta_input_receipts(current_source_owners=owners)
    view = zero._bind_ta_view(receipts, max_batch_rows=fold.BATCH_SIZE)
    need(candidate == view and view['full_ta_feature_profile'] is True, 'EXACT_CANDIDATE_VIEW')
    fold._require_ta_checkpoint_view({
        'contract_sha256': view['contract_sha256'],
        'prefix_normalization': view['prefix_normalization'],
        'research_objective_sha256': view['research_objective']['contract_sha256']}, verify_files=True)
    inspect_formula_inputs(plan['formula_inputs'])
    # Target bytes are loaded only inside the admitted producer. This checks
    # exact immutable references and small completion receipts during preflight.
    for group in target_input_refs().values():
        for item in group.values():
            files.regular(Path(item['path']))
            if Path(item['path']).suffix == '.json':
                files.bound_json(item)
    policy_ref = {'path': str(POLICY_PATH), 'sha256': zero._sha(POLICY_PATH)}
    if require_policy:
        _policy_gate(zero._read_ref(policy_ref, limit=16 * 1024**2), plan_ref,
                     {'path': str(Path(__file__).resolve()), 'sha256': zero._sha(Path(__file__))})
    return plan, policy_ref, head, receipts, view


def _provider_integrity_metadata(provider, provider_plan):
    "Use the bound provider's own metadata/immutable-reuse contract."
    info = provider.validate_metadata(provider_plan)
    return {'metadata_sha_files': len(info['metadata']),
        'payload_stat_files': len(provider_plan['payloads']),
        'readonly_reused_decode_files': len(info['reuse']['scratch_files']),
        'verification_owner': 'provider264.validate_metadata',
        'local_decode_scope': 'completed historical decode provenance plus exact dev/inode/mode/size/mtime/ctime; no fresh mmap body SHA',
        'metadata_or_stat_checks_are_not_fresh_payload_hashes': True}


def preflight(plan_ref):
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as mature
    plan, _, head, receipts, view = inspect(plan_ref, require_policy=False)
    provider = mature._provider_module()
    integrity = _provider_integrity_metadata(provider, receipts['provider_chain']['provider_plan'])
    return {'status': 'READY_NO_CLAIM_NO_FEATURE_PAYLOAD_NO_MODEL',
            'plan': plan_ref, 'source_head': head, 'source_owners_checked': len(plan['source_owners']),
            'candidate_view': plan['inputs']['candidate_view'],
            'candidate_view_sha256': view['contract_sha256'], 'audit_scope': audit_scope(),
            'historical_lineage': receipts['historical_lineage'],
            'provider_integrity_metadata': integrity,
            'endpoint_priors_fit_partition': 'mature_fold1_fit_only',
            'feature_payloads_read': False, 'model_constructions': 0,
            'model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False}


def run(plan_ref):
    import json
    import os
    import resource
    import signal
    from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as mature
    from gx1.contracts import market_state_five_endpoint_target_binding_v1 as state
    from gx1.contracts import market_state_future_displacement_candidate_v1 as future
    from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
    from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner
    from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_research_producer_execution
    cap = require_capped_cpu_research_producer_execution()
    need(all(cap[key] == LIMITS[key] for key in ('memory_max', 'memory_high', 'swap', 'pids', 'numerical_threads')),
         'PRODUCER_CAPS')
    began = time.monotonic()
    plan, policy_ref, head, receipts, view = inspect(plan_ref, require_policy=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    need(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'AMBIENT_TIMER')
    published_bytes = 0
    def publish(name, value):
        nonlocal published_bytes
        size = len((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())
        need(name in ('FAILED.json', 'TERMINAL.json')
             or published_bytes + size <= plan['max_json_bytes'], 'PRODUCER_JSON_BUDGET')
        item = zero._write(OUT_ROOT, name, value, limit=plan['max_json_bytes'])
        published_bytes += size
        return item
    provider = provider_plan = prepared = None
    input_phase_ref = None
    from gx1.audit.verify_market_state_remaining_mtf_numeric_v1 import TERMINAL_RESERVE_SECONDS
    remaining = WALL_SECONDS - TERMINAL_RESERVE_SECONDS - (time.monotonic() - began)
    need(remaining > 0., 'PRECLAIM_WALL_BUDGET')
    claim = publish('CLAIM.json', {'schema_version': SCHEMA + '_claim',
        'plan': plan_ref, 'policy': policy_ref, 'source_head': head, 'cap': cap,
        'candidate_view': plan['inputs']['candidate_view'], 'audit_scope': audit_scope(),
        'endpoint_prior_fit_rows': fold.FIT_ROWS, 'test_data_used': False})
    def source_bookend():
        need(zero._clean_head() == head and zero._sha(PLAN_PATH) == plan_ref['sha256']
             and zero._sha(POLICY_PATH) == policy_ref['sha256']
             and all(zero._sha(REPO / path) == digest for path, digest in plan['source_owners'].items())
             and zero._sha(Path(plan['inputs']['candidate_view']['path']))
                 == plan['inputs']['candidate_view']['sha256'], 'PRODUCER_SOURCE_BOOKEND')

    def publish_input_phase(measurement):
        nonlocal input_phase_ref
        need(input_phase_ref is None, 'DUPLICATE_INPUT_PHASE')
        integrity = _provider_integrity_metadata(provider, provider_plan)
        source_bookend()
        # Keep the completed numerical measurements without repeating the
        # large per-field extrema arrays in the eventual final RESULT.
        normalization = {surface: {key: values[key] for key in (
            'unique_rows', 'row_ids_sha256', 'checked_values',
            'maximum_absolute_error', 'field_names')}
            for surface, values in measurement['normalization'].items()}
        input_phase_ref = publish('INPUT_TENSOR_PHASE.json', {
            'schema_version': SCHEMA + '_input_tensor_phase',
            'status': 'COMPLETE_INPUT_TENSOR_PHASE_ONLY',
            'plan': plan_ref, 'claim': claim, 'policy': policy_ref,
            'source_head': head, 'source_owners': plan['source_owners'],
            'candidate_view': plan['inputs']['candidate_view'],
            'candidate_view_sha256': view['contract_sha256'],
            'audit_scope': audit_scope(), 'provider_metadata_bookend': integrity,
            'gather': measurement['gather'], 'normalization': normalization,
            'family_consumption': measurement['family_consumption'],
            'local_formula_alias_bridge': measurement['local_formula_alias_bridge'],
            'local_context_formula_evidence': measurement['local_context_formula_evidence'],
            'limits_of_evidence': measurement['limits_of_evidence'],
            'independent_formula_qualification_complete': False,
            'full_input_audit_pass': False,
            'model_constructions': 1, 'model_forwards': 1, 'backward_calls': 0,
            'optimizer_steps': 0, 'normalization_fits': 0,
            'targets_used_as_inputs': False, 'test_data_used': False,
            'diagnostic_only_no_resume_or_relaunch_authority': True})

    old_alarm = signal.getsignal(signal.SIGALRM)
    def expired(_signum, _frame):
        raise TimeoutError('TA_NATIVE_INPUT_3600_SECOND_SCOPE')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        setup = time.monotonic()
        # _prepared_fold1 validates provider metadata before opening inputs;
        # its provider uses independent ref/stat owners, not campaign's memo.
        provider, provider_plan, prepared, fit_rows, issues = runner._prepared_fold1(receipts, view, np=np)
        setup_seconds = time.monotonic() - setup
        measurement = audit_inputs(plan=plan, receipts=receipts, view=view, prepared=prepared,
                                   on_input_checks_complete=publish_input_phase)
        need(input_phase_ref is not None, 'INPUT_PHASE_NOT_PUBLISHED')
        # This separate reference fit never enters model.forward and has no
        # access to score outcomes through its exact fit-only gather call.
        state_bound = state.load_bound_five_endpoint_targets()
        future_bound = future.load_bound_future_displacement()
        priors = objective.fit_endpoint_priors(state_bound, future_bound,
            axes={'fit_parent_rows': fit_rows, 'fit_decision_ns': issues}, fold_id=1, ta_only=True)
        need(priors['fit_rows'] == fold.FIT_ROWS
             and priors['fit_parent_rows_sha256'] == axis_sha(fit_rows)
             and priors['fit_decision_ns_sha256'] == axis_sha(issues), 'ENDPOINT_PRIOR_FIT_AXIS')
        integrity = _provider_integrity_metadata(provider, provider_plan)
        mature._release_pages(provider, prepared)
        prepared = None
        source_bookend()
        priors_ref = publish('ENDPOINT_PRIORS.json', priors)
        result = publish('RESULT.json', {'schema_version': SCHEMA + '_result',
            'status': 'COMPLETE', 'plan': plan_ref, 'claim': claim, 'policy': policy_ref,
            'source_head': head, 'source_owners': plan['source_owners'],
            'candidate_view': plan['inputs']['candidate_view'], 'endpoint_priors': priors_ref,
            'input_audit': measurement, 'cap': cap, 'input_setup_seconds': setup_seconds,
            'limits_of_evidence': measurement['limits_of_evidence'],
            'input_tensor_phase': input_phase_ref,
            'input_integrity': {'provider_metadata_bookend': integrity,
                'provider_metadata_validated_before_prepare_and_after_measurement': True,
                'axes_archive_and_native_MTF_payloads_fresh_SHA_verified_by_prepare': True,
                'native_MTF_verification_owner': 'htf_features.load_multi_tf_v4_cache:descriptor_SHA_then_readonly_mmap',
                'reused_local_decode_fresh_body_hash_claimed': False},
            'elapsed_seconds': time.monotonic() - began,
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'model_constructions': 1, 'model_forwards': 1, 'backward_calls': 0,
            'optimizer_steps': 0, 'normalization_fits': 0, 'endpoint_prior_fit_rows': fold.FIT_ROWS,
            'source_unchanged': True, 'test_data_used': False, 'predictive_value_claimed': False})
        terminal = publish('TERMINAL.json', {'schema_version': SCHEMA + '_terminal',
            'status': 'COMPLETE', 'exit_code': 0, 'plan': plan_ref, 'claim': claim,
            'result': result, 'endpoint_priors': priors_ref,
            'input_tensor_phase': input_phase_ref,
            'no_relaunch_under_this_plan': True, 'test_data_used': False})
        return {'result': result, 'terminal': terminal, 'endpoint_priors': priors_ref}
    except BaseException as exc:
        failed = publish('FAILED.json', {'schema_version': SCHEMA + '_failure',
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'completed_input_tensor_phase': input_phase_ref,
            'elapsed_seconds': time.monotonic() - began, 'test_data_used': False})
        publish('TERMINAL.json', {'schema_version': SCHEMA + '_terminal',
            'status': 'FAILED_CONSUMED', 'exit_code': 1, 'plan': plan_ref,
            'claim': claim, 'failure': failed, 'no_relaunch_under_this_plan': True,
            'completed_input_tensor_phase': input_phase_ref,
            'model_forwards_upper_bound': 1, 'optimizer_steps': 0, 'test_data_used': False})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, old_alarm)
        if prepared is not None:
            mature._release_pages(provider, prepared)


def main():
    import argparse
    import json
    import sys
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    plan_ref = {'path': args.plan, 'sha256': args.plan_sha256}
    print(json.dumps(preflight(plan_ref) if args.preflight else run(plan_ref), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
