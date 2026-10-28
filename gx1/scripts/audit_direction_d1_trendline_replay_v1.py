"""Replay real D1 trendline cache from its causal M5 source; never fit or score a model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import signal
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path('/home/andre2/src/GX1_CURRENT')
ROOT = Path('/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63')
STAGE = 'independent_direction_d1_trendline_replay_20261028_001'
LOCALIZE_STAGE = 'independent_direction_d1_mismatch_localization_20261028_001'
H4_STAGE = 'independent_direction_h4_trendline_replay_20261028_001'
CUTOFF_NS = 1751328000000000000  # 2025-07-01 UTC, reused development only.
SOURCE_PATHS = (
    'gx1/scripts/audit_direction_d1_trendline_replay_v1.py',
    'gx1/features/htf_features.py',
    'gx1/features/entry_specialist_feature_groups_v1.py',
    'gx1/features/trendline_registry_v1.py',
    'gx1/features/technical_indicators_v1.py',
    'gx1/scripts/prebuild_multi_tf_cache_v4.py',
    'gx1/scripts/audit_entry_direction_candle_midpoints_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/entry_model_native_bundle_commit_v1.py',
    'gx1/contracts/entry_gc_mtf_history_v1.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_direction_d1_trendline_replay_v1.py',
)


def require(ok, reason):
    if not bool(ok):
        raise RuntimeError('DIRECTION_D1_TRENDLINE_REPLAY_' + reason)


def compare_fields(recomputed, cached, names, timestamps):
    """Compare float32 model values exactly, preserving NaN and signed-zero semantics."""
    a = np.ascontiguousarray(recomputed, dtype=np.float32)
    b = np.ascontiguousarray(cached, dtype=np.float32)
    require(a.ndim == 2 and a.shape == b.shape and a.shape[1] == len(names)
        and timestamps.shape == (len(a),) and not np.isinf(a).any() and not np.isinf(b).any(), 'COMPARE_SHAPE')
    mismatch = (np.isnan(a) != np.isnan(b)) | (~np.isnan(a) & ~np.isnan(b) & (a.view(np.uint32) != b.view(np.uint32)))
    rows = []
    for i, name in enumerate(names):
        at = np.flatnonzero(mismatch[:, i]); finite = np.isfinite(a[:, i]) & np.isfinite(b[:, i])
        error = float(np.max(np.abs(a[finite, i].astype(np.float64) - b[finite, i].astype(np.float64)))) if finite.any() else None
        rows.append({'field': name, 'mismatched_rows': int(len(at)), 'both_finite_rows': int(finite.sum()),
            'first_mismatch_timestamp_ns': int(timestamps[at[0]]) if len(at) else None, 'max_abs_error': error})
    return {'rows': len(a), 'fields': len(names), 'mismatched_cells': int(mismatch.sum()),
        'all_finite_float32_bits_and_missingness_equal': not bool(mismatch.any()), 'per_field': rows}


def load_m5_prefix(path, cutoff_ns):
    """Read only row groups whose minimum precedes cutoff, then drop later rows."""
    cutoff = datetime.fromtimestamp(cutoff_ns / 1e9, tz=timezone.utc)
    with path.open('rb') as handle:
        parquet = pq.ParquetFile(handle)
        column = parquet.schema.names.index('time')
        groups = []
        for i in range(parquet.num_row_groups):
            stats = parquet.metadata.row_group(i).column(column).statistics
            require(stats is not None and stats.has_min_max and stats.min.tzinfo is not None
                and stats.max.tzinfo is not None and stats.min <= stats.max, 'ROW_GROUP_TIME_STATS')
            if stats.min < cutoff:
                groups.append(i)
        require(groups and groups == list(range(len(groups))), 'PREFIX_ROW_GROUPS')
        frame = parquet.read_row_groups(groups, columns=['time', 'open', 'high', 'low', 'close', 'volume'],
            use_threads=False).to_pandas(use_threads=False)
    frame = frame.loc[frame['time'] < pd.Timestamp(cutoff)].copy()
    index = pd.DatetimeIndex(frame.pop('time')).as_unit('ns')
    require(len(index) > 0 and index.tz is not None and index.is_unique and index.is_monotonic_increasing
        and index[-1].value < cutoff_ns and index[0].value >= 0, 'SOURCE_CLOCK')
    frame.index = index
    require(np.isfinite(frame[['open', 'high', 'low', 'close']].to_numpy()).all(), 'SOURCE_OHLC')
    return frame, {'selected_row_groups': len(groups), 'm5_rows_before_cutoff': len(frame),
        'first_m5_ns': int(index[0].value), 'last_m5_ns': int(index[-1].value)}


def mismatch_matrix(recomputed, cached):
    a = np.ascontiguousarray(recomputed, dtype=np.float32)
    b = np.ascontiguousarray(cached, dtype=np.float32)
    require(a.ndim == 2 and a.shape == b.shape and not np.isinf(a).any() and not np.isinf(b).any(), 'LOCALIZE_SHAPE')
    missing = np.isnan(a) != np.isnan(b)
    finite = ~np.isnan(a) & ~np.isnan(b) & (a.view(np.uint32) != b.view(np.uint32))
    return missing, finite


def window_exposure(timestamps, mismatched_rows, decisions, length, *, shift_ns=86400 * 10**9, timeframe='D1'):
    timestamps = np.asarray(timestamps); bad = np.asarray(mismatched_rows); decisions = np.asarray(decisions)
    require(timestamps.dtype == decisions.dtype == np.int64 and bad.dtype == np.bool_
        and timestamps.ndim == bad.ndim == decisions.ndim == 1 and len(timestamps) == len(bad)
        and len(decisions) > 0 and np.all(np.diff(timestamps) > 0) and np.all(np.diff(decisions) > 0)
        and type(length) is int and length > 0 and type(shift_ns) is int and shift_ns > 0
        and timeframe in ('D1', 'H4', 'H1', 'M15', 'M5'), 'WINDOW_DOMAIN')
    right = np.searchsorted(timestamps, decisions - shift_ns, side='right')
    require(np.all(right >= length) and np.all(right <= len(timestamps)), 'WINDOW_BOUNDS')
    left = right - length
    cumulative = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(bad, dtype=np.int64)))
    impacted = cumulative[right] - cumulative[left] > 0
    coverage = np.zeros(len(timestamps) + 1, dtype=np.int64)
    np.add.at(coverage, left, 1); np.add.at(coverage, right, -1)
    used_bad = np.flatnonzero((np.cumsum(coverage[:-1]) > 0) & bad)
    return {'decisions': len(decisions), f'decisions_with_any_mismatched_{timeframe.lower()}_row': int(impacted.sum()),
        f'distinct_mismatched_{timeframe.lower()}_rows_in_windows': len(used_bad),
        'first_affected_decision_ns': int(decisions[np.flatnonzero(impacted)[0]]) if impacted.any() else None,
        'last_affected_decision_ns': int(decisions[np.flatnonzero(impacted)[-1]]) if impacted.any() else None,
        'earliest_window_row_ns': int(timestamps[left.min()]), 'latest_window_row_ns': int(timestamps[right.max()-1])}


def price_precision_summary(frame):
    names = ('open', 'high', 'low', 'close')
    raw = frame.loc[:, list(names)].to_numpy(dtype=np.float64)
    require(raw.ndim == 2 and raw.shape[1] == 4 and len(raw) > 0 and np.isfinite(raw).all(), 'PRICE_DOMAIN')
    rounded = raw.astype(np.float32).astype(np.float64)
    delta = np.abs(raw - rounded)
    raw_close = np.unique(raw[:, 3]); rounded_close = np.unique(rounded[:, 3])
    original_direction = np.sign(raw[:, 3] - raw[:, 0]); rounded_direction = np.sign(rounded[:, 3] - rounded[:, 0])
    require(not np.any(original_direction * rounded_direction < 0), 'FLOAT32_ORDER')
    return {'m5_rows':len(raw), 'price_cells':int(raw.size), 'price_cells_changed':int(np.count_nonzero(delta)),
        'median_absolute_error_usd':float(np.median(delta)), 'p99_absolute_error_usd':float(np.quantile(delta, 0.99)),
        'max_absolute_error_usd':float(delta.max()), 'distinct_original_closes':len(raw_close),
        'distinct_float32_closes':len(rounded_close), 'distinct_original_closes_collapsed':len(raw_close)-len(rounded_close),
        'm5_close_open_direction_changed':int(np.count_nonzero(original_direction != rounded_direction))}


def main(argv=None):
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('plan', 'plan-sha256', 'policy-sha256', 'source-head'):
        parser.add_argument('--' + option, required=True)
    parser.add_argument('--localize-windows', action='store_true')
    parser.add_argument('--timeframe', choices=('D1', 'H4', 'H1', 'M15', 'M5'), default='D1')
    parser.add_argument('--builder-float32', action='store_true')
    args = parser.parse_args(argv)
    other_tf = args.timeframe != 'D1'; builder = args.builder_float32
    require(not (other_tf and args.localize_windows) and (not builder or args.timeframe == 'H1'), 'MODE')
    localize = args.localize_windows or other_tf
    tf = args.timeframe
    shift_ns = {'D1':86400, 'H4':14400, 'H1':3600, 'M15':900, 'M5':300}[tf] * 10**9
    stage_name = ('independent_direction_' + tf.lower() + '_trendline_replay_20261028_001') if other_tf else LOCALIZE_STAGE if localize else STAGE
    schema = ('gx1_direction_' + tf.lower() + '_trendline_replay') if other_tf else 'gx1_direction_d1_mismatch_localization' if localize else 'gx1_direction_d1_trendline_replay'
    design_name = ('DIRECTION_' + tf + '_TRENDLINE_REPLAY_20261028.json') if other_tf else 'DIRECTION_D1_MISMATCH_LOCALIZATION_20261028.json' if localize else 'DIRECTION_D1_TRENDLINE_REPLAY_20261028.json'
    output_name = ('DIRECTION_' + tf + '_TRENDLINE_REPLAY_20261028_001') if other_tf else 'DIRECTION_D1_MISMATCH_LOCALIZATION_20261028_001' if localize else 'DIRECTION_D1_TRENDLINE_REPLAY_20261028_001'
    if builder:
        stage_name = 'independent_direction_h1_builder_precision_replay_20261028_001'
        schema = 'gx1_direction_h1_builder_precision_replay'
        design_name = 'DIRECTION_H1_BUILDER_PRECISION_REPLAY_20261028.json'
        output_name = 'DIRECTION_H1_BUILDER_PRECISION_REPLAY_20261028_001'
    feat_key, ts_key = ('tf_feats', 'tf_ts') if other_tf else ('d1_feats', 'd1_ts')
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import immutable_event_authority_v1 as publication
    from gx1.contracts import local_random_access_campaign_v2 as binding
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    from gx1.contracts.entry_gc_mtf_history_v1 import WINDOW_ROWS
    from gx1.features.entry_specialist_feature_groups_v1 import compile_gc_mtf_geometry_entry_execution_profile
    from gx1.features.htf_features import _resample_ohlcv, multi_tf_resample, build_multi_tf_v4_closed_timestamp_indices
    from gx1.features.technical_indicators_v1 import ema50_200_spread_atr_block, technical_indicator_contract_metadata
    from gx1.features.trendline_registry_v1 import TRENDLINE_REGISTRY_FEATURE_NAMES_V1, compute_trendline_registry_features_v1
    from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
    require(not any(name.split('.')[0] in ('torch', 'sklearn', 'joblib') for name in sys.modules), 'PRELOADED_MODEL')
    initial = candle.git_state(); require(initial == (args.source_head, 'work/gx1-current', ''), 'INITIAL_SOURCE')
    plan_ref = {'path': str(candle.regular(args.plan)), 'sha256': args.plan_sha256}
    policy_ref = {'path': str(REPO / 'NEXT_RUN_POLICY.json'), 'sha256': args.policy_sha256}
    with binding.fresh_file_verification_scope():
        plan = binding.read_bound_json(Path(plan_ref['path']), plan_ref['sha256'])
        require(set(plan) == {'schema_version', 'stage', 'design', 'inputs', 'sources', 'output_dir',
            'wall_seconds', 'cutoff_ns', 'test_data_used'} and plan['schema_version'] == schema + '_plan_v1'
            and plan['stage'] == stage_name and plan['wall_seconds'] == 1200 and plan['cutoff_ns'] == CUTOFF_NS
            and plan['test_data_used'] is False, 'PLAN')
        require(plan['design']['path'] == str(REPO / 'docs' / design_name), 'DESIGN_PATH')
        design = binding.read_bound_json(Path(plan['design']['path']), plan['design']['sha256'])
        require(design['schema_version'] == schema + '_design_v1'
            and design['inputs'] == plan['inputs'] and design['test_data_used'] is False, 'DESIGN')
        policy = binding.read_bound_json(Path(policy_ref['path']), policy_ref['sha256']); gate = policy[stage_name]
        require(gate['status'] == 'ADMITTED' and gate['plan'] == plan_ref and gate['execution_authorized'] is True
            and gate['max_invocations'] == 1 and gate['claimed_invocations'] == gate['completed_invocations'] == 0
            and gate['relaunch_allowed'] is False, 'ONE_SHOT_POLICY')
        require(all(policy[k] is False for k in ('training_enabled', 'native_training_enabled',
            'full_epoch_training_allowed', 'full_val_allowed', 'native_launch_authorized'))
            and all(gate[k] is False for k in ('model_forwards_authorized', 'optimizer_steps_authorized',
                'checkpoint_reads_authorized', 'test_data_used', 'broker_access_allowed')), 'CLOSED_GATES')
        require(set(plan['sources']) == set(SOURCE_PATHS)
            and all(ref['path'] == str(REPO / name) for name, ref in plan['sources'].items()), 'SOURCE_SET')
        for ref in plan['sources'].values():
            candle.regular(ref['path']); binding.require_binding(ref, label='source')
        keys = {'source_m5', 'cache_manifest', feat_key, ts_key, 'prefix_normalization', 'fit_result'}
        if localize: keys |= {'corrected_target', 'sample_pair', 'fold_design'}
        if localize and not other_tf: keys.add('previous_result')
        if builder: keys.add('prior_double_result')
        require(set(plan['inputs']) == keys, 'INPUT_SET')
        meta_names = ('cache_manifest', 'prefix_normalization', 'fit_result') + (('fold_design',) if localize else ()) + (('previous_result',) if localize and not other_tf else ()) + (('prior_double_result',) if builder else ())
        meta = {key: binding.read_bound_json(Path(plan['inputs'][key]['path']), plan['inputs'][key]['sha256']) for key in meta_names}
        if localize and not other_tf:
            prior = policy[STAGE]
            require(prior['status'] == 'COMPLETE_ONE_SHOT_CONSUMED' and prior['execution_authorized'] is False
                and prior['result'] == plan['inputs']['previous_result']
                and meta['previous_result']['status'] == 'COMPLETE'
                and meta['previous_result']['all_causal_history_vs_cache']['mismatched_cells'] == 7551
                and meta['previous_result']['test_data_used'] is False, 'PRIOR_REPLAY')
        manifest, norm, fit = (meta[k] for k in ('cache_manifest', 'prefix_normalization', 'fit_result'))
        if builder:
            previous = policy['independent_direction_h1_trendline_replay_20261028_001']
            require(previous['status'] == 'COMPLETE_ONE_SHOT_CONSUMED' and previous['execution_authorized'] is False
                and previous['result'] == plan['inputs']['prior_double_result']
                and meta['prior_double_result']['status'] == 'COMPLETE'
                and meta['prior_double_result']['test_data_used'] is False, 'PRIOR_DOUBLE_REPLAY')
        tf_metadata = manifest['tfs'][tf]; cache_dir = Path(plan['inputs']['cache_manifest']['path']).parent
        require(manifest['m5_prebuilt_source'] == plan['inputs']['source_m5']['path']
            and manifest['m5_prebuilt_source_sha256'] == plan['inputs']['source_m5']['sha256']
            and {'path': str(cache_dir / tf_metadata['feats_npy']), 'sha256': tf_metadata['feats_npy_sha256']} == plan['inputs'][feat_key]
            and {'path': str(cache_dir / tf_metadata['ts_npy']), 'sha256': tf_metadata['ts_npy_sha256']} == plan['inputs'][ts_key]
            and fit['feature_view_identity']['prefix_normalization']['binding'] == plan['inputs']['prefix_normalization']
            and fit['fold']['oof_cutoff_ns'] < CUTOFF_NS and fit['test_data_used'] is False, 'INPUT_LINEAGE')
        require(manifest['technical_indicator_owner'] == technical_indicator_contract_metadata(), 'TECHNICAL_OWNER_IDENTITY')
        profile = compile_gc_mtf_geometry_entry_execution_profile(norm['gc_mtf_history'])
        names = tuple(TRENDLINE_REGISTRY_FEATURE_NAMES_V1); columns = tuple(manifest['feature_names'].index(name) for name in names)
        require(len(names) == len(set(names)) == 31 and set(columns) <= set(profile['mtf_indices_by_specialist']['chart_geometry_encoder']), 'ACTIVE_FIELDS')
        control_snapshot = binding.capture_verified_file_snapshot()
    proof, lock = cap.require_capped_cpu_audit_execution(), cap.require_capped_lock_ancestry()
    require(lock['scope'] == proof['cgroup_path'][len('/sys/fs/cgroup'):], 'CAPPED_LOCK_SCOPE')
    out = Path(plan['output_dir']); require(out.parent == ROOT and out.name == output_name
        and ROOT.resolve(strict=True) == ROOT and not out.exists() and not out.is_symlink(), 'OUTPUT_SCOPE')
    out.mkdir(exist_ok=False)
    def publish(path, value): publication._publish_bytes(path, binding.canonical_bytes(value))
    publish(out / 'INVOCATION_CLAIM.json', {'plan': plan_ref, 'policy': policy_ref, 'source_head': args.source_head, 'execution': proof, 'lock': lock})
    payload_names = ('source_m5', feat_key, ts_key) + (('corrected_target', 'sample_pair') if localize else ())
    payloads = [plan['inputs'][k] for k in payload_names]
    metadata = [plan_ref['path'], policy_ref['path'], plan['design']['path']] + [plan['inputs'][k]['path'] for k in meta_names]
    sys.addaudithook(candle.make_guard(out, [ref['path'] for ref in payloads], metadata,
        source_paths=[ref['path'] for ref in plan['sources'].values()]))
    started = time.monotonic(); old_handler = signal.getsignal(signal.SIGALRM)
    def expired(*unused): raise RuntimeError('DIRECTION_D1_TRENDLINE_REPLAY_WALL_LIMIT')
    signal.signal(signal.SIGALRM, expired); signal.alarm(plan['wall_seconds']); stage = out / '.OUTPUT.staging'
    try:
        stage.mkdir(exist_ok=False)
        with binding.fresh_file_verification_scope():
            for ref in payloads: candle.regular(ref['path']); binding.require_binding(ref, label='payload')
            source, source_witness = load_m5_prefix(Path(plan['inputs']['source_m5']['path']), CUTOFF_NS)
            precision = None
            if builder:
                clock = source.index.asi8
                bounds = (('fit_train', 1714521600000000000, fit['fold']['train_cutoff_ns']),
                    ('fit_oof_q1', fit['fold']['train_cutoff_ns'], fit['fold']['oof_cutoff_ns']),
                    ('later_q2', fit['fold']['oof_cutoff_ns'], CUTOFF_NS))
                precision = {name:price_precision_summary(source.loc[(clock >= start) & (clock < end)])
                    for name, start, end in bounds}
                source = source.copy()
                for name in ('open', 'high', 'low', 'close', 'volume'):
                    source[name] = source[name].astype(np.float32)
            index = build_multi_tf_v4_closed_timestamp_indices(source.index)[tf]
            if tf == 'M5':
                require(index.equals(source.index), 'NATIVE_M5_AXIS')
                bars = source.loc[index, ['open', 'high', 'low', 'close', 'volume']]
            elif tf in ('H1', 'M15'):
                # With finite OHLC, pandas last equals the owner's _last_valid.
                # Check that exact identity on a fixed prefix before the full fast resample.
                agg = {'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}
                witness_source = source.iloc[:4096]
                expected = _resample_ohlcv(witness_source, tf).dropna(subset=['open','high','low','close'])
                observed = multi_tf_resample(witness_source, tf).agg(agg).dropna(subset=['open','high','low','close'])
                require(expected.index.equals(observed.index) and np.array_equal(expected.to_numpy(), observed.to_numpy()), 'FAST_RESAMPLE_IDENTITY')
                bars = multi_tf_resample(source, tf).agg(agg).dropna(subset=['open','high','low','close']).loc[index]
            else:
                bars = _resample_ohlcv(source, tf).dropna(subset=['open', 'high', 'low', 'close']).loc[index]
            require(bars.index.equals(index), 'RESAMPLE_AXIS')
            atr = ema50_200_spread_atr_block(bars['high'], bars['low'], bars['close'])['atr14']
            data = bars[['high', 'low', 'close']].copy(); data['atr'] = atr
            constants = manifest['v29_registry_constants']; kwargs = {'timeframe':tf,
                'seq_len':int(constants['per_tf_seq_lens'][tf]), 'band_atr':constants['trendline_band_atr'][tf]}
            complete, _state = compute_trendline_registry_features_v1(data, **kwargs)
            require(tuple(complete.columns) == names and complete.index.equals(index), 'PRODUCER_AXIS')
            split = int(np.searchsorted(index.asi8, fit['fold']['train_cutoff_ns'], side='left'))
            require(0 < split < len(index), 'CHUNK_SPLIT')
            first, state = compute_trendline_registry_features_v1(data.iloc[:split], **kwargs)
            second, _state = compute_trendline_registry_features_v1(data.iloc[split:], state=state, **kwargs)
            chunked = pd.concat((first, second))
            require(chunked.index.equals(index), 'CHUNK_AXIS')
            chunk_comparison = compare_fields(complete.to_numpy(), chunked.to_numpy(), names, index.asi8)
            require(chunk_comparison['all_finite_float32_bits_and_missingness_equal'], 'REAL_CHUNK_PARITY')
            with Path(plan['inputs'][ts_key]['path']).open('rb') as handle:
                ts = np.load(handle, allow_pickle=False)
            cached_all = np.load(plan['inputs'][feat_key]['path'], mmap_mode='r', allow_pickle=False)
            at = np.searchsorted(ts, index.asi8)
            require(ts.dtype == np.int64 and np.all(at < len(ts)) and np.array_equal(ts[at], index.asi8)
                and cached_all.dtype == np.float32 and cached_all.shape == (tf_metadata['n_bars'], manifest['feature_count']), 'CACHE_EXACT_AXIS')
            actual = complete.to_numpy(dtype=np.float32); cached = cached_all[np.ix_(at, columns)]
            comparison = compare_fields(actual, cached, names, index.asi8)
            train_at = np.flatnonzero((index.asi8 + shift_ns >= 1714521600000000000)
                & (index.asi8 + shift_ns < fit['fold']['train_cutoff_ns']))
            later_at = np.flatnonzero((index.asi8 + shift_ns >= fit['fold']['train_cutoff_ns'])
                & (index.asi8 + shift_ns < CUTOFF_NS))
            require(len(train_at) and len(later_at), 'MODEL_PERIOD_ROWS')
            model_train = compare_fields(actual[train_at], cached[train_at], names, index.asi8[train_at])
            later = compare_fields(actual[later_at], cached[later_at], names, index.asi8[later_at])
            details = None
            if localize:
                require(np.array_equal(at, np.arange(len(at))), 'CACHE_PREFIX_AXIS')
                if not other_tf:
                    require(comparison['mismatched_cells'] == meta['previous_result']['all_causal_history_vs_cache']['mismatched_cells']
                        and comparison['rows'] == meta['previous_result']['all_causal_history_vs_cache']['rows'], 'PREVIOUS_REPLAY_REPRODUCED')
                missing, finite = mismatch_matrix(actual, cached); bad = np.any(missing | finite, axis=1)
                years = index.year.to_numpy(); year_rows = []
                for year in np.unique(years):
                    selected = years == year
                    year_rows.append({'year':int(year), 'mismatched_cells':int((missing[selected] | finite[selected]).sum()),
                        'missingness_mismatches':int(missing[selected].sum()), 'finite_bit_mismatches':int(finite[selected].sum()),
                        'native_rows_with_mismatch':int(np.any((missing | finite)[selected], axis=1).sum())})
                with np.load(plan['inputs']['corrected_target']['path'], allow_pickle=False) as archive:
                    parents = archive['parent_rows'].copy(); decisions = archive['decision_ns'].copy()
                with np.load(plan['inputs']['sample_pair']['path'], allow_pickle=False) as archive:
                    sample_parents = archive['parent_rows'].copy(); sample_decisions = archive['decision_ns'].copy()
                fold = fit['fold']; later_fold = meta['fold_design']['folds'][1]
                require(parents.dtype == decisions.dtype == sample_parents.dtype == sample_decisions.dtype == np.int64
                    and parents.ndim == decisions.ndim == 1 and len(parents) == len(decisions) == 118249
                    and np.all(np.diff(parents)>0) and np.all(np.diff(decisions)>0)
                    and fold['oof_end_row'] == later_fold['oof_start_row']
                    and np.searchsorted(decisions, fold['train_cutoff_ns']) == fold['oof_start_row']
                    and np.searchsorted(decisions, fold['oof_cutoff_ns']) == fold['oof_end_row']
                    and np.searchsorted(decisions, later_fold['oof_cutoff_ns']) == later_fold['oof_end_row']
                    and later_fold['oof_cutoff_ns'] == CUTOFF_NS
                    and WINDOW_ROWS[tf] == {'D1':252, 'H4':96, 'H1':96, 'M15':64, 'M5':16}[tf], 'DECISION_AXES')
                positions = np.searchsorted(parents, sample_parents)
                require(len(sample_parents)==1760 and np.all(positions >= later_fold['oof_start_row'])
                    and np.all(positions < later_fold['oof_end_row']) and np.array_equal(parents[positions], sample_parents)
                    and np.array_equal(decisions[positions], sample_decisions), 'LATER_SAMPLE_JOIN')
                windows = {
                    'fit_train':window_exposure(index.asi8, bad, decisions[:fold['oof_start_row']], WINDOW_ROWS[tf], shift_ns=shift_ns, timeframe=tf),
                    'fit_oof_q1':window_exposure(index.asi8, bad, decisions[fold['oof_start_row']:fold['oof_end_row']], WINDOW_ROWS[tf], shift_ns=shift_ns, timeframe=tf),
                    'intervention_q2_sample':window_exposure(index.asi8, bad, sample_decisions, WINDOW_ROWS[tf], shift_ns=shift_ns, timeframe=tf)}
                at_bad = np.flatnonzero(bad)
                details = {'mismatched_cells':int((missing | finite).sum()), 'missingness_mismatches':int(missing.sum()),
                    'finite_bit_mismatches':int(finite.sum()), f'mismatched_native_{tf.lower()}_rows':len(at_bad),
                    'first_mismatch_ns':int(index.asi8[at_bad[0]]) if len(at_bad) else None,
                    'last_mismatch_ns':int(index.asi8[at_bad[-1]]) if len(at_bad) else None,
                    'by_year':year_rows, f'model_window_length_{tf.lower()}_rows':WINDOW_ROWS[tf], 'window_exposure':windows}
            payload_snapshot = binding.capture_verified_file_snapshot()
        payload_snapshot.require_unchanged(); control_snapshot.require_unchanged()
        require(candle.git_state() == initial and cap.require_capped_lock_ancestry() == lock
            and not any(name.split('.')[0] in ('torch', 'sklearn', 'joblib') for name in sys.modules), 'FINAL_SOURCE_LOCK')
        result = {'schema_version':schema + '_result_v1','status':'COMPLETE',
            'plan':plan_ref,'source_head':args.source_head,'timeframe':tf,'source_witness':source_witness,
            'cutoff_ns':CUTOFF_NS,'source_clock_rows':len(index),'history_state_split_index':split,
            'window_configuration':kwargs,'chunk_replay':chunk_comparison,'all_causal_history_vs_cache':comparison,
            'model_train_period_vs_cache':model_train,'later_development_period_vs_cache':later,
            'replay_parity':comparison['all_finite_float32_bits_and_missingness_equal'],
            'mismatch_localization':details,
            'source_price_precision':precision,
            'input_price_precision':'builder_float32' if builder else 'source_float64_counterfactual',
            'limits':'Same current registry implementation recomputed from genuine M5 source and compared with cached model input; not an independent mathematical implementation, predictive-value test or proof of all fields/timeframes.',
            'new_model_forwards':0,'optimizer_steps':0,'test_data_used':False,'economic_edge_qualified':False}
        publish(stage/'RESULT.json',result)
        publish(stage/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'source_unchanged':True,'inputs_unchanged':True,
            'elapsed_seconds':time.monotonic()-started,'test_data_used':False})
        publish_bundle_directory_noreplace(stage,out/'OUTPUT')
        publish(out/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'plan':plan_ref,'result_path':str(out/'OUTPUT/RESULT.json')})
        return 0
    except Exception as exc:
        publish(out/'FAILED_TERMINAL.json',{'status':'FAILED','exit_code':1,'error':type(exc).__name__+': '+str(exc),'relaunch_allowed':False})
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM,old_handler)


if __name__ == '__main__':
    raise SystemExit(main())
