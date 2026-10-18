"""Fresh native-input rebuild from declared raw MBP-1, never a prior feature table."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gx1.contracts import gc_continuous_source_v1 as source
from gx1.contracts import gc_source_disposition_v1 as timing
from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution, require_capped_lock_ancestry
from gx1.features import gc_order_flow_v1 as features
from gx1.scripts import gc_source_disposition_v1 as io
from gx1.scripts.gc_continuous_source_v1 import load_metadata
from gx1.scripts.gc_order_flow_features_v1 import add_basis

CODE_PATHS = (
    'gx1/scripts/rebuild_native_gc_inputs_v1.py',
    'gx1/features/gc_order_flow_v1.py',
    'gx1/scripts/gc_order_flow_features_v1.py',
    'gx1/contracts/gc_continuous_source_v1.py',
    'gx1/contracts/gc_source_disposition_v1.py',
    'gx1/contracts/gc_order_flow_source_v1.py',
    'gx1/scripts/gc_source_disposition_v1.py',
    'gx1/scripts/gc_continuous_source_v1.py',
    'gx1/features/htf_features.py',
    'gx1/features/smc_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/entry_model_native_bundle_commit_v1.py',
    'scripts/gx1_capped_run.sh',
)


def selected_minutes(daily, maturities):
    """Existing previous-complete-day rule, available at 00:01 UTC."""
    chosen, basis = source.choose_daily_contract(daily, maturities)
    prefix, _ = source.choose_daily_contract(daily[:, :-1], maturities)
    if not np.array_equal(prefix, chosen[:-1]):
        raise RuntimeError('RAW_GC_ROLL_PREFIX')
    width = daily.shape[1] * source.MINUTES_PER_DAY
    roll_day = (np.arange(width) - 1) // source.MINUTES_PER_DAY
    selected = np.full(width, -1, np.int16)
    ready = roll_day >= 0
    selected[ready] = chosen[roll_day[ready]]
    return selected, chosen, basis


def l1_masks(quality, selected, covered, *, reject_duplicate_signature=True):
    clean = (quality['records'] > 0) & (quality['clean_complete_bbo'] > 0)
    for name in ('bad_bbo', 'gap', 'bad_recv', 'snapshot', 'reset',
                 'duplicate_trade_signature', 'file_seam'):
        if name != 'duplicate_trade_signature' or reject_duplicate_signature:clean &= quality[name] == 0
    prior = np.r_[False, clean[:-1] & (selected[1:] == selected[:-1]) & (selected[1:] >= 0)]
    ofi = clean & prior & (selected >= 0)
    return ofi, ofi & (covered == timing.MINUTE_NS)


def raw_worker(group, ci, tick, stage):
    width = (timing.WINDOW_END_NS - timing.STUDY_START_NS) // timing.MINUTE_NS
    wanted = np.arange(width, dtype=np.int64)
    quality = source.L1QualityCollector(wanted, tick)
    book = features.L1FeatureCollectorV2(wanted, tick, include_order_counts=True)
    ladder_path = stage / (group['raw_symbol'] + '_LADDER.parquet')
    trades = features.TradeLadderCollector(ladder_path, np.full(width, ci, np.int16),
                                          np.ones(width, bool), ci, tick)

    def start(item):
        quality.file_start(item)
        book.file_start(item)

    def consume(data, idx, trusted):
        quality(data, idx, trusted)
        book(data, idx, trusted)
        trades(data, idx, trusted)

    try:
        result = source.scan_symbol(group['items'], tick, on_batch=consume,
                                    extra_columns=source.L1_AUDIT_COLUMNS, on_file_start=start)
        book.finish()
    finally:
        trades.close()
    # Unqualified values are intermediate reducer output only. The selected
    # clock and source-quality masks below are their sole admission authority.
    values, _ = book.values(np.ones(width, bool))
    counts, _ = book.order_count_values(np.ones(width, bool))
    table = pa.table({**{k: result[k] for k in source.METRICS},
                      **{'quality_' + k: v for k, v in quality.values.items()},
                      **{k: trades.ohlc[:, j] for j, k in enumerate(
                          ('open_tick', 'high_tick', 'low_tick', 'close_tick'))},
                      'covered_ns': book.covered, 'opening_mid_tick': book.first,
                      'closing_mid_tick': book.last, 'order_covered_ns': book.order_covered, **values, **counts})
    path = stage / (group['raw_symbol'] + '_M1.parquet')
    pq.write_table(table, path, compression='zstd')
    return {'symbol': group['raw_symbol'], 'path': str(path), 'ladder': str(ladder_path),
            'daily': result['volume'].reshape(-1, source.MINUTES_PER_DAY).sum(axis=1),
            'files': result['files']}


def write_footprint(paths, minute, selected, metrics, masks, destination):
    """One exact observed quantity per selected M1/price; unknown side retained."""
    schema = pa.schema([('bar_start_ns', pa.int64()), ('contract_index', pa.int16()),
        ('price_tick', pa.int64()), ('buy', pa.uint64()), ('sell', pa.uint64()),
        ('unknown', pa.uint64()), ('signed_valid', pa.bool_())])
    total_rows = 0
    buffered_rows = 0
    buffered = []
    with pq.ParquetWriter(destination, schema, compression='zstd') as writer:
        for ci, path in enumerate(paths):
            current = None
            histogram = {}

            def flush():
                nonlocal total_rows, buffered_rows
                if current is None:
                    return
                keys = sorted(histogram)
                quantities = np.array([histogram[k] for k in keys], np.uint64)
                if any(sum(map(int, quantities[:, j])) != int(metrics[k][current])
                       for j, k in enumerate(('buy', 'sell', 'unknown'))):
                    raise RuntimeError('RAW_GC_FOOTPRINT_SIDE_CONSERVATION')
                if selected[current] != ci or not masks['volume'][current]:
                    raise RuntimeError('RAW_GC_FOOTPRINT_SOURCE_MASK')
                buffered.append(pa.table({
                    'bar_start_ns': np.full(len(keys), minute[current], np.int64),
                    'contract_index': np.full(len(keys), ci, np.int16),
                    'price_tick': np.array(keys, np.int64), 'buy': quantities[:, 0],
                    'sell': quantities[:, 1], 'unknown': quantities[:, 2],
                    'signed_valid': np.full(len(keys), masks['delta'][current], bool)}, schema=schema))
                total_rows += len(keys)
                buffered_rows += len(keys)
                if buffered_rows >= 65536:
                    writer.write_table(pa.concat_tables(buffered))
                    buffered.clear()
                    buffered_rows = 0

            for batch in pq.ParquetFile(path).iter_batches(batch_size=65536):
                arrays = {k: batch.column(batch.schema.get_field_index(k)).to_numpy()
                          for k in batch.schema.names}
                rows = arrays['minute_index']
                unique, first, counts = np.unique(rows, return_index=True, return_counts=True)
                for row, begin, count in zip(unique, first, counts):
                    if current != int(row):
                        flush()
                        current = int(row)
                        histogram = {}
                    for j in range(begin, begin + count):
                        key = int(arrays['price_tick'][j])
                        q = np.array([arrays[k][j] for k in ('buy', 'sell', 'unknown')], np.uint64)
                        if key in histogram:
                            histogram[key] += q
                        else:
                            histogram[key] = q
            flush()
        if buffered:
            writer.write_table(pa.concat_tables(buffered))
    return total_rows


def build(plan, stage):
    base = io.read_json(io.REPO / plan['source_plan']['path'])
    io.verify_digest(io.REPO / plan['source_plan']['path'], plan['source_plan']['sha256'], 'RAW_SOURCE_PLAN')
    groups, hashes = io._bound_gc_inputs(base)
    metadata, metadata_hashes = load_metadata(base, groups)
    ticks = {v[2] for v in metadata.values()}
    if len(ticks) != 1:
        raise RuntimeError('RAW_GC_TICK_NOT_UNIFORM')
    tick = ticks.pop()
    parts = stage / 'RAW_REDUCTIONS'
    parts.mkdir()
    with ProcessPoolExecutor(max_workers=plan['worker_count'], mp_context=mp.get_context('fork')) as pool:
        jobs = [pool.submit(raw_worker, g, ci, metadata[g['raw_symbol']][2], parts)
                for ci, g in enumerate(groups)]
        results = [job.result() for job in jobs]
    records = sum(f['records'] for r in results for f in r['files'])
    if records != plan['expected_source_records']:
        raise RuntimeError('RAW_GC_RECORD_COUNT')
    daily = np.stack([r['daily'] for r in results])
    selected, chosen, basis = selected_minutes(daily, [metadata[r['symbol']][:2] for r in results])
    pq.write_table(pa.table({r['symbol']: daily[j] for j, r in enumerate(results)}),
                   stage / 'DAILY_CONTRACT_VOLUMES.parquet', compression='zstd')
    width = len(selected)
    minute = timing.STUDY_START_NS + np.arange(width, dtype=np.int64) * timing.MINUTE_NS
    metrics = {k: np.zeros(width, np.uint64) for k in source.METRICS}
    quality = {k: np.zeros(width, np.uint64) for k in source.L1_QUALITY_METRICS}
    ohlc = np.zeros((width, 4), np.int64)
    book = {k: np.full(width, np.nan) for k in features.L1_FIELDS_V2 + features.L1_ORDER_COUNT_FIELDS_V1}
    covered = np.zeros(width, np.int64)
    order_covered = covered.copy()
    first = np.full(width, np.nan)
    last = first.copy()
    bad_recv = 0
    for ci, result in enumerate(results):
        table = pq.read_table(result['path'])
        take = selected == ci
        for k in metrics:
            metrics[k][take] = table[k].to_numpy()[take]
        for k in quality:
            quality[k][take] = table['quality_' + k].to_numpy()[take]
        for j, k in enumerate(('open_tick', 'high_tick', 'low_tick', 'close_tick')):
            ohlc[take, j] = table[k].to_numpy()[take]
        for k in book:
            book[k][take] = table[k].to_numpy()[take]
        covered[take] = table['covered_ns'].to_numpy()[take]
        order_covered[take] = table['order_covered_ns'].to_numpy()[take]
        first[take] = table['opening_mid_tick'].to_numpy()[take]
        last[take] = table['closing_mid_tick'].to_numpy()[take]
        for f in result['files']:
            lo = (f['start_ns'] - timing.STUDY_START_NS) // timing.MINUTE_NS
            hi = (f['end_ns'] - timing.STUDY_START_NS) // timing.MINUTE_NS
            if np.any(take[lo:hi]):
                bad_recv += f['bad_recv_trades']
        del table
    if bad_recv or not np.array_equal(metrics['volume'], metrics['buy'] + metrics['sell'] + metrics['unknown']):
        raise RuntimeError('RAW_GC_SOURCE_ADMISSION')
    masks = source.row_masks(selected, metrics, minute + timing.MINUTE_NS < timing.WINDOW_END_NS)
    ofi, bbo = l1_masks(quality, selected, covered)
    order_valid = ofi & (order_covered == timing.MINUTE_NS)
    for j, k in enumerate(features.L1_FIELDS_V2 + features.L1_ORDER_COUNT_FIELDS_V1):
        mask = ofi if j < 6 else bbo if j < len(features.L1_FIELDS_V2) else order_valid
        book[k][~mask] = np.nan
    m1 = pa.table({'bar_start_ns': minute, 'contract_index': selected, **metrics,
                   **{'quality_' + k: v for k, v in quality.items()},
                   'volume_valid': masks['volume'], 'signed_valid': masks['delta'],
                   'ofi_valid': ofi, 'bbo_valid': bbo, 'covered_ns': covered,
                   'order_count_valid': order_valid, 'order_covered_ns': order_covered,
                   'opening_mid_tick': first, 'closing_mid_tick': last,
                   **{k: ohlc[:, j] for j, k in enumerate(('open_tick', 'high_tick', 'low_tick', 'close_tick'))},
                   **book})
    pq.write_table(m1, stage / 'M1_GC_INPUTS.parquet', compression='zstd')
    start = minute[::5]
    contracts = selected.reshape(-1, 5)
    same = (contracts >= 0).all(axis=1) & (contracts == contracts[:, :1]).all(axis=1)
    valid = masks['volume'].reshape(-1, 5).all(axis=1) & same
    volume = metrics['volume'].reshape(-1, 5).sum(axis=1)
    histograms = [{} for _ in start]
    surfaces, minute_maps = {}, {}
    for tf in ('M5', 'M15', 'H1', 'H4', 'D1'):
        surfaces[tf], minute_maps[tf] = features.timeframe_skeleton(
            minute, selected, metrics, masks, ohlc, tf, tick)
    for ci, result in enumerate(results):
        selected_path = parts / (result['symbol'] + '_SELECTED_LADDER.parquet')
        schema = pq.ParquetFile(result['ladder']).schema_arrow
        writer = pq.ParquetWriter(selected_path, schema, compression='zstd')
        for batch in pq.ParquetFile(result['ladder']).iter_batches(batch_size=65536):
            frame = batch.to_pandas()
            rows = frame.minute_index.to_numpy().astype(np.int64)
            bar = rows // 5
            selected_rows = (selected[rows] == ci) & masks['volume'][rows]
            writer.write_table(pa.Table.from_batches([batch]).filter(pa.array(selected_rows)))
            take = selected_rows & valid[bar]
            frame = frame.loc[take].copy()
            frame['bar'] = bar[take]
            frame['volume'] = frame.buy + frame.sell + frame.unknown
            for (row, price), qty in frame.groupby(['bar', 'price_tick']).volume.sum().items():
                h = histograms[int(row)]
                h[int(price)] = h.get(int(price), 0) + int(qty)
        writer.close()
        features.consume_ladder(selected_path, surfaces, minute_maps, tick)
    footprint_rows = write_footprint([parts / (r['symbol'] + '_SELECTED_LADDER.parquet')
        for r in results], minute, selected, metrics, masks, stage / 'FOOTPRINT_M1_LEVELS.parquet')
    cvd = features.relative_cvd(minute, selected, masks, metrics)
    for tf, surface in surfaces.items():
        features.add_structure_and_cvd(surface, cvd, minute_maps, minute)
        pq.write_table(pa.table(surface), stage / ('TRADE_FEATURES_' + tf + '.parquet'), compression='zstd')
    for row in np.flatnonzero(valid):
        if sum(histograms[row].values()) != int(volume[row]):
            raise RuntimeError('RAW_GC_PROFILE_VOLUME_CONSERVATION')
    close = ohlc[4::5, 3].astype(float) * tick / features.PRICE_SCALE
    levels = features.dynamic_profiles_v2(start, histograms, valid, contracts[:, 0],
                                         close, tick / features.PRICE_SCALE)
    cut = len(start) // 2
    prefix = features.dynamic_profiles_v2(start[:cut], histograms[:cut], valid[:cut],
                                          contracts[:cut, 0], close[:cut], tick / features.PRICE_SCALE)
    for k in levels:
        if not np.array_equal(levels[k][:cut], prefix[k], equal_nan=True):
            raise RuntimeError('RAW_GC_PROFILE_CAUSAL_PREFIX:' + k)
    out = legacy_strict_m5_surface(minute, selected, metrics, masks, ohlc,
        {**book,'ofi_valid':ofi,'bbo_valid':bbo,'order_count_valid':order_valid,
         'opening_mid_tick':first,'closing_mid_tick':last},tick,levels)
    output = pa.table(out)
    path = stage / 'M5_GC_FEATURES.parquet'
    pq.write_table(output, path, compression='zstd')
    if not features.exact_feature_readback_v2(output, pq.read_table(path)):
        raise RuntimeError('RAW_GC_READBACK')
    io._rehash_sources(hashes, metadata_hashes)
    months = pd.DatetimeIndex(start, tz='UTC').strftime('%Y-%m').to_numpy()
    coverage = {k: {m: int(np.count_nonzero(v & (months == m))) for m in sorted(set(months))}
                for k, v in out.items() if k.endswith('_valid')}
    io.write_json(stage / 'ROLL_SCHEDULE.json', {'contract_order': [r['symbol'] for r in results],
        'metadata_identity': metadata, 'selected_daily': chosen.tolist(), 'basis_daily': basis.tolist(),
        'origin_ns': timing.STUDY_START_NS, 'availability_offset_ns': timing.MINUTE_NS})
    return {'raw_records': records, 'raw_files': sum(len(r['files']) for r in results),
        'full_GC_M1_rows': width, 'M5_rows': len(start), 'coverage': coverage,
        'footprint_price_rows': footprint_rows,
        'footprint_side_conservation': 'EXACT_EVERY_VALID_SELECTED_M1',
        'source_files': [f for r in results for f in r['files']],
        'trade_feature_support': {tf: {k: int(np.count_nonzero(np.isfinite(v) & (v != 0)))
            for k, v in surf.items() if k not in ('bar_start_ns', 'available_ns', 'contract_index')}
            for tf, surf in surfaces.items()},
        'features_built_from_raw': True, 'prior_feature_values_reused': False,
        'profile_volume_conservation': 'EXACT_ALL_VALID_M5',
        'profile_prefix_rows': cut, 'session_holiday_calendar_qualified': False,
        'native_model_wiring_qualified': False, 'native_dataset_complete': False,
        'test_data_used': False, 'optimizer_steps': 0}


COMPLETION_SCHEMA = 'gx1_native_mbp1_information_completion_plan_v1'
COMPLETION_MODE = 'complete_mbp1_from_completed_m1_and_raw_events'

DERIVED_TIMEFRAME_SCHEMA = 'gx1_native_derived_gc_timeframes_plan_v1'
DERIVED_TIMEFRAME_MODE = 'derive_timeframes_from_completed_m1'
DERIVED_TIMEFRAMES = ('M5', 'M15', 'H1', 'H4', 'D1')
DERIVED_INPUT_NAMES = {
    'm1_inputs': 'M1_GC_INPUTS.parquet',
    'footprint_m1_levels': 'FOOTPRINT_M1_LEVELS.parquet',
    'roll_schedule': 'ROLL_SCHEDULE.json',
}


def derived_timeframe_mode(plan):
    """An explicit new plan cannot inherit the consumed raw-stage authority."""
    if plan.get('schema_version') != DERIVED_TIMEFRAME_SCHEMA:
        if 'mode' in plan:
            raise RuntimeError('DERIVED_GC_PLAN_MODE')
        return False
    if (plan.get('mode') != DERIVED_TIMEFRAME_MODE
            or tuple(plan.get('timeframes', ())) != DERIVED_TIMEFRAMES
            or pd.Timestamp(plan['study_start_inclusive']).value != timing.STUDY_START_NS
            or pd.Timestamp(plan['study_end_exclusive']).value != timing.WINDOW_END_NS):
        raise RuntimeError('DERIVED_GC_PLAN_SCOPE')
    return True


def completion_mode(plan):
    if plan.get('schema_version') != COMPLETION_SCHEMA:return False
    if (plan.get('mode') != COMPLETION_MODE or tuple(plan.get('timeframes', ())) != DERIVED_TIMEFRAMES
            or pd.Timestamp(plan['study_start_inclusive']).value != timing.STUDY_START_NS
            or pd.Timestamp(plan['study_end_exclusive']).value != timing.WINDOW_END_NS
            or not isinstance(plan.get('worker_count'), int) or plan['worker_count'] < 1
            or not isinstance(plan.get('profile_state_max_bytes'), int) or plan['profile_state_max_bytes'] < 1
            or not isinstance(plan.get('expected_source_records'), int) or plan['expected_source_records'] < 1
            or not isinstance(plan.get('expected_source_files'), int) or plan['expected_source_files'] < 1
            or set(plan.get('source_plan', {})) != {'path', 'sha256'}):
        raise RuntimeError('MBP1_COMPLETION_PLAN_SCOPE')
    return True


QUALITY_REDERIVE_SCHEMA = 'gx1_native_gc_source_quality_rederive_plan_v1'
QUALITY_REDERIVE_MODE = 'rederive_completed_gc_after_bound_source_incident'


def quality_rederive_mode(plan):
    if plan.get('schema_version') != QUALITY_REDERIVE_SCHEMA:return False
    if (plan.get('mode') != QUALITY_REDERIVE_MODE or tuple(plan.get('timeframes', ())) != DERIVED_TIMEFRAMES
            or pd.Timestamp(plan['study_start_inclusive']).value != timing.STUDY_START_NS
            or pd.Timestamp(plan['study_end_exclusive']).value != timing.WINDOW_END_NS
            or not isinstance(plan.get('profile_state_max_bytes'), int) or plan['profile_state_max_bytes'] < 1
            or set(plan.get('source_plan', {})) != {'path','sha256'}
            or set(plan.get('source_incidents', {})) != {'path','sha256'}
            or set(plan.get('retained_registry_source', {})) != {'path','sha256'}
            or set(plan.get('event_reductions', {})) != {'path','sha256','source_terminal'}):
        raise RuntimeError('GC_QUALITY_REDERIVE_SCOPE')
    return True



TRUSTED_RESTORATION_SCHEMA = 'gx1_native_gc_trusted_record_restoration_plan_v1'
TRUSTED_RESTORATION_MODE = 'restore_trusted_records_and_complete_event_statistics'


def trusted_restoration_mode(plan):
    if plan.get('schema_version') != TRUSTED_RESTORATION_SCHEMA:return False
    probe = dict(plan,schema_version=QUALITY_REDERIVE_SCHEMA,mode=QUALITY_REDERIVE_MODE)
    quality_rederive_mode(probe)
    if (plan.get('mode') != TRUSTED_RESTORATION_MODE
            or any(not isinstance(plan.get(name),int) or isinstance(plan[name],bool) or plan[name]<1 for name in
                ('worker_count','expected_source_records','expected_source_files','expected_duplicate_only_m1','expected_duplicate_only_volume_contracts'))):
        raise RuntimeError('GC_TRUSTED_RESTORATION_SCOPE')
    return True


def bound_raw_reduction_inputs(paths, roll):
    """Resolve exact original inventory entries, never discover cache files by glob."""
    receipt = io.read_json(paths['source_terminal']);root = paths['source_terminal'].parent
    results = [];bindings = []
    for ci,symbol in enumerate(roll['contract_order']):
        item = {'symbol':symbol,'contract_index':ci}
        for key,suffix in (('m1','_M1.parquet'),('ladder','_LADDER.parquet')):
            relative = 'RAW_REDUCTIONS/'+symbol+suffix;entry = receipt['artifacts'].get(relative,{})
            path = root/relative
            if (not path.is_file() or path.is_symlink() or any(p.is_symlink() for p in path.parents)
                    or path.resolve(strict=True)!=path or entry.get('bytes')!=path.stat().st_size
                    or not entry.get('sha256')):
                raise RuntimeError('GC_TRUSTED_RAW_CACHE_INVENTORY:'+relative)
            io.verify_digest(path,entry['sha256'],'GC_TRUSTED_RAW_CACHE')
            item[key] = path;bindings.append({'path':str(path),'sha256':entry['sha256']})
        results.append(item)
    return results,bindings


def restore_selected_m1_from_raw_caches(original, caches, roll):
    """Recover reductions before the old signature-only whole-minute exclusion."""
    selected = original['contract_index'].to_numpy();width = len(selected)
    columns = list(source.METRICS)+['quality_'+k for k in source.L1_QUALITY_METRICS]
    columns += ['open_tick','high_tick','low_tick','close_tick','covered_ns','order_covered_ns',
        'opening_mid_tick','closing_mid_tick']+list(features.L1_FIELDS_V2)+list(features.L1_ORDER_COUNT_FIELDS_V1)
    restored = {name:original[name].to_numpy().copy() for name in original.column_names}
    daily = []
    for item in caches:
        raw = pq.read_table(item['m1'])
        if raw.num_rows!=width or set(raw.column_names)!=set(columns):
            raise RuntimeError('GC_TRUSTED_RAW_CACHE_M1_SCHEMA')
        take = selected==item['contract_index']
        for name in columns:
            values = raw[name].to_numpy()
            # All non-book reductions were already retained unmasked in001.
            if name not in features.L1_FIELDS_V2+features.L1_ORDER_COUNT_FIELDS_V1 and not np.array_equal(values[take],restored[name][take],equal_nan=True):
                raise RuntimeError('GC_TRUSTED_RAW_CACHE_ORIGINAL_REDUCTION_PARITY:'+name)
            restored[name][take] = values[take]
        daily.append(raw['volume'].to_numpy().reshape(-1,source.MINUTES_PER_DAY).sum(axis=1))
    selected_again,chosen,basis = selected_minutes(np.stack(daily),[tuple(roll['metadata_identity'][s][:2]) for s in roll['contract_order']])
    if (not np.array_equal(selected_again,selected) or not np.array_equal(chosen,roll['selected_daily'])
            or not np.array_equal(basis,roll['basis_daily'])):
        raise RuntimeError('GC_TRUSTED_RAW_CACHE_ROLL_PARITY')
    minutes = restored['bar_start_ns'];metrics = {name:restored[name] for name in source.METRICS}
    quality = {name:restored['quality_'+name] for name in source.L1_QUALITY_METRICS}
    eligible = minutes+timing.MINUTE_NS<timing.WINDOW_END_NS
    old = source.row_masks(selected,metrics,eligible)
    new = source.row_masks(selected,metrics,eligible,reject_duplicate_signature=False)
    old_ofi,old_bbo = l1_masks(quality,selected,restored['covered_ns'])
    new_ofi,new_bbo = l1_masks(quality,selected,restored['covered_ns'],reject_duplicate_signature=False)
    old_order = old_ofi & (restored['order_covered_ns']==timing.MINUTE_NS)
    for name,expected in (('volume_valid',old['volume']),('signed_valid',old['delta']),('ofi_valid',old_ofi),('bbo_valid',old_bbo),('order_count_valid',old_order)):
        if not np.array_equal(original[name].to_numpy(),expected):raise RuntimeError('GC_TRUSTED_ORIGINAL_MASK_PARITY:'+name)
    for j,name in enumerate(features.L1_FIELDS_V2+features.L1_ORDER_COUNT_FIELDS_V1):
        oldmask = old_ofi if j<6 else old_bbo if j<len(features.L1_FIELDS_V2) else old_order
        expected = np.where(oldmask,restored[name],np.nan)
        if not np.array_equal(original[name].to_numpy(),expected,equal_nan=True):
            raise RuntimeError('GC_TRUSTED_ORIGINAL_BOOK_PARITY:'+name)
    restored.update(volume_valid=new['volume'],signed_valid=new['delta'],ofi_valid=new_ofi,bbo_valid=new_bbo,
        order_count_valid=new_ofi & (restored['order_covered_ns']==timing.MINUTE_NS))
    return restored,old,new,{'volume_m1':int(np.count_nonzero(new['volume']&~old['volume'])),
        'volume_contracts':sum(map(int,metrics['volume'][new['volume']&~old['volume']])),
        'signed_m1':int(np.count_nonzero(new['delta']&~old['delta'])),
        'ofi_m1':int(np.count_nonzero(new_ofi&~old_ofi)),'bbo_m1':int(np.count_nonzero(new_bbo&~old_bbo))}


def select_cached_ladders(caches,selected,masks,stage):
    parts = stage/'RESTORED_LADDER_REDUCTIONS';parts.mkdir()
    outputs = []
    for item in caches:
        raw = pq.ParquetFile(item['ladder']);path = parts/(item['symbol']+'_SELECTED_LADDER.parquet')
        expected = pa.schema([('minute_index',pa.int32()),('price_tick',pa.int64()),('buy',pa.uint64()),('sell',pa.uint64()),('unknown',pa.uint64())])
        if raw.schema_arrow!=expected:raise RuntimeError('GC_TRUSTED_RAW_LADDER_SCHEMA')
        with pq.ParquetWriter(path,expected,compression='zstd') as writer:
            for batch in raw.iter_batches(batch_size=65536):
                table = pa.Table.from_batches([batch]);indices = table['minute_index'].to_numpy().astype(np.int64)
                if np.any((indices<0)|(indices>=len(selected))):raise RuntimeError('GC_TRUSTED_RAW_LADDER_CLOCK')
                take = (selected[indices]==item['contract_index']) & masks['volume'][indices]
                writer.write_table(table.filter(pa.array(take)))
        outputs.append(path)
    return outputs



def assert_restored_price_grid_original_subset(original_path, restored_path, minutes, prior_valid):
    """Exact original minute/price/side bytes on the unchanged prior population."""
    def chunks(path,filter_prior):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536):
            table = pa.Table.from_batches([batch])
            if filter_prior:
                offsets = table['bar_start_ns'].to_numpy()-minutes[0];indices = offsets//timing.MINUTE_NS
                if np.any((offsets%timing.MINUTE_NS!=0)|(indices<0)|(indices>=len(minutes))):
                    raise RuntimeError('GC_TRUSTED_GRID_SUBSET_CLOCK')
                table = table.filter(pa.array(prior_valid[indices]))
            if table.num_rows:yield table
    a,b = iter(chunks(original_path,False)),iter(chunks(restored_path,True))
    left,right = next(a,None),next(b,None);rows = 0
    while left is not None and right is not None:
        n = min(left.num_rows,right.num_rows)
        if not left.slice(0,n).equals(right.slice(0,n)):
            raise RuntimeError('GC_TRUSTED_ORIGINAL_PRICE_GRID_PARITY')
        rows += n
        left = left.slice(n) if n<left.num_rows else next(a,None)
        right = right.slice(n) if n<right.num_rows else next(b,None)
    if left is not None or right is not None:raise RuntimeError('GC_TRUSTED_ORIGINAL_PRICE_GRID_SIZE')
    return rows


def assert_original_event_subset(events, original_events, original_valid):
    if set(events)!=set(original_events):raise RuntimeError('GC_TRUSTED_019_EVENT_SUBSET_SCHEMA')
    for name,value in events.items():
        if value.shape!=original_valid.shape or original_events[name].shape!=original_valid.shape:
            raise RuntimeError('GC_TRUSTED_019_EVENT_SUBSET_SHAPE')
        if not np.array_equal(value[original_valid],original_events[name][original_valid],equal_nan=True):
            raise RuntimeError('GC_TRUSTED_019_EVENT_SUBSET_PARITY:'+name)


def build_trusted_restoration(plan,stage):
    """Recover bound raw reductions and execute exactly one missing-event raw pass."""
    paths = bound_derived_inputs(plan);quality_paths,quality_bindings = bound_quality_inputs(plan)
    manifest = io.read_json(quality_paths['source_incidents']);admission = manifest.get('trusted_trade_admission',{})
    if (admission.get('schema_version')!='gx1_gc_trusted_trade_source_admission_v1'
            or admission.get('signature_identity_is_bad_source_proof') is not False
            or admission.get('deduplicate_T') is not False or admission.get('reject_duplicate_signature') is not False
            or admission.get('all_other_trade_flag_price_grid_guards_preserved') is not True or not admission.get('source_url')):
        raise RuntimeError('GC_TRUSTED_SOURCE_ADMISSION_MANIFEST')
    roll = io.read_json(paths['roll_schedule']);caches,cache_bindings = bound_raw_reduction_inputs(paths,roll)
    original = pq.read_table(paths['m1_inputs'])
    restored,old,new,delta = restore_selected_m1_from_raw_caches(original,caches,roll)
    if (delta['volume_m1']!=plan['expected_duplicate_only_m1']
            or delta['volume_contracts']!=plan['expected_duplicate_only_volume_contracts']):
        raise RuntimeError('GC_TRUSTED_RESTORATION_MEASURED_BASELINE_PARITY')
    minutes = restored['bar_start_ns'];selected = restored['contract_index']
    old_events = load_quality_event_reductions(quality_paths['events'],minutes,selected,original['ofi_valid'].to_numpy(),restored['covered_ns'])
    base_path = io.REPO/plan['source_plan']['path'];io.verify_digest(base_path,plan['source_plan']['sha256'],'GC_TRUSTED_SOURCE_PLAN')
    groups,hashes = io._bound_gc_inputs(io.read_json(base_path));metadata,metadata_hashes = load_metadata(io.read_json(base_path),groups)
    if ([g['raw_symbol'] for g in groups]!=roll['contract_order']
            or {k:list(v) for k,v in metadata.items()}!=roll['metadata_identity']):
        raise RuntimeError('GC_TRUSTED_SOURCE_CONTRACT_METADATA')
    parts = stage/'RAW_EVENT_REDUCTIONS';parts.mkdir()
    with ProcessPoolExecutor(max_workers=plan['worker_count'],mp_context=mp.get_context('fork')) as pool:
        jobs = []
        for ci,group in enumerate(groups):
            wanted = np.flatnonzero(selected==ci)
            metrics = {name:restored[name][wanted] for name in source.METRICS}
            book = {name:restored[name][wanted] for name in features.L1_FIELDS_V2}
            jobs.append(pool.submit(completion_event_worker,group,ci,metadata[group['raw_symbol']][2],wanted,
                np.ones(len(wanted),bool),metrics,book,parts,include_trade_statistics=True))
        results = [job.result() for job in jobs]
    records = sum(f['records'] for r in results for f in r['source_files'])
    files = sum(len(r['source_files']) for r in results)
    if records!=plan['expected_source_records'] or files!=plan['expected_source_files']:
        raise RuntimeError('GC_TRUSTED_EVENT_SOURCE_COUNT')
    events = None;seen = np.zeros(len(selected),bool)
    trade_stats = {name:np.zeros(len(selected),np.uint64) for name in features.TRUSTED_TRADE_SIZE_FIELDS_V1}
    for item in results:
        part = pq.read_table(item['path']);indices = part['minute_index'].to_numpy().astype(np.int64)
        if seen[indices].any() or not np.array_equal(indices,np.flatnonzero(selected==item['contract_index'])):
            raise RuntimeError('GC_TRUSTED_EVENT_SELECTED_CLOCK')
        seen[indices] = True
        if events is None:events = {name:np.full(len(selected),np.nan) for name in part.column_names if name!='minute_index'}
        for name in events:events[name][indices] = part[name].to_numpy()
        trade_part = pq.read_table(item['trade_statistics_path'])
        if (set(trade_part.column_names)!={'minute_index'}|set(features.TRUSTED_TRADE_SIZE_FIELDS_V1)
                or not np.array_equal(trade_part['minute_index'].to_numpy(),indices)
                or any(trade_part.schema.field(name).type!=pa.uint64() for name in trade_stats)):
            raise RuntimeError('GC_TRADE_STAT_PART_CLOCK_OR_SCHEMA')
        for name in trade_stats:trade_stats[name][indices] = trade_part[name].to_numpy()
    if not np.array_equal(seen,selected>=0):raise RuntimeError('GC_TRUSTED_EVENT_SELECTED_COVERAGE')
    pq.write_table(pa.table({'bar_start_ns':minutes,'available_ns':minutes+timing.MINUTE_NS,
        'contract_index':selected,**trade_stats}),stage/'M1_TRADE_SIZE_STATISTICS_UNMASKED.parquet',compression='zstd')
    original_valid = original['ofi_valid'].to_numpy()
    assert_original_event_subset(events,old_events,original_valid)
    pq.write_table(pa.table({'bar_start_ns':minutes,'available_ns':minutes+timing.MINUTE_NS,'contract_index':selected,**events}),
        stage/'M1_EVENT_DIAGNOSTICS_UNMASKED.parquet',compression='zstd')
    # Validate the original excluded-grid receipt before replacing its admission.
    with tempfile.TemporaryDirectory(prefix='gx1-gc-original-grid-proof-') as temporary:
        stream_completed_footprint(paths['footprint_m1_levels'],minutes,selected,
            {name:restored[name] for name in source.METRICS},old,Path(temporary)/'ORIGINAL_LADDER.parquet')
    ladders = select_cached_ladders(caches,selected,new,stage)
    footprint = stage/'FOOTPRINT_M1_LEVELS_PRE_SOURCE_INCIDENTS.parquet'
    write_footprint(ladders,minutes,selected,{name:restored[name] for name in source.METRICS},new,footprint)
    original_grid_rows = assert_restored_price_grid_original_subset(paths['footprint_m1_levels'],footprint,minutes,old['volume'])
    result = build_derived_timeframes(plan,stage,event_inputs=events,trade_size_inputs=trade_stats,
        restored_inputs={'table':pa.table(restored),'footprint':footprint,'old_volume_valid':old['volume'],'delta':delta})
    quality_bindings.extend(cache_bindings)
    result['source_quality_bindings'].extend(quality_bindings)
    result.update(mode=TRUSTED_RESTORATION_MODE,raw_records_rescanned=records,raw_files_rescanned=files,
        raw_event_passes=1,raw_trade_ladders_rebuilt=False,raw_book_reductions_rebuilt=False,
        recovered_raw_cache_files=len(cache_bindings),original_trade_book_reduction_parity='EXACT_BEFORE_CHANGED_ADMISSION',
        original_019_event_parity='EXACT_ORIGINAL_VALID_SUBSET',
        original_price_grid_rows_verified=original_grid_rows,original_price_grid_parity='EXACT_EVERY_ORIGINAL_MINUTE_PRICE_SIDE_ROW',
        duplicate_signature_diagnostics_preserved=True,
        T_records_deduplicated=False,nontrade_action_semantics_admitted=False,duplicate_only_restoration=delta,
        raw_event_statistics_publication='SELECTED_CONTRACT_ALL_MINUTES_BEFORE_WHOLE_MINUTE_CANDIDATE_MASK',
        trusted_trade_size_statistics='EXACT_UINT64_SIDE_COUNTS_SIZE_SUM_SQUARES_MIN_MAX_BEFORE_CANDIDATE_MASK',
        trusted_trade_size_semantics='T-record population deviation; no parent-order identity; empty raw minimum=0 iff zeroTcount',
        source_file_reports=[f for r in results for f in r['source_files']])
    io._rehash_sources(hashes,metadata_hashes)
    for binding in cache_bindings:io.verify_digest(Path(binding['path']),binding['sha256'],'GC_TRUSTED_RAW_CACHE_POST')
    return result


def bound_quality_inputs(plan):
    paths = {};bindings = {}
    for name in ('source_incidents','retained_registry_source'):
        binding = plan[name];path = io.REPO / binding['path']
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents) or path.resolve(strict=True) != path:
            raise RuntimeError('GC_QUALITY_INPUT_PATH')
        io.verify_digest(path,binding['sha256'],'GC_QUALITY_INPUT')
        paths[name] = path;bindings[name] = {'path':str(path),'sha256':binding['sha256']}
    event = plan['event_reductions'];terminal = event['source_terminal']
    if set(terminal) != {'path','sha256'}:raise RuntimeError('GC_QUALITY_EVENT_TERMINAL_BINDING')
    for name,binding in (('events',event),('events_terminal',terminal)):
        path = Path(binding['path'])
        if (not path.is_absolute() or path.is_symlink() or any(parent.is_symlink() for parent in path.parents)
                or path.resolve(strict=True) != path):raise RuntimeError('GC_QUALITY_EVENT_PATH')
        io.verify_digest(path,binding['sha256'],'GC_QUALITY_EVENT_INPUT')
        paths[name] = path;bindings[name] = {'path':str(path),'sha256':binding['sha256']}
    receipt = io.read_json(paths['events_terminal'])
    artifact = receipt.get('artifacts',{}).get('M1_EVENT_DIAGNOSTICS.parquet',{})
    if (paths['events_terminal'].name != 'TERMINAL.json' or receipt.get('status') != 'COMPLETE'
            or receipt.get('exit_code') != 0 or receipt.get('source_unchanged') is not True
            or receipt.get('test_data_used') is not False
            or paths['events'] != paths['events_terminal'].parent/'M1_EVENT_DIAGNOSTICS.parquet'
            or artifact.get('sha256') != event['sha256'] or artifact.get('bytes') != paths['events'].stat().st_size):
        raise RuntimeError('GC_QUALITY_EVENT_SOURCE_INVENTORY')
    return paths,list(bindings.values())


def load_quality_event_reductions(path, minutes, selected, original_ofi_valid, covered):
    table = pq.read_table(path)
    fields = set(features.EVENT_DIAGNOSTIC_REDUCERS_V1) | {'qualified_covered_ns','qualified_events_per_observed_second'}
    fields |= {prefix+'_mean_ns' for prefix in ('receive_interval',)+features.EVENT_LATENCY_DOMAINS_V1}
    if set(table.column_names) != fields | {'bar_start_ns','available_ns','contract_index'}:
        raise RuntimeError('GC_QUALITY_EVENT_SCHEMA')
    if (table.num_rows != len(minutes) or not np.array_equal(table['bar_start_ns'].to_numpy(),minutes)
            or not np.array_equal(table['available_ns'].to_numpy(),minutes+timing.MINUTE_NS)
            or not np.array_equal(table['contract_index'].to_numpy(),selected)):
        raise RuntimeError('GC_QUALITY_EVENT_M1_CLOCK_OR_CONTRACT')
    events = {name:table[name].to_numpy() for name in fields}
    candidate = np.asarray(original_ofi_valid,bool)
    count = events['qualified_final_event_count']
    if (not np.array_equal(np.isfinite(count),candidate) or np.any(count[candidate] <= 0)
            or not np.array_equal(events['qualified_covered_ns'][candidate],covered[candidate])
            or any(np.isfinite(value[~candidate]).any() or np.isinf(value).any() for value in events.values())):
        raise RuntimeError('GC_QUALITY_EVENT_ORIGINAL_MASK_OR_COVERAGE_PARITY')
    return events


def legacy_strict_m5_surface(minutes, selected, metrics, masks, ohlc, book, tick, profiles):
    """Original001 strict M5 book/profile formula; only source admission changes."""
    if len(minutes)%5:raise RuntimeError('GC_LEGACY_M5_GRID')
    start = minutes[::5];contracts = selected.reshape(-1,5)
    same = (contracts>=0).all(axis=1) & (contracts==contracts[:,:1]).all(axis=1)
    valid = masks['volume'].reshape(-1,5).all(axis=1) & same
    ofi,bbo,order = (book[name].reshape(-1,5).all(axis=1)&same for name in ('ofi_valid','bbo_valid','order_count_valid'))
    out = {'bar_start_ns':start,'available_ns':start+5*timing.MINUTE_NS,'contract_index':contracts[:,0],
        'volume_valid':valid,'signed_valid':masks['delta'].reshape(-1,5).all(axis=1)&same,
        'ofi_valid':ofi,'bbo_valid':bbo,'order_count_valid':order,**profiles}
    for name in ('volume','trades','buy','sell','unknown'):
        value = metrics[name].reshape(-1,5).sum(axis=1).astype(float)
        value[~(out['signed_valid'] if name in ('buy','sell') else valid)] = np.nan
        out[name] = value
    for j,name in enumerate(features.L1_FIELDS_V2):
        raw = book[name].reshape(-1,5)
        value = raw.sum(axis=1) if j<6 else raw.mean(axis=1)
        if name == 'mid_response_ticks':
            value = book['closing_mid_tick'].reshape(-1,5)[:,-1]-book['opening_mid_tick'].reshape(-1,5)[:,0]
        value[~(ofi if j<6 else bbo)] = np.nan
        out[name] = value
    for name in features.L1_ORDER_COUNT_FIELDS_V1:
        value = book[name].reshape(-1,5).mean(axis=1);value[~order] = np.nan;out[name] = value
    out['close_usd'] = np.where(valid,ohlc[4::5,3].astype(float)*tick/features.PRICE_SCALE,np.nan)
    return out


def retained_quality_rebindings(plan, stage):
    """Exact declared legacy columns, formulas, masks and clocks; no name inference."""
    binding = plan['retained_registry_source'];path = io.REPO/binding['path']
    io.verify_digest(path,binding['sha256'],'GC_QUALITY_RETAINED_REGISTRY')
    original = io.read_json(path)
    if original.get('schema_version') != 'gx1_native_gc_context_registry_v1' or original.get('outcomes_accessed') is not False:
        raise RuntimeError('GC_QUALITY_RETAINED_REGISTRY_SCOPE')
    table_map = {'M5_GC_FEATURES.parquet':'M5_GC_FEATURES.parquet'}
    table_map.update({'TRADE_FEATURES_'+tf+'.parquet':'TRADE_FEATURES_'+tf+'.parquet' for tf in DERIVED_TIMEFRAMES})
    table_map.update({'BASIS_'+tf+'.parquet':'BASIS_'+tf+'.parquet' for tf in DERIVED_TIMEFRAMES})
    output_tables = {};proof = []
    for key in sorted(set(item['table'] for item in original['fields'])):
        if key not in table_map:raise RuntimeError('GC_QUALITY_RETAINED_TABLE_NOT_DECLARED')
        old = original['tables'][key];oldpath = Path(old['path']);newpath = stage/table_map[key]
        io.verify_digest(oldpath,old['sha256'],'GC_QUALITY_RETAINED_ORIGINAL')
        oldschema = pq.ParquetFile(oldpath).schema_arrow;newschema = pq.ParquetFile(newpath).schema_arrow
        rows = [item for item in original['fields'] if item['table']==key]
        columns = {'available_ns'}
        for item in rows:
            columns.add(item['column'])
            if item.get('validity_column'):columns.add(item['validity_column'])
            columns.add(item.get('source_clock_column') or 'available_ns')
        for column in columns:
            if column not in oldschema.names or column not in newschema.names or oldschema.field(column).type != newschema.field(column).type:
                raise RuntimeError('GC_QUALITY_RETAINED_COLUMN_DOMAIN_TYPE:'+key+':'+column)
        clocks = ('bar_start_ns','available_ns','contract_index')
        a = pq.read_table(oldpath,columns=list(clocks));b = pq.read_table(newpath,columns=list(clocks))
        if a.num_rows != b.num_rows or any(not np.array_equal(a[name].to_numpy(),b[name].to_numpy()) for name in clocks):
            raise RuntimeError('GC_QUALITY_RETAINED_CLOCK_OR_CONTRACT:'+key)
        target = Path(plan['run_root'])/'RESULTS'/table_map[key]
        output_tables[key] = {'path':str(target),'sha256':io.sha256_file(newpath),'bytes':newpath.stat().st_size}
        proof.append({'table':key,'old_source':old,'new_source':output_tables[key],
            'columns':[{'name':item['name'],'column':item['column'],'owner':item['owner'],
                'validity_column':item.get('validity_column'),'source_clock_column':item.get('source_clock_column') or 'available_ns'} for item in rows],
            'formula_semantics':'Original strict formula and null/observed field semantics; explicit manifest changes signature-only admission and source-incident censoring.',
            'bar_available_contract_clock_parity':'EXACT','observed_recipe_substitution':False})
    return {'schema_version':'gx1_gc_quality_retained_source_rebindings_v1','source_registry':binding,
        'retained_registry':{**original,'tables':output_tables},'table_rebindings':proof,
        'retained_field_count':len(original['fields']),'native_registry_admitted':False}


def completion_event_worker(group, ci, tick, wanted, candidate, expected_metrics, expected_book, stage, *, include_trade_statistics=False):
    """Only the missing event sufficient statistics; no new trade ladder/build."""
    collector = features.L1FeatureCollectorV2(wanted, tick, include_event_diagnostics=True,
        include_trusted_trade_stats=include_trade_statistics)
    result = source.scan_symbol(group['items'], tick, on_batch=collector,
        extra_columns=source.L1_AUDIT_COLUMNS, on_file_start=collector.file_start)
    collector.finish()
    for name in source.METRICS:
        if not np.array_equal(result[name][wanted], expected_metrics[name]):
            raise RuntimeError('MBP1_COMPLETION_ORIGINAL_METRIC_PARITY:' + name)
    values, _ = collector.values(candidate)
    for name in features.L1_FIELDS_V2:
        if not np.array_equal(values[name], expected_book[name], equal_nan=True):
            raise RuntimeError('MBP1_COMPLETION_ORIGINAL_BOOK_PARITY:' + name)
    events = collector.event_diagnostics(candidate)
    output = pa.table({'minute_index': wanted, **events})
    path = stage / (group['raw_symbol'] + '_EVENT_M1.parquet')
    pq.write_table(output, path, compression='zstd')
    report = {'symbol': group['raw_symbol'], 'contract_index': ci, 'path': str(path),
        'source_files': result['files'], 'selected_m1_rows': len(wanted)}
    if include_trade_statistics:
        stats = collector.trusted_trade_statistics()
        count = features.trusted_trade_record_count_v1(stats)
        if not np.array_equal(count,expected_metrics['trades']) or not np.array_equal(stats['size_sum_contracts'],expected_metrics['volume']):
            raise RuntimeError('GC_TRADE_STAT_RAW_RECORD_OR_VOLUME_PARITY')
        trade_path = stage/(group['raw_symbol']+'_TRADE_SIZE_M1.parquet')
        pq.write_table(pa.table({'minute_index':wanted,**stats}),trade_path,compression='zstd')
        report['trade_statistics_path'] = str(trade_path)
    return report


def build_completion(plan, stage):
    """One new raw event pass; existing completed M1/grid/roll remain inputs."""
    paths = bound_derived_inputs(plan)
    base_path = io.REPO / plan['source_plan']['path']
    io.verify_digest(base_path, plan['source_plan']['sha256'], 'MBP1_COMPLETION_SOURCE_PLAN')
    base = io.read_json(base_path)
    groups, hashes = io._bound_gc_inputs(base)
    metadata, metadata_hashes = load_metadata(base, groups)
    roll = io.read_json(paths['roll_schedule'])
    if ([group['raw_symbol'] for group in groups] != roll['contract_order']
            or {key: list(value) for key, value in metadata.items()} != roll['metadata_identity']):
        raise RuntimeError('MBP1_COMPLETION_ORIGINAL_CONTRACT_METADATA')
    table = pq.read_table(paths['m1_inputs'], columns=['contract_index', 'ofi_valid'] + list(source.METRICS) + list(features.L1_FIELDS_V2))
    selected = table['contract_index'].to_numpy().astype(np.int16)
    parts = stage / 'RAW_EVENT_REDUCTIONS';parts.mkdir()
    with ProcessPoolExecutor(max_workers=plan['worker_count'], mp_context=mp.get_context('fork')) as pool:
        jobs = []
        for ci, group in enumerate(groups):
            wanted = np.flatnonzero(selected == ci)
            metrics = {name: table[name].to_numpy()[wanted] for name in source.METRICS}
            book = {name: table[name].to_numpy()[wanted] for name in features.L1_FIELDS_V2}
            jobs.append(pool.submit(completion_event_worker, group, ci, metadata[group['raw_symbol']][2],
                wanted, table['ofi_valid'].to_numpy()[wanted], metrics, book, parts))
        results = [job.result() for job in jobs]
    records = sum(item['records'] for result in results for item in result['source_files'])
    if records != plan['expected_source_records'] or sum(len(result['source_files']) for result in results) != plan['expected_source_files']:
        raise RuntimeError('MBP1_COMPLETION_RECORD_COUNT')
    events = None;seen = np.zeros(len(selected), bool)
    for result in results:
        part = pq.read_table(result['path']);indices = part['minute_index'].to_numpy().astype(np.int64)
        if (seen[indices].any() or not np.array_equal(indices, np.flatnonzero(selected == result['contract_index']))):
            raise RuntimeError('MBP1_COMPLETION_EVENT_CLOCK')
        seen[indices] = True
        if events is None:events = {name: np.full(len(selected), np.nan) for name in part.column_names if name != 'minute_index'}
        for name in events:events[name][indices] = part[name].to_numpy()
    if not np.array_equal(seen, selected >= 0):raise RuntimeError('MBP1_COMPLETION_SELECTED_CLOCK')
    minutes = timing.STUDY_START_NS + np.arange(len(selected), dtype=np.int64) * timing.MINUTE_NS
    pq.write_table(pa.table({'bar_start_ns': minutes, 'available_ns': minutes + timing.MINUTE_NS,
        'contract_index': selected, **events}), stage / 'M1_EVENT_DIAGNOSTICS.parquet', compression='zstd')
    del table
    io._rehash_sources(hashes, metadata_hashes)
    result = build_derived_timeframes(plan, stage, event_inputs=events)
    result.update(mode=COMPLETION_MODE, raw_records_rescanned=records,
        raw_files_rescanned=sum(len(result['source_files']) for result in results),
        raw_event_passes=1, raw_trade_ladders_rebuilt=False, original_trade_and_book_parity='EXACT_SELECTED_M1',
        source_file_reports=[item for result in results for item in result['source_files']],
        native_candidates=features.completion_native_candidates_v1(include_trade_size_stats=False))
    return result


def bound_derived_inputs(plan):
    bindings = plan['inputs']
    if set(bindings) != set(DERIVED_INPUT_NAMES) | {'source_terminal'}:
        raise RuntimeError('DERIVED_GC_INPUT_INVENTORY')
    paths = {}
    for name, binding in bindings.items():
        if set(binding) != {'path', 'sha256'}:
            raise RuntimeError('DERIVED_GC_INPUT_BINDING')
        path = Path(binding['path'])
        if (not path.is_absolute() or path.is_symlink()
                or any(parent.is_symlink() for parent in path.parents)
                or path.resolve(strict=True) != path or not path.is_file()):
            raise RuntimeError('DERIVED_GC_INPUT_PATH')
        io.verify_digest(path, binding['sha256'], 'DERIVED_GC_INPUT')
        paths[name] = path
    terminal = io.read_json(paths['source_terminal'])
    if (paths['source_terminal'].name != 'TERMINAL.json'
            or terminal.get('status') != 'COMPLETE' or terminal.get('exit_code') != 0
            or terminal.get('source_unchanged') is not True
            or terminal.get('test_data_used') is not False):
        raise RuntimeError('DERIVED_GC_SOURCE_NOT_COMPLETE')
    for name, filename in DERIVED_INPUT_NAMES.items():
        entry = terminal['artifacts'].get(filename, {})
        if (paths[name] != paths['source_terminal'].parent / filename
                or entry.get('sha256') != bindings[name]['sha256']
                or entry.get('bytes') != paths[name].stat().st_size):
            raise RuntimeError('DERIVED_GC_SOURCE_INVENTORY')
    return paths


def stream_completed_footprint(path, minutes, selected, metrics, masks, destination, *, admitted_minutes=None):
    """Map bounded price batches to the exact M1 clock and conserve every side."""
    schema = pa.schema([('minute_index', pa.int32()), ('price_tick', pa.int64()),
                        ('buy', pa.uint64()), ('sell', pa.uint64()), ('unknown', pa.uint64())])
    if admitted_minutes is not None:
        admitted_minutes = np.asarray(admitted_minutes)
        if (admitted_minutes.shape != minutes.shape or admitted_minutes.dtype.kind != 'b'
                or np.any(admitted_minutes & ~masks['volume'])):
            raise RuntimeError('DERIVED_GC_FOOTPRINT_ADMISSION_NOT_SUBSET')
    required = {'bar_start_ns', 'contract_index', 'price_tick', 'buy', 'sell', 'unknown', 'signed_valid'}
    parquet = pq.ParquetFile(path)
    if set(parquet.schema_arrow.names) != required:
        raise RuntimeError('DERIVED_GC_FOOTPRINT_SCHEMA')
    seen = np.zeros(len(minutes), bool)
    current = None
    sides = [0, 0, 0]
    rows_seen = 0

    def flush():
        if current is not None:
            if seen[current] or any(sides[j] != int(metrics[k][current])
                                    for j, k in enumerate(('buy', 'sell', 'unknown'))):
                raise RuntimeError('DERIVED_GC_FOOTPRINT_SIDE_CONSERVATION')
            seen[current] = True

    with pq.ParquetWriter(destination, schema, compression='zstd') as writer:
        for batch in parquet.iter_batches(batch_size=65536, use_threads=False):
            values = {name: batch.column(batch.schema.get_field_index(name)).to_numpy(zero_copy_only=False)
                      for name in required}
            indices = np.searchsorted(minutes, values['bar_start_ns'])
            if (np.any(indices >= len(minutes))
                    or not np.array_equal(minutes[np.minimum(indices, len(minutes) - 1)], values['bar_start_ns'])
                    or np.any(selected[indices] != values['contract_index'])
                    or not masks['volume'][indices].all()
                    or not np.array_equal(masks['delta'][indices], values['signed_valid'])
                    or np.any(values['price_tick'] <= 0)):
                raise RuntimeError('DERIVED_GC_FOOTPRINT_M1_MAPPING')
            # Published raw001 rows are ordered by contract and then M1/price,
            # not globally by clock. A repeated M1 group is rejected by seen.
            changes = np.r_[0, np.flatnonzero(np.diff(indices) != 0) + 1, len(indices)]
            for begin, end in zip(changes[:-1], changes[1:]):
                row = int(indices[begin])
                if row != current:
                    flush()
                    current, sides = row, [0, 0, 0]
                for j, name in enumerate(('buy', 'sell', 'unknown')):
                    sides[j] += sum(map(int, values[name][begin:end]))
            output = pa.table({'minute_index': indices.astype(np.int32),
                **{name: values[name] for name in ('price_tick', 'buy', 'sell', 'unknown')}}, schema=schema)
            if admitted_minutes is not None:output = output.filter(pa.array(admitted_minutes[indices]))
            writer.write_table(output)
            rows_seen += len(indices)
        flush()
    if rows_seen != parquet.metadata.num_rows or not np.array_equal(seen, masks['volume']):
        raise RuntimeError('DERIVED_GC_FOOTPRINT_MINUTE_CONSERVATION')
    return rows_seen


def add_completion_basis(plan, surfaces, minutes):
    """Reuse the source plan's bound spot partitions, never fetch vendor data."""
    path = io.REPO / plan['source_plan']['path']
    io.verify_digest(path, plan['source_plan']['sha256'], 'MBP1_COMPLETION_BASIS_SOURCE_PLAN')
    source_plan = io.read_json(path)
    hashes = [(path, plan['source_plan']['sha256'])]
    hashes.extend(add_basis(source_plan, surfaces, minutes, include_observed=True))
    return [{'path': str(path), 'sha256': digest} for path, digest in hashes]


def build_derived_timeframes(plan, stage, *, event_inputs=None, restored_inputs=None, trade_size_inputs=None):
    """Reuse completed M1 bytes; no raw MBP-1 or previous feature-table scan."""
    paths = bound_derived_inputs(plan)
    restoring = plan.get('mode') == TRUSTED_RESTORATION_MODE
    repairing = plan.get('mode') == QUALITY_REDERIVE_MODE or restoring
    if restoring != (restored_inputs is not None):raise RuntimeError('GC_TRUSTED_RESTORED_INPUT_OWNER')
    if restoring != (trade_size_inputs is not None):raise RuntimeError('GC_TRUSTED_TRADE_STAT_INPUT_OWNER')
    quality_paths,quality_bindings = bound_quality_inputs(plan) if repairing else ({},[])
    roll = io.read_json(paths['roll_schedule'])
    ticks = {value[2] for value in roll['metadata_identity'].values()}
    if len(ticks) != 1 or roll['origin_ns'] != timing.STUDY_START_NS:
        raise RuntimeError('DERIVED_GC_SOURCE_TICK')
    tick = ticks.pop()
    table = restored_inputs['table'] if restoring else pq.read_table(paths['m1_inputs'])
    minutes = table['bar_start_ns'].to_numpy(zero_copy_only=False).astype(np.int64)
    expected = np.arange(timing.STUDY_START_NS, timing.WINDOW_END_NS, timing.MINUTE_NS, dtype=np.int64)
    if not np.array_equal(minutes, expected):
        raise RuntimeError('DERIVED_GC_M1_CLOCK')
    selected = table['contract_index'].to_numpy(zero_copy_only=False).astype(np.int16)
    if np.any(selected < -1) or np.any(selected >= len(roll['contract_order'])):
        raise RuntimeError('DERIVED_GC_CONTRACT_INDEX')
    metrics = {name: table[name].to_numpy(zero_copy_only=False).astype(np.uint64) for name in source.METRICS}
    if not np.array_equal(metrics['volume'], metrics['buy'] + metrics['sell'] + metrics['unknown']):
        raise RuntimeError('DERIVED_GC_M1_SIDE_CONSERVATION')
    masks = source.row_masks(selected, metrics, minutes + timing.MINUTE_NS < timing.WINDOW_END_NS,
        reject_duplicate_signature=not restoring)
    for name, stored in (('volume', 'volume_valid'), ('delta', 'signed_valid')):
        if not np.array_equal(masks[name], table[stored].to_numpy(zero_copy_only=False)):
            raise RuntimeError('DERIVED_GC_M1_ADMISSION_MASK')
    ohlc = np.column_stack([table[name + '_tick'].to_numpy(zero_copy_only=False) for name in ('open', 'high', 'low', 'close')])
    book_names = ('ofi_valid', 'bbo_valid', 'covered_ns', 'order_count_valid', 'order_covered_ns',
                  'opening_mid_tick', 'closing_mid_tick') + features.L1_FIELDS_V2 + features.L1_ORDER_COUNT_FIELDS_V1
    book = {name: table[name].to_numpy(zero_copy_only=False) for name in book_names}
    original_masks = {name:value.copy() for name,value in masks.items()}
    source_quality = None
    if repairing:
        if not restoring:
            event_inputs = load_quality_event_reductions(quality_paths['events'],minutes,selected,book['ofi_valid'],book['covered_ns'])
        elif event_inputs is None:raise RuntimeError('GC_TRUSTED_UNMASKED_EVENTS_REQUIRED')
        quality = {name:table['quality_'+name].to_numpy() for name in source.L1_QUALITY_METRICS}
        source_quality = source.source_incident_masks_v1(minutes,selected,quality,io.read_json(quality_paths['source_incidents']),
            reject_duplicate_signature=not restoring)
        receive_excluded = source_quality['receive_clock_excluded']
        state_excluded = source_quality['book_event_state_excluded']
        for name in masks:masks[name] = masks[name] & ~receive_excluded
        for name in ('ofi_valid','bbo_valid','order_count_valid'):book[name] = book[name] & ~state_excluded
        for name,value in event_inputs.items():event_inputs[name] = np.where(book['ofi_valid'],value,np.nan)
        quality_output = {'bar_start_ns':minutes,'available_ns':minutes+timing.MINUTE_NS,'contract_index':selected,
            'receive_clock_excluded':receive_excluded,'book_event_state_excluded':state_excluded,
            'original_volume_valid':original_masks['volume'],'original_signed_valid':original_masks['delta'],
            'volume_valid':masks['volume'],'signed_valid':masks['delta'],
            **{name:book[name] for name in ('ofi_valid','bbo_valid','order_count_valid')}}
        if restoring:
            quality_output['prior_receipt_volume_valid'] = restored_inputs['old_volume_valid']
        pq.write_table(pa.table(quality_output),stage/'M1_SOURCE_QUALITY_MASKS.parquet',compression='zstd')
        # Publish repaired M1 as the sole feature input, retaining original raw
        # metric/quality columns rather than erasing diagnostic evidence.
        repaired = {name:table[name].to_numpy() for name in table.column_names}
        repaired.update(volume_valid=masks['volume'],signed_valid=masks['delta'])
        repaired.update({name:book[name] for name in ('ofi_valid','bbo_valid','order_count_valid')})
        for j,name in enumerate(features.L1_FIELDS_V2+features.L1_ORDER_COUNT_FIELDS_V1):
            valid = book['ofi_valid'] if j<6 else book['bbo_valid'] if j<len(features.L1_FIELDS_V2) else book['order_count_valid']
            book[name] = np.where(valid,book[name],np.nan);repaired[name] = book[name]
        pq.write_table(pa.table(repaired),stage/'M1_GC_INPUTS.parquet',compression='zstd')
        pq.write_table(pa.table({'bar_start_ns':minutes,'available_ns':minutes+timing.MINUTE_NS,
            'contract_index':selected,**event_inputs}),stage/'M1_EVENT_DIAGNOSTICS.parquet',compression='zstd')
        source_footprint = restored_inputs['footprint'] if restoring else paths['footprint_m1_levels']
        # The qualified grid is published separately; pre-censor input is kept as
        # evidence and checked completely before filtering, including lost rows.
        qualified_grid = stage/'FOOTPRINT_M1_LEVELS.parquet'
        writer = pq.ParquetWriter(qualified_grid,pq.ParquetFile(source_footprint).schema_arrow,compression='zstd')
        try:
            for batch in pq.ParquetFile(source_footprint).iter_batches(batch_size=65536):
                part = pa.Table.from_batches([batch]);indices = (part['bar_start_ns'].to_numpy()-minutes[0])//timing.MINUTE_NS
                if np.any((indices<0)|(indices>=len(minutes))):raise RuntimeError('GC_QUALITY_FOOTPRINT_GRID')
                writer.write_table(part.filter(pa.array(masks['volume'][indices])))
        finally:writer.close()
        # Copy the bound roll bytes; contract selection was not recomputed for admission.
        (stage/'ROLL_SCHEDULE.json').write_bytes(paths['roll_schedule'].read_bytes())
    surfaces, minute_maps = {}, {}
    for tf in plan['timeframes']:
        surfaces[tf], minute_maps[tf] = features.timeframe_skeleton(minutes, selected, metrics, masks, ohlc, tf, tick)
    complete = plan.get('mode') == COMPLETION_MODE or repairing
    profiles = features.dynamic_profile_arrays_v2(surfaces['M5']['bar_start_ns']) if complete else None
    strict_profiles = features.dynamic_profile_arrays_v2(surfaces['M5']['bar_start_ns']) if repairing else None
    with tempfile.TemporaryDirectory(prefix='gx1-gc-derived-ladder-') as temporary:
        ladder = Path(temporary) / 'SELECTED_LADDER.parquet'
        footprint_source = restored_inputs['footprint'] if restoring else paths['footprint_m1_levels']
        footprint_rows = stream_completed_footprint(footprint_source, minutes,
            selected, metrics, original_masks, ladder, admitted_minutes=masks['volume'] if repairing else None)
        stream_evidence = features.consume_ladder(ladder, surfaces, minute_maps, tick,
            profile_output=profiles, profile_state_max_bytes=plan['profile_state_max_bytes'] if complete else None,
            strict_profile_output=strict_profiles)
    cvd = features.relative_cvd(minutes, selected, masks, metrics)
    known_cumulative = features.known_cumulative_contributions_v1(minutes, selected, masks, metrics) if complete else None
    spot_basis_sources = add_completion_basis(plan, surfaces, minutes) if complete else []
    support = {}
    for tf, surface in surfaces.items():
        features.add_structure_and_cvd(surface, cvd, minute_maps, minutes)
        features.add_observed_structure(surface)
        if complete:
            features.add_known_cumulative_contributions_v1(surface, known_cumulative, minutes)
            features.add_projected_m5_profiles_v2(surface, surfaces['M5']['available_ns'], profiles,
                surfaces['M5']['observed_coverage_fraction'])
            if event_inputs is None:raise RuntimeError('MBP1_COMPLETION_EVENT_INPUTS_REQUIRED')
            surface.update(features.aggregate_observed_event_diagnostics_v1(minutes, selected, event_inputs, tf))
            if restoring:
                size_values=features.aggregate_observed_trade_size_v1(minutes,selected,metrics,masks,trade_size_inputs,tf)
                total_records=sum(size_values['observed_'+side+'_trade_records'] for side in ('buy','sell','unknown'))
                if (not np.array_equal(np.isfinite(total_records),surface['observed_valid'])
                        or not np.array_equal(total_records,surface['observed_trades'],equal_nan=True)):
                    raise RuntimeError('GC_TRADE_STAT_TF_OBSERVED_DOMAIN_OR_RECORD_PARITY:'+tf)
                surface.update(size_values)
        observed_book = features.aggregate_observed_book_v1(minutes, selected, book, tf)
        for key, values in observed_book.items():
            if key in surface:
                if key not in ('bar_start_ns', 'available_ns', 'contract_index') or not np.array_equal(surface[key], values):
                    raise RuntimeError('DERIVED_GC_BOOK_GRID_COLLISION:' + key)
            else:
                surface[key] = values
        take = surface['observed_valid'][minute_maps[tf]] & masks['volume']
        totals = {name: sum(map(int, metrics[name][take])) for name in ('volume', 'buy', 'sell', 'unknown')}
        emitted = {name: sum(map(int, surface['observed_' + target][surface['observed_valid']]))
            for name, target in (('volume', 'volume'), ('buy', 'buy_known'), ('sell', 'sell_known'), ('unknown', 'unknown'))}
        if totals != emitted:
            raise RuntimeError('DERIVED_GC_TIMEFRAME_SIDE_CONSERVATION:' + tf)
        output = pa.table(surface)
        path = stage / ('TRADE_FEATURES_' + tf + '.parquet')
        pq.write_table(output, path, compression='zstd')
        if not features.exact_feature_readback_v2(output, pq.read_table(path)):
            raise RuntimeError('DERIVED_GC_READBACK:' + tf)
        if repairing:
            basis = pa.table({name:surface[name] for name in ('bar_start_ns','available_ns','contract_index','volume_valid','close_usd','spot_basis_bps')})
            basis_path = stage/('BASIS_'+tf+'.parquet');pq.write_table(basis,basis_path,compression='zstd')
            if not features.exact_feature_readback_v2(basis,pq.read_table(basis_path)):
                raise RuntimeError('GC_QUALITY_BASIS_READBACK:'+tf)
        support[tf] = {'rows': len(surface['bar_start_ns']),
            'mask_support': {name: int(np.count_nonzero(value)) for name, value in surface.items()
                if np.asarray(value).dtype == np.dtype(bool)},
            'observed_accepted_m1_minutes': int(np.count_nonzero(take)),
            'observed_source_side_contracts': totals, 'observed_emitted_side_contracts': emitted,
            'observed_side_conservation': 'EXACT_ACCEPTED_M1_WITHIN_ADMITTED_SAME_CONTRACT_BUCKETS',
            'finite_feature_support': {name: int(np.isfinite(value).sum()) for name, value in surface.items()
                if name not in ('bar_start_ns', 'available_ns', 'contract_index')}}
    retained_rebindings = None
    if repairing:
        legacy = pa.table(legacy_strict_m5_surface(minutes,selected,metrics,masks,ohlc,book,tick,strict_profiles))
        legacy_path = stage/'M5_GC_FEATURES.parquet';pq.write_table(legacy,legacy_path,compression='zstd')
        if not features.exact_feature_readback_v2(legacy,pq.read_table(legacy_path)):
            raise RuntimeError('GC_QUALITY_LEGACY_M5_READBACK')
        retained_rebindings = retained_quality_rebindings(plan,stage)
        quality_bindings.extend({'path':item['old_source']['path'],'sha256':item['old_source']['sha256']}
            for item in retained_rebindings['table_rebindings'])
        io.write_json(stage/'RETAINED_SOURCE_REBINDINGS.json',retained_rebindings)
        bound_quality_inputs(plan)
    # Sources remain immutable throughout derivation, not just at admission.
    bound_derived_inputs(plan)
    result = {'mode': QUALITY_REDERIVE_MODE if repairing else DERIVED_TIMEFRAME_MODE, 'source_inputs': plan['inputs'],
        'raw_records_rescanned': 0, 'prior_feature_tables_reused': False,
        'completed_m1_inputs_reused': True, 'footprint_price_rows': footprint_rows,
        'footprint_side_conservation': 'EXACT_EVERY_VALID_SELECTED_M1', 'timeframes': support,
        'strict_complete_clock_features_preserved': True, 'observed_partial_tape_features_separate': True,
        'profile_stream_evidence': stream_evidence,
        'spot_basis_sources': spot_basis_sources,
        'spot_basis_clock_semantics': 'same observed last M1 interval: GC trade close and XAU bid/ask close mid; available at M1 close, no event-level simultaneity',
        'profile_context_semantics': 'closed M5 physical15/60minute and NYsession snapshots projected to each TF, no backsearch',
        'historical_exchange_calendar_qualified': False, 'native_model_wiring_qualified': False,
        'native_dataset_complete': False, 'native_input_qualification_complete': False,
        'test_data_used': False, 'optimizer_steps': 0}
    if repairing:
        excluded = source_quality['receive_clock_excluded']
        result.update(source_incidents=plan['source_incidents'],source_quality_bindings=quality_bindings,
            source_quality_incidents=source_quality['incidents'],
            receive_clock_excluded_m1_buckets=int(excluded.sum()),
            book_event_state_excluded_m1_buckets=int(source_quality['book_event_state_excluded'].sum()),
            original_raw_metrics_preserved=True,original_price_grid_side_conservation='EXACT_BEFORE_SOURCE_CENSOR',
            source_excluded_original_valid_m1=int(np.count_nonzero(original_masks['volume']&excluded)),
            source_excluded_side_contracts={name:sum(map(int,metrics[name][original_masks['volume']&excluded])) for name in ('volume','buy','sell','unknown')},
            strict_complete_clock_features_preserved=False,strict_feature_formulas_preserved=True,
            strict_source_admission_requalified=True,retained_source_field_count=retained_rebindings['retained_field_count'],
            native_candidates=features.completion_native_candidates_v1(include_trade_size_stats=restoring),nontrade_action_semantics_admitted=False)
        if restoring:
            prior = restored_inputs['old_volume_valid'];added = original_masks['volume']&~prior
            result['restoration_after_incident'] = {
                'duplicate_only_m1_excluded_by_incident':int(np.count_nonzero(added&excluded)),
                'duplicate_only_volume_excluded_by_incident':sum(map(int,metrics['volume'][added&excluded])),
                'net_added_m1_vs_prior_receipt':int(masks['volume'].sum())-int(prior.sum()),
                'net_added_contracts_vs_prior_receipt':sum(map(int,metrics['volume'][masks['volume']]))-sum(map(int,metrics['volume'][prior])),
                'originally_valid_m1_censored':int(np.count_nonzero(prior&excluded)),
                'originally_valid_contracts_censored':sum(map(int,metrics['volume'][prior&excluded]))}
    return result


def run(binding_path):
    began = time.monotonic()
    binding = io.read_json(binding_path)
    plan_path = io.REPO / binding['plan_path']
    io.verify_digest(plan_path, binding['plan_sha256'], 'RAW_REBUILD_PLAN')
    plan = io.read_json(plan_path)
    restoring = trusted_restoration_mode(plan)
    repairing = False if restoring else quality_rederive_mode(plan)
    complete = False if repairing or restoring else completion_mode(plan)
    derived = True if repairing or restoring else False if complete else derived_timeframe_mode(plan)
    root = Path(plan['run_root'])
    if binding_path != root / 'EXECUTION_BINDING.json':
        raise RuntimeError('RAW_REBUILD_BINDING_PATH')
    io.verify_digest(io.REPO / 'NEXT_RUN_POLICY.json', binding['policy_sha256'], 'RAW_REBUILD_POLICY')
    policy = io.read_json(io.REPO / 'NEXT_RUN_POLICY.json')
    gate = policy['current_work']['native_gc_rebuild_20261017'][
        'trusted_record_restoration_stage' if restoring else 'source_quality_rederive_stage' if repairing else 'mbp1_completion_stage' if complete else 'derived_timeframe_stage' if derived else 'raw_stage']
    if (gate['plan_sha256'] != binding['plan_sha256'] or gate['status'] != 'EXECUTION_CLAIMED'
            or gate['claimed_invocations'] != 1 or gate['completed_invocations'] != 0
            or gate['execution_authorized'] is not False or policy['training_enabled'] is not False
            or io.git_head() != binding['source_head']
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=io.REPO)):
        raise RuntimeError('RAW_REBUILD_AUTHORITY')
    if set(binding['code_sha256']) != set(CODE_PATHS):
        raise RuntimeError('RAW_REBUILD_CODE_INVENTORY')
    for path, digest in binding['code_sha256'].items():
        io.verify_digest(io.REPO / path, digest, 'RAW_REBUILD_CODE')
    proof = require_capped_cpu_audit_execution()
    proof['lock'] = require_capped_lock_ancestry()
    stage = root / '.RESULTS.staging'
    stage.mkdir(exist_ok=False)
    io.write_json(root / 'EXECUTION_CLAIM.json', {'invocation': 1, 'source_head': binding['source_head']})
    result = build_trusted_restoration(plan,stage) if restoring else build_completion(plan, stage) if complete else build_derived_timeframes(plan, stage) if derived else build(plan, stage)
    for path, digest in binding['code_sha256'].items():
        io.verify_digest(io.REPO / path, digest, 'RAW_REBUILD_CODE_POST')
    io.verify_digest(io.REPO / 'NEXT_RUN_POLICY.json', binding['policy_sha256'], 'RAW_REBUILD_POLICY_POST')
    if io.git_head() != binding['source_head'] or subprocess.check_output(['git', 'status', '--porcelain'], cwd=io.REPO):
        raise RuntimeError('RAW_REBUILD_SOURCE_CHANGED')
    result.update({'status': ('GC_TRUSTED_RECORDS_RESTORED' if restoring else 'GC_SOURCE_QUALITY_REDERIVED' if repairing else 'MBP1_INFORMATION_COMPLETED' if complete else 'DERIVED_GC_TIMEFRAMES_BUILT' if derived else 'RAW_GC_INPUTS_REBUILT'),
                   'source_head': binding['source_head'],
                   'limits': proof, 'elapsed_seconds': time.monotonic() - began})
    io.write_json(stage / 'RESULT.json', result)
    inventory = {}
    for path in sorted(stage.rglob('*')):
        if path.is_file():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
            inventory[str(path.relative_to(stage))] = {'sha256': io.sha256_file(path), 'bytes': path.stat().st_size}
    io.write_json(stage / 'TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0, 'artifacts': inventory,
        'finished_utc': datetime.now(timezone.utc).isoformat(), 'source_unchanged': True, 'test_data_used': False})
    for name, entry in inventory.items():
        io.verify_digest(stage / name, entry['sha256'], 'RAW_REBUILD_PUBLISH_ARTIFACT')
        if name.endswith('.parquet'):
            pq.ParquetFile(stage / name).metadata
    if {str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file()} != set(inventory) | {'TERMINAL.json'}:
        raise RuntimeError('RAW_REBUILD_PUBLICATION_INVENTORY')
    for directory in ((stage / 'RAW_EVENT_REDUCTIONS',stage/'RESTORED_LADDER_REDUCTIONS',stage) if restoring else (stage / 'RAW_EVENT_REDUCTIONS', stage) if complete else (stage,) if derived else (stage / 'RAW_REDUCTIONS', stage)):
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    for item in result.get('spot_basis_sources', ()):
        io.verify_digest(Path(item['path']), item['sha256'], 'MBP1_COMPLETION_BASIS_PREPUBLISH')
    for item in result.get('source_quality_bindings', ()):
        io.verify_digest(Path(item['path']),item['sha256'],'GC_QUALITY_SOURCE_PREPUBLISH')
    publish_bundle_directory_noreplace(stage, root / 'RESULTS')
    print('NATIVE_GC_TRUSTED_RECORD_RESTORATION_COMPLETE' if restoring else 'NATIVE_GC_SOURCE_QUALITY_REDERIVE_COMPLETE' if repairing else 'NATIVE_MBP1_INFORMATION_COMPLETION_COMPLETE' if complete else 'NATIVE_DERIVED_GC_TIMEFRAMES_COMPLETE' if derived else 'NATIVE_RAW_GC_REBUILD_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.binding)
    except Exception as exc:
        root = args.binding.parent
        if root.is_dir():
            source_unchanged = None
            try:
                binding = io.read_json(args.binding)
                source_unchanged = (io.git_head() == binding['source_head']
                    and not subprocess.check_output(['git', 'status', '--porcelain'], cwd=io.REPO)
                    and all(io.sha256_file(io.REPO / name) == digest
                            for name, digest in binding['code_sha256'].items()))
            except (OSError, RuntimeError, KeyError, TypeError):
                pass
            failure = {'status': 'FAILED_NO_ADMISSION', 'exit_code': 1,
                'error': str(exc), 'source_unchanged': source_unchanged,
                'test_data_used': None,
                'test_access_observation': 'unproved_after_failure',
                'finished_utc': datetime.now(timezone.utc).isoformat()}
            if not (root / 'FAILURE.json').exists():
                io.write_json(root / 'FAILURE.json', failure)
            if not (root / 'TERMINAL.json').exists():
                io.write_json(root / 'TERMINAL.json', failure)
        raise


if __name__ == '__main__':
    main()
