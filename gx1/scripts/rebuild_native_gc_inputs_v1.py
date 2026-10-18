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

CODE_PATHS = (
    'gx1/scripts/rebuild_native_gc_inputs_v1.py',
    'gx1/features/gc_order_flow_v1.py',
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


def l1_masks(quality, selected, covered):
    clean = (quality['records'] > 0) & (quality['clean_complete_bbo'] > 0)
    for name in ('bad_bbo', 'gap', 'bad_recv', 'snapshot', 'reset',
                 'duplicate_trade_signature', 'file_seam'):
        clean &= quality[name] == 0
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
    out = {'bar_start_ns': start, 'available_ns': start + 5 * timing.MINUTE_NS,
           'contract_index': contracts[:, 0], 'volume_valid': valid,
           'signed_valid': masks['delta'].reshape(-1, 5).all(axis=1) & same,
           'ofi_valid': ofi.reshape(-1, 5).all(axis=1) & same,
           'bbo_valid': bbo.reshape(-1, 5).all(axis=1) & same,
           'order_count_valid': order_valid.reshape(-1, 5).all(axis=1) & same, **levels}
    for k in ('volume', 'trades', 'buy', 'sell', 'unknown'):
        out[k] = metrics[k].reshape(-1, 5).sum(axis=1).astype(float)
        out[k][~(out['signed_valid'] if k in ('buy', 'sell') else valid)] = np.nan
    for j, k in enumerate(features.L1_FIELDS_V2):
        a = book[k].reshape(-1, 5)
        v = a.sum(axis=1) if j < 6 else a.mean(axis=1)
        if k == 'mid_response_ticks':
            v = last.reshape(-1, 5)[:, -1] - first.reshape(-1, 5)[:, 0]
        v[~out['ofi_valid' if j < 6 else 'bbo_valid']] = np.nan
        out[k] = v
    for k in features.L1_ORDER_COUNT_FIELDS_V1:
        v = book[k].reshape(-1, 5).mean(axis=1)
        v[~out['order_count_valid']] = np.nan
        out[k] = v
    out['close_usd'] = np.where(valid, close, np.nan)
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


def stream_completed_footprint(path, minutes, selected, metrics, masks, destination):
    """Map bounded price batches to the exact M1 clock and conserve every side."""
    schema = pa.schema([('minute_index', pa.int32()), ('price_tick', pa.int64()),
                        ('buy', pa.uint64()), ('sell', pa.uint64()), ('unknown', pa.uint64())])
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
            writer.write_table(pa.table({'minute_index': indices.astype(np.int32),
                **{name: values[name] for name in ('price_tick', 'buy', 'sell', 'unknown')}}, schema=schema))
            rows_seen += len(indices)
        flush()
    if rows_seen != parquet.metadata.num_rows or not np.array_equal(seen, masks['volume']):
        raise RuntimeError('DERIVED_GC_FOOTPRINT_MINUTE_CONSERVATION')
    return rows_seen


def build_derived_timeframes(plan, stage):
    """Reuse completed M1 bytes; no raw MBP-1 or previous feature-table scan."""
    paths = bound_derived_inputs(plan)
    roll = io.read_json(paths['roll_schedule'])
    ticks = {value[2] for value in roll['metadata_identity'].values()}
    if len(ticks) != 1 or roll['origin_ns'] != timing.STUDY_START_NS:
        raise RuntimeError('DERIVED_GC_SOURCE_TICK')
    tick = ticks.pop()
    table = pq.read_table(paths['m1_inputs'])
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
    masks = source.row_masks(selected, metrics, minutes + timing.MINUTE_NS < timing.WINDOW_END_NS)
    for name, stored in (('volume', 'volume_valid'), ('delta', 'signed_valid')):
        if not np.array_equal(masks[name], table[stored].to_numpy(zero_copy_only=False)):
            raise RuntimeError('DERIVED_GC_M1_ADMISSION_MASK')
    ohlc = np.column_stack([table[name + '_tick'].to_numpy(zero_copy_only=False) for name in ('open', 'high', 'low', 'close')])
    book_names = ('ofi_valid', 'bbo_valid', 'covered_ns', 'order_count_valid', 'order_covered_ns',
                  'opening_mid_tick', 'closing_mid_tick') + features.L1_FIELDS_V2 + features.L1_ORDER_COUNT_FIELDS_V1
    book = {name: table[name].to_numpy(zero_copy_only=False) for name in book_names}
    surfaces, minute_maps = {}, {}
    for tf in plan['timeframes']:
        surfaces[tf], minute_maps[tf] = features.timeframe_skeleton(minutes, selected, metrics, masks, ohlc, tf, tick)
    with tempfile.TemporaryDirectory(prefix='gx1-gc-derived-ladder-') as temporary:
        ladder = Path(temporary) / 'SELECTED_LADDER.parquet'
        footprint_rows = stream_completed_footprint(paths['footprint_m1_levels'], minutes,
            selected, metrics, masks, ladder)
        features.consume_ladder(ladder, surfaces, minute_maps, tick)
    cvd = features.relative_cvd(minutes, selected, masks, metrics)
    support = {}
    for tf, surface in surfaces.items():
        features.add_structure_and_cvd(surface, cvd, minute_maps, minutes)
        features.add_observed_structure(surface)
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
        support[tf] = {'rows': len(surface['bar_start_ns']),
            'mask_support': {name: int(np.count_nonzero(value)) for name, value in surface.items()
                if np.asarray(value).dtype == np.dtype(bool)},
            'observed_accepted_m1_minutes': int(np.count_nonzero(take)),
            'observed_source_side_contracts': totals, 'observed_emitted_side_contracts': emitted,
            'observed_side_conservation': 'EXACT_ACCEPTED_M1_WITHIN_ADMITTED_SAME_CONTRACT_BUCKETS',
            'finite_feature_support': {name: int(np.isfinite(value).sum()) for name, value in surface.items()
                if name not in ('bar_start_ns', 'available_ns', 'contract_index')}}
    # Sources remain immutable throughout derivation, not just at admission.
    bound_derived_inputs(plan)
    return {'mode': DERIVED_TIMEFRAME_MODE, 'source_inputs': plan['inputs'],
        'raw_records_rescanned': 0, 'prior_feature_tables_reused': False,
        'completed_m1_inputs_reused': True, 'footprint_price_rows': footprint_rows,
        'footprint_side_conservation': 'EXACT_EVERY_VALID_SELECTED_M1', 'timeframes': support,
        'strict_complete_clock_features_preserved': True, 'observed_partial_tape_features_separate': True,
        'historical_exchange_calendar_qualified': False, 'native_model_wiring_qualified': False,
        'native_dataset_complete': False, 'native_input_qualification_complete': False,
        'test_data_used': False, 'optimizer_steps': 0}


def run(binding_path):
    began = time.monotonic()
    binding = io.read_json(binding_path)
    plan_path = io.REPO / binding['plan_path']
    io.verify_digest(plan_path, binding['plan_sha256'], 'RAW_REBUILD_PLAN')
    plan = io.read_json(plan_path)
    derived = derived_timeframe_mode(plan)
    root = Path(plan['run_root'])
    if binding_path != root / 'EXECUTION_BINDING.json':
        raise RuntimeError('RAW_REBUILD_BINDING_PATH')
    io.verify_digest(io.REPO / 'NEXT_RUN_POLICY.json', binding['policy_sha256'], 'RAW_REBUILD_POLICY')
    policy = io.read_json(io.REPO / 'NEXT_RUN_POLICY.json')
    gate = policy['current_work']['native_gc_rebuild_20261017'][
        'derived_timeframe_stage' if derived else 'raw_stage']
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
    result = build_derived_timeframes(plan, stage) if derived else build(plan, stage)
    for path, digest in binding['code_sha256'].items():
        io.verify_digest(io.REPO / path, digest, 'RAW_REBUILD_CODE_POST')
    io.verify_digest(io.REPO / 'NEXT_RUN_POLICY.json', binding['policy_sha256'], 'RAW_REBUILD_POLICY_POST')
    if io.git_head() != binding['source_head'] or subprocess.check_output(['git', 'status', '--porcelain'], cwd=io.REPO):
        raise RuntimeError('RAW_REBUILD_SOURCE_CHANGED')
    result.update({'status': ('DERIVED_GC_TIMEFRAMES_BUILT' if derived else 'RAW_GC_INPUTS_REBUILT'),
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
    for directory in ((stage,) if derived else (stage / 'RAW_REDUCTIONS', stage)):
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    publish_bundle_directory_noreplace(stage, root / 'RESULTS')
    print('NATIVE_DERIVED_GC_TIMEFRAMES_COMPLETE' if derived else 'NATIVE_RAW_GC_REBUILD_COMPLETE', flush=True)


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
