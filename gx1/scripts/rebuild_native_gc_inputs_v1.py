"""Fresh native-input rebuild from declared raw MBP-1, never a prior feature table."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
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
        'source_files': [f for r in results for f in r['files']],
        'trade_feature_support': {tf: {k: int(np.count_nonzero(np.isfinite(v) & (v != 0)))
            for k, v in surf.items() if k not in ('bar_start_ns', 'available_ns', 'contract_index')}
            for tf, surf in surfaces.items()},
        'features_built_from_raw': True, 'prior_feature_values_reused': False,
        'profile_volume_conservation': 'EXACT_ALL_VALID_M5',
        'profile_prefix_rows': cut, 'session_holiday_calendar_qualified': False,
        'native_model_wiring_qualified': False, 'native_dataset_complete': False,
        'test_data_used': False, 'optimizer_steps': 0}


def run(binding_path):
    began = time.monotonic()
    binding = io.read_json(binding_path)
    plan_path = io.REPO / binding['plan_path']
    io.verify_digest(plan_path, binding['plan_sha256'], 'RAW_REBUILD_PLAN')
    plan = io.read_json(plan_path)
    root = Path(plan['run_root'])
    if binding_path != root / 'EXECUTION_BINDING.json':
        raise RuntimeError('RAW_REBUILD_BINDING_PATH')
    io.verify_digest(io.REPO / 'NEXT_RUN_POLICY.json', binding['policy_sha256'], 'RAW_REBUILD_POLICY')
    policy = io.read_json(io.REPO / 'NEXT_RUN_POLICY.json')
    gate = policy['current_work']['native_gc_rebuild_20261017']['raw_stage']
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
    result = build(plan, stage)
    for path, digest in binding['code_sha256'].items():
        io.verify_digest(io.REPO / path, digest, 'RAW_REBUILD_CODE_POST')
    io.verify_digest(io.REPO / 'NEXT_RUN_POLICY.json', binding['policy_sha256'], 'RAW_REBUILD_POLICY_POST')
    if io.git_head() != binding['source_head'] or subprocess.check_output(['git', 'status', '--porcelain'], cwd=io.REPO):
        raise RuntimeError('RAW_REBUILD_SOURCE_CHANGED')
    result.update({'status': 'RAW_GC_INPUTS_REBUILT', 'source_head': binding['source_head'],
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
    publish_bundle_directory_noreplace(stage, root / 'RESULTS')
    print('NATIVE_RAW_GC_REBUILD_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.binding)
    except Exception as exc:
        root = args.binding.parent
        if root.is_dir() and not (root / 'FAILURE.json').exists():
            io.write_json(root / 'FAILURE.json', {'status': 'FAILED_NO_ADMISSION', 'error': str(exc),
                'finished_utc': datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == '__main__':
    main()
