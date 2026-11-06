"""Bounded independent five-field MTF formula check on frozen 2024 TRAIN bars."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path('/home/andre2/src/GX1_CURRENT')
sys.path.insert(0, str(REPO))  # Direct script launch must resolve the project package.

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import lfilter

from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution, require_capped_lock_ancestry
from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace

ROOT = Path('/home/andre2/GX1_RUNS/NATIVE_M5_GENERIC_SCALAR5_NUMERIC_20261106_002')
STAGE = 'generic_scalar5_numeric_20261106_002'
TFS = ('M5', 'M15', 'H1', 'H4', 'D1')
FIELDS = ('rsi14_z48', 'macd_line_atr_change_3', 'macd_line_atr_change_5',
          'range_z20', 'mid_ema200_distance_change_1_atr')
DURATION = {'M5': 300, 'M15': 900, 'H1': 3600, 'H4': 14400, 'D1': 86400}
NS = 1_000_000_000


def require(condition, reason):
    if not bool(condition):
        raise RuntimeError('GENERIC_SCALAR5_NUMERIC_' + reason)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def bound_json(ref):
    path = Path(ref['path'])
    require(path.is_absolute() and path.is_file() and path.resolve() == path
            and sha256(path) == ref['sha256'], 'BOUND_JSON:' + str(path))
    return json.loads(path.read_text())


def publish(path, value):
    data = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def snapshot(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_mode, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def verified(ref, snapshots):
    path = Path(ref['path'])
    require(path.is_absolute() and path.is_file() and path.resolve() == path, 'INPUT_PATH')
    before = snapshot(path)
    require(sha256(path) == ref['sha256'] and snapshot(path) == before, 'INPUT_DIGEST:' + str(path))
    snapshots[str(path)] = before
    return path


def native_prefix(parts, cutoff):
    chunks = {name: [] for name in ('time', 'open', 'high', 'low', 'close')}
    previous = None
    for ref in parts:
        table = pq.read_table(ref['path'], columns=list(chunks), use_threads=False)
        require(not any(table[name].null_count for name in chunks), 'NATIVE_NULL')
        clock = table['time'].cast(pa.int64()).to_numpy()
        keep = int(np.searchsorted(clock, cutoff, side='left'))
        require(keep and np.all(np.diff(clock) > 0) and (previous is None or clock[0] > previous), 'NATIVE_CLOCK')
        previous = int(clock[-1])
        chunks['time'].append(clock[:keep])
        for name in ('open', 'high', 'low', 'close'):
            chunks[name].append(table[name].to_numpy()[:keep])
    result = {name: np.concatenate(values) for name, values in chunks.items()}
    require(len(result['time']) > 1_000_000 and np.all(np.diff(result['time']) > 0)
            and result['time'][-1] < cutoff, 'PREFIX_ORIGIN')
    for name in ('open', 'high', 'low', 'close'):
        require(result[name].dtype == np.float64 and np.isfinite(result[name]).all(), 'PRICE_DOMAIN')
    require(np.all(result['high'] >= np.maximum(result['open'], result['close']))
            and np.all(result['low'] <= np.minimum(result['open'], result['close'])), 'OHLC_DOMAIN')
    return result


def resample(source, tf):
    if tf == 'M5':
        return {key: value for key, value in source.items()}
    duration = DURATION[tf] * NS
    offset = (22 * 3600 * NS) % duration
    labels = (source['time'] - offset) // duration * duration + offset
    stamps, first, count = np.unique(labels, return_index=True, return_counts=True)
    complete = (stamps >= source['time'][0]) & (stamps + duration <= source['time'][-1] + 300 * NS)
    return {'time': stamps[complete], 'high': np.maximum.reduceat(source['high'], first)[complete],
            'low': np.minimum.reduceat(source['low'], first)[complete],
            'close': source['close'][first + count - 1][complete]}


def smooth(source, period, *, wilder):
    """SciPy C recurrence with independent SMA seed, never the producer helper."""
    x = np.asarray(source, dtype=np.float64)
    result = np.full(len(x), np.nan)
    if len(x) < period:
        return result
    seed = float(np.mean(x[:period], dtype=np.float64))
    result[period - 1] = seed
    alpha = 1.0 / period if wilder else 2.0 / (period + 1.0)
    if len(x) > period:
        result[period:], _ = lfilter([alpha], [1.0, -(1.0 - alpha)], x[period:],
                                     zi=[(1.0 - alpha) * seed])
    return result


def reference(bars, positions):
    high, low, close = (bars[name] for name in ('high', 'low', 'close'))
    n = len(close)
    require(n > 220 and np.all((positions >= 220) & (positions < n)), 'REFERENCE_WARMUP')
    tr = high - low
    tr[1:] = np.maximum.reduce((tr[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr = smooth(tr, 14, wilder=True)
    changes = np.diff(close)
    gain = smooth(np.maximum(changes, 0), 14, wilder=True)
    loss = smooth(np.maximum(-changes, 0), 14, wilder=True)
    rsi = np.full(n, np.nan)
    g, ell = gain[13:], loss[13:]
    rsi[14:] = np.where(ell == 0, np.where(g == 0, 50.0, 100.0),
                         np.where(g == 0, 0.0, 100.0 - 100.0 / (1.0 + g / np.where(ell == 0, 1.0, ell))))
    ema12 = smooth(close, 12, wilder=False)
    ema26 = smooth(close, 26, wilder=False)
    macd = (ema12 - ema26) / atr
    mid = (high + low) * 0.5
    mid200 = smooth(mid, 200, wilder=False)
    gap = mid - mid200
    width = high - low
    result = np.empty((len(positions), len(FIELDS)), dtype=np.float64)
    for row, i in enumerate(positions):
        r = rsi[i - 47:i + 1]
        w = width[i - 19:i + 1]
        r_std = float(np.std(r, ddof=0)); w_std = float(np.std(w, ddof=0))
        require(np.isfinite(r_std) and r_std > 0 and np.isfinite(w_std) and w_std > 0
                and atr[i] > 0 and np.isfinite(macd[i - 5:i + 1]).all(), 'REFERENCE_DEFINED')
        result[row] = ((rsi[i] - np.mean(r)) / r_std, macd[i] - macd[i - 3],
                       macd[i] - macd[i - 5], (width[i] - np.mean(w)) / w_std,
                       (gap[i] - gap[i - 1]) / atr[i])
    require(np.isfinite(result).all(), 'REFERENCE_NONFINITE')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'plan-sha256', 'policy-sha256', 'source-head'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    require(not any(name.split('.')[0] in ('torch', 'sklearn', 'joblib') for name in sys.modules), 'MODEL_PRELOAD')
    require(subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip() == 'work/gx1-current'
            and subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == args.source_head
            and not subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True), 'SOURCE_STATE')
    plan_path = Path(args.plan)
    require(plan_path == ROOT / 'PLAN.json', 'PLAN_PATH')
    plan_ref = {'path': str(plan_path), 'sha256': args.plan_sha256}
    policy_ref = {'path': str(REPO / 'NEXT_RUN_POLICY.json'), 'sha256': args.policy_sha256}
    plan = bound_json(plan_ref)
    policy = bound_json(policy_ref)
    require(plan['schema_version'] == 'gx1_generic_scalar5_numeric_plan_v1'
            and plan['stage'] == STAGE and plan['output_dir'] == str(ROOT / 'OUTPUT')
            and plan['timeframes'] == list(TFS) and plan['fields'] == list(FIELDS)
            and plan['train_cutoff_ns'] == 1735689600000000000
            and plan['sample_window_ns'] == [1704067200000000000, 1733011200000000000]
            and plan['tolerance'] == {'atol': 0.00001, 'rtol': 0.00005}
            and plan['wall_seconds'] == 1800 and plan['memory_max'] == '4G'
            and plan['max_invocations'] == 1 and plan['test_data_used'] is False, 'PLAN_CONTRACT')
    gate = policy[STAGE]
    require(gate['status'] == 'EXECUTION_CLAIMED' and gate['plan'] == plan_ref
            and gate['claimed_invocations'] == 1 and gate['completed_invocations'] == 0
            and gate['execution_authorized'] is False and gate['relaunch_allowed'] is False
            and gate['max_invocations'] == 1 and gate['memory_max'] == '4G'
            and gate['wall_seconds'] == 1800 and gate['test_data_used'] is False
            and all(gate[k] is False for k in ('model_forwards_authorized', 'checkpoint_reads_authorized',
                'optimizer_steps_authorized', 'broker_access_allowed'))
            and all(policy[k] is False for k in ('training_enabled', 'native_training_enabled',
                'full_epoch_training_allowed', 'full_val_allowed', 'native_launch_authorized')), 'ONE_USE_GATE')
    sources = plan['sources']
    require(set(sources) == {'gx1/scripts/audit_generic_scalar5_numeric_v1.py',
            'gx1/features/htf_features.py', 'gx1/features/technical_indicators_v1.py',
            'gx1/contracts/gx1_capped_execution_v1.py',
            'gx1/contracts/entry_model_native_bundle_commit_v1.py',
            'scripts/gx1_capped_run.sh'}, 'SOURCE_SET')
    snapshots = {}
    for name, ref in sources.items():
        require(ref['path'] == str(REPO / name), 'SOURCE_PATH')
        verified(ref, snapshots)
    inputs = plan['inputs']
    require(set(inputs) == {'cache_manifest', 'enriched_manifest', 'native_manifest', 'fit_result', 'native_parts', 'cache_arrays'}
            and set(inputs['cache_arrays']) == set(TFS), 'INPUT_SET')
    manifest = bound_json(inputs['cache_manifest'])
    enriched = bound_json(inputs['enriched_manifest'])
    native = bound_json(inputs['native_manifest'])
    fit = bound_json(inputs['fit_result'])
    require(manifest['feature_count'] == 253 and all(manifest['feature_names'].index(name) == j
            for name, j in zip(FIELDS, (230, 231, 232, 234, 238)))
            and fit['fold']['train_cutoff_ns'] == plan['train_cutoff_ns']
            and native['schema_version'] == 'xau_canonical_native_source_v3'
            and native['requested_end_utc_exclusive'] == '2026-07-01T00:00:00+00:00', 'LINEAGE')
    parts = inputs['native_parts']
    lineage = enriched['native_m5_source']
    require(enriched['decision'] == 'PASS' and enriched['timeframe'] == 'M5'
            and manifest['m5_prebuilt_source'] == enriched['output_parquet']
            and manifest['m5_prebuilt_source_sha256'] == enriched['output_parquet_sha256']
            and lineage['manifest_path'] == inputs['native_manifest']['path']
            and lineage['manifest_sha256'] == inputs['native_manifest']['sha256']
            and lineage['root'] == native['out_root'], 'SIDECAR_LINEAGE')
    require(len(parts) == 16 and [Path(ref['path']).parent.name for ref in parts]
            == ['year=' + str(year) for year in range(2009, 2025)]
            and [ref['path'] for ref in parts] == lineage['part_paths'][:16]
            and all(native['year_sha256'][Path(ref['path']).parent.name] == ref['sha256']
                    == lineage['part_sha256'][ref['path']] for ref in parts), 'PART_SET')
    cache_root = Path(inputs['cache_manifest']['path']).parent
    for tf in TFS:
        item = manifest['tfs'][tf]
        refs = inputs['cache_arrays'][tf]
        require(refs == {'timestamps': {'path': str(cache_root / item['ts_npy']), 'sha256': item['ts_npy_sha256']},
                         'features': {'path': str(cache_root / item['feats_npy']), 'sha256': item['feats_npy_sha256']}},
                'CACHE_BINDING')
        chosen = np.asarray(plan['selected_bar_starts_ns'][tf], dtype=np.int64)
        require(len(chosen) == 64 and np.all(np.diff(chosen) > 0)
                and np.all(chosen >= plan['sample_window_ns'][0])
                and np.all(chosen < plan['sample_window_ns'][1])
                and np.all(chosen + DURATION[tf] * NS <= plan['train_cutoff_ns']), 'FROZEN_TRAIN_SAMPLE')
    # Finish physical input verification before creating the one-use claim.
    for ref in [inputs['cache_manifest'], inputs['enriched_manifest'], inputs['native_manifest'],
                inputs['fit_result'], *parts]:
        verified(ref, snapshots)
    for tf in TFS:
        for ref in inputs['cache_arrays'][tf].values():
            verified(ref, snapshots)
    proof = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    require(lock['scope'] == proof['cgroup_path'][len('/sys/fs/cgroup'):], 'CAP_LOCK')
    out = Path(plan['output_dir'])
    require(ROOT.is_dir() and out.parent == ROOT and not out.exists(), 'OUTPUT_SCOPE')
    out.mkdir(exist_ok=False)
    publish(out / 'CLAIM.json', {'stage': STAGE, 'plan': plan_ref, 'policy': policy_ref,
                                 'source_head': args.source_head, 'capped_execution': proof, 'lock': lock})
    started = time.monotonic()
    previous_alarm = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(RuntimeError('GENERIC_SCALAR5_NUMERIC_WALL_LIMIT')))
    signal.alarm(plan['wall_seconds'])
    staging = out / '.OUTPUT.staging'
    try:
        staging.mkdir(exist_ok=False)
        source = native_prefix(parts, plan['train_cutoff_ns'])
        reports = {}
        indices = [manifest['feature_names'].index(name) for name in FIELDS]
        for tf in TFS:
            bars = resample(source, tf)
            stamps = np.asarray(plan['selected_bar_starts_ns'][tf], dtype=np.int64)
            positions = np.searchsorted(bars['time'], stamps)
            require(np.all(positions < len(bars['time'])) and np.array_equal(bars['time'][positions], stamps), 'NATIVE_SAMPLE_AXIS:' + tf)
            refs = inputs['cache_arrays'][tf]
            cache_clock = np.load(refs['timestamps']['path'], mmap_mode='r', allow_pickle=False)
            cache = np.load(refs['features']['path'], mmap_mode='r', allow_pickle=False)
            at = np.searchsorted(cache_clock, stamps)
            require(np.all(at < len(cache_clock)) and np.array_equal(cache_clock[at], stamps)
                    and cache.dtype == np.float32 and cache.shape == (manifest['tfs'][tf]['n_bars'], 253), 'CACHE_SAMPLE_AXIS:' + tf)
            expected = reference(bars, positions)
            actual = np.asarray(cache[np.ix_(at, indices)], dtype=np.float64)
            require(np.isfinite(actual).all(), 'CACHE_NONFINITE:' + tf)
            delta = np.abs(actual - expected)
            tolerance = plan['tolerance']['atol'] + plan['tolerance']['rtol'] * np.abs(expected)
            passed = delta <= tolerance
            reports[tf] = {'sampled_train_bars': len(stamps), 'earliest_bar_start_ns': int(stamps[0]),
                           'latest_bar_start_ns': int(stamps[-1]), 'source_prefix_bars': len(bars['time']),
                           'fields': {name: {'max_abs_error': float(delta[:, i].max()),
                                            'max_allowed_error': float(tolerance[:, i].max()),
                                            'failed_rows': int((~passed[:, i]).sum())}
                                      for i, name in enumerate(FIELDS)}}
            del bars, cache, cache_clock
        require(all(snapshot(path) == before for path, before in snapshots.items()), 'POST_INPUT_CHANGE')
        require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == args.source_head
                and not subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True), 'POST_SOURCE_CHANGE')
        all_pass = all(field['failed_rows'] == 0 for row in reports.values() for field in row['fields'].values())
        result = {'schema_version': 'gx1_generic_scalar5_numeric_result_v1', 'stage': STAGE,
                  'status': 'COMPLETE', 'all_sampled_within_prebound_tolerance': all_pass,
                  'timeframes': reports, 'source_m5_rows': len(source['time']),
                  'sampled_train_values': len(TFS) * len(FIELDS) * 64,
                  'reference': 'independent NumPy/SciPy native-prefix formulas; production indicator code not imported',
                  'limits': 'Fixed 2024 TRAIN sample only; not exhaustive cache qualification, learned usefulness, generalization or economic edge.',
                  'model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False}
        publish(staging / 'RESULT.json', result)
        publish(staging / 'TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0, 'source_unchanged': True,
                                           'inputs_unchanged': True, 'elapsed_seconds': time.monotonic() - started,
                                           'test_data_used': False})
        publish_bundle_directory_noreplace(staging, out / 'RESULTS')
        publish(out / 'TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0,
                                       'result': str(out / 'RESULTS/RESULT.json')})
        return 0
    except Exception as exc:
        publish(out / 'FAILED_TERMINAL.json', {'status': 'FAILED', 'exit_code': 1,
                                               'reason': type(exc).__name__ + ': ' + str(exc),
                                               'relaunch_allowed': False})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)


if __name__ == '__main__':
    raise SystemExit(main())
