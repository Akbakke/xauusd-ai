"""Materialize one additive EMA-spread field on the frozen cold FIT/later clocks.

Old signal and MTF matrices stay immutable. Only source bars before the fixed
2024-12-31 cutoff enter the calculation; selected rows are the already
qualified Fold2 anchor-span physical union (including all history windows).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from gx1.features import entry_specialist_feature_groups_v1 as owners
from gx1.features import htf_features as htf
from gx1.features.technical_indicators_v1 import ema50_200_spread_atr_block
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution
from gx1.scripts.prebuild_multi_tf_cache_v4 import (
    _fsync_directory, _rename_dir_noreplace,
)

BASE = Path('/home/andre2/GX1_DATA/data/data/prebuilt/XAU_GC_NATIVE_26M_20261018_003')
SOURCE = BASE / 'M5_ENRICHED.parquet'
LOCAL = BASE / 'SURFACES_050/M5_FEATURE_BASE_050.parquet'
LOCAL_MANIFEST = Path(str(LOCAL) + '.manifest.json')
LOCAL_SIGNAL = Path('/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_NATIVE_PREFIX_NORMALIZATION_20261024_261/SUPPORT_SCRATCH/source050/signal.mmap')
CACHE = BASE / 'MULTI_TF_V4_CACHE'
CACHE_MANIFEST = CACHE / 'manifest.json'
SPANS = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_FOLD2_SPAN_REUSE_20261113_002')
SPAN_RESULT = SPANS / 'RESULT.json'
PREFIX_SPANS = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_TRENDLINE_ANCHOR_SPAN_INPUTS_20261112_001')
PREFIX_RESULT = PREFIX_SPANS / 'RESULT.json'
CUTOFF_NS = 1735603200000000000
LANES = ('local', 'M5', 'M15', 'H1', 'H4', 'D1')
REPO = Path('/home/andre2/src/GX1_CURRENT')
STAGE = 'ta_m5_ema_delta_sidecars_20261117_001'
ROOT = Path('/home/andre2/GX1_RUNS/TA_M5_EMA_DELTA_SIDECARS_20261117_001')
PLAN = REPO / 'configs/research/TA_M5_EMA_DELTA_SIDECARS_20261117_001.json'
POLICY = REPO / 'NEXT_RUN_POLICY.json'
WALL_SECONDS = 1800
LOCAL_FIELD = 'chart.local_ema50_200_spread_delta_5_atr'
MTF_FIELD = 'ema50_200_spread_delta_5_atr'


def _need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError('TA_EMA_DELTA_' + reason)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _ref(path: Path) -> dict:
    _need(path.is_file() and not path.is_symlink(), 'ARTIFACT:' + str(path))
    return {'path': str(path), 'sha256': _sha_file(path)}


def _array_sha(value: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=dtype).tobytes()).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _read_prefix(path: Path, columns: list[str]) -> pd.DataFrame:
    # Arrow prunes later row groups. The output contains no cutoff-or-later row.
    cutoff = pd.Timestamp(CUTOFF_NS, tz='UTC').to_pydatetime()
    table = ds.dataset(path, format='parquet').to_table(
        columns=columns, filter=ds.field('time') < cutoff,
    )
    frame = table.to_pandas()
    times = pd.DatetimeIndex(pd.to_datetime(frame.pop('time'), utc=True))
    _need(len(times) > 0 and times.is_unique and times.is_monotonic_increasing
          and int(times[-1].value) < CUTOFF_NS, 'SOURCE_PREFIX_CLOCK')
    frame.index = times
    return frame


def _delta_and_slopes(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    block = ema50_200_spread_atr_block(
        frame['high'], frame['low'], frame['close'],
    )
    denom = block['atr14_positive']
    delta = (block['spread'] - block['spread'].shift(5)) / denom
    fast = (block['ema50'] - block['ema50'].shift(5)) / denom
    slow = (block['ema200'] - block['ema200'].shift(20)) / denom
    return tuple(series.to_numpy(dtype=np.float32) for series in (delta, fast, slow))


def _checked_names(local_manifest: dict, cache_manifest: dict) -> tuple[list[str], list[str]]:
    local_old = list(local_manifest['feature_field_order'])
    mtf_old = list(cache_manifest['feature_names'])
    current_local = list(owners.compile_full_ta_entry_execution_profile()['source_field_names']['signal'])
    current_mtf = list(htf.MULTI_TF_PER_BAR_FEATURES_V4)
    spans_local = {'chart.geomline_above_anchor_span_bars', 'chart.geomline_below_anchor_span_bars'}
    spans_mtf = {'geomline_above_anchor_span_bars', 'geomline_below_anchor_span_bars'}
    _need(len(local_old) == 2725 and len(mtf_old) == 253
          and [name for name in current_local if name not in spans_local | {LOCAL_FIELD}] == local_old
          and [name for name in current_mtf if name not in spans_mtf | {MTF_FIELD}] == mtf_old
          and len(current_local) == len(local_old) + 3
          and len(current_mtf) == len(mtf_old) + 3,
          'EXACT_ADDITIVE_NAMES')
    return local_old, mtf_old


def _span_payloads(result: dict) -> tuple[dict, dict]:
    _need(result.get('status') == 'COMPLETE' and result.get('test_data_used') is False,
          'SPAN_RESULT')
    refs, payloads = {}, {}
    for lane in LANES:
        binding = result['surfaces'][lane]['artifact']
        path = Path(binding['path'])
        _need(path == SPANS / (lane + '_ANCHOR_SPANS.npz')
              and _ref(path) == binding, 'SPAN_REF:' + lane)
        with np.load(path, allow_pickle=False) as archive:
            _need(set(archive.files) == {'row_indices', 'times_ns', 'native_bar_positions',
                                         'anchor_span_bars'}, 'SPAN_MEMBERS:' + lane)
            rows, times = archive['row_indices'], archive['times_ns']
        _need(rows.dtype == times.dtype == np.int64
              and rows.ndim == times.ndim == 1 and rows.shape == times.shape
              and len(rows) > 0 and np.all(np.diff(rows) == 1)
              and np.all(np.diff(times) > 0) and int(times[-1]) < CUTOFF_NS,
              'SPAN_AXIS:' + lane)
        refs[lane], payloads[lane] = binding, (rows, times)
    return refs, payloads


def _check_prefix_spans(payloads: dict) -> None:
    result = _read_json(PREFIX_RESULT)
    _need(result.get('status') == 'COMPLETE_INPUT_EXTENSION_ONLY'
          and result.get('test_data_used') is False, 'PREFIX_SPAN_RESULT')
    for lane in LANES:
        binding = result['surfaces'][lane]['artifact']
        path = Path(binding['path'])
        _need(path == PREFIX_SPANS / (lane + '_ANCHOR_SPANS.npz')
              and _ref(path) == binding, 'PREFIX_SPAN_REF:' + lane)
        with np.load(path, allow_pickle=False) as archive:
            old_rows, old_times = archive['row_indices'], archive['times_ns']
        rows, times = payloads[lane]
        positions = np.searchsorted(rows, old_rows)
        _need(np.all(positions < len(rows))
              and np.array_equal(rows[positions], old_rows)
              and np.array_equal(times[positions], old_times),
              'PREFIX_SPAN_SUBSET:' + lane)


def _check_selected(
    lane: str, rows: np.ndarray, times: np.ndarray, clock: np.ndarray,
    delta: np.ndarray, fast: np.ndarray, slow: np.ndarray,
    old_fast: np.ndarray, old_slow: np.ndarray,
) -> np.ndarray:
    _need(np.all(rows >= 0) and np.all(rows < len(clock))
          and np.all(rows < len(delta))
          and np.array_equal(clock[rows], times), 'CLOCK:' + lane)
    at = np.searchsorted(clock, times)
    _need(np.array_equal(at, rows), 'ROW_CLOCK:' + lane)
    chosen = np.ascontiguousarray(delta[rows], dtype=np.float32)
    _need(np.isfinite(chosen).all()
          and np.array_equal(fast[rows], old_fast, equal_nan=False)
          and np.array_equal(slow[rows], old_slow, equal_nan=False),
          'OLD_SLOPE_PARITY:' + lane)
    return chosen


def _write_json(path: Path, value: dict) -> None:
    with path.open('xb') as handle:
        handle.write((json.dumps(value, indent=2, allow_nan=False) + '\n').encode())
        handle.flush()
        os.fsync(handle.fileno())


def materialize(out_dir: Path) -> dict:
    _need(SOURCE.is_file() and not SOURCE.is_symlink()
          and LOCAL.is_file() and not LOCAL.is_symlink()
          and LOCAL_SIGNAL.is_file() and not LOCAL_SIGNAL.is_symlink(),
          'FROZEN_SOURCE_FILES')
    _need(out_dir.is_absolute() and out_dir.parent.is_dir()
          and not out_dir.exists() and not out_dir.is_symlink(),
          'OUTPUT')
    local_ref, cache_ref, span_ref = (_ref(path) for path in
                                      (LOCAL_MANIFEST, CACHE_MANIFEST, SPAN_RESULT))
    local_manifest, cache_manifest, span_result = (
        _read_json(path) for path in (LOCAL_MANIFEST, CACHE_MANIFEST, SPAN_RESULT)
    )
    _need(local_manifest['source_parquet'] == cache_manifest['m5_prebuilt_source'] == str(SOURCE)
          and local_manifest['source_sha256'] == cache_manifest['m5_prebuilt_source_sha256']
          and local_manifest['source_sha256']
              == '5e5cb256fc60a5e9a7ffe1a161aa875660f01806b8e8de2bf601fc2f528e7f99',
          'FROZEN_SOURCE')
    local_names, mtf_names = _checked_names(local_manifest, cache_manifest)
    span_refs, payloads = _span_payloads(span_result)
    _check_prefix_spans(payloads)

    raw = _read_prefix(SOURCE, ['time', 'open', 'high', 'low', 'close', 'volume'])
    _need(np.isfinite(raw[['open', 'high', 'low', 'close', 'volume']].to_numpy(dtype=np.float64)).all()
          and (raw[['open', 'high', 'low', 'close']].to_numpy(dtype=np.float64) > 0).all(),
          'SOURCE_OHLC')
    source_clock = raw.index.asi8
    digest = hashlib.sha256()
    for name in ('open', 'high', 'low', 'close'):
        digest.update(np.ascontiguousarray(raw[name].to_numpy(dtype='<f8')).tobytes())
    digest.update(np.ascontiguousarray(raw['volume'].to_numpy(dtype='<i8')).tobytes())
    source_binding = {
        'path': str(SOURCE),
        'sha256': local_manifest['source_sha256'],
        'prefix_rows': len(raw),
        'prefix_time_sha256': _array_sha(source_clock, '<i8'),
        'prefix_ohlcv_sha256': digest.hexdigest(),
    }
    staging = Path(tempfile.mkdtemp(prefix='.' + out_dir.name + '.staging.', dir=out_dir.parent))
    published = False
    surfaces = {}
    try:
        local_clock_frame = _read_prefix(LOCAL, ['time'])
        local_clock = local_clock_frame.index.asi8
        rows, times = payloads['local']
        _need(int(rows[-1]) < len(local_clock)
              and np.array_equal(local_clock[rows], times), 'LOCAL_BASE_CLOCK')
        raw_delta, raw_fast, raw_slow = _delta_and_slopes(raw)
        local_at = np.searchsorted(source_clock, local_clock)
        _need(np.all(local_at < len(source_clock))
              and np.array_equal(source_clock[local_at], local_clock), 'LOCAL_SOURCE_CLOCK')
        _need(LOCAL_SIGNAL.stat().st_size == len(local_manifest['feature_field_order'])
              * local_manifest['rows'] * 4, 'LOCAL_OLD_MATRIX_SIZE')
        old_local = np.memmap(LOCAL_SIGNAL, dtype=np.float32, mode='r',
                              shape=(local_manifest['rows'], len(local_names)))
        selected = _check_selected(
            'local', rows, times, local_clock,
            raw_delta[local_at], raw_fast[local_at], raw_slow[local_at],
            np.asarray(old_local[rows, local_names.index('chart.local_ema50_slope_atr')]),
            np.asarray(old_local[rows, local_names.index('chart.local_ema200_slope_atr')]),
        )
        lane_values = {'local': selected}
        del old_local, raw_delta, raw_fast, raw_slow
        mtf_raw = raw.copy()
        for name in ('open', 'high', 'low', 'close', 'volume'):
            mtf_raw[name] = mtf_raw[name].astype(np.float32)
        expected = htf.build_multi_tf_v4_closed_timestamp_indices(mtf_raw.index)
        for lane in LANES[1:]:
            frame = htf._resample_ohlcv(mtf_raw, lane).dropna(subset=['open', 'high', 'low', 'close'])
            frame = frame.loc[expected[lane]]
            delta, fast, slow = _delta_and_slopes(frame)
            clock = np.load(CACHE / (lane + '_ts.npy'), mmap_mode='r', allow_pickle=False)
            old = np.load(CACHE / (lane + '_feats.npy'), mmap_mode='r', allow_pickle=False)
            _need(old.shape == (len(clock), len(mtf_names))
                  and np.array_equal(clock[:len(frame)], frame.index.asi8),
                  'MTF_PREFIX_CLOCK:' + lane)
            rows, times = payloads[lane]
            lane_values[lane] = _check_selected(
                lane, rows, times, clock, delta, fast, slow,
                np.asarray(old[rows, mtf_names.index('ema50_slope_atr')]),
                np.asarray(old[rows, mtf_names.index('ema200_slope_atr')]),
            )
        for lane in LANES:
            rows, times = payloads[lane]
            values = lane_values[lane]
            path = staging / (lane + '_EMA_DELTA.npz')
            with path.open('xb') as handle:
                np.savez_compressed(handle, row_indices=rows, times_ns=times,
                                    delta_5_atr=values)
                handle.flush()
                os.fsync(handle.fileno())
            surfaces[lane] = {
                'artifact': {'path': str(out_dir / path.name), 'sha256': _sha_file(path)},
                'span_artifact': span_refs[lane],
                'row_indices_sha256': _array_sha(rows, '<i8'),
                'times_ns_sha256': _array_sha(times, '<i8'),
                'delta_5_atr_sha256': _array_sha(values, '<f4'),
                'rows': len(rows),
                'first_time_ns': int(times[0]),
                'last_time_ns': int(times[-1]),
                'old_slope_parity_rows': len(rows),
            }
        manifest = {
            'schema_version': 'entry_ta_ema_delta_sidecars_v1',
            'status': 'COMPLETE',
            'cutoff_ns': CUTOFF_NS,
            'source': source_binding,
            'local_surface_manifest': local_ref,
            'mtf_cache_manifest': cache_ref,
            'span_reuse_result': span_ref,
            'fields': {'local': LOCAL_FIELD, 'mtf': MTF_FIELD},
            'surfaces': surfaces,
            'test_data_used': False,
            'model_forwards': 0,
            'optimizer_steps': 0,
        }
        manifest_path = staging / 'MANIFEST.json'
        _write_json(manifest_path, manifest)
        result = {
            'schema_version': 'entry_ta_ema_delta_materialization_result_v1',
            'status': 'COMPLETE',
            'manifest': {'path': str(out_dir / 'MANIFEST.json'),
                         'sha256': _sha_file(manifest_path)},
            'surfaces': surfaces,
            'source': source_binding,
            'cutoff_ns': CUTOFF_NS,
            'fit_rows': 29574,
            'later_parent_start': 29849,
            'later_rows': 17489,
            'test_data_used': False,
            'model_forwards': 0,
            'optimizer_steps': 0,
        }
        _write_json(staging / 'RESULT.json', result)
        _fsync_directory(staging)
        _rename_dir_noreplace(staging, out_dir)
        published = True
        _fsync_directory(out_dir.parent)
        return result
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def _stat(path: Path) -> dict:
    _need(path.is_file() and not path.is_symlink(), 'SOURCE_STAT')
    value = path.stat()
    return {'size': value.st_size, 'mtime_ns': value.st_mtime_ns,
            'inode': value.st_ino, 'device': value.st_dev}


def build_plan() -> dict:
    """Deterministic one-use plan; the integrator publishes and admits it."""
    local = _read_json(LOCAL_MANIFEST)
    cache = _read_json(CACHE_MANIFEST)
    _checked_names(local, cache)
    _need(local['source_sha256'] == cache['m5_prebuilt_source_sha256']
          and local['source_parquet'] == cache['m5_prebuilt_source'] == str(SOURCE),
          'PLAN_SOURCE')
    span = _read_json(SPAN_RESULT)
    inputs = {
        'local_surface_manifest': _ref(LOCAL_MANIFEST),
        'mtf_cache_manifest': _ref(CACHE_MANIFEST),
        'span_reuse_result': _ref(SPAN_RESULT),
        'prefix_span_result': _ref(PREFIX_RESULT),
        **{lane + '_span': _ref(Path(span['surfaces'][lane]['artifact']['path']))
           for lane in LANES},
    }
    _need(all(inputs[lane + '_span'] == span['surfaces'][lane]['artifact']
              for lane in LANES), 'PLAN_SPAN_REFS')
    paths = (
        'gx1/audit/materialize_ta_ema_delta_sidecars_v1.py',
        'gx1/features/entry_specialist_feature_groups_v1.py',
        'gx1/features/htf_features.py',
        'gx1/features/technical_indicators_v1.py',
        'gx1/contracts/gx1_capped_execution_v1.py',
        'gx1/scripts/prebuild_multi_tf_cache_v4.py',
        'tests/test_ta_ema_delta_sidecars_v1.py',
    )
    return {
        'schema_version': 'entry_ta_ema_delta_one_use_plan_v1',
        'stage': STAGE,
        'output_root': str(ROOT),
        'source_sha256': local['source_sha256'],
        'source_stat': _stat(SOURCE),
        'local_signal_stat': _stat(LOCAL_SIGNAL),
        'source_owners': {name: _ref(REPO / name) for name in paths},
        'inputs': inputs,
        'scope': {
            'cutoff_ns': CUTOFF_NS, 'fit_rows': 29574,
            'later_parent_start': 29849, 'later_rows': 17489,
            'physical_surfaces': 6, 'model_forwards': 0,
            'optimizer_steps': 0, 'test_data_used': False,
        },
        'execution': {
            'module': 'gx1.audit.materialize_ta_ema_delta_sidecars_v1',
            'cwd': str(REPO), 'runner': 'scripts/gx1_capped_run.sh',
            'job_class': 'audit', 'memory': '4G', 'swap': '512M',
            'timeout_seconds': WALL_SECONDS, 'kill_after_seconds': 10,
        },
    }


def _inspect_plan(plan_sha256: str, *, execute: bool) -> tuple[dict, dict, dict | None, str, bool]:
    _need(len(plan_sha256) == 64 and all(c in '0123456789abcdef' for c in plan_sha256),
          'PLAN_SHA_FORMAT')
    plan_ref = {'path': str(PLAN), 'sha256': plan_sha256}
    _need(_ref(PLAN) == plan_ref, 'PLAN_FILE')
    plan = _read_json(PLAN)
    _need(plan == build_plan(), 'PLAN_IDENTITY')
    _need(Path.cwd().resolve() == REPO, 'CWD')
    branch = subprocess.check_output(
        ['git', 'branch', '--show-current'], cwd=REPO, text=True,
    ).strip()
    head = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True,
    ).strip()
    clean = subprocess.check_output(
        ['git', 'status', '--porcelain'], cwd=REPO, text=True,
    ) == ''
    _need(branch == 'work/gx1-current' and not ROOT.exists(), 'BRANCH_OR_ONE_USE_ROOT')
    span_refs, payloads = _span_payloads(_read_json(SPAN_RESULT))
    _need(all(span_refs[lane] == plan['inputs'][lane + '_span'] for lane in LANES),
          'PREFLIGHT_SPAN_REFS')
    _check_prefix_spans(payloads)
    if not execute:
        return plan, plan_ref, None, head, clean
    _need(clean, 'CLEAN_SOURCE')
    # Hash-only integrity check; no cutoff-or-later feature values are decoded.
    _need(_sha_file(SOURCE) == plan['source_sha256'], 'SOURCE_FILE_SHA')
    policy_ref = _ref(POLICY)
    policy = _read_json(POLICY)
    _need(all(policy.get(key) is False for key in (
        'training_enabled', 'native_training_enabled', 'native_launch_authorized',
        'full_val_allowed', 'full_epoch_training_allowed',
    )), 'TRAINING_CLOSED')
    gate = policy.get(STAGE, {})
    _need(gate.get('status') == 'ADMITTED'
          and gate.get('plan') == plan_ref
          and gate.get('scope') == plan['scope']
          and gate.get('execution_authorized') is True
          and gate.get('data_reads_authorized') is True
          and gate.get('training_authorized') is False
          and gate.get('test_authorized') is False
          and gate.get('max_invocations') == gate.get('remaining_invocations') == 1
          and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
          and gate.get('relaunch_allowed') is False,
          'EXACT_POLICY_ADMISSION')
    return plan, plan_ref, policy_ref, head, clean


def _bookend(plan: dict, plan_ref: dict, policy_ref: dict, head: str) -> None:
    _need(_ref(PLAN) == plan_ref and _ref(POLICY) == policy_ref
          and _stat(SOURCE) == plan['source_stat']
          and _stat(LOCAL_SIGNAL) == plan['local_signal_stat']
          and all(_ref(Path(item['path'])) == item
                  for item in (*plan['source_owners'].values(), *plan['inputs'].values()))
          and subprocess.check_output(
              ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True,
          ).strip() == head
          and subprocess.check_output(
              ['git', 'status', '--porcelain'], cwd=REPO, text=True,
          ) == '', 'SOURCE_INPUT_BOOKEND')


def _publish_root(name: str, value: dict) -> dict:
    path = ROOT / name
    _write_json(path, value)
    return _ref(path)


def _terminal_record(
    status: str, *, plan: dict, claim: dict, result: dict | None,
    failure: dict | None, source_unchanged: bool,
) -> dict:
    _need((status == 'COMPLETE' and result is not None and failure is None)
          or (status == 'FAILED_CONSUMED' and result is None and failure is not None),
          'TERMINAL_STATE')
    return {
        'status': status, 'exit_code': 0 if status == 'COMPLETE' else 1,
        'no_relaunch': True, 'plan': plan, 'claim': claim, 'result': result,
        'failure': failure, 'source_unchanged': source_unchanged,
    }


def run(plan_sha256: str) -> tuple[dict, str]:
    cap = require_capped_cpu_audit_execution()
    plan, plan_ref, policy_ref, head, _ = _inspect_plan(
        plan_sha256, execute=True,
    )
    _need(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0),
          'AMBIENT_TIMER')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    ROOT.mkdir()
    claim = _publish_root('CLAIM.json', {
        'plan': plan_ref, 'policy': policy_ref, 'source_head': head,
        'cap': cap, 'scope': plan['scope'],
    })
    old_alarm = signal.getsignal(signal.SIGALRM)
    result_ref = failure_ref = None
    status = 'FAILED_CONSUMED'
    def expired(*_):
        raise TimeoutError('TA_EMA_DELTA_1800_SECONDS')
    try:
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, WALL_SECONDS - 30)
        materialize(ROOT / 'OUTPUT')
        _bookend(plan, plan_ref, policy_ref, head)
        result_ref = _ref(ROOT / 'OUTPUT/RESULT.json')
        status = 'COMPLETE'
    except BaseException as exc:
        failure_ref = _publish_root('FAILED.json', {
            'status': 'FAILED_CONSUMED', 'plan': plan_ref, 'claim': claim,
            'error_type': type(exc).__name__, 'error': str(exc),
        })
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        unchanged = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True,
        ).strip() == head
        terminal = _publish_root('TERMINAL.json', _terminal_record(
            status, plan=plan_ref, claim=claim, result=result_ref,
            failure=failure_ref, source_unchanged=unchanged,
        ))
    return terminal, status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    if args.preflight:
        plan, plan_ref, _, head, clean = _inspect_plan(
            args.plan_sha256, execute=False,
        )
        print(json.dumps({
            'status': 'PREFLIGHT_PASS' if clean else 'PREFLIGHT_SOURCE_DIRTY',
            'plan': plan_ref, 'source_head': head, 'source_clean': clean,
            'output_root_absent': True, 'scope': plan['scope'],
            'claim_created': False,
        }, sort_keys=True))
        return
    terminal, status = run(args.plan_sha256)
    print(json.dumps({'status': status, 'terminal': terminal}, sort_keys=True))
    if status != 'COMPLETE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
