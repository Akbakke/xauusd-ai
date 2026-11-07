"""One-use TRAIN-only M5/M15/H1 trendline replay against independent oracle.

The old conditional receipt supplies coordinates and pinned files, never
producer-selected line identities. No model, target selection, TEST or trade.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
import sys
from pathlib import Path
import subprocess
import time

import numpy as np

from gx1.audit import verify_h4_trendline_lifecycle_oracle_v1 as h4
from gx1.audit.verify_d1_trendline_lifecycle_oracle_v1 import (
    LIFECYCLE_NAMES, NAMES, oracle_rows,
)
from gx1.contracts.immutable_event_authority_v1 import _fsync_directory, _publish_bytes

REPO = Path('/home/andre2/src/GX1_CURRENT')
RUNS = Path('/home/andre2/GX1_RUNS')
STAGE = 'm5_m15_h1_trendline_physical_oracle_20261107_001'
SCHEMA = 'gx1_m5_m15_h1_trendline_physical_plan_v1'
PLAN_PATH = REPO / 'configs/research' / (STAGE.upper() + '.json')
OUT_ROOT = RUNS / STAGE.upper()
POLICY = REPO / 'NEXT_RUN_POLICY.json'
TRAIN_START_NS = 1714521600000000000
TRAIN_END_NS = 1735689600000000000
TIMEFRAMES = ('M5', 'M15', 'H1')
TF_SCOPE = {
    'M5': {'bar_ns': 300_000_000_000, 'seq_len': 16,
           'band_atr': 0.9161632622285967, 'prefix_rows': 1070080,
           'first_row': 987340, 'last_row': 1035124, 'witnesses': 11},
    'M15': {'bar_ns': 900_000_000_000, 'seq_len': 64,
            'band_atr': 0.906513313573848, 'prefix_rows': 357385,
            'first_row': 329804, 'last_row': 345732, 'witnesses': 10},
    'H1': {'bar_ns': 3_600_000_000_000, 'seq_len': 96,
           'band_atr': 0.7319144469085033, 'prefix_rows': 90075,
           'first_row': 83174, 'last_row': 87159, 'witnesses': 10},
}
INPUT_REFS = h4.OLD_REFS
SOURCE_PATHS = frozenset({
    'gx1/audit/verify_m5_m15_h1_trendline_oracle_v1.py',
    'gx1/audit/verify_h4_trendline_lifecycle_oracle_v1.py',
    'gx1/audit/verify_d1_trendline_lifecycle_oracle_v1.py',
    'gx1/audit/verify_d1_trendline_lifecycle_physical_v1.py',
    'gx1/scripts/audit_direction_d1_trendline_replay_v1.py',
    'gx1/features/htf_features.py',
    'gx1/features/technical_indicators_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'scripts/gx1_capped_run.sh',
})
LIMITS = {'memory_max': 4 * 1024**3, 'memory_high': 4 * 1024**3,
          'swap': 512 * 1024**2, 'max_wall_seconds': 1800}
EXECUTION = {'class': 'audit', 'runner': 'scripts/gx1_capped_run.sh',
             'memory': '4G', 'swap': '512M', 'timeout_seconds': 1800,
             'kill_after_seconds': 10,
             'module': 'gx1.audit.verify_m5_m15_h1_trendline_oracle_v1',
             'model_forwards': 0, 'optimizer_steps': 0}


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError('THREE_TF_TRENDLINE_' + reason)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _ref(path: Path) -> dict:
    return {'path': str(path), 'sha256': _sha(path)}


def _write(root: Path, name: str, value: dict) -> dict:
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _need(len(data) <= 1024 * 1024 and json.loads(data) == value, 'PUBLICATION')
    path = root / name
    _publish_bytes(path, data)
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}


def _clean_head() -> str:
    _need(Path.cwd().resolve() == REPO
          and subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
          == 'work/gx1-current'
          and subprocess.check_output(['git', 'status', '--porcelain'], text=True) == '',
          'CLEAN_SOURCE')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def _small_receipts() -> tuple[dict, dict, dict]:
    """Bind old small receipts and cache metadata without cache/native payloads."""
    old = h4.bound_json(INPUT_REFS['plan'])
    result = h4.bound_json(INPUT_REFS['result'])
    terminal = h4.bound_json(INPUT_REFS['terminal'])
    _need(result.get('plan') == INPUT_REFS['plan']
          and result.get('status') == terminal.get('status') == 'COMPLETE'
          and result.get('test_data_used') is False
          and terminal.get('result') == INPUT_REFS['result']
          and terminal.get('exit_code') == 0
          and terminal.get('test_data_used') is False
          and result.get('remaining_lifecycle_fields') == 19
          and result.get('full_31_field_independent_numeric_qualification') is False
          and old.get('test_data_used') is False
          and old.get('scope', {}).get('witness_start_ns') == TRAIN_START_NS
          and old.get('scope', {}).get('witness_end_exclusive_ns') == TRAIN_END_NS,
          'OLD_RECEIPT_CHAIN')
    inputs = old['inputs']
    _need(len(inputs['native_parts']) == 16
          and [int(Path(ref['path']).parent.name.split('=')[1])
               for ref in inputs['native_parts']] == list(range(2009, 2025)),
          'NATIVE_YEAR_SCOPE')
    manifest = h4.bound_json(inputs['cache_manifest'], max_bytes=64 * 1024 * 1024)
    fit = h4.bound_json(inputs['fit_result'])
    enriched = h4.bound_json(inputs['enriched_manifest'])
    native = h4.bound_json(inputs['native_manifest'])
    _need(fit['fold']['train_cutoff_ns'] == TRAIN_END_NS
          and fit['test_data_used'] is False
          and enriched['native_m5_source']['manifest_path'] == inputs['native_manifest']['path']
          and enriched['native_m5_source']['manifest_sha256'] == inputs['native_manifest']['sha256']
          and all(native['year_sha256'][Path(ref['path']).parent.name] == ref['sha256']
                  for ref in inputs['native_parts'])
          and manifest['feature_count'] == 253
          and list(manifest['feature_names'][159:190]) == list(NAMES)
          and len(NAMES) == 31 and len(LIFECYCLE_NAMES) == 19,
          'MANIFEST_LINEAGE')
    witnesses = result['physical_train_witnesses']
    _need(set(witnesses) == {'M5', 'M15', 'H1', 'H4', 'D1'}
          and sum(len(v) for v in witnesses.values()) == 43,
          'WITNESS_SCOPE')
    for tf in TIMEFRAMES:
        c = TF_SCOPE[tf]
        _need(manifest['v29_registry_constants']['per_tf_seq_lens'][tf] == c['seq_len']
              and manifest['v29_registry_constants']['trendline_band_atr'][tf] == c['band_atr']
              and manifest['tfs'][tf]['feature_count'] == 253,
              'FEATURE_CONSTANTS:' + tf)
        refs = inputs['per_tf'][tf]
        _need(set(refs) == {'feats', 'ts', 'replay_plan', 'replay_result', 'replay_terminal'}
              and refs['feats'] == {
                  'path': str(Path(inputs['cache_manifest']['path']).parent /
                              manifest['tfs'][tf]['feats_npy']),
                  'sha256': manifest['tfs'][tf]['feats_npy_sha256']}
              and refs['ts'] == {
                  'path': str(Path(inputs['cache_manifest']['path']).parent /
                              manifest['tfs'][tf]['ts_npy']),
                  'sha256': manifest['tfs'][tf]['ts_npy_sha256']},
              'CACHE_REFS:' + tf)
        replay_plan = h4.bound_json(refs['replay_plan'])
        replay_result = h4.bound_json(refs['replay_result'])
        replay_terminal = h4.bound_json(refs['replay_terminal'])
        _need(replay_plan['inputs']['tf_feats'] == refs['feats']
              and replay_plan['inputs']['tf_ts'] == refs['ts']
              and replay_plan['inputs']['cache_manifest'] == inputs['cache_manifest']
              and replay_result['status'] == replay_terminal['status'] == 'COMPLETE'
              and replay_terminal['exit_code'] == 0
              and replay_result['test_data_used'] is False
              and replay_result['input_price_precision'] == 'native_float64_full_history'
              and replay_result['replay_parity'] is True
              and replay_result['source_clock_rows'] == c['prefix_rows']
              and replay_result['window_configuration'] == {
                  'timeframe': tf, 'seq_len': c['seq_len'], 'band_atr': c['band_atr']}
              and replay_result['all_causal_history_vs_cache']['mismatched_cells'] == 0
              and replay_result['model_train_period_vs_cache']['mismatched_cells'] == 0
              and replay_result['model_train_period_vs_cache']['rows']
                  == c['last_row'] - c['first_row'] + 1
              and len(replay_result['model_train_period_vs_cache']['per_field']) == len(NAMES)
              and all(field['both_finite_rows'] == c['last_row'] - c['first_row'] + 1
                      for field in replay_result['model_train_period_vs_cache']['per_field']),
              'REPLAY_RECEIPT:' + tf)
        records = witnesses[tf]
        _need(len(records) == c['witnesses']
              and len({r['cache_row'] for r in records}) == len(records)
              and all(c['first_row'] <= r['cache_row'] <= c['last_row']
                      and r['producer_cache_float32_bit_parity'] is True
                      and math.isfinite(r['source_atr14']) and r['source_atr14'] > 0
                      for r in records),
              'WITNESSES:' + tf)
        for ref in (*inputs['native_parts'], refs['feats'], refs['ts']):
            path = Path(ref['path'])
            _need(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
                  'PAYLOAD_HEADER:' + tf)
    return inputs, manifest, witnesses


def _inspect_plan(plan_path: Path, plan_sha256: str, out_dir: Path
                  ) -> tuple[dict, dict, str, dict, dict, dict]:
    head = _clean_head()
    _need(plan_path == PLAN_PATH and out_dir == OUT_ROOT
          and plan_path.is_file() and not out_dir.exists()
          and len(plan_sha256) == 64 and _sha(plan_path) == plan_sha256,
          'PLAN_IDENTITY')
    plan = json.loads(plan_path.read_text())
    _need(plan.get('schema_version') == SCHEMA and plan.get('stage') == STAGE
          and plan.get('status') == 'PREPARED_NOT_ADMITTED'
          and plan.get('output_root') == str(OUT_ROOT)
          and plan.get('source_base_head') is not None
          and isinstance(plan['source_base_head'], str)
          and len(plan['source_base_head']) == 40
          and subprocess.run(['git', 'merge-base', '--is-ancestor',
                              plan['source_base_head'], head],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL).returncode == 0
          and plan.get('max_invocations') == 1
          and plan.get('scope') == {'train_start_ns': TRAIN_START_NS,
                                    'train_end_exclusive_ns': TRAIN_END_NS,
                                    'timeframes': list(TIMEFRAMES),
                                    'tf_scope': TF_SCOPE, 'feature_columns': [159, 190]}
          and plan.get('input_refs') == INPUT_REFS
          and plan.get('limits') == LIMITS
          and plan.get('execution') == EXECUTION
          and plan.get('model_forwards_authorized') == 0
          and plan.get('optimizer_steps_authorized') == 0
          and plan.get('training_authorized') is False
          and plan.get('test_data_used') is False,
          'PLAN_SCOPE')
    owners = plan.get('source_owners')
    _need(isinstance(owners, dict) and set(owners) == SOURCE_PATHS
          and all(_sha(REPO / path) == digest for path, digest in owners.items()),
          'SOURCE_OWNERS')
    inputs, manifest, witnesses = _small_receipts()
    return plan, {'path': str(plan_path), 'sha256': plan_sha256}, head, inputs, manifest, witnesses


def preflight_read_only(plan_path: Path, plan_sha256: str, out_dir: Path) -> dict:
    plan, ref, head, inputs, _manifest, witnesses = _inspect_plan(
        plan_path, plan_sha256, out_dir)
    return {'status': 'READY_NO_CLAIM_NO_PAYLOAD', 'plan': ref,
            'source_head': head, 'source_owners_checked': len(plan['source_owners']),
            'small_receipts_checked': 16, 'native_parts_stat_checked': len(inputs['native_parts']),
            'witnesses': {tf: len(witnesses[tf]) for tf in TIMEFRAMES},
            'claim_created': False, 'model_forwards': 0,
            'optimizer_steps': 0, 'test_data_used': False}


def _align_rows(cache_ts: np.ndarray, native_ns: np.ndarray, tf: str
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = TF_SCOPE[tf]
    _need(cache_ts.dtype == native_ns.dtype == np.int64
          and cache_ts.ndim == native_ns.ndim == 1
          and len(cache_ts) > c['last_row'] + 1
          and len(native_ns) > 0
          and np.all(np.diff(cache_ts) > 0)
          and np.all(np.diff(native_ns) > 0),
          'CLOCK_AXES:' + tf)
    rows = np.flatnonzero((cache_ts[:c['prefix_rows']] + c['bar_ns'] >= TRAIN_START_NS)
                          & (cache_ts[:c['prefix_rows']] + c['bar_ns'] < TRAIN_END_NS))
    _need(np.array_equal(rows, np.arange(c['first_row'], c['last_row'] + 1))
          and int(cache_ts[c['last_row'] + 1]) + c['bar_ns'] >= TRAIN_END_NS,
          'TRAIN_ROW_RANGE:' + tf)
    timestamps = np.asarray(cache_ts[rows], dtype=np.int64)
    positions = np.searchsorted(native_ns, timestamps)
    _need(int(positions[-1]) < len(native_ns)
          and np.array_equal(native_ns[positions], timestamps)
          and np.all(np.diff(positions) > 0),
          'NATIVE_JOIN:' + tf)
    return rows, positions, timestamps


def _compare(oracle: np.ndarray, cache: np.ndarray,
             rows: np.ndarray, times: np.ndarray, witness_rows: set[int]) -> dict:
    a = np.asarray(oracle, dtype=np.float64)
    b = np.asarray(cache, dtype=np.float32).astype(np.float64)
    _need(a.shape == b.shape == (len(rows), len(NAMES))
          and times.shape == rows.shape
          and np.all(np.diff(rows) == 1) and np.all(np.diff(times) > 0)
          and not np.isinf(a).any() and not np.isinf(b).any(),
          'COMPARISON_SHAPE')
    per_field = []
    examples = []
    total = anchor_total = lifecycle_total = 0
    lifecycle = set(LIFECYCLE_NAMES)
    for j, name in enumerate(NAMES):
        expected, observed = a[:, j], b[:, j]
        finite = np.isfinite(expected) & np.isfinite(observed)
        tolerance = (np.zeros(int(finite.sum())) if j in h4.DISCRETE
                     else np.maximum(h4.FLOAT_ABS_TOL,
                                     h4.FLOAT_REL_TOL * np.abs(expected[finite])))
        match = np.isnan(expected) & np.isnan(observed)
        match[finite] = np.abs(expected[finite] - observed[finite]) <= tolerance
        bad = np.flatnonzero(~match)
        total += len(bad)
        anchor_total += int(np.isin(rows[bad], list(witness_rows)).sum())
        lifecycle_total += len(bad) if name in lifecycle else 0
        for i in bad[:max(0, 128 - len(examples))]:
            examples.append({'cache_row': int(rows[i]), 'timestamp_ns': int(times[i]),
                             'field': name,
                             'oracle': float(expected[i]) if np.isfinite(expected[i]) else None,
                             'cache': float(observed[i]) if np.isfinite(observed[i]) else None})
        source_events = int(np.count_nonzero(np.isfinite(expected) & (expected != 0)))
        cache_events = int(np.count_nonzero(np.isfinite(observed) & (observed != 0)))
        per_field.append({'field': name, 'lifecycle': name in LIFECYCLE_NAMES,
                          'oracle_finite_rows': int(np.isfinite(expected).sum()),
                          'cache_finite_rows': int(np.isfinite(observed).sum()),
                          'oracle_nonzero_or_event_rows': source_events,
                          'cache_nonzero_or_event_rows': cache_events,
                          'exercised': bool(source_events and cache_events),
                          'mismatched_rows': len(bad),
                          'max_finite_absolute_error': (float(np.abs(expected[finite] - observed[finite]).max())
                                                       if finite.any() else None)})
    return {'rows': len(rows), 'fields': len(NAMES),
            'compared_cells': len(rows) * len(NAMES),
            'mismatched_cells': total,
            'lifecycle_mismatched_cells': lifecycle_total,
            'anchor_mismatched_cells': anchor_total,
            'all_train_window_cells_match': total == 0,
            'matched_and_exercised_lifecycle_fields': [x['field'] for x in per_field
                                                       if x['lifecycle'] and x['exercised']
                                                       and x['mismatched_rows'] == 0],
            'unexercised_lifecycle_fields': [x['field'] for x in per_field
                                             if x['lifecycle'] and not x['exercised']],
            'float_abs_tolerance': h4.FLOAT_ABS_TOL,
            'float_rel_tolerance': h4.FLOAT_REL_TOL,
            'per_field': per_field, 'mismatch_examples': examples,
            'mismatch_examples_truncated': total > len(examples)}


def _measure(plan: dict, inputs: dict, manifest: dict, witnesses: dict) -> dict:
    from gx1.scripts.audit_direction_d1_trendline_replay_v1 import load_native_parts_prefix
    from gx1.features.htf_features import (
        _resample_ohlcv, build_multi_tf_v4_closed_timestamp_indices,
    )
    from gx1.features.technical_indicators_v1 import ema50_200_spread_atr_block
    stats = {}
    for ref in inputs['native_parts']:
        path, stat = h4.checked(ref)
        stats[str(path)] = stat
    for tf in TIMEFRAMES:
        for ref in (inputs['per_tf'][tf]['feats'], inputs['per_tf'][tf]['ts']):
            path, stat = h4.checked(ref)
            stats[str(path)] = stat
    source = load_native_parts_prefix(inputs['native_parts'], TRAIN_END_NS)
    _need(len(source) > 0 and source.index[-1].value < TRAIN_END_NS,
          'NATIVE_TRAIN_ONLY')
    index_by_tf = build_multi_tf_v4_closed_timestamp_indices(source.index)
    result = {}
    for tf in TIMEFRAMES:
        c = TF_SCOPE[tf]
        refs = inputs['per_tf'][tf]
        index = index_by_tf[tf]
        bars = _resample_ohlcv(source, tf).dropna(
            subset=['open', 'high', 'low', 'close']).loc[index]
        _need(bars.index.equals(index), 'RESAMPLE_AXIS:' + tf)
        atr = ema50_200_spread_atr_block(
            bars['high'], bars['low'], bars['close'])['atr14']
        cache_ts = np.load(refs['ts']['path'], mmap_mode='r', allow_pickle=False)
        cache = np.load(refs['feats']['path'], mmap_mode='r', allow_pickle=False)
        _need(cache_ts.dtype == np.int64 and cache_ts.ndim == 1
              and cache.dtype == np.float32 and cache.ndim == 2
              and cache.shape == (len(cache_ts), manifest['feature_count']),
              'CACHE_SHAPE:' + tf)
        rows, positions, times = _align_rows(cache_ts, index.asi8, tf)
        for record in witnesses[tf]:
            row = int(record['cache_row'])
            pos = int(record['native_bar_index'])
            _need(int(cache_ts[row]) == int(record['timestamp_ns'])
                  and int(index.asi8[pos]) == int(record['timestamp_ns'])
                  and int(positions[row - c['first_row']]) == pos
                  and float(bars.iloc[pos]['close']) == record['source_close']
                  and math.isclose(float(atr.iloc[pos]), record['source_atr14'],
                                   rel_tol=0, abs_tol=1e-12),
                  'WITNESS_CLOCK:' + tf)
        end = int(positions[-1]) + 1
        oracle, events = oracle_rows(
            bars['high'].to_numpy()[:end], bars['low'].to_numpy()[:end],
            bars['close'].to_numpy()[:end], atr.to_numpy()[:end],
            seq_len=c['seq_len'], band_atr=c['band_atr'],
            witness_positions=positions)
        comparison = _compare(oracle, cache[rows, 159:190], rows, times,
                              {int(r['cache_row']) for r in witnesses[tf]})
        result[tf] = {'train_cache_row_first': c['first_row'],
                      'train_cache_row_last_inclusive': c['last_row'],
                      'timestamp_rule': tf + ' cache label plus its bar duration is TRAIN close',
                      'comparison': comparison, 'oracle_event_counts': events,
                      'witness_count': len(witnesses[tf])}
        del cache_ts, cache, bars, atr, rows, positions, times, oracle
    for path, stat in stats.items():
        _need(h4.regular(Path(path)) == stat, 'FINAL_PAYLOAD_STAT:' + Path(path).name)
    return {'scope_limit': 'May-Dec 2024 TRAIN only; formula/source parity, not predictor influence, model learning, or economic edge.',
            'timeframes': result,
            'native_parts_hashed': len(inputs['native_parts']),
            'cache_arrays_hashed': 2 * len(TIMEFRAMES),
            'model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False}


def run(plan_path: Path, plan_sha256: str, out_dir: Path) -> dict:
    plan, plan_ref, head, inputs, manifest, witnesses = _inspect_plan(
        plan_path, plan_sha256, out_dir)
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry,
    )
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    _need(all(cap.get(key) == value for key, value in LIMITS.items()
              if key != 'max_wall_seconds')
          and cap.get('pids') == 64 and cap.get('numerical_threads') == 1
          and lock.get('scope') == cap.get('cgroup_path', '')[len('/sys/fs/cgroup'):],
          'CAP_LIMITS')
    _need(not any(name.split('.')[0] in ('torch', 'sklearn', 'joblib')
                  for name in sys.modules), 'MODEL_IMPORT')
    policy_ref = _ref(POLICY)
    policy = json.loads(POLICY.read_text())
    _need(_sha(POLICY) == policy_ref['sha256'], 'POLICY_STABLE_READ')
    gate = policy.get(STAGE, {})
    _need(policy.get('training_enabled') is False
          and policy.get('native_training_enabled') is False
          and policy.get('native_launch_authorized') is False
          and policy.get('full_val_allowed') is False
          and gate.get('status') == gate.get('decision') == 'ADMITTED'
          and gate.get('plan') == plan_ref
          and gate.get('execution_authorized') is True
          and gate.get('data_reads_authorized') is True
          and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
          and gate.get('max_invocations') == 1
          and gate.get('relaunch_allowed') is False
          and gate.get('memory_max') == '4G'
          and gate.get('swap_max') == '512M'
          and gate.get('wall_seconds') == LIMITS['max_wall_seconds']
          and gate.get('test_data_used') is False
          and gate.get('checkpoint_reads_authorized') is False
          and gate.get('broker_access_allowed') is False
          and gate.get('model_forwards_authorized') == 0
          and gate.get('optimizer_steps_authorized') == 0
          and gate.get('training_authorized') is False
          and gate.get('test_authorized') is False
          and gate.get('output_root') == str(OUT_ROOT),
          'POLICY_GATE')
    _need(_sha(POLICY) == policy_ref['sha256'], 'POLICY_BEFORE_CLAIM')
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    claim = None
    old_alarm = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
            TimeoutError('THREE_TF_TRENDLINE_WALL_LIMIT')))
        signal.alarm(LIMITS['max_wall_seconds'])
        _fsync_directory(out_dir.parent)
        claim = _write(out_dir, 'CLAIM.json', {
            'schema_version': 'gx1_three_tf_trendline_claim_v1',
            'plan': plan_ref, 'policy': policy_ref, 'source_head': head,
            'timeframes': list(TIMEFRAMES), 'invocation': 1})
        data = _measure(plan, inputs, manifest, witnesses)
        _need(_sha(plan_path) == plan_sha256
              and _sha(POLICY) == policy_ref['sha256']
              and _clean_head() == head
              and _sha(out_dir / 'CLAIM.json') == claim['sha256']
              and {entry.name for entry in out_dir.iterdir()} == {'CLAIM.json'},
              'FINAL_SOURCE_IDENTITY')
        comparisons = [data['timeframes'][tf]['comparison'] for tf in TIMEFRAMES]
        decision = ('FAIL_M5_M15_H1_TRENDLINE_PARITY'
                    if any(c['mismatched_cells'] for c in comparisons) else
                    'PASS_M5_M15_H1_TRENDLINE_19_EXERCISED'
                    if all(not c['unexercised_lifecycle_fields'] for c in comparisons) else
                    'PASS_M5_M15_H1_TRENDLINE_WITH_SUPPORT_GAPS')
        result = _write(out_dir, 'RESULT.json', {
            'schema_version': 'gx1_three_tf_trendline_result_v1',
            'status': 'COMPLETE', 'decision': decision,
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'elapsed_seconds': time.monotonic() - started, **data})
        terminal = _write(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_three_tf_trendline_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0,
            'plan': plan_ref, 'claim': claim, 'result': result,
            'model_forwards': 0, 'optimizer_steps': 0,
            'test_data_used': False, 'no_relaunch_under_this_plan': True,
            'output_inventory': ['CLAIM.json', 'RESULT.json', 'TERMINAL.json']})
        return {'result': result, 'terminal': terminal}
    except Exception as exc:
        failure = _write(out_dir, 'FAILED.json', {
            'schema_version': 'gx1_three_tf_trendline_failure_v1',
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'no_relaunch_under_this_plan': True})
        _write(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_three_tf_trendline_terminal_v1',
            'status': 'FAILED', 'exit_code': 1,
            'plan': plan_ref, 'claim': claim, 'failure': failure,
            'model_forwards': 0, 'optimizer_steps': 0,
            'test_data_used': False, 'no_relaunch_under_this_plan': True,
            'output_inventory': (['CLAIM.json', 'FAILED.json', 'TERMINAL.json']
                                 if claim else ['FAILED.json', 'TERMINAL.json'])})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight_read_only(args.plan, args.plan_sha256,
                                             args.out_dir), sort_keys=True))
        return 0
    print(json.dumps(run(args.plan, args.plan_sha256, args.out_dir), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
