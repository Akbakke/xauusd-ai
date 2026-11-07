"""One-use, TRAIN-only physical D1 trendline lifecycle witness audit.

The independent oracle receives only causal OHLC/ATR arrays and TRAIN-fitted
constants. The old conditional receipt supplies immutable witness coordinates,
never producer line identities or state. This worker does not fit or score a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True

import numpy as np

from gx1.audit.verify_d1_trendline_lifecycle_oracle_v1 import (
    LIFECYCLE_NAMES, NAMES, oracle_rows,
)

REPO = Path('/home/andre2/src/GX1_CURRENT')
STAGE = 'd1_trendline_lifecycle_physical_oracle_20261107_001'
TRAIN_START_NS = 1714521600000000000
TRAIN_END_NS = 1735689600000000000
DAY_NS = 86400 * 10**9
WITNESSES = (
    (3588, 3857, 1714514400000000000),
    (3595, 3864, 1715292000000000000),
    (3596, 3865, 1715551200000000000),
    (3604, 3873, 1716415200000000000),
)
TRAIN_CACHE_ROWS = tuple(range(3588, 3762))  # Exact May-Dec 2024 D1 closes.
OLD_ROOT = Path('/home/andre2/GX1_RUNS/TRENDLINE_CONDITIONAL_PREFIX_20261106_001')
OLD_REFS = {
    'plan': {'path': str(REPO / 'configs/research/TRENDLINE_CONDITIONAL_PREFIX_20261106_001.json'),
             'sha256': 'bc4469117ef476250e50ed308941b790da67ed30b134b5bf350e100240f17491'},
    'result': {'path': str(OLD_ROOT / 'RESULT.json'),
               'sha256': '09be782ab60eaf522ee121254354ef8af33077d6d71a9ee173c779f0b77add3f'},
    'terminal': {'path': str(OLD_ROOT / 'TERMINAL.json'),
                 'sha256': '02a100486d68e5afb59df94bbb7c5d8372d901f40cb1d5a3de6c5945c65acd2e'},
}
SOURCE_PATHS = {
    'worker': 'gx1/audit/verify_d1_trendline_lifecycle_physical_v1.py',
    'oracle': 'gx1/audit/verify_d1_trendline_lifecycle_oracle_v1.py',
    'replay_loader': 'gx1/scripts/audit_direction_d1_trendline_replay_v1.py',
    'resample': 'gx1/features/htf_features.py',
    'atr': 'gx1/features/technical_indicators_v1.py',
    'capped_contract': 'gx1/contracts/gx1_capped_execution_v1.py',
    'capped_runner': 'scripts/gx1_capped_run.sh',
    'publication': 'gx1/contracts/immutable_event_authority_v1.py',
}
LIMITS = {'memory_max': '4G', 'swap_max': '512M', 'wall_seconds': 1800,
          'execution_class': 'audit', 'model_forwards': 0, 'optimizer_steps': 0}
SCOPE = {'timeframe': 'D1', 'train_start_ns': TRAIN_START_NS,
         'train_end_exclusive_ns': TRAIN_END_NS,
         'train_cache_row_first': 3588, 'train_cache_row_last_inclusive': 3761,
         'train_cache_row_count': 174,
         'anchor_cache_rows': [row for row, _, _ in WITNESSES],
         'test_data_used': False}
# Discrete state/count/age fields must agree exactly, irrespective of float tolerance.
DISCRETE = frozenset((0, 3, 4, 5, 7, 10, 11, 12, 14, 15, 16, 17,
                      18, 19, 20, 21, 22, 23, 24, 25, 29))
FLOAT_ABS_TOL = 2e-6
FLOAT_REL_TOL = 2e-6


def need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError('D1_TRENDLINE_PHYSICAL_' + reason)


def regular(path: Path) -> tuple[int, int, int, int]:
    need(path.is_absolute() and path.is_file() and not path.is_symlink()
         and path.resolve(strict=True) == path, 'REGULAR_' + path.name)
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def checked(ref: dict) -> tuple[Path, tuple[int, int, int, int]]:
    need(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}
         and isinstance(ref['sha256'], str) and len(ref['sha256']) == 64,
         'REF_SHAPE')
    path = Path(ref['path'])
    before = regular(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    need(digest.hexdigest() == ref['sha256'] and regular(path) == before,
         'REF_SHA_' + path.name)
    return path, before


def bound_json(ref: dict, *, max_bytes: int = 16 * 1024 * 1024) -> dict:
    path, before = checked(ref)
    need(before[2] <= max_bytes, 'JSON_SIZE_' + path.name)
    value = json.loads(path.read_text())
    need(regular(path) == before and isinstance(value, dict), 'JSON_BOOKEND_' + path.name)
    return value


def read_claim(output: Path, plan_ref: dict, worker_ref: dict,
               source_head: str) -> tuple[dict, dict, tuple[int, int, int, int]]:
    need(output.is_dir() and not output.is_symlink()
         and {entry.name for entry in output.iterdir()} == {'CLAIM.json'},
         'CLAIM_ONLY_ROOT')
    path = output / 'CLAIM.json'
    before = regular(path)
    need(before[2] <= 1024 * 1024, 'CLAIM_SIZE')
    raw = path.read_bytes()
    need(regular(path) == before, 'CLAIM_BOOKEND')
    claim = json.loads(raw)
    need(isinstance(claim, dict) and set(claim) == {
        'schema_version', 'stage', 'plan', 'policy', 'source_head',
        'worker', 'source_only_oracle', 'test_data_used',
    } and claim['schema_version'] == 'gx1_d1_trendline_lifecycle_physical_claim_v1'
         and claim['stage'] == STAGE and claim['plan'] == plan_ref
         and claim['worker'] == worker_ref and claim['source_head'] == source_head
         and claim['source_only_oracle'] is True and claim['test_data_used'] is False
         and isinstance(claim['policy'], dict)
         and claim['policy'].get('path') == str(REPO / 'NEXT_RUN_POLICY.json'),
         'CLAIM_BINDING')
    ref = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
    return claim, ref, before


def git_state() -> tuple[str, str, str]:
    def cmd(*args: str) -> str:
        return subprocess.check_output(('git', *args), cwd=REPO, text=True).strip()
    return cmd('rev-parse', 'HEAD'), cmd('branch', '--show-current'), cmd('status', '--porcelain')


def check_witness_receipt(result: dict) -> list[dict]:
    need(result.get('status') == 'COMPLETE' and result.get('test_data_used') is False
         and result.get('remaining_lifecycle_fields') == 19
         and result.get('semantic_line_selection_qualified') is False
         and result.get('full_31_field_independent_numeric_qualification') is False,
         'OLD_RESULT_SCOPE')
    all_witnesses = result.get('physical_train_witnesses')
    need(isinstance(all_witnesses, dict) and set(all_witnesses) == {'M5', 'M15', 'H1', 'H4', 'D1'}
         and sum(map(len, all_witnesses.values())) == 43, 'OLD_WITNESS_COUNT')
    records = all_witnesses['D1']
    need(len(records) == len(WITNESSES), 'D1_WITNESS_COUNT')
    for record, (row, pos, timestamp) in zip(records, WITNESSES):
        need((record['cache_row'], record['native_bar_index'], record['timestamp_ns'])
             == (row, pos, timestamp)
             and record['producer_cache_float32_bit_parity'] is True
             and math.isfinite(record['source_atr14']) and record['source_atr14'] > 0,
             'D1_WITNESS_IDENTITY')
    scan = result['cache_prefix_representation']['D1']
    need(scan['scanned_rows'] == scan['valid_rows'] == 3889
         and scan['all_12_train_witness_classes_present'] is True
         and scan['all_31_representation_invariants_pass'] is True,
         'D1_PREFIX_RECEIPT')
    return records


def align_witnesses(cache_ts: np.ndarray, native_ns: np.ndarray,
                    records: list[dict]) -> list[int]:
    need(cache_ts.ndim == native_ns.ndim == 1 and cache_ts.dtype == native_ns.dtype == np.int64
         and len(native_ns) > WITNESSES[-1][1]
         and np.all(np.diff(native_ns) > 0), 'AXIS_SHAPE')
    positions = []
    for record, (row, pos, timestamp) in zip(records, WITNESSES):
        need(row < len(cache_ts) and int(cache_ts[row]) == timestamp
             and int(native_ns[pos]) == timestamp
             and int(np.searchsorted(native_ns, timestamp)) == pos
             and TRAIN_START_NS <= timestamp + DAY_NS < TRAIN_END_NS,
             'WITNESS_CLOCK_' + str(row))
        positions.append(pos)
    return positions


def align_train_rows(cache_ts: np.ndarray,
                     native_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map every May-Dec 2024 D1 close to its independent raw D1 bar."""
    need(cache_ts.dtype == native_ns.dtype == np.int64
         and cache_ts.ndim == native_ns.ndim == 1
         and len(cache_ts) >= 3889 and len(native_ns) > WITNESSES[-1][1]
         and np.all(np.diff(cache_ts) > 0)
         and np.all(np.diff(native_ns) > 0), 'TRAIN_AXIS_SHAPE')
    labels = np.asarray(cache_ts[:3889], dtype=np.int64)
    need(np.all(np.diff(labels) > 0), 'CACHE_PREFIX_CLOCK')
    closed = labels + DAY_NS
    rows = np.flatnonzero((closed >= TRAIN_START_NS) & (closed < TRAIN_END_NS))
    need(np.array_equal(rows, np.asarray(TRAIN_CACHE_ROWS, dtype=np.int64))
         and int(cache_ts[3762]) + DAY_NS >= TRAIN_END_NS,
         'TRAIN_ROW_RANGE')
    timestamps = labels[rows]
    positions = np.searchsorted(native_ns, timestamps)
    need(int(positions[-1]) < len(native_ns)
         and np.array_equal(native_ns[positions], timestamps)
         and np.all(np.diff(positions) > 0), 'TRAIN_NATIVE_JOIN')
    return rows, positions, timestamps


def compare_train_rows(oracle: np.ndarray, cache: np.ndarray,
                       cache_rows: np.ndarray, timestamps: np.ndarray) -> dict:
    """Compare 174x31 cells; report compact field support and mismatches."""
    a = np.asarray(oracle, dtype=np.float64)
    b = np.asarray(cache, dtype=np.float32).astype(np.float64)
    rows = np.asarray(cache_rows, dtype=np.int64)
    times = np.asarray(timestamps, dtype=np.int64)
    need(a.shape == b.shape == (174, 31)
         and np.array_equal(rows, np.asarray(TRAIN_CACHE_ROWS, dtype=np.int64))
         and times.shape == (174,) and np.all(np.diff(times) > 0)
         and len(NAMES) == 31 and len(LIFECYCLE_NAMES) == 19
         and not np.isinf(a).any() and not np.isinf(b).any(), 'COMPARE_SHAPE')
    lifecycle = set(LIFECYCLE_NAMES)
    per_field = []
    examples = []
    mismatched = 0
    lifecycle_mismatched = 0
    anchor_mismatched = 0
    anchor_rows = {row for row, _, _ in WITNESSES}
    for j, name in enumerate(NAMES):
        expected, observed = a[:, j], b[:, j]
        finite = np.isfinite(expected) & np.isfinite(observed)
        both_missing = np.isnan(expected) & np.isnan(observed)
        error = np.abs(expected[finite] - observed[finite])
        tolerance = (np.zeros(len(error), dtype=np.float64) if j in DISCRETE
                     else np.maximum(FLOAT_ABS_TOL,
                                     FLOAT_REL_TOL * np.abs(expected[finite])))
        matched = both_missing.copy()
        matched[finite] = error <= tolerance
        bad = np.flatnonzero(~matched)
        mismatched += len(bad)
        lifecycle_mismatched += len(bad) if name in lifecycle else 0
        anchor_mismatched += sum(int(rows[i] in anchor_rows) for i in bad)
        for i in bad:
            if len(examples) >= 128:
                break
            x, y = float(expected[i]), float(observed[i])
            examples.append({
                'cache_row': int(rows[i]), 'timestamp_ns': int(times[i]),
                'field': name, 'oracle': None if math.isnan(x) else x,
                'cache': None if math.isnan(y) else y,
                'absolute_error': abs(x - y) if math.isfinite(x) and math.isfinite(y) else None,
            })
        oracle_finite = np.isfinite(expected)
        cache_finite = np.isfinite(observed)
        oracle_nonzero = int(np.count_nonzero(oracle_finite & (expected != 0)))
        cache_nonzero = int(np.count_nonzero(cache_finite & (observed != 0)))
        per_field.append({
            'field': name, 'lifecycle': name in lifecycle,
            'oracle_finite_rows': int(np.count_nonzero(oracle_finite)),
            'cache_finite_rows': int(np.count_nonzero(cache_finite)),
            'both_missing_rows': int(np.count_nonzero(both_missing)),
            'oracle_nonzero_or_event_rows': oracle_nonzero,
            'cache_nonzero_or_event_rows': cache_nonzero,
            'oracle_positive_rows': int(np.count_nonzero(oracle_finite & (expected > 0))),
            'oracle_negative_rows': int(np.count_nonzero(oracle_finite & (expected < 0))),
            'exercised': bool(oracle_nonzero and cache_nonzero),
            'mismatched_rows': int(len(bad)),
            'max_finite_absolute_error': float(np.max(error)) if len(error) else None,
        })
    exercised = [x['field'] for x in per_field
                 if x['lifecycle'] and x['exercised'] and x['mismatched_rows'] == 0]
    need(len(per_field) == 31 and sum(x['lifecycle'] for x in per_field) == 19,
         'COMPARISON_COVERAGE')
    return {
        'rows': 174, 'fields': 31, 'compared_cells': 5394,
        'lifecycle_fields': 19, 'lifecycle_cells': 3306,
        'anchor_rows': 4, 'anchor_cells': 124,
        'mismatched_cells': mismatched,
        'lifecycle_mismatched_cells': lifecycle_mismatched,
        'anchor_mismatched_cells': anchor_mismatched,
        'all_train_window_cells_match': mismatched == 0,
        'matched_and_exercised_lifecycle_fields': exercised,
        'unexercised_lifecycle_fields': [x['field'] for x in per_field
                                         if x['lifecycle'] and not x['exercised']],
        'mismatched_lifecycle_fields': [x['field'] for x in per_field
                                        if x['lifecycle'] and x['mismatched_rows']],
        'float_abs_tolerance': FLOAT_ABS_TOL,
        'float_rel_tolerance': FLOAT_REL_TOL,
        'per_field': per_field,
        'mismatch_examples': examples,
        'mismatch_examples_truncated': mismatched > len(examples),
    }


def preflight(plan_ref: dict) -> tuple[dict, dict, dict, list[dict], dict, tuple[str, str, str]]:
    plan = bound_json(plan_ref)
    need(set(plan) == {'schema_version', 'stage', 'status', 'inputs', 'sources',
                       'scope', 'limits', 'output_dir', 'test_data_used'}
         and plan['schema_version'] == 'gx1_d1_trendline_lifecycle_physical_plan_v1'
         and plan['stage'] == STAGE and plan['status'] == 'PREPARED_NOT_ADMITTED'
         and plan['inputs'] == OLD_REFS and plan['scope'] == SCOPE
         and plan['limits'] == LIMITS and plan['test_data_used'] is False,
         'PLAN_CONTRACT')
    state = git_state()
    need(state[1:] == ('work/gx1-current', ''), 'SOURCE_TREE')
    need(set(plan['sources']) == set(SOURCE_PATHS), 'SOURCE_KEYS')
    source_stats = {}
    for name, relative in SOURCE_PATHS.items():
        ref = plan['sources'][name]
        need(ref['path'] == str(REPO / relative), 'SOURCE_PATH_' + name)
        _, source_stats[name] = checked(ref)
    need(plan['sources']['replay_loader']['sha256'] ==
         'afe228b61eb5ec5a10e39f80c1bf4131b806ebf95ea028c6545c74f89530b56a'
         and plan['sources']['resample']['sha256'] ==
         '9ca968805e3eca72525e14c8b53e5b31817cd2baa7c0cdb9b636cb112f092520'
         and plan['sources']['atr']['sha256'] ==
         '3a81bec825639e57a3071cc28d4dc2d3d394fef031dc89d71da3b468ca28ecad',
         'SOURCE_LINEAGE')
    output = Path(plan['output_dir'])
    need(output.is_absolute() and output.parent == Path('/home/andre2/GX1_RUNS')
         and output.name == STAGE.upper() and output.parent.is_dir(),
         'OUTPUT_SCOPE')
    claim, claim_ref, claim_stat = read_claim(
        output, plan_ref, plan['sources']['worker'], state[0])
    policy = bound_json(claim['policy'])
    gate = policy.get(STAGE)
    need(isinstance(gate, dict) and gate.get('status') == 'ADMITTED'
         and gate.get('plan') == plan_ref
         and gate.get('worker') == plan['sources']['worker']
         and gate.get('output_root') == plan['output_dir']
         and gate.get('execution_authorized') is True
         and gate.get('data_reads_authorized') is True
         and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
         and gate.get('max_invocations') == 1
         and gate.get('relaunch_allowed') is False
         and gate.get('memory_max') == '4G' and gate.get('swap_max') == '512M'
         and gate.get('wall_seconds') == 1800
         and gate.get('test_data_used') is False
         and gate.get('model_forwards_authorized') is False
         and gate.get('optimizer_steps_authorized') is False
         and gate.get('checkpoint_reads_authorized') is False
         and gate.get('broker_access_allowed') is False
         and policy.get('training_enabled') is False
         and policy.get('native_training_enabled') is False
         and policy.get('native_launch_authorized') is False,
         'POLICY_GATE')
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry,
    )
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 4294967296
         and cap['swap'] == 536870912 and cap['pids'] == 64
         and cap['numerical_threads'] == 1
         and lock['scope'] == cap['cgroup_path'][len('/sys/fs/cgroup'):],
         'CAP_LOCK')
    need(not any(x.split('.')[0] in ('torch', 'sklearn', 'joblib')
                 for x in sys.modules), 'MODEL_IMPORT')
    old = bound_json(OLD_REFS['plan'])
    prior = bound_json(OLD_REFS['result'])
    terminal = bound_json(OLD_REFS['terminal'])
    need(prior['plan'] == OLD_REFS['plan'] and terminal['status'] == 'COMPLETE'
         and terminal['exit_code'] == 0 and terminal['result'] == OLD_REFS['result']
         and terminal['test_data_used'] is False, 'OLD_TERMINAL')
    records = check_witness_receipt(prior)
    inputs = old['inputs']
    need(old['scope']['witness_end_exclusive_ns'] == TRAIN_END_NS
         and old['test_data_used'] is False and len(inputs['native_parts']) == 16
         and [int(Path(x['path']).parent.name.split('=')[1]) for x in inputs['native_parts']]
             == list(range(2009, 2025)), 'OLD_INPUT_SCOPE')
    manifest = bound_json(inputs['cache_manifest'], max_bytes=64 * 1024 * 1024)
    fit = bound_json(inputs['fit_result'])
    enriched = bound_json(inputs['enriched_manifest'])
    native = bound_json(inputs['native_manifest'])
    need(fit['fold']['train_cutoff_ns'] == TRAIN_END_NS
         and fit['test_data_used'] is False
         and enriched['native_m5_source']['manifest_path'] == inputs['native_manifest']['path']
         and enriched['native_m5_source']['manifest_sha256'] == inputs['native_manifest']['sha256']
         and all(native['year_sha256'][Path(ref['path']).parent.name] == ref['sha256']
                 for ref in inputs['native_parts']), 'INPUT_LINEAGE')
    constants = manifest['v29_registry_constants']
    need(constants['per_tf_seq_lens']['D1'] == 252
         and constants['trendline_band_atr']['D1'] == 0.6968664882937518
         and [manifest['feature_names'].index(name) for name in NAMES]
             == list(range(159, 190)), 'D1_FEATURE_CONTRACT')
    refs = inputs['per_tf']['D1']
    need(set(refs) == {'feats', 'ts', 'replay_plan', 'replay_result', 'replay_terminal'},
         'D1_REFS')
    replay_plan = bound_json(refs['replay_plan'])
    replay_result = bound_json(refs['replay_result'])
    replay_terminal = bound_json(refs['replay_terminal'])
    need(replay_plan['inputs']['d1_feats'] == refs['feats']
         and replay_plan['inputs']['d1_ts'] == refs['ts']
         and replay_plan['inputs']['cache_manifest'] == inputs['cache_manifest']
         and replay_result['status'] == 'COMPLETE'
         and replay_result['input_price_precision'] == 'native_float64_full_history'
         and replay_result['source_clock_rows'] == 3889
         and replay_result['replay_parity'] is True
         and replay_result['all_causal_history_vs_cache']['mismatched_cells'] == 0
         and replay_result['window_configuration'] == {
             'timeframe': 'D1', 'seq_len': 252,
             'band_atr': 0.6968664882937518}
         and replay_result['test_data_used'] is False
         and replay_terminal['status'] == 'COMPLETE'
         and replay_terminal['exit_code'] == 0, 'D1_REPLAY_BINDING')
    return plan, inputs, manifest, records, {
        'execution': cap, 'lock': lock, 'source_stats': source_stats,
        'claim_ref': claim_ref, 'claim_stat': claim_stat,
    }, state


def run(plan_path: Path, plan_sha256: str) -> dict:
    plan_ref = {'path': str(plan_path), 'sha256': plan_sha256}
    plan, inputs, manifest, records, caps, original_state = preflight(plan_ref)
    output = Path(plan['output_dir'])
    old_alarm = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
        TimeoutError('D1_TRENDLINE_PHYSICAL_WALL_LIMIT')))
    signal.alarm(LIMITS['wall_seconds'])
    started = time.monotonic()
    try:
        refs = inputs['per_tf']['D1']
        source_stats = {}
        for ref in (*inputs['native_parts'], refs['feats'], refs['ts']):
            path, stat = checked(ref)
            source_stats[str(path)] = stat
        from gx1.scripts.audit_direction_d1_trendline_replay_v1 import load_native_parts_prefix
        from gx1.features.htf_features import (
            _resample_ohlcv, build_multi_tf_v4_closed_timestamp_indices,
        )
        from gx1.features.technical_indicators_v1 import ema50_200_spread_atr_block
        source = load_native_parts_prefix(inputs['native_parts'], TRAIN_END_NS)
        need(len(source) > 0 and source.index[-1].value < TRAIN_END_NS,
             'NATIVE_TRAIN_ONLY')
        index = build_multi_tf_v4_closed_timestamp_indices(source.index)['D1']
        bars = _resample_ohlcv(source, 'D1').dropna(
            subset=['open', 'high', 'low', 'close']).loc[index]
        need(bars.index.equals(index), 'D1_RESAMPLE_AXIS')
        atr = ema50_200_spread_atr_block(
            bars['high'], bars['low'], bars['close'])['atr14']
        cache_ts = np.load(refs['ts']['path'], mmap_mode='r', allow_pickle=False)
        cache = np.load(refs['feats']['path'], mmap_mode='r', allow_pickle=False)
        need(cache_ts.dtype == np.int64 and cache_ts.ndim == 1
             and cache.dtype == np.float32 and cache.ndim == 2
             and cache.shape == (len(cache_ts), manifest['feature_count']),
             'D1_CACHE_SHAPE')
        rows, positions, timestamps = align_train_rows(cache_ts, index.asi8)
        anchors = align_witnesses(cache_ts, index.asi8, records)
        for record, pos in zip(records, anchors):
            need(int(positions[record['cache_row'] - TRAIN_CACHE_ROWS[0]]) == pos,
                 'ANCHOR_IN_TRAIN_RANGE')
            need(float(bars.iloc[pos]['close']) == record['source_close']
                 and math.isclose(float(atr.iloc[pos]), record['source_atr14'],
                                  rel_tol=0, abs_tol=1e-12),
                 'SOURCE_WITNESS_' + str(record['cache_row']))
        causal_end = positions[-1] + 1
        oracle, event_counts = oracle_rows(
            bars['high'].to_numpy()[:causal_end],
            bars['low'].to_numpy()[:causal_end],
            bars['close'].to_numpy()[:causal_end],
            atr.to_numpy()[:causal_end],
            seq_len=252, band_atr=0.6968664882937518,
            witness_positions=positions)
        columns = list(range(159, 190))
        cached = np.asarray(cache[rows][:, columns], dtype=np.float32)
        comparison = compare_train_rows(oracle, cached, rows, timestamps)
        need(git_state() == original_state, 'FINAL_SOURCE_TREE')
        for path, stat in source_stats.items():
            need(regular(Path(path)) == stat, 'FINAL_SOURCE_STAT_' + Path(path).name)
        for name, stat in caps['source_stats'].items():
            need(regular(Path(plan['sources'][name]['path'])) == stat,
                 'FINAL_OWNER_STAT_' + name)
        need(regular(output / 'CLAIM.json') == caps['claim_stat']
             and {entry.name for entry in output.iterdir()} == {'CLAIM.json'},
             'FINAL_CLAIM_ONLY_ROOT')
        result = {
            'schema_version': 'gx1_d1_trendline_lifecycle_physical_result_v1',
            'status': 'COMPLETE',
            'decision': (
                'FAIL_D1_TRAIN_WINDOW' if comparison['mismatched_cells'] else
                'PASS_D1_TRAIN_WINDOW_19_EXERCISED'
                if not comparison['unexercised_lifecycle_fields'] else
                'PASS_D1_TRAIN_WINDOW_WITH_SUPPORT_GAPS'),
            'plan': plan_ref, 'claim': caps['claim_ref'],
            'source_head': original_state[0],
            'old_receipts': OLD_REFS, 'source_refs': plan['sources'],
            'native_parts': inputs['native_parts'],
            'cache_refs': {'feats': refs['feats'], 'ts': refs['ts']},
            'train_end_exclusive_ns': TRAIN_END_NS,
            'timestamp_rule': 'D1 cache label plus one UTC day is the TRAIN close',
            'anchor_witnesses': records,
            'train_cache_rows': {'first': 3588, 'last_inclusive': 3761, 'count': 174},
            'comparison': comparison, 'oracle_event_counts': event_counts,
            'remaining_lifecycle_fields_before_pilot': 19,
            'all_19_lifecycle_fields_globally_qualified': False,
            'all_five_timeframes_qualified': False,
            'model_forwards': 0, 'optimizer_steps': 0,
            'test_data_used': False, 'elapsed_seconds': time.monotonic() - started,
            'cap': {'execution': caps['execution'], 'lock': caps['lock']},
        }
        from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
        payload = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
        _publish_bytes(output / 'RESULT.json', payload)
        return result
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    run(Path(args.plan), args.plan_sha256)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
