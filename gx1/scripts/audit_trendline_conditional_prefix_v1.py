"""Bounded, conditional trendline arithmetic and representation audit.

This does not validate line discovery, touch history, lifecycle, or predictive use.
Numerical checks use only the replay-matched pre-2025-07-01 cache prefix;
whole-file SHA integrity reads cache bytes through 2026-06-30, but no sealed TEST.
Physical witnesses use the earlier May-December 2024 TRAIN interval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path('/home/andre2/src/GX1_CURRENT')
sys.path.insert(0, str(REPO))  # Direct script launch resolves the project package.

import numpy as np
from gx1.features.smc_v1 import SWING_LOOKBACK

sys.dont_write_bytecode = True

STAGE = 'trendline_conditional_prefix_20261106_001'
TF_ORDER = ('M5', 'M15', 'H1', 'H4', 'D1')
TF_SECONDS = {'M5': 300, 'M15': 900, 'H1': 3600, 'H4': 14400, 'D1': 86400}
CUTOFF_NS = 1751328000000000000  # Prior replay scope; not a TEST boundary.
TRAIN_START_NS = 1714521600000000000
TRAIN_END_NS = 1735689600000000000
SLOT_NUMERIC = (1, 2, 4, 5, 8, 9, 11, 12)
CHANNEL_NUMERIC = (26, 27, 28, 30)
NUMERIC_COLUMNS = SLOT_NUMERIC + CHANNEL_NUMERIC
CLASSES = {
    'above': lambda b: b[:, 0] > 0,
    'below': lambda b: b[:, 7] > 0,
    'parallel_channel': lambda b: (b[:, 25] == 1) & (b[:, 29] == 0),
    'converging_channel': lambda b: (b[:, 25] == 1) & (b[:, 29] == 1),
    'touch_above': lambda b: b[:, 14] == 1,
    'touch_below': lambda b: b[:, 15] == 1,
    'break_up': lambda b: b[:, 16] == 1,
    'break_down': lambda b: b[:, 17] == 1,
    'retest_hold_up': lambda b: b[:, 20] == 1,
    'retest_fail_up': lambda b: b[:, 21] == 1,
    'retest_hold_down': lambda b: b[:, 22] == 1,
    'retest_fail_down': lambda b: b[:, 23] == 1,
}


def require(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError('TRENDLINE_CONDITIONAL_' + reason)


def sha256(path: Path) -> str:
    require(path.is_absolute() and path.is_file() and not path.is_symlink()
            and path.resolve(strict=True) == path, 'REGULAR_PATH_' + path.name)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def bound_json(ref: dict) -> dict:
    require(set(ref) == {'path', 'sha256'} and isinstance(ref['sha256'], str)
            and len(ref['sha256']) == 64, 'REFERENCE_SHAPE')
    path = Path(ref['path'])
    require(sha256(path) == ref['sha256'], 'REFERENCE_SHA_' + path.name)
    return json.loads(path.read_text())


def publish(path: Path, value: dict) -> None:
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    _publish_bytes(path, payload)


def git_state() -> tuple[str, str, str]:
    def run(*args: str) -> str:
        return subprocess.check_output(('git', *args), cwd=REPO, text=True).strip()
    return run('rev-parse', 'HEAD'), run('branch', '--show-current'), run('status', '--porcelain')


def preflight(plan_ref: dict) -> tuple[dict, dict, dict, dict]:
    plan = bound_json(plan_ref)
    require(plan['schema_version'] == 'gx1_trendline_conditional_prefix_plan_v1'
            and plan['stage'] == STAGE
            and plan['status'] == 'PREPARED_NOT_ADMITTED'
            and plan['scope'] == {'cache_prefix_exclusive_ns': CUTOFF_NS,
                                  'witness_start_ns': TRAIN_START_NS,
                                  'witness_end_exclusive_ns': TRAIN_END_NS,
                                  'timeframes': list(TF_ORDER),
                                  'test_data_used': False}
            and plan['limits'] == {'memory_max': '4G', 'swap_max': '512M',
                                   'wall_seconds': 1800, 'execution_class': 'audit',
                                   'model_forwards': 0, 'optimizer_steps': 0}
            and plan['test_data_used'] is False, 'PLAN_CONTRACT')
    head, branch, dirty = git_state()
    require(branch == 'work/gx1-current' and dirty == '', 'SOURCE_TREE')
    policy = json.loads((REPO / 'NEXT_RUN_POLICY.json').read_text())
    gate = policy.get(STAGE)
    require(gate is not None and gate.get('status') == 'ADMITTED'
            and gate.get('plan') == plan_ref
            and gate.get('execution_authorized') is True
            and gate.get('claimed_invocations') == 0
            and gate.get('completed_invocations') == 0
            and gate.get('max_invocations') == 1
            and gate.get('relaunch_allowed') is False
            and gate.get('test_data_used') is False
            and gate.get('model_forwards_authorized') is False
            and gate.get('optimizer_steps_authorized') is False
            and gate.get('checkpoint_reads_authorized') is False
            and gate.get('broker_access_allowed') is False, 'POLICY_GATE')
    require(policy.get('training_enabled') is False
            and policy.get('native_training_enabled') is False
            and policy.get('native_launch_authorized') is False, 'TRAINING_GATE')
    require(set(plan['sources']) == {'worker', 'producer', 'replay_loader',
                                    'resample', 'atr', 'pivot', 'capped_runner',
                                    'publication'}, 'SOURCE_KEYS')
    for name, ref in plan['sources'].items():
        require(sha256(Path(ref['path'])) == ref['sha256'], 'SOURCE_SHA_' + name)
    require(Path(plan['sources']['worker']['path']) == Path(__file__).resolve(), 'WORKER_PATH')
    require(set(plan['inputs']) == {'cache_manifest', 'enriched_manifest',
                                    'source_m5', 'native_manifest', 'native_parts',
                                    'fit_result', 'qualified_index', 'per_tf'},
            'INPUT_KEYS')
    meta = {name: bound_json(plan['inputs'][name])
            for name in ('cache_manifest', 'enriched_manifest',
                         'native_manifest', 'fit_result', 'qualified_index')}
    require(meta['fit_result']['fold']['train_cutoff_ns'] == TRAIN_END_NS
            and meta['fit_result']['test_data_used'] is False
            and meta['qualified_index']['qualified_per_tf'] == 158
            and meta['qualified_index']['remaining_per_tf'] == 95
            and meta['enriched_manifest']['output_parquet'] == plan['inputs']['source_m5']['path']
            and meta['enriched_manifest']['output_parquet_sha256'] == plan['inputs']['source_m5']['sha256'],
            'METADATA_LINEAGE')
    native = meta['enriched_manifest']['native_m5_source']
    require(native['manifest_path'] == plan['inputs']['native_manifest']['path']
            and native['manifest_sha256'] == plan['inputs']['native_manifest']['sha256'],
            'NATIVE_MANIFEST_LINEAGE')
    parts = plan['inputs']['native_parts']
    expected_parts = [path for path in native['part_paths']
                      if int(Path(path).parent.name.split('=')[1]) <= 2024]
    require(len(parts) == 16 and [r['path'] for r in parts] == expected_parts
            and all(r['sha256'] == native['part_sha256'][r['path']] for r in parts),
            'NATIVE_PART_LINEAGE')
    require(set(plan['inputs']['per_tf']) == set(TF_ORDER), 'TF_KEYS')
    for tf in TF_ORDER:
        refs = plan['inputs']['per_tf'][tf]
        require(set(refs) == {'feats', 'ts', 'replay_plan', 'replay_result',
                              'replay_terminal'}, 'TF_REF_KEYS_' + tf)
        replay_plan = bound_json(refs['replay_plan'])
        replay_result = bound_json(refs['replay_result'])
        replay_terminal = bound_json(refs['replay_terminal'])
        ck = 'd1_feats' if tf == 'D1' else 'tf_feats'
        tk = 'd1_ts' if tf == 'D1' else 'tf_ts'
        require(replay_plan['inputs']['cache_manifest'] == plan['inputs']['cache_manifest']
                and replay_plan['inputs'][ck] == refs['feats']
                and replay_plan['inputs'][tk] == refs['ts']
                and replay_plan['sources']['gx1/features/trendline_registry_v1.py']['sha256']
                    == plan['sources']['producer']['sha256']
                and replay_result['replay_parity'] is True
                and replay_result['all_causal_history_vs_cache']['mismatched_cells'] == 0
                and replay_result['all_causal_history_vs_cache']['fields'] == 31
                and replay_result['test_data_used'] is False
                and replay_terminal['status'] == 'COMPLETE'
                and replay_terminal['exit_code'] == 0
                and policy['independent_direction_' + tf.lower()
                           + '_native_history_replay_20261028_001']['result']
                    == refs['replay_result'], 'REPLAY_BINDING_' + tf)
    from gx1.contracts import gx1_capped_execution_v1 as cap
    execution = cap.require_capped_cpu_audit_execution()
    lock = cap.require_capped_lock_ancestry()
    require(lock['scope'] == execution['cgroup_path'][len('/sys/fs/cgroup'):],
            'CAPPED_LOCK')
    require(not any(x.split('.')[0] in ('torch', 'sklearn', 'joblib')
                    for x in sys.modules), 'MODEL_MODULE_IMPORTED')
    return (plan, meta, {'head': head, 'branch': branch},
            {'execution': execution, 'lock': lock})


def cache_prefix_scan(tf: str, refs: dict, manifest: dict,
                      replay_result: dict, band: float) -> tuple[dict, dict]:
    names = manifest['feature_names']
    from gx1.features.trendline_registry_v1 import TRENDLINE_REGISTRY_FEATURE_NAMES_V1
    columns = [names.index(name) for name in TRENDLINE_REGISTRY_FEATURE_NAMES_V1]
    require(len(columns) == len(set(columns)) == 31, 'COLUMN_LAYOUT_' + tf)
    ts = np.load(refs['ts']['path'], mmap_mode='r', allow_pickle=False)
    feats = np.load(refs['feats']['path'], mmap_mode='r', allow_pickle=False)
    limit = int(replay_result['all_causal_history_vs_cache']['rows'])
    require(ts.dtype == np.int64 and ts.ndim == 1
            and feats.dtype == np.float32 and feats.ndim == 2
            and feats.shape == (len(ts), manifest['feature_count'])
            and 0 < limit <= len(ts)
            and int(ts[limit-1]) + TF_SECONDS[tf] * 10**9 <= CUTOFF_NS
            and (limit == len(ts) or (int(ts[limit]) > int(ts[limit-1])
                 and int(ts[limit]) + TF_SECONDS[tf] * 10**9 > CUTOFF_NS)),
            'PREFIX_AXIS_' + tf)
    selected: dict[str, int] = {}
    valid_rows = 0
    age_last_break = None
    age_last_sign = None
    for left in range(0, limit, 8192):
        right = min(left + 8192, limit)
        block = np.asarray(feats[left:right, columns], dtype=np.float64)
        clock = np.asarray(ts[left:right], dtype=np.int64)
        require(np.all(np.diff(clock) > 0) and not np.isinf(block).any(),
                'CACHE_DOMAIN_' + tf + '_' + str(left))
        valid = np.isfinite(block[:, 0])
        require(np.isnan(block[~valid]).all()
                and np.isfinite(block[valid][:, [i for i in range(31) if i != 24]]).all(),
                'CACHE_MISSINGNESS_' + tf + '_' + str(left))
        b = block[valid]
        valid_rows += len(b)
        if len(b):
            for base in (0, 7):
                count = b[:, base]
                require(np.all(count >= 0) and np.all(count == np.floor(count)),
                        'SLOT_COUNT_' + tf + '_' + str(base))
                empty = count == 0
                require(np.all(b[empty, base+1:base+7] == 0),
                        'EMPTY_SLOT_' + tf + '_' + str(base))
                occupied = ~empty
                require(np.all(b[occupied, base+1] >= 0)
                        and np.all(b[occupied, base+3] >= 3)
                        and np.all(b[occupied, base+3] == np.floor(b[occupied, base+3]))
                        and np.all(b[occupied, base+4:base+6] >= 0)
                        and np.all(b[occupied, base+4:base+6] == np.floor(b[occupied, base+4:base+6]))
                        and np.all((b[occupied, base+6] >= 0)
                                   & (b[occupied, base+6] <= band + 1e-5)),
                        'OCCUPIED_SLOT_' + tf + '_' + str(base))
            flags = b[:, [14, 15, 16, 17, 20, 21, 22, 23, 25, 29]]
            require(np.all((flags == 0) | (flags == 1)), 'BINARY_FLAGS_' + tf)
            breaks = (b[:, 16] == 1) | (b[:, 17] == 1)
            require(np.all(b[~breaks, 18:20] == 0)
                    and np.all(b[breaks, 18] >= 3)
                    and np.all(b[breaks, 18:20] == np.floor(b[breaks, 18:20]))
                    and np.all(b[breaks, 19] >= 0), 'BREAK_METADATA_' + tf)
            inactive = b[:, 25] == 0
            active = ~inactive
            require(np.all(b[inactive, 26:31] == 0)
                    and np.all(b[active, 26] > 0)
                    and np.all((b[active, 27] >= 0) & (b[active, 27] <= 1))
                    and np.all((b[active, 30] >= 0) & (b[active, 30] <= 1))
                    and np.all(b[active & (b[:, 29] == 0), 30] == 0)
                    and np.all((b[b[:, 29] == 1, 0] > 0)
                               & (b[b[:, 29] == 1, 7] > 0)),
                    'CHANNEL_REPRESENTATION_' + tf)
            valid_global = np.flatnonzero(valid) + left
            for j, global_row in enumerate(valid_global):
                up = b[j, 16] == 1
                down = b[j, 17] == 1
                if up or down:
                    age_last_break = int(global_row)
                    age_last_sign = 1 if up and not down else -1 if down and not up else None
                if age_last_break is not None:
                    age = b[j, 24]
                    require(np.isfinite(age) and abs(age) == global_row - age_last_break
                            and (age_last_sign is None or age == age_last_sign * (global_row-age_last_break)),
                            'BREAK_CLOCK_' + tf + '_' + str(global_row))
        train = (clock + TF_SECONDS[tf] * 10**9 >= TRAIN_START_NS) & (clock + TF_SECONDS[tf] * 10**9 < TRAIN_END_NS)
        for label, predicate in CLASSES.items():
            if label not in selected:
                at = np.flatnonzero(train & valid & predicate(block))
                if len(at):
                    selected[label] = left + int(at[0])
    require(valid_rows > 0 and set(selected) == set(CLASSES),
            'TRAIN_WITNESS_COVERAGE_' + tf)
    return {'scanned_rows': limit, 'valid_rows': valid_rows,
            'selected_witness_cache_rows': selected,
            'all_12_train_witness_classes_present': True,
            'all_31_representation_invariants_pass': True,
            'scope': 'numerical scan of replay-matched pre-2025-07-01 cache prefix; full cache files read for SHA-256 integrity only; no sealed TEST'}, selected


def independent_projection(line: object, t: int) -> tuple[float, float]:
    a = float(line.anchor1_bar)
    b = float(line.anchor2_bar)
    p = float(line.anchor1_price)
    q = float(line.anchor2_price)
    slope = (q - p) / (b - a)
    projected = p + slope * (float(t) - a)
    return slope, projected


def conditional_numeric(state: object, t: int, close: float,
                        atr: float, band: float, seq_len: int) -> dict[int, float]:
    active = [line for line in state.active_lines if line.state == 'active']
    above = min((line for line in active if line.projection(t) >= close),
                key=lambda line: (line.projection(t)-close, line.line_id), default=None)
    below = min((line for line in active if line.projection(t) < close),
                key=lambda line: (close-line.projection(t), line.line_id), default=None)
    expected = {i: 0.0 for i in NUMERIC_COLUMNS}
    for base, line in ((0, above), (7, below)):
        if line is None:
            continue
        slope, projected = independent_projection(line, t)
        expected[base+1] = abs(projected-close) / atr
        expected[base+2] = slope / atr
        expected[base+4] = float(t - (line.anchor1_bar + SWING_LOOKBACK))
        expected[base+5] = float(t - line.last_touch_bar)
    if above is not None and below is not None:
        s_up, p_up = independent_projection(above, t)
        s_dn, p_dn = independent_projection(below, t)
        if below.side == 1 and above.side == -1 and s_dn > s_up and p_up > p_dn:
            width = p_up-p_dn
            apex = ((above.anchor1_price - s_up*above.anchor1_bar)
                    - (below.anchor1_price - s_dn*below.anchor1_bar)) / (s_dn-s_up)
            expected.update({26: width/atr,
                             27: min(1.0, max(0.0, (close-p_dn)/width)),
                             28: (s_up+s_dn)/(2.0*atr),
                             30: min(1.0, max(0.0, 1.0-(apex-t)/seq_len))})
            return expected
    candidates = []
    for line in (above, below):
        if line is None:
            continue
        slope, _ = independent_projection(line, t)
        own_offset = line.anchor1_price-slope*line.anchor1_bar
        opposite = state.resistance_pivots if line.side == 1 else state.support_pivots
        offsets = [price-slope*bar for bar, price in opposite]
        eligible = [x for x in offsets if x > own_offset] if line.side == 1 else [x for x in offsets if x < own_offset]
        if not eligible:
            continue
        extreme = max(eligible) if line.side == 1 else min(eligible)
        members = [x for x in eligible if abs(x-extreme) <= band*atr]
        if len(members) < 2:
            continue
        width = abs(extreme-own_offset)
        if width <= 0:
            continue
        lower = own_offset if line.side == 1 else extreme
        lower_projection = lower+slope*t
        vals = {26: width/atr,
                27: min(1.0, max(0.0, (close-lower_projection)/width)),
                28: slope/atr, 30: 0.0}
        distance = abs(line.projection(t)-close)
        candidates.append((distance, 0 if line is above else 1, vals))
    if candidates:
        _, _, vals = min(candidates, key=lambda item: (item[0], item[1]))
        expected.update(vals)
    return expected


def witness_replay(tf: str, source: object, selected: dict, refs: dict,
                   manifest: dict, band: float, seq_len: int) -> list[dict]:
    from gx1.features.htf_features import (_resample_ohlcv, multi_tf_resample,
                                           build_multi_tf_v4_closed_timestamp_indices)
    from gx1.features.technical_indicators_v1 import ema50_200_spread_atr_block
    from gx1.features.trendline_registry_v1 import compute_trendline_registry_features_v1
    index = build_multi_tf_v4_closed_timestamp_indices(source.index)[tf]
    if tf == 'M5':
        bars = source.loc[index, ['open', 'high', 'low', 'close', 'volume']]
    elif tf in ('M15', 'H1'):
        bars = multi_tf_resample(source, tf).agg({'open': 'first', 'high': 'max',
              'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna(subset=['open','high','low','close']).loc[index]
    else:
        bars = _resample_ohlcv(source, tf).dropna(subset=['open','high','low','close']).loc[index]
    require(bars.index.equals(index), 'WITNESS_SOURCE_AXIS_' + tf)
    atr = ema50_200_spread_atr_block(bars['high'], bars['low'], bars['close'])['atr14']
    data = bars[['high','low','close']].copy()
    data['atr'] = atr
    ts = np.load(refs['ts']['path'], mmap_mode='r', allow_pickle=False)
    feats = np.load(refs['feats']['path'], mmap_mode='r', allow_pickle=False)
    columns = [manifest['feature_names'].index(name)
               for name in __import__('gx1.features.trendline_registry_v1', fromlist=['TRENDLINE_REGISTRY_FEATURE_NAMES_V1']).TRENDLINE_REGISTRY_FEATURE_NAMES_V1]
    by_row: dict[int, list[str]] = {}
    for label, row in selected.items():
        by_row.setdefault(row, []).append(label)
    targets = []
    for row in sorted(by_row):
        timestamp = int(ts[row])
        pos = int(np.searchsorted(index.asi8, timestamp))
        require(pos < len(index) and int(index.asi8[pos]) == timestamp,
                'WITNESS_JOIN_' + tf + '_' + str(row))
        targets.append((pos, row, timestamp))
    state = None
    cursor = 0
    records = []
    for pos, row, timestamp in targets:
        while cursor < pos:
            end = min(pos, cursor+8192)
            _, state = compute_trendline_registry_features_v1(
                data.iloc[cursor:end], timeframe=tf, seq_len=seq_len,
                band_atr=band, state=state)
            cursor = end
        emitted, state = compute_trendline_registry_features_v1(
            data.iloc[pos:pos+1], timeframe=tf, seq_len=seq_len,
            band_atr=band, state=state)
        cursor = pos+1
        actual = np.asarray(feats[row, columns], dtype=np.float32)
        produced = emitted.to_numpy(dtype=np.float32)[0]
        require(np.array_equal(np.isnan(actual), np.isnan(produced))
                and np.array_equal(actual[~np.isnan(actual)].view(np.uint32),
                                   produced[~np.isnan(produced)].view(np.uint32)),
                'WITNESS_PRODUCER_CACHE_' + tf + '_' + str(row))
        t = state.bar_count-1
        raw_close = float(data.iloc[pos]['close'])
        raw_atr = float(data.iloc[pos]['atr'])
        require(math.isfinite(raw_atr) and raw_atr > 0 and t == pos,
                'WITNESS_CLOCK_' + tf + '_' + str(row))
        independent = conditional_numeric(state, t, raw_close, raw_atr, band, seq_len)
        errors = {str(i): abs(float(actual[i])-value) for i, value in independent.items()}
        require(all(math.isclose(float(actual[i]), value, rel_tol=2e-5, abs_tol=1e-4)
                    for i, value in independent.items()),
                'NUMERIC_WITNESS_' + tf + '_' + str(row))
        records.append({'cache_row': row, 'timestamp_ns': timestamp,
                        'classes': by_row[row], 'native_bar_index': t,
                        'source_close': raw_close, 'source_atr14': raw_atr,
                        'conditional_numeric_columns': list(NUMERIC_COLUMNS),
                        'max_absolute_error': max(errors.values()),
                        'producer_cache_float32_bit_parity': True})
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    plan_ref = {'path': args.plan, 'sha256': args.plan_sha256}
    plan, meta, source_state, cap_state = preflight(plan_ref)
    out = Path(plan['output_dir'])
    require(out.parent.is_dir() and not out.exists() and not out.is_symlink(), 'OUTPUT_SCOPE')
    started = time.monotonic()
    old_alarm = signal.getsignal(signal.SIGALRM)
    def expired(*unused: object) -> None:
        raise RuntimeError('TRENDLINE_CONDITIONAL_WALL_LIMIT')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(1800)
    out.mkdir(exist_ok=False)
    try:
        for ref in plan['inputs']['native_parts']:
            require(sha256(Path(ref['path'])) == ref['sha256'], 'PART_SHA')
        scans = {}
        selections = {}
        for tf in TF_ORDER:
            refs = plan['inputs']['per_tf'][tf]
            for key in ('feats', 'ts'):
                require(sha256(Path(refs[key]['path'])) == refs[key]['sha256'],
                        'CACHE_SHA_' + tf + '_' + key)
            replay_result = bound_json(refs['replay_result'])
            band = float(meta['cache_manifest']['v29_registry_constants']['trendline_band_atr'][tf])
            scans[tf], selections[tf] = cache_prefix_scan(tf, refs,
                meta['cache_manifest'], replay_result, band)
        from gx1.scripts.audit_direction_d1_trendline_replay_v1 import load_native_parts_prefix
        source = load_native_parts_prefix(plan['inputs']['native_parts'], TRAIN_END_NS)
        require(len(source) > 0 and source.index[-1].value < TRAIN_END_NS,
                'NATIVE_TRAIN_ONLY')
        witnesses = {}
        for tf in TF_ORDER:
            band = float(meta['cache_manifest']['v29_registry_constants']['trendline_band_atr'][tf])
            seq_len = int(meta['cache_manifest']['v29_registry_constants']['per_tf_seq_lens'][tf])
            witnesses[tf] = witness_replay(tf, source, selections[tf],
                plan['inputs']['per_tf'][tf], meta['cache_manifest'], band, seq_len)
        require(git_state() == (source_state['head'], source_state['branch'], ''),
                'FINAL_SOURCE_TREE')
        result = {
            'schema_version': 'gx1_trendline_conditional_prefix_result_v1',
            'status': 'COMPLETE', 'plan': plan_ref, 'source_head': source_state['head'],
            'cache_prefix_exclusive_ns': CUTOFF_NS,
            'witness_start_ns': TRAIN_START_NS,
            'witness_end_exclusive_ns': TRAIN_END_NS,
            'cache_prefix_representation': scans,
            'physical_train_witnesses': witnesses,
            'independently_checked_numeric_columns': list(NUMERIC_COLUMNS),
            'conditionally_checked_numeric_fields': [
                'geomline_above_dist_atr', 'geomline_above_slope_atr_per_bar',
                'geomline_above_age_bars', 'geomline_above_last_touch_age_bars',
                'geomline_below_dist_atr', 'geomline_below_slope_atr_per_bar',
                'geomline_below_age_bars', 'geomline_below_last_touch_age_bars',
                'geomchan_width_atr', 'geomchan_pos_0_1',
                'geomchan_slope_atr_per_bar', 'geomchan_apex_proximity'],
            'remaining_lifecycle_fields': 19,
            'semantic_line_selection_qualified': False,
            'full_31_field_independent_numeric_qualification': False,
            'new_model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False,
            'limits': 'Current producer supplies line state and identity. Arithmetic is recomputed from anchors and physical TRAIN candles. Prefix invariants check representation, not lifecycle truth. No predictive or economic conclusion.'}
        publish(out / 'RESULT.json', result)
        result_ref = {'path': str(out / 'RESULT.json'),
                      'sha256': sha256(out / 'RESULT.json')}
        publish(out / 'TERMINAL.json', {
            'status': 'COMPLETE', 'exit_code': 0, 'elapsed_seconds': time.monotonic()-started,
            'source_unchanged': True, 'cap': cap_state,
            'result': result_ref, 'test_data_used': False})
        return 0
    except Exception as exc:
        publish(out / 'TERMINAL.json', {
            'status': 'FAILED', 'exit_code': 1,
            'error': type(exc).__name__ + ': ' + str(exc),
            'relaunch_allowed': False, 'test_data_used': False})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)


if __name__ == '__main__':
    raise SystemExit(main())
