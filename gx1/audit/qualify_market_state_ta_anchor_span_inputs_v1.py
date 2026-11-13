"""One-use TRAIN input extension: recover two spans, preserve all 31 old fields.

Old physical caches and normalization remain immutable. No target decode,
normalization fit, model, checkpoint, optimizer, TEST or trading access.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

import numpy as np

from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO = Path('/home/andre2/src/GX1_CURRENT')
RUNS = Path('/home/andre2/GX1_RUNS')
STAGE = 'market_state_ta_trendline_anchor_span_inputs_20261112_001'
ROOT = RUNS / STAGE.upper()
PLAN = REPO / 'configs/research' / (STAGE.upper() + '.json')
POLICY = REPO / 'NEXT_RUN_POLICY.json'
SOURCE_BASE = 'e687294415d6376af14f8a00f105cfd9bb8c091e'
FIELDS = ('geomline_above_anchor_span_bars', 'geomline_below_anchor_span_bars')
TFS = ('M5', 'M15', 'H1', 'H4', 'D1')
BAR_NS = dict(M5=300, M15=900, H1=3600, H4=14400, D1=86400)
RANGES = dict(local=[987027, 1016809], M5=[987326, 1017028],
              M15=[329741, 339700], H1=[83079, 85650],
              H4=[21368, 22112], D1=[3336, 3696])
CUTOFF = 1727654400000000000
WALL_SECONDS = 1800
OLD_NORM = dict(path=str(RUNS / 'MARKET_STATE_MATURE_FOLD1_NORMALIZATION_20261107_001'
                         / 'PREFIX_NORMALIZATION_FOLD_1.json'),
                sha256='e24a2b8d1ae09f6f10c2de7a7097353b52aa2f2118287535e242778272022d54')
LOCAL_PLAN = REPO / 'configs/research/MARKET_STATE_LOCAL_TRENDLINE_FORMULA_DIAGNOSTIC_20261108_001.json'
REUSE = RUNS / 'NATIVE_M5_DIRECTION_NATIVE_PREFIX_INPUT_BINDING_REPAIR_20261024_264/DECODE_REUSE_BINDING.json'
SOURCE_QUAL = RUNS / 'MARKET_STATE_TA_TRENDLINE_IDENTITY_SOURCE_20261112_001/SOURCE_QUALIFICATION.json'
SOURCES = (
    'gx1/audit/qualify_market_state_ta_anchor_span_inputs_v1.py',
    'gx1/scripts/audit_direction_d1_trendline_replay_v1.py',
    'gx1/features/trendline_registry_v1.py',
    'gx1/features/technical_indicators_v1.py',
    'gx1/features/htf_features.py',
    'gx1/features/entry_model_native_feature_layers_v1.py',
    'gx1/features/entry_specialist_feature_groups_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/models/entry_v10/entry_v10_input_normalization.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_market_state_ta_anchor_span_inputs_v1.py',
)
SCOPE = dict(fold_id=1, parent_rows=29574, ranges=RANGES,
             history_end_exclusive_ns=CUTOFF, new_fields=list(FIELDS),
             old_registry_fields=31, new_registry_fields=33,
             producer_calls=6, target_arrays_decoded=0,
             normalization_fits=0, model_constructions=0, model_forwards=0,
             checkpoint_reads=0, optimizer_steps=0, test_data_used=False)


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError('ANCHOR_SPAN_INPUT_' + reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def read(item, limit=64 * 1024**2):
    p = Path(item['path'])
    need(p.is_file() and not p.is_symlink() and p.resolve() == p
         and p.stat().st_size <= limit and sha(p) == item['sha256'], 'JSON_BINDING:' + p.name)
    return json.loads(p.read_text())


def stat(path):
    p = Path(path)
    need(p.is_file() and not p.is_symlink() and p.resolve() == p, 'REGULAR:' + p.name)
    s = p.stat()
    return {k: getattr(s, k) for k in
            ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}


def axis_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype='<i8').tobytes()).hexdigest()


def row_sha(rows, lane):
    from gx1.models.entry_v10.entry_v10_input_normalization import _hash_int64_indices
    return _hash_int64_indices(rows, namespace='entry_local_m5_rows' if lane == 'local' else 'mtf:' + lane)


def build_plan():
    """Metadata and file stats only; no market/target array decoding."""
    from gx1.features.trendline_registry_v1 import TRENDLINE_REGISTRY_FEATURE_NAMES_V1, SWING_LOOKBACK
    from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4
    old = read(ref(LOCAL_PLAN))
    normal = read(OLD_NORM)
    source = read(old['inputs']['source_manifest'])
    manifest = read(old['inputs']['registry_manifest'])
    reuse = read(ref(REUSE))
    qualified = read(ref(SOURCE_QUAL))
    need(qualified['registry_field_count'] == 33 and qualified['real_market_data_reads'] == 0,
         'SOURCE_QUALIFICATION')
    need(all(sha(REPO / p) == x['sha256'] for p, x in qualified['source_owners'].items()),
         'QUALIFIED_SOURCE_CHANGED')
    names = list(TRENDLINE_REGISTRY_FEATURE_NAMES_V1)
    legacy = [name for name in names if name not in FIELDS]
    need(len(names) == 33 and len(legacy) == 31 and SWING_LOOKBACK == 3,
         'CURRENT_REGISTRY')
    need(old['parameters'] == dict(seq_len=96, band_atr=0.9865694402555613),
         'LOCAL_PARAMETERS')
    need(manifest['feature_count'] == 253 and len(source['feature_field_order']) == 2725
         and [x for x in MULTI_TF_PER_BAR_FEATURES_V4 if x not in FIELDS] == manifest['feature_names'],
         'NAME_ADDITION_ONLY')
    need(normal['base_normalization']['lineage']['mtf_cache_manifest_sha256']
         == old['inputs']['registry_manifest']['sha256'], 'NORMALIZATION_CACHE_BINDING')
    pop = dict(local=normal['prefix_qualification']['proof']['local'],
               **normal['base_normalization']['lineage']['per_tf_fit_windows'])
    populations = {}
    for lane, bounds in RANGES.items():
        left, right = bounds
        q = pop[lane]
        need(right - left == q['selected_unique_row_count']
             and row_sha(np.arange(left, right, dtype=np.int64), lane) == q['selected_row_indices_sha256'],
             'EXACT_RETAINED_ROWS:' + lane)
        if lane != 'local':
            need([q['left_index_inclusive'], q['right_index_exclusive']] == bounds,
                 'EXACT_RETAINED_INTERVAL:' + lane)
        populations[lane] = {k: q[k] for k in
            ('selected_unique_row_count', 'selected_row_indices_sha256', 'time_min_utc', 'time_max_utc')}
        populations[lane]['available_ns_sha256'] = (normal['prefix_qualification']['proof']['local'] if lane == 'local'
            else normal['prefix_qualification']['proof']['tf'][lane])['available_ns_sha256']
    signal_paths = [p for p in reuse['scratch_files'] if p.endswith('/source050/signal.mmap')]
    need(len(signal_paths) == 1, 'LOCAL_SIGNAL_OWNER')
    local_path = signal_paths[0]
    need(stat(local_path) == reuse['scratch_files'][local_path]
         and not stat(local_path)['st_mode'] & 0o222, 'READONLY_SIGNAL_STAT')
    caches = {}
    for tf in TFS:
        data = manifest['tfs'][tf]
        parent = Path(old['inputs']['registry_manifest']['path']).parent
        caches[tf] = {
            name: dict(path=str(parent / data[name + '_npy']), sha256=data[name + '_npy_sha256'])
            for name in ('feats', 'ts')}
    parameters = dict(local={**old['parameters'], 'timeframe': 'M5', 'swing_lookback': SWING_LOOKBACK})
    for tf in TFS:
        parameters[tf] = dict(timeframe=tf, seq_len=manifest['v29_registry_constants']['per_tf_seq_lens'][tf],
                              band_atr=manifest['v29_registry_constants']['trendline_band_atr'][tf],
                              swing_lookback=SWING_LOOKBACK)
    inputs = {key: old['inputs'][key] for key in
              ('enriched', 'enriched_manifest', 'source_manifest', 'registry_manifest', 'native_manifest', 'native_parts')}
    inputs.update(old_normalization=OLD_NORM, local_producer_plan=ref(LOCAL_PLAN),
                  decode_reuse_binding=ref(REUSE), source_qualification=ref(SOURCE_QUAL))
    return dict(schema_version='gx1_ta_anchor_span_input_plan_v1', stage=STAGE,
                source_base_head=SOURCE_BASE,
                source_owners={p: sha(REPO / p) for p in SOURCES}, inputs=inputs,
                populations=populations, parameters=parameters, caches=caches,
                local_signal=dict(path=local_path, stat=stat(local_path), shape=[1140763, 2725],
                                  columns=[source['feature_field_order'].index('chart.' + n) for n in legacy]),
                local_clock_path=source['output_parquet'], local_clock_stat=stat(source['output_parquet']),
                legacy_names=legacy, registry_names=names,
                mtf_legacy_columns=[manifest['feature_names'].index(n) for n in legacy],
                scope=SCOPE, output_root=str(ROOT),
                execution=dict(module=__name__ if __name__ != '__main__' else
                               'gx1.audit.qualify_market_state_ta_anchor_span_inputs_v1',
                               runner='scripts/gx1_capped_run.sh', job_class='audit',
                               memory='4G', swap='512M', timeout_seconds=WALL_SECONDS,
                               kill_after_seconds=10))


def inspect(plan_sha, *, execute):
    plan_ref = dict(path=str(PLAN), sha256=plan_sha)
    plan = read(plan_ref)
    need(plan == build_plan(), 'PLAN_IDENTITY')
    need(Path.cwd().resolve() == REPO
         and subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'work/gx1-current'
         and subprocess.check_output(['git', 'status', '--porcelain'], text=True) == '', 'CLEAN_SOURCE')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    need(subprocess.run(['git', 'merge-base', '--is-ancestor', SOURCE_BASE, head]).returncode == 0,
         'SOURCE_ANCESTRY')
    need(not ROOT.exists() or (ROOT.is_dir() and not ROOT.is_symlink() and not list(ROOT.iterdir())),
         'UNCONSUMED_OUTPUT_ROOT')
    policy_ref = ref(POLICY)
    policy = read(policy_ref)
    need(all(policy.get(k) is False for k in
             ('training_enabled', 'native_training_enabled', 'native_launch_authorized', 'full_val_allowed')),
         'TRAINING_CLOSED')
    if execute:
        gate = policy.get(STAGE, {})
        need(gate.get('status') == 'ADMITTED' and gate.get('plan') == plan_ref
             and gate.get('execution_authorized') is True and gate.get('data_reads_authorized') is True
             and gate.get('training_authorized') is False
             and gate.get('max_invocations') == gate.get('remaining_invocations') == 1
             and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
             and gate.get('relaunch_allowed') is False and gate.get('scope') == SCOPE, 'EXACT_ADMISSION')
    return plan, plan_ref, policy_ref, head


def compare_and_extract(values, observed, names, positions, *, lookback):
    """Require old-field parity before admitting any new values."""
    names = list(names)
    legacy = [n for n in names if n not in FIELDS]
    a = np.ascontiguousarray(values[:, [names.index(n) for n in legacy]], dtype=np.float32)
    b = np.ascontiguousarray(observed, dtype=np.float32)
    need(a.shape == b.shape == (len(positions), 31) and np.isfinite(a).all()
         and np.isfinite(b).all(), 'LEGACY_FINITE_SHAPE')
    mismatch = a.view(np.uint32) != b.view(np.uint32)
    need(not mismatch.any(), 'LEGACY_FLOAT32_BITS:' + str(int(mismatch.sum())))
    spans = np.ascontiguousarray(values[:, [names.index(n) for n in FIELDS]], dtype=np.float32)
    need(np.isfinite(spans).all(), 'SPAN_FINITE')
    reports = {}
    for j, side in enumerate(('above', 'below')):
        active = values[:, names.index('geomline_' + side + '_active_count')] > 0
        age = values[:, names.index('geomline_' + side + '_age_bars')]
        span = spans[:, j]
        need(np.all(span[~active] == 0) and np.all(span[active] > 0)
             and np.all(span == np.floor(span)) and np.all(age == np.floor(age)),
             'SPAN_DOMAIN:' + side)
        first = positions - age - lookback
        second = first + span
        need(np.all(first[active] >= 0) and np.all(second[active] <= positions[active] - lookback),
             'ANCHOR_CAUSALITY:' + side)
        reports[side] = dict(active_rows=int(active.sum()), absent_rows=int((~active).sum()),
                            unique_positive_spans=int(len(np.unique(span[active]))),
                            min_positive_span=float(span[active].min()) if active.any() else None,
                            max_positive_span=float(span[active].max()) if active.any() else None)
    return spans, dict(rows=len(a), fields=31, float32_bit_mismatched_cells=0,
                       legacy_values_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                       new_fields=reports)


def clock_prefix(path, stop):
    import pyarrow.parquet as pq
    with Path(path).open('rb') as stream:
        parquet = pq.ParquetFile(stream)
        groups, rows = [], 0
        for i in range(parquet.num_row_groups):
            if rows >= stop:
                break
            groups.append(i)
            rows += parquet.metadata.row_group(i).num_rows
        times = parquet.read_row_groups(groups, columns=['time'], use_threads=False).column('time')
    result = np.asarray(times.to_numpy()[:stop], dtype='datetime64[ns]').view(np.int64)
    need(len(result) == stop and np.all(np.diff(result) > 0), 'LOCAL_CLOCK')
    return result


def selected_positions(clock, times):
    pos = np.searchsorted(clock, times)
    need(np.all(pos < len(clock)) and np.array_equal(clock[pos], times), 'EXACT_SOURCE_CLOCK')
    return pos


def publish_json(name, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    need(len(raw) <= 2 * 1024**2, 'JSON_OUTPUT_BUDGET')
    _publish_bytes(ROOT / name, raw)
    return ref(ROOT / name)


def measure(plan, *, ranges=None, cutoff=None, output_root=None, lanes=None):
    import pandas as pd
    from gx1.scripts.audit_direction_d1_trendline_replay_v1 import load_m5_prefix, load_native_parts_prefix
    from gx1.features.htf_features import build_multi_tf_v4_closed_timestamp_indices, _resample_ohlcv, multi_tf_resample
    from gx1.features.technical_indicators_v1 import wilder_atr, ema50_200_spread_atr_block
    from gx1.features.trendline_registry_v1 import compute_trendline_registry_features_v1

    default_lanes = ('local', *TFS)
    explicit_lanes = lanes is not None
    lanes = default_lanes if lanes is None else lanes
    need(type(lanes) in (list, tuple) and 0 < len(lanes) <= len(default_lanes)
         and tuple(lanes) == tuple(lane for lane in default_lanes if lane in lanes)
         and (not explicit_lanes or plan['scope'].get('replay_lanes') == list(lanes)),
         'EXPLICIT_REPLAY_LANES')
    ranges = RANGES if ranges is None else ranges
    cutoff = CUTOFF if cutoff is None else cutoff
    output_root = ROOT if output_root is None else Path(output_root)
    need(type(ranges) is dict and set(ranges) == set(lanes)
         and all(type(v) in (list, tuple) and len(v) == 2
                 and all(type(i) is int for i in v) and 0 <= v[0] < v[1] for v in ranges.values())
         and ranges == plan['scope']['ranges'] and type(cutoff) is int
         and cutoff == plan['scope']['history_end_exclusive_ns']
         and str(output_root) == plan['output_root'], 'EXPLICIT_REPLAY_SCOPE')

    def publish_qualification(name, value):
        if output_root == ROOT:
            return publish_json(name, value)
        raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
        need(len(raw) <= 2 * 1024**2, 'JSON_OUTPUT_BUDGET')
        _publish_bytes(output_root / name, raw)
        return ref(output_root / name)

    payloads = [plan['inputs']['enriched'], *plan['inputs']['native_parts'],
                *(r for lane in plan['caches'].values() for r in lane.values())]
    stats = {}
    for item in payloads:
        before = stat(item['path'])
        need(sha(item['path']) == item['sha256'] and stat(item['path']) == before, 'PAYLOAD_SHA')
        stats[item['path']] = before
    need(stat(plan['local_signal']['path']) == plan['local_signal']['stat']
         and stat(plan['local_clock_path']) == plan['local_clock_stat'], 'LOCAL_SIGNAL_AND_CLOCK_STAT')
    stats[plan['local_clock_path']] = plan['local_clock_stat']
    native = None
    reports = {}
    for lane in lanes:
        left, right = ranges[lane]
        started = time.monotonic()
        rows = np.arange(left, right, dtype=np.int64)
        if lane == 'local':
            clock = clock_prefix(plan['local_clock_path'], right)
            times = clock[rows].copy()
            base = np.memmap(plan['local_signal']['path'], mode='r', dtype=np.float32,
                             shape=tuple(plan['local_signal']['shape']))
            observed = np.ascontiguousarray(base[np.ix_(rows, plan['local_signal']['columns'])])
            del base, clock
            bars, _ = load_m5_prefix(Path(plan['inputs']['enriched']['path']), cutoff)
            atr = wilder_atr(bars['high'], bars['low'], bars['close'], 14)
        else:
            cached_clock = np.load(plan['caches'][lane]['ts']['path'], mmap_mode='r', allow_pickle=False)
            times = np.asarray(cached_clock[rows], dtype=np.int64).copy()
            base = np.load(plan['caches'][lane]['feats']['path'], mmap_mode='r', allow_pickle=False)
            observed = np.ascontiguousarray(base[np.ix_(rows, plan['mtf_legacy_columns'])], dtype=np.float32)
            del base, cached_clock
            if native is None:
                native = load_native_parts_prefix(plan['inputs']['native_parts'], cutoff)
                native_indices = build_multi_tf_v4_closed_timestamp_indices(native.index)
            index = native_indices[lane]
            if lane == 'M5':
                bars = native.loc[index, ['open', 'high', 'low', 'close', 'volume']]
            elif lane in ('M15', 'H1'):
                agg = dict(open='first', high='max', low='min', close='last', volume='sum')
                witness = native.iloc[:4096]
                slow = _resample_ohlcv(witness, lane).dropna(subset=['open', 'high', 'low', 'close'])
                fast = multi_tf_resample(witness, lane).agg(agg).dropna(subset=['open', 'high', 'low', 'close'])
                need(slow.index.equals(fast.index) and np.array_equal(slow.to_numpy(), fast.to_numpy()),
                     'FAST_RESAMPLE_PARITY:' + lane)
                bars = multi_tf_resample(native, lane).agg(agg).dropna(subset=['open', 'high', 'low', 'close']).loc[index]
            else:
                bars = _resample_ohlcv(native, lane).dropna(subset=['open', 'high', 'low', 'close']).loc[index]
            atr = ema50_200_spread_atr_block(bars['high'], bars['low'], bars['close'])['atr14']
        pop = plan['populations'][lane]
        need(row_sha(rows, lane) == pop['selected_row_indices_sha256']
             and axis_sha(times + BAR_NS['M5' if lane == 'local' else lane] * 10**9) == pop['available_ns_sha256']
             and pd.Timestamp(int(times[0]), tz='UTC').isoformat() == pop['time_min_utc']
             and pd.Timestamp(int(times[-1]), tz='UTC').isoformat() == pop['time_max_utc']
             and np.all(times + BAR_NS['M5' if lane == 'local' else lane] * 10**9 < cutoff),
             'PHYSICAL_TRAIN_POPULATION:' + lane)
        positions = selected_positions(bars.index.asi8, times)
        bars = bars.iloc[:int(positions[-1]) + 1]
        data = bars[['high', 'low', 'close']].copy()
        data['atr'] = atr.loc[bars.index]
        produced, state = compute_trendline_registry_features_v1(data, **plan['parameters'][lane])
        need(list(produced.columns) == plan['registry_names'] and produced.index.equals(bars.index)
             and state.bar_count == len(bars), 'PRODUCER_INTERFACE')
        values = np.ascontiguousarray(produced.iloc[positions].to_numpy(), dtype=np.float32)
        spans, parity = compare_and_extract(values, observed, produced.columns, positions,
                                           lookback=plan['parameters'][lane]['swing_lookback'])
        buffer = io.BytesIO()
        np.savez(buffer, row_indices=rows, times_ns=times, native_bar_positions=positions.astype(np.int64),
                 anchor_span_bars=spans)
        need(buffer.tell() <= 2 * 1024**2, 'ARRAY_OUTPUT_BUDGET')
        _publish_bytes(output_root / (lane + '_ANCHOR_SPANS.npz'), buffer.getvalue())
        reports[lane] = dict(population=pop, parity=parity, parameters=plan['parameters'][lane],
                             replay_rows=len(bars), first_replay_time_ns=int(bars.index.asi8[0]),
                             fresh_state_at_source_row_zero=True,
                             times_ns_sha256=axis_sha(times),
                             artifact=ref(output_root / (lane + '_ANCHOR_SPANS.npz')),
                             elapsed_seconds=time.monotonic() - started)
        publish_qualification(lane + '_QUALIFICATION.json', reports[lane])
        del data, bars, atr, produced, state, values, observed, spans, buffer
        gc.collect()
    need(all(stat(p) == s for p, s in stats.items())
         and stat(plan['local_signal']['path']) == plan['local_signal']['stat'], 'PAYLOAD_BOOKENDS')
    return reports


def terminal_record(*, status, plan_ref, result, failure, head, elapsed):
    need(status in ('COMPLETE', 'FAILED_CONSUMED'), 'TERMINAL_STATUS')
    need((status == 'COMPLETE') == (result is not None and failure is None), 'TERMINAL_OUTCOME')
    return dict(status=status, exit_code=0 if status == 'COMPLETE' else 1,
                plan=plan_ref, result=result, failure=failure, source_head=head,
                source_unchanged=True, elapsed_seconds=elapsed, test_data_used=False,
                remaining_invocations=0, relaunch_allowed=False)


def run(plan_sha):
    from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution
    began = time.monotonic()
    cap = require_capped_cpu_audit_execution()
    plan, plan_ref, policy_ref, head = inspect(plan_sha, execute=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    need(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'AMBIENT_TIMER')
    old_alarm = signal.getsignal(signal.SIGALRM)
    ROOT.mkdir(exist_ok=True)
    claim = publish_json('CLAIM.json', dict(plan=plan_ref, policy=policy_ref, source_head=head,
                                         cap=cap, scope=SCOPE, test_data_used=False))
    def bookend():
        need(subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == head
             and subprocess.check_output(['git', 'status', '--porcelain'], text=True) == ''
             and sha(PLAN) == plan_sha and sha(POLICY) == policy_ref['sha256']
             and all(sha(REPO / p) == h for p, h in plan['source_owners'].items())
             and all(sha(x['path']) == x['sha256'] for k, x in plan['inputs'].items()
                     if k not in ('native_parts', 'enriched')), 'SOURCE_BOOKEND')
    def expired(*_args):
        raise TimeoutError('ANCHOR_SPAN_INPUT_1800_SECONDS')
    result = failure = None
    status = 'FAILED_CONSUMED'
    try:
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, max(1, WALL_SECONDS - 30 - (time.monotonic() - began)))
        reports = measure(plan)
        bookend()
        result = publish_json('RESULT.json', dict(status='COMPLETE_INPUT_EXTENSION_ONLY', plan=plan_ref,
            claim=claim, policy=policy_ref, source_head=head, source_unchanged=True, scope=SCOPE,
            surfaces=reports, normalization_fitted=False, fresh_model_baseline_complete=False,
            old_cache_modified=False, predictive_value_claimed=False, test_data_used=False,
            elapsed_seconds=time.monotonic() - began,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024))
        status = 'COMPLETE'
    except BaseException as exc:
        failure = publish_json('FAILED.json', dict(type=type(exc).__name__, error=str(exc)[:4096],
                                                   plan=plan_ref, test_data_used=False))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        bookend_error = None
        try:
            bookend()
        except BaseException as exc:
            bookend_error = exc
            status = 'FAILED_CONSUMED'
            if failure is None:
                failure = publish_json('FAILED.json', dict(type=type(exc).__name__, error=str(exc)[:4096],
                                                           plan=plan_ref, test_data_used=False))
        t = terminal_record(status=status, plan_ref=plan_ref, result=result, failure=failure,
                            head=head, elapsed=time.monotonic() - began)
        t['source_unchanged'] = bookend_error is None
        publish_json('TERMINAL.json', t)
        if bookend_error is not None:
            raise RuntimeError('ANCHOR_SPAN_INPUT_SOURCE_BOOKEND') from bookend_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-plan', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--plan-sha256')
    args = parser.parse_args()
    if args.build_plan:
        need(not args.preflight and args.plan_sha256 is None and not PLAN.exists(), 'PLAN_BUILD_CLI')
        _publish_bytes(PLAN, (json.dumps(build_plan(), sort_keys=True, indent=2) + '\n').encode())
        print(json.dumps(ref(PLAN)))
    else:
        need(args.plan_sha256 is not None, 'PLAN_SHA_REQUIRED')
        if args.preflight:
            plan, plan_ref, _policy, head = inspect(args.plan_sha256, execute=False)
            print(json.dumps(dict(status='READY_NO_CLAIM_NO_ARRAY_DECODE', plan=plan_ref,
                                  source_head=head, scope=plan['scope'])))
        else:
            run(args.plan_sha256)


if __name__ == '__main__':
    main()
