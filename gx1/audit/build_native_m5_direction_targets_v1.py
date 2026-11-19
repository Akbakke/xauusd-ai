"""One capped native-M5 TRAIN target audit with immutable one-use receipts."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import signal
import stat
import subprocess
import time

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

from gx1.contracts.gx1_capped_execution_v1 import (
    require_capped_cpu_audit_execution, require_capped_lock_ancestry,
)
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO = Path('/home/andre2/src/GX1_CURRENT')
STAGE = 'native_m5_direction_targets_20261119_001'
PLAN = REPO / 'configs/research/NATIVE_M5_DIRECTION_TARGETS_20261119_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_TARGETS_20261119_001')
POLICY = REPO / 'NEXT_RUN_POLICY.json'
NATIVE = Path('/home/andre2/GX1_DATA/data/native_xau/XAU_M5_NATIVE_2009_20260701_PAIR_WEEKCLOSED_20260927')
MANIFEST_SHA = 'ca0012e5e30f7808876d666fe4d32e3fb873fddb6fc6178bd47b97eb89194cca'
M5_NS = 300_000_000_000
ISSUE_START_NS = 1714521600000000000
CUTOFF_NS = 1727740800000000000
COLUMNS = ['time', 'open', 'high', 'low', 'close']
SOURCE_PATHS = (
    'gx1/audit/build_native_m5_direction_targets_v1.py',
    'gx1/contracts/market_state_native_m5_targets_v1.py',
    'gx1/features/htf_features.py', 'gx1/features/technical_indicators_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py', 'scripts/gx1_capped_run.sh',
    'tests/test_build_native_m5_direction_targets_v1.py',
    'tests/test_market_state_native_m5_targets_v1.py',
)
EXECUTION = {'class': 'audit', 'memory': '4G', 'swap': '512M',
             'outer_seconds': 900, 'internal_seconds': 870, 'kill_after_seconds': 10}
CLOSED_FLAGS = ('training_enabled', 'native_training_enabled',
                'native_launch_authorized', 'full_epoch_training_allowed', 'full_val_allowed')


def need(ok, reason):
    if not bool(ok):
        raise ValueError('NATIVE_M5_TARGET_AUDIT_' + reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def stamp(path):
    path = Path(path)
    need(path.is_absolute() and not path.is_symlink(), 'REGULAR_PATH')
    value = path.stat()
    need(stat.S_ISREG(value.st_mode) and value.st_size > 0, 'REGULAR_FILE')
    return {'size': value.st_size, 'device': value.st_dev, 'inode': value.st_ino,
            'mode': value.st_mode, 'mtime_ns': value.st_mtime_ns, 'ctime_ns': value.st_ctime_ns}


def ref(path):
    stamp(path)
    return {'path': str(path), 'sha256': sha(path)}


def read(binding):
    need(type(binding) is dict and set(binding) == {'path', 'sha256'}
         and ref(binding['path']) == binding, 'JSON_BINDING')
    return json.loads(Path(binding['path']).read_text())


def ns(value):
    return int(datetime.fromisoformat(str(value).replace('Z', '+00:00')).timestamp()) * 10**9


def source_manifest(binding):
    need(binding == {'path': str(NATIVE / 'MANIFEST.json'), 'sha256': MANIFEST_SHA},
         'NATIVE_MANIFEST_IDENTITY')
    manifest = read(binding)
    expected = {'schema_version': 'xau_canonical_native_source_v3',
        'instrument': 'XAU_USD', 'timeframe': 'M5', 'source_granularity': 'M5',
        'source_kind': 'oanda_native_mba_candles', 'prices': 'MBA',
        'bar_duration_seconds': 300, 'decision_available_offset_seconds': 300,
        'timestamp_semantics': 'bar_start_utc', 'completion_field': 'complete',
        'completion_value': True}
    need(all(manifest.get(key) == value for key, value in expected.items()),
         'NATIVE_M5_SOURCE_GRAIN')
    return manifest


def part_metadata(part):
    before = stamp(part['path'])
    parquet = pq.ParquetFile(part['path'])
    names = parquet.schema_arrow.names
    need(all(name in names for name in COLUMNS), 'SOURCE_COLUMNS')
    groups = []
    for index in part['row_groups']:
        need(type(index) is int and 0 <= index < parquet.num_row_groups, 'ROW_GROUP')
        row = parquet.metadata.row_group(index)
        clock = row.column(names.index('time')).statistics
        need(clock is not None and clock.has_min_max, 'CLOCK_STATISTICS')
        groups.append({'index': index, 'rows': row.num_rows,
                       'first_bar_ns': ns(clock.min), 'last_bar_ns': ns(clock.max)})
    need(stamp(part['path']) == before, 'METADATA_STABLE')
    return {'stat': before, 'total_rows': parquet.metadata.num_rows,
            'total_row_groups': parquet.num_row_groups, 'selected_groups': groups}


def native_source_spec():
    binding = {'path': str(NATIVE / 'MANIFEST.json'), 'sha256': MANIFEST_SHA}
    manifest = source_manifest(binding)
    parts = []
    for year in range(2009, 2025):
        path = NATIVE / f'year={year}/part-000.parquet'
        parquet = pq.ParquetFile(path)
        column = parquet.schema_arrow.names.index('time')
        groups = [i for i in range(parquet.num_row_groups)
                  if ns(parquet.metadata.row_group(i).column(column).statistics.min) + M5_NS < CUTOFF_NS]
        parts.append({'path': str(path), 'sha256': manifest['year_sha256'][f'year={year}'],
                      'row_groups': groups})
    return {'manifest': binding, 'parts': parts, 'closure_authority': None,
            'issue_start_ns': ISSUE_START_NS, 'cutoff_ns': CUTOFF_NS}


def build_plan(*, source_spec, source_base_commit=None):
    from gx1.contracts.market_state_native_m5_targets_v1 import contract
    need(source_spec == native_source_spec(), 'EXACT_SOURCE_SPEC')
    head = source_base_commit or subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    return {'schema_version': 'native_m5_direction_target_audit_plan_v1',
        'stage': STAGE, 'status': 'PREPARED_NOT_ADMITTED', 'output_root': str(ROOT),
        'source_base_commit': head, 'source_spec': source_spec,
        'source_metadata': {p['path']: part_metadata(p) for p in source_spec['parts']},
        'source_owners': {name: sha(REPO / name) for name in SOURCE_PATHS},
        'target_contract': contract(), 'columns': COLUMNS, 'execution': EXECUTION,
        'runtime_versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'pyarrow': pyarrow.__version__},
        'max_invocations': 1, 'source_resolution_seconds': 300,
        'native_prehistory': 'all_declared_native_M5_from_2009_before_cutoff',
        'decoded_crossing_group': 'filter_bar_start_plus_300s_strictly_before_cutoff',
        'unknown_gaps': 'conservative_unknown_without_closure_authority',
        'target_tapes': 1, 'model_forwards': 0, 'optimizer_steps': 0,
        'normalization_fits': 0, 'test_data_used': False, 'economic_edge_proven': False}


def source_clean(plan):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    return (subprocess.check_output(['git', 'branch', '--show-current'],
            cwd=REPO, text=True).strip() == 'work/gx1-current'
        and subprocess.run(['git', 'merge-base', '--is-ancestor',
            plan['source_base_commit'], head], cwd=REPO, capture_output=True).returncode == 0
        and not subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)
        and subprocess.run(['git', 'diff', '--quiet', plan['source_base_commit'],
            head, '--', 'gx1', 'scripts'], cwd=REPO).returncode == 0
        and set(plan['source_owners']) == set(SOURCE_PATHS)
        and all(sha(REPO / name) == digest for name, digest in plan['source_owners'].items()))


def inspect(plan_sha, *, admitted):
    need(Path.cwd().resolve() == REPO and not ROOT.exists(), 'CWD_OUTPUT')
    plan_ref = {'path': str(PLAN), 'sha256': plan_sha}
    plan = read(plan_ref)
    need(plan == build_plan(source_spec=plan['source_spec'],
                           source_base_commit=plan['source_base_commit'])
         and source_clean(plan), 'PLAN_SOURCE_IDENTITY')
    policy_ref = ref(POLICY)
    policy = read(policy_ref)
    gate = policy.get(STAGE, {})
    need(all(policy.get(key) is False for key in CLOSED_FLAGS)
         and gate.get('plan') == plan_ref and gate.get('output_root') == str(ROOT)
         and gate.get('relaunch_allowed') is False and gate.get('max_invocations') == 1
         and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
         and gate.get('training_authorized') is False and gate.get('test_authorized') is False
         and gate.get('broker_access_allowed') is False
         and gate.get('model_forwards') == gate.get('optimizer_steps') == 0, 'CLOSED_POLICY')
    expected = ('ADMITTED_ONE_USE', True, 1) if admitted else ('PREPARED_NOT_ADMITTED', False, 0)
    need((gate.get('status'), gate.get('execution_authorized'),
          gate.get('remaining_invocations')) == expected, 'ONE_USE_GATE')
    return plan, plan_ref, policy_ref


def checked_inputs(plan):
    for part in plan['source_spec']['parts']:
        need(stamp(part['path']) == plan['source_metadata'][part['path']]['stat'], 'PART_STAT')
        need(sha(part['path']) == part['sha256'], 'PART_SHA')
    source_manifest(plan['source_spec']['manifest'])


def read_source(plan):
    times, prices = [], []
    decoded_rows = 0
    for part in plan['source_spec']['parts']:
        parquet = pq.ParquetFile(part['path'])
        for group in part['row_groups']:
            table = parquet.read_row_group(group, columns=COLUMNS, use_threads=False)
            need(all(table[name].null_count == 0 for name in COLUMNS), 'NULL_SOURCE_VALUE')
            starts = table['time'].to_numpy(zero_copy_only=False).astype('datetime64[ns]').astype(np.int64)
            ohlc = np.column_stack([table[name].to_numpy(zero_copy_only=False)
                                    for name in COLUMNS[1:]]).astype(np.float64)
            selected = starts + M5_NS < plan['source_spec']['cutoff_ns']
            times.append(starts[selected])
            prices.append(ohlc[selected])
            decoded_rows += len(starts)
    starts, ohlc = np.concatenate(times), np.concatenate(prices)
    need(decoded_rows == sum(group['rows'] for meta in plan['source_metadata'].values()
                             for group in meta['selected_groups']), 'DECODED_ROWS')
    need(len(starts) > 0 and np.all(np.diff(starts) > 0)
         and np.all(starts % M5_NS == 0)
         and np.all(starts + M5_NS < plan['source_spec']['cutoff_ns'])
         and len(starts) <= sum(meta['total_rows'] for meta in plan['source_metadata'].values()),
         'SOURCE_CLOCK')
    return starts, ohlc, decoded_rows


def calculate(plan):
    from gx1.contracts.market_state_native_m5_targets_v1 import build_native_m5_targets, NATIVE_K, TIMEFRAMES
    starts, ohlc, decoded = read_source(plan)
    spec = plan['source_spec']
    targets = build_native_m5_targets(starts, ohlc, source_ref=spec['manifest'], cutoff_ns=spec['cutoff_ns'])
    selected = np.flatnonzero((targets.issue_ns >= spec['issue_start_ns'])
                              & (targets.issue_ns < spec['cutoff_ns']))
    need(len(selected) > 0, 'EMPTY_ISSUES')
    issues = targets.issue_ns[selected]
    query = np.broadcast_to(np.asarray(NATIVE_K, dtype=np.int64), (len(issues), 5)).copy()
    state = targets.gather_state(issues, query, fold_upper_ns=spec['cutoff_ns'])
    future = targets.gather_displacement(issues, fold_upper_ns=spec['cutoff_ns'])
    arrays = {'issue_ns': issues, 'source_row': targets.source_row[selected],
              'close': targets.close[selected], 'atr': targets.atr[selected],
              'atr_available_ns': targets.atr_available_ns[selected],
              'tf_bar_m5_counts': targets.tf_bar_m5_counts[selected],
              'source_gap': targets.source_gap[selected],
              'current_s_allowed': targets.s_allowed[selected],
              'current_s_span_complete': targets.s_span_complete[selected],
              'current_s_local_range': targets.s_local_range[selected]}
    for prefix, result in (('state', state), ('future', future)):
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                arrays[prefix + '_' + key] = value
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue(), {'source_rows': len(starts), 'decoded_rows': decoded,
        'selected_issue_rows': len(issues), 'first_issue_ns': int(issues[0]),
        'last_issue_ns': int(issues[-1]), 'array_shapes': {key: list(value.shape) for key, value in arrays.items()},
        'state_allowed_counts': {str(bit): (state['allowed'] == bit).sum(axis=0).tolist()
                                  for bit in (1, 2, 4, 5, 7)},
        'future_allowed_counts': {str(bit): (future['allowed'] == bit).sum(axis=0).tolist()
                                   for bit in (1, 2, 4, 5, 7)},
        'source_row_axis': 'zero_based_ordinal_in_ordered_native_2009_prefix_after_cutoff_filter',
        'source_first_bar_ns': int(starts[0]), 'source_last_bar_ns': int(starts[-1]),
        'source_gap_count': int(targets.source_gap.sum()),
        'selected_source_gap_count': int(targets.source_gap[selected].sum()),
        'current_s_allowed_counts': {str(bit): (targets.s_allowed[selected] == bit).sum(axis=0).tolist()
                                    for bit in (1, 2, 4, 5, 7)},
        'current_s_span_complete_counts': targets.s_span_complete[selected].sum(axis=0).tolist(),
        'coverage_by_timeframe': {
            tf: {
                'state_mature': int(state['mature'][:, col].sum()),
                'state_eligible': int(state['eligible'][:, col].sum()),
                'state_known_side': int(np.isin(state['allowed'][:, col], (1, 4)).sum()),
                'future_mature': int(future['mature'][:, col].sum()),
                'future_eligible': int(future['eligible'][:, col].sum()),
                'future_endpoint_present': int(future['endpoint_present'][:, col].sum()),
                'future_gap_between': int(future['gap_between'][:, col].sum()),
                'causal_atr_known': int(np.isfinite(targets.atr[selected, col]).sum()),
                'last_closed_tf_m5_count_distribution': {
                    str(int(count)): int(n) for count, n in zip(
                        *np.unique(targets.tf_bar_m5_counts[selected, col], return_counts=True))
                },
            } for col, tf in enumerate(TIMEFRAMES)},
        'timeframe_bar_m5_count_min': targets.tf_bar_m5_counts[selected].min(axis=0).tolist(),
        'timeframe_bar_m5_count_max': targets.tf_bar_m5_counts[selected].max(axis=0).tolist()}


def publish(name, value):
    raw = value if isinstance(value, bytes) else (
        json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    _publish_bytes(ROOT / name, raw)
    return {'path': str(ROOT / name), 'sha256': hashlib.sha256(raw).hexdigest()}


def timeout_handler(_signum, _frame):
    raise TimeoutError('NATIVE_M5_TARGET_AUDIT_INTERNAL_LIMIT')


def run(plan_sha):
    started = time.monotonic()
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 4 * 1024**3
         and cap['swap'] == 512 * 1024**2 and cap['numerical_threads'] == 1
         and lock['scope'] == cap['cgroup_path'][len('/sys/fs/cgroup'):], 'CAP_LIMITS')
    plan, plan_ref, policy_ref = inspect(plan_sha, admitted=True)
    checked_inputs(plan)
    ROOT.mkdir(parents=True, exist_ok=False)
    claim = publish('CLAIM.json', {'status': 'CLAIMED_ONE_USE', 'stage': STAGE,
        'plan': plan_ref, 'policy_at_claim': policy_ref, 'remaining_invocations': 0,
        'max_invocations': 1, 'test_data_used': False})
    old_handler = signal.getsignal(signal.SIGALRM)
    result_ref = None
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        remaining = EXECUTION['internal_seconds'] - (time.monotonic() - started)
        need(remaining > 0, 'TIME_BEFORE_CALCULATE')
        signal.setitimer(signal.ITIMER_REAL, remaining)
        payload, report = calculate(plan)
        checked_inputs(plan)
        need(source_clean(plan) and ref(POLICY) == policy_ref and ref(PLAN) == plan_ref,
             'SOURCE_INPUT_POLICY_BOOKEND')
        tape_ref = publish('TARGETS.npz', payload)
        result_ref = publish('RESULT.json', {'status': 'COMPLETE_NATIVE_M5_TARGET_AUDIT',
            'stage': STAGE, 'plan': plan_ref, 'claim': claim, 'policy_at_claim': policy_ref,
            'source_spec': plan['source_spec'], 'target_contract': plan['target_contract'],
            'source_unchanged': True, 'target_tape': tape_ref, 'report': report,
            'model_forwards': 0, 'optimizer_steps': 0, 'normalization_fits': 0,
            'test_data_used': False, 'economic_edge_proven': False,
            'elapsed_seconds': time.monotonic() - started})
        terminal = publish('TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0,
            'stage': STAGE, 'plan': plan_ref, 'claim': claim, 'result': result_ref,
            'target_tape': tape_ref, 'remaining_invocations': 0,
            'relaunch_allowed': False, 'test_data_used': False})
        return {'result': result_ref, 'terminal': terminal, 'target_tape': tape_ref}
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        publish('TERMINAL.json', {'status': 'FAILED_CONSUMED', 'exit_code': 1,
            'stage': STAGE, 'plan': plan_ref, 'claim': claim, 'result': result_ref,
            'error': type(error).__name__ + ':' + str(error)[:2000],
            'remaining_invocations': 0, 'relaunch_allowed': False, 'test_data_used': False})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    need(args.plan == str(PLAN) and args.out_dir == str(ROOT), 'CLI_SCOPE')
    if args.preflight:
        plan, plan_ref, policy_ref = inspect(args.plan_sha256, admitted=False)
        checked_inputs(plan)
        print(json.dumps({'status': 'PREFLIGHT_PASS_NO_TARGET_READ', 'plan': plan_ref,
            'policy': policy_ref, 'claim_created': False, 'output_root_absent': True,
            'parquet_values_decoded': 0, 'model_forwards': 0, 'test_data_used': False}))
    else:
        print(json.dumps(run(args.plan_sha256), sort_keys=True))


if __name__ == '__main__':
    main()
