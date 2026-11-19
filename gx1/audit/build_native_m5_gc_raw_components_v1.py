"""One capped direct-native GC raw-component pass with durable file checkpoints."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import json
import multiprocessing as mp
from pathlib import Path
import signal
import subprocess
import time

_FIREWALL = None
if __name__ in ('__main__', '__mp_main__'):
    from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
    _FIREWALL = install_native_m5_direction_io_firewall()

import numpy as np
import pyarrow.parquet as pq
from gx1.audit import build_native_m5_direction_targets_v1 as common
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
from gx1.contracts.gx1_capped_execution_v1 import (
    require_capped_cpu_research_producer_execution, require_capped_lock_ancestry,
)

REPO = common.REPO
STAGE = 'gc_native_m5_raw_components_20261119_001'
PLAN = REPO / 'configs/research/GC_NATIVE_M5_RAW_COMPONENTS_20261119_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001/GC_NATIVE_M5_RAW_COMPONENTS_20261119_001')
POLICY = REPO / 'NEXT_RUN_POLICY.json'
INPUT_KEYS = {'source_audit_plan', 'source_audit_result', 'source_incidents'}
EXECUTION = {'class': 'producer', 'memory': '12G', 'swap': '512M', 'workers': 4,
    'outer_seconds': 7200, 'internal_seconds': 7170, 'kill_after_seconds': 10,
    'numerical_threads': 1, 'models': 0, 'model_forwards': 0, 'optimizer_steps': 0}
SOURCE_PATHS = (
    'gx1/audit/build_native_m5_gc_raw_components_v1.py',
    'gx1/audit/build_native_m5_direction_targets_v1.py',
    'gx1/contracts/gc_native_m5_source_inputs_v1.py',
    'gx1/contracts/gc_continuous_source_v1.py',
    'gx1/contracts/gc_source_disposition_v1.py',
    'gx1/contracts/gc_order_flow_source_v1.py',
    'gx1/features/gc_native_m5_features_v1.py',
    'gx1/features/gc_order_flow_v1.py',
    'gx1/contracts/native_m5_direction_io_firewall_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/entry_model_native_bundle_commit_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_build_native_m5_gc_raw_components_v1.py',
)
sha, ref, read = common.sha, common.ref, common.read


def need(ok, reason):
    if not bool(ok):
        raise ValueError('GC_NATIVE_M5_RAW_' + reason)


def publish(path, value):
    raw = value if isinstance(value, bytes) else (
        json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    _publish_bytes(Path(path), raw)
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


def firewall_state():
    from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
    state = install_native_m5_direction_io_firewall()
    need(state['denied_attempts'] == 0, 'IO_FIREWALL')
    return json.loads(json.dumps(state))


def build_plan(*, inputs, source_base_commit=None):
    from gx1.contracts.gc_native_m5_source_inputs_v1 import bind_native_m5_raw_sources
    from gx1.contracts import gc_continuous_source_v1 as source
    from gx1.contracts.native_m5_direction_io_firewall_v1 import contract
    need(type(inputs) is dict and set(inputs) == INPUT_KEYS, 'INPUT_ROLES')
    need(inputs['source_audit_plan'] == {
        'path': str(REPO / 'configs/research/GC_SOURCE_AUDIT_20261016_002.json'),
        'sha256': '3cca1eef4bd965e0194a5f353f76645fbb6a030c509d0f3d24850745f4d0310d'},
         'ORIGINAL_INVENTORY')
    need(inputs['source_audit_result'] == {
        'path': '/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001/GC_SOURCE_QUALIFICATION_20261016_002/AUDIT_RESULTS/RESULT.json',
        'sha256': 'c869e3d1395c5892fab19d7c3056faaca1ca4aaac41f70d3f05943d4e63dcbab'},
        'ORIGINAL_RECORD_AUDIT')
    need(inputs['source_incidents'] == ref(REPO / 'configs/research/GC_VENDOR_SOURCE_INCIDENTS_20261018_001.json'),
         'INCIDENT_CONTRACT')
    binding = bind_native_m5_raw_sources(source_audit_plan_ref=inputs['source_audit_plan'])
    clock = binding['source_clock']
    need(clock['start_ns'] == 1714521600000000000 and clock['end_ns'] == 1782864000000000000,
         'EXACT_PRETEST_SCOPE')
    audit = read(inputs['source_audit_result'])
    original = {item['path']: item for item in audit['files']}
    need(audit['record_audit_performed'] is True and audit['test_accessed'] is False
         and set(original) == {item['path'] for item in binding['raw_files']}, 'AUDITED_FILE_INVENTORY')
    expected = {}
    for item in binding['raw_files']:
        prior = original[item['path']]
        need(prior['sha256'] == item['sha256'] and prior['receipt_sha256'] == item['receipt_sha256'],
             'AUDITED_FILE_IDENTITY')
        counters = prior['counters']
        expected[item['path']] = {'records': counters['records'], 'raw_trade_records': counters.get('trade_records', 0),
            'bad_receive_records': counters.get('bad_receive_timestamp_records', 0)}
    need(sum(x['records'] for x in expected.values()) == 892100712, 'EXACT_FULL_RECORD_COUNT')
    incidents = read(inputs['source_incidents'])
    admission = incidents['trusted_trade_admission']
    need(admission['deduplicate_T'] is False and admission['reject_duplicate_signature'] is False
         and incidents['nontrade_action_semantics_admitted'] is False, 'TRUSTED_T_EDITION')
    columns = {**source.COLUMNS, **source.L1_AUDIT_COLUMNS}
    for item in binding['raw_files']:
        receipt = read({'path': item['receipt_path'], 'sha256': item['receipt_sha256']})
        need(set(receipt['header_fields']) == set(columns) and len(receipt['header_fields']) == 20,
             'RAW_HEADER_CONTRACT')
    head = source_base_commit or subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    return {'schema_version': 'gx1_gc_native_m5_raw_components_plan_v1',
        'stage': STAGE, 'output_root': str(ROOT), 'status': 'PREPARED_NOT_ADMITTED',
        'source_base_commit': head, 'source_owners': {p: sha(REPO / p) for p in SOURCE_PATHS},
        'inputs': inputs, 'source_binding': binding, 'expected_per_file': expected,
        'parser_columns': {key: str(value) for key, value in columns.items()},
        'execution': EXECUTION, 'io_firewall': contract(),
        'checkpoint_unit': 'one_original_raw_file_native_300_second_components',
        'selection': 'all_contracts_unselected_raw_components_only_no_observation_admission',
        'future_assembly': 'source_incidents_and_previous_complete_UTC_day_roll_288_native_bars_per_day',
        'models': 0, 'model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False,
        'native_training_admitted': False, 'economic_edge_proven': False}


def source_clean(plan):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    return (subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip()
        == 'work/gx1-current'
        and not subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)
        and subprocess.run(['git', 'merge-base', '--is-ancestor', plan['source_base_commit'], head],
                           cwd=REPO, capture_output=True).returncode == 0
        and subprocess.run(['git', 'diff', '--quiet', plan['source_base_commit'], head,
                           '--', 'gx1', 'scripts'], cwd=REPO).returncode == 0
        and set(plan['source_owners']) == set(SOURCE_PATHS)
        and all(sha(REPO / path) == digest for path, digest in plan['source_owners'].items()))


def checked_metadata(plan):
    bindings = list(plan['inputs'].values())
    source = plan['source_binding']
    bindings += [source['source_inventory'], source['original_mapping_audit']]
    for item in source['raw_files']:
        bindings.append({'path': item['receipt_path'], 'sha256': item['receipt_sha256']})
    for item in source['definition_sources']:
        bindings += [item['source'], item['receipt']]
    for item in bindings:
        need(ref(item['path']) == item, 'METADATA_BOOKEND')


def inspect(plan_sha, *, admitted):
    need(Path.cwd().resolve() == REPO and not ROOT.exists(), 'CWD_OUTPUT')
    plan_ref = {'path': str(PLAN), 'sha256': plan_sha}
    plan = read(plan_ref)
    need(plan == build_plan(inputs=plan['inputs'], source_base_commit=plan['source_base_commit'])
         and source_clean(plan), 'PLAN_SOURCE')
    policy_ref = ref(POLICY)
    policy = read(policy_ref)
    gate = policy.get(STAGE, {})
    need(all(policy.get(key) is False for key in common.CLOSED_FLAGS)
         and gate.get('plan') == plan_ref and gate.get('output_root') == str(ROOT)
         and gate.get('max_invocations') == 1 and gate.get('relaunch_allowed') is False
         and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
         and gate.get('training_authorized') is False and gate.get('test_authorized') is False
         and gate.get('models') == gate.get('optimizer_steps') == 0, 'CLOSED_POLICY')
    expected = ('ADMITTED_ONE_USE', True, 1) if admitted else ('PREPARED_NOT_ADMITTED', False, 0)
    need((gate.get('status'), gate.get('execution_authorized'), gate.get('remaining_invocations')) == expected,
         'ONE_USE_GATE')
    checked_metadata(plan)
    return plan, plan_ref, policy_ref


def ladder_conservation(path, metrics):
    width = len(metrics['volume'])
    totals = np.zeros((width, 3), np.uint64)
    rows = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65536):
        data = {name: batch.column(batch.schema.get_field_index(name)).to_numpy()
                for name in batch.schema.names}
        idx = data['bucket_index']
        need(idx.dtype == np.int32 and np.all((idx >= 0) & (idx < width))
             and np.all(data['price_tick'] > 0), 'LADDER_DOMAIN')
        for j, name in enumerate(('buy', 'sell', 'unknown')):
            need(data[name].dtype == np.uint64, 'LADDER_INTEGER')
            np.add.at(totals[:, j], idx, data[name])
        rows += len(idx)
    need(all(np.array_equal(totals[:, j], metrics[name])
             for j, name in enumerate(('buy', 'sell', 'unknown'))), 'LADDER_SIDE_CONSERVATION')
    return rows


def reduce_file(plan, ordinal, producer_plan, producer_claim):
    firewall_state()
    from gx1.contracts import gc_continuous_source_v1 as source
    from gx1.contracts.gc_native_m5_source_inputs_v1 import verify_native_raw_payload
    from gx1.features.gc_native_m5_features_v1 import native_m5_collectors
    from gx1.features import gc_order_flow_v1 as formulas
    binding = plan['source_binding']
    item = binding['raw_files'][ordinal]
    expected = plan['expected_per_file'][item['path']]
    need(expected['records'] <= np.iinfo(np.uint64).max // np.iinfo(np.uint32).max,
         'EXACT_VOLUME_INTEGER_BOUND')
    before = verify_native_raw_payload(item)
    scope = {**binding['source_clock'], 'start_ns': item['start_ns'], 'end_ns': item['end_ns']}
    origin, end, interval = source.reduction_clock_v1(scope)
    width = (end - origin) // interval
    symbols = list(binding['contract_metadata'])
    ci = symbols.index(item['raw_symbol'])
    tick = binding['contract_metadata'][item['raw_symbol']][2]
    destination = ROOT / 'FILES' / ('FILE_' + str(ordinal + 1).zfill(3))
    staging = destination.with_name('.' + destination.name + '.staging')
    staging.mkdir(parents=True, exist_ok=False)
    ladder = staging / 'LADDER.parquet'
    selected = np.full(width, ci, np.int16)
    collectors = native_m5_collectors(scope=scope, ladder_path=ladder, selected=selected,
        masks={'volume': np.ones(width, bool)}, symbol_index=ci, tick_size=tick)
    quality = source.L1QualityCollector(np.arange(width, dtype=np.int64), tick, native_m5_scope=scope)
    def start(raw):
        quality.file_start(raw)
        collectors['on_file_start'](raw)
    def consume(data, idx, trusted):
        quality(data, idx, trusted)
        collectors['on_batch'](data, idx, trusted)
    try:
        result = source.scan_symbol([item], tick, on_batch=consume,
            extra_columns=source.L1_AUDIT_COLUMNS, on_file_start=start, native_m5_scope=scope)
        collectors['book'].finish()
    finally:
        collectors['trades'].close()
    book = collectors['book']
    stats = book.trusted_trade_statistics()
    report = result['files'][0]
    need(report['records'] == expected['records']
         and report['raw_trade_records'] == expected['raw_trade_records']
         and int(quality.values['records'].sum()) == expected['records']
         and int(quality.values['bad_recv'].sum()) == expected['bad_receive_records'],
         'ORIGINAL_RECORD_COUNTS')
    metrics = {name: result[name] for name in source.METRICS}
    need(np.array_equal(metrics['volume'], metrics['buy'] + metrics['sell'] + metrics['unknown'])
         and np.array_equal(formulas.trusted_trade_record_count_v1(stats), metrics['trades'])
         and np.array_equal(stats['size_sum_contracts'], metrics['volume']), 'TRADE_STAT_CONSERVATION')
    ladder_rows = ladder_conservation(ladder, metrics)
    arrays = {'bar_start_ns': origin + np.arange(width, dtype=np.int64) * interval,
        'ohlc_tick': collectors['trades'].ohlc,
        'first_trade_recv_ns': collectors['trades'].first_trade_recv_ns,
        'last_trade_recv_ns': collectors['trades'].last_trade_recv_ns,
        'book_sums': book.sums, 'book_covered_ns': book.covered, 'book_first_mid_tick': book.first,
        'book_last_mid_tick': book.last, 'book_first_recv_ns': book.first_time,
        'book_last_recv_ns': book.last_time, 'order_sums': book.order_sums,
        'order_covered_ns': book.order_covered}
    for prefix, group in (('metric', metrics), ('quality', quality.values),
                          ('event', book.event_stats), ('trade_size', stats)):
        arrays.update({prefix + '_' + name: value for name, value in group.items()})
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    publish(staging / 'COMPONENTS.npz', stream.getvalue())
    stream.close()
    with np.load(staging / 'COMPONENTS.npz', allow_pickle=False) as saved:
        need(set(saved.files) == set(arrays), 'COMPONENT_INVENTORY')
        for key, original in arrays.items():
            value = saved[key]
            need(value.dtype == original.dtype and value.shape == original.shape
                 and np.array_equal(value, original, equal_nan=True), 'COMPONENT_ROUNDTRIP')
    after = verify_native_raw_payload(item)
    need(before == after, 'RAW_HASH_BOOKEND')
    components_ref = {'path': str(destination / 'COMPONENTS.npz'), 'sha256': sha(staging / 'COMPONENTS.npz')}
    ladder_ref = {'path': str(destination / 'LADDER.parquet'), 'sha256': sha(ladder)}
    receipt = {'schema_version': 'gx1_gc_native_m5_raw_file_components_v1',
        'status': 'COMPLETE_RAW_COMPONENTS_NOT_FEATURE_ADMISSION', 'ordinal': ordinal,
        'source': item, 'source_identity_sha256': binding['source_identity_sha256'],
        'producer_plan': producer_plan, 'producer_claim': producer_claim,
        'source_clock': scope, 'global_bucket_offset': (origin - binding['source_clock']['start_ns']) // interval,
        'contract_index': ci, 'raw_symbol': item['raw_symbol'], 'tick_size_fixed_point': tick,
        'original_record_counts': expected, 'actual_scan': report,
        'components': components_ref, 'ladder': ladder_ref, 'ladder_rows': ladder_rows,
        'arrays': {key: {'dtype': value.dtype.str, 'shape': list(value.shape),
            'sha256': hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
            for key, value in arrays.items()},
        'raw_hash_bookends_unchanged': True, 'ladder_and_trade_size_integer_conservation': True,
        'book_integrals': 'raw_sums_and_covered_ns_preserved_no_means_of_means',
        'file_seam': 'book_quality_event_anchor_reset_at_original_file_start',
        'raw_missing_sentinels': 'empty_event_extrema_inf_and_empty_book_NaN_kept_with_counts_no_imputation',
        'observation_admission_performed': False, 'model_forwards': 0, 'optimizer_steps': 0,
        'clean_process_io_firewall': firewall_state()}
    publish(staging / 'RECEIPT.json', receipt)
    for path in staging.iterdir():
        with path.open('rb') as handle:
            __import__('os').fsync(handle.fileno())
    publish_bundle_directory_noreplace(staging, destination)
    return ref(destination / 'RECEIPT.json')


def completed_receipts():
    result = []
    for path in sorted((ROOT / 'FILES').glob('FILE_*/RECEIPT.json')):
        binding = ref(path)
        value = read(binding)
        need(value['status'] == 'COMPLETE_RAW_COMPONENTS_NOT_FEATURE_ADMISSION', 'CHECKPOINT_STATUS')
        need(all(ref(value[key]['path']) == value[key] for key in ('components', 'ladder')),
             'CHECKPOINT_HASH')
        result.append(binding)
    return result


def run(plan_sha):
    began = time.monotonic()
    firewall_state()
    cap = require_capped_cpu_research_producer_execution()
    lock = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 12 * 1024**3
         and cap['swap'] == 512 * 1024**2 and cap['numerical_threads'] == 1
         and lock['scope'] == cap['cgroup_path'][len('/sys/fs/cgroup'):]
         and signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'CAP_LIMITS')
    plan, plan_ref, policy_ref = inspect(plan_sha, admitted=True)
    ROOT.mkdir(parents=True, exist_ok=False)
    claim = publish(ROOT / 'CLAIM.json', {'stage': STAGE, 'status': 'CLAIMED_ONE_USE',
        'plan': plan_ref, 'policy_at_claim': policy_ref, 'max_invocations': 1,
        'remaining_invocations': 0, 'test_data_used': False})
    pool = None
    old = signal.getsignal(signal.SIGALRM)
    def expired(*args):
        raise TimeoutError('GC_NATIVE_M5_RAW_INTERNAL_LIMIT')
    try:
        signal.signal(signal.SIGALRM, expired)
        remaining = plan['execution']['internal_seconds'] - (time.monotonic() - began)
        need(remaining > 0, 'TIME_BEFORE_RAW_WORK')
        signal.setitimer(signal.ITIMER_REAL, remaining)
        pool = ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn'))
        futures = [pool.submit(reduce_file, plan, ordinal, plan_ref, claim)
                   for ordinal in range(len(plan['source_binding']['raw_files']))]
        receipts = [future.result() for future in as_completed(futures)]
        pool.shutdown(wait=True)
        pool = None
        checked_metadata(plan)
        need(source_clean(plan) and ref(POLICY) == policy_ref and ref(PLAN) == plan_ref,
             'SOURCE_POLICY_BOOKEND')
        need(len(receipts) == 66, 'FILE_COMPLETENESS')
        receipts = completed_receipts()
        need(len(receipts) == 66, 'DURABLE_FILE_COMPLETENESS')
        result = publish(ROOT / 'RESULT.json', {'stage': STAGE, 'status': 'COMPLETE_RAW_COMPONENTS_ONLY',
            'plan': plan_ref, 'claim': claim, 'source_unchanged': True, 'files': receipts,
            'records': 892100712, 'native_input_interval_ns': 300000000000,
            'all_raw_payload_hash_bookends_unchanged': True, 'model_forwards': 0, 'optimizer_steps': 0,
            'native_training_admitted': False, 'test_data_used': False,
            'clean_process_io_firewall': firewall_state(), 'elapsed_seconds': time.monotonic() - began})
        terminal = publish(ROOT / 'TERMINAL.json', {'stage': STAGE, 'status': 'COMPLETE',
            'exit_code': 0, 'plan': plan_ref, 'claim': claim, 'result': result,
            'remaining_invocations': 0, 'relaunch_allowed': False, 'test_data_used': False})
        return {'result': result, 'terminal': terminal}
    except BaseException as error:
        if pool is not None:
            processes = list(pool._processes.values())
            pool.shutdown(wait=False, cancel_futures=True)
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
        publish(ROOT / 'TERMINAL.json', {'stage': STAGE, 'status': 'FAILED_CONSUMED',
            'exit_code': 1, 'plan': plan_ref, 'claim': claim,
            'completed_file_receipts': completed_receipts(),
            'error': type(error).__name__ + ':' + str(error)[:2000],
            'remaining_invocations': 0, 'relaunch_allowed': False, 'test_data_used': False})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    firewall_state()
    need(args.plan == str(PLAN) and args.out_dir == str(ROOT), 'CLI_SCOPE')
    if args.preflight:
        _, plan_ref, policy_ref = inspect(args.plan_sha256, admitted=False)
        print(json.dumps({'status': 'PREFLIGHT_METADATA_ONLY', 'plan': plan_ref, 'policy': policy_ref,
            'raw_MBP_records_decoded': 0, 'claim_created': False, 'model_forwards': 0,
            'test_data_used': False, 'clean_process_io_firewall': firewall_state()}))
    else:
        print(json.dumps(run(args.plan_sha256), sort_keys=True))


if __name__ == '__main__':
    main()
