"""One native GC input assembly from durable file components; no raw MBP replay."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import signal
import subprocess
import time

if __name__ == '__main__':
    from gx1.contracts.native_m5_direction_io_firewall_v1 import install_native_m5_direction_io_firewall
    install_native_m5_direction_io_firewall()

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from gx1.audit import build_native_m5_gc_raw_components_v1 as raw
from gx1.audit import build_native_m5_direction_targets_v1 as native

REPO = raw.REPO
STAGE = 'gc_native_m5_assembly_20261119_001'
PLAN = REPO / 'configs/research/GC_NATIVE_M5_ASSEMBLY_20261119_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001/GC_NATIVE_M5_ASSEMBLY_20261119_001')
POLICY = REPO / 'NEXT_RUN_POLICY.json'
EXECUTION = {**raw.EXECUTION, 'workers': 1}
INPUT_KEYS = {'raw_components_result', 'raw_components_terminal', 'native_source_plan', 'source_incidents'}
SOURCE_PATHS = (
    'gx1/audit/assemble_native_m5_gc_inputs_v1.py', 'tests/test_assemble_native_m5_gc_inputs_v1.py',
    'gx1/audit/build_native_m5_gc_raw_components_v1.py',
    'gx1/audit/build_native_m5_direction_targets_v1.py',
    'gx1/contracts/gc_native_m5_source_inputs_v1.py',
    'gx1/contracts/gc_continuous_source_v1.py', 'gx1/contracts/gc_source_disposition_v1.py',
    'gx1/contracts/gc_order_flow_source_v1.py',
    'gx1/features/gc_native_m5_features_v1.py', 'gx1/features/gc_order_flow_v1.py',
    'gx1/features/htf_features.py', 'gx1/features/smc_v1.py',
    'gx1/time/session_detector.py', 'gx1/features/gc_native_context_v1.py',
    'gx1/contracts/entry_gc_mtf_history_v1.py',
    'gx1/contracts/market_state_native_m5_feature_inputs_v1.py',
    'gx1/contracts/market_state_native_m5_gc_feature_inputs_v1.py',
    'gx1/contracts/native_m5_direction_io_firewall_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py', 'gx1/contracts/immutable_event_authority_v1.py',
    'scripts/gx1_capped_run.sh',
)
sha, ref, read, publish = raw.sha, raw.ref, raw.read, raw.publish


def need(ok, reason):
    if not bool(ok):
        raise ValueError('GC_NATIVE_M5_ASSEMBLY_' + reason)


def file_metadata(receipt, ordinal, source_binding, raw_plan_ref, raw_claim_ref):
    item = source_binding['raw_files'][ordinal]
    clock = {**source_binding['source_clock'], 'start_ns': item['start_ns'], 'end_ns': item['end_ns']}
    interval = clock['bar_duration_ns']
    symbols = list(source_binding['contract_metadata'])
    ci = symbols.index(item['raw_symbol'])
    need(receipt['schema_version'] == 'gx1_gc_native_m5_raw_file_components_v1'
         and receipt['status'] == 'COMPLETE_RAW_COMPONENTS_NOT_FEATURE_ADMISSION'
         and receipt['ordinal'] == ordinal and receipt['source'] == item
         and receipt['producer_plan'] == raw_plan_ref and receipt['producer_claim'] == raw_claim_ref
         and receipt['source_identity_sha256'] == source_binding['source_identity_sha256']
         and receipt['source_clock'] == clock
         and receipt['global_bucket_offset'] == (clock['start_ns'] - source_binding['source_clock']['start_ns']) // interval
         and receipt['contract_index'] == ci
         and receipt['tick_size_fixed_point'] == source_binding['contract_metadata'][item['raw_symbol']][2]
         and receipt['raw_hash_bookends_unchanged'] is True
         and receipt['ladder_and_trade_size_integer_conservation'] is True
         and receipt['observation_admission_performed'] is False
         and receipt['model_forwards'] == receipt['optimizer_steps'] == 0,
         'FILE_RECEIPT_IDENTITY')
    need(clock['start_ns'] % 86400000000000 == clock['end_ns'] % 86400000000000 == 0
         and interval == 300000000000, 'ORIGINAL_FILE_DAY_CLOCK')
    from gx1.contracts import gc_continuous_source_v1 as source
    from gx1.features import gc_order_flow_v1 as formulas
    n=(clock['end_ns']-clock['start_ns'])//interval
    headers={'bar_start_ns':('<i8',[n]),'ohlc_tick':('<i8',[n,4]),
        'first_trade_recv_ns':('<i8',[n]),'last_trade_recv_ns':('<i8',[n]),
        'book_sums':('<f8',[n,12]),'book_covered_ns':('<i8',[n]),
        'book_first_mid_tick':('<f8',[n]),'book_last_mid_tick':('<f8',[n]),
        'book_first_recv_ns':('<i8',[n]),'book_last_recv_ns':('<i8',[n]),
        'order_sums':('<f8',[n,len(formulas.L1_ORDER_COUNT_FIELDS_V1)]),'order_covered_ns':('<i8',[n])}
    for prefix,names,dtype in (('metric',source.METRICS,'<u8'),
        ('quality',source.L1_QUALITY_METRICS,'<u8'),('event',formulas.EVENT_DIAGNOSTIC_REDUCERS_V1,'<f8'),
        ('trade_size',formulas.TRUSTED_TRADE_SIZE_FIELDS_V1,'<u8')):
        headers.update({prefix+'_'+name:(dtype,[n])for name in names})
    need(set(receipt['arrays'])==set(headers)and all(
        receipt['arrays'][name]['dtype']==dtype and receipt['arrays'][name]['shape']==shape
        for name,(dtype,shape)in headers.items()),'COMPONENT_METADATA_HEADERS')
    return receipt


def input_metadata(inputs):
    need(type(inputs) is dict and set(inputs) == INPUT_KEYS, 'INPUT_ROLES')
    result, terminal = read(inputs['raw_components_result']), read(inputs['raw_components_terminal'])
    raw_plan_ref, claim_ref = result['plan'], result['claim']
    plan, claim = read(raw_plan_ref), read(claim_ref)
    need(result['stage'] == terminal['stage'] == plan['stage'] == claim['stage'] == raw.STAGE
         and result['status'] == 'COMPLETE_RAW_COMPONENTS_ONLY'
         and terminal['status'] == 'COMPLETE' and terminal['exit_code'] == 0
         and terminal['result'] == inputs['raw_components_result']
         and terminal['plan'] == claim['plan'] == raw_plan_ref and terminal['claim'] == claim_ref
         and result['source_unchanged'] is True and result['all_raw_payload_hash_bookends_unchanged'] is True
         and result['test_data_used'] is terminal['test_data_used'] is False
         and result['model_forwards'] == result['optimizer_steps'] == 0
         and result['native_input_interval_ns'] == 300000000000
         and result['native_training_admitted'] is False and result['records'] == 892100712 and len(result['files']) == 66,
         'RAW_COMPONENTS_COMPLETE_CHAIN')
    binding = plan['source_binding']
    from gx1.contracts import gc_continuous_source_v1 as source
    from gx1.contracts.gc_native_m5_source_inputs_v1 import SOURCE_BINDING_SCHEMA
    source.reduction_clock_v1(binding['source_clock'])
    payload={key:value for key,value in binding.items()if key!='source_identity_sha256'}
    need(plan['schema_version']=='gx1_gc_native_m5_raw_components_plan_v1'
         and binding['schema_version']==SOURCE_BINDING_SCHEMA
         and binding['source_identity_sha256']==hashlib.sha256(json.dumps(payload,sort_keys=True,
             separators=(',',':'),allow_nan=False).encode()).hexdigest(),'RAW_BINDING_SEAL')
    need(binding['source_clock']['start_ns'] == 1714521600000000000
         and binding['source_clock']['end_ns'] == 1782864000000000000
         and len(binding['raw_files']) == 66 and len(binding['contract_metadata']) == 15
         and binding['m1_data_or_metadata_used'] is False
         and plan['inputs']['source_incidents'] == inputs['source_incidents'], 'NATIVE_SOURCE_SCOPE')
    refs, files = {}, {}
    for binding_ref in result['files']:
        receipt = read(binding_ref)
        ordinal = receipt['ordinal']
        need(type(ordinal) is int and 0 <= ordinal < 66 and ordinal not in files, 'FILE_ORDINAL')
        files[ordinal] = file_metadata(receipt, ordinal, binding, raw_plan_ref, claim_ref)
        expected=plan['expected_per_file'][receipt['source']['path']]
        need(receipt['original_record_counts']==expected
             and receipt['actual_scan']['records']==expected['records']
             and receipt['actual_scan']['raw_trade_records']==expected['raw_trade_records'],
             'ORIGINAL_PER_FILE_RECORD_COUNTS')
        refs[ordinal] = binding_ref
    need(set(files) == set(range(66)), 'EXACT_66_FILE_RECEIPTS')
    source = read(inputs['native_source_plan'])
    need(inputs['native_source_plan']['path'] == str(native.PLAN) and source['stage'] == native.STAGE
         and source['source_spec']['issue_start_ns'] == native.ISSUE_START_NS
         and source['source_spec']['cutoff_ns'] == 1782864000000000000, 'NATIVE_XAU_HISTORY003')
    incidents = read(inputs['source_incidents'])
    need(incidents['nontrade_action_semantics_admitted'] is False
         and incidents['trusted_trade_admission']['reject_duplicate_signature'] is False
         and incidents['trusted_trade_admission']['deduplicate_T'] is False, 'VENDOR_T_ADMISSION')
    return {'raw_components_plan': raw_plan_ref, 'raw_components_claim': claim_ref,
        'raw_source_binding': binding, 'files': [files[i] for i in range(66)],
        'raw_components': [refs[i] for i in range(66)],
        'native_xau_source_spec': {key: source['source_spec'][key] for key in ('manifest', 'parts', 'cutoff_ns')},
        'native_xau_source_metadata': source['source_metadata']}


def build_plan(*, inputs, source_base_commit=None):
    from gx1.features.gc_native_m5_features_v1 import native_m5_field_catalog, native_m5_source_unit_policy
    from gx1.contracts.native_m5_direction_io_firewall_v1 import contract
    metadata = input_metadata(inputs)
    from gx1.contracts.market_state_native_m5_feature_inputs_v1 import RAW_COLUMNS
    for item in metadata['native_xau_source_spec']['parts']:
        need(all(name in pq.ParquetFile(item['path']).schema_arrow.names for name in ('time',*RAW_COLUMNS)),
             'XAU_13_NATIVE_COLUMNS')
    head = source_base_commit or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    return {'schema_version': 'gx1_gc_native_m5_assembly_plan_v1', 'stage': STAGE,
        'output_root': str(ROOT), 'status': 'PREPARED_NOT_ADMITTED',
        'source_base_commit': head, 'source_owners': {p: sha(REPO / p) for p in SOURCE_PATHS},
        'inputs': inputs, 'metadata': metadata, 'execution': EXECUTION,
        'catalog_contract': native_m5_field_catalog(), 'unit_policy': native_m5_source_unit_policy(),
        'io_firewall': contract(), 'raw_replay': False, 'model_forwards': 0, 'optimizer_steps': 0,
        'training_authorized': False, 'test_data_used': False, 'economic_edge_proven': False}


def source_clean(plan):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    return (subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip()
        == 'work/gx1-current' and not subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)
        and subprocess.run(['git', 'merge-base', '--is-ancestor', plan['source_base_commit'], head],
                           cwd=REPO, capture_output=True).returncode == 0
        and subprocess.run(['git', 'diff', '--quiet', plan['source_base_commit'], head,
                           '--', 'gx1', 'scripts'], cwd=REPO).returncode == 0
        and set(plan['source_owners']) == set(SOURCE_PATHS)
        and all(sha(REPO / path) == digest for path, digest in plan['source_owners'].items()))


def checked_inputs(plan, *, payloads):
    meta = plan['metadata']
    bindings = [*plan['inputs'].values(), meta['raw_components_plan'], meta['raw_components_claim'],
                *meta['raw_components'], meta['native_xau_source_spec']['manifest']]
    if payloads:
        for receipt in meta['files']:
            bindings += [receipt['components'], receipt['ladder']]
        bindings += [{'path': item['path'], 'sha256': item['sha256']}
                     for item in meta['native_xau_source_spec']['parts']]
    for binding in bindings:
        need(ref(binding['path']) == binding, 'INPUT_HASH_BOOKEND')
    for item in meta['native_xau_source_spec']['parts']:
        need(native.part_metadata(item) == meta['native_xau_source_metadata'][item['path']],
             'NATIVE_XAU_METADATA')


def inspect(plan_sha, *, admitted):
    need(Path.cwd().resolve() == REPO and not ROOT.exists(), 'CWD_OUTPUT')
    plan_ref = {'path': str(PLAN), 'sha256': plan_sha}
    plan = read(plan_ref)
    need(plan == build_plan(inputs=plan['inputs'], source_base_commit=plan['source_base_commit'])
         and source_clean(plan), 'PLAN_SOURCE')
    policy_ref = ref(POLICY)
    policy = read(policy_ref)
    gate = policy.get(STAGE, {})
    need(all(policy.get(key) is False for key in native.CLOSED_FLAGS)
         and gate.get('plan') == plan_ref and gate.get('output_root') == str(ROOT)
         and gate.get('max_invocations') == 1 and gate.get('relaunch_allowed') is False
         and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
         and gate.get('training_authorized') is False and gate.get('test_authorized') is False
         and gate.get('model_forwards') == gate.get('optimizer_steps') == 0, 'CLOSED_POLICY')
    expected = ('ADMITTED_ONE_USE', True, 1) if admitted else ('PREPARED_NOT_ADMITTED', False, 0)
    need((gate.get('status'), gate.get('execution_authorized'), gate.get('remaining_invocations')) == expected,
         'ONE_USE_GATE')
    checked_inputs(plan, payloads=False)
    return plan, plan_ref, policy_ref


def component_arrays(receipt, names):
    with np.load(receipt['components']['path'], allow_pickle=False) as saved:
        need(set(names).issubset(saved.files), 'COMPONENT_COLUMNS')
        values = {}
        for name in names:
            value = saved[name]
            descriptor = receipt['arrays'][name]
            need(value.dtype.str == descriptor['dtype'] and list(value.shape) == descriptor['shape']
                 and hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == descriptor['sha256'],
                 'COMPONENT_ARRAY_BYTES:' + name)
            values[name] = value
    return values


def receive_clock_exclusion(source_clock, incidents):
    need(incidents.get('schema_version')=='gx1_gc_vendor_source_incidents_v1'
         and incidents.get('dataset')=='GLBX.MDP3' and incidents.get('incidents')
         and incidents.get('nontrade_action_semantics_admitted') is False, 'ROLL_INCIDENT_SCOPE')
    starts=np.arange(source_clock['start_ns'],source_clock['end_ns'],
                     source_clock['bar_duration_ns'],dtype=np.int64)
    excluded=np.zeros(len(starts),bool)
    for incident in incidents['incidents']:
        begin,end=(pd.Timestamp(incident[key])for key in ('start_inclusive_utc','end_exclusive_utc'))
        need(begin.tz is not None and end.tz is not None
             and begin.utcoffset()==end.utcoffset()==pd.Timedelta(0)
             and begin.value<end.value and incident.get('incident_id')and incident.get('source_url'),
             'ROLL_INCIDENT_INTERVAL')
        excluded|=(starts<end.value)&(starts+source_clock['bar_duration_ns']>begin.value)
    return excluded


def daily_roll(source_binding, receipts, incidents):
    from gx1.contracts.gc_native_m5_source_inputs_v1 import selected_native_m5_contracts
    clock = source_binding['source_clock']
    days = (clock['end_ns']-clock['start_ns']) // 86400000000000
    symbols = list(source_binding['contract_metadata'])
    daily = np.zeros((len(symbols), days), np.uint64)
    raw_daily=daily.copy();excluded_volume=daily.copy();excluded_trades=daily.copy()
    excluded_buckets=np.zeros_like(daily,np.int64)
    covered = np.zeros_like(daily, bool)
    receive_excluded=receive_clock_exclusion(clock,incidents)
    for receipt in receipts:
        values=component_arrays(receipt, ['metric_volume','metric_trades'])
        volume,trades=values['metric_volume'],values['metric_trades']
        span = receipt['source_clock']
        ci = receipt['contract_index']
        offset = (span['start_ns']-clock['start_ns']) // 86400000000000
        n = (span['end_ns']-span['start_ns']) // 86400000000000
        need(volume.dtype==trades.dtype==np.uint64 and volume.shape==trades.shape==(n*288,)
             and not covered[ci, offset:offset+n].any(), 'DAILY_SOURCE_COVERAGE')
        reject=receive_excluded[offset*288:(offset+n)*288]
        raw_daily[ci,offset:offset+n]=volume.reshape(n,288).sum(axis=1,dtype=np.uint64)
        excluded_volume[ci,offset:offset+n]=np.where(reject,volume,0).reshape(n,288).sum(axis=1,dtype=np.uint64)
        excluded_trades[ci,offset:offset+n]=np.where(reject,trades,0).reshape(n,288).sum(axis=1,dtype=np.uint64)
        excluded_buckets[ci,offset:offset+n]=reject.reshape(n,288).sum(axis=1,dtype=np.int64)
        daily[ci,offset:offset+n]=np.where(reject,0,volume).reshape(n,288).sum(axis=1,dtype=np.uint64)
        covered[ci, offset:offset+n] = True
    need(np.array_equal(raw_daily,daily+excluded_volume),'ROLL_INCIDENT_VOLUME_CONSERVATION')
    maturities = [tuple(source_binding['contract_metadata'][symbol][:2]) for symbol in symbols]
    selected, chosen, basis = selected_native_m5_contracts(daily, maturities, source_clock=clock)
    evidence={'receive_clock_excluded':receive_excluded,'daily_volume_before_incident':raw_daily,
        'daily_volume_admitted_to_roll':daily,'daily_excluded_volume':excluded_volume,
        'daily_excluded_trusted_T_records':excluded_trades,'daily_excluded_native_buckets':excluded_buckets,
        'partial_observation_days_due_to_incident':covered&(excluded_buckets>0),
        'full_feed_or_288_observed_bars_per_day_claimed':False,
        'roll_volume_admission':'per_record_trusted_T_volume_minus_published_receive_clock_incident_buckets',
        'feature_volume_admission':'separate_conservative_whole_bucket_row_masks',
        'bad_T_or_book_gap_does_not_erase_other_trusted_T_from_roll':True,
        'receive_clock_incident_excludes_entire_native_bucket_from_roll':True,
        'basis_available_ns':np.repeat(np.where(basis>=0,
            clock['start_ns']+(basis.astype(np.int64)+1)*86400000000000,-1),288)}
    return selected, daily, covered, chosen, basis, evidence


def selected_components(source_binding, receipts, selected, incidents):
    from gx1.contracts import gc_continuous_source_v1 as source
    from gx1.features import gc_order_flow_v1 as formulas
    clock = source_binding['source_clock']
    interval = clock['bar_duration_ns']
    starts = np.arange(clock['start_ns'], clock['end_ns'], interval, dtype=np.int64)
    need(selected.dtype == np.int16 and selected.shape == starts.shape, 'SELECTED_NATIVE_AXIS')
    ticks = {meta[2] for meta in source_binding['contract_metadata'].values()}
    need(len(ticks) == 1, 'SINGLE_FIXED_TICK')
    tick = next(iter(ticks))
    width = len(starts)
    collector = formulas.L1FeatureCollectorV2(np.arange(width, dtype=np.int64), tick,
        origin_ns=clock['start_ns'], include_order_counts=True, include_event_diagnostics=True,
        include_trusted_trade_stats=True, input_interval_ns=interval)
    metrics = {name: np.zeros(width, np.uint64) for name in source.METRICS}
    quality = {name: np.zeros(width, np.uint64) for name in source.L1_QUALITY_METRICS}
    ohlc = np.zeros((width, 4), np.int64)
    first_recv = np.full(width, -1, np.int64); last_recv = first_recv.copy()
    filled = np.zeros(width, bool)
    attr_names = {'book_sums':'sums', 'book_covered_ns':'covered', 'book_first_mid_tick':'first',
        'book_last_mid_tick':'last', 'book_first_recv_ns':'first_time', 'book_last_recv_ns':'last_time',
        'order_sums':'order_sums', 'order_covered_ns':'order_covered'}
    for receipt in receipts:
        n = (receipt['source_clock']['end_ns']-receipt['source_clock']['start_ns']) // interval
        offset = receipt['global_bucket_offset']
        ci = receipt['contract_index']
        need(0 <= offset < width and offset+n <= width, 'COMPONENT_GLOBAL_OFFSET')
        group_names = [('metric', metrics), ('quality', quality),
            ('event', collector.event_stats), ('trade_size', collector.trade_size_stats)]
        names = ['bar_start_ns','ohlc_tick','first_trade_recv_ns','last_trade_recv_ns',*attr_names]
        names += [prefix+'_'+name for prefix, group in group_names for name in group]
        arrays = component_arrays(receipt, names)
        need(np.array_equal(arrays['bar_start_ns'], starts[offset:offset+n]), 'COMPONENT_CLOCK')
        take = selected[offset:offset+n] == ci
        global_rows = offset + np.flatnonzero(take)
        need(not filled[global_rows].any(), 'DUPLICATE_SELECTED_COVERAGE')
        filled[global_rows] = True
        for prefix, group in group_names:
            for name, destination in group.items():
                value = arrays[prefix+'_'+name]
                need(value.shape == (n,) and value.dtype == destination.dtype, 'COMPONENT_DOMAIN:'+prefix+'_'+name)
                destination[global_rows] = value[take]
        for name, attr in attr_names.items():
            destination, value = getattr(collector, attr), arrays[name]
            need(value.shape == (n, *destination.shape[1:]) and value.dtype == destination.dtype,
                 'COMPONENT_BOOK_DOMAIN:'+name)
            destination[global_rows] = value[take]
        need(arrays['ohlc_tick'].dtype == np.int64 and arrays['ohlc_tick'].shape == (n,4), 'COMPONENT_OHLC')
        ohlc[global_rows] = arrays['ohlc_tick'][take]
        fr,lr=arrays['first_trade_recv_ns'],arrays['last_trade_recv_ns']
        has_trade=arrays['metric_trades']>0
        local_starts=starts[offset:offset+n]
        need(fr.dtype==lr.dtype==np.int64 and fr.shape==lr.shape==(n,)
             and np.all((fr[has_trade]>=local_starts[has_trade])&(lr[has_trade]<local_starts[has_trade]+interval)
                        &(fr[has_trade]<=lr[has_trade]))
             and np.all(fr[~has_trade]==-1) and np.all(lr[~has_trade]==-1),'TRUSTED_T_RECV_CLOCK')
        first_recv[global_rows] = fr[take]
        last_recv[global_rows] = lr[take]
    need(np.array_equal(metrics['volume'], metrics['buy']+metrics['sell']+metrics['unknown']),
         'SELECTED_SIDE_CONSERVATION')
    need(np.all((collector.covered>=0)&(collector.covered<=interval))
         and np.all((collector.order_covered>=0)&(collector.order_covered<=interval)), 'BOOK_INTERVAL_COVERAGE')
    clean = (quality['records'] > 0) & (quality['clean_complete_bbo'] > 0)
    for name in ('bad_bbo','gap','bad_recv','snapshot','reset','file_seam'):
        clean &= quality[name] == 0
    prior = np.r_[False, clean[:-1] & (selected[1:] == selected[:-1]) & (selected[1:] >= 0)]
    ofi = clean & prior & (selected >= 0) & filled
    masks = source.row_masks(selected, metrics, starts+interval < clock['end_ns'],
                             reject_duplicate_signature=False)
    exclusion = source.source_incident_masks_v1(starts, selected, quality, incidents,
        reject_duplicate_signature=False, input_interval_ns=interval)
    for name in masks:
        masks[name] &= ~exclusion['receive_clock_excluded']
    ofi &= ~exclusion['book_event_state_excluded']
    book, bbo = collector.values(ofi)
    orders, order_valid = collector.order_count_values(ofi)
    book.update(orders)
    book.update(ofi_valid=ofi, bbo_valid=bbo, order_count_valid=order_valid,
        covered_ns=collector.covered, order_covered_ns=collector.order_covered,
        opening_mid_tick=collector.first, closing_mid_tick=collector.last)
    events = collector.event_diagnostics(ofi)
    stats = collector.trusted_trade_statistics()
    need(np.array_equal(formulas.trusted_trade_record_count_v1(stats), metrics['trades'])
         and np.array_equal(stats['size_sum_contracts'], metrics['volume']), 'SELECTED_TRADE_STAT_PARITY')
    return {'starts':starts,'selected':selected,'metrics':metrics,'quality':quality,'masks':masks,'ohlc':ohlc,
        'book':book,'events':events,'trade_stats':stats,'first_recv':first_recv,'last_recv':last_recv,
        'filled':filled,'source_exclusions':exclusion,'tick':tick}


def selected_ladder(source_binding, receipts, state, destination):
    width = len(state['starts'])
    need(width < 2**31, 'LADDER_INDEX_WIDTH')
    totals = np.zeros((width,3), np.uint64)
    symbols = list(source_binding['contract_metadata'])
    ordered = sorted(receipts, key=lambda item: (
        *source_binding['contract_metadata'][symbols[item['contract_index']]][:2],
        item['source_clock']['start_ns']))
    schema = pa.schema([('bucket_index',pa.int32()),('price_tick',pa.int64()),
        ('buy',pa.uint64()),('sell',pa.uint64()),('unknown',pa.uint64())])
    last = -1; rows = 0
    with pq.ParquetWriter(destination, schema, compression='zstd') as writer:
        for receipt in ordered:
            offset = receipt['global_bucket_offset']; ci = receipt['contract_index']
            n = (receipt['source_clock']['end_ns']-receipt['source_clock']['start_ns']) // 300000000000
            for batch in pq.ParquetFile(receipt['ladder']['path']).iter_batches(batch_size=65536):
                data = {name:batch.column(batch.schema.get_field_index(name)).to_numpy()
                        for name in batch.schema.names}
                local = data['bucket_index']
                need(local.dtype == np.int32 and np.all((local >= 0)&(local < n))
                     and np.all(data['price_tick'] > 0), 'LADDER_LOCAL_CLOCK')
                global_rows = local.astype(np.int64)+offset
                take = (state['selected'][global_rows] == ci)&state['masks']['volume'][global_rows]
                ids = global_rows[take]
                if not len(ids):continue
                need(ids[0] >= last and np.all(np.diff(ids) >= 0), 'LADDER_GLOBAL_ORDER')
                last = int(ids[-1])
                columns = {'bucket_index':ids.astype(np.int32), 'price_tick':data['price_tick'][take]}
                for j,name in enumerate(('buy','sell','unknown')):
                    need(data[name].dtype == np.uint64, 'LADDER_UNSIGNED')
                    columns[name] = data[name][take]
                    np.add.at(totals[:,j],ids,columns[name])
                writer.write_table(pa.table(columns,schema=schema))
                rows += len(ids)
    need(all(np.array_equal(totals[:,j],np.where(state['masks']['volume'],state['metrics'][name],0))
             for j,name in enumerate(('buy','sell','unknown'))), 'SELECTED_LADDER_SIDE_CONSERVATION')
    return {'rows':rows,'side_volume_contracts':{name:int(totals[:,j].sum())for j,name in enumerate(('buy','sell','unknown'))},
            'global_bucket_order_nondecreasing':True,'per_side_integer_conservation':True}


def read_native_xau(plan):
    from gx1.contracts.market_state_native_m5_feature_inputs_v1 import RAW_COLUMNS, bind_native_m5_feature_source
    spec = plan['metadata']['native_xau_source_spec']
    pieces = []
    need(spec['cutoff_ns'] == 1782864000000000000, 'XAU_EXCLUSIVE_TEST_CUTOFF')
    for item in spec['parts']:
        need(native.part_metadata(item) == plan['metadata']['native_xau_source_metadata'][item['path']],
             'XAU_PART_METADATA')
        parquet = pq.ParquetFile(item['path'])
        for group in item['row_groups']:
            frame = parquet.read_row_group(group, columns=['time',*RAW_COLUMNS], use_threads=False).to_pandas()
            starts = pd.to_datetime(frame.pop('time'),utc=True,errors='raise')
            frame.index = pd.DatetimeIndex(starts).as_unit('ns')
            pieces.append(frame.loc[frame.index.asi8+300000000000 < spec['cutoff_ns'],list(RAW_COLUMNS)])
    value = pd.concat(pieces)
    binding = bind_native_m5_feature_source(value,native_m5_manifest=spec['manifest'],source_spec=spec)
    return value, binding


def publish_arrays(stem, value, *, packet=False):
    arrays, layout = {}, {}
    def visit(node,path):
        if isinstance(node,np.ndarray):
            key='__'.join(path)
            need(node.dtype.kind in 'biuf' and key not in arrays, 'NUMERIC_PACKET_ARRAY')
            arrays[key]=node
            layout[key]={'path':list(path),'dtype':node.dtype.str,'shape':list(node.shape),
                'sha256':hashlib.sha256(np.ascontiguousarray(node).tobytes()).hexdigest()}
            return None
        if isinstance(node,dict):return {key:visit(part,(*path,key))for key,part in node.items()}
        if isinstance(node,(list,tuple)):return [visit(part,(*path,str(i)))for i,part in enumerate(node)]
        if isinstance(node,np.generic):return node.item()
        return node
    metadata = visit(value,())
    if packet:metadata={key:part for key,part in value.items()if key not in ('timeframes','m5_book_features')}
    stream=io.BytesIO();np.savez_compressed(stream,**arrays)
    archive=publish(ROOT/(stem+'.npz'),stream.getvalue());stream.close()
    with np.load(archive['path'],allow_pickle=False)as saved:
        need(set(saved.files)==set(arrays),'PUBLISHED_ARRAY_INVENTORY')
        for key,value in arrays.items():
            loaded=saved[key]
            need(loaded.dtype==value.dtype and loaded.shape==value.shape
                 and np.array_equal(loaded,value,equal_nan=True)
                 and hashlib.sha256(np.ascontiguousarray(loaded).tobytes()).hexdigest()==layout[key]['sha256'],
                 'PUBLISHED_ARRAY_ROUNDTRIP')
    envelope={'schema_version':'gx1_gc_native_m5_numeric_packet_v1' if packet else 'gx1_gc_native_m5_assembly_state_v1',
        'packet_metadata':metadata,'array_layout':layout,'archive':archive,
        'all_array_dtype_shape_equal_nan_sha256_verified':True}
    if packet:
        from gx1.features.gc_native_m5_features_v1 import NATIVE_NULLABLE_TIMESTAMP_COLUMNS
        envelope['nullable_timestamp_columns']=list(NATIVE_NULLABLE_TIMESTAMP_COLUMNS)
    return {'json':publish(ROOT/(stem+'.json'),envelope),'npz':archive}


def assemble(plan, plan_ref, claim):
    from gx1.features import gc_native_m5_features_v1 as features
    from gx1.contracts.market_state_native_m5_gc_feature_inputs_v1 import bind_native_m5_gc_feature_packet
    meta = plan['metadata'];binding=meta['raw_source_binding'];receipts=meta['files']
    incidents=read(plan['inputs']['source_incidents'])
    selected,daily,covered,chosen,basis,roll_evidence=daily_roll(binding,receipts,incidents)
    state=selected_components(binding,receipts,selected,incidents)
    need(np.array_equal(roll_evidence['receive_clock_excluded'],
         state['source_exclusions']['receive_clock_excluded']),'ROLL_AND_FEATURE_CLOCK_EXCLUSION_PARITY')
    ladder_path=ROOT/'SELECTED_LADDER.parquet'
    ladder_proof=selected_ladder(binding,receipts,state,ladder_path)
    state_artifacts=publish_arrays('ASSEMBLY_STATE',{**state,'daily_volume':daily,
        'daily_source_covered':covered,'chosen_daily':chosen,'basis_daily':basis,'roll_evidence':roll_evidence})
    native_xau,xau_binding=read_native_xau(plan)
    packet=features.build_native_m5_gc_surfaces(scope=binding['source_clock'],selected=selected,
        metrics=state['metrics'],masks=state['masks'],ohlc=state['ohlc'],ladder_path=ladder_path,
        book=state['book'],events=state['events'],trade_stats=state['trade_stats'],
        tick_size=state['tick'],native_xau_m5=native_xau,
        first_trusted_trade_recv_ns=state['first_recv'],last_trusted_trade_recv_ns=state['last_recv'])
    del native_xau
    coverage=features.source_coverage_report(packet)
    need(coverage['all_declared_source_columns_present'] is True
         and not coverage['source_boolean_fields_without_binary_declaration'], 'GENUINE_FIELD_DOMAINS')
    artifacts=publish_arrays('GC_PACKET',packet,packet=True)
    checked_inputs(plan,payloads=True)
    need(source_clean(plan) and ref(PLAN)==plan_ref
         and ref(POLICY)==read(claim)['policy_at_claim'], 'SOURCE_BOOKEND_BEFORE_PROOF')
    proof={'schema_version':'gx1_gc_native_m5_assembly_proof_v1','status':'COMPLETE',
        'source_identity_sha256':binding['source_identity_sha256'],'source_clock':binding['source_clock'],
        'native_xau_m5_source_binding_sha256':xau_binding['contract_sha256'],
        'raw_components_plan':meta['raw_components_plan'],'raw_components_claim':meta['raw_components_claim'],
        'raw_components_result':plan['inputs']['raw_components_result'],
        'raw_components_terminal':plan['inputs']['raw_components_terminal'],
        'raw_components':meta['raw_components'],'m1_data_or_metadata_used':False,'raw_replay':False,
        'packet_artifacts':artifacts,'catalog_contract_sha256':features.native_m5_field_catalog()['contract_sha256'],
        'unit_policy_sha256':features.native_m5_source_unit_policy()['contract_sha256'],
        'source_unchanged':True,'test_data_used':False}
    proof_ref=publish(ROOT/'ASSEMBLY_PROOF.json',proof)
    feature_binding=bind_native_m5_gc_feature_packet(packet,raw_source_binding=binding,
        native_xau_m5_source_binding=xau_binding,assembly_proof_ref=proof_ref,artifact_refs=artifacts)
    binding_ref=publish(ROOT/'GC_SOURCE_BINDING.json',feature_binding)
    return {'packet_artifacts':artifacts,'assembly_proof':proof_ref,'source_binding':binding_ref,
        'assembly_state_artifacts':state_artifacts,'selected_ladder':ref(ladder_path),'ladder_proof':ladder_proof,
        'source_coverage':coverage,'native_xau_source_binding':xau_binding,
        'native_rows':len(selected),'unknown_contract_rows':int((selected<0).sum()),
        'volume_admitted_rows':int(state['masks']['volume'].sum()),
        'signed_admitted_rows':int(state['masks']['delta'].sum()),
        'roll_incident_excluded_volume_contracts':int(roll_evidence['daily_excluded_volume'].sum()),
        'roll_incident_excluded_trusted_T_records':int(roll_evidence['daily_excluded_trusted_T_records'].sum()),
        'roll_partial_observation_contract_days':int(roll_evidence['partial_observation_days_due_to_incident'].sum()),'model_forwards':0,'optimizer_steps':0}


def run(plan_sha):
    began=time.monotonic();raw.firewall_state()
    cap=raw.require_capped_cpu_research_producer_execution();lock=raw.require_capped_lock_ancestry()
    need(cap['memory_max']==cap['memory_high']==12*1024**3 and cap['swap']==512*1024**2
         and cap['numerical_threads']==1 and lock['scope']==cap['cgroup_path'][len('/sys/fs/cgroup'):]
         and signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'CAP_LIMITS')
    plan,plan_ref,policy_ref=inspect(plan_sha,admitted=True)
    ROOT.mkdir(parents=True,exist_ok=False)
    claim=publish(ROOT/'CLAIM.json',{'stage':STAGE,'status':'CLAIMED_ONE_USE','plan':plan_ref,
        'policy_at_claim':policy_ref,'max_invocations':1,'remaining_invocations':0,'test_data_used':False})
    old=signal.getsignal(signal.SIGALRM);result=None
    def expired(*args):raise TimeoutError('GC_NATIVE_M5_ASSEMBLY_INTERNAL_LIMIT')
    try:
        remaining=plan['execution']['internal_seconds']-(time.monotonic()-began)
        need(remaining>0,'TIME_BEFORE_COMPONENT_READ')
        signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
        checked_inputs(plan,payloads=True)
        report=assemble(plan,plan_ref,claim)
        checked_inputs(plan,payloads=True)
        need(source_clean(plan)and ref(POLICY)==policy_ref and ref(PLAN)==plan_ref,'SOURCE_POLICY_BOOKEND')
        result=publish(ROOT/'RESULT.json',{'stage':STAGE,'status':'COMPLETE_NATIVE_GC_INPUTS_ONLY',
            'plan':plan_ref,'claim':claim,'source_unchanged':True,'report':report,'raw_replay':False,
            'model_forwards':0,'optimizer_steps':0,'training_authorized':False,'test_data_used':False,
            'economic_edge_proven':False,'clean_process_io_firewall':raw.firewall_state(),
            'elapsed_seconds':time.monotonic()-began})
        terminal=publish(ROOT/'TERMINAL.json',{'stage':STAGE,'status':'COMPLETE','exit_code':0,
            'plan':plan_ref,'claim':claim,'result':result,'remaining_invocations':0,
            'relaunch_allowed':False,'test_data_used':False})
        return {'result':result,'terminal':terminal}
    except BaseException as error:
        publish(ROOT/'TERMINAL.json',{'stage':STAGE,'status':'FAILED_CONSUMED','exit_code':1,
            'plan':plan_ref,'claim':claim,'result':result,
            'error':type(error).__name__+':'+str(error)[:2000],
            'remaining_invocations':0,'relaunch_allowed':False,'test_data_used':False})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',required=True);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--out-dir',required=True);parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args();raw.firewall_state()
    need(args.plan==str(PLAN)and args.out_dir==str(ROOT),'CLI_SCOPE')
    if args.preflight:
        _,plan_ref,policy_ref=inspect(args.plan_sha256,admitted=False)
        print(json.dumps({'status':'PREFLIGHT_GENUINE_COMPLETED_COMPONENT_METADATA_ONLY',
            'plan':plan_ref,'policy':policy_ref,'claim_created':False,'raw_replay':False,
            'component_arrays_decoded':0,'model_forwards':0,'test_data_used':False}))
    else:print(json.dumps(run(args.plan_sha256),sort_keys=True))


if __name__=='__main__':main()
