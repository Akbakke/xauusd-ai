"""Fresh raw MBP-1 bindings and direct M5 roll; never a legacy feature parent."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from gx1.contracts.gc_continuous_source_v1 import (
    DAY_NS, NATIVE_M5_INTERVAL_NS, NATIVE_M5_REDUCTION_CLOCK_SCHEMA,
    choose_daily_contract, reduction_clock_v1,
)

GC_ROOT = Path('/home/andre2/GX1_DATA/research/GC_ORDER_FLOW_RESEARCH_001')
SOURCE_BINDING_SCHEMA = 'gx1_gc_native_m5_raw_source_binding_v1'


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(part)
    return h.hexdigest()


def _bound_json(ref):
    if (not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}
            or not isinstance(ref['sha256'], str) or len(ref['sha256']) != 64):
        raise ValueError('GC_NATIVE_M5_METADATA_REF')
    path = Path(ref['path'])
    if not path.is_absolute() or _digest(path) != ref['sha256']:
        raise ValueError('GC_NATIVE_M5_METADATA_HASH')
    return json.loads(path.read_text())


def _raw_path(path):
    path = Path(path)
    if not path.is_absolute() or not path.is_file() or not path.resolve().is_relative_to(GC_ROOT.resolve()):
        raise ValueError('GC_NATIVE_M5_RAW_PATH')
    return path


def bind_native_m5_raw_sources(*, source_audit_plan_ref):
    """Check actual inventory/receipts/definitions, without decoding raw MBP.

    Original compressed payload hashes are independently checked by the capped
    raw producer. This metadata gate neither reads old M1 inputs nor accepts
    any derived GC table, registry or parent model clock as a source.
    """
    audit = _bound_json(source_audit_plan_ref)
    if (audit.get('schema_version') != 'gx1_gc_order_flow_source_audit_plan_v1'
            or audit.get('source_audit_schema') != 'gx1_gc_order_flow_source_audit_v2'
            or audit.get('file_count') != 66 or len(audit.get('files', [])) != 66):
        raise ValueError('GC_NATIVE_M5_RAW_INVENTORY_SCHEMA')
    inventory_ref = audit['input_binding_plan']
    inventory = _bound_json(inventory_ref)
    mapping_ref = {k: audit['definition_mapping_audit'][k] for k in ('path', 'sha256')}
    mapping = _bound_json(mapping_ref)
    identity = audit['definition_mapping_audit']['mapping_by_raw_symbol']
    if (mapping.get('mapping_by_raw_symbol') != identity or mapping.get('raw_symbol_mapping_conflicts')
            or mapping.get('source_input_binding_plan_sha256') != inventory_ref['sha256']):
        raise ValueError('GC_NATIVE_M5_ORIGINAL_MAPPING_MISMATCH')
    # The mapping audit is a separately bound original source record. Its exact
    # schema varies from the inline plan; both inventories must name the same IDs.
    inventory_files = {x['local_file_path']: x for x in inventory['source_files'] if x['schema'] == 'mbp-1'}
    raw_files = []
    for item in audit['files']:
        path = _raw_path(item['path'])
        receipt_ref = dict(path=item['receipt_path'], sha256=item['receipt_sha256'])
        _raw_path(receipt_ref['path'])
        receipt = _bound_json(receipt_ref)
        original = inventory_files.get(str(path))
        if (original is None or item['raw_symbol'] not in identity
                or path.stat().st_size != item['bytes']
                or item['sha256'] != original['local_file_sha256_expected_from_receipt']
                or item['receipt_sha256'] != original['receipt_sha256']
                or item['bytes'] != original['local_file_bytes']
                or item['max_records'] != receipt.get('decompressed_csv_bytes')
                or receipt.get('bytes') != item['bytes'] or receipt.get('sha256') != item['sha256']
                or receipt.get('local_path') != str(path)
                or any(receipt.get(k) != x for k, x in dict(provider='databento', dataset='GLBX.MDP3',
                    schema='mbp-1', compression='gzip', pretty_ts=False, pretty_px=False).items())
                or (item['instrument_id'], item['publisher_id']) != (
                    identity[item['raw_symbol']]['instrument_id'], identity[item['raw_symbol']]['publisher_id'])):
            raise ValueError('GC_NATIVE_M5_RAW_RECEIPT_OR_IDENTITY')
        if item['start_ns'] % DAY_NS or item['end_ns'] % DAY_NS or item['start_ns'] >= item['end_ns']:
            raise ValueError('GC_NATIVE_M5_RAW_FILE_DAY_BOUNDARY')
        raw_files.append({k: item[k] for k in ('path', 'sha256', 'bytes', 'receipt_path', 'receipt_sha256',
            'raw_symbol', 'instrument_id', 'publisher_id', 'start_ns', 'end_ns', 'max_records')})
    if len({x['path'] for x in raw_files}) != 66 or len({x['raw_symbol'] for x in raw_files}) != 15:
        raise ValueError('GC_NATIVE_M5_RAW_INVENTORY_DUPLICATE')
    source_clock = dict(schema_version=NATIVE_M5_REDUCTION_CLOCK_SCHEMA,
        start_ns=min(x['start_ns'] for x in raw_files), end_ns=max(x['end_ns'] for x in raw_files),
        bar_duration_ns=NATIVE_M5_INTERVAL_NS)
    reduction_clock_v1(source_clock)
    contract_metadata = {symbol: set() for symbol in identity}
    definition_refs = []
    for item in inventory['source_files']:
        if item['schema'] != 'definition':
            continue
        path = _raw_path(item['local_file_path'])
        ref = dict(path=str(path), sha256=item['local_file_sha256_expected_from_receipt'])
        receipt_ref = dict(path=item['receipt_path'], sha256=item['receipt_sha256'])
        _raw_path(receipt_ref['path'])
        receipt = _bound_json(receipt_ref)
        if (path.stat().st_size != item['local_file_bytes'] or _digest(path) != ref['sha256']
                or receipt.get('sha256') != ref['sha256'] or receipt.get('bytes') != item['local_file_bytes']
                or receipt.get('schema') != 'definition' or item['raw_symbol'] not in identity):
            raise ValueError('GC_NATIVE_M5_DEFINITION_BINDING')
        rows = 0
        with gzip.open(path, 'rt', newline='') as f:
            for row in csv.DictReader(f):
                symbol = row['raw_symbol']
                if (symbol != item['raw_symbol'] or (int(row['instrument_id']), int(row['publisher_id'])) != (
                        identity[symbol]['instrument_id'], identity[symbol]['publisher_id'])):
                    raise ValueError('GC_NATIVE_M5_DEFINITION_IDENTITY')
                contract_metadata[symbol].add((int(row['maturity_year']), int(row['maturity_month']),
                                               int(row['min_price_increment'])))
                rows += 1
        if not rows:
            raise ValueError('GC_NATIVE_M5_EMPTY_DEFINITION')
        definition_refs.append(dict(source=ref, receipt=receipt_ref, rows=rows))
    if len(definition_refs) != 23 or any(len(v) != 1 for v in contract_metadata.values()):
        raise ValueError('GC_NATIVE_M5_DEFINITION_INVENTORY_OR_NONCONSTANT_CONTRACT')
    metadata = {s: next(iter(v)) for s, v in contract_metadata.items()}
    if any(y <= 0 or not 1 <= m <= 12 or tick <= 0 for y, m, tick in metadata.values()):
        raise ValueError('GC_NATIVE_M5_CONTRACT_METADATA_DOMAIN')
    # Reject overlapping raw windows instead of silently double-counting events.
    for symbol in metadata:
        files = sorted((i for i in raw_files if i['raw_symbol'] == symbol), key=lambda x: x['start_ns'])
        if any(a['end_ns'] > b['start_ns'] for a, b in zip(files, files[1:])):
            raise ValueError('GC_NATIVE_M5_RAW_OVERLAP')
    payload = dict(schema_version=SOURCE_BINDING_SCHEMA, source_clock=source_clock,
        source_audit_plan=source_audit_plan_ref, source_inventory=inventory_ref, original_mapping_audit=mapping_ref,
        raw_files=sorted(raw_files, key=lambda x: (x['raw_symbol'], x['start_ns'])),
        definition_sources=definition_refs, contract_metadata={s: list(v) for s, v in sorted(metadata.items())},
        raw_payload_hashes_checked=False, derived_feature_parents=[], m1_data_or_metadata_used=False,
        future_trade_volume_used_for_roll=False)
    payload['source_identity_sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return payload


def verify_native_raw_payload(item):
    path = _raw_path(item['path'])
    before = path.stat()
    if before.st_size != item['bytes'] or _digest(path) != item['sha256']:
        raise ValueError('GC_NATIVE_M5_RAW_PAYLOAD_HASH')
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('GC_NATIVE_M5_RAW_PAYLOAD_CHANGED_DURING_HASH')
    return dict(path=str(path), sha256=item['sha256'], bytes=after.st_size)


def selected_native_m5_contracts(daily_volumes, maturities, *, source_clock):
    """Day d uses only completed d-1; no legacy one-minute roll allowance."""
    begin, end, interval = reduction_clock_v1(source_clock)
    daily = np.asarray(daily_volumes)
    days = (end - begin) // DAY_NS
    if (begin % DAY_NS or end % DAY_NS or daily.ndim != 2 or daily.shape != (len(maturities), days)
            or daily.dtype != np.dtype('uint64')):
        raise ValueError('GC_NATIVE_M5_DAILY_VOLUME_CLOCK')
    chosen, basis_day = choose_daily_contract(daily, maturities)
    selected = np.repeat(chosen, DAY_NS // interval)
    basis_available_ns = np.where(basis_day >= 0, begin + (basis_day.astype(np.int64) + 1) * DAY_NS, -1)
    decision_ns = begin + (np.arange(len(selected), dtype=np.int64) + 1) * interval
    row_basis = np.repeat(basis_available_ns, DAY_NS // interval)
    if np.any((selected >= 0) & ((row_basis < 0) | (row_basis >= decision_ns))):
        raise ValueError('GC_NATIVE_M5_ROLL_FUTURE_DAY')
    return selected, chosen, basis_day
