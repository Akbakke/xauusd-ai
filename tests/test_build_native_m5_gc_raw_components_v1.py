"""Synthetic native file checkpoints and one-use guards; no genuine event scans."""
from concurrent.futures import Future
import csv
import gzip
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gx1.audit import build_native_m5_gc_raw_components_v1 as audit


def raw_plan(monkeypatch, tmp_path):
    from gx1.contracts import gc_continuous_source_v1 as source
    from gx1.contracts import gc_native_m5_source_inputs_v1 as binder
    origin = 1714521600000000000
    day = 86400000000000
    start = origin + day
    values = dict(rtype=1, publisher_id=1, instrument_id=42, symbol='GCM4',
        action='T', side='B', depth=0, price=100100000000, size=1,
        flags=128, ts_in_delta=0, sequence=0, bid_px_00=100000000000,
        ask_px_00=100100000000, bid_sz_00=10, ask_sz_00=12, bid_ct_00=2, ask_ct_00=3)
    rows = []
    for offset, side, size, action, sequence in (
        (1, 'B', 1, 'T', 0), (1, 'B', 1, 'T', 0),
        (300000000000, 'A', 2, 'T', 1), (900000000000, 'N', 3, 'T', 2),
        (1000000000000, 'B', 100, 'F', 3)):
        rows.append({**values, 'ts_recv': start + offset, 'ts_event': start + offset,
                     'side': side, 'size': size, 'action': action, 'sequence': sequence})
    path = tmp_path / 'raw.csv.gz'
    columns = {**source.COLUMNS, **source.L1_AUDIT_COLUMNS}
    with gzip.open(path, 'wt', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)
    item = {'path': str(path), 'sha256': audit.sha(path), 'bytes': path.stat().st_size,
        'raw_symbol': 'GCM4', 'instrument_id': 42, 'publisher_id': 1,
        'start_ns': start, 'end_ns': start + day, 'max_records': 100,
        'receipt_path': str(tmp_path / 'RAW_RECEIPT.json'), 'receipt_sha256': 'a' * 64}
    clock = {'schema_version': source.NATIVE_M5_REDUCTION_CLOCK_SCHEMA,
        'start_ns': origin, 'end_ns': start + day, 'bar_duration_ns': source.NATIVE_M5_INTERVAL_NS}
    plan = {'source_binding': {'raw_files': [item], 'source_clock': clock,
        'source_identity_sha256': 'b' * 64, 'contract_metadata': {'GCM4': [2024, 6, 100000000]}},
        'expected_per_file': {str(path): {'records': 5, 'raw_trade_records': 4, 'bad_receive_records': 0}}}
    monkeypatch.setattr(audit, 'ROOT', tmp_path / 'OUT')
    monkeypatch.setattr(binder, 'GC_ROOT', tmp_path)
    monkeypatch.setattr(audit, 'firewall_state', lambda: {'denied_attempts': 0})
    return plan


def test_native_original_file_checkpoint_keeps_all_T_and_exact_sufficient_statistics(monkeypatch, tmp_path):
    plan = raw_plan(monkeypatch, tmp_path)
    proof = {'path': '/synthetic/PRODUCER.json', 'sha256': 'c' * 64}
    binding = audit.reduce_file(plan, 0, proof, proof)
    receipt = audit.read(binding)
    assert receipt['global_bucket_offset'] == 288
    assert receipt['producer_plan'] == receipt['producer_claim'] == proof
    assert receipt['observation_admission_performed'] is False
    assert receipt['source_clock']['bar_duration_ns'] == 300000000000
    with np.load(receipt['components']['path'], allow_pickle=False) as saved:
        assert saved['bar_start_ns'].shape == (288,)
        assert saved['metric_volume'].sum() == 7
        assert saved['metric_buy'].sum() == 2
        assert saved['metric_sell'].sum() == 2
        assert saved['metric_unknown'].sum() == 3
        assert saved['metric_duplicate_trade'].sum() == 1
        assert saved['trade_size_size_sum_squared_contracts2'].sum() == 15
        assert saved['book_sums'].shape == (288, 12)
        assert saved['metric_records'].sum() == 5
        assert saved['metric_trades'].sum() == 4
        assert all('minute' not in name.lower() for name in saved.files)
    assert audit.completed_receipts() == [binding]
    assert not list((audit.ROOT / 'FILES').glob('.*.staging'))


@pytest.mark.parametrize('fault', ('hash', 'count'))
def test_failed_original_file_never_publishes_reusable_checkpoint(monkeypatch, tmp_path, fault):
    plan = raw_plan(monkeypatch, tmp_path)
    if fault == 'hash':
        plan['source_binding']['raw_files'][0]['sha256'] = 'f' * 64
    else:
        next(iter(plan['expected_per_file'].values()))['records'] = 6
    with pytest.raises(ValueError, match='(PAYLOAD_HASH|ORIGINAL_RECORD_COUNTS)'):
        audit.reduce_file(plan, 0, {}, {})
    assert audit.completed_receipts() == []


def test_build_plan_is_metadata_only_before_any_raw_or_feature_decode(monkeypatch):
    from gx1.contracts import gc_continuous_source_v1 as source
    from gx1.contracts import gc_native_m5_source_inputs_v1 as binder
    inputs = {'source_audit_plan': {
        'path': str(audit.REPO / 'configs/research/GC_SOURCE_AUDIT_20261016_002.json'),
        'sha256': '3cca1eef4bd965e0194a5f353f76645fbb6a030c509d0f3d24850745f4d0310d'},
        'source_audit_result': {'path': '/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001/GC_SOURCE_QUALIFICATION_20261016_002/AUDIT_RESULTS/RESULT.json',
            'sha256': 'c869e3d1395c5892fab19d7c3056faaca1ca4aaac41f70d3f05943d4e63dcbab'},
        'source_incidents': audit.ref(audit.REPO / 'configs/research/GC_VENDOR_SOURCE_INCIDENTS_20261018_001.json')}
    files = [{'path': '/synthetic/file' + str(i), 'sha256': 'a' * 64,
        'receipt_path': '/synthetic/receipt' + str(i), 'receipt_sha256': 'b' * 64} for i in range(66)]
    original = [{'path': x['path'], 'sha256': x['sha256'], 'receipt_sha256': x['receipt_sha256'],
        'counters': {'records': 892100647 if i == 0 else 1}}
        for i, x in enumerate(files)]
    binding = {'source_clock': {'start_ns': 1714521600000000000, 'end_ns': 1782864000000000000},
               'raw_files': files}
    monkeypatch.setattr(binder, 'bind_native_m5_raw_sources', lambda **kwargs: binding)
    incidents = {'trusted_trade_admission': {'deduplicate_T': False, 'reject_duplicate_signature': False},
                 'nontrade_action_semantics_admitted': False}
    def metadata(ref):
        if ref == inputs['source_audit_result']:
            return {'files': original, 'record_audit_performed': True, 'test_accessed': False}
        if ref == inputs['source_incidents']:
            return incidents
        return {'header_fields': list({**source.COLUMNS, **source.L1_AUDIT_COLUMNS})}
    monkeypatch.setattr(audit, 'read', metadata)
    monkeypatch.setattr(source, 'scan_symbol', lambda *a, **k: pytest.fail('raw decode before claim'))
    monkeypatch.setattr(audit.pq, 'ParquetFile', lambda *a, **k: pytest.fail('prior table opened'))
    plan = audit.build_plan(inputs=inputs, source_base_commit='c' * 40)
    assert plan['execution']['workers'] == 4
    assert plan['models'] == plan['optimizer_steps'] == 0
    assert plan['native_training_admitted'] is False
    assert len(plan['expected_per_file']) == 66


def lifecycle(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, 'ROOT', tmp_path / 'OUT')
    plan = {'execution': audit.EXECUTION, 'source_binding': {'raw_files': [{}] * 2}}
    plan_ref = {'path': '/synthetic/PLAN.json', 'sha256': '1' * 64}
    policy_ref = {'path': '/synthetic/POLICY.json', 'sha256': '2' * 64}
    monkeypatch.setattr(audit, 'firewall_state', lambda: {'denied_attempts': 0})
    monkeypatch.setattr(audit, 'inspect', lambda *a, **k: (plan, plan_ref, policy_ref))
    monkeypatch.setattr(audit, 'require_capped_cpu_research_producer_execution', lambda: {
        'memory_max': 12 * 1024**3, 'memory_high': 12 * 1024**3, 'swap': 512 * 1024**2,
        'numerical_threads': 1, 'cgroup_path': '/sys/fs/cgroup/synthetic'})
    monkeypatch.setattr(audit, 'require_capped_lock_ancestry', lambda: {'scope': '/synthetic'})
    return plan


def test_failed_raw_job_preserves_completed_file_refs_and_consumes_claim(monkeypatch, tmp_path):
    lifecycle(monkeypatch, tmp_path)
    completed = {'path': '/original/FILE_001/RECEIPT.json', 'sha256': '3' * 64}
    monkeypatch.setattr(audit, 'completed_receipts', lambda: [completed])
    class Pool:
        _processes = {}
        def __init__(self, **kwargs):
            assert kwargs['max_workers'] == 4
        def submit(self, *args):
            future = Future()
            future.set_exception(ValueError('synthetic-bad-file'))
            return future
        def shutdown(self, **kwargs):
            pass
    monkeypatch.setattr(audit, 'ProcessPoolExecutor', Pool)
    with pytest.raises(ValueError, match='synthetic-bad-file'):
        audit.run('1' * 64)
    terminal = json.loads((audit.ROOT / 'TERMINAL.json').read_text())
    assert terminal['status'] == 'FAILED_CONSUMED' and terminal['remaining_invocations'] == 0
    assert terminal['completed_file_receipts'] == [completed]
    assert terminal['relaunch_allowed'] is False
    assert not (audit.ROOT / 'RESULT.json').exists()


def test_preclaim_refusal_creates_no_raw_checkpoint_or_claim(monkeypatch, tmp_path):
    lifecycle(monkeypatch, tmp_path)
    def refused(*args, **kwargs):
        raise ValueError('missing-native-source')
    monkeypatch.setattr(audit, 'inspect', refused)
    with pytest.raises(ValueError, match='missing-native-source'):
        audit.run('1' * 64)
    assert not audit.ROOT.exists()
