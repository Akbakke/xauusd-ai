"""Synthetic runner boundaries, publication and cutoff checks without market data."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gx1.audit import build_native_m5_direction_targets_v1 as audit


def _lifecycle(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, 'ROOT', tmp_path / 'OUT')
    plan_ref = {'path': str(tmp_path / 'PLAN.json'), 'sha256': '1' * 64}
    policy_ref = {'path': str(tmp_path / 'POLICY.json'), 'sha256': '2' * 64}
    plan = {'source_spec': {}, 'target_contract': {'contract_sha256': '3' * 64}}
    monkeypatch.setattr(audit, 'require_capped_cpu_audit_execution', lambda: {
        'memory_max': 4 * 1024**3, 'memory_high': 4 * 1024**3,
        'swap': 512 * 1024**2, 'numerical_threads': 1,
        'cgroup_path': '/sys/fs/cgroup/synthetic'})
    monkeypatch.setattr(audit, 'require_capped_lock_ancestry',
                        lambda: {'scope': '/synthetic'})
    monkeypatch.setattr(audit, 'inspect', lambda *a, **kw: (plan, plan_ref, policy_ref))
    monkeypatch.setattr(audit, 'checked_inputs', lambda _plan: None)
    monkeypatch.setattr(audit, 'source_clean', lambda _plan: True)
    monkeypatch.setattr(audit, 'ref', lambda path: policy_ref if path == audit.POLICY else plan_ref)
    return plan_ref, policy_ref


def test_success_publishes_bound_target_result_terminal_and_consumes_root(monkeypatch, tmp_path):
    _lifecycle(monkeypatch, tmp_path)
    monkeypatch.setattr(audit, 'calculate', lambda plan: (b'synthetic-target-bytes', {'rows': 2}))
    result = audit.run('1' * 64)
    terminal = json.loads((audit.ROOT / 'TERMINAL.json').read_text())
    saved = json.loads((audit.ROOT / 'RESULT.json').read_text())
    assert terminal['status'] == 'COMPLETE' and terminal['exit_code'] == 0
    assert terminal['result'] == result['result']
    assert terminal['source_unchanged'] is True
    assert saved['target_tape'] == terminal['target_tape'] == result['target_tape']
    assert saved['model_forwards'] == saved['optimizer_steps'] == 0
    assert terminal['remaining_invocations'] == 0 and terminal['relaunch_allowed'] is False
    assert (audit.ROOT / 'TARGETS.npz').read_bytes() == b'synthetic-target-bytes'
    with pytest.raises(FileExistsError):
        audit.run('1' * 64)


def test_failed_calculation_publishes_consumed_terminal_and_no_result(monkeypatch, tmp_path):
    _lifecycle(monkeypatch, tmp_path)
    def failed(_plan):
        raise ValueError('synthetic-failure')
    monkeypatch.setattr(audit, 'calculate', failed)
    with pytest.raises(ValueError, match='synthetic-failure'):
        audit.run('1' * 64)
    terminal = json.loads((audit.ROOT / 'TERMINAL.json').read_text())
    assert terminal['status'] == 'FAILED_CONSUMED' and terminal['result'] is None
    assert terminal['remaining_invocations'] == 0 and terminal['relaunch_allowed'] is False
    assert terminal['error'] == 'ValueError:synthetic-failure'
    assert (audit.ROOT / 'CLAIM.json').is_file()
    assert not (audit.ROOT / 'RESULT.json').exists()
    assert not (audit.ROOT / 'TARGETS.npz').exists()


def test_preflight_checks_same_input_hash_gate_without_decode_or_output(monkeypatch, tmp_path, capsys):
    plan_ref, policy_ref = _lifecycle(monkeypatch, tmp_path)
    events = []
    monkeypatch.setattr(audit, 'checked_inputs', lambda plan: events.append('inputs-hash-gate'))
    def unexpected(*args, **kwargs):
        pytest.fail('preflight decoded data or created output')
    monkeypatch.setattr(audit, 'calculate', unexpected)
    monkeypatch.setattr(audit, 'publish', unexpected)
    monkeypatch.setattr(audit, 'read_source', unexpected)
    monkeypatch.setattr(sys, 'argv', ['audit', '--plan', str(audit.PLAN),
        '--plan-sha256', plan_ref['sha256'], '--out-dir', str(audit.ROOT), '--preflight'])
    audit.main()
    report = json.loads(capsys.readouterr().out)
    assert events == ['inputs-hash-gate']
    assert report['policy'] == policy_ref and report['parquet_values_decoded'] == 0
    assert report['claim_created'] is False and not audit.ROOT.exists()


def test_source_reader_discards_cutoff_close_and_retains_own_m5_clock(tmp_path):
    path = tmp_path / 'part.parquet'
    starts = pd.date_range('2024-09-30 23:45', periods=4, freq='5min', tz='UTC')
    table = pa.table({'time': pa.Array.from_pandas(starts),
        'open': [1.] * 4, 'high': [2.] * 4, 'low': [0.] * 4, 'close': [1.] * 4})
    pq.write_table(table, path, row_group_size=2)
    part = {'path': str(path), 'sha256': audit.sha(path), 'row_groups': [0, 1]}
    plan = {'source_spec': {'parts': [part], 'cutoff_ns': audit.CUTOFF_NS},
            'source_metadata': {str(path): audit.part_metadata(part)}}
    times, ohlc, decoded = audit.read_source(plan)
    assert decoded == 4
    assert times.tolist() == starts.asi8[:2].tolist()
    assert ohlc.shape == (2, 4)
    assert np.all(times + audit.M5_NS < audit.CUTOFF_NS)


def test_native_m1_manifest_rejected_before_parquet_metadata(monkeypatch):
    binding = {'path': str(audit.NATIVE / 'MANIFEST.json'), 'sha256': audit.MANIFEST_SHA}
    monkeypatch.setattr(audit, 'read', lambda _binding: {'timeframe': 'M1'})
    with pytest.raises(ValueError, match='NATIVE_M5_SOURCE_GRAIN'):
        audit.source_manifest(binding)


def test_input_hash_and_stat_must_match_before_claim(monkeypatch, tmp_path):
    path = tmp_path / 'part.parquet'
    path.write_bytes(b'synthetic-source')
    part = {'path': str(path), 'sha256': audit.sha(path), 'row_groups': [0]}
    plan = {'source_spec': {'parts': [part], 'manifest': {}},
            'source_metadata': {str(path): {'stat': audit.stamp(path)}}}
    monkeypatch.setattr(audit, 'source_manifest', lambda binding: {})
    audit.checked_inputs(plan)
    part['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='PART_SHA'):
        audit.checked_inputs(plan)
    part['sha256'] = audit.sha(path)
    path.write_bytes(b'changed-source')
    with pytest.raises(ValueError, match='PART_STAT'):
        audit.checked_inputs(plan)


@pytest.mark.parametrize('fault', ['training', 'unadmitted'])
def test_policy_rejects_before_calculation(monkeypatch, tmp_path, fault):
    monkeypatch.setattr(audit, 'REPO', tmp_path)
    monkeypatch.setattr(audit, 'ROOT', tmp_path / 'OUT')
    plan_ref = {'path': str(audit.PLAN), 'sha256': '1' * 64}
    policy_ref = {'path': str(audit.POLICY), 'sha256': '2' * 64}
    plan = {'source_spec': {}, 'source_base_commit': '3' * 40}
    gate = {'plan': plan_ref, 'output_root': str(audit.ROOT), 'relaunch_allowed': False,
            'max_invocations': 1, 'claimed_invocations': 0, 'completed_invocations': 0,
            'training_authorized': False, 'test_authorized': False,
            'broker_access_allowed': False, 'model_forwards': 0, 'optimizer_steps': 0,
            'status': 'ADMITTED_ONE_USE', 'execution_authorized': True,
            'remaining_invocations': 1}
    policy = {key: False for key in audit.CLOSED_FLAGS}
    policy[audit.STAGE] = gate
    if fault == 'training':
        policy['training_enabled'] = True
    else:
        gate['status'] = 'PREPARED_NOT_ADMITTED'
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(audit, 'build_plan', lambda **kwargs: plan)
    monkeypatch.setattr(audit, 'source_clean', lambda _plan: True)
    monkeypatch.setattr(audit, 'ref', lambda _path: policy_ref)
    monkeypatch.setattr(audit, 'read', lambda binding: plan if binding == plan_ref else policy)
    with pytest.raises(ValueError, match='CLOSED_POLICY|ONE_USE_GATE'):
        audit.inspect(plan_ref['sha256'], admitted=True)
    assert not audit.ROOT.exists()


def test_calculate_publishes_all_bar_comparisons_and_native_future_rows(monkeypatch):
    import io
    from gx1.contracts.market_state_native_m5_targets_v1 import TIMEFRAMES
    starts = pd.date_range('2024-01-01', periods=900, freq='5min', tz='UTC').asi8
    opens = 2000.0 + np.arange(len(starts)) % 4
    closes = np.full(len(starts), 2002.0)
    ohlc = np.column_stack((opens, np.maximum(opens, closes) + .2,
                           np.minimum(opens, closes) - .2, closes))
    monkeypatch.setattr(audit, 'read_source', lambda _plan: (starts, ohlc, len(starts)))
    plan = {'source_spec': {
        'manifest': {'path': '/synthetic/NATIVE_M5_MANIFEST.json', 'sha256': '1' * 64},
        'issue_start_ns': int(starts[300] + audit.M5_NS),
        'cutoff_ns': int(starts[-1] + 2 * audit.M5_NS),
        'parts': [{'path': '/synthetic/year=2024/part-000.parquet'}]}}
    payload, report = audit.calculate(plan)
    with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
        assert len(saved['issue_ns']) == report['selected_issue_rows'] == 600
        assert not any('atr' in key for key in saved.files)
        for col, tf in enumerate(TIMEFRAMES):
            prefix = 'bars_' + tf + '_'
            known = saved[prefix + 'observed_previous']
            bars = saved[prefix + 'ohlc']
            for name, coordinate in (('open', 0), ('close', 3)):
                expected = np.sign(bars[known, coordinate] - saved[prefix + 'previous_' + name][known])
                np.testing.assert_array_equal(saved[prefix + name + '_sign'][known], expected)
            assert np.all(saved[prefix + 'close_sign'][known] == 0)
            coverage = report['coverage_by_timeframe'][tf]
            assert coverage['observed_bars'] == len(bars)
            assert coverage['prior_observed_bars'] == int(known.sum())
            assert sum(coverage['open_sign_counts'].values()) == int(known.sum())
            eligible = saved['future_eligible'][:, col]
            positions = np.searchsorted(saved[prefix + 'row'], saved['future_target_row'][eligible, col])
            np.testing.assert_array_equal(
                saved['future_open_sign'][eligible, col],
                saved[prefix + 'open_sign'][positions])
            np.testing.assert_array_equal(
                saved['future_open_known_at_issue'][eligible, col],
                saved[prefix + 'open_available_ns'][positions] <= saved['issue_ns'][eligible])
            assert coverage['future_unique_eligible_bar_rows'] == len(np.unique(
                saved['future_target_row'][eligible, col]))
            assert np.all(saved['future_validity_ns'][eligible, col] > saved['issue_ns'][eligible])
