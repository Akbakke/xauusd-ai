"""Focused tests for the physical GC reference's source and axis guards."""
import hashlib

import numpy as np
import pytest

from gx1.audit import score_market_state_gc_flow_reference_v1 as worker
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as target


def _sha(values):
    return hashlib.sha256(values.tobytes()).hexdigest()


def test_axis_identity_requires_parent_row_index_and_registry_clock():
    parent = np.arange(4, dtype=np.int64)
    decision = (np.arange(4, dtype=np.int64) + 2) * target.M5_NS
    provider = {'axes': {'rows': 4, 'parent_rows_sha256': _sha(parent),
                         'bar_start_ns_sha256': _sha(decision - target.M5_NS)}}
    registry = {'train_parent_clock_binding': {
        'bar_clock_sha256': provider['axes']['bar_start_ns_sha256']},
        'train_decision_clock_sha256': _sha(decision)}
    assert worker._axis_identity(parent, decision, provider, registry) == {
        'parent_rows_sha256': _sha(parent),
        'bar_start_ns_sha256': _sha(decision - target.M5_NS),
        'decision_ns_sha256': _sha(decision)}
    wrong_parent = parent.copy()
    wrong_parent[-1] = 8
    with pytest.raises(RuntimeError, match='TARGET_PARENT_AXIS'):
        worker._axis_identity(wrong_parent, decision, provider, registry)
    wrong_clock = decision.copy()
    wrong_clock[-1] += target.M5_NS
    with pytest.raises(RuntimeError, match='TARGET_GC_CLOCK'):
        worker._axis_identity(parent, wrong_clock, provider, registry)


def test_masked_hash_uses_observation_bytes_and_stored_float32_bytes():
    values = np.array([3.0, 0.0, -2.0], dtype=np.float32)
    observed = np.array([True, False, True], dtype=np.bool_)
    expected = hashlib.sha256(b'\x01\x00\x01' + values.tobytes()).hexdigest()
    assert worker._masked_hash(values, observed) == expected
    changed = values.copy()
    changed[1] = 10.0
    assert worker._masked_hash(changed, observed) != expected
    with pytest.raises(RuntimeError, match='MASKED_HASH_INPUT'):
        worker._masked_hash(values.astype(np.float64), observed)


def test_volume_and_delta_parity_fail_closed_on_joint_observed_rows():
    names = ('gc_m5_observed_volume', 'gc_m5_observed_buy_known',
             'gc_m5_observed_sell_known', 'gc_m5_observed_unknown',
             'gc_m5_observed_known_delta_contracts')
    arrays = (
        [10.0, 7.0, 4.0], [6.0, 3.0, 1.0], [3.0, 2.0, 2.0],
        [1.0, 2.0, 1.0], [3.0, 1.0, -1.0],
    )
    values = {}
    for name, data in zip(names, arrays):
        values[name] = np.asarray(data, dtype=np.float32)
        values[name + '_observed'] = np.asarray([True, True, False])
    assert worker._flow_parity(values)['joint_observed_rows'] == 2
    values['gc_m5_observed_known_delta_contracts'][0] = 4.0
    with pytest.raises(RuntimeError, match='TRADE_ARITHMETIC'):
        worker._flow_parity(values)
    values['gc_m5_observed_known_delta_contracts'][0] = 3.0
    values['gc_m5_observed_volume'][1] = 8.0
    with pytest.raises(RuntimeError, match='TRADE_ARITHMETIC'):
        worker._flow_parity(values)


def test_weekly_result_writer_supports_bounded_payload_and_no_replace(tmp_path):
    payload = {'weekly_evidence': 'x' * (300 * 1024)}
    ref = worker._write(tmp_path, 'RESULT.json', payload)
    stored = (tmp_path / 'RESULT.json').read_bytes()
    assert len(stored) > 256 * 1024
    assert len(stored) <= worker.MAX_JSON_BYTES
    assert hashlib.sha256(stored).hexdigest() == ref['sha256']
    with pytest.raises(RuntimeError, match='immutable event already exists'):
        worker._write(tmp_path, 'RESULT.json', payload)
    with pytest.raises(RuntimeError, match='PUBLICATION_SIZE_OR_ROUNDTRIP'):
        worker._write(tmp_path, 'TOO_BIG.json',
                      {'weekly_evidence': 'x' * worker.MAX_JSON_BYTES})
    assert not (tmp_path / 'TOO_BIG.json').exists()
