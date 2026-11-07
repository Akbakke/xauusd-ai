"""Small contract tests; never run a physical oracle or read cache payloads."""
import json
import signal
from types import SimpleNamespace

import numpy as np
import pytest

from gx1.audit import verify_m5_m15_h1_trendline_oracle_v1 as audit
from gx1.contracts import gx1_capped_execution_v1 as cap_contract


def test_small_prior_receipts_are_bound_without_payload_hashing(monkeypatch):
    def no_payload_hash(path):
        if str(path).endswith('.npy') or str(path).endswith('.parquet'):
            raise AssertionError('payload hash before claim')
        return original_checked(path)

    original_checked = audit.h4.checked
    monkeypatch.setattr(audit.h4, 'checked', no_payload_hash)
    inputs, manifest, witnesses = audit._small_receipts()
    assert len(inputs['native_parts']) == 16
    assert manifest['feature_names'][159:190] == list(audit.NAMES)
    assert {tf: len(witnesses[tf]) for tf in audit.TIMEFRAMES} == {
        'M5': 11, 'M15': 10, 'H1': 10}


def test_closed_bar_window_and_exact_native_clock(monkeypatch):
    monkeypatch.setattr(audit, 'TRAIN_START_NS', 1000)
    monkeypatch.setattr(audit, 'TRAIN_END_NS', 5000)
    monkeypatch.setitem(audit.TF_SCOPE, 'H1', {
        'bar_ns': 10, 'prefix_rows': 5, 'first_row': 1, 'last_row': 3})
    cache = np.array([980, 990, 1990, 3990, 4990, 5990], dtype=np.int64)
    native = np.array([100, 980, 990, 1990, 3990, 4990], dtype=np.int64)
    rows, positions, times = audit._align_rows(cache, native, 'H1')
    assert rows.tolist() == [1, 2, 3]
    assert positions.tolist() == [2, 3, 4]
    assert times.tolist() == [990, 1990, 3990]
    with pytest.raises(RuntimeError, match='NATIVE_JOIN'):
        audit._align_rows(cache, np.array([100, 980, 990, 1991, 3990, 4990],
                                          dtype=np.int64), 'H1')


def test_full_mismatch_and_anchor_counts_ignore_example_cap():
    rows = np.arange(10, 15, dtype=np.int64)
    times = rows * 100
    oracle = np.ones((5, 31), dtype=np.float64)
    cache = np.zeros((5, 31), dtype=np.float32)
    comparison = audit._compare(oracle, cache, rows, times, {13, 14})
    assert comparison['mismatched_cells'] == 155
    assert comparison['anchor_mismatched_cells'] == 62
    assert comparison['mismatch_examples_truncated'] is True
    assert len(comparison['mismatch_examples']) == 128
    assert comparison['lifecycle_mismatched_cells'] == 5 * 19
    assert comparison['all_train_window_cells_match'] is False


def test_float_tolerance_and_equal_nan():
    rows = np.array([10, 11], dtype=np.int64)
    times = np.array([100, 200], dtype=np.int64)
    oracle = np.zeros((2, 31), dtype=np.float64)
    cache = np.zeros((2, 31), dtype=np.float32)
    j = next(i for i in range(31) if i not in audit.h4.DISCRETE)
    oracle[0, j] = 1.0
    cache[0, j] = 1.0 + 1e-7
    oracle[1, j] = np.nan
    cache[1, j] = np.nan
    comparison = audit._compare(oracle, cache, rows, times, set())
    assert comparison['mismatched_cells'] == 0
    assert comparison['all_train_window_cells_match'] is True
    assert comparison['per_field'][j]['oracle_finite_rows'] == 1


@pytest.mark.parametrize('failure', [None, 'error', 'timeout'])
def test_one_use_claim_and_terminal_on_success_or_failure(tmp_path, monkeypatch, failure):
    out = tmp_path / 'out'
    plan_file = tmp_path / 'plan.json'
    plan_file.write_text('{}')
    plan_ref = {'path': str(plan_file), 'sha256': audit._sha(plan_file)}
    policy_file = tmp_path / 'policy.json'
    policy_file.write_text(json.dumps({
        'training_enabled': False, 'native_training_enabled': False,
        'native_launch_authorized': False, 'full_val_allowed': False,
        audit.STAGE: {'status': 'ADMITTED', 'decision': 'ADMITTED',
                      'plan': plan_ref,
                      'execution_authorized': True, 'data_reads_authorized': True,
                      'claimed_invocations': 0, 'completed_invocations': 0,
                      'max_invocations': 1, 'relaunch_allowed': False,
                      'memory_max': '4G', 'swap_max': '512M',
                      'wall_seconds': 1800, 'test_data_used': False,
                      'checkpoint_reads_authorized': False,
                      'broker_access_allowed': False,
                      'model_forwards_authorized': 0,
                      'optimizer_steps_authorized': 0,
                      'training_authorized': False, 'test_authorized': False,
                      'output_root': str(out)}}))
    monkeypatch.setattr(audit, 'POLICY', policy_file)
    monkeypatch.setattr(audit, 'OUT_ROOT', out)
    monkeypatch.setattr(audit, '_clean_head', lambda: 'a' * 40)
    monkeypatch.setattr(audit, 'sys', SimpleNamespace(modules={}))
    monkeypatch.setattr(audit, '_inspect_plan', lambda *args: (
        {}, plan_ref, 'a' * 40, {}, {}, {}))
    monkeypatch.setattr(cap_contract, 'require_capped_cpu_audit_execution',
                        lambda: {**audit.LIMITS, 'pids': 64,
                                 'numerical_threads': 1,
                                 'cgroup_path': '/sys/fs/cgroup/test'})
    monkeypatch.setattr(cap_contract, 'require_capped_lock_ancestry',
                        lambda: {'scope': '/test'})

    def measure(*_args):
        if failure == 'error':
            raise RuntimeError('synthetic oracle failure')
        if failure == 'timeout':
            signal.raise_signal(signal.SIGALRM)
        return {'timeframes': {
            tf: {'comparison': {'mismatched_cells': 0,
                                'unexercised_lifecycle_fields': []}}
            for tf in audit.TIMEFRAMES},
            'model_forwards': 0, 'optimizer_steps': 0,
            'test_data_used': False}

    monkeypatch.setattr(audit, '_measure', measure)
    old_handler = signal.getsignal(signal.SIGALRM)
    if failure:
        expected = TimeoutError if failure == 'timeout' else RuntimeError
        with pytest.raises(expected):
            audit.run(plan_file, plan_ref['sha256'], out)
        assert {p.name for p in out.iterdir()} == {
            'CLAIM.json', 'FAILED.json', 'TERMINAL.json'}
        assert json.loads((out / 'TERMINAL.json').read_text())['status'] == 'FAILED'
    else:
        audit.run(plan_file, plan_ref['sha256'], out)
        assert {p.name for p in out.iterdir()} == {
            'CLAIM.json', 'RESULT.json', 'TERMINAL.json'}
        assert json.loads((out / 'RESULT.json').read_text())['decision'] == (
            'PASS_M5_M15_H1_TRENDLINE_19_EXERCISED')
    assert signal.getsignal(signal.SIGALRM) == old_handler
    terminal = json.loads((out / 'TERMINAL.json').read_text())
    assert terminal['no_relaunch_under_this_plan'] is True
    assert terminal['test_data_used'] is False
    with pytest.raises(FileExistsError):
        audit.run(plan_file, plan_ref['sha256'], out)
