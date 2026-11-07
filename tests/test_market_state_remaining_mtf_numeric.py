"""Synthetic gates for the one-use 63-column TRAIN physical verifier."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.audit import verify_market_state_remaining_mtf_numeric_v1 as worker
from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4
from gx1.features.technical_indicators_v1 import wilder_atr, wilder_rsi


def _manifest_and_index() -> tuple[dict, dict]:
    names = list(MULTI_TF_PER_BAR_FEATURES_V4)
    old = set(range(253)) - set(worker.EXPECTED_COLUMNS) - set(range(159, 190)) - {236}
    rows = [{'tf': tf, 'cache_column': col, 'name': names[col],
             'numeric_reference_qualified': True}
            for tf in worker.TIMEFRAMES for col in sorted(old)]
    return ({'feature_count': 253, 'feature_names': names},
            {'qualified_per_tf': 158, 'rows': rows, 'cache_manifest': {},
             'test_data_used': False})


def test_exact_63_ordinal_name_complement_and_corruption_gates() -> None:
    manifest, qualified = _manifest_and_index()
    names = worker._field_contract(manifest, qualified)
    assert len(names) == len(set(names)) == 63
    assert [manifest['feature_names'][i] for i in worker.EXPECTED_COLUMNS] == list(names)
    broken = dict(qualified)
    broken['rows'] = [dict(row) for row in qualified['rows']]
    broken['rows'][0]['cache_column'] = 23  # steals a residual field
    with pytest.raises(RuntimeError, match='QUALIFIED_INDEX_M5'):
        worker._field_contract(manifest, broken)
    broken = dict(qualified)
    broken['rows'] = [dict(row) for row in qualified['rows']]
    broken['rows'][0]['name'] = 'wrong-name'
    with pytest.raises(RuntimeError, match='QUALIFIED_INDEX_M5'):
        worker._field_contract(manifest, broken)
    broken_manifest = dict(manifest)
    broken_manifest['feature_names'] = list(manifest['feature_names'])
    broken_manifest['feature_names'][23] = 'wrong-name'
    with pytest.raises(RuntimeError, match='ORACLE_ORDINAL_NAMES'):
        worker._field_contract(broken_manifest, qualified)


def test_clock_alignment_requires_exact_closed_train_window(monkeypatch) -> None:
    monkeypatch.setattr(worker, 'TRAIN_START_NS', 20)
    monkeypatch.setattr(worker, 'TRAIN_END_NS', 40)
    monkeypatch.setitem(worker.TF_SCOPE, 'M5', {
        'bar_ns': 5, 'prefix_rows': 9, 'first_row': 3, 'last_row': 6})
    cache = np.arange(0, 55, 5, dtype=np.int64)
    native = np.arange(-10, 55, 5, dtype=np.int64)
    rows, pos, times = worker._align_rows(cache, native, 'M5')
    np.testing.assert_array_equal(rows, [3, 4, 5, 6])
    np.testing.assert_array_equal(times, [15, 20, 25, 30])
    np.testing.assert_array_equal(native[pos], times)
    bad = native.copy()
    bad[np.where(bad == 25)[0][0]] = 26
    with pytest.raises(RuntimeError, match='NATIVE_JOIN_M5'):
        worker._align_rows(cache, bad, 'M5')


def test_continuous_tolerance_mask_and_discrete_age_exactness() -> None:
    manifest, qualified = _manifest_and_index()
    names = worker._field_contract(manifest, qualified)
    rows = np.arange(3, dtype=np.int64)
    times = np.arange(3, dtype=np.int64) + 10
    expected = np.zeros((3, 63), dtype=np.float32)
    observed = expected.copy()
    observed[0, names.index('vol_z_20')] = np.float32(1e-6)
    assert worker._compare(expected, observed, rows, times, names)['mismatched_cells'] == 0
    observed[0, names.index('volatility.squeeze_release_age_bars')] = np.float32(1.0)
    report = worker._compare(expected, observed, rows, times, names)
    assert report['mismatched_cells'] == 1
    assert report['mismatch_examples'][0]['field'] == 'volatility.squeeze_release_age_bars'
    observed = expected.copy()
    observed[0, names.index('vol_z_20')] = np.nan
    assert worker._compare(expected, observed, rows, times, names)['mismatched_cells'] == 1


def test_raw_float64_rsi_atr_prerequisite_binds_prior_qualified_cache_columns() -> None:
    high = np.array([101.0, 102.0], dtype=np.float64)
    low = np.array([99.0, 100.0], dtype=np.float64)
    close = np.array([100.0, 101.0], dtype=np.float64)
    rsi = np.array([70.00000001, 30.00000001], dtype=np.float64)
    atr = np.array([1.5, 1.6], dtype=np.float64)
    cached = np.column_stack((atr / close * 1e4, (rsi - 50.0) / 50.0)).astype(np.float32)
    report = worker._require_prerequisites(high, low, close, rsi, atr, cached)
    assert report['rsi14_raw_float64'] is True
    assert rsi[0] > 70.0 and np.float32(rsi[0]) == np.float32(70.0)
    cached[0, 1] += 0.1
    with pytest.raises(RuntimeError, match='QUALIFIED_RAW_RSI_ATR_CACHE_BINDING'):
        worker._require_prerequisites(high, low, close, rsi, atr, cached)


def test_nine_oracles_fill_one_synthetic_full_prefix_without_skipped_column() -> None:
    manifest, qualified = _manifest_and_index()
    names = worker._field_contract(manifest, qualified)
    n = 260
    row = np.arange(n, dtype=np.float64)
    close = 2000 + 0.07 * row + 2 * np.sin(row * 0.27)
    span = 0.5 + 0.2 * (1 + np.sin(row * 0.19))
    index = pd.date_range('2024-05-01', periods=n, freq='5min', tz='UTC')
    bars = pd.DataFrame({'open': close - 0.1, 'high': close + span,
                         'low': close - span, 'close': close,
                         'volume': 100.0 + row % 13}, index=index)
    rsi = wilder_rsi(bars['close'], 14).to_numpy(dtype=np.float64)
    atr = wilder_atr(bars['high'], bars['low'], bars['close'], 14).to_numpy(dtype=np.float64)
    pos = np.array([219, 250, 259], dtype=np.int64)
    fit = {'bandwidth_mean': [0.002, 0.006],
           'bandwidth_variance': [1e-6, 1e-5],
           'transition_probability': [[0.98, 0.02], [0.03, 0.97]],
           'initial_probability': [0.5, 0.5]}
    actual = worker._oracle_matrix(bars, pos, 'M5', fit, names, rsi,
                                   np.where(atr > 0, atr, np.nan))
    assert actual.shape == (3, 63)
    assert actual.dtype == np.float32
    assert not np.isinf(actual).any()
    assert np.isfinite(actual[:, names.index('vol_z_20')]).all()
    assert np.isfinite(actual[:, names.index('_v1_pk_sigma20')]).all()
    assert np.isfinite(actual[:, names.index('mid_ema200_distance_change_1_atr')]).all()


def _fake_inspection(out: Path, plan_path: Path) -> tuple:
    manifest, qualified = _manifest_and_index()
    return ({'source_owners': {}}, {'path': str(plan_path), 'sha256': _sha(plan_path)},
            'synthetic-source-head', {'native_parts': []}, manifest, {}, qualified)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_measure(mismatches: int = 0) -> dict:
    return {'timeframes': {tf: {'comparison': {'compared_cells': 63,
                                              'mismatched_cells': mismatches if tf == 'M5' else 0,
                                              'unexercised_fields': []}}
                           for tf in worker.TIMEFRAMES},
            'model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False}


@pytest.mark.parametrize('mismatches', [0, 1])
def test_preflight_is_read_only_and_success_publishes_claim_result_terminal(monkeypatch, tmp_path, mismatches) -> None:
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text('{}')
    out = tmp_path / 'output'
    monkeypatch.setattr(worker, '_inspect_plan', lambda *a: _fake_inspection(out, plan_path))
    status = worker.preflight_read_only(plan_path, _sha(plan_path), out)
    assert status['status'] == 'READY_NO_CLAIM_NO_PAYLOAD'
    assert status['claim_created'] is False and status['npz_or_raw_payload_read'] is False
    assert not out.exists()
    policy_path = tmp_path / 'policy.json'
    policy_path.write_text('{}')
    monkeypatch.setattr(worker, 'POLICY', policy_path)
    monkeypatch.setattr(worker, '_cap_gate', lambda: {'synthetic': True})
    monkeypatch.setattr(worker, '_policy_gate', lambda *a: None)
    monkeypatch.setattr(worker, '_measure', lambda *a: _fake_measure(mismatches))
    monkeypatch.setattr(worker, '_clean_head', lambda: 'synthetic-source-head')
    result = worker.run(plan_path, _sha(plan_path), out)
    assert {p.name for p in out.iterdir()} == {'CLAIM.json', 'RESULT.json', 'TERMINAL.json'}
    result_payload = json.loads((out / 'RESULT.json').read_text())
    assert result_payload['total_compared_cells'] == 315
    assert result_payload['total_mismatched_cells'] == mismatches
    assert result_payload['decision'].startswith('FAIL_' if mismatches else 'PASS_')
    assert json.loads((out / 'TERMINAL.json').read_text())['status'] == 'COMPLETE'
    assert result['result']['sha256'] == _sha(out / 'RESULT.json')


@pytest.mark.parametrize('failure', [ValueError('synthetic data failure'),
                                     TimeoutError('synthetic wall limit')])
def test_policy_rejection_before_claim_and_failure_after_claim_are_terminal(monkeypatch, tmp_path, failure) -> None:
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text('{}')
    policy_path = tmp_path / 'policy.json'
    policy_path.write_text('{}')
    monkeypatch.setattr(worker, 'POLICY', policy_path)
    monkeypatch.setattr(worker, '_inspect_plan', lambda *a: _fake_inspection(tmp_path / 'unused', plan_path))
    monkeypatch.setattr(worker, '_cap_gate', lambda: {'synthetic': True})
    before = tmp_path / 'before'
    with pytest.raises(RuntimeError, match='POLICY_GATE'):
        worker.run(plan_path, _sha(plan_path), before)
    assert not before.exists()
    monkeypatch.setattr(worker, '_policy_gate', lambda *a: None)
    monkeypatch.setattr(worker, '_measure', lambda *a: (_ for _ in ()).throw(failure))
    after = tmp_path / 'after'
    with pytest.raises(type(failure), match=str(failure)):
        worker.run(plan_path, _sha(plan_path), after)
    assert {p.name for p in after.iterdir()} == {'CLAIM.json', 'FAILED.json', 'TERMINAL.json'}
    terminal = json.loads((after / 'TERMINAL.json').read_text())
    failed = json.loads((after / 'FAILED.json').read_text())
    assert failed['error_type'] == type(failure).__name__
    assert terminal['status'] == 'FAILED' and terminal['no_relaunch_under_this_plan'] is True


def test_exact_policy_admission_and_capability_denials(tmp_path) -> None:
    out = tmp_path / 'one_use_output'
    ref = {'path': str(tmp_path / 'plan.json'), 'sha256': '0' * 64}
    gate = {'status': 'ADMITTED', 'decision': 'ADMITTED',
            'plan': ref, 'execution_authorized': True,
            'data_reads_authorized': True,
            'claimed_invocations': 0, 'completed_invocations': 0,
            'max_invocations': 1, 'relaunch_allowed': False,
            'memory_max': '4G', 'swap_max': '512M', 'wall_seconds': 3600,
            'test_data_used': False, 'checkpoint_reads_authorized': False,
            'broker_access_allowed': False, 'model_forwards_authorized': 0,
            'optimizer_steps_authorized': 0, 'training_authorized': False,
            'test_authorized': False, 'output_root': str(out)}
    policy = {'training_enabled': False, 'native_training_enabled': False,
              'native_launch_authorized': False, 'full_val_allowed': False,
              worker.STAGE: gate}
    worker._policy_gate(policy, ref, out)
    for key, unsafe in (
        ('data_reads_authorized', False), ('claimed_invocations', 1),
        ('relaunch_allowed', True), ('checkpoint_reads_authorized', True),
        ('broker_access_allowed', True), ('model_forwards_authorized', 1),
        ('optimizer_steps_authorized', 1), ('training_authorized', True),
        ('test_authorized', True), ('test_data_used', True),
        ('memory_max', '8G'), ('wall_seconds', 7200),
    ):
        altered = dict(policy)
        altered[worker.STAGE] = {**gate, key: unsafe}
        with pytest.raises(RuntimeError, match='POLICY_GATE'):
            worker._policy_gate(altered, ref, out)
    altered = {**policy, 'training_enabled': True}
    with pytest.raises(RuntimeError, match='POLICY_GATE'):
        worker._policy_gate(altered, ref, out)


def test_internal_alarm_reserves_outer_timeout_terminal_margin(monkeypatch) -> None:
    assert worker.EXECUTION['timeout_seconds'] == worker.LIMITS['wall_seconds'] == 3600
    assert worker.EXECUTION['kill_after_seconds'] == 10
    assert worker.TERMINAL_RESERVE_SECONDS == 30
    monkeypatch.setattr(worker.time, 'monotonic', lambda: 100.0)
    assert worker._remaining_alarm_seconds(100.0) == 3570
    monkeypatch.setattr(worker.time, 'monotonic', lambda: 3650.0)
    assert worker._remaining_alarm_seconds(100.0) == 20
    monkeypatch.setattr(worker.time, 'monotonic', lambda: 3670.0)
    with pytest.raises(RuntimeError, match='WALL_BUDGET_PRECLAIM'):
        worker._remaining_alarm_seconds(100.0)
