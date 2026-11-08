"""Focused input-origin/comparison/reporting checks; no market data or model."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gx1.audit import diagnose_market_state_local_trendline_formula_v1 as owner


def test_all_intervals_and_missing_masks_survive_a_first_mismatch():
    rows = np.array([2, 3, 10, 11], dtype=np.int64)
    times = np.array([10, 20, 50, 60], dtype=np.int64)
    oracle = np.zeros((4, 31), dtype=np.float64)
    cache = np.zeros((4, 31), dtype=np.float32)
    oracle[:3, 0] = [1, 2, 3]
    oracle[3, 1] = np.nan
    result = owner.compare_all(oracle, cache, rows, times)
    assert result['all_intervals_compared'] == 2
    assert result['mismatched_cells'] == 4
    assert result['mask_mismatched_cells'] == 1
    assert result['per_field'][0]['mismatched_rows'] == 3
    assert result['per_field'][1]['mask_mismatched_rows'] == 1
    assert {item['cache_row'] for item in result['mismatch_examples']} == {2, 3, 10, 11}
    assert result['discrete_tolerance'] == 0
    assert result['float_abs_tolerance'] == result['float_rel_tolerance'] == 2e-6


def test_actual_source_clock_and_price_parity_drive_the_declared_condition():
    index = pd.date_range('2010-06-13 22:00', periods=8, freq='5min', tz='UTC')
    native = pd.DataFrame({name: np.arange(8, dtype=np.float64) + 100 for name in ('open', 'high', 'low', 'close')}, index=index)
    actual = native.iloc[2:].copy()
    same = owner.price_parity(native, actual)
    assert same['all_OHLC_bits_and_clocks_exact'] is True
    assert same['native_rows_from_actual_start'] == same['actual_rows'] == 6
    actual.iloc[2, actual.columns.get_loc('high')] += 0.001
    different = owner.price_parity(native, actual)
    assert different['all_OHLC_bits_and_clocks_exact'] is False
    assert different['OHLC']['high']['mismatched_rows'] == 1
    missing_clock = owner.price_parity(native, actual.drop(actual.index[3]))
    assert missing_clock['clocks_exact'] is False
    assert missing_clock['native_only_clock_rows'] == 1


def test_real_local96_oracle_and_producer_use_identical_source_origin_and_atr():
    from gx1.features.trendline_registry_v1 import compute_trendline_registry_features_v1
    t = np.arange(320, dtype=np.float64)
    close = 100 + 0.02*t + 3*np.sin(0.31*t) + 0.2*np.cos(1.47*t)
    frame = pd.DataFrame({'open': close, 'high': close + .5, 'low': close - .5, 'close': close},
                         index=pd.date_range('2010-06-13 22:00', periods=len(t), freq='5min', tz='UTC'))
    positions = np.arange(220, 320, dtype=np.int64)
    times = frame.index.asi8[positions]
    parameters = {'seq_len': 96, 'band_atr': 0.9865694402555613}
    oracle, atr, found, events = owner._oracle(frame, times, parameters)
    assert np.array_equal(found, positions)
    assert np.isnan(atr[:13]).all() and np.isfinite(atr[13:]).all()
    production_input = frame[['high', 'low', 'close']].copy()
    production_input['atr'] = atr
    production, state = compute_trendline_registry_features_v1(production_input, timeframe='M5', **parameters)
    report = owner.compare_all(oracle, production.iloc[positions].to_numpy(), positions, times)
    assert report['mismatched_cells'] == 0
    assert state.bar_count == len(frame)
    assert events['confirmed_pivots'] > 0 and events['promotions'] > 0


@pytest.mark.parametrize('fail_after_measurement', [False, True])
def test_run_preserves_measured_mismatch_or_failure_before_terminal(tmp_path, monkeypatch, fail_after_measurement):
    from gx1.contracts import gx1_capped_execution_v1 as guards
    output = tmp_path / 'run'
    output.mkdir()
    plan_path, policy_path, phase_path = (tmp_path / name for name in ('plan.json', 'policy.json', 'phase.json'))
    for path in (plan_path, policy_path, phase_path):
        path.write_text('{}\n')
    monkeypatch.setattr(owner, 'OUT_ROOT', output)
    monkeypatch.setattr(owner, 'PLAN_PATH', plan_path)
    monkeypatch.setattr(owner, 'POLICY_PATH', policy_path)
    cap = dict(owner.LIMITS)
    monkeypatch.setattr(guards, 'require_capped_cpu_research_producer_execution', lambda: cap)
    monkeypatch.setattr(owner.io, '_clean_head', lambda: 'test-head')
    plan_ref, policy_ref, phase_ref = map(owner._ref, (plan_path, policy_path, phase_path))
    plan = {'scope': dict(owner.SCOPE), 'source_owners': {},
            'inputs': {'input_phase': phase_ref}, 'historical_source_binding': {}}
    provider = SimpleNamespace(validate_metadata=lambda _plan: {})
    def inspect(_ref, *, require_policy):
        assert _ref == plan_ref and require_policy is True
        owner.need(not list(output.iterdir()), 'UNCONSUMED_OUTPUT_ROOT')
        return plan, policy_ref, 'test-head', provider, {}, {}, {}
    monkeypatch.setattr(owner, 'inspect', inspect)
    def measure(_plan, _provider, _provider_plan, _info, _phase, *, publish_measurement):
        measurement_ref = publish_measurement('NATIVE2009_ORACLE.json', {
            'mismatched_cells': 7, 'mismatch_examples': [{'field': 'geomline_above_active_count', 'oracle': 2., 'cache': 1.}]})
        if fail_after_measurement:
            raise RuntimeError('injected later formula failure')
        return {'decision': 'MISMATCH', 'measurement_refs': {'native2009': measurement_ref},
                'full_input_audit_pass': False}
    monkeypatch.setattr(owner, 'measure', measure)
    if fail_after_measurement:
        with pytest.raises(RuntimeError, match='injected later formula failure'):
            owner.run(plan_ref)
        assert not (output / 'RESULT.json').exists()
        failure = json.loads((output / 'FAILED.json').read_text())
        terminal = json.loads((output / 'TERMINAL.json').read_text())
        assert terminal['status'] == 'FAILED_CONSUMED'
        assert terminal['failure'] == owner._ref(output / 'FAILED.json')
        assert failure['completed_measurements']['NATIVE2009_ORACLE.json'] == owner._ref(output / 'NATIVE2009_ORACLE.json')
    else:
        returned = owner.run(plan_ref)
        result = json.loads((output / 'RESULT.json').read_text())
        terminal = json.loads((output / 'TERMINAL.json').read_text())
        assert returned['decision'] == result['decision'] == terminal['decision'] == 'MISMATCH'
        assert result['status'] == terminal['status'] == 'COMPLETE_DIAGNOSTIC'
        assert result['full_input_audit_pass'] is False
        assert result['diagnostic']['measurement_refs']['native2009'] == owner._ref(output / 'NATIVE2009_ORACLE.json')
        assert terminal['result'] == owner._ref(output / 'RESULT.json')
        assert not (output / 'FAILED.json').exists()
    assert terminal['no_relaunch_under_this_plan'] is True
    measurement = json.loads((output / 'NATIVE2009_ORACLE.json').read_text())
    assert measurement['measurement']['mismatched_cells'] == 7
    assert measurement['claim'] == owner._ref(output / 'CLAIM.json')
    with pytest.raises(RuntimeError, match='UNCONSUMED_OUTPUT_ROOT'):
        owner.run(plan_ref)


def test_registry_parameters_reuse_only_the_exact_verified_historical_manifest():
    ref = {'path': '/bound/manifest.json', 'sha256': 'a' * 64}
    enriched = {'path': '/bound/M5_ENRICHED.parquet', 'sha256': 'b' * 64}
    binding = {'artifact_path': ref['path'], 'artifact_sha256': ref['sha256'],
        'lane': 'M5', 'params_schema_version': 'registry-v9',
        'params_module': 'gx1.features.htf_features', 'params_contract_sha256': 'c' * 64}
    constants = {'schema_version': binding['params_schema_version'],
        'provenance': {'module': binding['params_module']}, 'contract_sha256': binding['params_contract_sha256'],
        'entry_m5': {'seq_len': 96, 'trendline_band_atr': 0.9865694402555613},
        'per_tf_seq_lens': {'M5': 16}, 'trendline_band_atr': {'M5': 0.9161632622285967}}
    manifest = {'v29_registry_constants': constants,
        'm5_prebuilt_source': enriched['path'], 'm5_prebuilt_source_sha256': enriched['sha256']}
    assert owner._bound_local_parameters(ref, dict(ref), manifest, binding, enriched) == {
        'seq_len': 96, 'band_atr': 0.9865694402555613}
    with pytest.raises(RuntimeError, match='REGISTRY_MANIFEST_REF'):
        owner._bound_local_parameters(ref, {**ref, 'sha256': 'd' * 64}, manifest, binding, enriched)
    with pytest.raises(RuntimeError, match='HISTORICAL_REGISTRY_BINDING'):
        owner._bound_local_parameters(ref, ref, manifest,
            {**binding, 'params_contract_sha256': 'd' * 64}, enriched)
