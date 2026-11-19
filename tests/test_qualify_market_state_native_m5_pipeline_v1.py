"""Synthetic lifecycle and exact native target-axis qualification boundaries."""
from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest

from gx1.audit import qualify_market_state_native_m5_pipeline_v1 as audit


def lifecycle(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, 'ROOT', tmp_path / 'OUT')
    plan_ref = {'path': str(tmp_path / 'PLAN.json'), 'sha256': '1' * 64}
    policy_ref = {'path': str(tmp_path / 'POLICY.json'), 'sha256': '2' * 64}
    plan = {'execution': dict(audit.EXECUTION), 'target_source_spec': {},
            'inputs': {'target_tape': {}, 'history_verification': {
                'path': '/previous/HISTORY_VERIFICATION.json', 'sha256': '3' * 64}}}
    monkeypatch.setattr(audit, 'require_capped_cpu_research_producer_execution', lambda: {
        'memory_max': 12 * 1024**3, 'memory_high': 12 * 1024**3,
        'swap': 512 * 1024**2, 'numerical_threads': 1,
        'cgroup_path': '/sys/fs/cgroup/synthetic'})
    monkeypatch.setattr(audit, 'require_capped_lock_ancestry', lambda: {'scope': '/synthetic'})
    monkeypatch.setattr(audit, 'inspect', lambda *args, **kwargs: (plan, plan_ref, policy_ref))
    monkeypatch.setattr(audit, 'checked_inputs', lambda plan: None)
    monkeypatch.setattr(audit, 'source_clean', lambda plan: True)
    monkeypatch.setattr(audit, 'ref', lambda path: policy_ref if path == audit.POLICY else plan_ref)
    return plan_ref


def test_success_checks_reused_history_and_parameters_before_pipeline(monkeypatch, tmp_path):
    lifecycle(monkeypatch, tmp_path)
    events = []
    monkeypatch.setattr(audit, 'reused_history_verification',
        lambda inputs: events.append('history') or {'verification': {'array_count': 76}})
    monkeypatch.setattr(audit, 'load_parameters',
        lambda plan: events.append('parameters') or {'registry_constants': {}, 'parameter_fits': 0})
    monkeypatch.setattr(audit, 'qualify_physical_pipeline',
        lambda plan, parameters: events.append('pipeline') or {'status': 'TECHNICAL_INITIAL_BATCH_ONLY'})
    result = audit.run('1' * 64)
    saved = json.loads((audit.ROOT / 'RESULT.json').read_text())
    terminal = json.loads((audit.ROOT / 'TERMINAL.json').read_text())
    assert events == ['history', 'parameters', 'pipeline']
    assert terminal['result'] == result['result']
    assert terminal['status'] == 'COMPLETE' and terminal['exit_code'] == 0
    assert terminal['remaining_invocations'] == 0 and terminal['relaunch_allowed'] is False
    assert saved['report']['status'] == 'TECHNICAL_INITIAL_BATCH_ONLY'
    assert saved['model_forwards'] == 1 and saved['optimizer_steps'] == 0
    assert saved['parameters']['parameter_fits'] == 0
    assert saved['economic_edge_proven'] is False
    assert saved['history_verification_artifact']['path'] == '/previous/HISTORY_VERIFICATION.json'
    assert not (audit.ROOT / 'HISTORY_VERIFICATION.json').exists()
    with pytest.raises(FileExistsError):
        audit.run('1' * 64)


def test_reused_history_failure_consumes_claim_before_parameter_load(monkeypatch, tmp_path):
    lifecycle(monkeypatch, tmp_path)
    def failed(inputs):
        raise RuntimeError('independent-history-mismatch')
    monkeypatch.setattr(audit, 'reused_history_verification', failed)
    monkeypatch.setattr(audit, 'load_parameters',
                        lambda plan: pytest.fail('parameter loading reached'))
    with pytest.raises(RuntimeError, match='independent-history-mismatch'):
        audit.run('1' * 64)
    terminal = json.loads((audit.ROOT / 'TERMINAL.json').read_text())
    assert terminal['status'] == 'FAILED_CONSUMED'
    assert terminal['result'] is None and terminal['remaining_invocations'] == 0
    assert terminal['relaunch_allowed'] is False
    assert not (audit.ROOT / 'RESULT.json').exists()


def test_preclaim_missing_genuine_metadata_creates_no_output(monkeypatch, tmp_path):
    lifecycle(monkeypatch, tmp_path)
    def missing(*args, **kwargs):
        raise ValueError('MACRO_COMPONENT_MISSING')
    monkeypatch.setattr(audit, 'inspect', missing)
    monkeypatch.setattr(audit, 'publish', lambda *a: pytest.fail('claim created'))
    with pytest.raises(ValueError, match='MACRO_COMPONENT_MISSING'):
        audit.run('1' * 64)
    assert not audit.ROOT.exists()


def test_preflight_never_decodes_or_constructs_model(monkeypatch, tmp_path, capsys):
    plan_ref = lifecycle(monkeypatch, tmp_path)
    def unexpected(*args, **kwargs):
        pytest.fail('physical work reached during preflight')
    for name in ('read_feature_source', 'reused_history_verification',
                 'load_parameters', 'qualify_physical_pipeline', 'publish'):
        monkeypatch.setattr(audit, name, unexpected)
    monkeypatch.setattr(sys, 'argv', ['audit', '--plan', str(audit.PLAN),
        '--plan-sha256', plan_ref['sha256'], '--out-dir', str(audit.ROOT), '--preflight'])
    audit.main()
    output = json.loads(capsys.readouterr().out)
    assert output['claim_created'] is False
    assert output['model_forwards'] == output['parquet_values_decoded'] == 0
    assert not audit.ROOT.exists()


def test_full_fit_axis_uses_input_readiness_and_per_head_maturity(tmp_path):
    n = 25
    issue = audit.FIT_START_NS + np.arange(n, dtype=np.int64) * 300_000_000_000
    shape = (n, 5)
    truth = {
        'issue_ns': issue, 'target_row': np.broadcast_to(np.arange(n)[:, None], shape).copy(),
        'validity_ns': np.broadcast_to(issue[:, None] + 300_000_000_000, shape).copy(),
        'target_present': np.ones(shape, bool), 'observed_previous': np.ones(shape, bool),
        'mature': np.ones(shape, bool), 'eligible': np.ones(shape, bool),
        'open_known_at_issue': np.zeros(shape, bool),
        'open_sign': np.ones(shape, np.int8), 'close_sign': np.zeros(shape, np.int8)}
    truth['target_row'] = truth['target_row'].astype(np.int64)
    truth['observed_previous'][3] = False
    truth['validity_ns'][-1] = audit.FIT_END_NS
    truth['validity_ns'][5, -1] = audit.FIT_END_NS
    source = np.arange(100, 100 + n, dtype=np.int64)
    path = tmp_path / 'TARGETS.npz'
    np.savez(path, source_row=source, issue_ns=issue,
             **{'future_' + key: value for key, value in truth.items()})
    ready = np.ones(n, bool)
    ready[2] = False
    provider = SimpleNamespace(source_row=source, issue_ns=issue, input_ready=ready)
    rows, selected, report = audit.target_axis({'inputs': {'target_tape': {'path': str(path)}}}, provider)
    np.testing.assert_equal(rows, np.setdiff1d(source, [102, 103, 124]))
    assert len(rows) == report['fit_issue_rows'] == 22
    assert not selected['mature'][np.flatnonzero(rows == 105)[0], -1]
    assert selected['target_row'][np.flatnonzero(rows == 105)[0], -1] == -1
    assert selected['close_sign'].dtype == np.int8
    assert report['input_not_ready_rows'] == 1
    assert report['no_mature_supervision_rows'] == 2
    assert report['fit_first_issue_ns'] == int(issue[0])
    assert report['fit_last_issue_ns'] == int(issue[23])


def test_build_plan_rejects_partial_history_before_feature_or_macro_work(monkeypatch):
    inputs = {key: {'path': '/synthetic/' + key, 'sha256': '1' * 64}
              for key in audit.INPUT_KEYS}
    inputs['native_source_plan']['path'] = str(audit.data.PLAN)
    source = {'stage': audit.data.STAGE, 'source_spec': {
        'issue_start_ns': audit.data.ISSUE_START_NS, 'cutoff_ns': audit.FIT_END_NS}}
    monkeypatch.setattr(audit, 'fit_plans', lambda inputs: ({}, {}))
    def source_only(binding):
        assert binding == inputs['native_source_plan']
        return source
    monkeypatch.setattr(audit, 'read', source_only)
    monkeypatch.setattr(audit, 'feature_spec',
                        lambda source: pytest.fail('partial-history feature work reached'))
    monkeypatch.setattr(audit, 'macro_metadata',
                        lambda binding: pytest.fail('partial-history macro work reached'))
    with pytest.raises(ValueError, match='FULL_HISTORY_003_IDENTITY'):
        audit.build_plan(inputs=inputs)


def history_fixture(monkeypatch):
    old_root = '/home/andre2/GX1_RUNS/NATIVE_M5_PIPELINE_20261119_001'
    def binding(path):
        return {'path': path, 'sha256': '1' * 64}
    inputs = {
        'history_verification': binding(old_root + '/HISTORY_VERIFICATION.json'),
        'target_tape': binding('/synthetic/TARGETS.npz'),
        'target_result': binding('/synthetic/RESULT.json'),
        'native_source_plan': binding('/synthetic/SOURCE_PLAN.json')}
    plan_ref = binding(str(audit.REPO / 'configs/research/NATIVE_M5_PIPELINE_20261119_001.json'))
    claim_ref = binding(old_root + '/CLAIM.json')
    spec = {'parts': [{}] * 18, 'cutoff_ns': audit.data.CUTOFF_NS}
    contract = {'schema_version': 'gx1_market_state_native_m5_bar_directions_v2'}
    coverage = {'observed_bars': 100, 'missing_previous_bars': 1,
        'cross_year_previous_bars': 17, 'future_target_present': 90,
        'future_mature': 90, 'future_eligible': 89, 'future_unique_mature_bar_rows': 90}
    proof_coverage = {'observed_bars': 100, 'missing_first_previous': 1,
        'cross_year_previous_bars': 17, 'future_present': 90,
        'future_mature': 90, 'future_eligible': 89, 'future_unique_mature_rows': 90}
    keys = ['synthetic_field_' + str(i) for i in range(76)]
    proof = {'schema_version': 'native_m5_independent_all_history_facts_verification_v1',
        'array_count': 76, 'compared_arrays': keys, 'all_arrays_exact_equal_nan': True,
        'raw_and_archive_hash_bookends_unchanged': True, 'raw_parts': 18,
        'retained_source_rows': 100, 'issue_rows': 100,
        'strict_exclusive_close_cutoff_ns': audit.data.CUTOFF_NS,
        'coverage_by_timeframe': {'M5': proof_coverage}}
    target = {'target_contract': contract, 'report': {
        'array_shapes': {key: [100] for key in keys}, 'source_rows': 100,
        'selected_issue_rows': 100, 'coverage_by_timeframe': {'M5': coverage}}}
    saved = {'plan': plan_ref, 'claim': claim_ref,
             'target_tape': dict(inputs['target_tape']), 'verification': proof}
    old_plan = {'stage': 'native_m5_pipeline_20261119_001',
        'inputs': {'target_tape': dict(inputs['target_tape']),
                   'native_source_plan': dict(inputs['native_source_plan'])},
        'target_contract': contract, 'target_source_spec': spec}
    old_claim = {'stage': old_plan['stage'], 'plan': plan_ref}
    records = {inputs['history_verification']['path']: saved,
        inputs['target_result']['path']: target,
        inputs['native_source_plan']['path']: {'source_spec': spec},
        plan_ref['path']: old_plan, claim_ref['path']: old_claim}
    monkeypatch.setattr(audit, 'read', lambda ref: records[ref['path']])
    return inputs, saved


def test_reuse_keeps_original_plan_claim_and_all_76_proof(monkeypatch):
    inputs, saved = history_fixture(monkeypatch)
    actual = audit.reused_history_verification(inputs)
    assert actual is saved
    assert actual['plan']['path'].endswith('NATIVE_M5_PIPELINE_20261119_001.json')
    assert actual['claim']['path'].endswith('_001/CLAIM.json')


@pytest.mark.parametrize('fault', ('tape', 'incomplete', 'coverage'))
def test_reuse_rejects_changed_scope_before_parameter_work(monkeypatch, fault):
    inputs, saved = history_fixture(monkeypatch)
    if fault == 'tape':
        inputs['target_tape'] = {**inputs['target_tape'], 'sha256': '2' * 64}
    elif fault == 'incomplete':
        saved['verification']['array_count'] = 75
    else:
        saved['verification']['coverage_by_timeframe']['M5']['observed_bars'] = 99
    with pytest.raises(ValueError, match='HISTORY_REUSE_(LINEAGE|COMPLETENESS|COVERAGE)'):
        audit.reused_history_verification(inputs)


def parameter_fixture(monkeypatch):
    from gx1.features import volatility_squeeze_state_v1 as squeeze
    from gx1.features import htf_features as htf
    previous = '/home/andre2/GX1_RUNS/NATIVE_M5_PIPELINE_20261119_002'
    binding = lambda path: {'path': path, 'sha256': '1' * 64}
    inputs = {key: binding('/synthetic/' + key) for key in
              ('squeeze_fit_plan', 'registry_fit_plan', 'native_source_plan')}
    inputs.update(prefitted_squeeze_manifest=binding(previous + '/SQUEEZE/manifest.json'),
                  prefitted_registry_constants=binding(previous + '/REGISTRY_CONSTANTS.json'))
    start, end = (audit.pd.Timestamp(ns, tz='UTC').isoformat()
                  for ns in (audit.PARAMETER_START_NS, audit.PARAMETER_END_NS))
    plan = {'native_m5_manifest': binding('/native/MANIFEST.json'),
        'source_clock_contract': squeeze.VOLATILITY_SQUEEZE_DIRECTION_CLOCK_CONTRACT,
        'declared_train_window_start': start, 'declared_train_window_end': end,
        'source_spec': {'parts': ['physical_sha_bound']}}
    monkeypatch.setattr(audit, 'fit_plans', lambda inputs: (plan, plan))
    geometry = {'local': 96, 'M5': 16, 'M15': 64, 'H1': 96, 'H4': 96, 'D1': 252}
    sources = audit.registry_provenance(inputs['registry_fit_plan'], plan)
    params, artifacts, level, trend = {}, {}, {}, {}
    for tf in squeeze.VOLATILITY_SQUEEZE_DIRECTION_CLOCKS:
        source = dict(sources[tf])
        for key in ('source_clock_contract', 'declared_train_window_start',
                    'declared_train_window_end'):
            source.pop(key)
        source.update(clock_contract=plan['source_clock_contract'],
            bar_grid=squeeze.volatility_squeeze_bar_grid(tf),
            fit_plan_artifact=inputs['squeeze_fit_plan']['path'],
            fit_plan_sha256=inputs['squeeze_fit_plan']['sha256'])
        params[tf] = {'source_provenance': source, 'declared_train_window_start': start,
            'declared_train_window_end': end, 'train_ohlcv_rows': 101,
            'train_ohlcv_sha256': '2' * 64}
        artifacts[tf] = {'params_artifact': previous + '/SQUEEZE/' + tf.lower() + '_params.json',
            'params_file_sha256': '3' * 64, 'train_ohlcv_rows': 101,
            'train_ohlcv_sha256': '2' * 64}
        item = {'source_provenance': sources[tf], 'outer_train_rows': 100,
                'frame_sha256': '4' * 64}
        level[tf], trend[tf] = dict(item), dict(item)
    constants = {'declared_train_window_start': start, 'declared_train_window_end': end,
        'declared_inner_fit_window_end': audit.INNER_END, 'entry_m5': {'seq_len': 96},
        'per_tf_seq_lens': {k: v for k, v in geometry.items() if k != 'local'},
        'contract_sha256': '5' * 64, 'provenance': {
            'level_recurrence_threshold': level, 'trendline_band': trend,
            'entry_m5_trendline_band': dict(trend['M5']), 'n_train_m5_rows': 100}}
    old_plan_ref = binding(str(audit.REPO / 'configs/research/NATIVE_M5_PIPELINE_20261119_002.json'))
    old_claim_ref = binding(previous + '/CLAIM.json')
    old_plan = {'stage': 'native_m5_pipeline_20261119_002', 'parameter_source_spec': plan['source_spec'],
                'inputs': inputs, 'geometry': geometry}
    records = {inputs['prefitted_squeeze_manifest']['path']: {'artifacts': artifacts},
        inputs['prefitted_registry_constants']['path']: constants,
        old_plan_ref['path']: old_plan,
        old_claim_ref['path']: {'stage': old_plan['stage'], 'plan': old_plan_ref}}
    monkeypatch.setattr(audit, 'read', lambda ref: records[ref['path']])
    monkeypatch.setattr(audit, 'ref', lambda path: old_claim_ref)
    carrier = SimpleNamespace(require_params=lambda tf: params[tf],
                              binding=lambda: {'native_clocks': list(params)})
    monkeypatch.setattr(squeeze, 'load_volatility_squeeze_artifact_manifest',
                        lambda *args, **kwargs: carrier)
    monkeypatch.setattr(htf, 'require_native_m5_registry_constants', lambda value: value)
    monkeypatch.setattr(audit.pq, 'ParquetFile',
                        lambda *args, **kwargs: pytest.fail('calibration prices decoded'))
    return inputs, geometry, params, constants, old_plan


def test_completed_native_parameters_reuse_preserves_owner_frame_domains(monkeypatch):
    inputs, geometry, _, constants, _ = parameter_fixture(monkeypatch)
    metadata, actual = audit.parameter_artifact_metadata(inputs, geometry)
    assert actual is constants and metadata['parameter_fits'] == 0
    assert metadata['recorded_fit_frames']['M15']['squeeze_rows'] == 101
    assert metadata['recorded_fit_frames']['M15']['registry_rows'] == 100
    assert len(metadata['parameter_dependencies']) == 7
    published = []
    monkeypatch.setattr(audit, 'publish', lambda name, value: published.append(name) or {'path': name})
    loaded = audit.load_parameters({'inputs': inputs, 'geometry': geometry, 'parameter_reuse': metadata})
    assert published == ['PARAMETER_REUSE.json']
    assert loaded['squeeze_manifest'] is inputs['prefitted_squeeze_manifest']
    assert loaded['registry_artifact'] is inputs['prefitted_registry_constants']
    assert loaded['parameter_fits'] == 0


@pytest.mark.parametrize('fault', ('path', 'source', 'geometry', 'frame', 'origin'))
def test_parameter_reuse_rejects_wrong_native_lineage_before_raw_read(monkeypatch, fault):
    inputs, geometry, params, constants, old_plan = parameter_fixture(monkeypatch)
    if fault == 'path':
        inputs['prefitted_squeeze_manifest']['path'] = '/other/manifest.json'
    elif fault == 'source':
        params['H1']['source_provenance']['source_lane'] = 'M1'
    elif fault == 'geometry':
        constants['per_tf_seq_lens']['D1'] = 251
    elif fault == 'frame':
        constants['provenance']['trendline_band']['H1']['frame_sha256'] = '6' * 64
    else:
        old_plan['parameter_source_spec'] = {'parts': ['different']}
    with pytest.raises(ValueError, match='PARAMETER_REUSE_'):
        audit.parameter_artifact_metadata(inputs, geometry)


def test_bounded_feature_snapshot_saves_all_arrays_with_verified_reload(monkeypatch, tmp_path):
    import pandas as pd
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    starts = np.arange(10, dtype=np.int64) * 300_000_000_000
    surfaces = {'schema_version': 'synthetic_native_surfaces',
        'bar_start_ns': starts, 'retained_source_rows': np.array([7, 8, 9], np.int64),
        'retained_issue_rows': np.array([8, 9], np.int64),
        'signal': np.array([[1., np.nan], [2., 3.], [4., 5.]], np.float32),
        'ctx_cont': np.ones((3, 1), np.float32), 'ctx_cat': np.zeros((3, 1), np.int64),
        'observed': np.array([[True, False], [True, True], [True, True]]),
        'axes': {'full_rows': 10, 'retained_rows': 3}, 'field_lineage': {'field': 'native'},
        'multi_tf_sources': {}}
    for tf in ('M5', 'M15', 'H1', 'H4', 'D1'):
        matrix = np.ones((2, 4), np.float32)
        times = np.array([100, 200], np.int64)
        frame = pd.DataFrame(matrix, index=pd.to_datetime(times, utc=True), copy=False)
        frame.attrs.update(ts_int64=times, feats_np=matrix,
            owner_row_indices=np.array([70, 71], np.int64),
            model_native_mtf_scalars_np_v4=np.ones((2, 1), np.float32),
            causal_warmup_rows=0, model_native_mtf_scalar_fields_v4=('scalar',),
            htf_feature_contract='native', model_native_mtf_scalar_warmup_rows_v4=0,
            model_native_mtf_scalar_contract_v4='native_scalar')
        surfaces['multi_tf_sources'][tf] = frame
    snapshot_ref = audit.publish_feature_snapshot(SimpleNamespace(surfaces=surfaces))
    metadata = audit.read(snapshot_ref)
    assert metadata['reload_all_arrays_dtype_shape_equal_nan_and_bytes_sha256'] is True
    assert metadata['surfaces_metadata']['axes'] == surfaces['axes']
    assert len(metadata['arrays']) == 27
    with np.load(metadata['archive']['path'], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['retained_source_rows'], [7, 8, 9])
        np.testing.assert_array_equal(saved['tf_D1_owner_row_indices'], [70, 71])
        np.testing.assert_array_equal(saved['signal'], surfaces['signal'])
    assert metadata['timeframes']['D1']['scalar_attributes']['model_native_mtf_scalar_fields_v4'] == ['scalar']
