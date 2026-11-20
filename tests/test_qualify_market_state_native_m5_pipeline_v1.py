"""Synthetic lifecycle and exact native target-axis qualification boundaries."""
from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest

from gx1.audit import qualify_market_state_native_m5_pipeline_v1 as audit


def lifecycle(monkeypatch, tmp_path):
    from gx1.contracts.native_m5_direction_io_firewall_v1 import contract
    monkeypatch.setattr(audit, '_IO_FIREWALL', {'contract': contract(),
        'denied_attempts': 0, 'denied_operations': {}})
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


def candidate_fixture(monkeypatch):
    binding = lambda path: {'path': str(path), 'sha256': '1' * 64}
    inputs = {key: binding('/synthetic/' + key) for key in audit.BASE_INPUT_KEYS}
    for key, name in {'candidate_snapshot': 'FEATURE_INPUTS.json',
        'candidate_archive': 'FEATURE_INPUTS.npz', 'candidate_support': 'FEATURE_SUPPORT.json',
        'candidate_terminal': 'TERMINAL.json'}.items():
        inputs[key] = binding(audit.CANDIDATE_ROOT / name)
    from gx1.contracts.entry_model_native_input_normalization_v1 import native_m5_input_domain_contract
    before = {'schema_version': 'gx1_native_m5_feature_provider_v1', 'execution_profile': {'name': 'native_TA', 'contract_sha256': '2' * 64},
        'field_names': {'signal': ['genuine_field']}, 'contract_sha256': '3' * 64}
    after = {**before, 'schema_version': 'gx1_native_m5_feature_provider_v2',
             'input_semantic_domains': native_m5_input_domain_contract(), 'execution_profile': {'name': 'native_TA', 'contract_sha256': '4' * 64},
             'contract_sha256': '5' * 64}
    plan_ref = binding(audit.REPO / 'configs/research/NATIVE_M5_PIPELINE_20261119_003.json')
    claim_ref = binding(audit.CANDIDATE_ROOT / 'CLAIM.json')
    old_plan = {'stage': 'native_m5_pipeline_20261119_003',
        'inputs': {key: inputs[key] for key in audit.BASE_INPUT_KEYS},
        'feature_contract': before, 'feature_source_spec': {}}
    saved = {'schema_version': 'gx1_native_m5_bounded_feature_snapshot_v1',
        'archive': inputs['candidate_archive'], 'arrays': {str(i): {} for i in range(27)},
        'timeframes': {tf: {} for tf in ('M5','M15','H1','H4','D1')},
        'surfaces_metadata': {'provider_contract': before},
        'reload_all_arrays_dtype_shape_equal_nan_and_bytes_sha256': True,
        'model_forwards': 0, 'optimizer_steps': 0}
    terminal = {'plan': plan_ref, 'claim': claim_ref, 'stage': old_plan['stage'],
        'status': 'FAILED_CONSUMED', 'exit_code': 1, 'result': None,
        'relaunch_allowed': False, 'test_data_used': False,
        'error': 'RuntimeError:[ENTRY_INPUT_NORMALIZATION_UNSCALEABLE] field=ctx_cont.d1_ema_stack_aligned_v2'}
    records = {inputs['candidate_snapshot']['path']: saved,
        inputs['candidate_support']['path']: {'feature_source_spec': {}, 'model_forwards': 0, 'optimizer_steps': 0},
        inputs['candidate_terminal']['path']: terminal,
        plan_ref['path']: old_plan, claim_ref['path']: {'stage': old_plan['stage'], 'plan': plan_ref}}
    monkeypatch.setattr(audit, 'read', lambda ref: records[ref['path']])
    monkeypatch.setattr(audit, 'numeric_owner_identities', lambda plan: {name: '6' * 64 for name in audit.NUMERIC_OWNER_PATHS})
    monkeypatch.setattr(audit, 'candidate_clock_proof', lambda *args: {'synthetic_clock_proof': True})
    return inputs, before, after, saved, terminal


def test_candidate_requalification_declares_changed_catalog_and_fresh_domain_only(monkeypatch):
    inputs, before, after, saved, _ = candidate_fixture(monkeypatch)
    proof = audit.candidate_metadata(inputs, after)
    assert proof['original_execution_profile'] == before['execution_profile']
    assert proof['admitted_execution_profile'] == after['execution_profile']
    assert proof['original_full_process_M1_free'] is False
    assert saved['surfaces_metadata']['provider_contract'] is before


@pytest.mark.parametrize('fault', ('fields', 'terminal', 'archive'))
def test_candidate_metadata_rejects_changed_numeric_contract_or_origin(monkeypatch, fault):
    inputs, before, after, saved, terminal = candidate_fixture(monkeypatch)
    if fault == 'fields':
        after['field_names'] = {'signal': ['different_formula']}
    elif fault == 'terminal':
        terminal['status'] = 'COMPLETE'
    else:
        saved['archive'] = {**inputs['candidate_archive'], 'sha256': '7' * 64}
    with pytest.raises(ValueError, match='CANDIDATE_'):
        audit.candidate_metadata(inputs, after)


def test_numeric_owner_identity_ignores_docs_import_wiring_but_preserves_formulas():
    original = 'import old_catalog\nLIMIT=3\ndef formula(x):\n    "doc"\n    return x+LIMIT\n'
    moved = 'import pure_catalog\nLIMIT=3\ndef formula(x):\n    "new docs"\n    return x+LIMIT\n'
    assert audit.numeric_ast_identity(original) == audit.numeric_ast_identity(moved)
    assert audit.numeric_ast_identity(original) != audit.numeric_ast_identity(moved.replace('LIMIT=3','LIMIT=4'))
    assert audit.numeric_ast_identity(original) != audit.numeric_ast_identity(moved.replace('x+LIMIT','x-LIMIT'))


def cached_physical_fixture(monkeypatch, tmp_path):
    from tests.test_market_state_native_m5_feature_inputs_v1 import _fixture, _build
    from gx1.contracts import market_state_native_m5_feature_inputs_v1 as owner
    fixture = _fixture(tmp_path, monkeypatch)
    surfaces = _build(fixture)
    provider = owner.NativeM5FeatureProvider(surfaces)
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    snapshot = audit.publish_feature_snapshot(provider)
    saved = audit.read(snapshot)
    target_path = tmp_path / 'TARGETS.npz'
    target = {'issue_ns': provider.issue_ns, 'source_row': provider.source_row}
    for tf, frame in fixture[-1].items():
        target['bars_' + tf + '_bar_start_ns'] = frame.index.asi8
        target['bars_' + tf + '_available_ns'] = frame.index.asi8 + int(owner.htf.MULTI_TF_SHIFT[tf].value)
    np.savez(target_path, **target)
    plan = {'inputs': {'candidate_snapshot': snapshot, 'candidate_archive': saved['archive'],
                      'target_tape': {'path': str(target_path)}},
        'candidate_requalification': {}, 'feature_contract': owner.native_m5_feature_provider_contract(),
        'feature_source_spec': surfaces['source_binding']['source_spec'],
        'parameter_reuse': {'registry_contract_sha256': surfaces['registry_contract_sha256'],
                            'squeeze_binding': surfaces['squeeze_binding']}}
    # The existing wiring fixture deliberately fabricates HTF prehistory.
    # Its declared owner axes are injected here; real native geometry is checked
    # independently below, including initial partial bins and source gaps.
    monkeypatch.setattr(owner.htf, 'build_multi_tf_v4_closed_timestamp_indices',
                        lambda index: {tf: frame.index for tf, frame in fixture[-1].items()})
    clocks = {tf: {key: frame.attrs[key] for key in ('owner_row_indices', 'ts_int64')}
              for tf, frame in surfaces['multi_tf_sources'].items()}
    clock_proof = audit.validate_candidate_clock_domains(surfaces['bar_start_ns'], clocks, target)
    plan['candidate_requalification'] = {'cached_clock_admission': {
        **clock_proof, 'cached_clock_arrays_decoded': 11,
        'cached_feature_value_arrays_decoded': 0, 'raw_price_values_decoded': 0}}
    monkeypatch.setattr(audit, 'candidate_metadata', lambda *args: plan['candidate_requalification'])
    return plan, saved, provider


def test_cached_physical_requalification_preserves_all27_arrays_axes_and_aliases(monkeypatch, tmp_path):
    plan, _, original = cached_physical_fixture(monkeypatch, tmp_path)
    provider, proof = audit.load_candidate_provider(plan)
    assert proof['array_count'] == 27 and proof['raw_numeric_values_recomputed'] is False
    np.testing.assert_array_equal(provider.issue_ns, original.issue_ns)
    for key in ('signal', 'ctx_cont', 'ctx_cat', 'observed'):
        np.testing.assert_array_equal(provider.surfaces[key], original.surfaces[key])
        assert not provider.surfaces[key].flags.writeable


@pytest.mark.parametrize('fault', ('bytes', 'global_tf_id', 'field_order', 'alias', 'lineage'))
def test_cached_physical_requalification_rejects_mismatches(monkeypatch, tmp_path, fault):
    plan, saved, _ = cached_physical_fixture(monkeypatch, tmp_path)
    archive_path = plan['inputs']['candidate_archive']['path']
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if fault in ('bytes', 'alias'):
        key = 'ctx_cont'
        arrays[key][0, 0] += 1.
        if fault == 'alias':
            saved['arrays'][key]['sha256'] = audit.hashlib.sha256(arrays[key].tobytes()).hexdigest()
    elif fault == 'global_tf_id':
        key = 'tf_H1_owner_row_indices'
        arrays[key] += 1
        saved['arrays'][key]['sha256'] = audit.hashlib.sha256(arrays[key].tobytes()).hexdigest()
    elif fault == 'field_order':
        saved['timeframes']['M5']['columns'][0:2] = saved['timeframes']['M5']['columns'][1::-1]
    else:
        key = next(iter(saved['surfaces_metadata']['field_lineage']))
        saved['surfaces_metadata']['field_lineage'][key] = 'wrong_owner'
    np.savez(archive_path, **arrays)
    plan['inputs']['candidate_archive'] = audit.ref(archive_path)
    monkeypatch.setattr(audit, 'read', lambda ref: saved)
    with pytest.raises((ValueError, RuntimeError), match='(CANDIDATE_|CONTEXT_ALIAS_VALUES)'):
        audit.load_candidate_provider(plan)


def real_clock_geometry(first='2009-06-01T00:00:00Z'):
    import pandas as pd
    from gx1.features import htf_features as htf
    index = pd.date_range(first, '2009-06-17T23:50:00Z', freq='5min')
    index = index[index.dayofweek < 5]  # observed source gaps stay in both domains
    starts = index.asi8
    owner = htf.build_multi_tf_v4_closed_timestamp_indices(index)
    truth = {'issue_ns': starts + audit.data.M5_NS, 'source_row': np.arange(len(starts), dtype=np.int64)}
    clocks = {}
    for tf, axis in owner.items():
        observed = htf.multi_tf_bar_label(index, tf).drop_duplicates()
        duration = int(htf.MULTI_TF_SHIFT[tf].value)
        observed = observed[observed.asi8 + duration <= starts[-1] + audit.data.M5_NS]
        truth['bars_' + tf + '_bar_start_ns'] = observed.asi8
        truth['bars_' + tf + '_available_ns'] = observed.asi8 + duration
        ids = np.arange(max(0, len(axis)-5), len(axis), dtype=np.int64)
        clocks[tf] = {'owner_row_indices': ids, 'ts_int64': axis.asi8[ids]}
    return starts, clocks, truth


@pytest.mark.parametrize('tf', ('H4', 'D1'))
def test_real_first_partial_facts_and_feature_ids_have_separate_exact_domains(tf):
    starts, clocks, truth = real_clock_geometry()
    original_ids = clocks[tf]['owner_row_indices'].copy()
    proof = audit.validate_candidate_clock_domains(starts, clocks, truth)
    domain = proof['per_timeframe'][tf]
    assert domain['target_rows_before_feature_axis_start'] == 1
    assert domain['target_minus_feature_owner_id_counts'] == {'1': len(original_ids)}
    assert domain['exact_bijective_retained_time_and_availability_join'] is True
    np.testing.assert_array_equal(clocks[tf]['owner_row_indices'], original_ids)


def test_asymmetric_native_start_changes_affected_domains_without_date_or_offset_special_case():
    starts, clocks, truth = real_clock_geometry('2009-06-01T09:35:00Z')
    proof = audit.validate_candidate_clock_domains(starts, clocks, truth)
    assert proof['per_timeframe']['M5']['target_rows_before_feature_axis_start'] == 0
    for tf in ('M15', 'H1', 'H4', 'D1'):
        assert proof['per_timeframe'][tf]['target_rows_before_feature_axis_start'] == 1
        assert proof['per_timeframe'][tf]['target_minus_feature_owner_id_counts'] == {'1': 5}


@pytest.mark.parametrize('fault', ('owner_id', 'target_time', 'availability'))
def test_separate_clock_domain_proof_rejects_wrong_ids_or_nonexact_target_join(fault):
    starts, clocks, truth = real_clock_geometry()
    if fault == 'owner_id':
        clocks['D1']['owner_row_indices'] = clocks['D1']['owner_row_indices'] + 1
    else:
        key = 'bars_D1_' + ('bar_start_ns' if fault == 'target_time' else 'available_ns')
        truth[key] = truth[key].copy()
        truth[key][-1] += audit.data.M5_NS
    with pytest.raises(ValueError, match='CANDIDATE_(FEATURE_OWNER_AXIS|TARGET_TIME_JOIN|TARGET_AVAILABILITY_JOIN)'):
        audit.validate_candidate_clock_domains(starts, clocks, truth)


def test_preclaim_clock_proof_never_decodes_cached_feature_values(monkeypatch, tmp_path):
    starts, clocks, truth = real_clock_geometry()
    arrays = {'bar_start_ns': starts}
    for tf, clock in clocks.items():
        arrays.update({'tf_' + tf + '_' + key: value for key, value in clock.items()})
    candidate = tmp_path / 'FEATURES.npz'
    target = tmp_path / 'TARGETS.npz'
    # Accessing either object array with allow_pickle=False would fail.
    np.savez(candidate, **arrays, signal=np.array([object()], dtype=object))
    np.savez(target, **truth, bars_D1_ohlc=np.array([object()], dtype=object))
    saved = {'arrays': {key: {'dtype': value.dtype.str, 'shape': list(value.shape),
        'sha256': audit.hashlib.sha256(value.tobytes()).hexdigest()} for key, value in arrays.items()}}
    proof = audit.candidate_clock_proof({'candidate_archive': audit.ref(candidate),
                                       'target_tape': audit.ref(target)}, saved)
    assert proof['cached_clock_arrays_decoded'] == 11
    assert proof['cached_feature_value_arrays_decoded'] == proof['raw_price_values_decoded'] == 0


def diagnosis_fixture(monkeypatch):
    import copy
    prior='/home/andre2/GX1_RUNS/NATIVE_M5_PIPELINE_20261119_005'
    binding=lambda path:{'path':path,'sha256':'a'*64}
    inputs={'normalization_domain_observation':dict(audit.NORMALIZATION_DOMAIN_OBSERVATION),
        'candidate_archive':binding('/candidate/FEATURE_INPUTS.npz'),'target_tape':binding('/native/TARGETS.npz')}
    terminal_ref=binding(prior+'/TERMINAL.json')
    support_ref=binding(prior+'/FEATURE_SUPPORT.json')
    plan_ref=binding(str(audit.REPO/'configs/research/NATIVE_M5_PIPELINE_20261119_005.json'))
    claim_ref=binding(prior+'/CLAIM.json')
    population={'fit_issue_rows':29963,'fit_source_rows_sha256':'b'*64,'fit_issue_ns_sha256':'c'*64}
    shapes={'signal':(30058,318),'ctx_cont':(29963,132),'mtf_m5':(29978,256),
        'mtf_m15':(10051,256),'mtf_h1':(2594,256),'mtf_h4':(750,256),'mtf_d1':(361,256)}
    report={'archive':dict(inputs['candidate_archive']),'fit_population':copy.deepcopy(population),
        'plan':prior,'forbidden_io_attempts':0,'model_forwards':0,
        'normalization_fitted':False,'raw_data_rebuilt':False,
        'surfaces':{name:{'rows':n,'width':w,'unsupported_fields':int(name=='mtf_m5')}
                    for name,(n,w)in shapes.items()},
        'unsupported':[{'surface':'mtf_m5','field':'geomchan_apex_proximity','min':0.,
            'median':0.,'max':.59,'observation_gated':False,'positive_deviations':1,
            'observed_rows':29978,'physical_rows':29978,'unique_counts':[[0.,29977],[.59,1]]}]}
    observation={'schema_version':'gx1_external_native_normalization_domain_observation_v1',
        'role':'read_only_domain_diagnosis_not_native_job_terminal_or_normalization',
        'original_terminal':terminal_ref,'report':report}
    terminal={'stage':'native_m5_pipeline_20261119_005','status':'FAILED_CONSUMED',
        'result':None,'error':'RuntimeError:surface=mtf_m5 field=geomchan_apex_proximity',
        'plan':plan_ref,'claim':claim_ref}
    support={'fit_population':population,'consumed_support':{'surfaces':{
        name:{'physical_rows':n,'field_names':list(range(w))}for name,(n,w)in shapes.items()}}}
    records={inputs['normalization_domain_observation']['path']:observation,
        terminal_ref['path']:terminal,support_ref['path']:support,
        plan_ref['path']:{'stage':terminal['stage'],'inputs':inputs},
        claim_ref['path']:{'plan':plan_ref}}
    monkeypatch.setattr(audit,'read',lambda ref:records[ref['path']])
    monkeypatch.setattr(audit,'ref',lambda path:support_ref)
    return inputs,report


def test_006_requires_all7_diagnosis_and_exact_continuous_bounded_remedy(monkeypatch):
    inputs,report=diagnosis_fixture(monkeypatch)
    proof=audit.normalization_diagnosis(inputs)
    assert proof['all_seven_consumption_domains_diagnosed']is True
    assert proof['fit_population']==report['fit_population']
    assert proof['only_unsupported_case']=={'surface':'mtf_m5','field':'geomchan_apex_proximity'}
    assert proof['normalization_fitted_by_diagnosis']is False


@pytest.mark.parametrize('fault',('archive','fit_axis','extra_case','outside_bound','refitted'))
def test_006_diagnosis_gate_rejects_changed_archive_axis_or_uncovered_case(monkeypatch,fault):
    inputs,report=diagnosis_fixture(monkeypatch)
    if fault=='archive':report['archive']['sha256']='f'*64
    elif fault=='fit_axis':report['fit_population']['fit_issue_rows']-=1
    elif fault=='extra_case':report['surfaces']['mtf_h1']['unsupported_fields']=1
    elif fault=='outside_bound':report['unsupported'][0]['max']=2.
    else:report['normalization_fitted']=True
    with pytest.raises(ValueError,match='NORMALIZATION_DIAGNOSIS_'):
        audit.normalization_diagnosis(inputs)


def alias_numeric_owner_fixture(monkeypatch, tmp_path):
    from gx1.audit import build_native_m5_ta_feature_cache_v1 as bridge
    relative='gx1/contracts/market_state_native_m5_feature_inputs_v1.py'
    current=(audit.REPO/relative).read_text()
    original=current.replace(bridge._ALIAS_HELPER_SOURCE+'\n\n','').replace(
        'values[local]='+bridge._ALIAS_NEW_EXPRESSION,'values[local]='+bridge._ALIAS_OLD_EXPRESSION)
    assert original!=current and '_native_m5_alias_vector'not in original
    other='gx1/features/unit_numeric_owner.py'
    originals={relative:original.encode(),other:b'LIMIT=3\ndef formula(x): return x+LIMIT\n'}
    for name,value in {relative:current,other:originals[other].decode()}.items():
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
    monkeypatch.setattr(audit,'REPO',tmp_path)
    monkeypatch.setattr(audit,'NUMERIC_OWNER_PATHS',(relative,other))
    monkeypatch.setattr(audit.subprocess,'check_output',lambda args,**kwargs:originals[args[-1].split(':',1)[1]])
    plan={'source_base_commit':'unit','source_owners':{
        name:audit.hashlib.sha256(value).hexdigest()for name,value in originals.items()}}
    return relative,other,current,originals,plan


def test_canonical_numeric_owner_reuses_exact_qualified_alias_bridge(monkeypatch,tmp_path):
    relative,other,current,originals,plan=alias_numeric_owner_fixture(monkeypatch,tmp_path)
    identities=audit.numeric_owner_identities(plan)
    assert set(identities)=={relative,other}
    excluded=('native_m5_feature_provider_contract','fit_native_m5_feature_normalization')
    assert identities[relative]==audit.numeric_ast_identity(originals[relative].decode(),excluded_functions=excluded)
    assert identities[other]==audit.numeric_ast_identity(originals[other].decode())
    assert 'gx1/audit/build_native_m5_ta_feature_cache_v1.py'in audit.SOURCE_PATHS
    assert 'tests/test_build_native_m5_ta_feature_cache_v1.py'in audit.SOURCE_PATHS


def test_canonical_alias_bridge_rejects_helper_or_other_numeric_changes(monkeypatch,tmp_path):
    relative,other,current,originals,plan=alias_numeric_owner_fixture(monkeypatch,tmp_path)
    path=tmp_path/relative
    changed=current.replace('m5.columns.get_loc(per_bar)','0')
    assert changed!=current
    path.write_text(changed)
    with pytest.raises(ValueError,match='EXACT_ALIAS_HELPER_AST'):audit.numeric_owner_identities(plan)
    changed=current.replace('issues=starts+BAR_NS','issues=starts+BAR_NS+1')
    assert changed!=current
    path.write_text(changed)
    with pytest.raises(ValueError,match='CANDIDATE_NUMERIC_OWNER_CHANGED:'):audit.numeric_owner_identities(plan)
    path.write_text(current);(tmp_path/other).write_text('LIMIT=4\ndef formula(x): return x+LIMIT\n')
    with pytest.raises(ValueError,match='CANDIDATE_NUMERIC_OWNER_CHANGED:'):audit.numeric_owner_identities(plan)


def test_numeric_owner_closure_pins_actual_candle_and_cross_tf_formula_producers():
    import ast
    tree=ast.parse((audit.REPO/'gx1/features/htf_features.py').read_text())
    producer=next(node for node in tree.body if isinstance(node,ast.FunctionDef)
        and node.name=='compute_per_bar_features_v4')
    calls={node.func.id for node in ast.walk(producer)if isinstance(node,ast.Call)
        and isinstance(node.func,ast.Name)}
    expected={'gx1.features.entry_candle_primitives_v1':'build_entry_candle_primitive_layer',
        'gx1.scripts.augment_forward_outcome_v2':'_compute_closed_lane_group_a'}
    actual={node.module:alias.name for node in ast.walk(producer)if isinstance(node,ast.ImportFrom)
        and node.module in expected for alias in node.names if alias.name in calls}
    assert actual==expected
    for module in actual:
        path=module.replace('.','/')+'.py'
        assert path in audit.NUMERIC_OWNER_PATHS and path in audit.SOURCE_PATHS


def test_new_transitive_formula_owners_preserve_full_numeric_ast_guards(monkeypatch,tmp_path):
    names=('gx1/features/entry_candle_primitives_v1.py','gx1/scripts/augment_forward_outcome_v2.py',
        'gx1/features/swing_structure_v1.py','gx1/features/model_native_market_context_v1.py')
    original=b'LIMIT=3\ndef formula(x): return x+LIMIT\n'
    for name in names:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(original)
    monkeypatch.setattr(audit,'REPO',tmp_path)
    monkeypatch.setattr(audit,'NUMERIC_OWNER_PATHS',names)
    monkeypatch.setattr(audit.subprocess,'check_output',lambda *args,**kwargs:original)
    plan={'source_base_commit':'unit','source_owners':{
        name:audit.hashlib.sha256(original).hexdigest()for name in names}}
    assert set(audit.numeric_owner_identities(plan))==set(names)
    for name in names:
        path=tmp_path/name
        path.write_text('LIMIT=3\ndef formula(x): return x-LIMIT\n')
        with pytest.raises(ValueError,match='CANDIDATE_NUMERIC_OWNER_CHANGED:'):
            audit.numeric_owner_identities(plan)
        path.write_bytes(original)


def test_actual_006_candidate_receipt_keeps_17_owner_roles_after_all_21_validate_without_arrays(monkeypatch):
    plan=json.loads((audit.REPO/'configs/research/NATIVE_M5_PIPELINE_20261119_006.json').read_text())
    expected=plan['candidate_requalification'];calls=[]
    def no_array_decode(*args,**kwargs):pytest.fail('candidate receipt compatibility must not decode arrays')
    monkeypatch.setattr(np,'load',no_array_decode)
    monkeypatch.setattr(audit,'candidate_clock_proof',lambda *args:expected['cached_clock_admission'])
    validate=audit.numeric_owner_identities
    def full_validation(original_plan):
        result=validate(original_plan);calls.append(set(result));return result
    monkeypatch.setattr(audit,'numeric_owner_identities',full_validation)
    actual=audit.candidate_metadata(plan['inputs'],plan['feature_contract'])
    assert len(audit.CANDIDATE_RECEIPT_NUMERIC_OWNER_PATHS)==17 and len(audit.NUMERIC_OWNER_PATHS)==21
    assert calls==[set(audit.NUMERIC_OWNER_PATHS)]
    assert len(actual['numeric_owner_ast_sha256'])==17
    assert actual==expected
