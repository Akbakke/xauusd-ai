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
            'inputs': {'target_tape': {}}}
    monkeypatch.setattr(audit, 'require_capped_cpu_audit_execution', lambda: {
        'memory_max': 12 * 1024**3, 'memory_high': 12 * 1024**3,
        'swap': 512 * 1024**2, 'numerical_threads': 1,
        'cgroup_path': '/sys/fs/cgroup/synthetic'})
    monkeypatch.setattr(audit, 'require_capped_lock_ancestry', lambda: {'scope': '/synthetic'})
    monkeypatch.setattr(audit, 'inspect', lambda *args, **kwargs: (plan, plan_ref, policy_ref))
    monkeypatch.setattr(audit, 'checked_inputs', lambda plan: None)
    monkeypatch.setattr(audit, 'source_clean', lambda plan: True)
    monkeypatch.setattr(audit, 'ref', lambda path: policy_ref if path == audit.POLICY else plan_ref)
    return plan_ref


def test_success_checks_history_before_fits_and_publishes_one_use(monkeypatch, tmp_path):
    lifecycle(monkeypatch, tmp_path)
    events = []
    monkeypatch.setattr(audit, 'verify_native_m5_bar_facts_003',
        lambda **kwargs: events.append('history') or {'array_count': 76})
    monkeypatch.setattr(audit, 'physical_parameter_fits',
        lambda plan: events.append('parameters') or {'registry_constants': {}})
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
    assert saved['economic_edge_proven'] is False
    with pytest.raises(FileExistsError):
        audit.run('1' * 64)


def test_independent_history_failure_consumes_claim_before_parameter_fit(monkeypatch, tmp_path):
    lifecycle(monkeypatch, tmp_path)
    def failed(**kwargs):
        raise RuntimeError('independent-history-mismatch')
    monkeypatch.setattr(audit, 'verify_native_m5_bar_facts_003', failed)
    monkeypatch.setattr(audit, 'physical_parameter_fits',
                        lambda plan: pytest.fail('parameter fit reached'))
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
    for name in ('read_feature_source', 'verify_native_m5_bar_facts_003',
                 'physical_parameter_fits', 'qualify_physical_pipeline', 'publish'):
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
    source = np.arange(n, dtype=np.int64)
    path = tmp_path / 'TARGETS.npz'
    np.savez(path, source_row=source, issue_ns=issue,
             **{'future_' + key: value for key, value in truth.items()})
    ready = np.ones(n, bool)
    ready[2] = False
    provider = SimpleNamespace(source_row=source, issue_ns=issue, input_ready=ready)
    rows, selected, report = audit.target_axis({'inputs': {'target_tape': {'path': str(path)}}}, provider)
    np.testing.assert_equal(rows, np.setdiff1d(source, [2, 3, 24]))
    assert len(rows) == report['fit_issue_rows'] == 22
    assert not selected['mature'][np.flatnonzero(rows == 5)[0], -1]
    assert selected['target_row'][np.flatnonzero(rows == 5)[0], -1] == -1
    assert selected['close_sign'].dtype == np.int8
    assert report['input_not_ready_rows'] == 1
    assert report['no_mature_supervision_rows'] == 2


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
