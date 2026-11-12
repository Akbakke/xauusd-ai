"""Synthetic reporting/claim paths; no data, fit, model or real subprocess."""
from copy import deepcopy

import pytest

from gx1.audit import qualify_market_state_ta_anchor_span_normalization_v1 as worker


@pytest.mark.parametrize('fault', ('none', 'measure', 'late_bookend'))
def test_report_preserves_consumption_and_partial_evidence(monkeypatch, tmp_path, fault):
    from gx1.contracts import gx1_capped_execution_v1 as cap
    published = {}
    digest, head = 'a' * 64, 'b' * 40
    plan = {'source_owners': {}, 'inputs': {}}
    monkeypatch.setattr(worker, 'ROOT', tmp_path / 'run')
    monkeypatch.setattr(cap, 'require_capped_cpu_audit_execution', lambda: {'synthetic': True})
    monkeypatch.setattr(worker, 'inspect', lambda *a, **k: (
        plan, {'path': 'plan', 'sha256': digest}, {'path': 'policy', 'sha256': digest}, head))
    monkeypatch.setattr(worker.io, 'sha', lambda _: digest)
    monkeypatch.setattr(worker.subprocess, 'check_output', lambda args, **kw:
        ('changed' if fault == 'late_bookend' and 'RESULT.json' in published else head)
        if args[1] == 'rev-parse' else '')
    monkeypatch.setattr(worker.signal, 'getitimer', lambda _: (0., 0.))
    monkeypatch.setattr(worker.signal, 'getsignal', lambda _: 'original')
    monkeypatch.setattr(worker.signal, 'signal', lambda *a: None)
    timers = []
    monkeypatch.setattr(worker.signal, 'setitimer', lambda *args: timers.append(args))

    def publish(name, value):
        assert name not in published
        published[name] = deepcopy(value)
        return {'path': name, 'sha256': digest}

    def measure(_):
        publish('SIGNAL_NEW_STATS.json', {'synthetic_partial': True})
        if fault == 'measure':
            raise RuntimeError('synthetic_measure_failure')
        return {'normalization_qualified': True, 'actual_model_baseline_complete': False}

    monkeypatch.setattr(worker, 'publish', publish)
    monkeypatch.setattr(worker, 'measure', measure)
    if fault == 'none':
        worker.run(digest)
    else:
        with pytest.raises(RuntimeError, match='synthetic_measure_failure|SOURCE_BOOKEND'):
            worker.run(digest)
    terminal = published['TERMINAL.json']
    assert 'CLAIM.json' in published and 'SIGNAL_NEW_STATS.json' in published
    assert terminal['status'] == ('COMPLETE' if fault == 'none' else 'FAILED_CONSUMED')
    assert terminal['exit_code'] == (0 if fault == 'none' else 1)
    assert terminal['source_unchanged'] == (fault != 'late_bookend')
    assert ('RESULT.json' in published) == (fault != 'measure')
    assert ('FAILED.json' in published) == (fault != 'none')
    assert terminal['remaining_invocations'] == 0 and terminal['relaunch_allowed'] is False
    assert terminal['test_data_used'] is False and timers[-1][1] == 0


def test_execution_cap_precedes_claim_and_fit(monkeypatch):
    from gx1.contracts import gx1_capped_execution_v1 as cap
    def stop():
        raise RuntimeError('cap_required')
    def forbidden(*args, **kwargs):
        raise AssertionError('must not inspect, claim or fit')
    monkeypatch.setattr(cap, 'require_capped_cpu_audit_execution', stop)
    monkeypatch.setattr(worker, 'inspect', forbidden)
    monkeypatch.setattr(worker, 'publish', forbidden)
    monkeypatch.setattr(worker, 'measure', forbidden)
    with pytest.raises(RuntimeError, match='cap_required'):
        worker.run('a' * 64)
