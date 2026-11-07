"""Synthetic gates for the one-use mature-prefix statistics worker."""
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as worker


def _cohorts():
    return [dict(fold_id=i, official_fit_cutoff_ns=1000 + i,
                 effective_fit_cutoff_ns=900 + i, joined_fit_rows=i + 1,
                 fit_parent_rows_sha256=str(i) * 64,
                 fit_decision_ns_sha256=str(i) * 64)
            for i in worker.FOLD_IDS]


def test_five_raw_qualifications_precede_conditional_fits(monkeypatch, tmp_path):
    events = []
    cohorts = _cohorts()
    monkeypatch.setattr(worker, '_validate_axes', lambda *_: None)
    monkeypatch.setattr(worker, '_release_pages', lambda *_: None)
    fields = {name: {'field_names': [name]} for name in (
        'signal', 'ctx_cont', 'mtf_m5', 'mtf_m15', 'mtf_h1', 'mtf_h4', 'mtf_d1',
        'gc_mtf_m5', 'gc_mtf_m15', 'gc_mtf_h1', 'gc_mtf_h4', 'gc_mtf_d1')}

    def qualify(**kwargs):
        fold = kwargs['cutoff_ns'] - 900
        events.append(('qualify', fold))
        item = cohorts[fold - 1]
        allowed = fold != 3
        report = {'statistics_fits': 0, 'fit_allowed': allowed,
                  'proof': {'prefix': {'rows': item['joined_fit_rows'],
                                      'cutoff_ns': item['effective_fit_cutoff_ns'],
                                      'parent_rows_sha256': item['fit_parent_rows_sha256'],
                                      'bar_start_ns_sha256': worker._axis_sha(np.arange(10)[:item['joined_fit_rows']]),
                                      'decision_ns_sha256': item['fit_decision_ns_sha256']}},
                  'support': fields,
                  'blockers': [] if allowed else [{'surface': 'gc_mtf_d1',
                                                   'field': 'rare',
                                                   'reason': 'no_observed_prefix_values'}]}
        return {'report': report, 'population': {'inputs': {'fold': fold}} if allowed else None}

    def fit(value):
        fold = value['population']['inputs']['fold']
        events.append(('fit', fold))
        assert fold in (2, 4, 5)
        return {'prefix_qualification': value['report'],
                'old055_or223_normalization_identity_reused': False,
                'native_launch_authorized': False,
                'genuine_model_fit_authorized': False,
                'base_normalization': {'surfaces': dict(list(fields.items())[:7])},
                'gc_mtf_history': {'normalization': {'surfaces': dict(list(fields.items())[7:])}},
                'contract_sha256': 'a' * 64}

    monkeypatch.setattr(worker, 'qualify_native_direction_prefix_population', qualify)
    monkeypatch.setattr(worker, 'fit_native_direction_prefix_normalization', fit)
    prepared = {'axes': {'parent_rows': np.arange(10),
                         'bar_start_ns': np.arange(10),
                         'decision_ns': np.arange(10)},
                'expected_axes': {}, 'entry_sequence_source': {},
                'current': {'snap': None, 'ctx_cont': None, 'ctx_cat': None},
                'cache': None, 'gc': {'surfaces': None, 'layout': None}}
    info = {'source': {'feature_field_order': []}}
    worker.PUBLISHED_BYTES = 0
    reports, fits = worker._process({'cohorts': cohorts, 'source_owners': {}},
        {'base_lineage': {}}, info, None, prepared, {}, {}, 'head', tmp_path,
        time.monotonic() + 1000)
    assert events[:5] == [('qualify', i) for i in worker.FOLD_IDS]
    assert events[5:] == [('fit', 2), ('fit', 4), ('fit', 5)]
    assert reports[0]['action'] == 'REPORT_ONLY'
    assert reports[2]['action'] == 'FIT_BLOCKED'
    assert reports[2]['blockers'][0]['field'] == 'rare'
    assert [item['fold_id'] for item in fits] == [2, 4, 5]
    assert len(list(tmp_path.glob('QUALIFICATION_FOLD_*.json'))) == 5
    assert len(list(tmp_path.glob('PREFIX_NORMALIZATION_FOLD_*.json'))) == 3


def test_deadline_requires_five_minute_reserve():
    with pytest.raises(RuntimeError, match='DEADLINE_RESERVE'):
        worker._time_guard(time.monotonic() + 299)
    worker._time_guard(time.monotonic() + 301)


def test_claimed_failure_publishes_terminal_without_relaunch(monkeypatch, tmp_path):
    plan = {'cohorts': _cohorts()}
    plan_ref = {'path': 'plan', 'sha256': 'a' * 64}
    policy_ref = {'path': 'policy', 'sha256': 'b' * 64}
    monkeypatch.setattr(worker, 'preflight', lambda *args, **kwargs:
                        (plan, plan_ref, 'head', policy_ref))
    monkeypatch.setattr(worker, '_receipt_chain', lambda: {'provider_plan': {}, 'gc_layout': {}})
    provider = SimpleNamespace(validate_metadata=lambda _: {})
    monkeypatch.setattr(worker, '_provider_module', lambda: provider)
    monkeypatch.setattr(worker, '_prepare_inputs', lambda *args: {})
    monkeypatch.setattr(worker, '_process', lambda *args: (_ for _ in ()).throw(RuntimeError('measured blocker')))
    with pytest.raises(RuntimeError, match='measured blocker'):
        worker.run(worker.PLAN_PATH, 'a' * 64, tmp_path / 'run')
    root = tmp_path / 'run'
    assert (root / 'CLAIM.json').is_file()
    failed = json.loads((root / 'FAILED.json').read_text())
    terminal = json.loads((root / 'TERMINAL.json').read_text())
    assert failed['no_relaunch_under_this_plan'] is True
    assert terminal['status'] == 'FAILED' and terminal['exit_code'] == 1
    assert terminal['claim']['path'] == str(root / 'CLAIM.json')
    assert terminal['test_data_used'] is False


def test_cohort_metadata_must_match_raw_model_axes(monkeypatch):
    decisions = np.arange(1, 11, dtype=np.int64)
    axes = {'parent_rows': np.arange(10, dtype=np.int64),
            'bar_start_ns': decisions - 1,
            'decision_ns': decisions}
    monkeypatch.setattr(worker, 'SOURCE_ROWS', 10)
    monkeypatch.setattr(worker, 'MATURE_ROWS', [1, 3, 5, 7, 9])
    monkeypatch.setattr(worker, 'OLD_ROWS', [2, 4, 6, 8, 10])
    monkeypatch.setattr(worker, 'DAY_NS', 1)
    monkeypatch.setattr(worker.target, 'FOLD_BOUNDARIES_NS', (3, 5, 7, 9, 11, 13))
    cohorts = []
    for fold, official in enumerate(worker.target.FOLD_BOUNDARIES_NS[:5], 1):
        stop = worker.MATURE_ROWS[fold - 1]
        cohorts.append({'fold_id': fold, 'official_fit_cutoff_ns': official,
                        'effective_fit_cutoff_ns': official - 1,
                        'joined_fit_rows': stop,
                        'old_joined_fit_rows': worker.OLD_ROWS[fold - 1],
                        'fit_parent_rows_sha256': worker._axis_sha(axes['parent_rows'][:stop]),
                        'fit_decision_ns_sha256': worker._axis_sha(axes['decision_ns'][:stop])})
    worker._validate_cohort_contract(cohorts)
    worker._validate_axes(cohorts, axes)
    cohorts[2]['fit_decision_ns_sha256'] = '0' * 64
    with pytest.raises(RuntimeError, match='COHORT_AXIS:3'):
        worker._validate_axes(cohorts, axes)


def _admitted_policy(plan_ref, out_dir):
    return {'training_enabled': False, 'native_training_enabled': False,
            'native_launch_authorized': False, 'full_val_allowed': False,
            worker.STAGE: {'status': 'ADMITTED', 'decision': 'ADMITTED',
                'plan': plan_ref, 'execution_authorized': True,
                'data_reads_authorized': True, 'normalization_fit_authorized': True,
                'claimed_invocations': 0, 'completed_invocations': 0,
                'max_invocations': 1, 'relaunch_allowed': False,
                'memory_max': worker.EXECUTION['memory'],
                'swap_max': worker.EXECUTION['swap'],
                'wall_seconds': worker.EXECUTION['timeout_seconds'],
                'test_data_used': False, 'checkpoint_reads_authorized': False,
                'broker_access_allowed': False, 'model_forwards_authorized': 0,
                'optimizer_steps_authorized': 0, 'training_authorized': False,
                'test_authorized': False, 'output_root': str(out_dir)}}


@pytest.mark.parametrize('scope,key,bad', [
    ('global', 'training_enabled', True),
    ('global', 'native_training_enabled', True),
    ('global', 'native_launch_authorized', True),
    ('gate', 'status', 'PREPARED'),
    ('gate', 'relaunch_allowed', True),
    ('gate', 'memory_max', '8G'),
    ('gate', 'swap_max', '1G'),
    ('gate', 'wall_seconds', 3600),
    ('gate', 'test_data_used', True),
    ('gate', 'checkpoint_reads_authorized', True),
    ('gate', 'broker_access_allowed', True),
    ('gate', 'test_authorized', True),
])
def test_policy_scope_fails_closed(scope, key, bad, tmp_path):
    plan_ref = {'path': 'plan', 'sha256': 'a' * 64}
    policy = _admitted_policy(plan_ref, tmp_path)
    worker._policy_gate(policy, plan_ref, tmp_path)
    where = policy if scope == 'global' else policy[worker.STAGE]
    where[key] = bad
    with pytest.raises(RuntimeError, match='POLICY_GATE'):
        worker._policy_gate(policy, plan_ref, tmp_path)


def test_internal_alarm_publishes_failed_terminal(monkeypatch, tmp_path):
    import signal
    plan_ref = {'path': 'plan', 'sha256': 'a' * 64}
    policy_ref = {'path': 'policy', 'sha256': 'b' * 64}
    monkeypatch.setattr(worker, 'preflight', lambda *args, **kwargs:
                        ({'cohorts': _cohorts()}, plan_ref, 'head', policy_ref))
    monkeypatch.setattr(worker, '_receipt_chain', lambda: {'provider_plan': {}, 'gc_layout': {}})
    monkeypatch.setattr(worker, '_provider_module', lambda:
                        SimpleNamespace(validate_metadata=lambda _: {}))
    monkeypatch.setattr(worker, '_prepare_inputs', lambda *args: {})
    monkeypatch.setattr(worker, 'LIMITS', {**worker.LIMITS, 'internal_wall_seconds': 0.2})

    def long_unit(*_args):
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
        time.sleep(1.0)

    monkeypatch.setattr(worker, '_process', long_unit)
    prior = signal.getsignal(signal.SIGALRM)
    with pytest.raises(TimeoutError, match='INTERNAL_WALL_LIMIT'):
        worker.run(worker.PLAN_PATH, 'a' * 64, tmp_path / 'timeout_run')
    root = tmp_path / 'timeout_run'
    failure = json.loads((root / 'FAILED.json').read_text())
    terminal = json.loads((root / 'TERMINAL.json').read_text())
    assert failure['error_type'] == 'TimeoutError'
    assert terminal['status'] == 'FAILED' and terminal['exit_code'] == 1
    assert terminal['failure']['path'] == str(root / 'FAILED.json')
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert signal.getsignal(signal.SIGALRM) == prior
