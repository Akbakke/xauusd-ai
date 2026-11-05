"""Synthetic admission and axis checks; no genuine input files are opened."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from gx1.audit import dynamic_mtf_october_input_only as owner


def _stub():
    ref = {'path': '/synthetic/plan.json', 'sha256': '1' * 64}
    plan = {'schema_version': 'gx1_dynamic_m5_october_input_only_plan_v1',
            'status': 'PREPARED_NOT_ADMITTED',
            'plan_id': 'DYNAMIC_M5_OCTOBER_INPUT_ONLY_20261105_001',
            'max_invocations': 1,
            'output_root': '/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_INPUT_ONLY_20261105_001',
            'restrictions': {'target_arrays': False, 'model_construction': False,
                'model_forward': False, 'optimizer_updates': False, 'test': False,
                'broker_or_network': False, 'orders_or_trading': False, 'spending_usd': 0}}
    policy = {'training_enabled': False, 'full_val_allowed': False,
              'full_epoch_training_allowed': False,
              'direction_edge_research_goal_20261105': {'training_enabled': False,
                  'native_launch_authorized': False, 'test_access_authorized': False,
                  'broker_orders_or_trading_authorized': False},
              owner.POLICY_KEY: {'status': 'ADMITTED_ONCE', 'plan': ref,
                  'operator_approved': True, 'remaining_invocations': 1}}
    return plan, policy, ref


def test_explicit_admission_never_follows_prepared_plan_or_prior_approval():
    plan, policy, ref = _stub()
    assert owner._admission(plan, policy, ref) == policy[owner.POLICY_KEY]
    for change in ('no_new_admission', 'wrong_ref', 'old_approval', 'training_on', 'network_on'):
        p, q, r = deepcopy(plan), deepcopy(policy), deepcopy(ref)
        if change == 'no_new_admission':
            q[owner.POLICY_KEY]['status'] = 'PREPARED_NOT_ADMITTED'
        elif change == 'wrong_ref':
            q[owner.POLICY_KEY]['plan']['sha256'] = '2' * 64
        elif change == 'old_approval':
            q[owner.POLICY_KEY]['remaining_invocations'] = 0
        elif change == 'training_on':
            q['training_enabled'] = True
        else:
            p['restrictions']['broker_or_network'] = True
        with pytest.raises(RuntimeError):
            owner._admission(p, q, r)


def _digest(values):
    return hashlib.sha256(np.asarray(values, dtype='<i8').tobytes()).hexdigest()


def test_evaluation_axis_is_separate_and_target_free():
    start = 1_700_000_000_000_000_000
    starts = start + np.arange(6, dtype=np.int64) * 300_000_000_000
    axes = {'parent_rows': np.arange(6, dtype=np.int64),
            'bar_start_ns': starts, 'decision_ns': starts + 300_000_000_000}
    full = {'source': {'path': '/synthetic/axes.npz', 'sha256': 'a'*64}, 'rows': 6,
            **{name + '_sha256': _digest(values) for name, values in axes.items()}}
    fit = {'rows': 3, 'cutoff_ns': int(axes['decision_ns'][3]),
           **{name + '_sha256': _digest(values[:3]) for name, values in axes.items()}}
    eval_spec = {'parent_row_range': [3, 5], 'rows': 2,
                 'cutoff_ns': int(axes['decision_ns'][5])}
    result = owner._axis_receipt(axes, full, fit, eval_spec)
    assert result['evaluation']['rows'] == 2
    assert result['evaluation']['parent_rows_sha256'] == _digest(axes['parent_rows'][3:5])
    assert result['target_members_decoded'] == result['evaluation_outcomes_decoded'] == 0
    bad = deepcopy(fit)
    bad['decision_ns_sha256'] = '0'*64
    with pytest.raises(RuntimeError, match='EXACT_FIT_HASHES'):
        owner._axis_receipt(axes, full, bad, eval_spec)
    bad_eval = deepcopy(eval_spec)
    bad_eval['rows'] += 1
    with pytest.raises(RuntimeError, match='EXACT_OCTOBER_GEOMETRY'):
        owner._axis_receipt(axes, full, fit, bad_eval)


def test_prepared_plan_sources_are_bound_without_opening_data_payloads():
    plan = json.loads(owner.PLAN.read_text())
    assert plan['status'] == 'PREPARED_NOT_ADMITTED'
    assert plan['fit_prefix']['rows'] == 29849
    assert plan['evaluation_axis']['parent_row_range'] == [29849, 36172]
    assert plan['restrictions']['target_arrays'] is False
    for path, expected in plan['source_files_sha256'].items():
        assert owner._sha_file(owner.REPO / path) == expected
    with pytest.raises(RuntimeError, match='EXPLICIT_ADMISSION_REQUIRED'):
        prepared, policy, ref = _stub()
        policy[owner.POLICY_KEY]['status'] = 'PREPARED_NOT_ADMITTED'
        owner._admission(prepared, policy, ref)


def test_reader_touches_only_fit_feature_prefix(monkeypatch):
    from types import SimpleNamespace
    from gx1.audit import dynamic_mtf_december_normalization as loader
    from gx1.contracts import entry_exit_feature_surface_v1 as surface

    class Guarded:
        def __init__(self, values, limit):
            self.values, self.limit = values, limit
        def __getitem__(self, key):
            rows = key[0] if isinstance(key, tuple) else key
            if isinstance(rows, slice):
                assert rows.stop is not None and rows.stop <= self.limit
            elif isinstance(rows, np.ndarray):
                assert len(rows) and int(rows.max()) < self.limit
            else:
                assert int(rows) < self.limit
            return self.values[key]

    bar = 300_000_000_000
    times = np.arange(120, dtype=np.int64) * bar
    positions = np.array([100, 101, 102], dtype=np.int64)
    signal = np.column_stack((np.arange(120), np.arange(120) + 1)).astype(np.float32)
    current = signal[positions].copy()
    groups = {'source050': {'signal': Guarded(signal, 103)},
              'current051': {'snap': Guarded(current, 3),
                             'ctx_cont': Guarded(np.ones((6, 1), np.float32), 3),
                             'ctx_cat': Guarded(np.ones((6, 1), np.int64), 3)}}
    provider = SimpleNamespace(map_reuse=lambda *_: groups)
    monkeypatch.setattr(surface, 'load_m1_feature_surface_times',
                        lambda *_args, **_kwargs: SimpleNamespace(asi8=times))
    monkeypatch.setattr(loader, '_history_views',
        lambda plan, transition, layout, last_issue, **kwargs:
            ({'M5': 'prefix'}, {'M5': 'prefix'}, {'layout': 'prefix'}))
    axes = {'bar_start_ns': times[positions], 'decision_ns': times[positions] + bar}
    plan = {'october_gc_transition': {'path': '/synthetic/transition.json', 'sha256': '1'*64},
            'october_gc_layout': {'path': '/synthetic/layout.json', 'sha256': '2'*64}}
    info = {'source': {'output_parquet': '/synthetic/source.parquet'}, 'reuse': {}, 'audit': {}}
    got = owner._prepare_prefix_inputs(provider, plan, info, axes, fit_rows=3,
                                       last_issue_ns=int(axes['decision_ns'][-1]))
    assert got['current051_rows_readable'] == 3
    assert got['source050_rows_readable'] == 103
    assert len(got['entry_sequence_source']['signal']) == 103
    assert len(got['current']['snap']) == 3
