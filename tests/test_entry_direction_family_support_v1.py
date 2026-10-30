'''Synthetic streaming support counts and bounded adapter guards; no payloads.'''
from copy import deepcopy
import numpy as np
import pytest
from gx1.scripts import audit_entry_direction_family_support_v1 as audit
from tests.test_entry_direction_family_intervention_v1 import case


def test_streamed_missing_constant_and_unique_history_are_distinct():
    raw = np.array([[[0, 0, 7], [2, 1, 7]], [[2, 1, 7], [4, 1, 7]]], np.float32)
    ids = np.array([[10, 11], [11, 12]], np.int64)
    c = audit.SurfaceSupport(['value', 'observed', 'constant'], [[0, 1]])
    for row in range(2):
        c.update(raw[row:row+1], raw[row:row+1] - 7, ids[row:row+1])
    r = c.finish()
    assert r['consumed_physical_row_occurrences'] == 4
    assert r['unique_physical_rows'] == 3
    assert r['observed_consumed'] == [3, 4, 4]
    assert r['missing_consumed'] == [1, 0, 0]
    assert r['observed_unique_physical'] == [2, 3, 3]
    assert r['observed_history_rows_per_parent_min'] == [1, 2, 2]
    assert r['observed_history_rows_per_parent_max'] == [2, 2, 2]
    assert r['constant_when_observed'] == [False, False, True]
    assert r['nonzero_normalized_observed_consumed'][2] == 0
    assert r['observed_min'] == [2, 0, 7]
    whole = audit.SurfaceSupport(c.names, c.pairs)
    whole.update(raw, raw - 7, ids)
    assert whole.finish() == r


def test_later_support_is_unknown_where_train_unobserved():
    train = audit.SurfaceSupport(['value', 'observed'], [[0, 1]])
    train.update(np.array([[0, 0]], np.float32), np.zeros((1, 2), np.float32), np.array([0], np.int64))
    assert train.finish()['constant_when_observed'][0] is None
    later = audit.SurfaceSupport(train.names, train.pairs)
    later.update(np.array([[5, 1]], np.float32), np.ones((1, 2), np.float32), np.array([1], np.int64),
        train_range=(train.minimum, train.maximum, train.count > 0))
    assert later.finish()['outside_observed_train_range_consumed'] == [None, 1]


def test_masked_storage_and_negative_physical_ids_rejected():
    c = audit.SurfaceSupport(['value', 'observed'], [[0, 1]])
    with pytest.raises(RuntimeError, match='MASKED_STORAGE'):
        c.update(np.array([[2, 0]], np.float32), np.zeros((1, 2), np.float32), np.array([0], np.int64))
    with pytest.raises(RuntimeError, match='PHYSICAL_ROW_IDENTITY'):
        c.update(np.zeros((1, 2), np.float32), np.zeros((1, 2), np.float32), np.array([-1], np.int64))


def synthetic_adapter(monkeypatch, case):
    args, contract, features = case
    axes = {'parent_rows': np.arange(4, dtype=np.int64), 'bar_start_ns': np.arange(4, dtype=np.int64)*300000000000,
        'decision_ns': (np.arange(4, dtype=np.int64)+1)*300000000000}
    for x in axes.values(): x.flags.writeable = False
    view = {'contract_sha256': 'synthetic_view', 'entry_execution_profile': contract['profile_reference'],
        'prefix_normalization': {'base_sha256': args['base_normalization']['contract_sha256'], 'gc_normalization_sha256': args['gc_normalization']['contract_sha256']},
        'max_batch_rows': 2, 'prefix': {'rows': 2, 'cutoff_ns': 900000000000},
        'full_axes': {'rows': 4, **{k+'_sha256': audit.axis_sha(v) for k, v in axes.items()}}}
    prepared = {'arrays': axes, 'entry_sequence_source': None, 'current': dict.fromkeys(('snap', 'ctx_cont', 'ctx_cat')),
        'cache': None, 'gc': {'surfaces': None, 'layout': args['gc_mtf_history']['layout']}}
    plan = {'schema_version': audit.SCHEMA_VERSION, 'mode': 'full', 'train_parent_rows': [0, 1], 'later_parent_rows': [2, 3],
        'expected_train_rows': 2, 'expected_later_rows': 2, 'train_cutoff_ns': 900000000000, 'later_cutoff_ns': 1500000000000,
        'feature_view_sha256': view['contract_sha256'], 'intervention_sha256': contract['contract_sha256'], 'max_batch_rows': 2, 'max_wall_seconds': 120}
    plan['contract_sha256'] = audit.family._sha(plan)
    calls = []
    def gather(_view, **kw):
        ids = kw['batch_parent_rows']; calls.extend(ids.tolist())
        f = {k: v[:len(ids)].copy() for k, v in features.items()}
        return {'features': f, 'witness': {'parent_rows': ids.tolist(), 'source_positions': (ids+100).tolist(),
            'bar_start_ns': axes['bar_start_ns'][ids].tolist(), 'decision_ns': axes['decision_ns'][ids].tolist(),
            'multi_tf': {tf: [{'native_row_indices': list(range(int(i), int(i)+args['gc_mtf_history']['layout']['window_rows'][tf]))} for i in ids] for tf in args['gc_mtf_history']['layout']['timeframes']}}}
    monkeypatch.setattr(audit.view_owner, '_require_view', lambda v: None)
    monkeypatch.setattr(audit.view_owner, 'gather_native_direction_feature_batch', gather)
    def run(p=plan):
        return audit.audit_prepared_family_support(prepared, view, contract, plan=p, expected_plan_sha256=p['contract_sha256'],
            base_normalization=args['base_normalization'], gc_normalization=args['gc_normalization'])
    return run, plan, calls


def test_adapter_summarizes_all_14_surfaces_and_exact_axes(monkeypatch, case):
    run, plan, calls = synthetic_adapter(monkeypatch, case)
    result = run()
    assert calls == [0, 1, 2, 3]
    assert result['full_population_measured'] is True
    assert len(result['inventory']) == 14
    assert result['model_forwards'] == result['optimizer_steps'] == 0
    assert result['test_data_used'] is False
    for name in ('train', 'later'):
        assert result['partitions'][name]['parent_rows_sha256'] == audit.axis_sha(plan[name+'_parent_rows'])
        assert len(result['partitions'][name]['surfaces']) == 14
    assert result['structural_history_padding']['rows'] == 0


@pytest.mark.parametrize('change,reason', [({'train_parent_rows': [1]}, 'FULL_COHORT'),
    ({'train_parent_rows': [0, 2]}, 'SELECTION_BOUNDARY'), ({'later_parent_rows': [2.0, 3.0]}, 'SELECTION_INTEGER_IDS'),
    ({'later_cutoff_ns': audit.history.TEST_BOUNDARY_NS+1}, 'PLAN_SCOPE')])
def test_adapter_rejects_scope_before_gather(monkeypatch, case, change, reason):
    run, plan, calls = synthetic_adapter(monkeypatch, case)
    plan = deepcopy(plan); plan.update(change); plan.pop('contract_sha256')
    plan['contract_sha256'] = audit.family._sha(plan)
    with pytest.raises(RuntimeError, match=reason): run(plan)
    assert calls == []
