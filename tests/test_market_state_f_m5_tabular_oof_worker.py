"""Focused F-target separation and one-use terminal-path checks."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gx1.audit import run_market_state_f_m5_tabular_oof_v1 as worker


def _truth(bits):
    allowed = np.full((len(bits), 5), 7, dtype=np.uint8)
    allowed[:, 0] = bits
    eligible = np.zeros_like(allowed, dtype=np.bool_)
    eligible[:, 0] = True
    mature = np.ones_like(allowed, dtype=np.bool_)
    return {'allowed': allowed, 'eligible': eligible, 'mature': mature}



def test_preflight_rejects_unqualified_current_width_before_claim(monkeypatch):
    monkeypatch.setattr(worker.base, '_reference_chain', lambda: {'cohorts': []})
    monkeypatch.setattr(worker, 'INPUT_REFS', {})
    payloads = iter(({'schema_version': 'gx1_market_state_tabular_feature_schema_v1',
                      'no_future_targets_in_features': True, 'width': 3991,
                      'rows': 6, 'matrix_float32_c_order_sha256': 'a' * 64},
                     {'folds': [{}] * 5},
                     {'old_feature_schema': worker.OLD_FEATURE_SCHEMA,
                      'span_result': worker.SPAN_RESULT,
                      'old_projection_matches_immutable_schema': True,
                      'old_matrix_projection_sha256': 'a' * 64,
                      'current_width': 3991, 'model_rows': 6},
                     {'folds': {str(i): {} for i in range(1, 6)},
                      'surface_widths': [317, 132, 255, 255, 255, 255, 255],
                      'test_data_used': False}))
    monkeypatch.setattr(worker, '_checked_json', lambda _ref: next(payloads))
    with pytest.raises(RuntimeError, match='CURRENT_SCHEMA_AND_AUDIT'):
        worker._inputs()


def test_f_head_fits_only_f_m5_labels_and_scores_full_later_axis(monkeypatch, tmp_path):
    parents = np.arange(8, dtype=np.int64)
    issues = np.arange(10, 18, dtype=np.int64)
    fit_truth = _truth([1, 4, 1, 4])
    score_truth = _truth([1, 4, 1, 4])
    s_score = _truth([1, 4, 5, 5])
    cohort = {'fold_id': 1, 'joined_fit_rows': 4, 'old_joined_fit_rows': 4,
              'fit_parent_rows': parents[:4], 'fit_decision_ns': issues[:4],
              'score_parent_rows': parents[4:], 'score_decision_ns': issues[4:],
              'score_parent_rows_sha256': 'a' * 64,
              'score_decision_ns_sha256': 'b' * 64}
    declaration = {key: cohort[key] for key in ('fold_id', 'joined_fit_rows',
                                                'old_joined_fit_rows')}
    axis = {'fold_id': 1, 'score_rows': 4,
            'score_parent_rows_sha256': 'a' * 64,
            'score_decision_ns_sha256': 'b' * 64}
    plan = {'cohorts': [declaration], 'score_axes': [axis],
            'target_audit_counts': [{'score_f_known_n': 4}]}
    monkeypatch.setattr(worker, 'mature_fold_training_targets',
                        lambda _bound, *, fold_id: {'cohort': cohort})
    monkeypatch.setattr(worker.base, '_time_guard', lambda _: None)
    monkeypatch.setattr(worker.base, '_write_bytes', lambda *args: {'sha256': 'c' * 64})
    monkeypatch.setattr(worker.base, '_write_json', lambda *args: {'sha256': 'd' * 64})
    monkeypatch.setattr(worker.model, '_categorical_indices', lambda *args: [])
    seen = {}
    class LightGBM:
        @staticmethod
        def Dataset(matrix, *, label, **_kwargs):
            seen['train_shape'] = matrix.shape
            seen['labels'] = label.tolist()
            return object()

        @staticmethod
        def train(_params, _dataset, *, num_boost_round):
            assert num_boost_round == 64
            return SimpleNamespace(
                predict=lambda matrix, **_kwargs: np.array([.2, .8, .3, .7]),
                model_to_string=lambda: 'model')
    monkeypatch.setattr(worker.model, '_import_lightgbm', lambda: LightGBM)
    f_bound = SimpleNamespace(gather=lambda _p, _d, *, partition, **_k:
                              fit_truth if partition == 'fit' else score_truth)
    s_bound = SimpleNamespace(gather=lambda *_a, **_k: s_score)
    view = SimpleNamespace(parent_rows=parents, decision_ns=issues,
                           feature_names=('a', 'b'), categorical=())
    prior = worker._prior_record(fit_truth, fold_id=1, expected_count=4)
    record, weekly = worker._fold(plan, s_bound, f_bound, view,
                                  np.ones((8, 2), dtype=np.float32), prior,
                                  1, tmp_path, float('inf'))
    assert seen == {'train_shape': (4, 2), 'labels': [0, 1, 0, 1]}
    assert record['sha256'] == 'd' * 64
    assert weekly['allowed'] is score_truth['allowed']
    assert np.allclose(weekly['model_probabilities'][:, 0, 2], [.2, .8, .3, .7])
    assert worker._auc(np.array([0, 1, 0, 1]), np.array([.2, .8, .3, .7])) == 1.0
    assert worker._auc(np.array([0, 1]), np.array([.5, .5])) == .5


@pytest.mark.parametrize('fail_fold', [None, 3])
def test_claimed_run_publishes_terminal_on_success_or_failure(monkeypatch, tmp_path, fail_fold):
    monkeypatch.setattr(worker.base, 'RUNS', tmp_path)
    out_dir = tmp_path / 'RUN'
    parents = np.arange(6, dtype=np.int64)
    issues = np.arange(100, 106, dtype=np.int64)
    view = SimpleNamespace(parent_rows=parents, decision_ns=issues,
                           width=worker.model.EXPECTED_WIDTH,
                           feature_names=('old',), categorical=())
    plan = {'cohorts': [{}] * 5, 'source_owners': {}}
    plan_ref = {'path': 'plan', 'sha256': 'a' * 64}
    policy_ref = {'path': 'policy', 'sha256': 'b' * 64}
    monkeypatch.setattr(worker, 'preflight', lambda *_a, **_k:
                        (plan, plan_ref, 'head', policy_ref))
    schema = {'matrix_float32_c_order_sha256': 'd' * 64, 'rows': 6, 'width': 3991}
    audit = {'folds': [{'fit': {'F_m5_known_side_n': 2}}] * 5}
    monkeypatch.setattr(worker, '_inputs', lambda: ({'provider_plan': {}, 'gc_layout': {}},
                                                      schema, audit))
    provider = SimpleNamespace(validate_metadata=lambda _: {})
    monkeypatch.setattr(worker.source, '_provider_module', lambda: provider)
    monkeypatch.setattr(worker.source, '_validate_axes', lambda *_a: None)
    monkeypatch.setattr(worker.overlay, 'build', lambda *_a, **_k: (view, {'synthetic': True}))
    monkeypatch.setattr(worker, '_checked_json', lambda item: (
        {'current_matrix_float32_c_order_sha256': 'c' * 64}
        if item == worker.CURRENT_VIEW_REVIEW else {}))
    bound = SimpleNamespace(parent_rows=parents, decision_ns=issues,
                            gather=lambda *_a, **_k: _truth([1, 4]))
    monkeypatch.setattr(worker.base.target, 'load_bound_five_endpoint_targets', lambda: bound)
    monkeypatch.setattr(worker.future, 'load_bound_future_displacement', lambda: bound)
    monkeypatch.setattr(worker, 'mature_fold_training_targets', lambda *_a, **_k:
                        {'cohort': {'fit_parent_rows': parents[:2],
                                    'fit_decision_ns': issues[:2]}})
    monkeypatch.setattr(worker, '_current_matrix', lambda *_a:
                        (np.zeros((6, worker.model.EXPECTED_WIDTH), dtype=np.float32),
                         'c' * 64, 'd' * 64))
    monkeypatch.setattr(worker.base, '_time_guard', lambda _: None)
    monkeypatch.setattr(worker, '_bookend', lambda *_a: None)
    calls = []
    def fake_fold(_plan, _s, _f, _view, _matrix, _prior, fold_id, *_a):
        calls.append(fold_id)
        if fold_id == fail_fold:
            raise RuntimeError('synthetic fold failure')
        return {'fold_id': fold_id}, {'fold_id': fold_id}
    monkeypatch.setattr(worker, '_fold', fake_fold)
    monkeypatch.setattr(worker.review, 'paired_m5_weekly_review', lambda _items:
                        {'decision': 'NO_POSITIVE_GATE'})
    if fail_fold is None:
        output = worker.run(Path('plan'), 'a' * 64, out_dir)
        assert output['terminal']['path'] == str(out_dir / 'TERMINAL.json')
        assert calls == [1, 2, 3, 4, 5]
        assert (out_dir / 'RESULT.json').is_file()
    else:
        with pytest.raises(RuntimeError, match='synthetic fold failure'):
            worker.run(Path('plan'), 'a' * 64, out_dir)
        assert calls == [1, 2, 3]
        assert (out_dir / 'FAILED.json').is_file()
    import json
    terminal = json.loads((out_dir / 'TERMINAL.json').read_text())
    assert terminal['status'] == ('FAILED' if fail_fold else 'COMPLETE')
    assert (out_dir / 'CLAIM.json').is_file()
    assert (out_dir / 'PRIOR_FROZEN.json').is_file()
