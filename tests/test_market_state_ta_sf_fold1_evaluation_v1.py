"""Synthetic full-population evaluation plumbing; no model forward or data read."""
from __future__ import annotations

import copy
import hashlib
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gx1.contracts import market_state_time_query_native_F_batch_v1 as batch
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner


class _Model:
    def __init__(self):
        self.training = True
        self.child = SimpleNamespace(training=False)
        self.value = torch.tensor([2.0])

    def named_modules(self):
        return [('', self), ('child', self.child)]

    def eval(self):
        for _, module in self.named_modules():
            module.training = False

    def state_dict(self):
        return {'value': self.value}


def _batch_measurement(rows, issues, partition):
    n = len(rows)
    stats = {'count': n, 'brier_sum': n / 4, 'log_sum': n / 2}
    scores = {
        role: {task: {tf: {metric: dict(stats) for metric in objective.METRICS}
                     for tf in objective.TIMEFRAMES}
               for task in ('state', 'future')}
        for role in ('model', 'prior')
    }
    return {
        'schema_version': 'gx1_native_S_F_endpoint_batch_measurement_v1',
        'fold_id': 1, 'partition': partition, 'future_weight': 1.0,
        'native_query_k': list(objective.NATIVE_K),
        'loss_contract_sha256': 'a' * 64,
        'model_contract_sha256': 'b' * 64,
        'feature_view_sha256': 'c' * 64,
        'frozen_prior_contract_sha256': 'd' * 64,
        'rows': n, 'scores': scores,
        'observation_sha256': hashlib.sha256(rows.tobytes() + issues.tobytes()).hexdigest(),
    }


def _fake_evaluation(monkeypatch, *, partition, fail=None):
    n = runner.scope.FIT_ROWS if partition == 'fit' else runner.scope.SCORE_ROWS
    first = 0 if partition == 'fit' else runner.scope.FOLD['oof_start_row']
    rows = np.arange(first, first + n, dtype=np.int64)
    issues = (rows + 1) * np.int64(300_000_000_000)
    model = _Model()
    calls = []
    frozen_priors = {'synthetic_frozen_prior': True}
    endpoint_bound, future_bound = object(), object()

    def features(prepared, view, receipts, parents, clocks, left, right, **kwargs):
        assert np.array_equal(parents, rows) and np.array_equal(clocks, issues)
        return ({'seq_x': np.zeros((right - left, 1), np.float32)},
                parents[left:right], clocks[left:right])

    def evaluate(actual, features, s_bound, f_bound, parents, clocks, **kwargs):
        assert actual is model and (s_bound, f_bound) == (endpoint_bound, future_bound)
        assert not any(module.training for _, module in model.named_modules())
        assert kwargs == dict(fold_id=1, partition=partition, future_weight=1.0,
                              frozen_priors=frozen_priors)
        assert features['seq_x'].shape == (len(parents), 1)
        calls.append((parents.copy(), clocks.copy()))
        if fail == 'mutation':
            model.value.add_(1)
        return {'measurement': _batch_measurement(parents, clocks, partition)}

    monkeypatch.setattr(runner, '_feature_batch', features)
    monkeypatch.setattr(batch, 'evaluate_endpoint_batch', evaluate)
    if fail:
        ticks = iter((0.0, 1000.0))
        monkeypatch.setattr(runner, 'time', SimpleNamespace(monotonic=lambda: next(ticks)))
    else:
        monkeypatch.setattr(runner, 'time', SimpleNamespace(monotonic=lambda: 0.0))
    counters = {'forward_calls': 0}
    args = dict(model=model, prepared={}, view={}, receipts={},
                rows=rows, issues=issues, partition=partition,
                endpoint_bound=endpoint_bound, future_bound=future_bound,
                frozen_priors=frozen_priors, device=torch.device('cpu'),
                deadline=1000.0, counters=counters)
    return args, model, calls, counters


@pytest.mark.parametrize('partition,expected_rows,tail', [
    ('fit', 29574, 6), ('score', 17753, 9),
])
def test_evaluation_visits_whole_exact_axis_and_restores_modes_and_rng(
        monkeypatch, partition, expected_rows, tail):
    args, model, calls, counters = _fake_evaluation(monkeypatch, partition=partition)
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng = torch.random.get_rng_state().clone()
    measured = runner._ta_endpoint_measurement(**args)
    assert measured['rows'] == expected_rows
    assert measured['batches'] == counters['forward_calls'] == len(calls)
    assert len(calls[-1][0]) == tail
    assert np.array_equal(np.concatenate([part[0] for part in calls]), args['rows'])
    assert np.array_equal(np.concatenate([part[1] for part in calls]), args['issues'])
    assert model.training is True and model.child.training is False
    assert torch.equal(model.value, torch.tensor([2.0]))
    assert random.getstate() == python_rng
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_rng[0] and current_numpy[2:] == numpy_rng[2:]
    assert np.array_equal(current_numpy[1], numpy_rng[1])
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    assert measured['scores']['model']['state']['M5']['activity']['count'] == expected_rows
    assert measured['means']['model']['state']['M5']['activity']['log'] == 0.5


@pytest.mark.parametrize('fail', ['deadline', 'mutation'])
def test_interrupted_evaluation_restores_modes_and_rejects_changed_state(monkeypatch, fail):
    args, model, calls, counters = _fake_evaluation(monkeypatch, partition='fit', fail=fail)
    expected = runner._ScoreWindowExhausted if fail == 'deadline' else RuntimeError
    match = None if fail == 'deadline' else 'TA_MEASUREMENT_MUTATED_MODEL'
    with pytest.raises(expected, match=match):
        runner._ta_endpoint_measurement(**args)
    assert counters['forward_calls'] == len(calls) == 1
    assert model.training is True and model.child.training is False


def test_evaluation_rejects_incomplete_declared_population_before_forward(monkeypatch):
    args, model, calls, counters = _fake_evaluation(monkeypatch, partition='fit')
    args['rows'], args['issues'] = args['rows'][:-1], args['issues'][:-1]
    with pytest.raises(RuntimeError, match='TA_MEASUREMENT_POPULATION'):
        runner._ta_endpoint_measurement(**args)
    assert calls == [] and counters['forward_calls'] == 0 and model.training is True


def _measurement():
    rows = np.arange(3, dtype=np.int64)
    batch_value = _batch_measurement(rows, rows + 1, 'fit')
    return objective.finalize_endpoint_measurement(
        objective.merge_endpoint_measurements(None, batch_value))


def test_matched_evaluation_allows_predictor_scores_to_change():
    initial = _measurement()
    final = copy.deepcopy(initial)
    final['scores']['model']['state']['M5']['activity']['log_sum'] += 1.0
    runner._ta_require_matched_observations(initial, final)


@pytest.mark.parametrize('fault,reason', [
    ('rows', 'TA_UNMATCHED_MEASUREMENT:rows'),
    ('batches', 'TA_UNMATCHED_MEASUREMENT:batches'),
    ('partition', 'TA_UNMATCHED_MEASUREMENT:partition'),
    ('mask_chain', 'TA_UNMATCHED_MEASUREMENT:observation_chain_sha256'),
    ('prior_identity', 'TA_UNMATCHED_MEASUREMENT:frozen_prior_contract_sha256'),
    ('support', 'TA_UNMATCHED_SUPPORT'),
    ('prior_scores', 'TA_CHANGED_FROZEN_PRIOR'),
    ('score_roles', 'TA_UNMATCHED_SCORE_ROLES'),
])
def test_matched_evaluation_rejects_changed_observations_or_prior(fault, reason):
    initial = _measurement()
    final = copy.deepcopy(initial)
    if fault in ('rows', 'batches'):
        final[fault] += 1
    elif fault == 'partition':
        final['partition'] = 'score'
    elif fault == 'mask_chain':
        final['observation_chain_sha256'] = 'e' * 64
    elif fault == 'prior_identity':
        final['frozen_prior_contract_sha256'] = 'e' * 64
    elif fault == 'support':
        final['scores']['model']['state']['M5']['conditional_side']['count'] -= 1
    elif fault == 'prior_scores':
        final['scores']['prior']['future']['D1']['activity']['log_sum'] += 1.0
    else:
        del final['scores']['prior']
    with pytest.raises(RuntimeError, match=reason):
        runner._ta_require_matched_observations(initial, final)


def test_evaluation_payload_keeps_checkpoint_and_predictive_claims_separate():
    measured = _measurement()
    checkpoint = {'path': '/tmp/synthetic-FINAL_FOLD1_FIT_STATE.pt', 'sha256': 'f' * 64}
    payload = runner._ta_measurement_payload(measured, checkpoint=checkpoint,
        model_state_sha256='e' * 64, optimizer_updates=1849)
    assert payload['measurement'] is measured
    assert payload['evaluated_checkpoint'] == checkpoint
    assert payload['evaluated_model_state_sha256'] == 'e' * 64
    assert payload['optimizer_updates'] == 1849
    assert payload['test_data_used'] is payload['predictive_value_claimed'] is False
    assert payload['realized_economics_claimed'] is False
    with pytest.raises(RuntimeError, match='TA_MEASUREMENT_PAYLOAD'):
        runner._ta_measurement_payload({}, checkpoint=checkpoint,
            model_state_sha256='e' * 64, optimizer_updates=1849)


def test_only_ta_final_state_can_resume_before_final_evaluation_exists():
    view = {'synthetic': True}
    prior = {
        'outcome': 'RESUMABLE', 'phase_state': 'SCORING',
        'parent_order_sha256': 'a' * 64, 'feature_view_identity': view,
        'oof_cursor': 0, 'parent_cursor': runner.scope.FIT_ROWS,
        'optimizer_updates': runner.scope.MAX_UPDATES,
        'final_train_measurement': None,
        'full_state': {'path': '/tmp/FINAL_FOLD1_FIT_STATE.pt'},
    }
    args = dict(order_sha='a' * 64, view_identity=view)
    assert runner._require_prior_cursor(prior, **args, ta_sf=True) is True
    with pytest.raises(RuntimeError, match='SCORE_ONLY_RESUME_FROZEN_FIT'):
        runner._require_prior_cursor(prior, **args)
    prior['full_state']['path'] = '/tmp/intermediate.pt'
    with pytest.raises(RuntimeError, match='SCORE_ONLY_RESUME_FROZEN_FIT'):
        runner._require_prior_cursor(prior, **args, ta_sf=True)


def test_ta_source_freeze_contains_loss_scorer_and_frozen_prior_owners():
    required = runner.scope.required_source_paths(ta_sf=True)
    assert {
        'gx1/contracts/market_state_time_query_loss_v2.py',
        'gx1/contracts/market_state_time_query_scorer_v2.py',
        'gx1/contracts/market_state_five_endpoint_prior_v1.py',
    } <= required
