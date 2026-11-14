from __future__ import annotations

import numpy as np
import pytest
import torch

from gx1.audit import verify_market_state_time_query_v2_zero_step_v1 as zero
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as state
from gx1.contracts import market_state_future_displacement_candidate_v1 as future
from gx1.contracts import market_state_time_query_binding_v1 as query
from gx1.contracts import market_state_time_query_native_F_batch_v1 as batch
from gx1.contracts.market_state_time_query_model_v2 import factorized_log_probabilities
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K


class RecordingModel:
    """Only the adapter protocol matters; targets must not enter forward."""
    def __init__(self):
        self.state_model = type('State', (), {'query_native_k': torch.tensor(NATIVE_K)})()
        self.research_objective = {'contract_sha256': 'synthetic'}
        self.keys = None

    def __call__(self, **kwargs):
        self.keys = set(kwargs)
        n = kwargs['seq_x'].shape[0]
        raw = torch.zeros((n, 5, 2), dtype=torch.float32, requires_grad=True)
        future = raw.clone()
        return {'state_activity_side_logits': raw,
                'state_log_probs': factorized_log_probabilities(raw),
                'future_activity_side_logits': future,
                'future_log_probs': factorized_log_probabilities(future)}


def test_feature_only_forward_on_actual_bound_train_rows(monkeypatch):
    # Synthetic forward keeps the expensive backbone out; target data are real.
    bound_state = query.load_bound_time_query_targets()
    bound_future = future.load_bound_future_displacement()
    axes = state.load_bound_five_endpoint_targets().mature_oof_axes(fold_id=1)
    rows = axes['fit_parent_rows'][:8]
    issues = axes['fit_decision_ns'][:8]
    # The batch owner explicitly requires its dual model class; patch only the
    # class identity here so this test can inspect the exact forward boundary.
    monkeypatch.setattr(batch, 'NativeFutureDualModel', RecordingModel)
    model = RecordingModel()
    features = {key: torch.zeros((8, 1), dtype=torch.float32)
                for key in batch.FEATURE_KEYS}
    features['seq_x'] = torch.zeros((8, 1, 4), dtype=torch.float32)
    out = batch.evaluate_fit_batch(
        model, features, bound_state, bound_future, rows, issues,
        fold_id=1, pass_index=zero.PASS_INDEX, seed=zero.SEED,
        future_weight=0.25)
    assert model.keys == set(batch.FEATURE_KEYS) | {'query_k'}
    assert out['state_count'] > 0 and out['future_count'] == 40
    assert out['future_native_k'].tolist() == list(NATIVE_K)
    assert out['state_query_k'][0].tolist() != list(NATIVE_K)
    assert bool(torch.isfinite(out['loss']))


class RecordingEndpointModel(torch.nn.Module):
    def __init__(self, profile_name=None):
        super().__init__()
        from types import SimpleNamespace
        self.marker = torch.nn.Parameter(torch.ones(1))
        self.state_model = SimpleNamespace(
            query_native_k=torch.tensor(NATIVE_K),
            feature_view_identity={'contract_sha256': 'synthetic-view'})
        self.backbone = SimpleNamespace(entry_execution_profile=(
            {'name': profile_name} if profile_name else None))
        self.research_objective = {'contract_sha256': 'synthetic-model'}
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append({key for key in kwargs})
        assert torch.equal(kwargs['query_k'], torch.tensor(
            [NATIVE_K]*len(kwargs['seq_x']), dtype=torch.int64))
        raw = torch.zeros((len(kwargs['seq_x']), 5, 2), dtype=torch.float32)
        return {'state_activity_side_logits': raw,
                'state_log_probs': factorized_log_probabilities(raw),
                'future_activity_side_logits': raw.clone(),
                'future_log_probs': factorized_log_probabilities(raw)}


def test_endpoint_eval_uses_native_queries_then_targets_and_rejects_train_mode(monkeypatch):
    import numpy as np
    import pytest
    from tests.test_market_state_time_query_native_F_objective import _endpoint_fixture
    from gx1.features.entry_specialist_feature_groups_v1 import (
        FULL_TA_ENTRY_EXECUTION_PROFILE_NAME, FULL_TA_ENTRY_FEATURE_KEYS)
    monkeypatch.setattr(batch, 'NativeFutureDualModel', RecordingEndpointModel)
    endpoints, f = _endpoint_fixture()
    axes = endpoints.mature_oof_axes(fold_id=1)
    model = RecordingEndpointModel(FULL_TA_ENTRY_EXECUTION_PROFILE_NAME)
    features = {key: torch.zeros((2, 1), dtype=torch.float32)
                for key in FULL_TA_ENTRY_FEATURE_KEYS}
    args = (model, features, endpoints, f, axes['fit_parent_rows'], axes['fit_decision_ns'])
    kw = dict(fold_id=1, partition='fit', future_weight=1.0)
    with pytest.raises(ValueError, match='REQUIRES_EVAL'):
        batch.evaluate_endpoint_batch(*args, **kw)
    assert not model.calls
    s_gather, f_gather = type(endpoints).gather, type(f).gather
    def after_s(self, *a, **k):
        assert model.calls
        return s_gather(self, *a, **k)
    def after_f(self, *a, **k):
        assert model.calls
        return f_gather(self, *a, **k)
    monkeypatch.setattr(type(endpoints), 'gather', after_s)
    monkeypatch.setattr(type(f), 'gather', after_f)
    model.eval()
    before = model.marker.detach().clone()
    result = batch.evaluate_endpoint_batch(*args, **kw)
    assert torch.equal(model.marker, before) and model.marker.grad is None
    assert len(model.calls) == 1
    assert model.calls[0] == set(FULL_TA_ENTRY_FEATURE_KEYS) | {'query_k'}
    assert result['state_activity_side_logits'].dtype == np.float32
    assert result['measurement']['rows'] == 2
    assert result['measurement']['native_query_k'] == list(NATIVE_K)
    bad = dict(features, gc_seq_m5=torch.zeros((2, 1)))
    with pytest.raises(ValueError, match='BATCH_INPUT'):
        batch.evaluate_endpoint_batch(model, bad, *args[2:], **kw)
    assert len(model.calls) == 1


def test_gc_batch_input_keys_keep_the_explicit_original_contract():
    from gx1.contracts import market_state_time_query_batch_v1 as original
    from gx1.contracts import market_state_time_query_batch_v2 as factorized
    from gx1.features.entry_specialist_feature_groups_v1 import FULL_TA_ENTRY_FEATURE_KEYS
    expected_gc = FULL_TA_ENTRY_FEATURE_KEYS | {
        'gc_seq_m5', 'gc_seq_m15', 'gc_seq_h1', 'gc_seq_h4', 'gc_seq_d1'}
    assert original.FEATURE_KEYS == factorized.FEATURE_KEYS == expected_gc
    assert len(expected_gc) == 14 and 'gc_features' not in factorized.FEATURE_KEYS


def _fold2_weighted_batch(monkeypatch, start, size):
    from tests.test_market_state_m5_equal_observation_binding import _batch_case
    events, args, kwargs = _batch_case(monkeypatch, 0, size)
    parents = np.arange(47338, dtype=np.int64) * 3 + 7
    issues = state.TRAIN_START_NS + (np.arange(47338, dtype=np.int64) + 1) * state.M5_NS
    for bound in args[2:4]:
        bound.parent_rows, bound.decision_ns = parents.copy(), issues.copy()
    args[4], args[5] = parents[start:start + size].copy(), issues[start:start + size].copy()
    args[0].state_model._dedicated_tf_activity = True
    kwargs.update(fold_id=2, fit_batch_start=start)
    return events, args, kwargs


@pytest.mark.parametrize('start,size', [(16, 16), (47328, 10)])
def test_fold2_weighted_batch_and_tail_reach_own_loss_binding(monkeypatch, start, size):
    from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
    events, args, kwargs = _fold2_weighted_batch(monkeypatch, start, size)
    result = batch.evaluate_fit_batch(*args, **kwargs)
    assert events == ['forward', 'S-targets', 'F-targets']
    assert result['state_activity_side_logits'].shape == (size, 5, 2)
    assert result['loss_contract_sha256'] == objective.contract(
        ta_only=True, dedicated_m5_side=True, dedicated_tf_activity=True,
        m5_equal_observation=True, fold_id=2)['contract_sha256']
    # Zero logits, all known-UP: ten activity terms, eight ordinary side terms,
    # and size times the two distinct measured M5 coefficients, all divided by five.
    expected = np.log(2) * (18 + size * (2890 / 16101 + 2950 / 19072)) / 5
    assert float(result['loss']) == pytest.approx(expected, rel=1e-6)
    assert np.all(result['state_query_k'][:, 0] == 1)


def test_fold2_explicit_positions_accept_own_rows_beyond_fold1(monkeypatch):
    events, args, kwargs = _fold2_weighted_batch(monkeypatch, 0, 16)
    positions = np.array([47337, 29574, 0, 33000, 5, 40000, 7, 100,
                          47200, 9, 11, 13, 15, 17, 19, 21], np.int64)
    args[4] = args[2].parent_rows[positions].copy()
    args[5] = args[2].decision_ns[positions].copy()
    kwargs.pop('fit_batch_start')
    kwargs['fit_batch_positions'] = positions
    result = batch.evaluate_fit_batch(*args, **kwargs)
    assert events == ['forward', 'S-targets', 'F-targets']
    np.testing.assert_array_equal(result['state_query_k'], query.sampled_query_k(
        args[5], pass_index=0, seed=20260911))


@pytest.mark.parametrize('fault', [
    'fold1_tail', 'unsupported_fold', 'bool_fold', 'early_tail', 'past_end',
    'wrong_start', 'future_parent', 'state_issue', 'duplicate_positions', 'wrong_positions',
])
def test_fold2_bad_batch_binding_rejects_before_forward_or_targets(monkeypatch, fault):
    tail = fault == 'fold1_tail'
    size = 10 if tail or fault == 'early_tail' else 16
    events, args, kwargs = _fold2_weighted_batch(monkeypatch, 47328 if tail else 0, size)
    if fault == 'fold1_tail':
        kwargs['fold_id'] = 1
    elif fault == 'unsupported_fold':
        kwargs['fold_id'] = 3
    elif fault == 'bool_fold':
        kwargs['fold_id'] = True
    elif fault == 'past_end':
        kwargs['fit_batch_start'] = 47344
    elif fault == 'wrong_start':
        kwargs['fit_batch_start'] = 16
    elif fault == 'future_parent':
        args[3].parent_rows[0] += 1
    elif fault == 'state_issue':
        args[2].decision_ns[0] += 1
    elif fault in ('duplicate_positions', 'wrong_positions'):
        kwargs.pop('fit_batch_start')
        positions = np.arange(16, dtype=np.int64)
        positions[-1] = 0 if fault == 'duplicate_positions' else 40000
        kwargs['fit_batch_positions'] = positions
    with pytest.raises(ValueError, match='NATIVE_F_(M5_EQUAL_OBSERVATION|BATCH_INPUT)'):
        batch.evaluate_fit_batch(*args, **kwargs)
    assert not events
