import numpy as np
import torch

from gx1.contracts import market_state_five_endpoint_target_binding_v1 as state
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
from gx1.contracts import market_state_time_query_loss_v2 as legacy_loss
from gx1.contracts import market_state_time_query_side_normalized_loss_v1 as state_loss
from gx1.contracts.market_state_future_displacement_candidate_v1 import BoundFutureDisplacement
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K


def test_two_losses_keep_F_flat_side_gradient_zero_and_S_independent():
    s = torch.zeros((1, 5, 2), dtype=torch.float32, requires_grad=True)
    f = torch.zeros((1, 5, 2), dtype=torch.float32, requires_grad=True)
    state_truth = {'allowed': np.array([[5, 1, 4, 2, 7]], dtype=np.uint8),
                   'eligible': np.array([[True, True, True, True, False]]),
                   'mature': np.ones((1, 5), bool)}
    future_truth = {'allowed': np.array([[4, 2, 1, 7, 2]], dtype=np.uint8),
                    'eligible': np.array([[True, True, True, False, True]]),
                    'mature': np.ones((1, 5), bool)}
    q = torch.tensor([NATIVE_K], dtype=torch.int64)
    result = objective.combined_loss(state_raw=s, future_raw=f,
                                     state_truth=state_truth, future_truth=future_truth,
                                     state_query_k=q, future_weight=1.0)
    result['loss'].backward()
    assert result['state']['count'] == 4 and result['future']['count'] == 4
    assert s.grad[0, 0, 1] == 0 and s.grad[0, 1, 1] != 0
    assert f.grad[0, 0, 1] != 0 and f.grad[0, 2, 1] != 0
    assert torch.equal(f.grad[0, [1, 3, 4], 1], torch.zeros(3))
    assert result['skip_update'] is False


def test_native_F_score_uses_earlier_prefix_and_partitions_UTC_week():
    fit = np.array([state.TRAIN_START_NS + 2*state.M5_NS,
                    state.TRAIN_START_NS + 3*state.M5_NS], np.int64)
    score = np.array([state.FOLD_BOUNDARIES_NS[0] + state.M5_NS,
                      state.FOLD_BOUNDARIES_NS[0] + 2*state.M5_NS], np.int64)
    issues = np.r_[fit, score]
    parents = np.arange(4, dtype=np.int64)
    sign = np.array([[1]*5,[-1]*5,[1]*5,[-1]*5],dtype=np.int8)
    mag = np.ones((4,5),dtype=np.float64)
    valid = np.ones((4,5),dtype=bool)
    bound = BoundFutureDisplacement(parents,issues,valid,sign,mag,
                                    np.zeros((4,5),dtype=bool))
    axes = {'fit_parent_rows':parents[:2], 'fit_decision_ns':fit,
            'score_parent_rows':parents[2:], 'score_decision_ns':score,
            'fit_parent_rows_sha256':'fit','score_parent_rows_sha256':'score'}
    raw = np.zeros((2,5,2), dtype=np.float32)
    report=objective.score_native_F_fold(bound=bound,axes=axes,fold_id=1,
                                         score_raw=raw)
    assert len(report['by_utc_week']) == 1
    for tf in ('M5','M15','H1','H4','D1'):
        assert report['overall'][tf]['activity']['count']==2
        assert report['overall'][tf]['conditional_side']['count']==2
        assert report['overall'][tf]['conditional_side']['prior_minus_model_brier_sum']==0


def test_side_normalization_uses_only_known_side_rows_and_preserves_old_loss():
    allowed = torch.tensor([[1, 7, 7, 7, 7],
                            [5, 7, 7, 7, 7],
                            [2, 7, 7, 7, 7],
                            [4, 7, 7, 7, 7]], dtype=torch.uint8)
    eligible = torch.ones((4, 5), dtype=torch.bool)
    mature = torch.ones((4, 5), dtype=torch.bool)
    query = torch.tensor([NATIVE_K] * 4, dtype=torch.int64)
    raw = torch.zeros((4, 5, 2), dtype=torch.float32, requires_grad=True)
    old = legacy_loss.allowed_set_loss(raw, allowed, eligible, mature, query)
    balanced = state_loss.allowed_set_loss(
        raw, allowed, eligible, mature, query)
    assert torch.isclose(old['loss'], torch.tensor(1.5 * np.log(2) / 5, dtype=torch.float32))
    assert torch.isclose(balanced['loss'], torch.tensor(2 * np.log(2) / 5, dtype=torch.float32))
    assert balanced['contract_sha256'] == state_loss.contract()['contract_sha256']
    assert old['contract_sha256'] == legacy_loss.factorized_time_query_loss_contract()['contract_sha256']
    balanced['loss'].backward()
    assert raw.grad[1, 0, 1] == raw.grad[2, 0, 1] == 0
    assert torch.isclose(raw.grad[0, 0, 1], torch.tensor(0.05))
    assert torch.isclose(raw.grad[3, 0, 1], torch.tensor(-0.05))
    assert torch.isclose(raw.grad[0, 0, 0], torch.tensor(-0.025))
    assert raw.grad[0, 1:, :].abs().sum() == 0


def _endpoint_fixture():
    fit = np.array([state.TRAIN_START_NS + i * state.M5_NS for i in (2, 3)], np.int64)
    later = np.array([state.FOLD_BOUNDARIES_NS[0] + i * state.M5_NS for i in (1, 2)], np.int64)
    issues = np.r_[fit, later]
    parents = np.arange(4, dtype=np.int64)
    bits = np.array([[1]*5, [4]*5, [5]*5, [2]*5], dtype=np.uint8)
    endpoints = state.BoundFiveEndpointTargets(
        parents, issues, bits, np.zeros((4, 5), np.int64),
        np.arange(4, dtype=np.int64), 0)
    valid = np.ones((4, 5), dtype=bool)
    sign = np.array([[1]*5, [-1]*5, [1]*5, [-1]*5], dtype=np.int8)
    future = BoundFutureDisplacement(
        parents, issues, valid, sign, np.ones((4, 5), np.float64),
        np.zeros((4, 5), dtype=bool))
    return endpoints, future


def _endpoint_measurement(raw_s, raw_f, bits_s, bits_f, indices=None):
    n = len(raw_s)
    parents = np.arange(n, dtype=np.int64)
    issues = state.TRAIN_START_NS + (parents + 2) * state.M5_NS
    validity = issues[:, None] + np.asarray(state.LEAD_MINUTES)[None, :] * state.MINUTE_NS
    def truth(bits):
        return {'parent_rows': parents, 'issue_ns': issues, 'validity_ns': validity,
                'allowed': bits, 'eligible': np.ones((n, 5), dtype=bool),
                'mature': np.ones((n, 5), dtype=bool)}
    s, f = truth(bits_s), truth(bits_f)
    if indices is not None:
        parents, issues = parents[indices], issues[indices]
        s = {k: v[indices] for k, v in s.items()}
        f = {k: v[indices] for k, v in f.items()}
        raw_s, raw_f = raw_s[indices], raw_f[indices]
    return objective.endpoint_measurement_sums(
        state_raw=raw_s, future_raw=raw_f, state_truth=s, future_truth=f,
        parent_rows=parents, issue_ns=issues, fold_id=1, partition='fit',
        future_weight=1.0, model_contract_sha256='model', feature_view_sha256='view')


def test_endpoint_global_denominators_are_independent_of_unequal_batch_support():
    raw = np.zeros((4, 5, 2), dtype=np.float32)
    state_bits = np.array([[1]*5, [5]*5, [2]*5, [4]*5], dtype=np.uint8)
    future_bits = np.array([[2]*5, [2]*5, [4]*5, [7]*5], dtype=np.uint8)
    whole = objective.finalize_endpoint_measurement(
        objective.merge_endpoint_measurements(None,
            _endpoint_measurement(raw, raw, state_bits, future_bits)))
    accumulated = None
    for part in (slice(0, 1), slice(1, 4)):
        accumulated = objective.merge_endpoint_measurements(
            accumulated, _endpoint_measurement(raw, raw, state_bits, future_bits, part))
    split = objective.finalize_endpoint_measurement(accumulated)
    assert whole['scores'] == split['scores']
    assert whole['endpoint_loss'] == split['endpoint_loss']
    assert np.isclose(whole['endpoint_loss']['model']['combined'], 4*np.log(2))
    assert whole['scores']['model']['state']['M5']['activity']['count'] == 4
    assert whole['scores']['model']['state']['M5']['conditional_side']['count'] == 2
    assert whole['scores']['model']['future']['M5']['conditional_side']['count'] == 1


def test_endpoint_priors_are_frozen_on_exact_mature_fit_and_masks_survive():
    import pytest
    endpoints, future = _endpoint_fixture()
    axes = endpoints.mature_oof_axes(fold_id=1)
    frozen = objective.fit_endpoint_priors(endpoints, future, axes=axes, fold_id=1)
    assert objective.require_endpoint_priors(frozen, fold_id=1) == frozen
    with pytest.raises(ValueError, match='NATIVE_F_ENDPOINT_PRIOR_BINDING'):
        objective.require_endpoint_priors(frozen, fold_id=1, ta_only=True)
    assert frozen['fit_rows'] == 2
    assert frozen['fit_parent_rows_sha256'] == axes['fit_parent_rows_sha256']
    assert len(frozen['tasks']['state']['raw_logits']) == 5
    bad_axes = dict(axes, fit_parent_rows=axes['fit_parent_rows'][1:])
    with pytest.raises(ValueError, match='EXACT_FIT_AXIS'):
        objective.fit_endpoint_priors(endpoints, future, axes=bad_axes, fold_id=1)
    rows, issues = axes['score_parent_rows'], axes['score_decision_ns']
    s = endpoints.gather(rows, issues, fold_id=1, partition='score')
    f = future.gather(rows, issues, fold_id=1, partition='score')
    result = objective.endpoint_measurement_sums(
        state_raw=np.zeros((2, 5, 2), np.float32),
        future_raw=np.zeros((2, 5, 2), np.float32),
        state_truth=s, future_truth=f, parent_rows=rows, issue_ns=issues,
        fold_id=1, partition='score', future_weight=1.0,
        model_contract_sha256='model', feature_view_sha256='view', frozen_priors=frozen)
    final = objective.finalize_endpoint_measurement(
        objective.merge_endpoint_measurements(None, result))
    assert final['scores']['model']['state']['M5']['activity']['count'] == 2
    assert final['scores']['model']['state']['M5']['conditional_side']['count'] == 0
    assert 'conditional_side' not in final['means']['model']['state']['M5']
    assert final['scores']['prior']['state']['M5']['activity']['count'] == 2
    forged = dict(frozen, fit_rows=3)
    with pytest.raises(ValueError, match='PRIOR_BINDING'):
        objective._endpoint_prior_arrays(forged, 1)


def test_endpoint_merge_rejects_changed_scope_and_observation_hash_tracks_mask():
    import pytest
    raw = np.zeros((2, 5, 2), np.float32)
    bits = np.ones((2, 5), np.uint8)
    original = _endpoint_measurement(raw, raw, bits, bits)
    changed_bits = bits.copy()
    changed_bits[0, 0] = 5
    changed = _endpoint_measurement(raw, raw, changed_bits, bits)
    assert changed['observation_sha256'] != original['observation_sha256']
    for key, value in [('partition', 'score'), ('future_weight', .5),
                       ('feature_view_sha256', 'other'),
                       ('loss_contract_sha256', 'other')]:
        total = objective.merge_endpoint_measurements(None, original)
        with pytest.raises(ValueError, match='MERGE_BINDING'):
            objective.merge_endpoint_measurements(total, dict(original, **{key: value}))
