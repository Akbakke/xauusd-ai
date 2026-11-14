import numpy as np
import pytest
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


# The fold1 profile is frozen; fold2 is bound to its target-only measurement.
def test_measured_fold2_profile_preserves_fold1_and_binds_own_counts():
    first = objective.m5_equal_observation_profile()
    assert first == objective.m5_equal_observation_profile(fold_id=1)
    assert first['contract_sha256'] == 'c11930254fe54d89ccfb5594aac90e1f599674db1f1bd7fd7c1b5732e6476a04'
    second = objective.m5_equal_observation_profile(fold_id=2)
    assert (second['fit_rows'], second['fit_batches'], second['batch_size']) == (47338, 2959, 16)
    assert second['state'] == dict(known_side_rows=16101, nonempty_batches=2890)
    assert second['future'] == dict(known_side_rows=19072, nonempty_batches=2950)
    assert second['target_count_result']['sha256'] == 'ba50eb96f1a663ec8829554d67364d03a2a9cc642a4807554f614a47af645131'
    assert second['target_count_validation']['sha256'] == 'dd45bd05dfc7e4b5f081999303869c93101e2fa6daf1f2da28b6dda87ca31bdb'
    assert not {'fit_cache_sha256', 'weight_result_sha256', 'weight_verification_sha256'} & second.keys()
    kwargs = dict(ta_only=True, dedicated_m5_side=True, dedicated_tf_activity=True)
    default = objective.contract(**kwargs, m5_equal_observation=True)
    assert default == objective.contract(**kwargs, m5_equal_observation=True, fold_id=1)
    own = objective.contract(**kwargs, m5_equal_observation=True, fold_id=2)
    changed = {'contract_sha256', 'm5_side_weighting_profile',
               'S_loss_contract_sha256', 'F_loss_contract_sha256'}
    assert {k: v for k, v in own.items() if k not in changed} == {
        k: v for k, v in default.items() if k not in changed}
    assert own['m5_side_weighting_profile'] == second
    assert own['contract_sha256'] != default['contract_sha256']


@pytest.mark.parametrize('fold_id', [1, 2, 3, 4, 5])
def test_unweighted_objective_keeps_existing_identity_for_all_folds(fold_id):
    kwargs = dict(ta_only=True, dedicated_m5_side=True, dedicated_tf_activity=True)
    assert objective.contract(**kwargs, fold_id=fold_id) == objective.contract(**kwargs)


@pytest.mark.parametrize('fold_id', [True, False, 0, 3, 5, '2', 2.0, np.int64(2)])
def test_only_exact_measured_fold_ids_admit_m5_weighting(fold_id):
    with pytest.raises(ValueError, match='M5_EQUAL_OBSERVATION_FOLD'):
        objective.m5_equal_observation_profile(fold_id=fold_id)
    with pytest.raises(ValueError, match='M5_EQUAL_OBSERVATION_FOLD'):
        objective.contract(ta_only=True, dedicated_m5_side=True,
                           m5_equal_observation=True, fold_id=fold_id)


@pytest.mark.parametrize('size', [16, 10])
def test_fold2_loss_uses_own_coefficients_and_preserves_other_gradients(size):
    allowed = np.tile(np.array([4, 1, 4, 2, 5], np.uint8), (size, 1))
    allowed[:7, 0] = [1, 4, 2, 5, 7, 4, 1]
    eligible, mature = np.ones((size, 5), bool), np.ones((size, 5), bool)
    mature[5, 0] = False
    eligible[6, 0] = False
    truth = dict(allowed=allowed, eligible=eligible, mature=mature)
    s = torch.linspace(-1, 1, size * 10).reshape(size, 5, 2).requires_grad_()
    f = (-s.detach()).clone().requires_grad_()
    kwargs = dict(state_raw=s, future_raw=f, state_truth=truth, future_truth=truth,
                  state_query_k=torch.tensor([NATIVE_K] * size), future_weight=0.375,
                  ta_only=True, dedicated_m5_side=True, dedicated_tf_activity=True,
                  fold_id=2)
    weighted = objective.combined_loss(**kwargs, m5_equal_observation=True)
    ordinary = objective.combined_loss(**kwargs)
    weighted_grad = torch.autograd.grad(weighted['loss'], (s, f))
    ordinary_grad = torch.autograd.grad(ordinary['loss'], (s, f))
    known = torch.from_numpy(eligible[:, 0] & mature[:, 0] & np.isin(allowed[:, 0], (1, 4)))
    target = torch.from_numpy(allowed[:, 0] == 4).to(torch.float32)
    for raw, actual, old, coefficient in zip(
            (s, f), weighted_grad, ordinary_grad, (2890 / 16101 / 5, 0.375 * 2950 / 19072 / 5)):
        expected = torch.where(known, (raw[:, 0, 1].detach().sigmoid() - target) * coefficient,
                               torch.zeros(size))
        torch.testing.assert_close(actual[:, 0, 1], expected, rtol=1e-6, atol=2e-9)
        assert torch.equal(actual[~known, 0, 1], torch.zeros(int((~known).sum())))
        assert torch.equal(actual[..., 0], old[..., 0])
        assert torch.equal(actual[:, 1:, :], old[:, 1:, :])
    assert weighted['contract_sha256'] == objective.contract(
        ta_only=True, dedicated_m5_side=True, dedicated_tf_activity=True,
        m5_equal_observation=True, fold_id=2)['contract_sha256']


def test_fold2_full_axis_binding_rejects_shape_dtype_order_and_clock_changes(monkeypatch):
    from gx1.audit.qualify_market_state_mature_prefix_normalization_v1 import _axis_sha
    parents = np.arange(47338, dtype=np.int64) * 3 + 7
    issues = state.TRAIN_START_NS + (np.arange(47338, dtype=np.int64) + 1) * state.M5_NS
    original = objective.m5_equal_observation_profile
    # Synthetic axes retain the measured dimensions; no genuine target payload is opened.
    profile = dict(original(fold_id=2), parent_axis_sha256=_axis_sha(parents),
                   issue_axis_sha256=_axis_sha(issues))
    def synthetic_profile(*, fold_id=1):
        return profile if fold_id == 2 else original(fold_id=fold_id)
    monkeypatch.setattr(objective, 'm5_equal_observation_profile', synthetic_profile)
    assert objective.require_m5_equal_observation_axes(parents, issues, fold_id=2) == profile
    for rows, clocks in ((parents[:-1], issues[:-1]), (parents.astype(np.int32), issues),
                         (parents[::-1], issues), (parents, issues + state.M5_NS)):
        with pytest.raises(ValueError, match='M5_EQUAL_OBSERVATION_FIT_AXES'):
            objective.require_m5_equal_observation_axes(rows, clocks, fold_id=2)
    with pytest.raises(ValueError, match='M5_EQUAL_OBSERVATION_FIT_AXES'):
        objective.require_m5_equal_observation_axes(parents, issues, fold_id=1)
