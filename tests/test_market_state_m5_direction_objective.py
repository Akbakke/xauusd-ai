"""Exact existing M5 components; synthetic tensors only, no market payload."""
import numpy as np
import pytest
import torch

from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
from gx1.contracts import market_state_time_query_native_F_batch_v1 as batch
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K

PROFILE = dict(ta_only=True, dedicated_m5_side=True, m5_equal_observation=True,
               dedicated_tf_activity=True)
FROZEN_FULL_LOSS_SHA = 'ee227c6c07868520dce9fd3ee7fab536353b447e80b6cf4939cbc8ebfb69eb76'


def case(n=16):
    bits = np.tile(np.array([1, 4, 5, 2, 7], np.uint8), (n, 1))
    bits[:, 0] = np.resize(np.array([1, 4, 5, 2, 7, 1, 4], np.uint8), n)
    eligible = np.ones((n, 5), bool)
    mature = np.ones((n, 5), bool)
    eligible[min(n-1, 5), 0] = False
    mature[min(n-1, 6), 0] = False
    truths = [dict(allowed=bits.copy(), eligible=eligible.copy(), mature=mature.copy())
              for _ in range(2)]
    truths[1]['allowed'] = np.where(bits == 1, 4, np.where(bits == 4, 1, bits)).astype(np.uint8)
    s = torch.linspace(-2, 2, n*10).reshape(n, 5, 2).requires_grad_()
    f = torch.linspace(1.5, -1.5, n*10).reshape(n, 5, 2).requires_grad_()
    args = dict(state_raw=s, future_raw=f, state_truth=truths[0], future_truth=truths[1],
                state_query_k=torch.tensor([NATIVE_K]*n), future_weight=1.0, **PROFILE)
    return (s, f), truths, args


@pytest.mark.parametrize('n', [6, 16])
@pytest.mark.parametrize('future_weight', [0.5, 1.0])
def test_only_original_m5_derivatives_survive_with_exact_coefficients(n, future_weight):
    raws, truths, args = case(n)
    args['future_weight'] = future_weight
    full = objective.combined_loss(**args)
    selected = objective.combined_loss(**args, m5_direction_only=True)
    fg = torch.autograd.grad(full['loss'], raws)
    mg = torch.autograd.grad(selected['loss'], raws)
    expected_loss = 0.0
    for index, (raw, truth, original, chosen, ratio) in enumerate(zip(
            raws, truths, fg, mg, (1799/10010/5, 1844/11909/5*future_weight))):
        known = truth['eligible'][:, 0] & truth['mature'][:, 0] & np.isin(truth['allowed'][:, 0], (1, 4))
        target = torch.from_numpy((truth['allowed'][:, 0] == 4).astype(np.float32))
        expected = (raw.detach()[:, 0, 1].sigmoid()-target) * ratio
        expected[~known] = 0
        assert torch.allclose(chosen[:, 0, 1], expected, atol=4e-9, rtol=2e-7)
        assert torch.equal(chosen[:, 0, 1], original[:, 0, 1])
        assert torch.count_nonzero(chosen[..., 0]) == 0
        assert torch.count_nonzero(chosen[:, 1:, :]) == 0
        z = raw.detach()[:, 0, 1].double()
        cells = torch.logaddexp(torch.zeros_like(z), z) - z * target.double()
        expected_loss += float(cells[known].sum()) * ratio
    assert float(selected['loss'].detach()) == pytest.approx(expected_loss, abs=2e-7)
    assert selected['skip_update'] is False
    assert selected['contract_sha256'] != full['contract_sha256']


def test_no_m5_direction_cannot_train_from_other_tasks_or_imputed_labels():
    raws, truths, args = case()
    for truth in truths:
        truth['allowed'][:, 0] = 5
    full = objective.combined_loss(**args)
    selected = objective.combined_loss(**args, m5_direction_only=True)
    assert full['skip_update'] is False and float(full['loss'].detach()) > 0
    assert selected['skip_update'] is True and float(selected['loss'].detach()) == 0
    assert all(torch.count_nonzero(g) == 0 for g in torch.autograd.grad(selected['loss'], raws))


def test_full_default_contract_remains_frozen_and_forward_identity_is_shared():
    full = objective.contract(**PROFILE)
    selected = objective.contract(**PROFILE, m5_direction_only=True)
    assert full['contract_sha256'] == FROZEN_FULL_LOSS_SHA
    assert full == objective.contract(**PROFILE, m5_direction_only=False)
    for key in ('dual_model_contract_sha256', 'S_loss_contract_sha256',
                'F_loss_contract_sha256', 'F_target_contract_sha256',
                'F_score_contract_sha256', 'm5_side_weighting_profile'):
        assert selected[key] == full[key]


@pytest.mark.parametrize('value', [None, 0, 1, 'true'])
def test_nonboolean_mode_rejected(value):
    with pytest.raises(ValueError, match='M5_DIRECTION_ONLY_MODE'):
        objective.contract(**PROFILE, m5_direction_only=value)
    _, _, args = case()
    with pytest.raises(ValueError, match='M5_DIRECTION_ONLY_MODE'):
        objective.combined_loss(**args, m5_direction_only=value)


def test_direction_only_requires_original_equal_observation_profile():
    with pytest.raises(ValueError, match='M5_DIRECTION_ONLY_MODE'):
        objective.contract(m5_direction_only=True)


def test_batch_adapter_preserves_feature_only_forward_and_selects_owned_loss(monkeypatch):
    from tests.test_market_state_m5_equal_observation_binding import _batch_case
    events, args, kwargs = _batch_case(monkeypatch, 16, 16)
    result = batch.evaluate_fit_batch(*args, **kwargs, m5_direction_only=True)
    assert events == ['forward', 'S-targets', 'F-targets']
    assert result['loss_contract_sha256'] == objective.contract(
        ta_only=True, dedicated_m5_side=True, m5_equal_observation=True,
        m5_direction_only=True)['contract_sha256']
    assert result['skip_update'] is False


@pytest.mark.parametrize('value,equal', [(1, True), (True, False)])
def test_batch_rejects_unbound_mode_before_forward(monkeypatch, value, equal):
    from tests.test_market_state_m5_equal_observation_binding import _batch_case
    events, args, kwargs = _batch_case(monkeypatch, 16, 16)
    kwargs.update(m5_direction_only=value, m5_equal_observation=equal)
    with pytest.raises(ValueError, match='M5_DIRECTION_ONLY_MODE'):
        batch.evaluate_fit_batch(*args, **kwargs)
    assert events == []


def test_numerical_binding_changes_only_the_selected_training_objective():
    from gx1.contracts.market_state_time_query_fold1_fit_v1 import ta_sf_numerical_recipe
    args = {k: v for k, v in PROFILE.items() if k != 'ta_only'}
    full = ta_sf_numerical_recipe(**args)
    selected = ta_sf_numerical_recipe(**args, m5_direction_only=True)
    assert full['native_F_loss_contract_sha256'] == FROZEN_FULL_LOSS_SHA
    changed = {'m5_direction_only', 'native_F_loss_contract_sha256', 'contract_sha256'}
    assert {k: v for k, v in full.items() if k not in changed} == {
        k: v for k, v in selected.items() if k not in changed}
    assert selected['native_F_loss_contract_sha256'] == objective.contract(
        **PROFILE, m5_direction_only=True)['contract_sha256']


def test_zero_step_rebinding_preserves_weights_optimizer_and_rng(monkeypatch):
    from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
    from gx1.contracts.market_state_time_query_fold1_fit_v1 import ta_sf_numerical_recipe
    from tests.test_market_state_ta_dedicated_m5_binding_v1 import _model, _seed, _capture, ORDER, DEVICE, COMMIT
    _seed()
    model = _model(monkeypatch, True)
    args = dict(dedicated_m5_side=True, m5_equal_observation=True)
    full = ta_sf_numerical_recipe(**args)
    selected = ta_sf_numerical_recipe(**args, m5_direction_only=True)
    optimizer, _ = state.build_fold1_fit_optimizer(model, numerical_recipe=full)
    old = _capture(model, optimizer, full, 0)
    new = _capture(model, optimizer, selected, 0)
    for key in ('model_state_sha256', 'buffer_state_sha256', 'optimizer_state_sha256',
                'rng_state_sha256', 'module_training', 'cursor', 'optimizer_updates', 'skipped_batches'):
        assert old[key] == new[key]
    assert not optimizer.state
    assert old['binding'] != new['binding']
    kwargs = dict(feature_view_identity=model.state_model.feature_view_identity,
                  parent_order=ORDER, source_commit=COMMIT, device=DEVICE)
    with pytest.raises(RuntimeError):
        state.restore_fold1_fit_state(old, model, optimizer, numerical_recipe=selected, **kwargs)
    result = state.restore_fold1_fit_state(new, model, optimizer, numerical_recipe=selected, **kwargs)
    assert result == dict(cursor=0, optimizer_updates=0, skipped_batches=0)
