"""Focused query-loss semantics and endpoint score parity, no model fit."""
import math

import numpy as np
import pytest
import torch

from gx1.contracts import market_state_five_endpoint_contract_v1 as old
from gx1.contracts import market_state_time_query_loss_v1 as new


def _k_torch(n=1):
    return torch.tensor([new.K] * n, dtype=torch.int64)


def _k_numpy(n=1):
    return np.array([new.K] * n, dtype=np.int64)


def _one_known_loss(probabilities: tuple[float, float, float], bit: int) -> float:
    logits = torch.zeros((1, 5, 3), dtype=torch.float32)
    logits[0, 0] = torch.log(torch.tensor(probabilities, dtype=torch.float32))
    allowed = torch.full((1, 5), 7, dtype=torch.uint8)
    allowed[0, 0] = bit
    ready = torch.ones((1, 5), dtype=torch.bool)
    return float(new.allowed_set_loss(logits, allowed, ready, ready,
                                      _k_torch())['loss'])


def test_active_unknown_does_not_prefer_a_side_at_fixed_activity_mass():
    first = _one_known_loss((0.4, 0.2, 0.4), 5)
    second = _one_known_loss((0.7, 0.2, 0.1), 5)
    assert first == pytest.approx(-math.log(0.8) / 5, abs=2e-7)
    assert second == pytest.approx(first, abs=2e-7)


def test_unknown_and_immature_cells_have_zero_gradient_and_skip():
    logits = torch.randn((1, 5, 3), dtype=torch.float32, requires_grad=True)
    allowed = torch.full((1, 5), 7, dtype=torch.uint8)
    ready = torch.ones((1, 5), dtype=torch.bool)
    result = new.allowed_set_loss(logits, allowed, ready, ready, _k_torch())
    assert result['count'] == 0 and result['skip_update'] is True
    result['loss'].backward()
    assert torch.count_nonzero(logits.grad) == 0

    logits.grad.zero_()
    allowed[0, 0] = 1
    mature = ready.clone()
    mature[0, 0] = False
    result = new.allowed_set_loss(logits, allowed, ready, mature, _k_torch())
    assert result['count'] == 0 and result['skip_update'] is True
    result['loss'].backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_equal_tf_weighting_uses_mean_within_tf_and_fixed_divisor_five():
    logits = torch.zeros((2, 5, 3), dtype=torch.float32)
    logits[0, 0] = torch.log(torch.tensor((0.1, 0.2, 0.7)))
    logits[1, 0] = torch.log(torch.tensor((0.3, 0.4, 0.3)))
    logits[0, 2] = torch.log(torch.tensor((0.2, 0.5, 0.3)))
    allowed = torch.full((2, 5), 7, dtype=torch.uint8)
    allowed[0, 0], allowed[1, 0], allowed[0, 2] = 1, 2, 4
    ready = torch.ones((2, 5), dtype=torch.bool)
    result = new.allowed_set_loss(logits, allowed, ready, ready, _k_torch(2))
    expected = ((-math.log(0.1) - math.log(0.4)) / 2 - math.log(0.3)) / 5
    assert float(result['loss']) == pytest.approx(expected, abs=2e-7)
    assert result['per_tf_counts'] == {'M5': 2, 'M15': 0, 'H1': 1,
                                       'H4': 0, 'D1': 0}
    assert result['skip_update'] is False


def test_endpoint_scores_reuse_proper_math_with_new_identity():
    probabilities = np.array([
        [[.2, .3, .5], [.1, .8, .1], [.3, .2, .5], [.4, .2, .4], [.6, .2, .2]],
        [[.4, .1, .5], [.3, .4, .3], [.2, .7, .1], [.2, .5, .3], [.1, .2, .7]],
    ], dtype=np.float64)
    allowed = np.array([[1, 5, 4, 2, 7], [5, 2, 7, 1, 4]], dtype=np.uint8)
    ready = np.ones((2, 5), dtype=bool)
    mature = ready.copy()
    mature[1, 3] = False
    previous = old.proper_score_sums(probabilities, allowed, ready, mature)
    got = new.proper_score_sums(probabilities, allowed, ready, mature,
                                _k_numpy(2))
    assert got['per_tf'] == previous['per_tf']
    assert got['contract_sha256'] != previous['contract_sha256']
    assert got['per_tf']['M15']['conditional_side']['count'] == 0
    assert got['per_tf']['M15']['activity']['count'] == 2


def test_query_domain_rejects_bad_offset_before_loss_or_score():
    logits = torch.zeros((1, 5, 3), dtype=torch.float32)
    allowed = torch.full((1, 5), 7, dtype=torch.uint8)
    ready = torch.ones((1, 5), dtype=torch.bool)
    bad = _k_torch()
    bad[0, 0] = 2
    with pytest.raises(ValueError, match='TIME_QUERY_K'):
        new.allowed_set_loss(logits, allowed, ready, ready, bad)
    probabilities = np.full((1, 5, 3), 1 / 3, dtype=np.float64)
    with pytest.raises(ValueError, match='TIME_QUERY_K'):
        new.proper_score_sums(probabilities, allowed.numpy(), ready.numpy(),
                              ready.numpy(), bad.numpy())


def test_unknown_and_immature_cells_do_not_backpropagate_with_valid_neighbor():
    logits = torch.randn((2, 5, 3), dtype=torch.float32, requires_grad=True)
    allowed = torch.full((2, 5), 7, dtype=torch.uint8)
    allowed[0, 0] = 1
    allowed[1, 1] = 4
    ready = torch.ones((2, 5), dtype=torch.bool)
    mature = ready.clone()
    mature[1, 1] = False
    result = new.allowed_set_loss(logits, allowed, ready, mature, _k_torch(2))
    assert result['count'] == 1 and result['skip_update'] is False
    result['loss'].backward()
    assert torch.count_nonzero(logits.grad[0, 0]) > 0
    assert torch.count_nonzero(logits.grad[1]) == 0
    assert torch.count_nonzero(logits.grad[0, 1:]) == 0
