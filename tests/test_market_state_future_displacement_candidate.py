import numpy as np
import pytest
import torch

from gx1.contracts import market_state_five_endpoint_target_binding_v1 as state
from gx1.contracts import market_state_future_displacement_candidate_v1 as f
from gx1.contracts import market_state_time_query_loss_v2 as loss


def test_candidate_bits_keep_small_net_move_flat_and_missing_unknown():
    sign = np.array([[-1, 1, -1, 0, 0]], dtype=np.int8)
    magnitude = np.array([[0.49, 0.5, 1.0, 0.0, 0.0]], dtype=np.float64)
    valid = np.array([[True, True, True, False, True]])
    assert f.allowed_bits(sign, magnitude, valid).tolist() == [[2, 4, 1, 7, 2]]
    magnitude[0, 0] = 0.5
    assert f.allowed_bits(sign, magnitude, valid)[0, 0] == 1
    magnitude[0, 0] = -0.1
    with pytest.raises(ValueError, match='NUMERIC_DOMAIN'):
        f.allowed_bits(sign, magnitude, valid)


def test_exact_fold_clock_censors_boundary_without_changing_labels():
    start, end = state.FOLD_BOUNDARIES_NS[:2]
    issues = np.array([start, end - state.M5_NS], dtype=np.int64)
    parents = np.array([4, 9], dtype=np.int64)
    shape = (2, 5)
    target = f.BoundFutureDisplacement(
        parents, issues, np.ones(shape, dtype=bool),
        np.ones(shape, dtype=np.int8), np.ones(shape, dtype=np.float64),
        np.zeros(shape, dtype=bool))
    gathered = target.gather(parents, issues, fold_id=1, partition='score')
    assert gathered['allowed'].tolist() == [[4] * 5, [4] * 5]
    assert gathered['mature'][0].all()
    assert not gathered['mature'][1].any()
    with pytest.raises(ValueError, match='EXACT_JOIN'):
        target.gather(parents, issues + state.M5_NS, fold_id=1, partition='score')


def test_existing_factorized_loss_gives_no_F_side_gradient_on_flat_or_missing():
    bits = f.allowed_bits(
        np.array([[1, 1, 0, 0, -1]], dtype=np.int8),
        np.array([[0.8, 0.1, 0.0, 0.0, 0.6]], dtype=np.float64),
        np.array([[True, True, True, False, True]]))
    raw = torch.zeros((1, 5, 2), dtype=torch.float32, requires_grad=True)
    allowed = torch.from_numpy(bits)
    eligible = torch.tensor([[True, True, True, False, True]])
    mature = torch.ones((1, 5), dtype=torch.bool)
    k = torch.tensor([loss.K], dtype=torch.int64)
    report = loss.allowed_set_loss(raw, allowed, eligible, mature, k)
    report['loss'].backward()
    assert raw.grad is not None
    assert raw.grad[0, 0, 1] != 0 and raw.grad[0, 4, 1] != 0
    assert torch.equal(raw.grad[0, 1:4, 1], torch.zeros(3))
