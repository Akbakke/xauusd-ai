"""Equal known-row M5 gradients without changing any other task coordinate."""
import numpy as np
import pytest
import torch

from gx1.contracts import market_state_time_query_side_normalized_loss_v1 as loss
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K


def _batch(n):
    allowed = torch.tensor([[5, 1, 4, 2, 7]] * n, dtype=torch.uint8)
    eligible = torch.ones((n, 5), dtype=torch.bool)
    mature = torch.ones((n, 5), dtype=torch.bool)
    query = torch.tensor([NATIVE_K] * n, dtype=torch.int64)
    return allowed, eligible, mature, query


def test_identical_known_rows_get_equal_gradient_across_B16_and_tail_B6():
    a, e, m, q = _batch(38)
    a[[0, 16, 17, 32], 0] = torch.tensor([1, 4, 4, 1], dtype=torch.uint8)
    a[18, 0] = 4
    m[18, 0] = False
    normalizer = dict(known_side_rows=4, nonempty_batches=3)
    all_new = []
    all_old = []
    for left, right in ((0, 16), (16, 32), (32, 38)):
        raw = torch.zeros((right-left, 5, 2), dtype=torch.float32, requires_grad=True)
        args = a[left:right], e[left:right], m[left:right], q[left:right]
        old = loss.allowed_set_loss(raw, *args)
        new = loss.allowed_set_loss(raw, *args, m5_side_normalization=normalizer)
        go = torch.autograd.grad(old['loss'], raw)[0]
        gn = torch.autograd.grad(new['loss'], raw)[0]
        assert torch.equal(go[..., 0], gn[..., 0])
        assert torch.equal(go[:, 1:, :], gn[:, 1:, :])
        assert torch.equal(old['side_known_mask'], new['side_known_mask'])
        assert old['per_tf_counts'] == new['per_tf_counts']
        for tf in ('M15', 'H1', 'H4', 'D1'):
            assert torch.equal(old['per_tf_mean_loss'][tf], new['per_tf_mean_loss'][tf])
        all_new.append(gn[:, 0, 1]); all_old.append(go[:, 0, 1])
    gn, go = torch.cat(all_new), torch.cat(all_old)
    known = e[:, 0] & m[:, 0] & ((a[:, 0] == 1) | (a[:, 0] == 4))
    # At logit0, |d BCE/dz|=.5. Constant side coefficient q/(5N)=.15.
    assert torch.allclose(gn[known].abs(), torch.full((4,), .075), atol=1e-8, rtol=0)
    assert torch.equal(gn[~known], torch.zeros_like(gn[~known]))
    assert go[0].abs() > go[16].abs()
    assert float(gn[known].abs().sum()) == pytest.approx(float(go[known].abs().sum()), abs=3e-8)


def test_empty_M5_side_keeps_activity_and_other_TF_gradients():
    a, e, m, q = _batch(6)
    a[:, 0] = torch.tensor([2, 5, 7, 4, 1, 5], dtype=torch.uint8)
    m[3, 0] = False; e[4, 0] = False
    raw = torch.linspace(-2, 2, 60).reshape(6, 5, 2).requires_grad_(True)
    old = loss.allowed_set_loss(raw, a, e, m, q)
    new = loss.allowed_set_loss(raw, a, e, m, q,
        m5_side_normalization=dict(known_side_rows=10, nonempty_batches=2))
    go = torch.autograd.grad(old['loss'], raw)[0]
    gn = torch.autograd.grad(new['loss'], raw)[0]
    assert not new['skip_update'] and new['per_tf_side_counts']['M5'] == 0
    assert torch.equal(gn[:, 0, 1], torch.zeros(6))
    assert torch.equal(gn, go)
    assert bool(torch.count_nonzero(gn[:, 0, 0]))


def test_combined_route_applies_distinct_S_F_mass_and_only_training_identity_changes():
    a, e, m, q = _batch(16)
    a[:2, 0] = torch.tensor([1, 4], dtype=torch.uint8)
    truth = dict(allowed=a.numpy(), eligible=e.numpy(), mature=m.numpy())
    s = torch.zeros((16, 5, 2), requires_grad=True)
    f = torch.zeros((16, 5, 2), requires_grad=True)
    report = objective.combined_loss(state_raw=s, future_raw=f,
        state_truth=truth, future_truth=truth, state_query_k=q, future_weight=1.0,
        ta_only=True, dedicated_m5_side=True, m5_equal_observation=True)
    gs, gf = torch.autograd.grad(report['loss'], (s, f))
    assert float(gs[0, 0, 1]) == pytest.approx(.5 * 1799 / 10010 / 5, abs=1e-9)
    assert float(gf[0, 0, 1]) == pytest.approx(.5 * 1844 / 11909 / 5, abs=1e-9)
    assert torch.equal(gs[2:, 0, 1], torch.zeros(14))
    assert torch.equal(gf[2:, 0, 1], torch.zeros(14))
    default = objective.contract(ta_only=True, dedicated_m5_side=True)
    enabled = objective.contract(ta_only=True, dedicated_m5_side=True, m5_equal_observation=True)
    assert report['contract_sha256'] == enabled['contract_sha256'] != default['contract_sha256']
    assert default == objective.contract(ta_only=True, dedicated_m5_side=True, m5_equal_observation=False)
    for name in ('dual_model_contract_sha256', 'F_target_contract_sha256', 'F_score_contract_sha256',
                 'F_reference', 'F_score', 'S_scoring'):
        assert enabled[name] == default[name]


@pytest.mark.parametrize('normalizer', [
    {}, {'known_side_rows': 10}, {'known_side_rows': 0, 'nonempty_batches': 1},
    {'known_side_rows': 1, 'nonempty_batches': 2}, {'known_side_rows': 17, 'nonempty_batches': 1},
    {'known_side_rows': True, 'nonempty_batches': 1}, {'known_side_rows': 10., 'nonempty_batches': 1},
])
def test_bad_normalization_rejected(normalizer):
    with pytest.raises(ValueError, match='NORMALIZATION'):
        loss.contract(m5_side_normalization=normalizer)


@pytest.mark.parametrize('kwargs', [
    dict(m5_equal_observation=True),
    dict(ta_only=True, m5_equal_observation=True),
    dict(ta_only=True, dedicated_m5_side=True, m5_equal_observation=1),
])
def test_opt_in_cannot_escape_ta_dedicated_M5(kwargs):
    with pytest.raises(ValueError):
        objective.contract(**kwargs)
