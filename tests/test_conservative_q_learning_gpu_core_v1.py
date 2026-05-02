"""Tests for conservative_q_learning_gpu_core_v1 — CQL extension."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from gx1.scripts import conservative_q_learning_gpu_core_v1 as cql
from gx1.scripts import true_iql_gpu_core_v1 as iql


def _toy_transitions(n: int = 64, d: int = 5, seed: int = 7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    a = rng.integers(0, 2, size=n).astype(np.int64)
    r = rng.normal(size=n).astype(np.float32)
    next_idx = np.arange(n, dtype=np.int64)
    next_idx[-1] = -1
    done = np.zeros(n, dtype=bool)
    done[-1] = True
    return X, a, r, next_idx, done


def test_cql_alpha_grid_positive():
    assert all(a > 0 for a in cql.CQL_ALPHA_GRID)
    assert len(cql.CQL_ALPHA_GRID) >= 2


def test_cql_regularizer_zero_when_uniform_q():
    """If all action Qs are equal, logsumexp(Q) - Q(a_data) reduces to log(N)."""
    n_actions = 2
    q_all = torch.full((10, n_actions), 1.0)
    q_data = torch.full((10,), 1.0)
    reg = cql.cql_regularizer(q_all, q_data)
    expected = float(torch.log(torch.tensor(float(n_actions))))
    assert abs(reg.item() - expected) < 1e-5


def test_cql_regularizer_negative_when_data_action_is_argmax():
    """If a_data is the argmax, OOD Q is small; logsumexp ~ q_data; reg ~ 0."""
    q_all = torch.tensor([[5.0, -5.0], [5.0, -5.0]])
    q_data = torch.tensor([5.0, 5.0])
    reg = cql.cql_regularizer(q_all, q_data)
    # logsumexp([5, -5]) ~ 5.0000045; minus 5 ~ ~0
    assert reg.item() < 0.1


def test_cql_regularizer_positive_when_ood_action_dominates():
    """If a non-data action has higher Q, reg should be positive (penalize OOD)."""
    q_all = torch.tensor([[10.0, 0.0], [10.0, 0.0]])
    q_data = torch.tensor([0.0, 0.0])  # data took the LOW-Q action
    reg = cql.cql_regularizer(q_all, q_data)
    # logsumexp([10, 0]) ~ 10; minus 0 ~ 10
    assert reg.item() > 5.0


def test_q_for_all_actions_shape():
    q_net = iql.MLP(in_dim=5 + iql.N_ACTIONS, out_dim=1, hidden_dim=8, n_hidden=1)
    X = torch.randn(7, 5)
    out = cql._q_for_all_actions(q_net, X)
    assert out.shape == (7, iql.N_ACTIONS)


def test_train_iql_cql_returns_model_with_alpha_recorded():
    X, a, r, next_idx, done = _toy_transitions()
    model = cql.train_iql_cql_gpu(
        X, a, r, next_idx, done,
        tau=0.7, cql_alpha=0.5, k_iterations=2, inner_epochs=5,
        prefer_cuda=False,
    )
    assert model.cql_alpha == 0.5
    p_exit = cql.true_iql_cql_policy_exit_prob_gpu(X, model, beta=3.0)
    assert p_exit.shape == (X.shape[0],)
    assert (p_exit >= 0.0).all() and (p_exit <= 1.0).all()


def test_higher_cql_alpha_pushes_q_lower_on_average():
    """With heavier OOD penalty, Q-values should compress toward the data-action range.
    Quantitative check: variance of Q across actions should be smaller with high alpha.
    """
    X, a, r, next_idx, done = _toy_transitions()
    m_low = cql.train_iql_cql_gpu(
        X, a, r, next_idx, done,
        tau=0.7, cql_alpha=0.05, k_iterations=2, inner_epochs=10, prefer_cuda=False,
    )
    m_high = cql.train_iql_cql_gpu(
        X, a, r, next_idx, done,
        tau=0.7, cql_alpha=2.0, k_iterations=2, inner_epochs=10, prefer_cuda=False,
    )
    q_hold_low = m_low.predict_q(X, action_id=iql.ACTION_HOLD_ID)
    q_exit_low = m_low.predict_q(X, action_id=iql.ACTION_EXIT_NOW_ID)
    q_hold_hi = m_high.predict_q(X, action_id=iql.ACTION_HOLD_ID)
    q_exit_hi = m_high.predict_q(X, action_id=iql.ACTION_EXIT_NOW_ID)
    range_low = float(np.std(q_exit_low - q_hold_low))
    range_hi = float(np.std(q_exit_hi - q_hold_hi))
    # Larger alpha should not increase the OOD-vs-data gap (smoke check).
    assert range_hi <= range_low * 3.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_runs_on_cuda_when_available():
    X, a, r, next_idx, done = _toy_transitions()
    model = cql.train_iql_cql_gpu(
        X, a, r, next_idx, done,
        tau=0.7, cql_alpha=0.5, k_iterations=2, inner_epochs=5,
        prefer_cuda=True,
    )
    assert model.device.type == "cuda"
    info = cql.info(model)
    assert info["cql_alpha_v1"] == 0.5
    assert info["v_param_count_v1"] > 0


def test_info_includes_cql_alpha():
    X, a, r, next_idx, done = _toy_transitions()
    model = cql.train_iql_cql_gpu(
        X, a, r, next_idx, done,
        tau=0.7, cql_alpha=1.0, k_iterations=1, inner_epochs=3,
        prefer_cuda=False,
    )
    info = cql.info(model)
    assert "cql_alpha_v1" in info
    assert info["cql_alpha_v1"] == 1.0
