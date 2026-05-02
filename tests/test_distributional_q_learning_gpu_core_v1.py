"""Tests for distributional_q_learning_gpu_core_v1 — QR-DQN-style Q."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from gx1.scripts import distributional_q_learning_gpu_core_v1 as dist
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


def test_default_n_quantiles_matches_qrdqn_paper():
    """Dabney 2018 used N=51 for Atari; we keep the same default."""
    assert dist.DEFAULT_N_QUANTILES == 51


def test_quantile_midpoints_strictly_increasing_in_unit_interval():
    taus = dist._quantile_midpoints(11, torch.device("cpu"))
    assert taus.shape == (11,)
    assert (taus > 0).all() and (taus < 1).all()
    assert (taus[1:] > taus[:-1]).all()


def test_quantile_huber_loss_zero_when_target_in_distribution():
    """If predicted quantile equals scalar target everywhere, loss ~ 0."""
    z = torch.full((4, 11), 2.5)
    target = torch.full((4,), 2.5)
    loss = dist.quantile_huber_loss(z, target)
    assert loss.item() < 1e-6


def test_quantile_huber_loss_positive_when_misaligned():
    z = torch.zeros((4, 11))
    target = torch.ones((4,))
    loss = dist.quantile_huber_loss(z, target)
    assert loss.item() > 0


def test_train_distributional_returns_model_with_n_quantiles():
    X, a, r, next_idx, done = _toy_transitions()
    model = dist.train_distributional_iql_gpu(
        X, a, r, next_idx, done,
        tau=0.7, n_quantiles=11, k_iterations=2, inner_epochs=5,
        prefer_cuda=False,
    )
    assert model.n_quantiles == 11
    p_exit = dist.distributional_iql_policy_exit_prob_gpu(X, model, beta=3.0)
    assert p_exit.shape == (X.shape[0],)
    assert (p_exit >= 0.0).all() and (p_exit <= 1.0).all()


def test_predict_q_quantiles_shape():
    X, a, r, next_idx, done = _toy_transitions()
    model = dist.train_distributional_iql_gpu(
        X, a, r, next_idx, done,
        tau=0.7, n_quantiles=11, k_iterations=1, inner_epochs=3,
        prefer_cuda=False,
    )
    z = model.predict_q_quantiles(X, action_id=iql.ACTION_HOLD_ID)
    assert z.shape == (X.shape[0], 11)


def test_predict_expected_q_is_mean_of_quantiles():
    X, a, r, next_idx, done = _toy_transitions()
    model = dist.train_distributional_iql_gpu(
        X, a, r, next_idx, done,
        tau=0.7, n_quantiles=11, k_iterations=1, inner_epochs=3,
        prefer_cuda=False,
    )
    z = model.predict_q_quantiles(X, action_id=iql.ACTION_EXIT_NOW_ID)
    expected = model.predict_expected_q(X, action_id=iql.ACTION_EXIT_NOW_ID)
    np.testing.assert_array_almost_equal(expected, z.mean(axis=1))


def test_distributional_quantile_spread_grows_after_training():
    """After training on noisy rewards, the per-state quantile spread (std)
    should be > 0 — the network has learned a non-degenerate distribution.
    """
    X, a, r, next_idx, done = _toy_transitions(n=128, seed=99)
    model = dist.train_distributional_iql_gpu(
        X, a, r, next_idx, done,
        tau=0.7, n_quantiles=51, k_iterations=2, inner_epochs=10,
        prefer_cuda=False,
    )
    z = model.predict_q_quantiles(X, action_id=iql.ACTION_HOLD_ID)
    spreads = z.std(axis=1)
    assert (spreads > 1e-3).any(), "All quantile distributions are flat — likely degenerate fit"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_runs_on_cuda_when_available():
    X, a, r, next_idx, done = _toy_transitions()
    model = dist.train_distributional_iql_gpu(
        X, a, r, next_idx, done,
        tau=0.7, n_quantiles=11, k_iterations=1, inner_epochs=3,
        prefer_cuda=True,
    )
    assert model.device.type == "cuda"
    info = dist.info(model)
    assert info["n_quantiles_v1"] == 11
    assert info["q_param_count_v1"] > 0
