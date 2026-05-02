"""Tests for entry_iql_gpu_core_v1 — 3-action contextual bandit."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from gx1.scripts import entry_iql_gpu_core_v1 as entry_iql


def _toy_bandit(n: int = 64, d: int = 8, seed: int = 7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    a = rng.integers(0, 3, size=n).astype(np.int64)
    # Reward depends on action: SKIP=0, TAKE_NOW=signal, WAIT=signal*0.5
    base = X[:, 0]  # use first feature as signal
    r = np.where(a == 0, 0.0, np.where(a == 1, base, base * 0.5)).astype(np.float32)
    return X, a, r


def test_action_constants_named_correctly():
    assert entry_iql.ACTION_SKIP_ID == 0
    assert entry_iql.ACTION_TAKE_NOW_ID == 1
    assert entry_iql.ACTION_WAIT_ID == 2
    assert entry_iql.N_ACTIONS == 3
    assert entry_iql.ACTION_LABELS[0] == "SKIP"
    assert entry_iql.ACTION_LABELS[1] == "TAKE_NOW"
    assert entry_iql.ACTION_LABELS[2] == "WAIT"


def test_train_returns_model_with_expected_shapes():
    X, a, r = _toy_bandit()
    model = entry_iql.train_entry_iql_gpu(
        X, a, r, tau=0.7, k_iterations=2, epochs=5, prefer_cuda=False,
    )
    assert model.state_dim == X.shape[1]
    assert model.n_actions == 3
    q = model.predict_q(X)
    assert q.shape == (X.shape[0], 3)
    v = model.predict_v(X)
    assert v.shape == (X.shape[0],)


def test_policy_returns_valid_probabilities():
    X, a, r = _toy_bandit()
    model = entry_iql.train_entry_iql_gpu(
        X, a, r, tau=0.7, k_iterations=2, epochs=5, prefer_cuda=False,
    )
    probs = entry_iql.entry_iql_policy_action_probs(X, model, beta=3.0)
    assert probs.shape == (X.shape[0], 3)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_policy_argmax_returns_valid_action_ids():
    X, a, r = _toy_bandit()
    model = entry_iql.train_entry_iql_gpu(
        X, a, r, tau=0.7, k_iterations=2, epochs=5, prefer_cuda=False,
    )
    actions = entry_iql.entry_iql_policy_argmax_action(X, model)
    assert actions.shape == (X.shape[0],)
    assert ((actions >= 0) & (actions < 3)).all()


def test_higher_beta_gives_more_peaked_distribution():
    X, a, r = _toy_bandit(n=128, seed=99)
    model = entry_iql.train_entry_iql_gpu(
        X, a, r, tau=0.7, k_iterations=2, epochs=10, prefer_cuda=False,
    )
    probs_low = entry_iql.entry_iql_policy_action_probs(X, model, beta=0.5)
    probs_high = entry_iql.entry_iql_policy_action_probs(X, model, beta=20.0)
    # Higher beta → more peaked → max prob closer to 1
    assert probs_high.max(axis=1).mean() > probs_low.max(axis=1).mean()


def test_expectile_loss_at_tau_05_reduces_to_half_mse():
    diff = torch.tensor([1.0, -1.0, 2.0, -2.0])
    half = entry_iql.expectile_loss(diff, 0.5)
    half_mse = (diff.pow(2) * 0.5).mean()
    torch.testing.assert_close(half, half_mse)


def test_q_warmup_recovers_action_signal():
    """If reward = 5 for TAKE_NOW and 0 for others, Q(s, TAKE_NOW) > Q(s, SKIP)."""
    rng = np.random.default_rng(11)
    n = 120
    X = rng.normal(size=(n, 5)).astype(np.float32)
    # Equal action distribution
    a = np.tile([0, 1, 2], n // 3 + 1)[:n].astype(np.int64)
    r = np.where(a == 1, 5.0, 0.0).astype(np.float32)
    model = entry_iql.train_entry_iql_gpu(
        X, a, r, tau=0.7, k_iterations=3, epochs=30, prefer_cuda=False,
    )
    q = model.predict_q(X)
    assert q[:, 1].mean() > q[:, 0].mean(), "Q(TAKE_NOW) should exceed Q(SKIP) when TAKE_NOW gives reward 5"
    assert q[:, 1].mean() > q[:, 2].mean(), "Q(TAKE_NOW) should exceed Q(WAIT) when TAKE_NOW gives reward 5"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_runs_on_cuda():
    X, a, r = _toy_bandit()
    model = entry_iql.train_entry_iql_gpu(
        X, a, r, tau=0.7, k_iterations=1, epochs=3, prefer_cuda=True,
    )
    assert model.device.type == "cuda"
    info = entry_iql.info(model)
    assert info["device"] in ("cuda", "cuda:0")
    assert info["v_param_count_v1"] > 0
    assert info["q_param_count_v1"] > 0
    assert info["n_actions_v1"] == 3


def test_dropout_zero_gives_deterministic_inference():
    """With dropout=0 and same seed, two trainings produce identical predictions."""
    X, a, r = _toy_bandit(n=32)
    m1 = entry_iql.train_entry_iql_gpu(
        X, a, r, tau=0.7, k_iterations=1, epochs=3, dropout=0.0, seed=42, prefer_cuda=False,
    )
    m2 = entry_iql.train_entry_iql_gpu(
        X, a, r, tau=0.7, k_iterations=1, epochs=3, dropout=0.0, seed=42, prefer_cuda=False,
    )
    q1 = m1.predict_q(X)
    q2 = m2.predict_q(X)
    np.testing.assert_array_almost_equal(q1, q2, decimal=4)


def test_info_includes_action_labels_dict():
    X, a, r = _toy_bandit(n=20)
    model = entry_iql.train_entry_iql_gpu(
        X, a, r, tau=0.7, k_iterations=1, epochs=2, prefer_cuda=False,
    )
    info = entry_iql.info(model)
    assert "action_labels_v1" in info
    assert info["action_labels_v1"][0] == "SKIP"
    assert info["action_labels_v1"][1] == "TAKE_NOW"
    assert info["action_labels_v1"][2] == "WAIT"
