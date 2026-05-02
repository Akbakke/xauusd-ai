"""Tests for materialize_build_true_implicit_q_learning_v1."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import (
    materialize_build_true_implicit_q_learning_v1 as iql_gate,
)


# ---------------------------------------------------------------------------
# Expectile regression
# ---------------------------------------------------------------------------


def test_expectile_at_tau_05_recovers_ols() -> None:
    """tau=0.5 expectile = OLS."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 4))
    true_beta = np.array([1.0, -0.5, 0.0, 0.7])
    y = X @ true_beta + rng.normal(scale=0.1, size=200)
    beta_05 = iql_gate.fit_expectile_regression(X, y, tau=0.5, lam=1e-6)
    beta_ols = iql_gate._ridge_fit(X, y, lam=1e-6)
    np.testing.assert_allclose(beta_05, beta_ols, atol=0.01)


def test_expectile_at_tau_above_05_overestimates() -> None:
    """Higher tau weights positive residuals more, so the fit
    over-predicts compared to OLS."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 3))
    true_beta = np.array([0.5, 0.5, 0.5])
    y = X @ true_beta + rng.normal(scale=0.1, size=200)
    beta_07 = iql_gate.fit_expectile_regression(X, y, tau=0.7, lam=1e-6)
    pred_05 = X @ iql_gate._ridge_fit(X, y, lam=1e-6)
    pred_07 = X @ beta_07
    # Mean prediction at tau>0.5 should be higher than OLS mean prediction
    # (upper expectile is biased upward).
    assert pred_07.mean() > pred_05.mean()


def test_expectile_at_tau_below_05_underestimates() -> None:
    rng = np.random.default_rng(11)
    X = rng.normal(size=(200, 3))
    true_beta = np.array([0.5, 0.5, 0.5])
    y = X @ true_beta + rng.normal(scale=0.1, size=200)
    beta_03 = iql_gate.fit_expectile_regression(X, y, tau=0.3, lam=1e-6)
    pred_05 = X @ iql_gate._ridge_fit(X, y, lam=1e-6)
    pred_03 = X @ beta_03
    assert pred_03.mean() < pred_05.mean()


def test_weighted_ridge_recovers_unweighted_when_weights_uniform() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 4))
    y = rng.normal(size=80)
    weights = np.ones(80)
    beta_w = iql_gate._weighted_ridge_fit(X, y, weights, lam=1e-6)
    beta_u = iql_gate._ridge_fit(X, y, lam=1e-6)
    np.testing.assert_allclose(beta_w, beta_u, atol=1e-6)


# ---------------------------------------------------------------------------
# State-action features
# ---------------------------------------------------------------------------


def test_build_state_action_features_appends_one_hot() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    actions = np.array([0, 1, 0])
    sa = iql_gate._build_state_action_features(X, actions)
    assert sa.shape == (3, 4)
    np.testing.assert_array_equal(sa[:, :2], X)
    np.testing.assert_array_equal(sa[:, 2], [1.0, 0.0, 1.0])
    np.testing.assert_array_equal(sa[:, 3], [0.0, 1.0, 0.0])


def test_q_for_action_id_returns_correct_action_q() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    coef_q = np.array([1.0, 1.0, 5.0, -3.0])  # state·1 + 5*hold + -3*exit
    q_hold = iql_gate._q_for_action_id(X, coef_q, action_id=0)
    q_exit = iql_gate._q_for_action_id(X, coef_q, action_id=1)
    np.testing.assert_array_almost_equal(q_hold, np.array([3.0 + 5.0, 7.0 + 5.0]))
    np.testing.assert_array_almost_equal(q_exit, np.array([3.0 - 3.0, 7.0 - 3.0]))


# ---------------------------------------------------------------------------
# Transition tuples
# ---------------------------------------------------------------------------


def _per_bar_with_actions(uids, bars, actions, pnls):
    return pd.DataFrame(
        {
            "candidate_uid_v1": uids,
            "bars_held_v1": bars,
            "action_id_v1": actions,
            "running_pnl_at_close_bps_v1": pnls,
            "reward_realized_pnl_reward_v1": pnls,
        }
    )


def test_build_transition_tuples_assigns_terminal_to_exit_now() -> None:
    df = _per_bar_with_actions(
        uids=["A", "A", "B"],
        bars=[0, 1, 0],
        actions=[0, 1, 1],  # HOLD, EXIT_NOW, EXIT_NOW
        pnls=[10.0, 30.0, -20.0],
    )
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    out = iql_gate._build_transition_tuples(df, X, "reward_realized_pnl_reward_v1")
    # All EXIT_NOW rows are terminal.
    assert out["done_v1"][1] is np.True_ or out["done_v1"][1] == True
    assert out["done_v1"][2] is np.True_ or out["done_v1"][2] == True
    # EXIT_NOW reward is the variant column value.
    assert out["r_v1"][1] == 30.0
    assert out["r_v1"][2] == -20.0


def test_build_transition_tuples_finds_next_hold_for_hold_rows() -> None:
    df = _per_bar_with_actions(
        uids=["A", "A", "A", "A"],
        bars=[0, 0, 1, 1],
        actions=[0, 1, 0, 1],  # HOLD bar0, EXIT_NOW bar0, HOLD bar1, EXIT_NOW bar1
        pnls=[5.0, 10.0, 15.0, 20.0],
    )
    X = np.eye(4)
    out = iql_gate._build_transition_tuples(df, X, "reward_realized_pnl_reward_v1")
    # HOLD bar0 should pair with HOLD bar1 (same uid, same action, bar+1).
    next_idx = out["next_idx_v1"]
    # row 0 is HOLD bar0; its next HOLD-bar1 is row 2.
    assert next_idx[0] == 2
    # row 2 is HOLD bar1; no next HOLD bar -> -1, done.
    assert next_idx[2] == -1
    assert out["done_v1"][2]


# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------


def test_tau_grid_in_upper_quantile_range() -> None:
    for t in iql_gate.TAU_GRID:
        assert 0.5 < t < 1.0


def test_beta_grid_includes_practical_values() -> None:
    assert all(b > 0 for b in iql_gate.BETA_GRID)


def test_gamma_locked_per_mdp_contract() -> None:
    assert iql_gate.GAMMA_LOCKED == 0.99


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def test_true_iql_policy_returns_probabilities_in_unit_interval() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(20, 3))
    coef_q = rng.normal(size=5)  # 3 + 2 action one-hot
    coef_v = rng.normal(size=3)
    p = iql_gate.true_iql_policy_exit_prob(X, coef_q, coef_v, beta=3.0)
    assert (p >= 0.0).all() and (p <= 1.0).all()


def test_true_iql_policy_higher_q_exit_means_higher_probability() -> None:
    X = np.array([[1.0, 0.0]])
    # Q has structure [s_coef ⊕ hold_coef, exit_coef]
    coef_q_high_exit = np.array([0.0, 0.0, 0.0, 5.0])  # exit Q = 5, hold = 0
    coef_q_high_hold = np.array([0.0, 0.0, 5.0, 0.0])  # hold Q = 5, exit = 0
    coef_v = np.array([0.0, 0.0])
    p_high_exit = iql_gate.true_iql_policy_exit_prob(
        X, coef_q_high_exit, coef_v, beta=3.0
    )
    p_high_hold = iql_gate.true_iql_policy_exit_prob(
        X, coef_q_high_hold, coef_v, beta=3.0
    )
    assert p_high_exit[0] > 0.95
    assert p_high_hold[0] < 0.05


# ---------------------------------------------------------------------------
# Integration of train + policy
# ---------------------------------------------------------------------------


def test_train_true_iql_returns_v_and_q_coefficients() -> None:
    rng = np.random.default_rng(7)
    n = 50
    X_train = rng.normal(size=(n, 3))
    a_train = rng.integers(0, 2, size=n)
    r_train = rng.normal(size=n)
    next_idx = np.arange(n)
    next_idx[-1] = -1  # last row is terminal
    done = np.zeros(n, dtype=bool)
    done[-1] = True
    model = iql_gate.train_true_iql(
        X_train, a_train, r_train, next_idx, done, tau=0.7, k_iterations=3
    )
    assert model["coef_v"].shape == (3,)
    assert model["coef_q_sa"].shape == (3 + 2,)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def test_validate_final_status_rejects_unknown() -> None:
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        iql_gate.validate_final_status("MADE_UP", "BUILD_CONSERVATIVE_Q_LEARNING_V1")


def test_validate_final_status_rejects_unknown_action() -> None:
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        iql_gate.validate_final_status(
            "TRUE_IQL_PASS_MEETS_PROMOTION_CRITERIA", "TRAIN_NOW"
        )


def test_validate_no_deprecated_revival_passes_on_self() -> None:
    from pathlib import Path

    iql_gate.validate_no_deprecated_revival(Path(iql_gate.__file__))


# ---------------------------------------------------------------------------
# Hyperparameter constants
# ---------------------------------------------------------------------------


def test_k_vq_iterations_at_least_5() -> None:
    """V/Q backup needs enough iterations to converge."""
    assert iql_gate.K_VQ_ITERATIONS >= 5


def test_advantage_clip_bounded() -> None:
    assert iql_gate.ADVANTAGE_CLIP > 0
    assert iql_gate.ADVANTAGE_CLIP <= 10
