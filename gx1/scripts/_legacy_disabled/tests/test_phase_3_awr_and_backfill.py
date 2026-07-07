"""Tests for Phase 3 gates: M1 backfill and AWR proper IQL POC."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import (
    materialize_backfill_xauusd_m1_2020_2024_v1 as backfill_gate,
)
from gx1.scripts import (
    materialize_build_awr_proper_iql_poc_v1 as awr_gate,
)


# ---------------------------------------------------------------------------
# Backfill tests
# ---------------------------------------------------------------------------


def test_backfill_target_years_match_2020_2024() -> None:
    assert backfill_gate.TARGET_YEARS == [2020, 2021, 2022, 2023, 2024]


def test_backfill_canonical_root_matches_oanda_convention() -> None:
    p = str(backfill_gate.CANONICAL_M1_ROOT)
    assert "xauusd_m1_bid_ask__CANONICAL" in p
    assert "/oanda/canonical/" in p


def test_backfill_validate_year_df_passes_on_clean_year() -> None:
    idx = pd.date_range(
        "2020-06-15 00:00:00", "2020-06-15 23:59:00", freq="min", tz="UTC"
    )
    n = len(idx)
    df = pd.DataFrame(
        {
            "bid_open": np.random.uniform(1700, 1800, n),
            "bid_high": np.random.uniform(1700, 1800, n),
            "bid_low": np.random.uniform(1700, 1800, n),
            "bid_close": np.random.uniform(1700, 1800, n),
            "ask_open": np.random.uniform(1700, 1800, n),
            "ask_high": np.random.uniform(1700, 1800, n),
            "ask_low": np.random.uniform(1700, 1800, n),
            "ask_close": np.random.uniform(1700, 1800, n),
        },
        index=idx,
    )
    # Force OHLC consistency.
    for prefix in ("bid", "ask"):
        df[f"{prefix}_high"] = df[[f"{prefix}_open", f"{prefix}_close"]].max(axis=1) + 0.5
        df[f"{prefix}_low"] = df[[f"{prefix}_open", f"{prefix}_close"]].min(axis=1) - 0.5
    # Force bid <= ask.
    for c in ("open", "high", "low", "close"):
        df[f"ask_{c}"] = df[f"bid_{c}"] + 0.1
    audit = backfill_gate._validate_year_df(df, year=2020)
    # 1 day of M1 data has only 1440 candles; that's below the year-level
    # lower bound, so the audit will mark it FAIL on count range.
    assert audit["candle_count_v1"] == n
    assert audit["bid_le_ask_invariant_v1"] is True
    assert audit["high_low_invariant_v1"] is True
    assert audit["n_negative_or_zero_price_v1"] == 0
    assert audit["n_duplicate_ts_v1"] == 0


def test_backfill_validate_detects_duplicate_timestamps() -> None:
    idx = pd.DatetimeIndex(
        ["2020-06-15 00:00:00", "2020-06-15 00:00:00", "2020-06-15 00:01:00"], tz="UTC"
    )
    df = pd.DataFrame(
        {
            "bid_open": [1700.0, 1700.0, 1700.0],
            "bid_high": [1700.0, 1700.0, 1700.0],
            "bid_low": [1700.0, 1700.0, 1700.0],
            "bid_close": [1700.0, 1700.0, 1700.0],
            "ask_open": [1700.1, 1700.1, 1700.1],
            "ask_high": [1700.1, 1700.1, 1700.1],
            "ask_low": [1700.1, 1700.1, 1700.1],
            "ask_close": [1700.1, 1700.1, 1700.1],
        },
        index=idx,
    )
    audit = backfill_gate._validate_year_df(df, year=2020)
    assert audit["n_duplicate_ts_v1"] == 1
    assert "DUPLICATE_TIMESTAMPS" in audit["failures_v1"]


def test_backfill_validate_detects_bid_gt_ask() -> None:
    idx = pd.date_range("2020-06-15", periods=3, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "bid_open": [1700.0, 1700.0, 1700.0],
            "bid_high": [1700.0, 1700.0, 1700.0],
            "bid_low": [1700.0, 1700.0, 1700.0],
            "bid_close": [1700.0, 1700.0, 1700.0],
            "ask_open": [1699.5, 1699.5, 1699.5],  # ask < bid (impossible)
            "ask_high": [1699.5, 1699.5, 1699.5],
            "ask_low": [1699.5, 1699.5, 1699.5],
            "ask_close": [1699.5, 1699.5, 1699.5],
        },
        index=idx,
    )
    audit = backfill_gate._validate_year_df(df, year=2020)
    assert audit["bid_le_ask_invariant_v1"] is False
    assert "BID_GT_ASK" in audit["failures_v1"]


def test_backfill_validate_final_status_rejects_unknown() -> None:
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        backfill_gate.validate_final_status(
            "MADE_UP", "BUILD_EXTENDED_BASE34_PREBUILT_2020_2026_V1"
        )


def test_backfill_validate_no_deprecated_revival_passes_on_self() -> None:
    from pathlib import Path

    backfill_gate.validate_no_deprecated_revival(Path(backfill_gate.__file__))


# ---------------------------------------------------------------------------
# AWR tests
# ---------------------------------------------------------------------------


def test_awr_ridge_fit_recovers_least_squares() -> None:
    rng = np.random.default_rng(42)
    n, k = 100, 5
    X = rng.normal(size=(n, k))
    true_beta = np.array([0.5, -0.3, 1.0, 0.0, 0.7])
    y = X @ true_beta + rng.normal(scale=0.01, size=n)
    beta_hat = awr_gate._ridge_fit(X, y, lam=1e-6)
    np.testing.assert_allclose(beta_hat, true_beta, atol=0.05)


def test_awr_build_action_augmented_state_appends_one_hot() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    actions = np.array([0, 1, 0])  # HOLD, EXIT_NOW, HOLD
    sa = awr_gate._build_action_augmented_state(X, actions)
    assert sa.shape == (3, 4)
    # First two columns are state.
    np.testing.assert_array_equal(sa[:, :2], X)
    # Third column is is_hold.
    np.testing.assert_array_equal(sa[:, 2], [1.0, 0.0, 1.0])
    # Fourth column is is_exit_now.
    np.testing.assert_array_equal(sa[:, 3], [0.0, 1.0, 0.0])


def test_awr_compute_advantage_returns_v_q_and_diffs() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    coef_v = np.array([1.0, 0.5])  # V(s) = 1*x1 + 0.5*x2
    coef_q_sa = np.array([1.0, 0.5, 1.0, -1.0])  # Q(s,a) = state·coef + 1*hold - 1*exit
    out = awr_gate._compute_advantage(X, coef_v, coef_q_sa)
    expected_v = np.array([1.0 + 1.0, 3.0 + 2.0])  # = [2.0, 5.0]
    expected_q_hold = expected_v + 1.0  # adding hold one-hot * 1.0
    expected_q_exit = expected_v - 1.0  # adding exit one-hot * (-1.0)
    np.testing.assert_array_almost_equal(out["v_s_v1"], expected_v)
    np.testing.assert_array_almost_equal(out["q_hold_v1"], expected_q_hold)
    np.testing.assert_array_almost_equal(out["q_exit_v1"], expected_q_exit)
    np.testing.assert_array_almost_equal(
        out["advantage_exit_minus_hold_v1"], expected_q_exit - expected_q_hold
    )


def test_awr_policy_softmax_two_actions() -> None:
    advantage = np.array([0.0, 1.0, -1.0, 5.0, -5.0])
    p = awr_gate._awr_policy_exit_now_probability(advantage, beta=1.0, clip=10.0)
    # advantage 0 -> p = 0.5
    assert pytest.approx(p[0], abs=1e-6) == 0.5
    # positive advantage -> p > 0.5 (favors exit)
    assert p[1] > 0.5
    # negative advantage -> p < 0.5 (favors hold)
    assert p[2] < 0.5
    # very high positive -> close to 1
    assert p[3] > 0.99
    # very negative -> close to 0
    assert p[4] < 0.01


def test_awr_policy_clipping_bounds_extremes() -> None:
    advantage = np.array([1000.0, -1000.0])
    p = awr_gate._awr_policy_exit_now_probability(advantage, beta=1.0, clip=5.0)
    # Even with extreme advantage the clip caps at +/-5.
    assert p[0] < 1.0  # not exactly 1 because clip(adv) = 5, sigmoid(5*1)
    assert p[1] > 0.0
    # sigmoid(5) = 0.9933..., sigmoid(-5) = 0.0067...
    assert pytest.approx(p[0], abs=0.01) == 1.0 / (1.0 + np.exp(-5.0))
    assert pytest.approx(p[1], abs=0.01) == 1.0 / (1.0 + np.exp(5.0))


def test_awr_beta_grid_includes_practical_values() -> None:
    assert 1.0 in awr_gate.BETA_GRID
    assert 3.0 in awr_gate.BETA_GRID
    assert 10.0 in awr_gate.BETA_GRID


def test_awr_validate_final_status_rejects_unknown() -> None:
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        awr_gate.validate_final_status("MADE_UP", "BUILD_CONSERVATIVE_Q_LEARNING_V1")


def test_awr_validate_final_status_rejects_unknown_action() -> None:
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        awr_gate.validate_final_status(
            "AWR_PROPER_IQL_POC_PASS_MEETS_PROMOTION_CRITERIA", "TRAIN_NOW"
        )


def test_awr_validate_no_deprecated_revival_passes_on_self() -> None:
    from pathlib import Path

    awr_gate.validate_no_deprecated_revival(Path(awr_gate.__file__))
